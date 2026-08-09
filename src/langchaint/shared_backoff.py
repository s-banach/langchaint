"""A shared pause and paced admission for one provider backpressure domain.

SharedBackoff has one control action: holding a request from starting until a deadline its whole domain shares.
A domain is the set of requests the caller routes through one instance, usually one model on one account.
A request enters `admitted()`, the async-with block spanning one attempt.
Entry acquires a permit when `max_concurrent_requests` is set, then admission; normally both are immediate.
After the provider pushes back, every request in the domain waits at entry until the shared pause ends.
When the pause ends, waiting requests remain ordered by arrival.
`max_request_starts_per_second` spaces their request starts.
Exit parses a provider failure into a verdict, records it, then returns the permit, in that order by position.
SharedBackoff decides no retries and counts no tokens.
It also bounds no pending work: it cannot tell an unadmitted request from one that has not entered yet.
The bound on pending work belongs to whatever spawns the work.

The verdicts are PauseAll, PauseAllDoNotRetry, RetryThisOne, and DoNotRetry; only the two pausing
ones change shared state.
The verdict comes from the status, the error type, and the provider's own retry directive; a
`retry-after` header never sets it.
The header says how long to wait, not who has to wait.

Every deadline here uses a forward-only clock (`time.monotonic`), never wall-clock time.
The one exception is a `parse` converting a timestamp-formatted `retry-after`.
A timestamp can only be compared against wall-clock time.
This is for asyncio: safe for many tasks in one event loop, with no attempt to be safe across threads.
"""

import asyncio
import inspect
import logging
import math
import random
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Literal

from langchaint.exceptions import GaveUpWaiting, ParserContractError

_logger = logging.getLogger(__name__)

_NEVER = float("-inf")
"""The moment before every other: the initial pause end and the initial admission time."""


@dataclass(frozen=True, kw_only=True)
class PauseAll:
    """The provider told us to stop sending for a while.

    retry_after is the wait the provider named in seconds, None where it named none.
    Recording this verdict starts the shared pause, or extends a running one.
    """

    retry_after: float | None
    kind: Literal["pause_all"] = "pause_all"


@dataclass(frozen=True, kw_only=True)
class RetryThisOne:
    """Worth retrying, with no sign the provider wants less traffic overall.

    retry_after is a wait floor for this one request's next attempt, None where the provider named none.
    Recording this verdict changes no shared state; it exists so the caller reads one vocabulary off Admission.verdict.
    """

    retry_after: float | None
    kind: Literal["retry_this_one"] = "retry_this_one"


@dataclass(frozen=True, kw_only=True)
class DoNotRetry:
    """Waiting will not help; the caller stops retrying this request.

    No retry_after field: a stray `retry-after` header on a failure that can never succeed names no wait worth serving.
    """

    kind: Literal["do_not_retry"] = "do_not_retry"


@dataclass(frozen=True, kw_only=True)
class PauseAllDoNotRetry:
    """The provider told us to stop sending for a while, and told this request not to come back.

    retry_after is the wait the provider named in seconds, None where it named none.
    It serves the shared pause rather than this request, which is why this variant carries one and
    DoNotRetry does not.
    Recording this verdict starts the shared pause, or extends a running one, exactly as PauseAll
    does; the caller stops retrying this request, exactly as on DoNotRetry.
    """

    retry_after: float | None
    kind: Literal["pause_all_do_not_retry"] = "pause_all_do_not_retry"


type Verdict = PauseAll | PauseAllDoNotRetry | RetryThisOne | DoNotRetry
"""What one parsed provider failure means for this request and for the domain.

"Terminal verdict" names DoNotRetry and PauseAllDoNotRetry, on either of which the caller stops
retrying this request.
"""


def _random_up_to(ceiling: float) -> float:
    """Draw a wait greater than zero and no larger than ceiling.

    1 - random.random() lies in (0, 1], so the draw is never the zero-length pause
    random.uniform permits.
    """
    return ceiling * (1.0 - random.random())


def _validated_positive_float(name: str, value: float) -> float:
    """Return value as a positive finite float.

    bool is rejected explicitly because it subclasses int, so a type checker admits True here.

    Raises:
        ValueError: value is a bool, is not finite, is not greater than zero, or is an int too
            large to convert to float.
    """
    if isinstance(value, int):
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(
                f"{name} must be representable as a float, got an int of {value.bit_length()} bits"
            ) from None
    else:
        converted = value
    if isinstance(value, bool) or not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(
            f"{name} must be a non-bool finite number greater than zero, got {value!r}"
        )
    return converted


class Admission:
    """One `admitted()` block: entry waits until the request may start, exit reports how it ended.

    verdict is None until a failure_types exception exits the block, then holds the normalized parse result.
    It is the same object _record received, so both sides work from the same capped retry_after.
    Build one only through SharedBackoff.admitted, which validates the budget first.
    """

    def __init__(self, shared_backoff: "SharedBackoff", budget_seconds: float | None) -> None:
        """Bind the block to its SharedBackoff and store the validated budget."""
        self._shared_backoff = shared_backoff
        self._budget_seconds = budget_seconds
        self.verdict: Verdict | None = None

    async def __aenter__(self) -> "Admission":
        """Acquire a permit when max_concurrent_requests is set, then admission; normally both are immediate.

        Once this returns, the request is admitted, and a pause starting afterwards does not revoke that admission.
        Cancellation during entry leaves nothing held: the request leaves the queue and any acquired permit returns.

        Raises:
            GaveUpWaiting: the budget expired first; nothing is held and nothing was recorded.
        """
        shared_backoff = self._shared_backoff
        try:
            async with asyncio.timeout(self._budget_seconds):
                await shared_backoff._acquire_permit()  # noqa: SLF001 (same-module machinery)
                try:
                    await shared_backoff._wait_turn()  # noqa: SLF001 (same-module machinery)
                except BaseException:
                    shared_backoff._release_permit()  # noqa: SLF001 (same-module machinery)
                    raise
        except TimeoutError:
            shared_backoff.event_counts["gave_up_waiting"] += 1
            _logger.info(
                "gave up waiting for admission after a budget of %s seconds", self._budget_seconds
            )
            raise GaveUpWaiting(
                f"gave up waiting for admission after a budget of {self._budget_seconds} seconds"
            ) from None
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Parse a failure_types exception into a verdict, record it, then return the permit.

        The order matters: recording after releasing would let a waiter take the permit and be
        admitted into the refusal already observed, so no await sits between recording and release.
        The release sits in a finally covering all exit processing, so a raise anywhere in it still
        returns the permit and cannot starve the domain.
        Any exception outside failure_types is a fault in the attempt, not a provider failure: the
        permit returns and it propagates unparsed and unrecorded.
        On success nothing is recorded and verdict stays None.

        Raises:
            ParserContractError: parse violated its contract and on_parse_error is "raise";
                nothing was recorded and verdict stays None.
        """
        try:
            if isinstance(exc_value, self._shared_backoff.failure_types):
                verdict = self._shared_backoff._checked_parse(exc_value)  # noqa: SLF001 (same-module machinery)
                self.verdict = verdict
                self._shared_backoff._record(verdict)  # noqa: SLF001 (same-module machinery)
        finally:
            self._shared_backoff._release_permit()  # noqa: SLF001 (same-module machinery)
        return False


class SharedBackoff:
    """The shared pause, the admission queue, and the permits of one domain.

    Route every request in the domain through one instance, first attempts and retries alike.

    The wait ceiling is a ceiling, not a wait: each pause of our own choosing lasts a fresh random
    number greater than zero and no larger than it.
    One wait can fall far below minimum_wait_ceiling_seconds.
    There is no setting for how many requests may wait; the bound on pending work belongs to
    whatever spawns the work.

    Rules for the caller that no position in code enforces:

    - Raise provider failures and stream error events as failure_types exceptions inside the block,
      including failures you will not retry, and consume the whole stream inside the block.
    - Do not catch a failure_types exception inside the block: a swallowed failure exits as a
      success, unrecorded.
    - Begin the attempt immediately after entry, and await no other admission mechanism inside the
      block: blocking there after admission piles up admitted requests outside this object's reach.
    - Keep a transport failure that produced nothing parseable out of failure_types; it propagates
      unrecorded, and the caller treats it as RetryThisOne with no retry_after.
    - Gate every retry on replay safety, whatever the verdict.
    - On RetryThisOne, run a backoff private to the logical request between blocks, never inside one.
    - On PauseAll, compute no wait of your own; the next entry already covers it.
    - Turn off automatic retries in the SDK and the HTTP transport: left on, they run inside the
      block, invisible to SharedBackoff.
    """

    def __init__(  # noqa: PLR0913 (the settings table travels whole: five numeric settings plus parse, failure_types, max_concurrent_requests, on_parse_error)
        self,
        *,
        parse: Callable[[Exception], Verdict],
        failure_types: tuple[type[Exception], ...],
        max_concurrent_requests: int | None,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
        max_request_starts_per_second: float = 50.0,
        on_parse_error: Literal["raise", "retry_this_one"] = "raise",
    ) -> None:
        """Validate the configuration and initialize an unpaused domain.

        `parse` synchronously maps each provider failure to a `Verdict`.
        `failure_types` identifies the exceptions passed to `parse`.
        `max_concurrent_requests=None` delegates concurrency limits to the caller.
        `minimum_wait_ceiling_seconds` is the initial and minimum wait ceiling.
        `longest_wait_seconds` caps the wait ceiling and `retry_after`.
        `wait_multiplier` changes the wait ceiling by one step.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.
        `max_request_starts_per_second` limits request starts during queued demand.
        `on_parse_error="raise"` raises `ParserContractError`.
        `on_parse_error="retry_this_one"` produces `RetryThisOne` without `retry_after`.

        Raises:
            ValueError: A numeric setting is boolean, non-finite, or non-positive.
                Also raised when the request-rate reciprocal is non-finite.
                Also raised when `wait_multiplier` is at most one.
                Also raised when `longest_wait_seconds` is below `minimum_wait_ceiling_seconds`.
                Also raised when their ratio is non-finite.
                Also raised when `max_concurrent_requests` is boolean or below one.
                Also raised when `on_parse_error` is unsupported.
                Also raised when `failure_types` is empty.
                Also raised when a failure type is not an `Exception` subclass.
                Also raised when a failure type equals `Exception`.
                Also raised when `inspect.iscoroutinefunction(parse)` is true.
        """
        self.minimum_wait_ceiling_seconds = _validated_positive_float(
            "minimum_wait_ceiling_seconds", minimum_wait_ceiling_seconds
        )
        self.longest_wait_seconds = _validated_positive_float(
            "longest_wait_seconds", longest_wait_seconds
        )
        self.wait_multiplier = _validated_positive_float("wait_multiplier", wait_multiplier)
        self.quiet_seconds_per_decay_step = _validated_positive_float(
            "quiet_seconds_per_decay_step", quiet_seconds_per_decay_step
        )
        self.max_request_starts_per_second = _validated_positive_float(
            "max_request_starts_per_second", max_request_starts_per_second
        )
        self._seconds_between_request_starts = 1.0 / self.max_request_starts_per_second
        if not math.isfinite(self._seconds_between_request_starts):
            raise ValueError(
                "1 / max_request_starts_per_second must be finite, "
                f"got {self._seconds_between_request_starts!r} from "
                f"{max_request_starts_per_second!r}"
            )
        if self.wait_multiplier <= 1.0:
            raise ValueError(f"wait_multiplier must be greater than 1, got {wait_multiplier!r}")
        if self.longest_wait_seconds < self.minimum_wait_ceiling_seconds:
            raise ValueError(
                "longest_wait_seconds must be at least minimum_wait_ceiling_seconds, "
                f"got {longest_wait_seconds!r} < {minimum_wait_ceiling_seconds!r}"
            )
        ceiling_ratio = self.longest_wait_seconds / self.minimum_wait_ceiling_seconds
        if not math.isfinite(ceiling_ratio):
            raise ValueError(
                "longest_wait_seconds / minimum_wait_ceiling_seconds must be finite, "
                f"got {ceiling_ratio!r} from {longest_wait_seconds!r} / "
                f"{minimum_wait_ceiling_seconds!r}"
            )
        if max_concurrent_requests is not None and (
            isinstance(max_concurrent_requests, bool) or max_concurrent_requests < 1
        ):
            raise ValueError(
                f"max_concurrent_requests must be None or a positive int, "
                f"got {max_concurrent_requests!r}"
            )
        if on_parse_error not in ("raise", "retry_this_one"):
            raise ValueError(
                f'on_parse_error must be "raise" or "retry_this_one", got {on_parse_error!r}'
            )
        if not failure_types:
            raise ValueError(
                "failure_types must not be empty: the exit would parse nothing and record nothing"
            )
        for failure_type in failure_types:
            if (
                not isinstance(failure_type, type)
                or not issubclass(failure_type, Exception)
                or failure_type is Exception
            ):
                raise ValueError(
                    f"every failure_types entry must be a strict subclass of Exception, "
                    f"got {failure_type!r}"
                )
        if inspect.iscoroutinefunction(parse):
            raise ValueError("parse must be synchronous, got a coroutine function")
        self.parse = parse
        self.failure_types = tuple(failure_types)
        self._max_concurrent_requests = max_concurrent_requests
        self.on_parse_error: Literal["raise", "retry_this_one"] = on_parse_error
        self._steps_to_floor = math.ceil(math.log(ceiling_ratio) / math.log(self.wait_multiplier))
        """Quiet steps after which the ceiling has reached the floor, whatever it started at.

        This bounds the decay exponent because the ceiling never exceeds longest_wait_seconds.
        Afterward, the answer is minimum_wait_ceiling_seconds.
        For fewer steps, wait_multiplier ** steps cannot exceed the checked ceiling ratio.
        """
        self._pause_until = _NEVER
        """When the current pause ends; once it is over, still the end of the previous pause."""
        self._pause_started_at = _NEVER
        """When the current pause began; a metric for logging, read by no decision."""
        self._wait_ceiling = self.minimum_wait_ceiling_seconds
        """Longest pause this object will currently choose for itself."""
        self._last_admission_at = _NEVER
        """When a request was last admitted.

        Enforces _seconds_between_request_starts.
        Also answers whether traffic resumed after the previous pause.
        """
        self._queue: deque[asyncio.Future[None]] = deque()
        """Requests waiting for admission, released in the order they joined."""
        self._permits = (
            None if max_concurrent_requests is None else asyncio.Semaphore(max_concurrent_requests)
        )
        """The permits; None when max_concurrent_requests is None.

        asyncio.Semaphore grants waiters in the order they joined and keeps the permit count whole
        under cancellation: acquire's CancelledError branch passes a granted-but-unclaimed permit
        on to the next waiter (CPython 3.14 asyncio.locks.Semaphore).
        """
        self._admit_timer: asyncio.TimerHandle | None = None
        """Wakes _admit_waiting when the front of the queue becomes admissible."""
        self._clock: Callable[[], float] = time.monotonic
        """The forward-only clock every deadline reads."""
        self.event_counts: Counter[str] = Counter()
        """How often each noteworthy entry or exit event occurred, by tag.

        The correction tags: "retry_after_invalid", "retry_after_over_cap", and under
        on_parse_error="retry_this_one" also "parse_raised" and "parse_returned_non_verdict".
        The failure tags: "gave_up_waiting" for a budget that expired before admission, and
        "parser_contract_error" for a parse contract violation under on_parse_error="raise".
        A metric, read by no decision.
        """

    @property
    def max_concurrent_requests(self) -> int | None:
        """The number of requests allowed inside admitted() blocks at once, or None for no bound.

        Read-only: the permits are sized once in __init__, so assigning this would move what
        callers read while leaving the permit count they contend for unchanged.
        """
        return self._max_concurrent_requests

    def admitted(self, *, budget: float | None = None) -> Admission:
        """Return the block spanning one attempt; enter it to wait, exit it to report.

        budget bounds entry alone: the permit acquisition plus the wait for admission, and nothing
        after, so what bounds the attempt itself is the SDK's own timeout.
        None means entry may wait indefinitely.
        The budget is per attempt; nothing bounds the total across attempts unless the caller
        subtracts what each attempt spent and passes the remainder to the next call.

        Raises:
            ValueError: budget is not None and fails the acceptance rule (not a bool, finite,
                greater than zero); nothing was acquired.
        """
        budget_seconds = None if budget is None else _validated_positive_float("budget", budget)
        return Admission(self, budget_seconds)

    async def _acquire_permit(self) -> None:
        """Hold one permit, waiting behind earlier waiters; no-op when there is no bound.

        Raises:
            asyncio.CancelledError: the wait was cancelled; no permit is held.
        """
        if self._permits is None:
            return
        _ = await self._permits.acquire()

    def _release_permit(self) -> None:
        """Return one permit, waking the longest-waiting live waiter; no-op when there is no bound."""
        if self._permits is None:
            return
        self._permits.release()

    async def _wait_turn(self) -> None:
        """Wait until the shared pause and request-start interval permit admission.

        Cancellation before the grant removes the request from the queue.
        Cancellation after the grant may consume one request-start interval.
        The caller returns the permit.

        Raises:
            asyncio.CancelledError: the wait was cancelled; the request is out of the queue.
        """
        granted: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.append(granted)
        self._admit_waiting()
        try:
            await granted
        except asyncio.CancelledError:
            if not (granted.done() and not granted.cancelled()):
                try:
                    self._queue.remove(granted)
                except ValueError:
                    pass  # _admit_waiting already dropped this spent entry from the front
            raise

    def _admit_waiting(self) -> None:
        """Admit the front of the queue while nothing blocks it, else arm the timer for when it will.

        Admission requires no active pause and one elapsed request-start interval.
        Granting records the moment in _last_admission_at.
        A queued burst starts at max_request_starts_per_second.
        Spent entries (cancelled waiters) at the front are dropped, never granted.
        """
        while self._queue:
            if self._queue[0].done():
                _ = self._queue.popleft()
                continue
            now = self._clock()
            admissible_at = max(
                self._pause_until,
                self._last_admission_at + self._seconds_between_request_starts,
            )
            if now < admissible_at:
                self._arm_admit_timer(admissible_at - now)
                return
            self._log_pause_end()
            granted = self._queue.popleft()
            self._last_admission_at = now
            granted.set_result(None)

    def _arm_admit_timer(self, delay_seconds: float) -> None:
        """Schedule _admit_waiting for when the front becomes admissible, replacing any armed timer."""
        if self._admit_timer is not None:
            self._admit_timer.cancel()
        self._admit_timer = asyncio.get_running_loop().call_later(
            delay_seconds, self._on_admit_timer
        )

    def _on_admit_timer(self) -> None:
        self._admit_timer = None
        self._admit_waiting()

    def _log_pause_end(self) -> None:
        """Log the ended pause's length and the queue depth, on the first admission after its end.

        Queue depth here cannot exceed max_concurrent_requests, or the worker pool's size when it
        is None, so a full queue means every permit or worker is idle; it says nothing about work
        waiting further upstream, which whatever bounds that work has to report.
        """
        if self._pause_until == _NEVER or self._last_admission_at > self._pause_until:
            return
        _logger.info(
            "pause of %.3f seconds ended with %d requests waiting",
            self._pause_until - self._pause_started_at,
            len(self._queue),
        )

    def _checked_parse(self, failure: Exception) -> Verdict:
        """Call parse under its contract checks; correct what is safely correctable, raise on the rest.

        An awaitable return is treated as a contract violation, closed first when it is a
        coroutine, so no unawaited-coroutine warning fires; the construction-time coroutine check
        cannot catch a callable whose __call__ is asynchronous.

        Raises:
            ParserContractError: parse raised or returned a non-verdict, and on_parse_error is
                "raise"; a defect must not impersonate a provider classification, since the
                impersonation retries permanent failures and hides the bug behind ordinary-looking
                traffic.
        """
        try:
            result = self.parse(failure)
        except Exception as defect:  # noqa: BLE001 (parse is caller code; any Exception it raises is the defect handled here)
            return self._parse_defect_outcome(
                "parse raised instead of returning a verdict", "parse_raised", defect
            )
        if inspect.iscoroutine(result):
            result.close()
            return self._parse_defect_outcome(
                "parse returned a coroutine instead of a verdict",
                "parse_returned_non_verdict",
                None,
            )
        if not isinstance(result, PauseAll | PauseAllDoNotRetry | RetryThisOne | DoNotRetry):
            return self._parse_defect_outcome(
                f"parse returned {result!r} instead of a verdict",
                "parse_returned_non_verdict",
                None,
            )
        return self._normalized(result)

    def _parse_defect_outcome(
        self, description: str, tag: str, defect: BaseException | None
    ) -> RetryThisOne:
        """Turn one parse contract violation into the configured outcome.

        Raises:
            ParserContractError: on_parse_error is "raise"; defect is chained as __cause__ where
                parse raised one.
        """
        if self.on_parse_error == "raise":
            self.event_counts["parser_contract_error"] += 1
            _logger.error("parse violated its contract: %s", description)
            if defect is None:
                raise ParserContractError(description)
            raise ParserContractError(description) from defect
        self.event_counts[tag] += 1
        _logger.warning("corrected a parse contract violation to RetryThisOne: %s", description)
        return RetryThisOne(retry_after=None)

    def _normalized(self, verdict: Verdict) -> Verdict:
        """Return the verdict with retry_after validated and capped at longest_wait_seconds.

        Runs before either _record or Admission.verdict sees the verdict, so both sides work from
        the same number whatever parse returned.
        An invalid retry_after can corrupt state that outlives the verdict: a negative one plants a
        pause end in the past that a later report reads as history, and a NaN passes a greater-than
        cap into the quiet-step arithmetic.
        """
        if isinstance(verdict, DoNotRetry):
            return verdict
        if verdict.retry_after is None:
            return verdict
        return replace(verdict, retry_after=self._normalized_retry_after(verdict.retry_after))

    def _normalized_retry_after(self, stated: object) -> float | None:
        """Return a valid retry_after capped at longest_wait_seconds, or None.

        The type must be exactly int or float, which excludes bool; anything else becomes None,
        counted, because a much larger or malformed value is more likely a bug or something hostile
        than a real instruction.
        Separate int and float branches: math.isfinite converts an int to float first and raises
        OverflowError on a large enough one, while comparison does not, so a huge int caps cleanly.
        """
        if type(stated) is int:
            if stated <= 0:
                self._count_correction("retry_after_invalid")
                return None
            if stated > self.longest_wait_seconds:
                self._count_correction("retry_after_over_cap")
                return self.longest_wait_seconds
            return float(stated)
        if type(stated) is float:
            if not math.isfinite(stated) or stated <= 0.0:
                self._count_correction("retry_after_invalid")
                return None
            if stated > self.longest_wait_seconds:
                self._count_correction("retry_after_over_cap")
                return self.longest_wait_seconds
            return stated
        self._count_correction("retry_after_invalid")
        return None

    def _count_correction(self, tag: str) -> None:
        """Count and log one wrapper correction."""
        self.event_counts[tag] += 1
        _logger.warning("corrected a parse verdict: %s", tag)

    def _record(self, verdict: Verdict) -> None:
        """Record one parsed failure.

        Only PauseAll and PauseAllDoNotRetry change shared state.
        A report during a pause proposes another capped pause.
        The pauses merge by keeping the later end.
        Reports during one pause do not increase _wait_ceiling.
        The merged pause never shrinks.
        Waiting for admission satisfies every PauseAll.retry_after.
        The pause ends within longest_wait_seconds after the most recent report.
        """
        if not isinstance(verdict, PauseAll | PauseAllDoNotRetry):
            return
        now = self._clock()
        if now < self._pause_until:
            chosen_wait = self._chosen_wait(verdict)
            self._pause_until = max(self._pause_until, now + chosen_wait)
            _logger.info(
                "pause extended by a report with retry_after=%s; %.3f seconds remain; "
                "%d requests waiting",
                verdict.retry_after,
                self._pause_until - now,
                len(self._queue),
            )
            return
        previous_pause_end = self._pause_until
        ceiling_before = self._wait_ceiling
        self._set_wait_ceiling(now, previous_pause_end)
        chosen_wait = self._chosen_wait(verdict)
        self._pause_started_at = now
        self._pause_until = now + chosen_wait
        _logger.info(
            "pause of %.3f seconds started by a report with retry_after=%s; "
            "ceiling %.3f -> %.3f; %d requests waiting",
            chosen_wait,
            verdict.retry_after,
            ceiling_before,
            self._wait_ceiling,
            len(self._queue),
        )

    def _chosen_wait(self, verdict: PauseAll | PauseAllDoNotRetry) -> float:
        """Return how long this report proposes to pause.

        `is None`, not truthiness: the wrapper guarantees a present retry_after is finite and
        greater than zero, and this test stays correct even if that changes.
        """
        if verdict.retry_after is not None:
            return verdict.retry_after
        return _random_up_to(self._wait_ceiling)

    def _set_wait_ceiling(self, now: float, previous_pause_end: float) -> None:
        """Set _wait_ceiling from activity since previous_pause_end.

        The first pause uses minimum_wait_ceiling_seconds.
        Each full quiet_seconds_per_decay_step shrinks _wait_ceiling by wait_multiplier.
        _wait_ceiling never falls below minimum_wait_ceiling_seconds.
        Quiet time includes periods without requests.
        Resumed traffic without a full quiet step grows _wait_ceiling by wait_multiplier.
        _wait_ceiling never exceeds longest_wait_seconds.
        A full quiet step takes precedence over resumed traffic.
        _steps_to_floor prevents wait_multiplier ** steps from overflowing.
        """
        if previous_pause_end == _NEVER:
            self._wait_ceiling = self.minimum_wait_ceiling_seconds
            return
        steps = int((now - previous_pause_end) // self.quiet_seconds_per_decay_step)
        if steps >= 1:
            if steps >= self._steps_to_floor:
                self._wait_ceiling = self.minimum_wait_ceiling_seconds
            else:
                self._wait_ceiling = max(
                    self._wait_ceiling / self.wait_multiplier**steps,
                    self.minimum_wait_ceiling_seconds,
                )
        elif self._last_admission_at > previous_pause_end:
            self._wait_ceiling = min(
                self._wait_ceiling * self.wait_multiplier,
                self.longest_wait_seconds,
            )


class PrivateBackoff:
    """One logical request's waits between admitted() blocks, private to that request.

    The caller-side companion of RetryThisOne: recording that verdict changes no shared state,
    so the failing request spaces its own retries with this.
    Draw next_wait once per RetryThisOne and sleep it between blocks, never inside one.
    A PauseAll costs no draw, because the next entry already waits out the shared pause.
    A wait here overlaps a running pause rather than stacking on it, for the same reason.
    Keep one instance across every attempt for one logical request.
    Discard it when that logical request ends.
    """

    def __init__(self, shared_backoff: SharedBackoff) -> None:
        """Start the private ceiling at minimum_wait_ceiling_seconds."""
        self._ceiling = shared_backoff.minimum_wait_ceiling_seconds
        self._wait_multiplier = shared_backoff.wait_multiplier
        self._longest_wait_seconds = shared_backoff.longest_wait_seconds

    def next_wait(self, retry_after: float | None) -> float:
        """Return one failure's wait in seconds, then grow the ceiling one step.

        The wait is a positive random draw bounded by _ceiling.
        retry_after raises that wait when present.
        Admission.verdict provides a normalized retry_after.
        The normalized retry_after is capped at longest_wait_seconds.
        _ceiling grows by wait_multiplier under the same cap.
        """
        wait = _random_up_to(self._ceiling)
        if retry_after is not None:
            wait = max(wait, retry_after)
        self._ceiling = min(self._ceiling * self._wait_multiplier, self._longest_wait_seconds)
        return wait
