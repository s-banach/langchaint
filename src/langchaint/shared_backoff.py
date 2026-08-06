"""A shared pause and paced admission for one provider backpressure domain.

SharedBackoff has one control action: holding a request from starting until a deadline its whole domain shares.
A domain is the set of requests the caller routes through one instance, usually one model on one account.
A request enters `admitted()`, the async-with block spanning one attempt.
Entry acquires a capacity permit when `capacity` is set, then admission; normally both are immediate.
After the provider pushes back, every request in the domain waits at entry until the shared pause ends.
When the pause ends, waiting requests are released in the order they joined, spaced by `admission_gap`.
Exit parses a provider failure into a verdict, records it, then returns the permit, in that order by position.
SharedBackoff decides no retries and counts no tokens.
It also bounds no pending work: it cannot tell an unadmitted request from one that has not entered yet.
The bound on pending work belongs to whatever spawns the work.

The three verdicts are PauseAll, RetryThisOne, and DoNotRetry; only PauseAll changes shared state.
The verdict comes from the status and the error type; a `retry-after` header never sets it.
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


type Verdict = PauseAll | RetryThisOne | DoNotRetry


def _random_up_to(ceiling: float) -> float:
    """Draw a wait greater than zero and no larger than ceiling.

    1 - random.random() lies in (0, 1], so the draw is never the zero-length pause
    random.uniform permits.
    """
    return ceiling * (1.0 - random.random())


def _validated_positive_seconds(name: str, value: float) -> float:
    """Return value as a positive finite float, under the acceptance rule that the numeric settings and budget share.

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
        """Acquire a capacity permit when capacity is set, then admission; normally both are immediate.

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
    """The shared pause, the admission queue, and the capacity permits of one domain.

    Route every request in the domain through one instance, first attempts and retries alike.

    The wait ceiling is a ceiling, not a wait: each pause of our own choosing lasts a fresh random
    number greater than zero and no larger than it, so one wait can fall far below minimum_wait_ceiling.
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

    def __init__(  # noqa: PLR0913 (the settings table travels whole: five numeric settings plus parse, failure_types, capacity, on_parse_error)
        self,
        *,
        parse: Callable[[Exception], Verdict],
        failure_types: tuple[type[Exception], ...],
        capacity: int | None,
        minimum_wait_ceiling: float = 1.0,
        longest_wait: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_per_decay_step: float = 60.0,
        admission_gap: float = 0.02,
        on_parse_error: Literal["raise", "retry_this_one"] = "raise",
    ) -> None:
        """Validate the configuration and start with no pause, no queue, and every permit free.

        parse maps each failure_types exception to a verdict; the exit calls it only through the
        checking wrapper, so nothing else in the object sees a status code or an exception's contents.
        parse is synchronous, so the raised failure must already carry what parse needs; await any
        body reading inside the block before raising.
        failure_types are the exception types the exit parses; provide narrow provider-failure classes.
        capacity is the number of requests allowed inside admitted() blocks at once, or None when a
        fixed worker pool upstream already bounds concurrency; a permit held idle through a pause is
        acceptable because every other request in the domain is paused too.
        minimum_wait_ceiling is where the wait ceiling starts and decays back to.
        longest_wait caps the wait ceiling and any retry_after.
        wait_multiplier is how much the ceiling grows or shrinks in one step.
        quiet_per_decay_step is how long without a pause earns one shrink of the ceiling.
        admission_gap is the smallest interval between two admissions, so it caps the rate of
        request starts while demand is queued; check that cap against your own workload.
        on_parse_error is what a parse contract violation becomes: "raise" (the default) raises
        ParserContractError, and "retry_this_one" corrects it to RetryThisOne with no retry_after.

        Raises:
            ValueError: a numeric setting fails the acceptance rule (not a bool, finite,
                greater than zero); wait_multiplier is not greater than 1; longest_wait is below
                minimum_wait_ceiling; longest_wait / minimum_wait_ceiling is not a finite float,
                which the decay arithmetic assumes; capacity is a bool or an int below 1;
                on_parse_error is neither accepted string; failure_types is empty,
                which would make the exit parse nothing and record nothing; a failure_types entry
                is not a strict subclass of Exception (Exception itself would convert nearly every
                application bug into an apparent provider failure, and a type outside Exception,
                such as asyncio.CancelledError, would pause the domain over a Ctrl-C); or parse is
                a coroutine function, which the synchronous exit could never await.
        """
        self.minimum_wait_ceiling = _validated_positive_seconds(
            "minimum_wait_ceiling", minimum_wait_ceiling
        )
        self.longest_wait = _validated_positive_seconds("longest_wait", longest_wait)
        self.wait_multiplier = _validated_positive_seconds("wait_multiplier", wait_multiplier)
        self.quiet_per_decay_step = _validated_positive_seconds(
            "quiet_per_decay_step", quiet_per_decay_step
        )
        self.admission_gap = _validated_positive_seconds("admission_gap", admission_gap)
        if self.wait_multiplier <= 1.0:
            raise ValueError(f"wait_multiplier must be greater than 1, got {wait_multiplier!r}")
        if self.longest_wait < self.minimum_wait_ceiling:
            raise ValueError(
                f"longest_wait must be at least minimum_wait_ceiling, "
                f"got {longest_wait!r} < {minimum_wait_ceiling!r}"
            )
        ceiling_ratio = self.longest_wait / self.minimum_wait_ceiling
        if not math.isfinite(ceiling_ratio):
            raise ValueError(
                "longest_wait / minimum_wait_ceiling must be a finite float, "
                f"got {ceiling_ratio!r} from {longest_wait!r} / {minimum_wait_ceiling!r}"
            )
        if capacity is not None and (isinstance(capacity, bool) or capacity < 1):
            raise ValueError(f"capacity must be None or a positive int, got {capacity!r}")
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
        self.capacity = capacity
        self.on_parse_error: Literal["raise", "retry_this_one"] = on_parse_error
        self._steps_to_floor = math.ceil(math.log(ceiling_ratio) / math.log(self.wait_multiplier))
        """Quiet steps after which the ceiling has reached the floor, whatever it started at.

        Bounds the decay exponent, since the ceiling never exceeds longest_wait: past this many
        steps the answer is minimum_wait_ceiling by definition, and below it
        wait_multiplier ** steps cannot exceed the ceiling ratio checked finite above.
        """
        self._pause_until = _NEVER
        """When the current pause ends; once it is over, still the end of the previous pause."""
        self._pause_started_at = _NEVER
        """When the current pause began; a metric for logging, read by no decision."""
        self._wait_ceiling = self.minimum_wait_ceiling
        """Longest pause this object will currently choose for itself."""
        self._last_admission_at = _NEVER
        """When a request was last admitted.

        Enforces admission_gap, and answers whether traffic resumed after the previous pause.
        """
        self._queue: deque[asyncio.Future[None]] = deque()
        """Requests waiting for admission, released in the order they joined."""
        self._capacity_permits = None if capacity is None else asyncio.Semaphore(capacity)
        """The capacity permits; None when capacity is None.

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
        budget_seconds = None if budget is None else _validated_positive_seconds("budget", budget)
        return Admission(self, budget_seconds)

    async def _acquire_permit(self) -> None:
        """Hold one capacity permit, waiting behind earlier waiters; no-op when capacity is None.

        Raises:
            asyncio.CancelledError: the wait was cancelled; no permit is held.
        """
        if self._capacity_permits is None:
            return
        _ = await self._capacity_permits.acquire()

    def _release_permit(self) -> None:
        """Return one permit, waking the longest-waiting live waiter; no-op when capacity is None."""
        if self._capacity_permits is None:
            return
        self._capacity_permits.release()

    async def _wait_turn(self) -> None:
        """Wait in the admission queue until the shared pause and admission_gap admit this request.

        Cancellation before the grant removes the request from the queue.
        Cancellation after the grant may have consumed one admission_gap, which is wasted but
        harmless; the caller returns the permit.

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

        Admission requires that no pause is running and at least admission_gap has passed since the
        previous admission; granting records the moment in _last_admission_at, so a burst released
        after a pause is spread one admission per gap.
        Spent entries (cancelled waiters) at the front are dropped, never granted.
        """
        while self._queue:
            if self._queue[0].done():
                _ = self._queue.popleft()
                continue
            now = self._clock()
            admissible_at = max(self._pause_until, self._last_admission_at + self.admission_gap)
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

        Queue depth here cannot exceed capacity, or the worker pool's size when capacity is None,
        so a full queue means every permit or worker is idle; it says nothing about work waiting
        further upstream, which whatever bounds that work has to report.
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
        if not isinstance(result, PauseAll | RetryThisOne | DoNotRetry):
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
        """Return the verdict with its retry_after validated and capped at longest_wait.

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
        """Return the stated retry_after as a positive finite float capped at longest_wait, or None.

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
            if stated > self.longest_wait:
                self._count_correction("retry_after_over_cap")
                return self.longest_wait
            return float(stated)
        if type(stated) is float:
            if not math.isfinite(stated) or stated <= 0.0:
                self._count_correction("retry_after_invalid")
                return None
            if stated > self.longest_wait:
                self._count_correction("retry_after_over_cap")
                return self.longest_wait
            return stated
        self._count_correction("retry_after_invalid")
        return None

    def _count_correction(self, tag: str) -> None:
        """Count and log one wrapper correction."""
        self.event_counts[tag] += 1
        _logger.warning("corrected a parse verdict: %s", tag)

    def _record(self, verdict: Verdict) -> None:
        """Record one parsed failure; only PauseAll changes shared state.

        A report during a pause starts a fresh pause of its own, at most longest_wait long, and the
        two merge by keeping the later end; the ceiling is untouched, because one burst of trouble
        usually produces several reports within a second, and treating each as fresh evidence would
        multiply the wait for a single event.
        The merged pause never shrinks, which is one of the properties letting a PauseAll carry no
        per-request floor: the failing request's own wait is one of the proposals the pause end is
        the running maximum of, so waiting for admission already satisfies it.
        So at every moment the pause ends at most longest_wait after the most recent report: every
        merged-in wait is capped there, by the wrapper for a retry_after and by the ceiling for a
        chosen wait.
        """
        if not isinstance(verdict, PauseAll):
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

    def _chosen_wait(self, verdict: PauseAll) -> float:
        """Return how long this report proposes to pause.

        `is None`, not truthiness: the wrapper guarantees a present retry_after is finite and
        greater than zero, and this test stays correct even if that changes.
        """
        if verdict.retry_after is not None:
            return verdict.retry_after
        return _random_up_to(self._wait_ceiling)

    def _set_wait_ceiling(self, now: float, previous_pause_end: float) -> None:
        """Set the ceiling a fresh pause draws under, from what happened since the previous pause.

        Nothing has gone wrong yet, so guess small: until the first pause the ceiling sits at
        minimum_wait_ceiling.
        Time has passed without trouble, so give back what the trouble cost us: the ceiling shrinks
        one step per full quiet_per_decay_step since the previous pause ended, never below the
        floor, so a stray refusal after a busy afternoon does not cost the afternoon's whole ceiling.
        Quiet time also accrues while we send nothing, deliberately: requiring traffic first would
        leave an idle application holding a longest_wait ceiling indefinitely.
        Traffic resumed and we were refused again, so the last guess was too small: grow the
        ceiling, capped at longest_wait.
        The guard is on traffic having resumed, not on which request reported: a refusal is
        evidence about the provider's present state and counts whoever observed it, but one
        arriving before any traffic resumed is most likely another delayed result of the burst that
        caused the previous pause.
        When a full quiet step has passed and traffic has also resumed, decay wins, deliberately: a
        refusal after a full quiet step is a fresh incident, priced by the pause it starts, and not
        evidence that a guess made a step or more ago was too small.
        The short-circuit past _steps_to_floor is what keeps wait_multiplier ** steps from
        overflowing after a long quiet spell.
        The ceiling is only read at the instant a pause begins, so computing the decay here gives
        the same answer as decaying continuously, with no timer to run.
        """
        if previous_pause_end == _NEVER:
            self._wait_ceiling = self.minimum_wait_ceiling
            return
        steps = int((now - previous_pause_end) // self.quiet_per_decay_step)
        if steps >= 1:
            if steps >= self._steps_to_floor:
                self._wait_ceiling = self.minimum_wait_ceiling
            else:
                self._wait_ceiling = max(
                    self._wait_ceiling / self.wait_multiplier**steps,
                    self.minimum_wait_ceiling,
                )
        elif self._last_admission_at > previous_pause_end:
            self._wait_ceiling = min(
                self._wait_ceiling * self.wait_multiplier,
                self.longest_wait,
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
        """Start the private ceiling at the domain's minimum_wait_ceiling."""
        self._ceiling = shared_backoff.minimum_wait_ceiling
        self._wait_multiplier = shared_backoff.wait_multiplier
        self._longest_wait = shared_backoff.longest_wait

    def next_wait(self, retry_after: float | None) -> float:
        """Return one failure's wait in seconds, then grow the ceiling one step.

        The wait is a fresh random draw greater than zero and no larger than the private ceiling,
        raised to retry_after where the verdict carried one, so a server-stated floor is honored.
        Read retry_after off Admission.verdict, which is normalized, so it is already capped at
        longest_wait; the ceiling grows by wait_multiplier under the same cap.
        """
        wait = _random_up_to(self._ceiling)
        if retry_after is not None:
            wait = max(wait, retry_after)
        self._ceiling = min(self._ceiling * self._wait_multiplier, self._longest_wait)
        return wait
