"""Paced request admission for one rate-limit quota.

`SharedBackoff.admitted` applies concurrency, request-rate, queue-order, and shared-pause constraints.
`PauseAll` and `PauseAllDoNotRetry` pause the quota.
`RetryThisOne` and `DoNotRetry` leave shared state unchanged.
"""

import asyncio
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

    `retry_after` is the provider-specified wait in seconds.
    `retry_after` is `None` when the provider specified no wait.
    Recording this verdict starts or extends the shared pause.
    """

    retry_after: float | None
    kind: Literal["pause_all"] = "pause_all"


@dataclass(frozen=True, kw_only=True)
class RetryThisOne:
    """Worth retrying, with no sign the provider wants less traffic overall.

    `retry_after` is a wait floor for this request's next attempt.
    `retry_after` is `None` when the provider specified no wait.
    Recording this verdict changes no shared state.
    """

    retry_after: float | None
    kind: Literal["retry_this_one"] = "retry_this_one"


@dataclass(frozen=True, kw_only=True)
class DoNotRetry:
    """Tell the caller to stop retrying this request."""

    kind: Literal["do_not_retry"] = "do_not_retry"


@dataclass(frozen=True, kw_only=True)
class PauseAllDoNotRetry:
    """Pause shared requests and stop retrying this request.

    `retry_after` supplies the shared pause duration when present.
    """

    retry_after: float | None
    kind: Literal["pause_all_do_not_retry"] = "pause_all_do_not_retry"


type Verdict = PauseAll | PauseAllDoNotRetry | RetryThisOne | DoNotRetry
"""What one provider failure means for one request and its rate-limit quota.

`DoNotRetry` and `PauseAllDoNotRetry` are terminal.
Callers stop retrying this request for either verdict.
"""


def _random_up_to(ceiling: float) -> float:
    """Draw a wait greater than zero and no larger than ceiling.

    `1 - random.random()` lies in `(0, 1]`, so the pause is never zero.
    """
    return ceiling * (1.0 - random.random())


def _validated_positive_float(name: str, value: float) -> float:
    """Return value as a positive finite float.

    `bool` is rejected explicitly because it subclasses `int`.

    Raises:
        ValueError: `value` is boolean, non-finite, non-positive, or too large to convert to float.
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
    """Represent one `admitted()` block. Entry waits until the request may start. Exit reports how the request ended.

    `verdict` is `None` until a `failure_types` exception exits the block.
    `verdict` then holds the normalized `parse` result.
    Build `Admission` only through `SharedBackoff.admitted`, which validates `budget` first.
    """

    def __init__(self, shared_backoff: "SharedBackoff", budget_seconds: float | None) -> None:
        """Bind the block to its SharedBackoff and store the validated budget."""
        self._shared_backoff = shared_backoff
        self._budget_seconds = budget_seconds
        self.verdict: Verdict | None = None

    async def __aenter__(self) -> "Admission":
        """Acquire a permit when `max_concurrent_requests` is set.

        Once this returns, the request is admitted. A later pause does not revoke that admission.
        Cancellation during entry removes the request from the queue. Cancellation returns any acquired permit.

        Raises:
            GaveUpWaiting: `budget` expired before admission.
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
        """Parse and record a provider failure before releasing the permit.

        The permit returns even when parsing raises.

        Raises:
            ParserContractError: `parse` violates its contract and `on_parse_error="raise"`.
            BaseException: The admitted block raises it.
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
    """Coordinate request starts for one rate-limit quota.

    Run each complete provider request inside `admitted`.
    Raise `failure_types` inside that block so provider pushback updates shared state.
    Use `PrivateBackoff` between `RetryThisOne` attempts.
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
        """Validate configuration. Initialize an unpaused `SharedBackoff`.

        `parse` maps each `failure_types` exception to `Verdict`.
        `max_concurrent_requests=None` applies no concurrency limit.
        `longest_wait_seconds` caps generated waits and `retry_after`.

        Args:
            parse: The provider failure parser.
            failure_types: The exception types that `parse` accepts.
            max_concurrent_requests: The request concurrency limit, or `None`.
            minimum_wait_ceiling_seconds: The minimum private wait ceiling in seconds.
            longest_wait_seconds: The maximum generated or provider-specified wait in seconds.
            wait_multiplier: The factor that grows or shrinks the wait ceiling.
            quiet_seconds_per_decay_step: The quiet interval that shrinks the wait ceiling once.
            max_request_starts_per_second: The request-start rate limit.
            on_parse_error: The action when `parse` raises.

        Raises:
            ValueError: A numeric setting is boolean, non-finite, or non-positive.
            ValueError: `1 / max_request_starts_per_second` is non-finite.
            ValueError: `longest_wait_seconds / minimum_wait_ceiling_seconds` is non-finite.
            ValueError: `wait_multiplier` is at most one.
            ValueError: `longest_wait_seconds` is below `minimum_wait_ceiling_seconds`.
            ValueError: `max_concurrent_requests` is boolean or below one.
            ValueError: `failure_types` is empty or contains `Exception`.
        """
        self.minimum_wait_ceiling_seconds: float = _validated_positive_float(
            "minimum_wait_ceiling_seconds", minimum_wait_ceiling_seconds
        )
        self.longest_wait_seconds: float = _validated_positive_float(
            "longest_wait_seconds", longest_wait_seconds
        )
        self.wait_multiplier: float = _validated_positive_float("wait_multiplier", wait_multiplier)
        self.quiet_seconds_per_decay_step: float = _validated_positive_float(
            "quiet_seconds_per_decay_step", quiet_seconds_per_decay_step
        )
        self.max_request_starts_per_second: float = _validated_positive_float(
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
        if not failure_types:
            raise ValueError(
                "failure_types must not be empty: the exit would parse nothing and record nothing"
            )
        if Exception in failure_types:
            raise ValueError("failure_types must not contain Exception")
        self.parse: Callable[[Exception], Verdict] = parse
        self.failure_types: tuple[type[Exception], ...] = failure_types
        self._max_concurrent_requests = max_concurrent_requests
        self.on_parse_error: Literal["raise", "retry_this_one"] = on_parse_error
        self._steps_to_floor = math.ceil(math.log(ceiling_ratio) / math.log(self.wait_multiplier))
        """Quiet steps after which the ceiling has reached the floor, whatever it started at.

        This bounds the decay exponent because the ceiling never exceeds longest_wait_seconds.
        Afterward, the answer is minimum_wait_ceiling_seconds.
        For fewer steps, wait_multiplier ** steps cannot exceed the checked ceiling ratio.
        """
        self._pause_until = _NEVER
        self._pause_started_at = _NEVER
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
        self._admit_timer: asyncio.TimerHandle | None = None
        """Wakes _admit_waiting when the front of the queue becomes admissible."""
        self._clock: Callable[[], float] = time.monotonic
        """The forward-only clock every deadline reads."""
        self.event_counts: Counter[str] = Counter()
        """How often each noteworthy entry or exit event occurred, by tag.

        The correction tags are `"retry_after_invalid"` and `"retry_after_over_cap"`.
        `on_parse_error="retry_this_one"` also permits the `"parse_raised"` correction tag.
        The failure tags are `"gave_up_waiting"` and `"parser_contract_error"`.
        """

    @property
    def max_concurrent_requests(self) -> int | None:
        """The number of requests allowed inside admitted() blocks at once, or None for no bound.

        `__init__` fixes both this value and the permit count.
        The property is read-only to keep them consistent.
        """
        return self._max_concurrent_requests

    def admitted(self, *, budget: float | None = None) -> Admission:
        """Return an `Admission` block for one attempt.

        `budget` limits permit acquisition and admission waits.
        `budget=None` permits an indefinite wait.

        Args:
            budget: The admission wait budget in seconds, or `None`.

        Raises:
            ValueError: `budget` is boolean, non-finite, or non-positive.
        """
        budget_seconds = None if budget is None else _validated_positive_float("budget", budget)
        return Admission(self, budget_seconds)

    async def _acquire_permit(self) -> None:
        """Hold one permit after earlier waiters.

        Do nothing when there is no concurrency bound.

        Raises:
            asyncio.CancelledError: the wait was cancelled; no permit is held.
        """
        if self._permits is None:
            return
        _ = await self._permits.acquire()

    def _release_permit(self) -> None:
        """Return one permit and wake the longest-waiting live waiter.

        Do nothing when there is no concurrency bound.
        """
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
        """Admit the front of the queue or schedule _admit_waiting for its earliest admission.

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
        """Schedule _admit_waiting for the front's admission time, replacing any scheduled timer."""
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

        Queue depth cannot exceed `max_concurrent_requests` when configured.
        Otherwise it cannot exceed the worker pool size.
        A full queue means every permit or worker is idle.
        Upstream bounds must report additional waiting work.
        """
        if self._pause_until == _NEVER or self._last_admission_at > self._pause_until:
            return
        _logger.info(
            "pause of %.3f seconds ended with %d requests waiting",
            self._pause_until - self._pause_started_at,
            len(self._queue),
        )

    def _checked_parse(self, failure: Exception) -> Verdict:
        """Call parse, handle raised exceptions, and normalize the verdict.

        Raises:
            ParserContractError: parse raised and on_parse_error is "raise".
        """
        try:
            result = self.parse(failure)
        except Exception as defect:  # noqa: BLE001 (parse is caller code; any Exception it raises is the defect handled here)
            return self._parse_error_outcome(defect)
        return self._normalized(result)

    def _parse_error_outcome(self, defect: Exception) -> RetryThisOne:
        """Apply on_parse_error to an exception raised by parse.

        Raises:
            ParserContractError: on_parse_error is "raise".
        """
        description = "parse raised instead of returning a verdict"
        if self.on_parse_error == "raise":
            self.event_counts["parser_contract_error"] += 1
            _logger.error("parse violated its contract: %s", description)
            raise ParserContractError(description) from defect
        self.event_counts["parse_raised"] += 1
        _logger.warning("corrected a parse contract violation to RetryThisOne: %s", description)
        return RetryThisOne(retry_after=None)

    def _normalized(self, verdict: Verdict) -> Verdict:
        """Return the verdict with retry_after validated and capped at longest_wait_seconds.

        Runs before `_record` or `Admission.verdict` reads the verdict.
        Both therefore use the same normalized value.
        A negative `retry_after` creates a past pause end.
        A NaN bypasses the cap and corrupts quiet-step arithmetic.
        """
        if isinstance(verdict, DoNotRetry):
            return verdict
        if verdict.retry_after is None:
            return verdict
        return replace(verdict, retry_after=self._normalized_retry_after(verdict.retry_after))

    def _normalized_retry_after(self, stated: float) -> float | None:
        """Return a valid retry_after capped at longest_wait_seconds, or None.

        Accept exactly `int` or `float`, excluding `bool`.
        Count other values and return `None`.
        Check integers before `math.isfinite` so huge integers cap without `OverflowError`.
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

        Test `is None` because `retry_after` presence determines the branch.
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
    """Generate private waits between one request's `RetryThisOne` attempts.

    Keep one instance for the request's complete retry loop.
    Sleep returned waits outside `admitted` blocks.
    """

    def __init__(self, shared_backoff: SharedBackoff) -> None:
        """Start the private ceiling at minimum_wait_ceiling_seconds."""
        self._ceiling = shared_backoff.minimum_wait_ceiling_seconds
        self._wait_multiplier = shared_backoff.wait_multiplier
        self._longest_wait_seconds = shared_backoff.longest_wait_seconds

    def next_wait(self, retry_after: float | None) -> float:
        """Return one failure's wait in seconds, then grow the ceiling one step.

        The wait is a positive random draw bounded by `_ceiling`.
        `retry_after` raises that wait when present.
        The normalized `retry_after` is capped at `longest_wait_seconds`.
        """
        wait = _random_up_to(self._ceiling)
        if retry_after is not None:
            wait = max(wait, retry_after)
        self._ceiling = min(self._ceiling * self._wait_multiplier, self._longest_wait_seconds)
        return wait
