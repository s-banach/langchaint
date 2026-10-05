"""Paced request admission for one rate-limit quota.

`SharedBackoff.admitted` applies concurrency, request-rate, queue-order, and shared-pause constraints.
The caller maps each failed request to a `RequestFailure` and passes it to `Admission.record`.
A `RequestFailure` with `pauses_quota` pauses the quota, and any other leaves shared state unchanged.
"""

import asyncio
import logging
import math
import random
import time
from collections import Counter, deque
from dataclasses import replace
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from langchaint.common.exceptions import GaveUpWaitingError
from langchaint.common.request_failure import RequestFailure

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger("langchaint.shared_backoff")

_NEVER = float("-inf")
"""The moment before every other: the initial pause end and the initial admission time."""


def _random_up_to(ceiling: float, draw: float) -> float:
    """Map `draw` from `[0, 1)` to a wait greater than zero and no larger than ceiling.

    Callers pass `random.random()` as `draw`.
    `1 - draw` lies in `(0, 1]`, so the wait is never zero.
    """
    return ceiling * (1.0 - draw)


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
    """Represent one `admitted()` block. Entry waits until the request may start. Exit returns the permit.

    Build `Admission` only through `SharedBackoff.admitted`, which validates `budget_seconds` first.
    """

    def __init__(self, shared_backoff: "SharedBackoff", budget_seconds: float | None) -> None:
        """Bind the block to its SharedBackoff and store the validated budget."""
        self._shared_backoff = shared_backoff
        self._budget_seconds = budget_seconds

    async def __aenter__(self) -> "Admission":
        """Wait in the queue until the request may start.

        Once this returns, the request is admitted and holds a permit when `max_concurrent_requests` is set.
        A later pause does not revoke that admission.
        Cancellation during entry removes the request from the queue and returns any acquired permit.

        Raises:
            GaveUpWaitingError: `budget_seconds` expired before admission.
        """
        try:
            async with asyncio.timeout(self._budget_seconds):
                await self._shared_backoff._wait_turn()
        except TimeoutError:
            self._shared_backoff.event_counts["gave_up_waiting"] += 1
            _logger.info(
                "gave up waiting for admission after a budget of %s seconds", self._budget_seconds
            )
            raise GaveUpWaitingError(
                f"gave up waiting for admission after a budget of {self._budget_seconds} seconds"
            ) from None
        return self

    def record(self, request_failure: RequestFailure) -> RequestFailure:
        """Apply the request's failure to shared state and return it with `retry_after_seconds` normalized.

        Call it inside the block, so a pause starts before the permit passes to a waiter.
        """
        return self._shared_backoff._recorded(request_failure)

    def _release(self) -> None:
        """Return the permit.

        `StreamHandle` calls this directly because its admission spans several of its methods.
        """
        self._shared_backoff._release_permit()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Return the permit.

        Raises:
            BaseException: The admitted block raises it.
        """
        self._release()
        return False


class SharedBackoff:
    """Coordinate request starts for one rate-limit quota.

    Run each complete provider request inside `admitted`.
    Pass each failed request's `RequestFailure` to `Admission.record` inside that block.
    Provider pushback then updates shared state.
    Use `PrivateBackoff` between retries of a failure that does not pause the quota.
    Share one instance among every `LLM` and `EmbeddingModel` that draws on the same rate-limit quota.
    """

    def __init__(
        self,
        *,
        max_concurrent_requests: int | None = 8,
        min_wait_ceiling_seconds: float = 1.0,
        max_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
        max_request_starts_per_second: float = 50.0,
    ) -> None:
        """Validate configuration. Initialize an unpaused `SharedBackoff`.

        `max_concurrent_requests=None` applies no concurrency limit.
        `max_wait_seconds` caps generated waits and `retry_after_seconds`.

        Args:
            max_concurrent_requests: The request concurrency limit, or `None`.
            min_wait_ceiling_seconds: The minimum private wait ceiling in seconds.
            max_wait_seconds: The maximum generated or provider-specified wait in seconds.
            wait_multiplier: The factor that grows or shrinks the wait ceiling.
            quiet_seconds_per_decay_step: The quiet interval that shrinks the wait ceiling once.
            max_request_starts_per_second: The request-start rate limit.

        Raises:
            ValueError: A numeric setting is boolean, non-finite, or non-positive.
            ValueError: `1 / max_request_starts_per_second` is non-finite.
            ValueError: `max_wait_seconds / min_wait_ceiling_seconds` is non-finite.
            ValueError: `wait_multiplier` is at most one.
            ValueError: `max_wait_seconds` is below `min_wait_ceiling_seconds`.
            ValueError: `max_concurrent_requests` is boolean or below one.
        """
        self.min_wait_ceiling_seconds: float = _validated_positive_float(
            "min_wait_ceiling_seconds", min_wait_ceiling_seconds
        )
        self.max_wait_seconds: float = _validated_positive_float(
            "max_wait_seconds", max_wait_seconds
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
        if self.max_wait_seconds < self.min_wait_ceiling_seconds:
            raise ValueError(
                "max_wait_seconds must be at least min_wait_ceiling_seconds, "
                f"got {max_wait_seconds!r} < {min_wait_ceiling_seconds!r}"
            )
        ceiling_ratio = self.max_wait_seconds / self.min_wait_ceiling_seconds
        if not math.isfinite(ceiling_ratio):
            raise ValueError(
                "max_wait_seconds / min_wait_ceiling_seconds must be finite, "
                f"got {ceiling_ratio!r} from {max_wait_seconds!r} / "
                f"{min_wait_ceiling_seconds!r}"
            )
        if max_concurrent_requests is not None and (
            isinstance(max_concurrent_requests, bool) or max_concurrent_requests < 1
        ):
            raise ValueError(
                f"max_concurrent_requests must be None or a positive int, "
                f"got {max_concurrent_requests!r}"
            )
        self._max_concurrent_requests = max_concurrent_requests
        self._steps_to_floor = math.ceil(math.log(ceiling_ratio) / math.log(self.wait_multiplier))
        """Quiet steps after which the ceiling has reached the floor, whatever it started at.

        This bounds the decay exponent because the ceiling never exceeds max_wait_seconds.
        Afterward, the answer is min_wait_ceiling_seconds.
        For fewer steps, wait_multiplier ** steps cannot exceed the checked ceiling ratio.
        """
        self._pause_until = _NEVER
        self._pause_started_at = _NEVER
        self._wait_ceiling = self.min_wait_ceiling_seconds
        """Longest pause this object will currently choose for itself."""
        self._last_admission_at = _NEVER
        """When a request was last admitted.

        Enforces _seconds_between_request_starts.
        Also answers whether traffic resumed after the previous pause.
        """
        self._queue: deque[asyncio.Future[None]] = deque()
        """Requests waiting for admission, released in the order they joined."""
        self._permits_held = 0
        """Admitted requests that have not yet exited their block.

        _admit_waiting keeps this at or below max_concurrent_requests when that is set.
        A counter binds no event loop, so one SharedBackoff serves consecutive event loops.
        """
        self._admit_timer: asyncio.TimerHandle | None = None
        """Wakes _admit_waiting when the front of the queue becomes admissible."""
        self._clock: Callable[[], float] = time.monotonic
        """The forward-only clock every deadline reads."""
        self.event_counts: Counter[str] = Counter()
        """How often each noteworthy entry or exit event occurred, by tag.

        The correction tags are `"retry_after_invalid"` and `"retry_after_over_cap"`.
        The failure tag is `"gave_up_waiting"`.
        """

    @property
    def max_concurrent_requests(self) -> int | None:
        """The number of requests allowed inside admitted() blocks at once, or None for no bound.

        `__init__` validates and fixes this value, so the property is read-only.
        """
        return self._max_concurrent_requests

    def admitted(self, *, budget_seconds: float | None = None) -> Admission:
        """Return an `Admission` block for one request.

        `budget_seconds` limits the admission wait.
        `budget_seconds=None` permits an indefinite wait.

        Args:
            budget_seconds: The admission wait budget in seconds, or `None`.

        Raises:
            ValueError: `budget_seconds` is boolean, non-finite, or non-positive.
        """
        validated_budget_seconds = (
            None
            if budget_seconds is None
            else _validated_positive_float("budget_seconds", budget_seconds)
        )
        return Admission(self, validated_budget_seconds)

    def _release_permit(self) -> None:
        """Return one permit and admit the front of the queue when it may start."""
        self._permits_held -= 1
        self._admit_waiting()

    async def _wait_turn(self) -> None:
        """Wait until a permit, the shared pause, and the request-start interval permit admission.

        Cancellation before the grant removes the request from the queue.
        Cancellation after the grant returns the permit and may consume one request-start interval.

        Raises:
            asyncio.CancelledError: the wait was cancelled; the request is out of the queue and holds no permit.
        """
        granted: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.append(granted)
        self._admit_waiting()
        try:
            await granted
        except asyncio.CancelledError:
            if granted.done() and not granted.cancelled():
                self._release_permit()
            else:
                try:
                    self._queue.remove(granted)
                except ValueError:
                    pass  # _admit_waiting already dropped this spent entry from the front
            raise

    def _admit_waiting(self) -> None:
        """Admit the front of the queue or schedule _admit_waiting for its earliest admission.

        Admission requires a free permit, no active pause, and one elapsed request-start interval.
        No timer is armed while every permit is held because _release_permit calls this method.
        Granting takes a permit and records the moment in _last_admission_at.
        A queued burst starts at max_request_starts_per_second.
        Spent entries (cancelled waiters) at the front are dropped, never granted.
        """
        while self._queue:
            if self._queue[0].done():
                _ = self._queue.popleft()
                continue
            if (
                self._max_concurrent_requests is not None
                and self._permits_held >= self._max_concurrent_requests
            ):
                return
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
            self._permits_held += 1
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

        The queue holds every request waiting for admission, including requests waiting for a permit.
        """
        if self._pause_until == _NEVER or self._last_admission_at > self._pause_until:
            return
        _logger.info(
            "pause of %.3f seconds ended with %d requests waiting",
            self._pause_until - self._pause_started_at,
            len(self._queue),
        )

    def _recorded(self, request_failure: RequestFailure) -> RequestFailure:
        """Normalize one failed request's failure, apply it to shared state, and return the normalized failure.

        `Admission.record` calls this inside its block.
        `StreamHandle` also calls it for a failure that may arrive after its drained stream returned the permit.
        """
        normalized = self._normalized(request_failure)
        self._record(normalized)
        return normalized

    def _normalized(self, request_failure: RequestFailure) -> RequestFailure:
        """Return the failure with retry_after_seconds validated and capped at max_wait_seconds.

        Runs before `_record` or the retry loop reads the failure.
        Both therefore use the same normalized value.
        A negative `retry_after_seconds` creates a past pause end.
        A NaN bypasses the cap and corrupts quiet-step arithmetic.
        """
        if request_failure.retry_after_seconds is None:
            return request_failure
        return replace(
            request_failure,
            retry_after_seconds=self._normalized_retry_after_seconds(
                request_failure.retry_after_seconds
            ),
        )

    def _normalized_retry_after_seconds(self, stated: float) -> float | None:
        """Return a valid `retry_after_seconds` capped at `max_wait_seconds`, or None.

        Accept exactly `int` or `float`, excluding `bool`.
        Count other values and return `None`.
        Check integers before `math.isfinite` so huge integers cap without `OverflowError`.
        """
        if type(stated) is int:
            if stated <= 0:
                self._count_correction("retry_after_invalid")
                return None
            if stated > self.max_wait_seconds:
                self._count_correction("retry_after_over_cap")
                return self.max_wait_seconds
            return float(stated)
        if type(stated) is float:
            if not math.isfinite(stated) or stated <= 0.0:
                self._count_correction("retry_after_invalid")
                return None
            if stated > self.max_wait_seconds:
                self._count_correction("retry_after_over_cap")
                return self.max_wait_seconds
            return stated
        self._count_correction("retry_after_invalid")
        return None

    def _count_correction(self, tag: str) -> None:
        """Count and log one wrapper correction."""
        self.event_counts[tag] += 1
        _logger.warning("corrected a request failure: %s", tag)

    def _record(self, request_failure: RequestFailure) -> None:
        """Record one failed request.

        Only a failure with `pauses_quota` changes shared state.
        A report during a pause proposes another capped pause.
        The pauses merge by keeping the later end.
        Reports during one pause do not increase _wait_ceiling.
        The merged pause never shrinks.
        Waiting for admission satisfies every pausing failure's retry_after_seconds.
        The pause ends within max_wait_seconds after the most recent report.
        """
        if not request_failure.pauses_quota:
            return
        now = self._clock()
        if now < self._pause_until:
            chosen_wait = self._chosen_wait(request_failure)
            self._pause_until = max(self._pause_until, now + chosen_wait)
            _logger.info(
                "pause extended by a report with retry_after_seconds=%s; %.3f seconds remain; "
                "%d requests waiting",
                request_failure.retry_after_seconds,
                self._pause_until - now,
                len(self._queue),
            )
            return
        previous_pause_end = self._pause_until
        ceiling_before = self._wait_ceiling
        self._set_wait_ceiling(now, previous_pause_end)
        chosen_wait = self._chosen_wait(request_failure)
        self._pause_started_at = now
        self._pause_until = now + chosen_wait
        _logger.info(
            "pause of %.3f seconds started by a report with retry_after_seconds=%s; "
            "ceiling %.3f -> %.3f; %d requests waiting",
            chosen_wait,
            request_failure.retry_after_seconds,
            ceiling_before,
            self._wait_ceiling,
            len(self._queue),
        )

    def _chosen_wait(self, request_failure: RequestFailure) -> float:
        """Return how long this report proposes to pause.

        Test `is None` because `retry_after_seconds` presence determines the branch.
        """
        if request_failure.retry_after_seconds is not None:
            return request_failure.retry_after_seconds
        return _random_up_to(self._wait_ceiling, random.random())

    def _set_wait_ceiling(self, now: float, previous_pause_end: float) -> None:
        """Set _wait_ceiling from activity since previous_pause_end.

        The first pause uses min_wait_ceiling_seconds.
        Each full quiet_seconds_per_decay_step shrinks _wait_ceiling by wait_multiplier.
        _wait_ceiling never falls below min_wait_ceiling_seconds.
        Quiet time includes periods without requests.
        Resumed traffic without a full quiet step grows _wait_ceiling by wait_multiplier.
        _wait_ceiling never exceeds max_wait_seconds.
        A full quiet step takes precedence over resumed traffic.
        _steps_to_floor prevents wait_multiplier ** steps from overflowing.
        """
        if previous_pause_end == _NEVER:
            self._wait_ceiling = self.min_wait_ceiling_seconds
            return
        steps = int((now - previous_pause_end) // self.quiet_seconds_per_decay_step)
        if steps >= 1:
            if steps >= self._steps_to_floor:
                self._wait_ceiling = self.min_wait_ceiling_seconds
            else:
                self._wait_ceiling = max(
                    self._wait_ceiling / self.wait_multiplier**steps,
                    self.min_wait_ceiling_seconds,
                )
        elif self._last_admission_at > previous_pause_end:
            self._wait_ceiling = min(
                self._wait_ceiling * self.wait_multiplier,
                self.max_wait_seconds,
            )


class PrivateBackoff:
    """Generate private waits between the retries of one retry loop whose failures do not pause the quota.

    Keep one instance for a complete retry loop.
    Sleep returned waits outside `admitted` blocks.
    """

    def __init__(self, shared_backoff: SharedBackoff) -> None:
        """Start the private ceiling at min_wait_ceiling_seconds."""
        self._wait_ceiling = shared_backoff.min_wait_ceiling_seconds
        self._wait_multiplier = shared_backoff.wait_multiplier
        self._max_wait_seconds = shared_backoff.max_wait_seconds

    def next_wait(self, retry_after_seconds: float | None) -> float:
        """Return one failure's wait in seconds, then grow the ceiling one step.

        The wait is a positive random draw bounded by `_wait_ceiling`.
        `retry_after_seconds` raises that wait when present.
        The normalized `retry_after_seconds` is capped at `max_wait_seconds`.
        """
        wait = _random_up_to(self._wait_ceiling, random.random())
        if retry_after_seconds is not None:
            wait = max(wait, retry_after_seconds)
        self._wait_ceiling = min(
            self._wait_ceiling * self._wait_multiplier, self._max_wait_seconds
        )
        return wait
