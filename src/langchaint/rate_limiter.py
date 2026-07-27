"""Retry budget, backoff shape, and request pacing in one shared RateLimiter.

RateLimiter is the only place concurrency and retries are configured in langchaint.
Its admission gates every request start on every path: first attempts, retries, batch items, and stream openings,
so one budget covers the whole process.
There is deliberately no requests_per_minute: under a fixed in-flight bound, throughput follows request duration
(short cheap requests finish fast and run at a high rate, long token-heavy ones slow it),
while a client-side rate number models one dimension of the provider's multi-dimensional limit
and goes stale with the account tier.

Rate-limit errors drive a three-state admission cycle,
because an exhausted quota is account-level and letting the other slots fire into it only collects the same error:

1. Open: no unresolved rate-limit error; admission needs only a free in-flight slot.
2. Paused: a rate-limit error arrived; nobody is admitted until the pause expires.
   The pause is the server-stated retry-after when the error carried one,
   else the backoff delay computed from the failing task's error chain, and an existing later pause is never shortened.
3. Probing: the pause expired but no request has succeeded since the error;
   exactly one request (the probe) is admitted at a time.
   A registered success restores open admission; a probe that ends any other way
   (transient failure, non-retriable error, cancellation) lets the next waiter probe;
   a further rate-limit error starts a new pause.

Requests already in flight are never interrupted by a pause; the cycle gates admission only.
"""

import asyncio
import random
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

from langchaint.exceptions import TransientError

_RETRY_AFTER_MAX_SECONDS = 60.0
"""Cap on a server-stated retry-after, at the boundary both SDK clients draw.

An erroneous or hostile header (a wait of hours) must not stall the client indefinitely;
past this cap the account-wide pause reopens to a single probe, which re-pauses if the quota is still exhausted.
"""


class Admission:
    """One granted request start, returned by RateLimiter.acquire.

    Pass it back to RateLimiter.release exactly once when the request ends, however it ends;
    the limiter compares it by identity to know when the recovery probe has finished.
    """

    __slots__ = ("limiter",)

    limiter: "RateLimiter | None"
    """The RateLimiter whose in-flight slot this admission holds; None once released.

    Written by acquire and release only. Every RateLimiter method taking an admission
    raises RuntimeError when this is not that limiter, so a released, double-released,
    or foreign admission fails loudly instead of corrupting the slot count.
    """

    def __init__(self, limiter: "RateLimiter") -> None:
        """Bind the admission to the limiter granting it; called by RateLimiter.acquire only."""
        self.limiter = limiter


class Backoff:
    """One failure's backoff delay, drawn once so the sleep and any account-wide pause agree.

    Returned by RateLimiter.register_transient_error. A caller that will retry awaits sleep()
    after releasing the failing attempt's admission; a caller that will not retry drops the
    object, and any account-wide pause set with it still stands.
    """

    __slots__ = ("_admission", "delay_seconds")

    delay_seconds: float
    """The drawn delay sleep() waits, in seconds; public for callers that report the wait."""

    def __init__(self, delay_seconds: float, admission: Admission) -> None:
        """Store the drawn delay and the failing attempt's admission; built by register_transient_error only."""
        self.delay_seconds = delay_seconds
        self._admission = admission

    async def sleep(self) -> None:
        """Sleep the drawn delay; call after the failing attempt's admission is released.

        Raises:
            RuntimeError: the admission still holds its in-flight slot, so this sleep would
                hold capacity idle for the whole wait; release the admission first.
        """
        if self._admission.limiter is not None:
            raise RuntimeError(
                "backoff slept while the failing attempt's admission still holds its slot; "
                "release the admission before sleeping"
            )
        await asyncio.sleep(self.delay_seconds)


def _is_rate_limit_evidence(transient_error: TransientError) -> bool:
    """Whether the error is evidence of a rate limit, by its classification or by a server-stated wait.

    A server-stated retry_after_seconds qualifies because a server that names a wait is throttling,
    whatever the status code.
    """
    return transient_error.is_rate_limit or transient_error.retry_after_seconds is not None


class RateLimiter:
    """Retry budget, backoff shape, and in-flight bound; stateful and shareable.

    Share one instance across LLMs that hit the same provider account:
    the in-flight bound and the rate-limit admission cycle guard an account-level limit, not a per-model one.
    The internal primitives bind to the running event loop on first use, so one instance serves one event loop.

    max_attempts counts requests sent including the first, so 1 disables retrying.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        max_in_flight: int = 8,
    ) -> None:
        """Store the constructor parameters and create the unbound in-flight semaphore."""
        self.max_attempts = max_attempts
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.max_in_flight = max_in_flight
        self._in_flight_slots = asyncio.Semaphore(max_in_flight)
        self._paused_until = 0.0
        """Monotonic-clock moment before which acquire admits nothing."""
        self._recovering = False
        """Whether a rate-limit error is unresolved, so admission is probe-only after the pause."""
        self._probe_admission: Admission | None = None
        """The recovery probe currently in flight, compared by identity in release."""
        self._recovery_changed = asyncio.Event()
        """Wakes probe-gate waiters; replaced on each wake so late waiters do not see a stale set."""

    def _require_held(self, admission: Admission) -> None:
        """Raise unless admission currently holds one of this limiter's in-flight slots.

        Raises:
            RuntimeError: admission was already released, or was acquired from a different RateLimiter.
        """
        if admission.limiter is not self:
            raise RuntimeError(
                "admission does not hold a slot on this RateLimiter: "
                "it was already released, or was acquired from a different RateLimiter"
            )

    def _wake_recovery_waiters(self) -> None:
        self._recovery_changed.set()
        self._recovery_changed = asyncio.Event()

    def _is_probe_gate_closed(self) -> bool:
        return self._recovering and self._probe_admission is not None

    async def acquire(self) -> Admission:
        """Suspend until a request may start, then hold one in-flight slot.

        During recovery the returned Admission is the probe; the caller does not need to know,
        release handles both cases.
        Every acquire needs exactly one release of the returned Admission.
        """
        while True:
            pause_seconds = self._paused_until - time.monotonic()
            if pause_seconds > 0:
                await asyncio.sleep(pause_seconds)
                continue
            if self._is_probe_gate_closed():
                recovery_changed = self._recovery_changed
                await recovery_changed.wait()
                continue
            await self._in_flight_slots.acquire()
            if time.monotonic() < self._paused_until or self._is_probe_gate_closed():
                self._in_flight_slots.release()
                continue
            admission = Admission(self)
            if self._recovering:
                self._probe_admission = admission
            return admission

    def release(self, admission: Admission) -> None:
        """Return the in-flight slot admission holds; call exactly once, however the request ended.

        A probe released without a registered success did not prove recovery
        (it failed, was cancelled, or raised a non-transient error), so the next waiter becomes the probe.

        Raises:
            RuntimeError: admission does not hold a slot on this limiter
                (already released, or acquired from a different RateLimiter).
        """
        self._require_held(admission)
        admission.limiter = None
        self._in_flight_slots.release()
        if admission is self._probe_admission:
            self._probe_admission = None
            self._wake_recovery_waiters()

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[Admission]:
        """Hold one in-flight slot for the duration of the block.

        Yields:
            The Admission for this slot; pass it to register_success so recovery ends only when this slot is the probe.

        Raises:
            RuntimeError: the block released the admission itself, so the exit's release finds no held slot.
        """
        admission = await self.acquire()
        try:
            yield admission
        finally:
            self.release(admission)

    def register_success(self, admission: Admission) -> None:
        """Record that the recovery probe completed successfully, reopening full admission.

        A request admitted before the incident, or any other non-probe request,
        proves nothing about whether the exhausted quota reopened, so its success must not lift the probe-only gate:
        doing so would admit max_in_flight requests at once into a possibly-still-exhausted quota,
        which is the flooding the probe exists to prevent.
        Call while still holding the probe's slot,
        so the reopening is ordered before the release that would otherwise admit the next probe.
        _paused_until is deliberately not reset here: the probe's own pause has already elapsed by the time it succeeds,
        and a fresh pause another task set while the probe ran must survive.

        The probe identity is cleared with the recovery it served, rather than left for release to clear.
        An admission can outlive the request that proved recovery by a long way: a stream registers its success at
        the open and holds the same admission until it closes. Leaving the identity set means the next rate-limit
        error finds a probe already in flight, so the probe gate closes on an admission nobody will release until
        that stream ends, and every waiter on the limiter blocks with slots free.

        Raises:
            RuntimeError: admission does not hold a slot on this limiter
                (already released, or acquired from a different RateLimiter).
        """
        self._require_held(admission)
        if not (self._recovering and admission is self._probe_admission):
            return
        self._recovering = False
        self._probe_admission = None
        self._wake_recovery_waiters()

    def register_transient_error(
        self, admission: Admission, errors_from_attempts: Sequence[TransientError]
    ) -> Backoff:
        """Feed one task's failure chain back so admission can react account-wide.

        admission is the failing request's still-held admission: requiring it here puts the pause
        in place before the release admits anyone else, and the returned Backoff refuses to sleep
        until that release, so the wait never holds a slot.
        Only the chain's last error is the new failure; the earlier entries shape the pause length.
        Only a rate-limit error pauses admission account-wide,
        because a timeout or 5xx says nothing about the account's quota;
        an existing later pause is never shortened.

        Returns:
            The Backoff for the failing task's own retry; a task that will not retry drops it.
            Its delay is drawn once here, so it is exactly the account-wide pause length:
            the pause and the failing task's retry expire together,
            which makes the waking task the natural recovery probe.
            A non-rate-limit transient sets no pause but still carries its backoff for the failing task to sleep.

        Raises:
            RuntimeError: admission does not hold a slot on this limiter
                (already released, or acquired from a different RateLimiter).
        """
        self._require_held(admission)
        delay_seconds = self.delay_seconds(errors_from_attempts)
        if _is_rate_limit_evidence(errors_from_attempts[-1]):
            self._recovering = True
            self._paused_until = max(self._paused_until, time.monotonic() + delay_seconds)
        return Backoff(delay_seconds, admission)

    def delay_seconds(self, errors_from_attempts: Sequence[TransientError]) -> float:
        """Delay before the next attempt after the given failure chain.

        A server-stated wait is followed exactly, un-jittered,
        because retrying before it is a guaranteed wasted request;
        the _RETRY_AFTER_MAX_SECONDS cap keeps an erroneous or hostile header from stalling the client indefinitely.
        The backoff branch is AWS full jitter and draws once per call,
        so callers that need the pause and the retry sleep to agree must call once and share the value;
        register_transient_error does exactly that, carrying the one draw on the Backoff it returns.
        """
        last = errors_from_attempts[-1]
        if last.retry_after_seconds is not None:
            return min(last.retry_after_seconds, _RETRY_AFTER_MAX_SECONDS)
        ceiling = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * 2.0 ** (len(errors_from_attempts) - 1),
        )
        return random.uniform(0.0, ceiling)
