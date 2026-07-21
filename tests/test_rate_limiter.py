"""RateLimiter admission driven directly, without any adapter.

Each test drives acquire/release/register_success/register_transient_error by hand to pin the admission cycle:
the hard pause, probe-only recovery, and the success that restores full admission.
"""

import asyncio
import time

import pytest

from langchaint import RateLimiter, TransientError
from langchaint import rate_limiter as rate_limiter_module


def _rate_limit_error(*, retry_after_seconds: float | None = None) -> TransientError:
    """Build a TransientError classified as a rate limit."""
    return TransientError(
        "rate limited", retry_after_seconds=retry_after_seconds, is_rate_limit=True
    )


def test_retry_after_pauses_admission() -> None:
    """A rate-limit error with retry_after_seconds pauses acquire until that moment."""

    async def scenario() -> None:
        """Register the error, then time one slot acquisition."""
        rate_limiter = RateLimiter()
        rate_limiter.register_transient_error([_rate_limit_error(retry_after_seconds=0.05)])
        started_at = time.monotonic()
        async with rate_limiter.slot():
            pass
        assert time.monotonic() - started_at >= 0.04

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_retry_after_beyond_the_cap_is_clamped() -> None:
    """A server-stated retry-after past the 60s cap is clamped; a wait under it is honored."""
    rate_limiter = RateLimiter()
    assert rate_limiter.delay_seconds([_rate_limit_error(retry_after_seconds=600.0)]) == 60.0
    assert rate_limiter.delay_seconds([_rate_limit_error(retry_after_seconds=5.0)]) == 5.0


def test_rate_limit_error_without_retry_after_pauses_with_the_backoff_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A headerless rate-limit error pauses admission for the chain's backoff delay.

    The full-jitter draw is pinned to its ceiling so the pause length is deterministic here.
    """
    monkeypatch.setattr(rate_limiter_module.random, "uniform", lambda _low, high: high)

    async def scenario() -> None:
        """Register a one-error chain under a visible backoff base, then time acquisition."""
        rate_limiter = RateLimiter(backoff_base_seconds=0.05)
        rate_limiter.register_transient_error([_rate_limit_error()])
        started_at = time.monotonic()
        async with rate_limiter.slot():
            pass
        assert time.monotonic() - started_at >= 0.04

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_backoff_branch_draws_full_jitter_within_the_ceiling() -> None:
    """The exponential branch draws inside [0, ceiling] and the draw varies, so jitter is real.

    A ceiling that failed to double per failure is caught by test_backoff_branch_ceiling_doubles_per_failure_and_caps,
    not here;
    this test guards that the value is an actual random.uniform draw and not the ceiling returned deterministically.
    """
    rate_limiter = RateLimiter(backoff_base_seconds=1.0, backoff_max_seconds=30.0)
    one_error = [TransientError("boom")]
    two_errors = [TransientError("boom"), TransientError("boom")]
    one_error_draws = set()
    for _ in range(200):
        one_error_draw = rate_limiter.delay_seconds(one_error)
        one_error_draws.add(one_error_draw)
        assert 0.0 <= one_error_draw <= 1.0
        assert 0.0 <= rate_limiter.delay_seconds(two_errors) <= 2.0
    assert len(one_error_draws) > 1, "draws are constant, so the branch is not jittered"


def test_backoff_branch_ceiling_doubles_per_failure_and_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full-jitter ceiling is backoff_base_seconds doubled per failure, capped at backoff_max_seconds.

    Pinning random.uniform to its ceiling turns the jittered draw into the ceiling itself,
    so the returned value is exactly the ceiling under test.
    A branch that used backoff_base_seconds without doubling would return 1.0 every step and fail at the second.
    """
    monkeypatch.setattr(rate_limiter_module.random, "uniform", lambda _low, high: high)
    rate_limiter = RateLimiter(backoff_base_seconds=1.0, backoff_max_seconds=6.0)
    errors_from_attempts: list[TransientError] = []
    for expected_ceiling in (1.0, 2.0, 4.0, 6.0, 6.0):
        errors_from_attempts.append(TransientError("boom"))
        assert rate_limiter.delay_seconds(errors_from_attempts) == expected_ceiling


def test_backoff_draw_is_shared_by_the_pause_and_the_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """register_transient_error draws the jittered backoff once and returns that same pause length."""
    draws: list[tuple[float, float]] = []

    def spy_uniform(low: float, high: float) -> float:
        """Record the draw and return a fixed value."""
        draws.append((low, high))
        return 0.037

    monkeypatch.setattr(rate_limiter_module.random, "uniform", spy_uniform)
    rate_limiter = RateLimiter(backoff_base_seconds=0.05)
    before = time.monotonic()
    returned = rate_limiter.register_transient_error([_rate_limit_error()])
    # One draw for this failure, not two, over [0, backoff_base_seconds].
    assert draws == [(0.0, 0.05)]
    assert returned == 0.037
    # The returned backoff equals the account-wide pause offset it set, from the one draw.
    assert abs((rate_limiter._paused_until - before) - returned) < 0.005


def test_error_without_rate_limit_evidence_sets_no_admission_pause() -> None:
    """A plain transient error (timeout, 5xx) leaves admission open."""

    async def scenario() -> None:
        """Register a plain transient error, then time one slot acquisition."""
        rate_limiter = RateLimiter(backoff_base_seconds=0.05)
        rate_limiter.register_transient_error([TransientError("boom")])
        started_at = time.monotonic()
        async with rate_limiter.slot():
            pass
        assert time.monotonic() - started_at < 0.04

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_pause_is_never_shortened() -> None:
    """A later error with a shorter retry-after does not cut an existing pause."""

    async def scenario() -> None:
        """Register a long pause, then a short one, then time acquisition."""
        rate_limiter = RateLimiter()
        rate_limiter.register_transient_error([_rate_limit_error(retry_after_seconds=0.08)])
        rate_limiter.register_transient_error([_rate_limit_error(retry_after_seconds=0.001)])
        started_at = time.monotonic()
        async with rate_limiter.slot():
            pass
        assert time.monotonic() - started_at >= 0.06

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_recovery_admits_one_probe_and_a_success_reopens_admission() -> None:
    """After the pause, exactly one acquire is admitted until register_success."""

    async def scenario() -> None:
        """Race three acquires against a recovering limiter, then register the success."""
        rate_limiter = RateLimiter(backoff_base_seconds=0.01)
        rate_limiter.register_transient_error([_rate_limit_error()])
        tasks = [asyncio.create_task(rate_limiter.acquire()) for _ in range(3)]
        await asyncio.sleep(0.05)
        assert sum(task.done() for task in tasks) == 1
        probe = next(task.result() for task in tasks if task.done())
        rate_limiter.register_success(probe)
        admissions = await asyncio.gather(*tasks)
        for admission in admissions:
            rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_probe_released_without_success_admits_the_next_probe() -> None:
    """A probe that ends without a registered success hands the probe role to the next waiter."""

    async def scenario() -> None:
        """Let the first probe die unregistered and count who is admitted."""
        rate_limiter = RateLimiter(backoff_base_seconds=0.01)
        rate_limiter.register_transient_error([_rate_limit_error()])
        tasks = [asyncio.create_task(rate_limiter.acquire()) for _ in range(3)]
        await asyncio.sleep(0.05)
        first_probe = next(task for task in tasks if task.done())
        rate_limiter.release(first_probe.result())
        await asyncio.sleep(0.01)
        assert sum(task.done() for task in tasks) == 2
        second_probe = next(
            task.result() for task in tasks if task.done() and task is not first_probe
        )
        rate_limiter.register_success(second_probe)
        for task in tasks:
            if task is not first_probe:
                rate_limiter.release(await task)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_non_probe_success_does_not_end_recovery() -> None:
    """A success from a request that is not the probe leaves probe-only recovery in force."""

    async def scenario() -> None:
        """Succeed a pre-incident request during recovery, then confirm admission stays probe-only."""
        rate_limiter = RateLimiter(backoff_base_seconds=0.01)
        pre_incident = await rate_limiter.acquire()
        rate_limiter.register_transient_error([_rate_limit_error()])
        # The pre-incident request was admitted in the open state, so it is not the probe;
        # its success must be a no-op that leaves the one-probe gate closed.
        rate_limiter.register_success(pre_incident)
        rate_limiter.release(pre_incident)
        tasks = [asyncio.create_task(rate_limiter.acquire()) for _ in range(3)]
        await asyncio.sleep(0.05)
        assert sum(task.done() for task in tasks) == 1
        probe = next(task.result() for task in tasks if task.done())
        rate_limiter.register_success(probe)
        for admission in await asyncio.gather(*tasks):
            rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_probe_rate_limit_failure_pauses_admission_again() -> None:
    """A probe that collects another rate-limit error starts a new pause for everyone."""

    async def scenario() -> None:
        """Fail the probe with a fresh rate-limit error and time the next admission."""
        rate_limiter = RateLimiter(backoff_base_seconds=0.01)
        first_chain = [_rate_limit_error()]
        rate_limiter.register_transient_error(first_chain)
        probe_admission = await asyncio.wait_for(rate_limiter.acquire(), timeout=1.0)
        rate_limiter.register_transient_error(
            [*first_chain, _rate_limit_error(retry_after_seconds=0.05)]
        )
        rate_limiter.release(probe_admission)
        started_at = time.monotonic()
        async with rate_limiter.slot():
            pass
        assert time.monotonic() - started_at >= 0.04

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_max_in_flight_bounds_admission_outside_recovery() -> None:
    """With no incident registered, acquire admits max_in_flight requests and then blocks."""

    async def scenario() -> None:
        """Fill both slots, check the third waits, release one, check it proceeds."""
        rate_limiter = RateLimiter(max_in_flight=2)
        first = await rate_limiter.acquire()
        second = await rate_limiter.acquire()
        third_task = asyncio.create_task(rate_limiter.acquire())
        await asyncio.sleep(0.01)
        assert not third_task.done()
        rate_limiter.release(first)
        third = await asyncio.wait_for(third_task, timeout=1.0)
        rate_limiter.release(second)
        rate_limiter.release(third)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
