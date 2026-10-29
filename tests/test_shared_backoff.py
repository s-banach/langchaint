import asyncio
import math
import time
from typing import Literal

import pytest

from langchaint.common.exceptions import GaveUpWaitingError
from langchaint.common.request_failure import RequestFailure
from langchaint.concurrency.shared_backoff import (
    _NEVER,
    Admission,
    PrivateBackoff,
    SharedBackoff,
    _random_up_to,
)
from tests.helpers import run_with_timeout, yield_until


class ProviderError(Exception):
    """The failure type the tests raise inside admitted() blocks."""


def _shared_backoff(
    *,
    max_concurrent_requests: int | None = 1,
    min_wait_ceiling_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    wait_multiplier: float = 2.0,
    quiet_seconds_per_decay_step: float = 60.0,
    max_request_starts_per_second: float = 200.0,
) -> SharedBackoff:
    return SharedBackoff(
        max_concurrent_requests=max_concurrent_requests,
        min_wait_ceiling_seconds=min_wait_ceiling_seconds,
        max_wait_seconds=max_wait_seconds,
        wait_multiplier=wait_multiplier,
        quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        max_request_starts_per_second=max_request_starts_per_second,
    )


def _all_permits_free(shared_backoff: SharedBackoff) -> bool:
    return shared_backoff._permits_held == 0 and not shared_backoff._queue


async def _enter_empty_block(admission: Admission) -> None:
    async with admission:
        pass


def _pausing(retry_after_seconds: float | None) -> RequestFailure:
    return RequestFailure(
        kind="transient", pauses_quota=True, retry_after_seconds=retry_after_seconds
    )


async def _record_and_raise(
    shared_backoff: SharedBackoff, request_failure: RequestFailure, recorded: list[RequestFailure]
) -> None:
    async with shared_backoff.admitted() as admission:
        recorded.append(admission.record(request_failure))
        raise ProviderError("boom")


async def _fail_one_request(
    shared_backoff: SharedBackoff, request_failure: RequestFailure
) -> RequestFailure:
    recorded: list[RequestFailure] = []
    with pytest.raises(ProviderError):
        await _record_and_raise(shared_backoff, request_failure, recorded)
    (normalized,) = recorded
    return normalized


# --- construction ---


@pytest.mark.parametrize("max_concurrent_requests", [True, False, 0, -1])
def test_constructor_rejects_invalid_max_concurrent_requests(
    max_concurrent_requests: int,
) -> None:
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        _ = _shared_backoff(max_concurrent_requests=max_concurrent_requests)


@pytest.mark.parametrize("max_concurrent_requests", [None, 8])
def test_constructor_accepts_valid_max_concurrent_requests(
    max_concurrent_requests: int | None,
) -> None:
    _ = _shared_backoff(max_concurrent_requests=max_concurrent_requests)


@pytest.mark.parametrize(
    "name",
    [
        "min_wait_ceiling_seconds",
        "max_wait_seconds",
        "wait_multiplier",
        "quiet_seconds_per_decay_step",
        "max_request_starts_per_second",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 0, -1.0, float("inf"), float("nan")])
def test_constructor_rejects_invalid_numeric_settings(name: str, invalid_value: float) -> None:
    with pytest.raises(ValueError, match=name):
        _ = _shared_backoff(
            min_wait_ceiling_seconds=invalid_value if name == "min_wait_ceiling_seconds" else 1.0,
            max_wait_seconds=invalid_value if name == "max_wait_seconds" else 60.0,
            wait_multiplier=invalid_value if name == "wait_multiplier" else 2.0,
            quiet_seconds_per_decay_step=(
                invalid_value if name == "quiet_seconds_per_decay_step" else 60.0
            ),
            max_request_starts_per_second=(
                invalid_value if name == "max_request_starts_per_second" else 50.0
            ),
        )


@pytest.mark.parametrize("max_request_starts_per_second", [10**1000, 5e-324])
def test_constructor_rejects_unrepresentable_request_start_rates(
    max_request_starts_per_second: float,
) -> None:
    with pytest.raises(ValueError, match="max_request_starts_per_second"):
        _ = _shared_backoff(max_request_starts_per_second=max_request_starts_per_second)


def test_constructor_rejects_wait_multiplier_at_or_below_one() -> None:
    """A multiplier of 1 would never grow or shrink the ceiling."""
    with pytest.raises(ValueError, match="wait_multiplier"):
        _ = _shared_backoff(wait_multiplier=1.0)


def test_constructor_rejects_max_wait_seconds_below_the_floor() -> None:
    with pytest.raises(ValueError, match="max_wait_seconds"):
        _ = _shared_backoff(
            min_wait_ceiling_seconds=10.0,
            max_wait_seconds=1.0,
        )


def test_constructor_rejects_an_unrepresentable_ceiling_ratio() -> None:
    """`1e308 / 5e-324` overflows, and `10**1000` cannot convert to a float."""
    with pytest.raises(ValueError, match="finite"):
        _ = _shared_backoff(min_wait_ceiling_seconds=5e-324, max_wait_seconds=1e308)
    with pytest.raises(ValueError, match="min_wait_ceiling_seconds"):
        _ = _shared_backoff(
            min_wait_ceiling_seconds=10**1000,
            max_wait_seconds=1e308,
        )


def test_constructor_accepts_max_wait_seconds_equal_to_the_floor() -> None:
    _ = _shared_backoff(
        min_wait_ceiling_seconds=2.0,
        max_wait_seconds=2.0,
    )


# --- the exit ---


def test_success_returns_the_permit_and_records_nothing() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        await _enter_empty_block(shared_backoff.admitted())
        assert _all_permits_free(shared_backoff)
        assert shared_backoff._pause_until == _NEVER

    run_with_timeout(scenario())


@pytest.mark.parametrize("kind", ["transient", "provider_failed_terminally"])
def test_a_pausing_failure_is_recorded_and_propagated(
    kind: Literal["transient", "provider_failed_terminally"],
) -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        request_failure = RequestFailure(kind=kind, pauses_quota=True, retry_after_seconds=0.25)
        assert await _fail_one_request(shared_backoff, request_failure) == request_failure
        assert _all_permits_free(shared_backoff)
        assert shared_backoff._pause_until > shared_backoff._clock()

    run_with_timeout(scenario())


@pytest.mark.parametrize(
    "request_failure",
    [
        RequestFailure(kind="rejected", pauses_quota=False, retry_after_seconds=None),
        RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=0.5),
    ],
    ids=["rejected", "transient"],
)
def test_a_non_pausing_failure_changes_no_shared_state(request_failure: RequestFailure) -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        assert await _fail_one_request(shared_backoff, request_failure) == request_failure
        assert shared_backoff._pause_until == _NEVER

    run_with_timeout(scenario())


# --- retry_after_seconds normalization ---


@pytest.mark.parametrize(
    ("stated", "expected"),
    [
        (-5, None),
        (0, None),
        (float("nan"), None),
        (float("-inf"), None),
        (float("inf"), None),
        (True, None),
        (-(10**1000), None),
        (9999, 60.0),
        (10**1000, 60.0),
        (60, 60.0),
        (5.0, 5.0),
    ],
)
def test_retry_after_normalization(stated: float, expected: float | None) -> None:
    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        recorded = await _fail_one_request(shared_backoff, _pausing(stated))
        assert recorded.retry_after_seconds == expected

    run_with_timeout(scenario())


@pytest.mark.parametrize(
    ("stated", "tag"), [(-5, "retry_after_invalid"), (9999, "retry_after_over_cap")]
)
def test_retry_after_corrections_are_counted(stated: float, tag: str) -> None:
    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        _ = await _fail_one_request(shared_backoff, _pausing(stated))
        assert shared_backoff.event_counts[tag] == 1

    run_with_timeout(scenario())


# --- pauses and pacing ---


def test_a_pause_holds_the_next_admission_until_it_ends() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        _ = await _fail_one_request(shared_backoff, _pausing(10.0))
        entering = asyncio.create_task(_enter_empty_block(shared_backoff.admitted()))
        await yield_until(lambda: len(shared_backoff._queue) == 1)
        admit_timer = shared_backoff._admit_timer
        assert admit_timer is not None
        assert 9.0 < admit_timer.when() - asyncio.get_running_loop().time() <= 10.0
        _ = entering.cancel()
        with pytest.raises(asyncio.CancelledError):
            await entering

    run_with_timeout(scenario())


def test_recording_happens_before_the_permit_is_released() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        first_entered = asyncio.Event()

        async def fail_after_signalling() -> None:
            async with shared_backoff.admitted() as admission:
                first_entered.set()
                await yield_until(lambda: len(shared_backoff._queue) == 1)
                _ = admission.record(_pausing(0.02))
                raise ProviderError("429")

        async def failing_request() -> None:
            with pytest.raises(ProviderError):
                await fail_after_signalling()

        async def waiting_request() -> float:
            await first_entered.wait()
            started_at = time.monotonic()
            await _enter_empty_block(shared_backoff.admitted())
            return time.monotonic() - started_at

        failing = asyncio.create_task(failing_request())
        waiting = asyncio.create_task(waiting_request())
        await failing
        assert await waiting >= 0.02

    run_with_timeout(scenario())


def test_request_starts_respect_max_request_starts_per_second() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            max_concurrent_requests=None,
            max_request_starts_per_second=100.0,
        )
        admitted_at: list[float] = []

        async def request() -> None:
            async with shared_backoff.admitted():
                admitted_at.append(time.monotonic())

        await asyncio.gather(request(), request())
        assert abs(admitted_at[1] - admitted_at[0]) >= 0.008

    run_with_timeout(scenario())


def test_waiters_are_released_in_the_order_they_joined() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            max_concurrent_requests=None,
            max_request_starts_per_second=100.0,
        )
        shared_backoff._record(_pausing(0.01))
        admitted_order: list[int] = []

        async def request(index: int) -> None:
            async with shared_backoff.admitted():
                admitted_order.append(index)

        async with asyncio.TaskGroup() as group:
            for index in range(3):
                _ = group.create_task(request(index))
                await yield_until(lambda joined=index + 1: len(shared_backoff._queue) == joined)
        assert admitted_order == [0, 1, 2]

    run_with_timeout(scenario())


def test_entry_is_immediate_when_nothing_blocks_it() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        started_at = time.monotonic()
        await _enter_empty_block(shared_backoff.admitted())
        assert time.monotonic() - started_at < 0.05

    run_with_timeout(scenario())


# --- the budget ---


@pytest.mark.parametrize("budget", [True, False, 0, -1, float("inf"), float("nan"), 10**1000])
def test_admitted_rejects_invalid_budgets_before_acquiring_anything(budget: float) -> None:
    shared_backoff = _shared_backoff()
    with pytest.raises(ValueError, match="budget"):
        _ = shared_backoff.admitted(budget_seconds=budget)
    assert _all_permits_free(shared_backoff)


def test_a_budget_expiring_in_the_queue_leaves_nothing_held() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        shared_backoff._record(_pausing(0.5))
        with pytest.raises(GaveUpWaitingError):
            await _enter_empty_block(shared_backoff.admitted(budget_seconds=0.005))
        assert len(shared_backoff._queue) == 0
        assert shared_backoff.event_counts["gave_up_waiting"] == 1
        assert _all_permits_free(shared_backoff)

    run_with_timeout(scenario())


def test_a_budget_expiring_while_every_permit_is_held_takes_no_permit() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with shared_backoff.admitted():
                await release_holder.wait()

        holding = asyncio.create_task(holder())
        await yield_until(lambda: shared_backoff._permits_held == 1)
        with pytest.raises(GaveUpWaitingError):
            await _enter_empty_block(shared_backoff.admitted(budget_seconds=0.005))
        assert len(shared_backoff._queue) == 0
        release_holder.set()
        await holding
        assert _all_permits_free(shared_backoff)

    run_with_timeout(scenario())


def test_one_shared_backoff_serves_consecutive_event_loops() -> None:
    """Permit contention in one event loop does not bind the SharedBackoff to that loop."""
    shared_backoff = _shared_backoff()

    async def two_contending_requests() -> None:
        async with shared_backoff.admitted():
            contender = asyncio.create_task(_enter_empty_block(shared_backoff.admitted()))
            await yield_until(lambda: len(shared_backoff._queue) == 1)
        await contender

    run_with_timeout(two_contending_requests())
    run_with_timeout(two_contending_requests())
    assert _all_permits_free(shared_backoff)


def test_cancellation_while_queued_leaves_an_empty_queue_and_a_full_permit_count() -> None:

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        shared_backoff._record(_pausing(0.5))
        waiting = asyncio.create_task(_enter_empty_block(shared_backoff.admitted()))
        await yield_until(lambda: len(shared_backoff._queue) == 1)
        _ = waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert len(shared_backoff._queue) == 0
        assert _all_permits_free(shared_backoff)

    run_with_timeout(scenario())


# --- the merge rule ---


def test_reports_during_a_pause_extend_it_and_never_shrink_it() -> None:
    shared_backoff = _shared_backoff()
    moment = 0.0
    shared_backoff._clock = lambda: moment
    for report_at, expected_end in ((0.0, 60.0), (59.0, 119.0), (118.0, 178.0), (177.0, 237.0)):
        moment = report_at
        shared_backoff._record(_pausing(60.0))
        assert shared_backoff._pause_until == expected_end
    moment = 178.0
    shared_backoff._record(_pausing(1.0))
    assert shared_backoff._pause_until == 237.0


def test_pause_ends_within_max_wait_seconds_after_the_most_recent_report() -> None:
    shared_backoff = _shared_backoff()
    moment = 0.0
    shared_backoff._clock = lambda: moment
    shared_backoff._record(_pausing(60.0))
    for step in range(200):
        moment += 1.0
        retry_after_seconds = 60.0 if step % 2 == 0 else None
        shared_backoff._record(_pausing(retry_after_seconds))
        assert shared_backoff._pause_until - moment <= 60.0


# --- the wait ceiling ---


def test_the_first_pause_starts_from_the_floor() -> None:
    shared_backoff = _shared_backoff()
    shared_backoff._set_wait_ceiling(10.0, _NEVER)
    assert shared_backoff._wait_ceiling == 1.0


def test_the_ceiling_grows_only_after_traffic_resumed() -> None:
    shared_backoff = _shared_backoff()
    shared_backoff._wait_ceiling = 4.0
    shared_backoff._last_admission_at = 3.0
    shared_backoff._set_wait_ceiling(6.0, 4.0)
    assert shared_backoff._wait_ceiling == 4.0
    shared_backoff._last_admission_at = 5.0
    shared_backoff._set_wait_ceiling(6.0, 4.0)
    assert shared_backoff._wait_ceiling == 8.0


def test_the_ceiling_growth_caps_at_max_wait_seconds() -> None:
    shared_backoff = _shared_backoff()
    shared_backoff._wait_ceiling = 40.0
    shared_backoff._last_admission_at = 5.0
    shared_backoff._set_wait_ceiling(6.0, 4.0)
    assert shared_backoff._wait_ceiling == 60.0


@pytest.mark.parametrize(
    ("quiet_steps", "expected_ceiling"),
    list(enumerate([30.0, 15.0, 7.5, 3.75, 1.875, 1.0, 1.0], start=1)),
)
def test_the_ceiling_decays_one_step_per_quiet_step_down_to_the_floor(
    quiet_steps: int, expected_ceiling: float
) -> None:
    shared_backoff = _shared_backoff()
    shared_backoff._wait_ceiling = 60.0
    shared_backoff._set_wait_ceiling(100.0 + 60.0 * quiet_steps, 100.0)
    assert shared_backoff._wait_ceiling == expected_ceiling


def test_decay_wins_when_traffic_also_resumed() -> None:
    shared_backoff = _shared_backoff()
    shared_backoff._wait_ceiling = 60.0
    shared_backoff._last_admission_at = 150.0
    shared_backoff._set_wait_ceiling(160.0, 100.0)
    assert shared_backoff._wait_ceiling == 30.0


def test_a_long_quiet_spell_cannot_overflow_the_decay() -> None:
    """Past _steps_to_floor the answer is the floor, so no exponent is ever computed there."""
    a_day_quiet = _shared_backoff()
    a_day_quiet._wait_ceiling = 60.0
    a_day_quiet._set_wait_ceiling(86_400.0, 0.0)
    assert a_day_quiet._wait_ceiling == 1.0
    huge_multiplier = _shared_backoff(wait_multiplier=1e5)
    huge_multiplier._wait_ceiling = 60.0
    huge_multiplier._set_wait_ceiling(6_000.0, 0.0)
    assert huge_multiplier._wait_ceiling == 1.0


def test_a_multiplier_just_above_one_decays_without_a_clamp() -> None:
    """A tiny multiplier takes many steps to the floor and each is computed exactly."""
    shared_backoff = _shared_backoff(wait_multiplier=1.0000001)
    shared_backoff._wait_ceiling = 60.0
    shared_backoff._set_wait_ceiling(60.0 * 1_000_000.0, 0.0)
    assert 1.0 <= shared_backoff._wait_ceiling < 60.0


@pytest.mark.parametrize("ceiling", [math.ulp(0.0), 2 * math.ulp(0.0), 2.0])
@pytest.mark.parametrize("draw", [0.0, 0.75, math.nextafter(1.0, 0.0)])
def test_chosen_waits_are_positive_and_bounded_by_the_ceiling(ceiling: float, draw: float) -> None:
    """Subnormal ceilings can round the sampled product to zero."""
    wait = _random_up_to(ceiling, draw)
    assert 0.0 < wait <= ceiling
    if draw == 0.0:
        assert wait == ceiling
    elif ceiling > math.ulp(0.0):
        assert wait < ceiling


def test_private_backoff_ceilings_grow_to_the_cap() -> None:
    """Each wait lies under the ceiling in force before it, and the ceiling then grows one step."""
    private_backoff = PrivateBackoff(
        _shared_backoff(
            min_wait_ceiling_seconds=1.0,
            wait_multiplier=2.0,
            max_wait_seconds=4.0,
        )
    )
    ceilings_and_waits = [
        (private_backoff._wait_ceiling, private_backoff.next_wait_seconds(None)) for _ in range(4)
    ]
    assert [ceiling for ceiling, _ in ceilings_and_waits] == [1.0, 2.0, 4.0, 4.0]
    assert all(0.0 < wait <= ceiling for ceiling, wait in ceilings_and_waits)
