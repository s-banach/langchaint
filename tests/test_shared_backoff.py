"""Test SharedBackoff admission, pauses, and wait ceilings."""

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Literal, override

import pytest

from langchaint import shared_backoff as shared_backoff_module
from langchaint.exceptions import GaveUpWaiting, ParserContractError
from langchaint.shared_backoff import (
    _NEVER,
    Admission,
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    PrivateBackoff,
    RetryThisOne,
    SharedBackoff,
    Verdict,
)


class ProviderFailure(Exception):  # noqa: N818 (named for what it is, a raised provider failure)
    """The failure type the tests raise inside admitted() blocks."""


def _retry_verdict(_failure: Exception) -> Verdict:
    """Map every failure to RetryThisOne with no retry_after; the parse most tests bind."""
    return RetryThisOne(retry_after=None)


def _shared_backoff(
    *,
    parse: Callable[[Exception], Verdict] = _retry_verdict,
    max_concurrent_requests: int | None = 1,
    minimum_wait_ceiling_seconds: float = 1.0,
    longest_wait_seconds: float = 60.0,
    wait_multiplier: float = 2.0,
    quiet_seconds_per_decay_step: float = 60.0,
    max_request_starts_per_second: float = 200.0,
    on_parse_error: Literal["raise", "retry_this_one"] = "raise",
) -> SharedBackoff:
    """Build a SharedBackoff with test-friendly defaults, overridable per test."""
    return SharedBackoff(
        parse=parse,
        failure_types=(ProviderFailure,),
        max_concurrent_requests=max_concurrent_requests,
        minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
        longest_wait_seconds=longest_wait_seconds,
        wait_multiplier=wait_multiplier,
        quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        max_request_starts_per_second=max_request_starts_per_second,
        on_parse_error=on_parse_error,
    )


class _RecordingSharedBackoff(SharedBackoff):
    """SharedBackoff that keeps every verdict _record received, for identity assertions."""

    def __init__(self, *, parse: Callable[[Exception], Verdict]) -> None:
        super().__init__(parse=parse, failure_types=(ProviderFailure,), max_concurrent_requests=1)
        self.recorded: list[Verdict] = []

    @override
    def _record(self, verdict: Verdict) -> None:
        self.recorded.append(verdict)
        super()._record(verdict)


def _run(scenario: Callable[[], Coroutine[None, None, None]]) -> None:
    """Run one async scenario under a hang guard."""
    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def _all_permits_free(shared_backoff: SharedBackoff) -> bool:
    """Report whether every permit is free: none held, none leaked, no live waiter queued.

    Read CPython's private semaphore count for tests.
    """
    permits = shared_backoff._permits
    assert permits is not None
    return permits._value == shared_backoff.max_concurrent_requests and not permits.locked()


async def _raise_in_block(admission: Admission, failure: Exception) -> None:
    """Enter the block and raise failure as the attempt's ending."""
    async with admission:
        raise failure


async def _enter_empty_block(admission: Admission) -> None:
    """Enter the block and end the attempt at once, successfully."""
    async with admission:
        pass


async def _fail_one_attempt(shared_backoff: SharedBackoff) -> Admission:
    """Run one attempt that raises ProviderFailure, and return its Admission.

    Asserts the exit re-raised the provider failure to the caller.
    """
    admission = shared_backoff.admitted()
    with pytest.raises(ProviderFailure):
        await _raise_in_block(admission, ProviderFailure("boom"))
    return admission


# --- construction ---


def test_constructor_rejects_invalid_max_concurrent_requests() -> None:
    """Reject a bool max_concurrent_requests and an int below 1."""
    for max_concurrent_requests in (True, False, 0, -1):
        with pytest.raises(ValueError, match="max_concurrent_requests"):
            _ = _shared_backoff(max_concurrent_requests=max_concurrent_requests)
    _ = _shared_backoff(max_concurrent_requests=None)
    _ = _shared_backoff(max_concurrent_requests=8)


def test_constructor_rejects_exception_as_a_failure_type() -> None:
    """Exception would include caller defects outside provider failures."""
    with pytest.raises(ValueError, match="failure_types"):
        _ = SharedBackoff(
            parse=_retry_verdict,
            failure_types=(Exception,),
            max_concurrent_requests=1,
        )


def test_constructor_rejects_empty_failure_types() -> None:
    """An empty failure_types would make the exit parse nothing and record nothing."""
    with pytest.raises(ValueError, match="failure_types"):
        _ = SharedBackoff(parse=_retry_verdict, failure_types=(), max_concurrent_requests=1)


def test_constructor_rejects_invalid_numeric_settings() -> None:
    """Every numeric setting shares one acceptance rule: not a bool, finite, positive."""
    valid = {
        "minimum_wait_ceiling_seconds": 1.0,
        "longest_wait_seconds": 60.0,
        "wait_multiplier": 2.0,
        "quiet_seconds_per_decay_step": 60.0,
        "max_request_starts_per_second": 50.0,
    }
    for invalid_value in (True, 0, -1.0, float("inf"), float("nan")):
        for name in valid:
            settings = {**valid, name: invalid_value}
            with pytest.raises(ValueError, match=name):
                _ = SharedBackoff(
                    parse=_retry_verdict,
                    failure_types=(ProviderFailure,),
                    max_concurrent_requests=1,
                    minimum_wait_ceiling_seconds=settings["minimum_wait_ceiling_seconds"],
                    longest_wait_seconds=settings["longest_wait_seconds"],
                    wait_multiplier=settings["wait_multiplier"],
                    quiet_seconds_per_decay_step=settings["quiet_seconds_per_decay_step"],
                    max_request_starts_per_second=settings["max_request_starts_per_second"],
                )


def test_constructor_derives_seconds_between_request_starts_from_a_valid_rate() -> None:
    """Validate max_request_starts_per_second before deriving its reciprocal."""
    with pytest.raises(ValueError, match="max_request_starts_per_second"):
        _ = _shared_backoff(max_request_starts_per_second=10**1000)
    with pytest.raises(ValueError, match="max_request_starts_per_second"):
        _ = _shared_backoff(max_request_starts_per_second=5e-324)
    shared_backoff = _shared_backoff(max_request_starts_per_second=20.0)
    assert shared_backoff.max_request_starts_per_second == 20.0
    assert shared_backoff._seconds_between_request_starts == 0.05


def test_constructor_rejects_wait_multiplier_at_or_below_one() -> None:
    """A multiplier of 1 would never grow or shrink the ceiling."""
    with pytest.raises(ValueError, match="wait_multiplier"):
        _ = _shared_backoff(wait_multiplier=1.0)


def test_constructor_rejects_longest_wait_seconds_below_the_floor() -> None:
    """longest_wait_seconds must be at least minimum_wait_ceiling_seconds."""
    with pytest.raises(ValueError, match="longest_wait_seconds"):
        _ = _shared_backoff(
            minimum_wait_ceiling_seconds=10.0,
            longest_wait_seconds=1.0,
        )


def test_constructor_rejects_an_unrepresentable_ceiling_ratio() -> None:
    """A ratio the decay arithmetic cannot hold raises ValueError, never OverflowError.

    5e-324 under 1e308 makes the ceiling ratio infinite, and 10**1000 cannot become a float.
    """
    with pytest.raises(ValueError, match="finite"):
        _ = _shared_backoff(minimum_wait_ceiling_seconds=5e-324, longest_wait_seconds=1e308)
    with pytest.raises(ValueError, match="minimum_wait_ceiling_seconds"):
        _ = _shared_backoff(
            minimum_wait_ceiling_seconds=10**1000,
            longest_wait_seconds=1e308,
        )


def test_constructor_accepts_longest_wait_seconds_equal_to_the_floor() -> None:
    """A one-value ceiling range is legal; the ceiling then never moves."""
    shared_backoff = _shared_backoff(
        minimum_wait_ceiling_seconds=2.0,
        longest_wait_seconds=2.0,
    )
    assert shared_backoff._wait_ceiling == 2.0


# --- the exit ---


def test_success_returns_the_permit_and_records_nothing() -> None:
    """A block that raises nothing leaves verdict None, the permit free, and no pause."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        admission = shared_backoff.admitted()
        await _enter_empty_block(admission)
        assert admission.verdict is None
        assert _all_permits_free(shared_backoff)
        assert shared_backoff._pause_until == _NEVER

    _run(scenario)


def test_an_exception_outside_failure_types_propagates_unparsed() -> None:
    """A fault in the attempt returns the permit and is neither parsed nor recorded."""

    async def scenario() -> None:
        parsed: list[Exception] = []

        def parse(failure: Exception) -> Verdict:
            parsed.append(failure)
            return DoNotRetry()

        shared_backoff = _shared_backoff(parse=parse)
        admission = shared_backoff.admitted()
        with pytest.raises(ValueError, match="attempt fault"):
            await _raise_in_block(admission, ValueError("attempt fault"))
        assert admission.verdict is None
        assert parsed == []
        assert _all_permits_free(shared_backoff)
        assert shared_backoff._pause_until == _NEVER

    _run(scenario)


def test_a_failure_is_parsed_recorded_and_propagated() -> None:
    """The exit stores the normalized verdict, records the same object, and re-raises the failure."""

    async def scenario() -> None:
        shared_backoff = _RecordingSharedBackoff(parse=lambda _failure: PauseAll(retry_after=0.25))
        admission = await _fail_one_attempt(shared_backoff)
        assert admission.verdict == PauseAll(retry_after=0.25)
        assert shared_backoff.recorded == [admission.verdict]
        assert shared_backoff.recorded[0] is admission.verdict
        assert _all_permits_free(shared_backoff)
        assert shared_backoff._pause_until > shared_backoff._clock()

    _run(scenario)


def test_a_pause_all_do_not_retry_verdict_starts_the_shared_pause() -> None:
    """`PauseAllDoNotRetry` stops one request and pauses the rate-limit quota.

    `_record()` returns early for every verdict except two pausing verdicts.
    `DoNotRetry` would drop the shared pause evidenced by this failure.
    """

    async def scenario() -> None:
        shared_backoff = _RecordingSharedBackoff(
            parse=lambda _failure: PauseAllDoNotRetry(retry_after=0.25)
        )
        admission = await _fail_one_attempt(shared_backoff)
        assert admission.verdict == PauseAllDoNotRetry(retry_after=0.25)
        assert shared_backoff.recorded == [admission.verdict]
        assert shared_backoff._pause_until > shared_backoff._clock()

    _run(scenario)


def test_a_pause_all_do_not_retry_retry_after_is_capped_like_any_other() -> None:
    """The wrapper normalizes this variant's retry_after, since it carries one."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            parse=lambda _failure: PauseAllDoNotRetry(retry_after=10_000.0)
        )
        admission = await _fail_one_attempt(shared_backoff)
        assert admission.verdict == PauseAllDoNotRetry(
            retry_after=shared_backoff.longest_wait_seconds
        )

    _run(scenario)


def test_a_do_not_retry_verdict_changes_no_shared_state() -> None:
    """DoNotRetry lands on the admission and starts no pause."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(parse=lambda _failure: DoNotRetry())
        admission = await _fail_one_attempt(shared_backoff)
        assert admission.verdict == DoNotRetry()
        assert shared_backoff._pause_until == _NEVER

    _run(scenario)


def test_a_raising_record_still_returns_the_permit() -> None:
    """The release sits in a finally, so a raise out of recording cannot leak a permit."""

    class _BrokenRecordSharedBackoff(SharedBackoff):
        @override
        def _record(self, verdict: Verdict) -> None:
            raise RuntimeError("record broke")

    async def scenario() -> None:
        shared_backoff = _BrokenRecordSharedBackoff(
            parse=_retry_verdict, failure_types=(ProviderFailure,), max_concurrent_requests=1
        )
        with pytest.raises(RuntimeError, match="record broke"):
            await _raise_in_block(shared_backoff.admitted(), ProviderFailure("boom"))
        assert _all_permits_free(shared_backoff)

    _run(scenario)


# --- the parse contract ---


def test_a_raising_parse_raises_parser_contract_error_with_the_full_chain() -> None:
    """The chain reads ParserContractError, then the parse defect, then the provider failure."""

    def parse(_failure: Exception) -> Verdict:
        raise RuntimeError("parse bug")

    async def scenario() -> None:
        shared_backoff = _shared_backoff(parse=parse)
        admission = shared_backoff.admitted()
        with pytest.raises(ParserContractError) as caught:
            await _raise_in_block(admission, ProviderFailure("429"))
        defect = caught.value.__cause__
        assert isinstance(defect, RuntimeError)
        assert isinstance(defect.__context__, ProviderFailure)
        assert admission.verdict is None
        assert shared_backoff.event_counts["parser_contract_error"] == 1
        assert _all_permits_free(shared_backoff)
        assert shared_backoff._pause_until == _NEVER

    _run(scenario)


def test_the_retry_this_one_fallback_corrects_a_parse_defect() -> None:
    """Under the fallback the defect becomes RetryThisOne, counted, and the failure propagates."""

    def parse(_failure: Exception) -> Verdict:
        raise RuntimeError("parse bug")

    async def scenario() -> None:
        shared_backoff = _shared_backoff(parse=parse, on_parse_error="retry_this_one")
        admission = await _fail_one_attempt(shared_backoff)
        assert admission.verdict == RetryThisOne(retry_after=None)
        assert shared_backoff.event_counts["parse_raised"] == 1

    _run(scenario)


# --- retry_after normalization ---


def test_retry_after_normalization() -> None:
    """Accept a positive finite number capped at longest_wait_seconds.

    The wrapper normalizes before either _record or Admission.verdict sees the verdict.
    """
    cases: list[tuple[float, float | None]] = [
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
    ]

    async def scenario() -> None:
        for stated, expected in cases:
            shared_backoff = _shared_backoff(
                parse=lambda _failure, stated=stated: PauseAll(retry_after=stated)
            )
            admission = await _fail_one_attempt(shared_backoff)
            assert isinstance(admission.verdict, PauseAll), f"stated={stated!r}"
            assert admission.verdict.retry_after == expected, f"stated={stated!r}"
            if expected is not None:
                assert type(admission.verdict.retry_after) is float

    _run(scenario)


def test_retry_after_corrections_are_counted() -> None:
    """An invalid retry_after and one over the cap each land in event_counts."""

    async def scenario() -> None:
        for stated, tag in ((-5, "retry_after_invalid"), (9999, "retry_after_over_cap")):
            shared_backoff = _shared_backoff(
                parse=lambda _failure, stated=stated: PauseAll(retry_after=stated)
            )
            _ = await _fail_one_attempt(shared_backoff)
            assert shared_backoff.event_counts[tag] == 1

    _run(scenario)


# --- pauses and pacing ---


def test_a_pause_holds_the_next_admission_until_it_ends() -> None:
    """After a PauseAll, entry waits out the remaining pause before admitting."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(parse=lambda _failure: PauseAll(retry_after=0.15))
        _ = await _fail_one_attempt(shared_backoff)
        started_at = time.monotonic()
        await _enter_empty_block(shared_backoff.admitted())
        assert time.monotonic() - started_at >= 0.1

    _run(scenario)


def test_recording_happens_before_the_permit_is_released() -> None:
    """A waiter taking the failing request's permit finds the pause already recorded."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(parse=lambda _failure: PauseAll(retry_after=0.15))
        first_entered = asyncio.Event()

        async def fail_after_signalling() -> None:
            async with shared_backoff.admitted():
                first_entered.set()
                await asyncio.sleep(0.02)
                raise ProviderFailure("429")

        async def failing_request() -> None:
            with pytest.raises(ProviderFailure):
                await fail_after_signalling()

        async def waiting_request() -> float:
            await first_entered.wait()
            started_at = time.monotonic()
            await _enter_empty_block(shared_backoff.admitted())
            return time.monotonic() - started_at

        failing = asyncio.create_task(failing_request())
        waiting = asyncio.create_task(waiting_request())
        await failing
        assert await waiting >= 0.1

    _run(scenario)


def test_request_starts_respect_max_request_starts_per_second() -> None:
    """Two queued requests start at the configured maximum rate."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            max_concurrent_requests=None,
            max_request_starts_per_second=20.0,
        )
        assert shared_backoff._seconds_between_request_starts == 0.05
        admitted_at: list[float] = []

        async def request() -> None:
            async with shared_backoff.admitted():
                admitted_at.append(time.monotonic())

        await asyncio.gather(request(), request())
        assert abs(admitted_at[1] - admitted_at[0]) >= 0.04

    _run(scenario)


def test_waiters_are_released_in_the_order_they_joined() -> None:
    """A pause ending with several requests queued releases them in arrival order."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            max_concurrent_requests=None,
            max_request_starts_per_second=100.0,
        )
        shared_backoff._record(PauseAll(retry_after=0.1))
        admitted_order: list[int] = []

        async def request(index: int) -> None:
            async with shared_backoff.admitted():
                admitted_order.append(index)

        async with asyncio.TaskGroup() as group:
            for index in range(3):
                _ = group.create_task(request(index))
                await asyncio.sleep(0.005)
        assert admitted_order == [0, 1, 2]

    _run(scenario)


def test_entry_is_immediate_when_nothing_blocks_it() -> None:
    """With a free permit, no pause, and a passed gap, entry does not wait."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        started_at = time.monotonic()
        await _enter_empty_block(shared_backoff.admitted())
        assert time.monotonic() - started_at < 0.05

    _run(scenario)


# --- the budget ---


def test_admitted_rejects_invalid_budgets_before_acquiring_anything() -> None:
    """Booleans, zero, negatives, non-finite values, and unrepresentable ints raise."""
    shared_backoff = _shared_backoff()
    for budget in (True, False, 0, -1, float("inf"), float("nan"), 10**1000):
        with pytest.raises(ValueError, match="budget"):
            _ = shared_backoff.admitted(budget=budget)
    assert _all_permits_free(shared_backoff)


def test_a_budget_expiring_in_the_queue_leaves_nothing_held() -> None:
    """Expiry while waiting for admission leaves the queue and returns the permit."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        shared_backoff._record(PauseAll(retry_after=0.5))
        with pytest.raises(GaveUpWaiting):
            await _enter_empty_block(shared_backoff.admitted(budget=0.05))
        assert len(shared_backoff._queue) == 0
        assert shared_backoff.event_counts["gave_up_waiting"] == 1
        assert _all_permits_free(shared_backoff)

    _run(scenario)


def test_a_budget_expiring_on_the_permit_takes_no_permit() -> None:
    """Expiry during permit acquisition never reaches the queue."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with shared_backoff.admitted():
                await release_holder.wait()

        holding = asyncio.create_task(holder())
        await asyncio.sleep(0.01)
        with pytest.raises(GaveUpWaiting):
            await _enter_empty_block(shared_backoff.admitted(budget=0.05))
        assert len(shared_backoff._queue) == 0
        release_holder.set()
        await holding
        assert _all_permits_free(shared_backoff)

    _run(scenario)


def test_cancellation_while_queued_leaves_an_empty_queue_and_a_full_permit_count() -> None:
    """A task cancelled behind a pause holds nothing afterwards."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff()
        shared_backoff._record(PauseAll(retry_after=0.5))
        waiting = asyncio.create_task(_enter_empty_block(shared_backoff.admitted()))
        await asyncio.sleep(0.05)
        _ = waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert len(shared_backoff._queue) == 0
        assert _all_permits_free(shared_backoff)

    _run(scenario)


# --- the merge rule ---


def test_reports_during_a_pause_extend_it_and_never_shrink_it() -> None:
    """Each report during a pause merges by keeping the later end."""
    shared_backoff = _shared_backoff()
    moment = [0.0]
    shared_backoff._clock = lambda: moment[0]
    for report_at, expected_end in ((0.0, 60.0), (59.0, 119.0), (118.0, 178.0), (177.0, 237.0)):
        moment[0] = report_at
        shared_backoff._record(PauseAll(retry_after=60.0))
        assert shared_backoff._pause_until == expected_end
    moment[0] = 178.0
    shared_backoff._record(PauseAll(retry_after=1.0))
    assert shared_backoff._pause_until == 237.0


def test_pause_ends_within_longest_wait_seconds_after_the_most_recent_report() -> None:
    """Every merged wait keeps the remaining pause within longest_wait_seconds."""
    shared_backoff = _shared_backoff()
    moment = [0.0]
    shared_backoff._clock = lambda: moment[0]
    shared_backoff._record(PauseAll(retry_after=60.0))
    for step in range(200):
        moment[0] += 1.0
        retry_after = 60.0 if step % 2 == 0 else None
        shared_backoff._record(PauseAll(retry_after=retry_after))
        assert shared_backoff._pause_until - moment[0] <= 60.0


# --- the wait ceiling ---


def test_the_first_pause_starts_from_the_floor() -> None:
    """The first pause draws under minimum_wait_ceiling_seconds."""
    shared_backoff = _shared_backoff()
    shared_backoff._set_wait_ceiling(10.0, _NEVER)
    assert shared_backoff._wait_ceiling == 1.0


def test_the_ceiling_grows_only_after_traffic_resumed() -> None:
    """A refusal before any admission since the previous pause does not grow the ceiling."""
    shared_backoff = _shared_backoff()
    shared_backoff._wait_ceiling = 4.0
    shared_backoff._last_admission_at = 3.0
    shared_backoff._set_wait_ceiling(6.0, 4.0)
    assert shared_backoff._wait_ceiling == 4.0
    shared_backoff._last_admission_at = 5.0
    shared_backoff._set_wait_ceiling(6.0, 4.0)
    assert shared_backoff._wait_ceiling == 8.0


def test_the_ceiling_growth_caps_at_longest_wait_seconds() -> None:
    """Growth never passes longest_wait_seconds."""
    shared_backoff = _shared_backoff()
    shared_backoff._wait_ceiling = 40.0
    shared_backoff._last_admission_at = 5.0
    shared_backoff._set_wait_ceiling(6.0, 4.0)
    assert shared_backoff._wait_ceiling == 60.0


def test_the_ceiling_decays_one_step_per_quiet_step_down_to_the_floor() -> None:
    """A 60s ceiling comes down 30, 15, 7.5, 3.75, 1.875, then the floor."""
    expected = [30.0, 15.0, 7.5, 3.75, 1.875, 1.0, 1.0]
    for quiet_steps, expected_ceiling in enumerate(expected, start=1):
        shared_backoff = _shared_backoff()
        shared_backoff._wait_ceiling = 60.0
        shared_backoff._set_wait_ceiling(100.0 + 60.0 * quiet_steps, 100.0)
        assert shared_backoff._wait_ceiling == expected_ceiling, f"steps={quiet_steps}"


def test_decay_wins_when_traffic_also_resumed() -> None:
    """A refusal after a full quiet step is a fresh incident, not evidence to grow on."""
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


def test_chosen_waits_are_positive_and_bounded_by_the_ceiling() -> None:
    """A wait of our own choosing lies in (0, ceiling] and varies."""
    draws = {shared_backoff_module._random_up_to(2.0) for _ in range(200)}
    assert all(0.0 < draw <= 2.0 for draw in draws)
    assert len(draws) > 1, "draws are constant, so the wait is not jittered"


def test_private_backoff_waits_grow_to_the_cap_and_honor_a_stated_floor() -> None:
    """Waits respect the growing ceiling and retry_after floor."""
    private_backoff = PrivateBackoff(
        _shared_backoff(
            minimum_wait_ceiling_seconds=1.0,
            wait_multiplier=2.0,
            longest_wait_seconds=4.0,
        )
    )
    ceilings = (1.0, 2.0, 4.0, 4.0)
    for ceiling in ceilings:
        assert 0.0 < private_backoff.next_wait(None) <= ceiling
    assert private_backoff.next_wait(3.5) >= 3.5
