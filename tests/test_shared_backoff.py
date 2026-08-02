"""SharedBackoff driven directly, without any adapter.

Each test drives admitted() blocks by hand to pin the entry, exit, pause, and ceiling behavior.
The async tests use real short waits; the ceiling arithmetic tests drive _record and
_set_wait_ceiling under a fake clock, which no timer reads because no request queues there.
"""

import asyncio
import gc
import time
import warnings
from collections.abc import Callable, Coroutine
from typing import override

import pytest

from langchaint import shared_backoff as shared_backoff_module
from langchaint.exceptions import GaveUpWaiting, ParserContractError
from langchaint.shared_backoff import (
    _NEVER,
    Admission,
    DoNotRetry,
    PauseAll,
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
    capacity: int | None = 1,
    minimum_wait_ceiling: float = 1.0,
    longest_wait: float = 60.0,
    wait_multiplier: float = 2.0,
    quiet_per_decay_step: float = 60.0,
    admission_gap: float = 0.005,
    on_parse_error: str = "raise",
) -> SharedBackoff:
    """Build a SharedBackoff with test-friendly defaults, overridable per test."""
    return SharedBackoff(
        parse=parse,
        failure_types=(ProviderFailure,),
        capacity=capacity,
        minimum_wait_ceiling=minimum_wait_ceiling,
        longest_wait=longest_wait,
        wait_multiplier=wait_multiplier,
        quiet_per_decay_step=quiet_per_decay_step,
        admission_gap=admission_gap,
        # pyrefly: ignore[bad-argument-type]  # str, so tests can pass invalid values
        on_parse_error=on_parse_error,
    )


class _RecordingSharedBackoff(SharedBackoff):
    """SharedBackoff that keeps every verdict _record received, for identity assertions."""

    def __init__(self, *, parse: Callable[[Exception], Verdict]) -> None:
        super().__init__(parse=parse, failure_types=(ProviderFailure,), capacity=1)
        self.recorded: list[Verdict] = []

    @override
    def _record(self, verdict: Verdict) -> None:
        self.recorded.append(verdict)
        super()._record(verdict)


def _run(scenario: Callable[[], Coroutine[None, None, None]]) -> None:
    """Run one async scenario under a hang guard."""
    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def _all_permits_free(shared_backoff: SharedBackoff) -> bool:
    """Report whether every capacity permit is free: none held, none leaked, no live waiter queued.

    Reads the semaphore's _value, which is CPython-private, because no public member exposes the
    count; a stdlib rename breaks these tests visibly, not the shipped code.
    """
    permits = shared_backoff._capacity_permits
    assert permits is not None
    return permits._value == shared_backoff.capacity and not permits.locked()


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


def test_constructor_rejects_invalid_capacity() -> None:
    """Reject every capacity outside None and the positive non-bool ints."""
    for capacity in (True, False, 0, -1, 1.0, "8"):
        with pytest.raises(ValueError, match="capacity"):
            _ = _shared_backoff(capacity=capacity)  # pyrefly: ignore[bad-argument-type]
    _ = _shared_backoff(capacity=None)
    _ = _shared_backoff(capacity=8)


def test_constructor_rejects_unknown_on_parse_error() -> None:
    """on_parse_error takes exactly "raise" and "retry_this_one"."""
    with pytest.raises(ValueError, match="on_parse_error"):
        _ = _shared_backoff(on_parse_error="quiet")
    _ = _shared_backoff(on_parse_error="retry_this_one")


def test_constructor_rejects_a_coroutine_parse() -> None:
    """A coroutine parse can never be awaited by the synchronous exit."""

    async def parse(_failure: Exception) -> Verdict:
        return DoNotRetry()

    with pytest.raises(ValueError, match="parse"):
        # pyrefly: ignore[bad-argument-type]  # the rejection under test
        _ = _shared_backoff(parse=parse)


def test_constructor_rejects_failure_types_wider_than_exception() -> None:
    """Exception itself and anything reaching only BaseException are rejected."""
    for failure_type in (Exception, BaseException, KeyboardInterrupt, SystemExit):
        with pytest.raises(ValueError, match="failure_types"):
            _ = SharedBackoff(
                parse=_retry_verdict,
                # pyrefly: ignore[bad-argument-type]  # the rejection under test
                failure_types=(failure_type,),
                capacity=1,
            )
    with pytest.raises(ValueError, match="failure_types"):
        _ = SharedBackoff(
            parse=_retry_verdict,
            failure_types=(asyncio.CancelledError,),  # pyrefly: ignore[bad-argument-type]
            capacity=1,
        )


def test_constructor_rejects_empty_failure_types() -> None:
    """An empty failure_types would make the exit parse nothing and record nothing."""
    with pytest.raises(ValueError, match="failure_types"):
        _ = SharedBackoff(parse=_retry_verdict, failure_types=(), capacity=1)


def test_constructor_rejects_invalid_numeric_settings() -> None:
    """Every numeric setting shares one acceptance rule: exactly int or float, finite, positive."""
    valid = {
        "minimum_wait_ceiling": 1.0,
        "longest_wait": 60.0,
        "wait_multiplier": 2.0,
        "quiet_per_decay_step": 60.0,
        "admission_gap": 0.02,
    }
    for invalid_value in (True, 0, -1.0, float("inf"), float("nan"), "1"):
        for name in valid:
            settings = {**valid, name: invalid_value}
            with pytest.raises(ValueError, match=name):
                _ = SharedBackoff(
                    parse=_retry_verdict,
                    failure_types=(ProviderFailure,),
                    capacity=1,
                    # pyrefly: ignore[bad-argument-type]  # the rejection under test
                    minimum_wait_ceiling=settings["minimum_wait_ceiling"],
                    # pyrefly: ignore[bad-argument-type]
                    longest_wait=settings["longest_wait"],
                    # pyrefly: ignore[bad-argument-type]
                    wait_multiplier=settings["wait_multiplier"],
                    # pyrefly: ignore[bad-argument-type]
                    quiet_per_decay_step=settings["quiet_per_decay_step"],
                    # pyrefly: ignore[bad-argument-type]
                    admission_gap=settings["admission_gap"],
                )


def test_constructor_rejects_wait_multiplier_at_or_below_one() -> None:
    """A multiplier of 1 would never grow or shrink the ceiling."""
    with pytest.raises(ValueError, match="wait_multiplier"):
        _ = _shared_backoff(wait_multiplier=1.0)


def test_constructor_rejects_longest_wait_below_the_floor() -> None:
    """longest_wait must be at least minimum_wait_ceiling."""
    with pytest.raises(ValueError, match="longest_wait"):
        _ = _shared_backoff(minimum_wait_ceiling=10.0, longest_wait=1.0)


def test_constructor_rejects_an_unrepresentable_ceiling_ratio() -> None:
    """A ratio the decay arithmetic cannot hold raises ValueError, never OverflowError.

    5e-324 under 1e308 makes longest_wait / minimum_wait_ceiling infinite, and 10**1000 is an int
    float() cannot represent; unchecked, each surfaced as OverflowError out of construction.
    """
    with pytest.raises(ValueError, match="finite"):
        _ = _shared_backoff(minimum_wait_ceiling=5e-324, longest_wait=1e308)
    with pytest.raises(ValueError, match="minimum_wait_ceiling"):
        _ = _shared_backoff(minimum_wait_ceiling=10**1000, longest_wait=1e308)


def test_constructor_accepts_longest_wait_equal_to_the_floor() -> None:
    """A one-value ceiling range is legal; the ceiling then never moves."""
    shared_backoff = _shared_backoff(minimum_wait_ceiling=2.0, longest_wait=2.0)
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


def test_a_do_not_retry_verdict_changes_no_shared_state() -> None:
    """DoNotRetry lands on the admission and starts no pause."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(parse=lambda _failure: DoNotRetry())
        admission = await _fail_one_attempt(shared_backoff)
        assert admission.verdict == DoNotRetry()
        assert shared_backoff._pause_until == _NEVER

    _run(scenario)


def test_a_raising_record_still_returns_the_permit() -> None:
    """The release sits in a finally, so a raise out of recording cannot leak capacity."""

    class _BrokenRecordSharedBackoff(SharedBackoff):
        @override
        def _record(self, verdict: Verdict) -> None:
            raise RuntimeError("record broke")

    async def scenario() -> None:
        shared_backoff = _BrokenRecordSharedBackoff(
            parse=_retry_verdict, failure_types=(ProviderFailure,), capacity=1
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


def test_a_non_verdict_return_raises_parser_contract_error() -> None:
    """A parse returning something that is not a verdict is a contract violation."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            parse=lambda _failure: "nonsense"  # pyrefly: ignore[bad-argument-type]
        )
        with pytest.raises(ParserContractError) as caught:
            await _raise_in_block(shared_backoff.admitted(), ProviderFailure("429"))
        assert caught.value.__cause__ is None
        assert isinstance(caught.value.__context__, ProviderFailure)

    _run(scenario)


def test_a_parse_returning_a_coroutine_is_a_contract_violation_with_no_warning() -> None:
    """The wrapper closes the coroutine, so no unawaited-coroutine warning fires."""

    async def verdict_coroutine() -> Verdict:
        return DoNotRetry()

    async def scenario() -> None:
        shared_backoff = _shared_backoff(
            # A callable whose __call__ is synchronous but returns an awaitable, which the
            # construction-time coroutine-function check cannot catch.
            parse=lambda _failure: verdict_coroutine()  # pyrefly: ignore[bad-argument-type]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ParserContractError):
                await _raise_in_block(shared_backoff.admitted(), ProviderFailure("429"))
            _ = gc.collect()

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
    """Accept exactly a positive finite int or float, capped at longest_wait; anything else is None.

    The wrapper normalizes before either _record or Admission.verdict sees the verdict.
    """
    cases: list[tuple[object, float | None]] = [
        (-5, None),
        (0, None),
        (float("nan"), None),
        (float("-inf"), None),
        (float("inf"), None),
        (True, None),
        ("60", None),
        (-(10**1000), None),
        (9999, 60.0),
        (10**1000, 60.0),
        (60, 60.0),
        (5.0, 5.0),
    ]

    async def scenario() -> None:
        for stated, expected in cases:
            shared_backoff = _shared_backoff(
                # pyrefly: ignore[bad-argument-type]  # invalid retry_after is the case under test
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


def test_admissions_are_spaced_by_the_admission_gap() -> None:
    """Two requests entering together are admitted one gap apart."""

    async def scenario() -> None:
        shared_backoff = _shared_backoff(capacity=None, admission_gap=0.05)
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
        shared_backoff = _shared_backoff(capacity=None, admission_gap=0.01)
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
    """Booleans, zero, negatives, non-finite values, strings, and unrepresentable ints raise."""
    shared_backoff = _shared_backoff()
    for budget in (True, False, 0, -1, float("inf"), float("nan"), "5", 10**1000):
        with pytest.raises(ValueError, match="budget"):
            _ = shared_backoff.admitted(budget=budget)  # pyrefly: ignore[bad-argument-type]
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


def test_the_pause_never_ends_more_than_longest_wait_after_the_most_recent_report() -> None:
    """Every merged-in wait is capped, so the remaining pause never exceeds longest_wait."""
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
    """Nothing has gone wrong yet, so the first pause draws under minimum_wait_ceiling."""
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


def test_the_ceiling_growth_caps_at_longest_wait() -> None:
    """Growth never passes longest_wait."""
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
    """Each wait lies in (0, ceiling], the ceiling doubles to longest_wait, and retry_after floors a wait."""
    private_backoff = PrivateBackoff(
        _shared_backoff(minimum_wait_ceiling=1.0, wait_multiplier=2.0, longest_wait=4.0)
    )
    ceilings = (1.0, 2.0, 4.0, 4.0)
    for ceiling in ceilings:
        assert 0.0 < private_backoff.next_wait(None) <= ceiling
    assert private_backoff.next_wait(3.5) >= 3.5
