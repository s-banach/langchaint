"""Cover run_many: validation, the pending bound, input order, and the two cancellation paths."""

import asyncio
from contextvars import ContextVar
from enum import IntEnum

import pytest

from langchaint.run_many import run_many


class _RunManyFailure(BaseException):
    """Identify a run_many task failure."""


def test_run_many_validates_max_pending_before_running_items() -> None:
    """Invalid max_pending starts no run_one task."""

    async def scenario() -> None:
        """Try every invalid max_pending."""
        called = False

        async def run_one(input_value: int) -> int:
            """Record execution and return the input."""
            nonlocal called
            called = True
            return input_value

        for invalid_max_pending in (True, False, 0, -1):
            with pytest.raises(ValueError, match="max_pending"):
                _ = await run_many([1], run_one, max_pending=invalid_max_pending)
        assert not called

    asyncio.run(scenario())


def test_run_many_accepts_an_int_subclass_that_is_not_bool() -> None:
    """An IntEnum is a positive non-bool int, so run_many accepts it as max_pending."""

    class Concurrency(IntEnum):
        """Stand in for an application's own bound, spelled as an enum."""

        LOW = 2

    async def scenario() -> None:
        """Run two inputs under an IntEnum bound."""

        async def run_one(input_value: int) -> int:
            """Return the input."""
            return input_value

        assert await run_many([1, 2], run_one, max_pending=Concurrency.LOW) == [1, 2]

    asyncio.run(scenario())


def test_run_many_empty_inputs_start_no_task() -> None:
    """Empty inputs return an empty result after max_pending validation."""

    async def scenario() -> None:
        """Run an empty input sequence."""
        called = False

        async def run_one(input_value: int) -> int:
            """Record execution and return the input."""
            nonlocal called
            called = True
            return input_value

        assert await run_many([], run_one, max_pending=2) == []
        assert not called

    asyncio.run(scenario())


def test_run_many_snapshots_inputs_before_starting_tasks() -> None:
    """Mutating the input sequence cannot change the inputs still to start."""

    async def scenario() -> None:
        """Clear the input list from its first run_one call."""
        inputs = [0, 1, 2]

        async def run_one(input_value: int) -> int:
            """Clear inputs after run_many captured them."""
            inputs.clear()
            return input_value

        assert await run_many(inputs, run_one, max_pending=1) == [0, 1, 2]

    asyncio.run(scenario())


def test_run_many_bounds_pending_refills_and_preserves_order_and_context() -> None:
    """max_pending=2 refills immediately without sharing ContextVar bindings."""

    async def scenario() -> None:
        """Complete four inputs outside input order."""
        context_value = ContextVar("run_many_context", default="default")
        caller_token = context_value.set("caller")
        started_events = [asyncio.Event() for _ in range(4)]
        finish_events = [asyncio.Event() for _ in range(4)]
        started_inputs: list[int] = []
        pending_count = 0
        peak_pending_count = 0

        async def run_one(input_index: int) -> str:
            """Wait for this input's finish event."""
            nonlocal pending_count, peak_pending_count
            assert context_value.get() == "caller"
            _ = context_value.set(f"input-{input_index}")
            pending_count += 1
            peak_pending_count = max(peak_pending_count, pending_count)
            started_inputs.append(input_index)
            started_events[input_index].set()
            try:
                await finish_events[input_index].wait()
                assert context_value.get() == f"input-{input_index}"
                return str(input_index)
            finally:
                pending_count -= 1

        try:
            task_count_before_batch = len(asyncio.all_tasks())
            batch_task = asyncio.create_task(run_many(range(4), run_one, max_pending=2))
            await started_events[0].wait()
            await started_events[1].wait()
            assert started_inputs == [0, 1]
            # batch_task plus one task per pending input, so two run_one tasks live here, not four.
            # Creating every task up front and holding the extras on a semaphore fails this.
            assert len(asyncio.all_tasks()) == task_count_before_batch + 3
            finish_events[1].set()
            await started_events[2].wait()
            assert started_inputs == [0, 1, 2]
            finish_events[0].set()
            await started_events[3].wait()
            assert started_inputs == [0, 1, 2, 3]
            finish_events[2].set()
            finish_events[3].set()
            assert await batch_task == ["0", "1", "2", "3"]
            assert peak_pending_count == 2
        finally:
            context_value.reset(caller_token)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_run_many_cancellation_settles_started_tasks_and_skips_unstarted_inputs() -> None:
    """Outer cancellation settles two started tasks and starts no further input."""

    async def scenario() -> None:
        """Cancel a four-input run after two inputs start."""
        started_events = [asyncio.Event() for _ in range(2)]
        never_finishes = asyncio.Event()
        started_inputs: list[int] = []
        settled_inputs: list[int] = []

        async def run_one(input_index: int) -> None:
            """Wait until cancellation.

            Raises:
                asyncio.CancelledError: run_many cancelled this task.
            """
            started_inputs.append(input_index)
            if input_index < len(started_events):
                started_events[input_index].set()
            try:
                await never_finishes.wait()
            finally:
                settled_inputs.append(input_index)

        batch_task = asyncio.create_task(run_many(range(4), run_one, max_pending=2))
        await asyncio.gather(started_events[0].wait(), started_events[1].wait())
        _ = batch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch_task
        assert started_inputs == [0, 1]
        assert sorted(settled_inputs) == [0, 1]

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_run_many_base_exception_settles_sibling_and_skips_unstarted_input() -> None:
    """A BaseException settles its started sibling and starts no further input."""

    async def scenario() -> None:
        """Raise after two inputs start."""
        both_started = asyncio.Event()
        never_finishes = asyncio.Event()
        started_inputs: list[int] = []
        settled_inputs: list[int] = []

        async def run_one(input_index: int) -> None:
            """Fail input zero after input one starts.

            Raises:
                _RunManyFailure: input zero failed.
                asyncio.CancelledError: run_many cancelled input one.
            """
            started_inputs.append(input_index)
            if len(started_inputs) == 2:
                both_started.set()
            try:
                await both_started.wait()
                if input_index == 0:
                    raise _RunManyFailure("first input failed")
                await never_finishes.wait()
            finally:
                settled_inputs.append(input_index)

        with pytest.raises(_RunManyFailure, match="first input failed"):
            _ = await run_many(range(3), run_one, max_pending=2)
        assert started_inputs == [0, 1]
        assert sorted(settled_inputs) == [0, 1]

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_run_many_selects_the_lowest_input_failure_from_one_wait() -> None:
    """Simultaneous failures propagate the lowest input index's failure."""

    async def scenario() -> None:
        """Fail every initial task before run_many resumes."""

        async def run_one(input_index: int) -> None:
            """Fail with the input index.

            Raises:
                _RunManyFailure: always.
            """
            raise _RunManyFailure(str(input_index))

        with pytest.raises(_RunManyFailure, match=r"^0$"):
            _ = await run_many(range(3), run_one, max_pending=3)

    asyncio.run(scenario())
