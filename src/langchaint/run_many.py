"""Run many inputs concurrently under a bound on how many are pending at once.

An input is pending from the moment run_many starts its task until that task settles.
max_pending caps the pending count, creating each task as an earlier one settles.
This module models nothing about LLMs and imports nothing from langchaint.
"""

import asyncio
from collections.abc import Sequence
from typing import Protocol


class RunOne[InputT, OutputT](Protocol):
    """Produce one result from one input.

    Implement it with async def.
    run_many hands the returned coroutine to asyncio.create_task, which rejects an Awaitable that
    is not a Coroutine.
    """

    async def __call__(self, input_value: InputT, /) -> OutputT:
        """Produce the result.

        Raises:
            BaseException: run_many cancels the tasks still pending and propagates it.
        """
        ...


async def run_many[InputT, OutputT](
    inputs: Sequence[InputT],
    run_one: RunOne[InputT, OutputT],
    *,
    max_pending: int | None,
) -> list[OutputT]:
    """Run every input through run_one and return the results in input order.

    max_pending None starts every input at once.
    Each run_one call runs in its own task, so a ContextVar one call sets reaches no other input.
    inputs is copied before the first task starts, so mutating it afterwards changes nothing.
    A failure cancels the tasks still pending, waits for them to settle, then propagates.
    Neither a failure nor an outer cancellation starts a further input.
    Where one wait wave carries several failures, the one with the lowest input index propagates.

    Raises:
        ValueError: max_pending is a bool, or an int below 1.
        asyncio.CancelledError: an outer scope cancelled this call.
        BaseException: a run_one call raised it.
    """
    # bool is rejected explicitly because it subclasses int, so a type checker admits True here.
    if max_pending is not None and (isinstance(max_pending, bool) or max_pending < 1):
        raise ValueError(f"max_pending must be None or a positive int, got {max_pending!r}")
    input_snapshot = tuple(inputs)
    if max_pending is None:
        max_pending = len(input_snapshot)
    result_by_input_index: dict[int, OutputT] = {}
    pending_input_index_by_task: dict[asyncio.Task[OutputT], int] = {}
    next_input_index = 0

    try:
        while pending_input_index_by_task or next_input_index < len(input_snapshot):
            while len(pending_input_index_by_task) < max_pending and next_input_index < len(
                input_snapshot
            ):
                task = asyncio.create_task(run_one(input_snapshot[next_input_index]))
                pending_input_index_by_task[task] = next_input_index
                next_input_index += 1
            settled_tasks, _ = await asyncio.wait(
                pending_input_index_by_task, return_when=asyncio.FIRST_COMPLETED
            )
            failure_by_input_index: dict[int, BaseException] = {}
            for task in settled_tasks:
                input_index = pending_input_index_by_task.pop(task)
                try:
                    result_by_input_index[input_index] = task.result()
                except BaseException as failure:  # noqa: BLE001 (observe every BaseException)
                    failure_by_input_index[input_index] = failure
            if failure_by_input_index:
                lowest_failed_index = min(failure_by_input_index)
                raise failure_by_input_index[lowest_failed_index]  # noqa: TRY301 (handler below settles the pending tasks)
    except BaseException:
        # cancel() only requests cancellation: the task is still running when it returns, and its
        # coroutine resumes to handle the CancelledError. Await each one before propagating, so no
        # task outlives the run_many call that created it.
        for task in pending_input_index_by_task:
            _ = task.cancel()
        await asyncio.gather(*pending_input_index_by_task, return_exceptions=True)
        raise
    return [result_by_input_index[input_index] for input_index in range(len(input_snapshot))]
