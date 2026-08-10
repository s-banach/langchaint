"""Run many run_ones concurrently under a bound on how many are pending at once.

A run_one is pending from the moment run_many starts its task until that task settles.
max_pending caps the pending count, creating each task as an earlier one settles.
This module models nothing about LLMs and imports nothing from langchaint.
"""

import asyncio
from collections.abc import Callable, Coroutine, Sequence

_PENDING_TASKS_PER_CONCURRENT_REQUEST = 2
_SPARE_PENDING_TASKS = 8
_MAX_PENDING_TASKS_WITHOUT_A_CONCURRENCY_BOUND = 1000


def max_pending_for_requests(max_concurrent_requests: int | None) -> int:
    """Derive the pending task bound from the request concurrency bound.

    The extra tasks keep permits supplied while retries wait privately.
    An absent concurrency bound still receives a finite pending task bound.

    Raises:
        ValueError: `max_concurrent_requests` is boolean or below one.
    """
    if max_concurrent_requests is None:
        return _MAX_PENDING_TASKS_WITHOUT_A_CONCURRENCY_BOUND
    if isinstance(max_concurrent_requests, bool) or max_concurrent_requests < 1:
        raise ValueError(
            "max_concurrent_requests must be None or a positive int, "
            f"got {max_concurrent_requests!r}"
        )
    return max_concurrent_requests * _PENDING_TASKS_PER_CONCURRENT_REQUEST + _SPARE_PENDING_TASKS


async def run_many[OutputT](
    run_ones: Sequence[Callable[[], Coroutine[object, object, OutputT]]],
    *,
    max_pending: int | None,
) -> list[OutputT]:
    """Run every run_one and return results in run_ones order.

    asyncio.create_task rejects other Awaitable implementations.
    Bind every argument before passing each run_one.
    max_pending None starts every run_one at once.
    Each run_one runs in its own task, so its ContextVar changes reach no other run_one.
    run_one references are stored before the first task starts.
    Later run_ones mutations change nothing.
    A failure cancels the tasks still pending, waits for them to settle, then propagates.
    Neither a failure nor an outer cancellation starts a further run_one.
    Where one wait wave carries several failures, the lowest run_one index propagates.

    Raises:
        ValueError: max_pending is a bool, or an int below 1.
        TypeError: A run_one returned something other than a Coroutine.
        asyncio.CancelledError: an outer scope cancelled this call.
        BaseException: A run_one raised it.
    """
    # bool is rejected explicitly because it subclasses int, so a type checker admits True here.
    if max_pending is not None and (isinstance(max_pending, bool) or max_pending < 1):
        raise ValueError(f"max_pending must be None or a positive int, got {max_pending!r}")
    run_one_snapshot = tuple(run_ones)
    if max_pending is None:
        max_pending = len(run_one_snapshot)
    result_by_run_one_index: dict[int, OutputT] = {}
    pending_run_one_index_by_task: dict[asyncio.Task[OutputT], int] = {}
    next_run_one_index = 0

    try:
        while pending_run_one_index_by_task or next_run_one_index < len(run_one_snapshot):
            while len(pending_run_one_index_by_task) < max_pending and next_run_one_index < len(
                run_one_snapshot
            ):
                task = asyncio.create_task(run_one_snapshot[next_run_one_index]())
                pending_run_one_index_by_task[task] = next_run_one_index
                next_run_one_index += 1
            settled_tasks, _ = await asyncio.wait(
                pending_run_one_index_by_task, return_when=asyncio.FIRST_COMPLETED
            )
            failure_to_raise: BaseException | None = None
            settled_tasks_in_run_one_order = sorted(
                settled_tasks, key=pending_run_one_index_by_task.__getitem__
            )
            for task in settled_tasks_in_run_one_order:
                run_one_index = pending_run_one_index_by_task.pop(task)
                try:
                    result_by_run_one_index[run_one_index] = task.result()
                except BaseException as failure:  # noqa: BLE001 (observe every BaseException)
                    if failure_to_raise is None:
                        failure_to_raise = failure
            if failure_to_raise is not None:
                raise failure_to_raise  # noqa: TRY301 (handler below settles the pending tasks)
    except BaseException:
        # cancel() requests cancellation, but the task may remain active.
        # Its coroutine resumes to handle CancelledError.
        # Await every task before propagating.
        # No task outlives its run_many call.
        for task in pending_run_one_index_by_task:
            _ = task.cancel()
        while pending_run_one_index_by_task:
            try:
                settled_tasks, _ = await asyncio.wait(
                    pending_run_one_index_by_task,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                continue
            for task in settled_tasks:
                _ = pending_run_one_index_by_task.pop(task)
                try:
                    _ = task.result()
                except BaseException:  # noqa: BLE001 (settle every cancelled task)
                    pass
        raise
    return [
        result_by_run_one_index[run_one_index] for run_one_index in range(len(run_one_snapshot))
    ]
