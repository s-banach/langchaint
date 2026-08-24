"""Run zero-argument async callables under a pending-task bound.

`max_pending` limits tasks from creation through completion.
"""

import asyncio
from collections.abc import Callable, Coroutine, Sequence

_PENDING_TASKS_PER_CONCURRENT_REQUEST = 2
_SPARE_PENDING_TASKS = 8
_MAX_PENDING_TASKS_WITHOUT_A_CONCURRENCY_BOUND = 1000


def max_pending_for_requests(max_concurrent_requests: int | None) -> int:
    """Derive the pending task bound from the request concurrency bound.

    Extra tasks keep request permits supplied during retry waits.

    Args:
        max_concurrent_requests: The request concurrency bound, or `None` when unbounded.

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
    """Run each `run_one` and preserve input order.

    `max_pending=None` starts every call concurrently.
    A failure cancels and settles pending tasks before propagating.
    Concurrent failures propagate by the lowest input index.

    Args:
        run_ones: The zero-argument async callables to run.
        max_pending: The pending-task bound, or `None` to start every callable concurrently.

    Raises:
        ValueError: `max_pending` is boolean or below one.
        TypeError: A `run_one` returns a non-coroutine.
        asyncio.CancelledError: The caller cancels this function.
        BaseException: A `run_one` raises it.
    """
    # Reject `bool` explicitly because it subclasses `int`.
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
        # Settle every task before propagating.
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
