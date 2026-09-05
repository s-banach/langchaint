"""Run synchronous work without abandoning it during cancellation."""

import asyncio
from collections.abc import Callable


async def await_task_cancellation_safe[ResultT](task: asyncio.Task[ResultT]) -> ResultT:
    """Settle `task` before propagating caller cancellation.

    Args:
        task: The task to settle.

    Raises:
        asyncio.CancelledError: The caller was cancelled after `task` settled.
        BaseException: `task` raised it before caller cancellation.
    """
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        _ = task.exception()
        raise


async def to_thread_cancellation_safe[ResultT](function: Callable[[], ResultT]) -> ResultT:
    """Run `function` in a thread and settle it before propagating cancellation.

    The function receives no cancellation signal from `asyncio`.

    Args:
        function: The zero-argument synchronous function to run.

    Raises:
        asyncio.CancelledError: The caller was cancelled after `function` settled.
        BaseException: `function` raised it before caller cancellation.
    """
    task = asyncio.create_task(asyncio.to_thread(function))
    return await await_task_cancellation_safe(task)
