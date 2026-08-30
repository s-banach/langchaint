"""Cover cancellation-safe synchronous thread execution."""

import asyncio
import threading

import pytest

from langchaint._generate_many_records import _run_resume_io
from langchaint.cancellation import to_thread_cancellation_safe


def test_run_resume_io_settles_before_cancellation_propagates() -> None:
    """Assert `_run_resume_io` waits for started work after cancellation."""

    async def scenario() -> None:
        """Cancel the awaiting task while its thread remains blocked."""
        started = threading.Event()
        release = threading.Event()

        def block() -> None:
            started.set()
            assert release.wait(timeout=5.0)

        task = asyncio.create_task(_run_resume_io(block))
        assert await asyncio.to_thread(started.wait, 5.0)
        _ = task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_thread_work_settles_before_cancellation_propagates() -> None:
    """Cancellation waits for synchronous work to settle."""

    async def scenario() -> None:
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def work() -> int:
            started.set()
            _ = release.wait()
            finished.set()
            return 1

        task = asyncio.create_task(to_thread_cancellation_safe(work))
        await asyncio.to_thread(started.wait)
        _ = task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            _ = await task
        assert finished.is_set()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_thread_failure_propagates() -> None:
    """A synchronous failure reaches the caller unchanged."""

    class FailureError(Exception):
        """Identify the expected failure."""

    def fail() -> None:
        raise FailureError("boom")

    async def scenario() -> None:
        with pytest.raises(FailureError, match="boom"):
            await to_thread_cancellation_safe(fail)

    asyncio.run(scenario())
