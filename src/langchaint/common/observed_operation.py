"""Define the handle an observer returns for one input or one tool dispatch."""

import logging
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Protocol

_logger = logging.getLogger(__name__)


class ObservedOperation[OutcomeT](Protocol):
    """One operation that an observer follows from start to end.

    The observer creates the handle when the operation starts.
    langchaint calls `conclude` at most once, then `end` exactly once.
    An operation cancelled before its conclusion receives `end` without `conclude`.
    A stream instead concludes with its `abandoned` record.
    langchaint logs an exception raised by any method and continues, so a failing observer loses only its own record.
    """

    def current(self) -> AbstractContextManager[None]:
        """Mark the operation as the active context while langchaint awaits it.

        langchaint enters this around an input's retry loop and around a tool function.
        A stream never enters it, because the application's code runs between stream items.
        langchaint exits it without the block's exception, so the context can neither see nor suppress it.
        """
        ...

    def conclude(self, outcome: OutcomeT) -> None:
        """Receive the operation's outcome."""
        ...

    def end(self) -> None:
        """Close the operation."""
        ...


class _UnobservedOperation:
    """An `ObservedOperation` that records nothing."""

    def current(self) -> AbstractContextManager[None]:
        """Return a context that changes nothing."""
        return nullcontext()

    def conclude(self, outcome: object) -> None:
        """Ignore the outcome."""

    def end(self) -> None:
        """Do nothing."""


UNOBSERVED_OPERATION: ObservedOperation[object] = _UnobservedOperation()
"""The handle for an operation without an observer."""


@contextmanager
def _logging_failures(what: str) -> Generator[None]:
    """Log an `Exception` the block raises instead of letting it out.

    Only `Exception` is caught, so a cancellation still reaches the caller.
    """
    try:
        yield
    except Exception:
        _logger.warning(
            "the observer raised while %s; its record is incomplete", what, exc_info=True
        )


@contextmanager
def _guarded_context(current: Callable[[], AbstractContextManager[None]]) -> Generator[None]:
    """Enter and exit the context `current` returns around the block, logging the observer's failures.

    The context exits without the block's exception, so it can neither see nor suppress it.
    """
    context: AbstractContextManager[None] | None = None
    with _logging_failures("making the operation current"):
        entering = current()
        entering.__enter__()
        context = entering
    try:
        yield
    finally:
        if context is not None:
            with _logging_failures("restoring the previous context"):
                _ = context.__exit__(None, None, None)


class _GuardedOperation[OutcomeT]:
    """An `ObservedOperation` that logs the exceptions of the operation it delegates to."""

    def __init__(self, operation: ObservedOperation[OutcomeT]) -> None:
        self._operation = operation

    def current(self) -> AbstractContextManager[None]:
        """Enter the delegate's context, logging its failures."""
        return _guarded_context(self._operation.current)

    def conclude(self, outcome: OutcomeT) -> None:
        """Pass the outcome to the delegate, logging its failure."""
        with _logging_failures("concluding the operation"):
            self._operation.conclude(outcome)

    def end(self) -> None:
        """End the delegate, logging its failure."""
        with _logging_failures("ending the operation"):
            self._operation.end()


def start_observed_operation[OutcomeT](
    start: Callable[[], ObservedOperation[OutcomeT]],
) -> ObservedOperation[OutcomeT]:
    """Start an operation with an observer's `start`, so that no observer failure reaches the caller.

    A failing `start` leaves the operation unobserved.
    """
    with _logging_failures("starting the operation"):
        return _GuardedOperation(start())
    # Reached when `start` raises, because `_logging_failures` suppresses the exception.
    return UNOBSERVED_OPERATION
