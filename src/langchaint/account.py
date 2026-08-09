"""The SDK-free `Account` protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType

__all__ = ["Account"]


class Account(Protocol):
    """The common lifecycle of provider accounts."""

    async def aclose(self) -> None:
        """Close this account and its owned resources.

        Raises:
            asyncio.CancelledError: This caller is cancelled while resource closure continues.
            Exception: An owned resource close operation fails.
        """
        ...

    async def __aenter__(self) -> Self:
        """Enter this account once.

        Raises:
            RuntimeError: This account is closed or already entered.
        """
        ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close this account.

        Raises:
            asyncio.CancelledError: This caller is cancelled while resource closure continues.
            Exception: An owned resource close operation fails.
        """
        ...
