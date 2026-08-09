"""Shared implementation for concrete accounts."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Self

from langchaint.account_state import AccountState
from langchaint.llm import LLM
from langchaint.shared_backoff import SharedBackoff, Verdict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

    from langchaint.adapter import Adapter


class AccountBase:
    """Shared lifecycle and `SharedBackoff` construction for concrete accounts."""

    def __init__(  # noqa: PLR0913 (every request policy reaches SharedBackoff)
        self,
        *,
        parse: Callable[[Exception], Verdict],
        failure_types: tuple[type[Exception], ...],
        max_concurrent_requests: int | None,
        max_request_starts_per_second: float,
        minimum_wait_ceiling_seconds: float,
        longest_wait_seconds: float,
        wait_multiplier: float,
        quiet_seconds_per_decay_step: float,
    ) -> None:
        """Construct lifecycle state, owned resources, and `SharedBackoff`.

        Raises:
            ValueError: A `SharedBackoff` setting is invalid.
        """
        self._state = AccountState()
        self._owned_resources = AsyncExitStack()
        self._close_task: asyncio.Task[None] | None = None
        self._shared_backoff = SharedBackoff(
            parse=parse,
            failure_types=failure_types,
            max_concurrent_requests=max_concurrent_requests,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
            max_request_starts_per_second=max_request_starts_per_second,
        )

    def _register_owned_close(self, close: Callable[[], Awaitable[None]]) -> None:
        _ = self._owned_resources.push_async_callback(close)

    def _register_owned_sync_close(self, close: Callable[[], None]) -> None:
        _ = self._owned_resources.callback(close)

    def _llm(self, adapter: Adapter) -> LLM:
        """Build an `LLM` sharing this account's lifecycle and `SharedBackoff`."""
        llm = LLM(adapter, shared_backoff=self._shared_backoff)
        llm._account_state = self._state  # noqa: SLF001 (account-to-LLM boundary)
        return llm

    async def aclose(self) -> None:
        """Close this account and its owned resources once.

        Raises:
            asyncio.CancelledError: This caller is cancelled while resource closure continues.
            Exception: An owned resource close operation fails.
        """
        if self._close_task is None:
            self._state.close()
            self._close_task = asyncio.create_task(self._owned_resources.aclose())
        await asyncio.shield(self._close_task)

    async def __aenter__(self) -> Self:
        """Enter this account once.

        Raises:
            RuntimeError: This account is closed or already entered.
        """
        self._state.enter()
        return self

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
        await self.aclose()
