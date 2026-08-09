"""The lifecycle state shared by an account and its request clients."""

from __future__ import annotations


class AccountClosedError(RuntimeError):
    """An account rejected work after closure."""


class AccountState:
    """Share account closure state with every account-created request client."""

    def __init__(self) -> None:
        """Start open and outside a context manager."""
        self._closed = False
        self._entered = False

    def ensure_open(self) -> None:
        """Raise when this account is closed.

        Raises:
            RuntimeError: This account is closed.
        """
        if self._closed:
            raise AccountClosedError("Account is closed")

    def enter(self) -> None:
        """Record one context-manager entry.

        Raises:
            RuntimeError: This account is closed or already entered.
        """
        self.ensure_open()
        if self._entered:
            raise RuntimeError("Account is already entered")
        self._entered = True

    def close(self) -> None:
        """Mark this account closed."""
        self._closed = True
