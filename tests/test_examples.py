"""Verify the numbered examples' offline behavior."""

import asyncio
import importlib
from collections.abc import Sequence
from typing import Protocol, TypeIs

import pytest

from langchaint import (
    DispatchExceptionGroup,
    ToolCall,
    ToolMessage,
)


class _ToolFormsExample(Protocol):
    async def dispatch_with_approval(
        self,
        tool_calls: Sequence[ToolCall],
        approved_transfer_call_ids: frozenset[str],
    ) -> tuple[ToolMessage, ...]: ...


def _is_tool_forms_example(value: object) -> TypeIs[_ToolFormsExample]:
    return callable(getattr(value, "dispatch_with_approval", None))


def test_tool_approval_paths() -> None:
    """Dispatch approved calls and precompute denied calls."""
    example_module = importlib.import_module("examples.10_tool_forms_and_approval")
    assert _is_tool_forms_example(example_module)
    transfer_call = ToolCall(
        id="transfer_1",
        name="transfer_funds",
        args_json='{"transfer_id":"bank_1","recipient":"Ada","amount_in_usd":12.5}',
    )
    search_call = ToolCall(
        id="search_1",
        name="search_docs",
        args_json='{"query":"refunds"}',
    )

    approved_messages = asyncio.run(
        example_module.dispatch_with_approval(
            (transfer_call, search_call), frozenset({"transfer_1"})
        )
    )
    denied_messages = asyncio.run(
        example_module.dispatch_with_approval((transfer_call,), frozenset())
    )

    assert [message.is_error for message in approved_messages] == [False, False]
    assert denied_messages[0].is_error is True
    assert denied_messages[0].content == "The user declined this transfer."


def test_tool_defect_records_completed_app_data(capsys: pytest.CaptureFixture[str]) -> None:
    """Record settled `app_data` before propagating a tool defect."""
    example_module = importlib.import_module("examples.10_tool_forms_and_approval")
    assert _is_tool_forms_example(example_module)
    transfer_call = ToolCall(
        id="transfer_2",
        name="transfer_funds",
        args_json='{"transfer_id":"bank_2","recipient":"Grace","amount_in_usd":9.0}',
    )
    failing_call = ToolCall(
        id="search_2",
        name="search_docs",
        args_json='{"query":"raise server defect"}',
    )

    with pytest.raises(DispatchExceptionGroup):
        _ = asyncio.run(
            example_module.dispatch_with_approval(
                (transfer_call, failing_call), frozenset({"transfer_2"})
            )
        )

    output = capsys.readouterr().out
    assert "tool defects: 1" in output
    assert "recorded transfer: bank_2" in output
