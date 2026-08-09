"""Verify the numbered examples' offline behavior."""

import asyncio
import importlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Protocol, TypeIs

import pytest
from pydantic import BaseModel, ValidationError

from langchaint import (
    DispatchExceptionGroup,
    Message,
    ReasoningDelta,
    StreamItem,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
    messages_from_json,
)


class _StreamingExample(Protocol):
    def print_stream_item(self, item: StreamItem) -> None: ...


class _ToolFormsExample(Protocol):
    TransferArgs: type[BaseModel]

    async def dispatch_with_approval(
        self,
        tool_calls: Sequence[ToolCall],
        approved_transfer_call_ids: frozenset[str],
    ) -> tuple[ToolMessage, ...]: ...

    async def search_docs(self, arguments: dict[str, object]) -> object: ...


class _MessagesExample(Protocol):
    def serialize_messages(self) -> str: ...


def _is_streaming_example(value: object) -> TypeIs[_StreamingExample]:
    return callable(getattr(value, "print_stream_item", None))


def _is_tool_forms_example(value: object) -> TypeIs[_ToolFormsExample]:
    transfer_args = getattr(value, "TransferArgs", None)
    return (
        isinstance(transfer_args, type)
        and issubclass(transfer_args, BaseModel)
        and callable(getattr(value, "dispatch_with_approval", None))
        and callable(getattr(value, "search_docs", None))
    )


def _is_messages_example(value: object) -> TypeIs[_MessagesExample]:
    return callable(getattr(value, "serialize_messages", None))


def test_stream_item_routing(capsys: pytest.CaptureFixture[str]) -> None:
    """Route each `StreamItem` variant through its printed representation."""
    example_module = importlib.import_module("examples.03_streaming")
    assert _is_streaming_example(example_module)

    stream_items: tuple[StreamItem, ...] = (
        "answer",
        ReasoningDelta(text="because"),
        ToolCallDelta(
            id="call_1",
            name="get_weather",
            partial_args_json='{"city":',
        ),
        ToolCall(id="call_1", name="get_weather", args_json='{"city":"Oslo"}'),
    )
    for stream_item in stream_items:
        example_module.print_stream_item(stream_item)

    assert capsys.readouterr().out == (
        "answerreasoning: because\n"
        'get_weather[call_1] arguments: {"city":\n'
        'completed call: get_weather({"city":"Oslo"})\n'
    )


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


def test_tool_forms_type_guard_rejects_unrelated_classes() -> None:
    """Only BaseModel subclasses satisfy the TransferArgs protocol attribute."""
    invalid_module = SimpleNamespace(
        TransferArgs=int,
        dispatch_with_approval=print,
        search_docs=print,
    )
    assert not _is_tool_forms_example(invalid_module)


def test_tool_arguments_reject_infinite_transfers_and_missing_queries() -> None:
    """Both invalid inputs fail before producing outputs."""
    example_module = importlib.import_module("examples.10_tool_forms_and_approval")
    assert _is_tool_forms_example(example_module)

    with pytest.raises(ValidationError):
        _ = example_module.TransferArgs(
            transfer_id="bank_infinite",
            recipient="Ada",
            amount_in_usd=float("inf"),
        )
    with pytest.raises(KeyError):
        _ = asyncio.run(example_module.search_docs({}))


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


def test_multimodal_messages_round_trip() -> None:
    """Restore every multimodal part serialized by the messages example."""
    example_module = importlib.import_module("examples.12_messages")
    assert _is_messages_example(example_module)

    restored_messages: list[Message] = messages_from_json(example_module.serialize_messages())

    assert len(restored_messages) == 1
    message = restored_messages[0]
    assert message.kind == "user"
    assert not isinstance(message.content, str)
    assert [part.kind for part in message.content] == ["text", "image", "image_url", "audio"]
