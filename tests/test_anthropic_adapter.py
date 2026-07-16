"""Anthropic Messages adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot: usage partition arithmetic, the 5-minute/1-hour cache-write cost split,
stop-reason mapping, tool_use extraction, cache-breakpoint placement, tool-choice translation,
and the precomputed request the binding determines.
"""

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Sequence
from typing import override

import anthropic
import anthropic.types as at
import httpx
import pytest
from anthropic import AsyncAnthropic
from anthropic.lib.streaming import (
    AsyncMessageStream,
    ParsedContentBlockStopEvent,
    ParsedMessageStreamEvent,
)
from anthropic.types import MessageParam, ParsedMessage
from anthropic.types.parsed_message import ParsedTextBlock
from pydantic import BaseModel

from langchaint import (
    AbortBatchError,
    AssistantMessage,
    ExceededMaxCompletionTokensError,
    ImagePart,
    InferenceParams,
    PricingTable,
    ReasoningTrace,
    RefusalError,
    SpecificTool,
    StreamItem,
    TextPart,
    Tool,
    ToolCall,
    ToolManager,
    ToolMessage,
    UserMessage,
)
from langchaint.anthropic import AnthropicMessagesProvider
from langchaint.anthropic.messages_provider import (
    _AnthropicStream,
    _assistant_content_blocks,
    _assistant_message_from,
    _BoundAnthropicStructured,
    _cost_in_usd,
    _normalized_stop_reason,
    _normalized_usage,
    _provider_result,
    _user_content_blocks,
    _wire_messages,
    _wire_tool_choice,
)
from langchaint.exceptions import StreamProtocolError, TransientError
from langchaint.provider import Binding
from langchaint.tools import ToolSchema

_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_usd_per_million_tokens=3.75,
    cache_write_1h_usd_per_million_tokens=6.0,
)
_PRICING_NO_1H = PricingTable(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_usd_per_million_tokens=3.75,
)


def _as_dict(value: object) -> dict[str, object]:
    """View one wire TypedDict as a plain dict for structural assertions."""
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _content_blocks(message: MessageParam) -> list[dict[str, object]]:
    """Return one wire message's content blocks as plain dicts."""
    content = _as_dict(message)["content"]
    assert isinstance(content, list)
    return [_as_dict(block) for block in content]


def _block_list(value: object) -> list[dict[str, object]]:
    """View a wire block list (never the omit sentinel here) as plain dicts."""
    assert isinstance(value, list)
    return [_as_dict(block) for block in value]


class _EchoArgs(BaseModel):
    """Argument model for the test tool."""

    city: str


def _tool_schemas() -> tuple[ToolSchema, ...]:
    """Return the schemas of one tool named get_weather."""

    async def function(args: _EchoArgs) -> str:
        """Return the city unchanged; never called in these tests."""
        return args.city

    tool = Tool(
        name="get_weather",
        description="Look up the weather",
        args_model=_EchoArgs,
        function=function,
    )
    return ToolManager([tool]).schemas()


def _usage_with_cache_split() -> at.Usage:
    """Return a usage object exercising every input counter and the write split."""
    return at.Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=30,
        cache_creation=at.CacheCreation(
            ephemeral_5m_input_tokens=10, ephemeral_1h_input_tokens=20
        ),
    )


def test_normalized_usage_partitions_input_counters() -> None:
    """input_tokens is the uncached counter and no all-inclusive total exists."""
    usage = _normalized_usage(_usage_with_cache_split())
    assert usage.input_tokens_cache_read == 200
    assert usage.input_tokens_cache_write == 30
    assert usage.input_tokens_cache_none == 100
    assert usage.input_tokens_total == 330
    assert usage.input_tokens_total_provider_reported is None


def test_normalized_usage_treats_none_cache_counts_as_zero() -> None:
    """Absent cache counters normalize to zero, not None."""
    usage = _normalized_usage(at.Usage(input_tokens=7, output_tokens=3))
    assert usage.input_tokens_cache_read == 0
    assert usage.input_tokens_cache_write == 0
    assert usage.input_tokens_cache_none == 7


def test_cost_splits_five_minute_and_one_hour_cache_writes() -> None:
    """The two cache-write tiers bill at their own rates from cache_creation."""
    cost = _cost_in_usd(_usage_with_cache_split(), _PRICING)
    expected = (100 * 3.0 + 200 * 0.3 + 10 * 3.75 + 20 * 6.0 + 50 * 15.0) / 1e6
    assert abs(cost - expected) < 1e-12


def test_cost_without_cache_creation_prices_all_writes_at_five_minute_rate() -> None:
    """With cache_creation absent, cache_creation_input_tokens bills as 5-minute writes."""
    usage = at.Usage(
        input_tokens=100,
        output_tokens=0,
        cache_creation_input_tokens=40,
    )
    cost = _cost_in_usd(usage, _PRICING)
    expected = (100 * 3.0 + 40 * 3.75) / 1e6
    assert abs(cost - expected) < 1e-12


def test_cost_raises_abort_when_one_hour_writes_lack_a_rate() -> None:
    """A 1-hour write with no cache_write_1h rate is a configuration defect."""
    with pytest.raises(AbortBatchError):
        _cost_in_usd(_usage_with_cache_split(), _PRICING_NO_1H)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("pause_turn", "other"),
        (None, "other"),
    ],
)
def test_stop_reason_mapping(raw: str | None, expected: str) -> None:
    """Recognized stop reasons pass through; everything else becomes other."""
    assert _normalized_stop_reason(raw) == expected


def test_provider_result_extracts_text_and_tool_use() -> None:
    """Text blocks concatenate and tool_use blocks become ToolCalls with JSON args."""
    message = at.Message(
        id="msg_1",
        content=[
            at.TextBlock(type="text", text="hello "),
            at.TextBlock(type="text", text="world"),
            at.ToolUseBlock(
                type="tool_use", id="tu_1", name="get_weather", input={"city": "Nairobi"}
            ),
        ],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason="tool_use",
        type="message",
        usage=_usage_with_cache_split(),
    )
    result = _provider_result(message=message, output="hello world", pricing=_PRICING)
    assert result.output == "hello world"
    assert result.assistant_message.text == "hello world"
    tool_call = result.assistant_message.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.args_json) == {"city": "Nairobi"}
    assert result.stop_reason == "tool_use"
    assert result.raw is message


def _message_with_content(content: list[at.ContentBlock]) -> at.Message:
    """Build an SDK message carrying the given content blocks."""
    return at.Message(
        id="msg_1",
        content=content,
        model="claude-sonnet-5",
        role="assistant",
        stop_reason="tool_use",
        type="message",
        usage=at.Usage(input_tokens=1, output_tokens=1),
    )


def test_reasoning_round_trips_verbatim_in_position() -> None:
    """A thinking block round-trips verbatim and in its original position.

    Produce yields one ReasoningTrace where the thinking block sat.
    Consume re-emits the stored dict unchanged, in the same position, with one wire block per modeled block.
    """
    message = _message_with_content([
        at.ThinkingBlock(type="thinking", thinking="check first", signature="sig-1"),
        at.TextBlock(type="text", text="hello"),
        at.ToolUseBlock(type="tool_use", id="tu_1", name="get_weather", input={"city": "Nairobi"}),
    ])
    assistant_message = _assistant_message_from(message)
    assert [type(element) for element in assistant_message.turn] == [
        ReasoningTrace,
        TextPart,
        ToolCall,
    ]
    reasoning_trace = assistant_message.turn[0]
    assert isinstance(reasoning_trace, ReasoningTrace)
    assert reasoning_trace.reasoning == {
        "type": "thinking",
        "thinking": "check first",
        "signature": "sig-1",
    }
    assert assistant_message.text == "hello"
    assert assistant_message.tool_calls == (
        ToolCall(id="tu_1", name="get_weather", args_json='{"city": "Nairobi"}'),
    )
    blocks = _assistant_content_blocks(assistant_message)
    assert len(blocks) == len(message.content)
    assert blocks[0] == reasoning_trace.reasoning
    assert blocks[1] == {"type": "text", "text": "hello"}
    assert blocks[2] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "get_weather",
        "input": {"city": "Nairobi"},
    }


def test_redacted_thinking_round_trips_routed_by_its_type_key() -> None:
    """A redacted_thinking block round-trips as its own dump; the type key routes it on the wire."""
    message = _message_with_content([
        at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes")
    ])
    assistant_message = _assistant_message_from(message)
    assert _assistant_content_blocks(assistant_message) == [
        {"type": "redacted_thinking", "data": "opaque-bytes"}
    ]


def test_reasoning_with_a_key_the_installed_sdk_lacks_survives_the_wire_builder() -> None:
    """A stored dict carrying a field newer than the installed SDK param re-emits unchanged.

    A consume step that reshaped the dict to the pinned param keys would modify the thinking block
    across an SDK upgrade, which the API rejects on a tool-use continuation.
    """
    reasoning = {
        "type": "thinking",
        "thinking": "t",
        "signature": "s",
        "field_newer_than_sdk": "x",
    }
    assistant_message = AssistantMessage(turn=(ReasoningTrace(reasoning=reasoning),))
    assert _assistant_content_blocks(assistant_message) == [reasoning]


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """An openai-produced trace emits its dict as-is; the API rejects the unknown type key, not this adapter."""
    reasoning = {"type": "reasoning", "id": "rs_1"}
    assistant_message = AssistantMessage(
        turn=(ReasoningTrace(reasoning=reasoning), TextPart(text="hi"))
    )
    assert _assistant_content_blocks(assistant_message) == [
        reasoning,
        {"type": "text", "text": "hi"},
    ]


def test_wire_messages_groups_consecutive_tool_results() -> None:
    """Consecutive ToolMessages collapse into one user message of tool_result blocks."""
    conversation = [
        UserMessage(content="hi"),
        AssistantMessage(
            turn=(
                TextPart(text="checking"),
                ToolCall(id="tu_1", name="t", args_json='{"a": 1}'),
            ),
        ),
        ToolMessage(tool_call_id="tu_1", content="r1", is_error=False),
        ToolMessage(tool_call_id="tu_2", content="r2", is_error=True),
    ]
    wire = _wire_messages(conversation, automatic_prompt_caching=False)
    assert [message["role"] for message in wire] == ["user", "assistant", "user"]
    tool_results = _content_blocks(wire[2])
    assert len(tool_results) == 2
    assert tool_results[0]["is_error"] is False
    assert tool_results[1]["is_error"] is True


def test_wire_messages_marks_only_the_last_block_when_caching() -> None:
    """The per-request breakpoint lands on the last block of the last message."""
    conversation = [
        ToolMessage(tool_call_id="tu_1", content="r1", is_error=False),
        ToolMessage(tool_call_id="tu_2", content="r2", is_error=True),
    ]
    wire = _wire_messages(conversation, automatic_prompt_caching=True)
    tool_results = _content_blocks(wire[0])
    assert tool_results[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tool_results[0]


def test_wire_messages_writes_no_breakpoint_on_a_thinking_last_block() -> None:
    """A conversation ending on a thinking block writes no breakpoint that request.

    The thinking wire params carry no cache_control key, so the marker has nowhere valid to go.
    """
    conversation = [
        AssistantMessage(
            turn=(
                TextPart(text="t"),
                ReasoningTrace(
                    provider_name="anthropic_messages",
                    reasoning={"type": "thinking", "thinking": "x", "signature": "s"},
                ),
            )
        )
    ]
    wire = _wire_messages(conversation, automatic_prompt_caching=True)
    assert all("cache_control" not in block for block in _content_blocks(wire[0]))


def test_wire_messages_writes_no_breakpoint_when_caching_disabled() -> None:
    """With caching off, no block anywhere carries a cache_control marker."""
    conversation = [
        UserMessage(content="hi"),
        ToolMessage(tool_call_id="tu_1", content="r1"),
    ]
    wire = _wire_messages(conversation, automatic_prompt_caching=False)
    assert all(
        "cache_control" not in block
        for message in wire
        for block in _content_blocks(message)
    )


def test_wire_messages_converts_tool_result_parts_to_text_and_image_blocks() -> None:
    """A ToolMessage carrying parts becomes a tool_result whose content is the text and image blocks.

    A dropped part or a mis-encoded image would change this exact block list.
    """
    conversation = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="saw"), ImagePart(data=b"png", media_type="image/png")),
        )
    ]
    wire = _wire_messages(conversation, automatic_prompt_caching=False)
    tool_result = _content_blocks(wire[0])[0]
    assert tool_result["content"] == [
        {"type": "text", "text": "saw"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(b"png").decode("ascii"),
            },
        },
    ]


def test_wire_messages_rejects_tool_result_image_with_unsupported_media_type() -> None:
    """A tool_result image media type outside the accepted set aborts the batch as a request defect."""
    conversation = [
        ToolMessage(
            tool_call_id="tu_1", content=(ImagePart(data=b"x", media_type="image/tiff"),)
        )
    ]
    with pytest.raises(AbortBatchError):
        _wire_messages(conversation, automatic_prompt_caching=False)


def test_wire_tool_choice_required_becomes_any_and_inverts_parallel() -> None:
    """Neutral required maps to any; disable_parallel_tool_use is the inverse."""
    for parallel in (True, False):
        assert _wire_tool_choice("required", parallel_tool_calls=parallel) == {
            "type": "any",
            "disable_parallel_tool_use": not parallel,
        }


def test_wire_tool_choice_specific_tool_names_the_tool() -> None:
    """A SpecificTool becomes the named-tool form."""
    assert _wire_tool_choice(SpecificTool(tool_name="x"), parallel_tool_calls=True) == {
        "type": "tool",
        "name": "x",
        "disable_parallel_tool_use": False,
    }


def test_wire_tool_choice_none_forbids_calls() -> None:
    """Neutral none maps to the none form with no parallel flag."""
    assert _wire_tool_choice("none", parallel_tool_calls=True) == {"type": "none"}


def _provider() -> AnthropicMessagesProvider:
    """Build an adapter over a keyless client, valid because no request is sent."""
    return AnthropicMessagesProvider(
        client=AsyncAnthropic(api_key="test"), model="m", pricing=_PRICING
    )


def _binding(
    *,
    system_prompt: str | None,
    tool_schemas: tuple[ToolSchema, ...],
    automatic_prompt_caching: bool,
) -> Binding:
    """Assemble a binding with the fields these request tests vary."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        tool_choice="required",
        parallel_tool_calls=False,
        inference_params=InferenceParams(reasoning_effort="high"),
        automatic_prompt_caching=automatic_prompt_caching,
    )


def test_request_omits_tool_sentinels_without_tools() -> None:
    """No tools leaves both tools and tool_choice at the omit sentinel."""
    request = _provider()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert request.max_tokens == 4096
    assert isinstance(request.tools, anthropic.Omit)
    assert isinstance(request.tool_choice, anthropic.Omit)
    assert request.output_config == {"effort": "high"}


def test_request_marks_the_system_block_only_when_caching() -> None:
    """The system block carries a breakpoint under caching and none without it."""
    cached = _provider()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert _block_list(cached.system)[0]["cache_control"] == {"type": "ephemeral"}
    uncached = _provider()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    assert "cache_control" not in _block_list(uncached.system)[0]


def test_request_marks_last_tool_only_without_a_system_prompt() -> None:
    """The prefix breakpoint sits on the last tool only when no system prompt follows."""
    schemas = _tool_schemas()
    without_system = _provider()._request(
        _binding(system_prompt=None, tool_schemas=schemas, automatic_prompt_caching=True)
    )
    assert _block_list(without_system.tools)[-1]["cache_control"] == {"type": "ephemeral"}
    with_system = _provider()._request(
        _binding(system_prompt="sys", tool_schemas=schemas, automatic_prompt_caching=True)
    )
    assert "cache_control" not in _block_list(with_system.tools)[-1]


def test_user_content_blocks_rejects_unsupported_image_media_type() -> None:
    """An image media type outside the accepted set aborts the batch as a request defect."""
    message = UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),))
    with pytest.raises(AbortBatchError):
        _user_content_blocks(message)


class _FakeSDKMessageStream(AsyncMessageStream[None]):
    """Replays constructed events without a connection.

    Overrides exactly the surface _AnthropicStream uses (iteration, close,
    and the current_message_snapshot the stop-reason check reads); the base __init__ is deliberately not called,
    so the untouched base machinery stays unusable.
    """

    def __init__(
        self,
        replay_events: Sequence[ParsedMessageStreamEvent[None]],
        message_snapshot: ParsedMessage[None],
    ) -> None:
        self._replay_events = list(replay_events)
        self._message_snapshot = message_snapshot

    @override
    async def __aiter__(self) -> AsyncIterator[ParsedMessageStreamEvent[None]]:
        for replay_event in self._replay_events:
            yield replay_event

    @override
    async def close(self) -> None:
        return

    @property
    @override
    def current_message_snapshot(self) -> ParsedMessage[None]:
        return self._message_snapshot

    @override
    async def get_final_message(self) -> ParsedMessage[None]:
        return self._message_snapshot


def _message_snapshot(
    stop_reason: at.StopReason | None, content: list[at.ContentBlock] | None = None
) -> ParsedMessage[None]:
    """Build the accumulated message the SDK stream would hold after draining."""
    message = at.Message(
        id="msg_1",
        content=content if content is not None else [],
        model="claude-sonnet-4-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=at.Usage(input_tokens=1, output_tokens=1),
    )
    return ParsedMessage[None].model_validate(message.model_dump())


def _anthropic_stream(
    replay_events: Sequence[ParsedMessageStreamEvent[None]],
    message_snapshot: ParsedMessage[None],
) -> _AnthropicStream[str]:
    """Build a text-content adapter stream over replayed events."""
    return _AnthropicStream(
        sdk_stream=_FakeSDKMessageStream(replay_events, message_snapshot),
        pricing=_PRICING,
        output_from_message=lambda _message: "",
    )


def _text_delta_event(text: str, index: int) -> at.RawContentBlockDeltaEvent:
    """Build one raw text-delta event."""
    return at.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=at.TextDelta(type="text_delta", text=text),
    )


def test_stream_yields_bare_text_and_one_complete_tool_call() -> None:
    """Text deltas pass through as the SDK's own strings.

    A closing tool_use block yields one complete ToolCall whose args_json is the JSON text of the SDK-accumulated input;
    argument fragments and text block closes yield nothing.
    """
    args_fragment = at.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=1,
        delta=at.InputJSONDelta(type="input_json_delta", partial_json='{"city"'),
    )
    text_block_stop = ParsedContentBlockStopEvent(
        type="content_block_stop",
        index=0,
        content_block=ParsedTextBlock(type="text", text="hey"),
    )
    tool_use_block_stop = ParsedContentBlockStopEvent(
        type="content_block_stop",
        index=1,
        content_block=at.ToolUseBlock(
            type="tool_use", id="tu_1", name="get_weather", input={"city": "Nairobi"}
        ),
    )

    async def scenario() -> list[StreamItem]:
        adapter_stream = _anthropic_stream(
            [
                _text_delta_event("he", 0),
                _text_delta_event("y", 0),
                text_block_stop,
                args_fragment,
                tool_use_block_stop,
            ],
            _message_snapshot("tool_use"),
        )
        return [item async for item in adapter_stream.items()]

    assert asyncio.run(scenario()) == [
        "he",
        "y",
        ToolCall(id="tu_1", name="get_weather", args_json='{"city": "Nairobi"}'),
    ]


def test_stream_final_turn_carries_reasoning() -> None:
    """final()'s assistant turn includes the thinking block from the SDK-assembled message."""

    async def scenario() -> None:
        snapshot = _message_snapshot(
            "end_turn",
            content=[
                at.ThinkingBlock(type="thinking", thinking="check", signature="sig-1"),
                at.TextBlock(type="text", text="hey"),
            ],
        )
        adapter_stream = _anthropic_stream([], snapshot)
        async for _item in adapter_stream.items():
            pass
        result = await adapter_stream.final()
        reasoning_trace = result.assistant_message.turn[0]
        assert isinstance(reasoning_trace, ReasoningTrace)
        assert reasoning_trace.reasoning == {
            "type": "thinking",
            "thinking": "check",
            "signature": "sig-1",
        }

    asyncio.run(scenario())


def test_stream_without_stop_reason_raises() -> None:
    """Ending with no stop reason on the accumulated message is a protocol violation."""

    async def scenario() -> None:
        adapter_stream = _anthropic_stream(
            [_text_delta_event("he", 0)], _message_snapshot(None)
        )
        with pytest.raises(StreamProtocolError):
            async for _item in adapter_stream.items():
                pass

    asyncio.run(scenario())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundAnthropicStructured[_StructuredReport]:
    """Build a structured-bound provider over a keyless client; no request is sent."""
    provider = _provider()
    request = provider._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    return _BoundAnthropicStructured(
        adapter=provider, request=request, response_format=_StructuredReport
    )


def _parsed_message(
    parsed_output: _StructuredReport | None,
    stop_reason: at.StopReason = "end_turn",
) -> ParsedMessage[_StructuredReport]:
    """Build the SDK parse result whose first text block carries the given parsed output."""
    return ParsedMessage[_StructuredReport](
        id="msg_1",
        content=[
            ParsedTextBlock[_StructuredReport](
                type="text", text="{}", parsed_output=parsed_output
            )
        ],
        model="claude-sonnet-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=at.Usage(input_tokens=1, output_tokens=1),
    )


def test_structured_bind_returns_the_sdk_parsed_instance() -> None:
    """The structured bound provider hands back the SDK-parsed response_format instance as content."""
    report = _StructuredReport(city="Nairobi", celsius=25)
    assert _structured_bound()._parsed_output(_parsed_message(report)) == report


def test_structured_bind_raises_transient_without_parsed_output() -> None:
    """A turn with no parsed output is transient: a later attempt may still produce it."""
    with pytest.raises(TransientError) as raised:
        _structured_bound()._parsed_output(_parsed_message(None))
    # The rejected 200's billing rides on the transient error so the retry record is not zero.
    assert raised.value.cost_in_usd is not None
    assert raised.value.stop_reason == "end_turn"


def test_structured_bind_raises_refusal_on_a_refusal_stop_reason() -> None:
    """A refusal stop_reason with no parsed output is the terminal refusal leaf, carrying its billing."""
    with pytest.raises(RefusalError) as raised:
        _structured_bound()._parsed_output(_parsed_message(None, stop_reason="refusal"))
    assert raised.value.stop_reason == "refusal"
    assert raised.value.cost_in_usd > 0.0


def test_structured_bind_raises_truncation_on_a_max_tokens_stop_reason() -> None:
    """A max_tokens stop_reason with no parsed output is the terminal truncation leaf."""
    with pytest.raises(ExceededMaxCompletionTokensError) as raised:
        _structured_bound()._parsed_output(_parsed_message(None, stop_reason="max_tokens"))
    assert raised.value.stop_reason == "max_tokens"
    assert raised.value.cost_in_usd > 0.0


def _rate_limit_error(headers: dict[str, str]) -> anthropic.RateLimitError:
    """Build the SDK's 429 exception around a constructed httpx response."""
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def test_retry_after_seconds_prefers_the_millisecond_header() -> None:
    """retry-after-ms wins over retry-after because it is more precise."""
    error = _rate_limit_error({"retry-after-ms": "1500", "retry-after": "49"})
    assert _provider().retry_after_seconds(error) == 1.5


def test_retry_after_seconds_parses_the_seconds_header() -> None:
    """Without retry-after-ms, retry-after is parsed as float seconds."""
    error = _rate_limit_error({"retry-after": "49"})
    assert _provider().retry_after_seconds(error) == 49.0


def test_retry_after_seconds_is_none_without_headers_or_status() -> None:
    """No headers, an unparseable value, and a non-SDK error all yield None."""
    provider = _provider()
    assert provider.retry_after_seconds(_rate_limit_error({})) is None
    assert (
        provider.retry_after_seconds(
            _rate_limit_error({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        )
        is None
    )
    assert provider.retry_after_seconds(ValueError("boom")) is None


def test_adapter_pins_sdk_retries_off() -> None:
    """The stored client copy carries max_retries=0 so only the package retries."""
    assert _provider().client.max_retries == 0
