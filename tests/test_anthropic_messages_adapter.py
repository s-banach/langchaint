"""Test Anthropic Messages adapters with constructed SDK objects."""

import asyncio
import base64
import inspect
import json
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TypeIs, get_args, override

import anthropic
import anthropic.types as at
import httpx2
import pytest
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    transform_schema,
)
from anthropic.lib.streaming import (
    AsyncMessageStream,
    ParsedContentBlockStopEvent,
    ParsedMessageStreamEvent,
)
from anthropic.types import ContentBlockParam, MessageParam, ParsedMessage
from anthropic.types.parsed_message import ParsedTextBlock
from pydantic import BaseModel, TypeAdapter

from langchaint import (
    LLM,
    AllowedToolsChoice,
    AssistantMessage,
    AudioPart,
    Billing,
    ImagePart,
    ImageUrlPart,
    InferenceParams,
    Message,
    PydanticTool,
    RawPart,
    ReasoningDelta,
    ReasoningPart,
    SpecificToolChoice,
    StreamItem,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolManager,
    ToolMessage,
    UserMessage,
)
from langchaint.adapter import (
    REASONING_PART_SEPARATOR,
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    ContextWindowExceeded,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutput,
    NoOutputOutcome,
    ProviderBilling,
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.anthropic import (
    ANTHROPIC_BEDROCK,
    ANTHROPIC_BEDROCK_PRICING,
    Anthropic,
    AnthropicBedrock,
    AnthropicBedrockModelName,
    AnthropicMessagesAdapter,
    AnthropicPricingTable,
    AnthropicRates,
)
from langchaint.anthropic.messages_adapter import (
    _NO_ANTHROPIC_PROVIDER_TOOLS,
    PARSE_FALLTHROUGH_COUNTS,
    _adapter_result,
    _AnthropicProviderTools,
    _AnthropicRequestParams,
    _AnthropicStream,
    _assistant_content_blocks,
    _assistant_message_from,
    _BoundAnthropic,
    _BoundAnthropicStructured,
    _BoundAnthropicText,
    _normalized_stop_reason,
    _NotSendableError,
    _user_content_blocks,
    _wire_messages,
    _wire_tool_choice,
    parse_anthropic,
)
from langchaint.anthropic.messages_adapter import (
    _billing_from_sdk_usage as _provider_billing_from_sdk_usage,
)
from langchaint.call import ResponseIdentity
from langchaint.conformance import AdapterConformance
from langchaint.exceptions import TransientError
from langchaint.shared_backoff import (
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    RetryThisOne,
    Verdict,
)
from langchaint.tools import ToolSchema


def _billing_from_sdk_usage(
    usage: at.Usage,
    pricing: AnthropicPricingTable,
    *,
    provider_tools: _AnthropicProviderTools = _NO_ANTHROPIC_PROVIDER_TOOLS,
    billing_complete: bool = True,
) -> Billing:
    return _provider_billing_from_sdk_usage(
        usage,
        pricing,
        provider_tools=provider_tools,
        billing_complete=billing_complete,
    ).billing


_STANDARD_RATES = AnthropicRates(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_5m_usd_per_million_tokens=3.75,
    cache_write_1h_usd_per_million_tokens=6.0,
)

_PRICING = AnthropicPricingTable(
    standard=_STANDARD_RATES,
    web_search_usd_per_invocation=0.01,
)
"""The standard tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = AnthropicRates(
    input_cache_none_usd_per_million_tokens=6.0,
    output_usd_per_million_tokens=30.0,
    cache_read_usd_per_million_tokens=0.6,
    cache_write_5m_usd_per_million_tokens=7.5,
    cache_write_1h_usd_per_million_tokens=12.0,
)
"""Twice the standard rates, so a tier-selection test reads as a doubling."""


def _content_blocks(message: MessageParam) -> list[ContentBlockParam]:
    """Return one wire message's content blocks."""
    content = message["content"]
    assert isinstance(content, list)
    blocks: list[ContentBlockParam] = []
    for block in content:
        assert _is_content_block_param(block)
        blocks.append(block)
    return blocks


def _is_content_block_param(value: object) -> TypeIs[ContentBlockParam]:
    """Distinguish request TypedDicts from response models."""
    return isinstance(value, dict)


def _block_list[BlockT](value: list[BlockT] | anthropic.Omit) -> list[BlockT]:
    """Return populated Anthropic blocks."""
    assert isinstance(value, list)
    return value


class _EchoArgs(BaseModel):
    """Argument model for the test tool."""

    city: str


def _tool_schemas() -> tuple[ToolSchema, ...]:
    """Return the schemas of one tool named get_weather."""

    async def function(args: _EchoArgs) -> str:
        """Return the city unchanged. Never called in these tests."""
        return args.city

    tool = PydanticTool(
        name="get_weather",
        description="Look up the weather",
        args_model=_EchoArgs,
        function=function,
    )
    return ToolManager([tool]).schemas()


def _usage_with_cache_split() -> at.Usage:
    return at.Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=200,
        cache_creation=at.CacheCreation(
            ephemeral_5m_input_tokens=10, ephemeral_1h_input_tokens=20
        ),
    )


def test_billing_partitions_and_prices_complete_usage() -> None:
    """Expose one complete SDK usage object's neutral counters, costs, tier, and applied rates."""
    usage_raw = _usage_with_cache_split()
    billing = _billing_from_sdk_usage(usage_raw, _PRICING)
    usage = billing.usage
    assert _provider_billing_from_sdk_usage(usage_raw, _PRICING).usage_raw is usage_raw
    assert billing.service_tier == "standard"
    assert billing.input_cache_none_usd_per_million_tokens == 3.0
    assert billing.cache_read_usd_per_million_tokens == 0.3
    assert billing.output_usd_per_million_tokens == 15.0
    assert usage.input_tokens_cache_read == 200
    assert usage.input_tokens_cache_write == 30
    assert usage.input_tokens_cache_none == 100
    assert usage.input_tokens_cache_none_cost_in_usd == 100 * 3.0 / 1e6
    assert usage.input_tokens_cache_read_cost_in_usd == 200 * 0.3 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == (10 * 3.75 + 20 * 6.0) / 1e6
    assert usage.output_tokens_cost_in_usd == 50 * 15.0 / 1e6


def test_an_unpriced_tier_keeps_its_counters_and_its_name() -> None:
    """A response served at a tier no table prices still reports what it billed and who served it."""
    billing = _billing_from_sdk_usage(
        at.Usage(input_tokens=100, output_tokens=50, service_tier="priority"), _PRICING
    )
    assert billing.service_tier == "priority"
    assert billing.usage.input_tokens_total == 100
    assert billing.usage.output_tokens == 50


def test_billing_treats_none_cache_counts_as_zero() -> None:
    """Absent cache counters normalize to zero, not None."""
    usage = _billing_from_sdk_usage(at.Usage(input_tokens=7, output_tokens=3), _PRICING).usage
    assert usage.input_tokens_cache_read == 0
    assert usage.input_tokens_cache_write == 0
    assert usage.input_tokens_cache_none == 7


def test_web_search_requests_add_the_cataloged_provider_executed_tool_cost() -> None:
    """Anthropic's raw invocation count prices web search exactly."""
    raw = at.Usage(
        input_tokens=7,
        output_tokens=3,
        server_tool_use=at.ServerToolUsage(web_search_requests=2, web_fetch_requests=0),
    )
    usage = _billing_from_sdk_usage(raw, _PRICING).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(0.02)
    assert usage.cost_in_usd > usage.provider_executed_tool_cost_in_usd


def test_anthropic_zero_fee_server_tools_preserve_zero_cost() -> None:
    """Web fetch, tool search, and exempt code execution add no separate fee."""
    provider_tools = (
        _adapter()
        ._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=(
                    {"type": "web_fetch_20260209"},
                    {"type": "tool_search_tool_bm25"},
                    {"type": "code_execution_20260120"},
                ),
                automatic_cache_breakpoints=False,
            )
        )
        .provider_tools
    )
    server_tool_use = at.ServerToolUsage.model_validate({
        "web_search_requests": 0,
        "web_fetch_requests": 2,
        "tool_search_requests": 3,
        "code_execution_requests": 4,
    })
    usage_raw = at.Usage(
        input_tokens=1,
        output_tokens=1,
        server_tool_use=server_tool_use,
    )
    usage = _billing_from_sdk_usage(usage_raw, _PRICING, provider_tools=provider_tools).usage
    assert usage.provider_executed_tool_cost_in_usd == 0.0


@pytest.mark.parametrize(
    "counter_name",
    ["web_fetch_requests", "tool_search_requests", "code_execution_requests"],
)
def test_unconfigured_anthropic_server_counter_produces_nan(counter_name: str) -> None:
    """A nonzero unconfigured server-tool counter is unexpected billing evidence."""
    server_tool_counters = {
        "web_search_requests": 0,
        "web_fetch_requests": 0,
    }
    server_tool_counters[counter_name] = 1
    server_tool_use = at.ServerToolUsage.model_validate(server_tool_counters)
    usage_raw = at.Usage(
        input_tokens=1,
        output_tokens=1,
        server_tool_use=server_tool_use,
    )
    cost = _billing_from_sdk_usage(usage_raw, _PRICING).usage.provider_executed_tool_cost_in_usd
    assert math.isnan(cost)


@pytest.mark.parametrize("unexpected_count", [0, 2])
def test_unexpected_anthropic_server_counter_controls_nan(unexpected_count: int) -> None:
    """Only a nonzero unexpected request counter proves an unpriced charge."""
    server_tool_use = at.ServerToolUsage.model_validate({
        "web_search_requests": 0,
        "web_fetch_requests": 0,
        "future_requests": unexpected_count,
    })
    usage_raw = at.Usage(
        input_tokens=1,
        output_tokens=1,
        server_tool_use=server_tool_use,
    )
    cost = _billing_from_sdk_usage(usage_raw, _PRICING).usage.provider_executed_tool_cost_in_usd
    if unexpected_count:
        assert math.isnan(cost)
    else:
        assert cost == 0.0


def test_truncated_anthropic_web_search_billing_produces_nan() -> None:
    """A partial usage snapshot cannot prove the final web-search count."""
    provider_tools = (
        _adapter()
        ._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=({"type": "web_search_20260318"},),
                automatic_cache_breakpoints=False,
            )
        )
        .provider_tools
    )
    usage_raw = at.Usage(input_tokens=1, output_tokens=1)
    usage = _billing_from_sdk_usage(
        usage_raw,
        _PRICING,
        provider_tools=provider_tools,
        billing_complete=False,
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


@pytest.mark.parametrize("rate", [None, True, math.nan, math.inf, -0.01])
def test_configured_anthropic_web_search_rate_must_be_usable(rate: float | None) -> None:
    """A configured search rejects an unusable caller rate before requests."""
    pricing = AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=3.0,
            output_usd_per_million_tokens=15.0,
            cache_read_usd_per_million_tokens=0.3,
            cache_write_5m_usd_per_million_tokens=3.75,
            cache_write_1h_usd_per_million_tokens=6.0,
        ),
        web_search_usd_per_invocation=rate,
    )
    adapter = AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=pricing,
        provider_name="anthropic",
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _ = adapter._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=({"type": "web_search_20250305"},),
                automatic_cache_breakpoints=False,
            )
        )


def test_billing_reads_reasoning_tokens_and_defaults_to_zero() -> None:
    """output_tokens_reasoning reads thinking_tokens, and is zero when output_tokens_details is absent."""
    with_details = _billing_from_sdk_usage(
        at.Usage(
            input_tokens=1,
            output_tokens=9,
            output_tokens_details=at.OutputTokensDetails(thinking_tokens=4),
        ),
        _PRICING,
    ).usage
    assert with_details.output_tokens_reasoning == 4
    without_details = _billing_from_sdk_usage(
        at.Usage(input_tokens=1, output_tokens=9), _PRICING
    ).usage
    assert without_details.output_tokens_reasoning == 0


def test_cost_without_cache_creation_prices_all_writes_at_five_minute_rate() -> None:
    """With cache_creation absent, cache_creation_input_tokens bills as 5-minute writes."""
    usage = at.Usage(
        input_tokens=100,
        output_tokens=0,
        cache_creation_input_tokens=40,
    )
    cost = _billing_from_sdk_usage(usage, _PRICING).usage.cost_in_usd
    expected = (100 * 3.0 + 40 * 3.75) / 1e6
    assert abs(cost - expected) < 1e-12


def test_equal_write_rates_store_that_rate_as_the_write_price() -> None:
    """With both TTLs priced alike the blend is that rate, so blending adds no artifact."""
    equal_write_rates = AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=3.0,
            output_usd_per_million_tokens=15.0,
            cache_read_usd_per_million_tokens=0.3,
            cache_write_5m_usd_per_million_tokens=3.75,
            cache_write_1h_usd_per_million_tokens=3.75,
        ),
        web_search_usd_per_invocation=0.01,
    )
    billing = _billing_from_sdk_usage(_usage_with_cache_split(), equal_write_rates)
    assert billing.cache_write_usd_per_million_tokens == pytest.approx(3.75)


def test_a_response_that_wrote_no_cache_stores_the_five_minute_write_rate() -> None:
    """With nothing written there is nothing to blend, so the write price is the default TTL's rate."""
    billing = _billing_from_sdk_usage(at.Usage(input_tokens=7, output_tokens=3), _PRICING)
    assert billing.usage.input_tokens_cache_write == 0
    assert billing.cache_write_usd_per_million_tokens == 3.75


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response, standard rates a response reporting no tier."""
    pricing = AnthropicPricingTable(
        standard=_STANDARD_RATES,
        priority=_PRIORITY_RATES,
    )
    at_priority = _billing_from_sdk_usage(
        at.Usage(input_tokens=100, output_tokens=50, service_tier="priority"), pricing
    )
    at_standard = _billing_from_sdk_usage(
        at.Usage(input_tokens=100, output_tokens=50, service_tier="standard"), pricing
    )
    reporting_none = _billing_from_sdk_usage(at.Usage(input_tokens=100, output_tokens=50), pricing)
    assert at_priority.usage.cost_in_usd == pytest.approx(2 * at_standard.usage.cost_in_usd)
    assert reporting_none.usage.cost_in_usd == at_standard.usage.cost_in_usd


def test_cache_ttl_is_stored_on_the_adapter() -> None:
    """`Anthropic.model()` and `AnthropicBedrock.model()` carry `cache_ttl`."""
    assert (
        _anthropic_adapter_of(
            Anthropic(client=AsyncAnthropic(api_key="test")).model(
                "claude-sonnet-5",
                cache_ttl="1h",
            )
        ).cache_ttl
        == "1h"
    )
    assert (
        _anthropic_adapter_of(
            AnthropicBedrock(aws_region="us-east-1").model(
                "anthropic.claude-sonnet-5",
                cache_ttl="1h",
            )
        ).cache_ttl
        == "1h"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("model_context_window_exceeded", "context_window_exceeded"),
        ("pause_turn", "other"),
        (None, "other"),
    ],
)
def test_stop_reason_mapping(raw: str | None, expected: str) -> None:
    """Map recognized stop reasons and use other as fallback."""
    assert _normalized_stop_reason(raw) == expected


def test_adapter_result_extracts_text_and_tool_use() -> None:
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
    result = _adapter_result(message, "hello world", _assistant_message_from(message))
    assert result.output == "hello world"
    assert result.assistant_message.text == "hello world"
    tool_call = result.assistant_message.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.args_json) == {"city": "Nairobi"}
    assert result.stop_reason == "tool_use"


def _message_with_content(
    content: list[at.ContentBlock],
    stop_reason: at.StopReason = "tool_use",
    usage: at.Usage | None = None,
) -> at.Message:
    """Build an SDK message carrying the given content blocks."""
    return at.Message(
        id="msg_1",
        content=content,
        model="claude-sonnet-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=usage if usage is not None else at.Usage(input_tokens=1, output_tokens=1),
    )


def test_reasoning_round_trips_verbatim_in_position() -> None:
    """A thinking block round-trips verbatim and in its original position.

    Produce yields one ReasoningPart where the thinking block sat.
    Consume re-emits the stored dict unchanged, in the same position, with one wire block per modeled block.
    """
    message = _message_with_content([
        at.ThinkingBlock(type="thinking", thinking="check first", signature="sig-1"),
        at.TextBlock(type="text", text="hello"),
        at.ToolUseBlock(type="tool_use", id="tu_1", name="get_weather", input={"city": "Nairobi"}),
    ])
    assistant_message = _assistant_message_from(message)
    assert [type(part) for part in assistant_message.turn] == [
        ReasoningPart,
        TextPart,
        ToolCall,
    ]
    reasoning_part = assistant_message.turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.raw == {
        "type": "thinking",
        "thinking": "check first",
        "signature": "sig-1",
    }
    assert reasoning_part.text == "check first"
    assert assistant_message.text == "hello"
    assert assistant_message.tool_calls == (
        ToolCall(id="tu_1", name="get_weather", args_json='{"city": "Nairobi"}'),
    )
    blocks = _assistant_content_blocks(assistant_message)
    assert len(blocks) == len(message.content)
    assert blocks[0] == reasoning_part.raw
    assert blocks[1] == {"type": "text", "text": "hello"}
    assert blocks[2] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "get_weather",
        "input": {"city": "Nairobi"},
    }


def test_empty_thinking_text_normalizes_to_none() -> None:
    """A thinking block whose text is "" yields text None, the single text-free condition.

    ReasoningPart.text uses None for missing reasoning across adapters.
    """
    message = _message_with_content([
        at.ThinkingBlock(type="thinking", thinking="", signature="sig")
    ])
    reasoning_part = _assistant_message_from(message).turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.text is None
    assert reasoning_part.raw["thinking"] == ""


def test_redacted_thinking_round_trips_routed_by_its_type_key() -> None:
    """A redacted_thinking block round-trips as its own dump. The type key routes it on the wire.

    ReasoningPart.text is None because the block has no readable text.
    """
    message = _message_with_content([
        at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes")
    ])
    assistant_message = _assistant_message_from(message)
    reasoning_part = assistant_message.turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.text is None
    assert _assistant_content_blocks(assistant_message) == [
        {"type": "redacted_thinking", "data": "opaque-bytes"}
    ]


def test_a_server_tool_block_becomes_a_raw_part_and_replays_as_itself() -> None:
    """A server tool block becomes RawPart and returns unchanged.

    The billed raw block remains in the turn for replay.
    """
    assistant_message = _assistant_message_from(
        _message_with_content([
            at.ServerToolUseBlock(
                type="server_tool_use",
                id="srvtoolu_1",
                name="web_search",
                input={"query": "langchaint"},
            )
        ])
    )
    (raw_part,) = assistant_message.turn
    assert isinstance(raw_part, RawPart)
    assert _assistant_content_blocks(assistant_message) == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "langchaint"},
        }
    ]


def test_produced_reasoning_parts_survive_the_message_json_round_trip() -> None:
    """Produced ReasoningPart values re-validate equal from JSON.

    Persistence serializes Sequence[Message] through TypeAdapter.
    A changed raw payload causes API rejection.
    The round trip must restore each ReasoningPart.raw exactly.
    """
    message = _message_with_content([
        at.ThinkingBlock(type="thinking", thinking="check first", signature="sig-1"),
        at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes"),
        at.TextBlock(type="text", text="hello"),
    ])
    messages_type_adapter: TypeAdapter[tuple[Message, ...]] = TypeAdapter(tuple[Message, ...])
    messages: tuple[Message, ...] = (_assistant_message_from(message),)
    restored = messages_type_adapter.validate_json(messages_type_adapter.dump_json(messages))
    assert restored == messages


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """A foreign ReasoningPart sends ReasoningPart.raw unchanged for provider validation."""
    raw = {"type": "reasoning", "id": "rs_1"}
    assistant_message = AssistantMessage(turn=(ReasoningPart(raw=raw), TextPart(text="hi")))
    assert _assistant_content_blocks(assistant_message) == [
        raw,
        {"type": "text", "text": "hi"},
    ]


def test_wire_messages_groups_consecutive_tool_results() -> None:
    """Consecutive ToolMessages collapse into one user message of tool_result blocks."""
    messages = [
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
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    assert [message["role"] for message in wire] == ["user", "assistant", "user"]
    tool_results = _content_blocks(wire[2])
    assert len(tool_results) == 2
    assert tool_results[0]["type"] == "tool_result"
    assert tool_results[1]["type"] == "tool_result"
    assert tool_results[0].get("is_error") is False
    assert tool_results[1].get("is_error") is True


def test_wire_messages_marks_only_the_last_block_when_caching() -> None:
    """The per-request breakpoint lands on the last block of the last message."""
    messages = [
        ToolMessage(tool_call_id="tu_1", content="r1", is_error=False),
        ToolMessage(tool_call_id="tu_2", content="r2", is_error=True),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=True, cache_ttl="5m", message_mark_budget=2
    )
    tool_results = _content_blocks(wire[0])
    assert tool_results[-1]["type"] == "tool_result"
    assert tool_results[-1].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in tool_results[0]


def test_wire_messages_writes_no_breakpoint_on_a_thinking_last_block() -> None:
    """A Sequence[Message] ending on a thinking block writes no breakpoint that request.

    The thinking wire params carry no cache_control key, so the marker has nowhere valid to go.
    """
    messages = [
        AssistantMessage(
            turn=(
                TextPart(text="t"),
                ReasoningPart(raw={"type": "thinking", "thinking": "x", "signature": "s"}),
            )
        )
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=True, cache_ttl="5m", message_mark_budget=2
    )
    assert all("cache_control" not in block for block in _content_blocks(wire[0]))


def test_wire_messages_writes_no_automatic_breakpoint_when_disabled() -> None:
    """False writes no automatic `cache_control` marker."""
    messages = [
        UserMessage(content="hi"),
        ToolMessage(tool_call_id="tu_1", content="r1"),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    assert all(
        "cache_control" not in block for message in wire for block in _content_blocks(message)
    )


def test_wire_messages_converts_tool_result_parts_to_text_and_image_blocks() -> None:
    """A ToolMessage carrying parts becomes a tool_result whose content is the text and image blocks.

    A dropped part or a mis-encoded image would change this exact block list.
    """
    messages = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="saw"), ImagePart(data=b"png", media_type="image/png")),
        )
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    tool_result = _content_blocks(wire[0])[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("content") == [
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


def test_wire_messages_sends_image_url_part_unchanged() -> None:
    """ImageUrlPart uses URLImageSourceParam in UserMessage and ToolMessage."""
    image_url = ImageUrlPart(
        url="https://example.com/image.png",
        media_type="image/png",
        cache_breakpoint=True,
    )
    wire = _wire_messages(
        [
            UserMessage(content=(image_url,)),
            ToolMessage(tool_call_id="tu_1", content=(image_url,)),
        ],
        automatic_cache_breakpoints=False,
        cache_ttl="5m",
        message_mark_budget=4,
    )
    expected_unmarked_block = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/image.png"},
    }
    assert _content_blocks(wire[0]) == [
        {**expected_unmarked_block, "cache_control": {"type": "ephemeral"}}
    ]
    tool_result = _content_blocks(wire[1])[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("content") == [expected_unmarked_block]
    assert tool_result.get("cache_control") == {"type": "ephemeral"}


@pytest.mark.parametrize(
    "message",
    [
        UserMessage(content=(AudioPart(data=b"wav", media_type="audio/wav"),)),
        ToolMessage(
            tool_call_id="tu_1",
            content=(AudioPart(data=b"wav", media_type="audio/wav"),),
        ),
    ],
)
def test_build_request_reports_audio_part_as_invalid_request(message: Message) -> None:
    """AnthropicMessagesAdapter returns InvalidRequest for AudioPart."""
    request = _structured_bound().build_request([message])
    assert isinstance(request, InvalidRequest)
    assert "AudioPart" in request.reason
    assert type(message).__name__ in request.reason


def test_wire_messages_rejects_tool_result_image_with_unsupported_media_type() -> None:
    """A tool_result image media type outside the accepted set is not sendable."""
    messages = [
        ToolMessage(tool_call_id="tu_1", content=(ImagePart(data=b"x", media_type="image/tiff"),))
    ]
    with pytest.raises(_NotSendableError, match="image/tiff"):
        _ = _wire_messages(
            messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
        )


@pytest.mark.parametrize("parallel_tool_calls", [True, False])
@pytest.mark.parametrize(
    ("tool_choice", "expected_without_parallel_flag"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        (SpecificToolChoice(tool_name="x"), {"type": "tool", "name": "x"}),
    ],
    ids=["auto", "required", "specific_tool"],
)
def test_wire_tool_choice_carries_the_inverted_parallel_flag(
    tool_choice: ToolChoice,
    expected_without_parallel_flag: dict[str, object],
    *,
    parallel_tool_calls: bool,
) -> None:
    """Neutral required maps to any, and every form carrying the flag inverts it.

    disable_parallel_tool_use inverts parallel_tool_calls.
    """
    assert _wire_tool_choice(tool_choice, parallel_tool_calls=parallel_tool_calls) == {
        **expected_without_parallel_flag,
        "disable_parallel_tool_use": not parallel_tool_calls,
    }


@pytest.mark.parametrize("parallel_tool_calls", [True, False])
def test_wire_tool_choice_none_forbids_calls_and_carries_no_parallel_flag(
    *, parallel_tool_calls: bool
) -> None:
    """Neutral none maps to the none form, which takes no parallel flag at either binding."""
    assert _wire_tool_choice("none", parallel_tool_calls=parallel_tool_calls) == {"type": "none"}


def _adapter() -> AnthropicMessagesAdapter:
    """Build an adapter over a keyless client, valid because no request is sent."""
    return AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
    )


def test_config_fingerprint_data_contains_only_stored_request_configuration() -> None:
    """Fingerprint data includes constructor request settings and excludes billing settings."""
    adapter = AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
        default_max_completion_tokens=8192,
        cache_ttl="1h",
        service_tier="standard_only",
        inference_geo="us",
    )
    assert adapter.config_fingerprint_data() == {
        "cache_ttl": "1h",
        "default_max_completion_tokens": 8192,
        "inference_geo": "us",
        "service_tier": "standard_only",
    }


def _binding(
    *,
    system_prompt: str | tuple[TextPart, ...] | None,
    tool_schemas: tuple[ToolSchema, ...],
    automatic_cache_breakpoints: bool,
    provider_executed_tools: tuple[Mapping[str, object], ...] = (),
    tool_choice: ToolChoice = "required",
    extra_body: Mapping[str, object] | None = None,
    temperature: float | None = None,
) -> Binding:
    """Assemble a binding with the fields these request tests vary."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        provider_executed_tools=provider_executed_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=False,
        inference_params=InferenceParams(reasoning_effort="high", temperature=temperature),
        automatic_cache_breakpoints=automatic_cache_breakpoints,
        extra_body=extra_body,
    )


def test_request_omits_tool_sentinels_without_tools() -> None:
    """No tools leaves both tools and tool_choice at the omit sentinel."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    assert precomputed_fields.max_tokens == 4096
    assert isinstance(precomputed_fields.tools, anthropic.Omit)
    assert isinstance(precomputed_fields.tool_choice, anthropic.Omit)
    assert precomputed_fields.output_config == {"effort": "high"}
    assert precomputed_fields.thinking == {"type": "adaptive"}


def test_anthropic_rejects_allowed_tools_choice_at_text_bind() -> None:
    """AnthropicMessagesAdapter rejects AllowedToolsChoice during text binding."""
    with pytest.raises(TypeError, match="does not support AllowedToolsChoice"):
        _ = _adapter().bind_text(
            _binding(
                system_prompt="sys",
                tool_schemas=_tool_schemas(),
                automatic_cache_breakpoints=False,
                tool_choice=AllowedToolsChoice(mode="auto", tool_names=("get_weather",)),
            )
        )


def test_provider_executed_tools_follow_function_tools_and_receive_automatic_caching() -> None:
    """Provider-executed tools keep order and can carry the automatic cache marker."""
    provider_tool: dict[str, object] = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }
    precomputed = _adapter()._precompute_fields(
        _binding(
            system_prompt=None,
            tool_schemas=_tool_schemas(),
            provider_executed_tools=(provider_tool,),
            automatic_cache_breakpoints=True,
        )
    )
    tools = _block_list(precomputed.tools)
    assert tools[0].get("name") == "get_weather"
    assert tools[1].get("type") == "web_search_20250305"
    assert tools[1].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in provider_tool


def test_provider_executed_tool_binds_without_function_tools() -> None:
    """A provider-executed tool does not require an application function."""
    precomputed = _adapter()._precompute_fields(
        _binding(
            system_prompt="system",
            tool_schemas=(),
            provider_executed_tools=({"type": "web_search_20250305", "name": "web_search"},),
            automatic_cache_breakpoints=False,
        )
    )
    tools = _block_list(precomputed.tools)
    assert tools == [{"type": "web_search_20250305", "name": "web_search"}]
    assert precomputed.tool_choice == {"type": "any", "disable_parallel_tool_use": True}


@pytest.mark.parametrize(
    "tool_type",
    [
        "tool_search_tool_bm25",
        "tool_search_tool_bm25_20251119",
        "tool_search_tool_regex",
        "tool_search_tool_regex_20251119",
        "web_fetch_20250910",
        "web_fetch_20260209",
        "web_fetch_20260309",
        "web_fetch_20260318",
        "web_search_20250305",
        "web_search_20260209",
        "web_search_20260318",
    ],
)
def test_every_supported_anthropic_provider_type_binds(tool_type: str) -> None:
    """Each reviewed Anthropic provider-executed `type` reaches Messages unchanged."""
    provider_tool: dict[str, object] = {"type": tool_type}
    precomputed = _adapter()._precompute_fields(
        _binding(
            system_prompt="system",
            tool_schemas=(),
            provider_executed_tools=(provider_tool,),
            automatic_cache_breakpoints=False,
        )
    )
    assert _block_list(precomputed.tools) == [provider_tool]


@pytest.mark.parametrize(
    "code_execution_type", ["code_execution_20260120", "code_execution_20260521"]
)
@pytest.mark.parametrize(
    "web_tool_type",
    [
        "web_fetch_20260209",
        "web_fetch_20260309",
        "web_fetch_20260318",
        "web_search_20260209",
        "web_search_20260318",
    ],
)
def test_supported_code_execution_requires_a_qualifying_web_tool(
    code_execution_type: str, web_tool_type: str
) -> None:
    """Every reviewed code-execution type is free beside each qualifying web family."""
    precomputed = _adapter()._precompute_fields(
        _binding(
            system_prompt="system",
            tool_schemas=(),
            provider_executed_tools=(
                {"type": web_tool_type},
                {"type": code_execution_type},
            ),
            automatic_cache_breakpoints=False,
        )
    )
    assert precomputed.provider_tools.code_execution_exempt


@pytest.mark.parametrize(
    "tool_type",
    [
        "advisor_20260301",
        "bash_20250124",
        "code_execution_20250522",
        "code_execution_20250825",
        "computer_20250124",
        "custom",
        "mcp_toolset",
        "memory_20250818",
        "text_editor_20250124",
        "text_editor_20250429",
        "text_editor_20250728",
        "unknown",
    ],
)
def test_every_unlisted_anthropic_provider_type_is_rejected(tool_type: str) -> None:
    """Messages rejects every reviewed client-executed or unaudited `type`."""
    with pytest.raises(ValueError, match="supported string type"):
        _ = _adapter()._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=({"type": tool_type},),
                automatic_cache_breakpoints=False,
            )
        )


@pytest.mark.parametrize("provider_tool", [{}, {"type": 1}])
def test_anthropic_provider_type_must_be_a_supported_string(
    provider_tool: Mapping[str, object],
) -> None:
    """Missing and non-string `type` values fail before requests."""
    with pytest.raises(ValueError, match="supported string type"):
        _ = _adapter()._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=(provider_tool,),
                automatic_cache_breakpoints=False,
            )
        )


@pytest.mark.parametrize(
    "code_execution_type", ["code_execution_20260120", "code_execution_20260521"]
)
def test_standalone_anthropic_code_execution_is_rejected(code_execution_type: str) -> None:
    """Standalone code execution lacks exact response billing evidence."""
    with pytest.raises(ValueError, match="qualifying web tool"):
        _ = _adapter()._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=({"type": code_execution_type},),
                automatic_cache_breakpoints=False,
            )
        )


def test_anthropic_bedrock_rejects_provider_executed_tools() -> None:
    """Anthropic pricing does not establish Bedrock provider-tool billing."""
    adapter = AnthropicMessagesAdapter(
        client=AsyncAnthropicBedrock(aws_region="us-east-1"),
        model="m",
        pricing=_PRICING,
        provider_name="aws.bedrock",
    )
    with pytest.raises(ValueError, match="provider_name='anthropic'"):
        _ = adapter._precompute_fields(
            _binding(
                system_prompt="system",
                tool_schemas=(),
                provider_executed_tools=({"type": "web_search_20250305"},),
                automatic_cache_breakpoints=False,
            )
        )


def test_anthropic_provider_rate_defaults_to_unavailable() -> None:
    """Ordinary custom pricing requires no unused web-search rate."""
    parameter = inspect.signature(AnthropicPricingTable).parameters[
        "web_search_usd_per_invocation"
    ]
    assert parameter.default is None


def test_provider_executed_cache_markers_reduce_the_message_budget() -> None:
    """Provider-executed cache markers count toward Anthropic's request limit."""
    provider_tools = tuple(
        {
            "type": "web_search_20250305",
            "name": f"web_search_{index}",
            "cache_control": {"type": "ephemeral"},
        }
        for index in range(2)
    )
    precomputed = _adapter()._precompute_fields(
        _binding(
            system_prompt=None,
            tool_schemas=(),
            provider_executed_tools=provider_tools,
            automatic_cache_breakpoints=False,
        )
    )
    assert precomputed.message_mark_budget == 2


def test_request_passes_widened_reasoning_effort_through() -> None:
    """A value outside anthropic's own effort literal ("minimal") reaches the request unchanged."""
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        provider_executed_tools=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(reasoning_effort="minimal"),
        automatic_cache_breakpoints=False,
    )
    precomputed_fields = _adapter()._precompute_fields(binding)
    assert precomputed_fields.output_config == {"effort": "minimal"}
    assert precomputed_fields.thinking == {"type": "adaptive"}


def test_request_omits_thinking_and_output_config_without_reasoning_effort() -> None:
    """A None reasoning_effort leaves both output_config and thinking at the omit sentinel."""
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        provider_executed_tools=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(),
        automatic_cache_breakpoints=False,
    )
    precomputed_fields = _adapter()._precompute_fields(binding)
    assert isinstance(precomputed_fields.output_config, anthropic.Omit)
    assert isinstance(precomputed_fields.thinking, anthropic.Omit)


def test_request_maps_temperature_and_omits_it_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound temperature enters extra_body. None leaves the omit sentinel."""
    adapter = _adapter()
    unset = adapter._precompute_fields(
        _binding(system_prompt=None, tool_schemas=(), automatic_cache_breakpoints=False)
    )
    assert isinstance(unset.temperature, anthropic.Omit)
    precomputed_fields = adapter._precompute_fields(
        _binding(
            system_prompt=None,
            tool_schemas=(),
            automatic_cache_breakpoints=False,
            temperature=0.2,
        )
    )
    assert precomputed_fields.temperature == 0.2
    text_bound = _BoundAnthropicText(adapter=adapter, precomputed_fields=precomputed_fields)
    assert _kwarg_sent(monkeypatch, text_bound, "extra_body") == {"temperature": 0.2}


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request. None leaves the omit sentinel.

    The sentinel omits an unstated tier from the request.
    """
    binding = _binding(system_prompt=None, tool_schemas=(), automatic_cache_breakpoints=False)
    assert isinstance(_adapter()._precompute_fields(binding).service_tier, anthropic.Omit)
    stated = AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
        service_tier="standard_only",
    )
    assert stated._precompute_fields(binding).service_tier == "standard_only"


def test_request_marks_the_system_block_for_automatic_cache_breakpoints() -> None:
    """The system block follows `automatic_cache_breakpoints`."""
    cached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    assert _block_list(cached.system)[0].get("cache_control") == {"type": "ephemeral"}
    uncached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=False)
    )
    assert "cache_control" not in _block_list(uncached.system)[0]


def test_request_marks_last_tool_only_without_a_system_prompt() -> None:
    """The prefix breakpoint sits on the last tool only when no system prompt follows."""
    schemas = _tool_schemas()
    without_system = _adapter()._precompute_fields(
        _binding(system_prompt=None, tool_schemas=schemas, automatic_cache_breakpoints=True)
    )
    assert _block_list(without_system.tools)[-1].get("cache_control") == {"type": "ephemeral"}
    with_system = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=schemas, automatic_cache_breakpoints=True)
    )
    assert "cache_control" not in _block_list(with_system.tools)[-1]


def test_user_content_blocks_rejects_unsupported_image_media_type() -> None:
    """An image media type outside the accepted set is not sendable."""
    message = UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),))
    with pytest.raises(_NotSendableError):
        _ = _user_content_blocks(message)


class _FakeSDKMessageStream(AsyncMessageStream[None]):
    """Replay constructed events without a connection."""

    def __init__(  # pyrefly: ignore[missing-super-call]
        self,
        replay_events: Sequence[ParsedMessageStreamEvent],
        message_snapshot: ParsedMessage[None],
        headers: dict[str, str] | None = None,
    ) -> None:
        self._replay_events = list(replay_events)
        self._message_snapshot = message_snapshot
        self._http_response = httpx2.Response(
            200,
            headers=headers,
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

    @property
    @override
    def response(self) -> httpx2.Response:
        return self._http_response

    @override
    async def __aiter__(self) -> AsyncIterator[ParsedMessageStreamEvent]:
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
    replay_events: Sequence[ParsedMessageStreamEvent],
    message_snapshot: ParsedMessage[None],
    headers: dict[str, str] | None = None,
) -> _AnthropicStream:
    """Build an adapter stream over replayed events, reading headers off a constructed response."""
    return _AnthropicStream(
        sdk_stream=_FakeSDKMessageStream(replay_events, message_snapshot, headers),
        pricing=_PRICING,
    )


def test_a_stream_reports_the_request_id_header_of_the_response_it_reads() -> None:
    """Read a streamed request ID from the stream response."""
    snapshot = _message_snapshot("end_turn")
    with_header = _anthropic_stream([], snapshot, {"request-id": "req_stream"})
    assert with_header.request_id() == "req_stream"
    assert _anthropic_stream([], snapshot).request_id() is None


def _text_delta_event(text: str, index: int) -> at.RawContentBlockDeltaEvent:
    """Build one raw text-delta event."""
    return at.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=at.TextDelta(type="text_delta", text=text),
    )


def _thinking_delta_event(thinking: str, index: int) -> at.RawContentBlockDeltaEvent:
    """Build one thinking-delta event, belonging to the numbered thinking block."""
    return at.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=at.ThinkingDelta(type="thinking_delta", thinking=thinking),
    )


def _thinking_block_stop_event(thinking: str, index: int) -> ParsedContentBlockStopEvent:
    """Build the stop event closing the numbered thinking block."""
    return ParsedContentBlockStopEvent(
        type="content_block_stop",
        index=index,
        content_block=at.ThinkingBlock(
            type="thinking", thinking=thinking, signature=f"sig-{index}"
        ),
    )


def _streamed_reasoning(translated: Sequence[StreamItem]) -> str:
    """Concatenate the reasoning deltas, as an application rendering them as flowing text would."""
    return "".join(item.text for item in translated if isinstance(item, ReasoningDelta))


def _collected_items(
    replay_events: Sequence[ParsedMessageStreamEvent],
    message_snapshot: ParsedMessage[None] | None = None,
) -> list[StreamItem]:
    """Drain the translated items into a list. None means a bare end_turn snapshot."""
    snapshot = message_snapshot if message_snapshot is not None else _message_snapshot("end_turn")

    async def scenario() -> list[StreamItem]:
        adapter_stream = _anthropic_stream(replay_events, snapshot)
        return [item async for item in adapter_stream.items()]

    return asyncio.run(scenario())


def test_stream_yields_bare_text_argument_fragments_and_one_complete_tool_call() -> None:
    """Stream text, argument fragments, and completed ToolCall values."""

    def args_fragment(partial_json: str) -> at.RawContentBlockDeltaEvent:
        return at.RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=at.InputJSONDelta(type="input_json_delta", partial_json=partial_json),
        )

    text_block_stop = ParsedContentBlockStopEvent(
        type="content_block_stop",
        index=0,
        content_block=ParsedTextBlock(type="text", text="hey"),
    )
    tool_use_block = at.ToolUseBlock(
        type="tool_use", id="tu_1", name="get_weather", input={"city": "Nairobi"}
    )
    tool_use_block_stop = ParsedContentBlockStopEvent(
        type="content_block_stop", index=1, content_block=tool_use_block
    )

    translated = _collected_items(
        [
            _text_delta_event("he", 0),
            _text_delta_event("y", 0),
            text_block_stop,
            args_fragment(""),
            args_fragment('{"city"'),
            args_fragment(': "Nairobi"}'),
            tool_use_block_stop,
        ],
        _message_snapshot("tool_use", [at.TextBlock(type="text", text="hey"), tool_use_block]),
    )
    assert translated == [
        "he",
        "y",
        ToolCallDelta(id="tu_1", name="get_weather", partial_args_json='{"city"'),
        ToolCallDelta(id="tu_1", name="get_weather", partial_args_json=': "Nairobi"}'),
        ToolCall(id="tu_1", name="get_weather", args_json='{"city": "Nairobi"}'),
    ]


def test_a_server_tool_use_blocks_argument_fragments_yield_nothing() -> None:
    """input_json_delta also grows a server_tool_use block, which is not a langchaint tool call."""
    fragment = at.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=at.InputJSONDelta(type="input_json_delta", partial_json='{"query": "x"}'),
    )
    snapshot = _message_snapshot(
        "end_turn",
        [at.ServerToolUseBlock(type="server_tool_use", id="st_1", name="web_search", input={})],
    )
    assert _collected_items([fragment], snapshot) == []


def test_two_thinking_blocks_stream_separated_by_a_blank_line() -> None:
    """A block boundary separates reasoning text with a blank line."""
    translated = _collected_items([
        _thinking_delta_event("First, water ", 0),
        _thinking_delta_event("evaporates.", 0),
        _thinking_block_stop_event("First, water evaporates.", 0),
        _thinking_delta_event("Then it condenses.", 1),
        _thinking_block_stop_event("Then it condenses.", 1),
    ])
    assert _streamed_reasoning(translated) == "First, water evaporates.\n\nThen it condenses."


def test_a_block_stop_with_no_delta_after_it_streams_no_trailing_separator() -> None:
    """The separator precedes the next thinking delta, so a last block contributes none.

    Only a thinking delta consumes a pending separator.
    """
    translated = _collected_items([
        _thinking_delta_event("thought it over", 0),
        _thinking_block_stop_event("thought it over", 0),
        _text_delta_event("hey", 1),
    ])
    assert translated == [ReasoningDelta(text="thought it over"), "hey"]


def test_a_thinking_delta_carrying_no_characters_is_dropped_rather_than_streamed() -> None:
    """An empty delta is not text: it yields no item and leaves the next block unseparated."""
    translated = _collected_items([
        _thinking_delta_event("", 0),
        _thinking_block_stop_event("", 0),
        _thinking_delta_event("thought it over", 1),
    ])
    assert translated == [ReasoningDelta(text="thought it over")]


def test_an_empty_thinking_delta_keeps_the_separator_for_the_next_delta_with_text() -> None:
    """A pending separator before a dropped delta remains before the next block's text."""
    translated = _collected_items([
        _thinking_delta_event("First.", 0),
        _thinking_block_stop_event("First.", 0),
        _thinking_delta_event("", 1),
        _thinking_delta_event("Then.", 1),
    ])
    assert _streamed_reasoning(translated) == "First.\n\nThen."


def test_a_redacted_thinking_block_streams_no_text_and_no_extra_blank_line() -> None:
    """A redacted block yields no delta of its own and does not double the blank line around it."""
    redacted_stop = ParsedContentBlockStopEvent(
        type="content_block_stop",
        index=1,
        content_block=at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes"),
    )
    translated = _collected_items([
        _thinking_delta_event("First.", 0),
        _thinking_block_stop_event("First.", 0),
        redacted_stop,
        _thinking_delta_event("Then.", 2),
    ])
    assert translated == [
        ReasoningDelta(text="First."),
        ReasoningDelta(text=REASONING_PART_SEPARATOR),
        ReasoningDelta(text="Then."),
    ]


def test_a_redacted_block_after_a_thinking_block_with_no_text_arms_no_separator() -> None:
    """A redacted block does not separate what its neighbors left unseparated.

    An empty preceding block adds no separator before redacted reasoning.
    """
    redacted_stop = ParsedContentBlockStopEvent(
        type="content_block_stop",
        index=1,
        content_block=at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes"),
    )
    translated = _collected_items([
        _thinking_delta_event("", 0),
        _thinking_block_stop_event("", 0),
        redacted_stop,
        _thinking_delta_event("Then.", 2),
    ])
    assert translated == [ReasoningDelta(text="Then.")]


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
        assistant_message = _assistant_message_from(await adapter_stream.final())
        reasoning_part = assistant_message.turn[0]
        assert isinstance(reasoning_part, ReasoningPart)
        assert reasoning_part.raw == {
            "type": "thinking",
            "thinking": "check",
            "signature": "sig-1",
        }

    asyncio.run(scenario())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundAnthropicStructured[_StructuredReport]:
    """Build a structured-bound adapter over a keyless client. No request is sent."""
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=False)
    )
    return _BoundAnthropicStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )


def _structured_parse(message: at.Message) -> ResponseOutcome[_StructuredReport | None]:
    """Run the structured binding's parse over one message, with the turn that message carries."""
    return _structured_bound()._parsed_outcome(message, _assistant_message_from(message))


_REPORT_JSON = '{"city": "Nairobi", "celsius": 25}'
"""Text that validates into _StructuredReport."""


def _kwarg_sent[OutputT](
    monkeypatch: pytest.MonkeyPatch, bound: _BoundAnthropic[OutputT], key: str
) -> object:
    """Open one stream through a fake, capturing the request kwarg key it was passed."""
    captured: list[object] = []

    class _FakeStreamManager:
        async def __aenter__(self) -> _FakeSDKMessageStream:
            return _FakeSDKMessageStream([], _message_snapshot("end_turn"))

    def fake_stream(**request_kwargs: object) -> _FakeStreamManager:
        captured.append(request_kwargs[key])
        return _FakeStreamManager()

    monkeypatch.setattr(bound._adapter.client.messages, "stream", fake_stream)

    async def scenario() -> None:
        request = bound.build_request([UserMessage(content="q")])
        assert not isinstance(request, InvalidRequest)
        await bound.open_stream(request)

    asyncio.run(scenario())
    (kwarg,) = captured
    return kwarg


def test_identity_reads_the_messages_own_id_and_served_model() -> None:
    """Both values come off the message verbatim, neither from the id the binding sent.

    A streamed message carries no request ID.
    """
    identity = _structured_bound().identity_from_raw(_message_with_content([]), request_id=None)
    assert identity == ResponseIdentity(
        model_served="claude-sonnet-5", response_id="msg_1", request_id=None
    )


def test_identity_reads_the_adapter_stream_request_id() -> None:
    """The AdapterStream request id reaches ResponseIdentity unchanged."""
    message = _message_with_content([])
    identity = _structured_bound().identity_from_raw(message, request_id="req_anthropic")
    assert identity.request_id == "req_anthropic"


def _structured_message(
    text: str | None,
    stop_reason: at.StopReason | None = "end_turn",
) -> at.Message:
    """Build a message whose first text block carries the given text. None gives no text block."""
    return at.Message(
        id="msg_1",
        content=[at.TextBlock(type="text", text=text)] if text is not None else [],
        model="claude-sonnet-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=at.Usage(input_tokens=1, output_tokens=1),
    )


def test_structured_bind_merges_the_sdk_schema_into_the_bindings_output_config() -> None:
    """output_config carries the response schema and bound effort."""
    adapted_type: TypeAdapter[_StructuredReport] = TypeAdapter(_StructuredReport)
    assert _structured_bound()._precomputed_fields.output_config == {
        "effort": "high",
        "format": {"schema": transform_schema(adapted_type.json_schema()), "type": "json_schema"},
    }


def test_the_structured_request_sends_the_output_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_stream sends the precomputed output_config."""
    structured_bound = _structured_bound()
    output_config = _kwarg_sent(monkeypatch, structured_bound, "output_config")
    assert output_config == structured_bound._precomputed_fields.output_config


def test_request_rejects_an_extra_body_key_the_adapter_populates() -> None:
    """An extra_body key that open_stream passes as its own keyword raises at bind time.

    Rejecting the duplicate key prevents extra_body from overriding the binding.
    """
    with pytest.raises(ValueError, match="max_tokens"):
        _ = _adapter()._precompute_fields(
            _binding(
                system_prompt=None,
                tool_schemas=(),
                automatic_cache_breakpoints=True,
                extra_body={"max_tokens": 10},
            )
        )


def test_the_request_sends_extra_body_by_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_stream passes the binding's extra_body to the SDK's extra_body parameter."""
    adapter = _adapter()
    extra_body = {"top_k": 5}
    text_bound = _BoundAnthropicText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(
            _binding(
                system_prompt=None,
                tool_schemas=(),
                automatic_cache_breakpoints=True,
                extra_body=extra_body,
            )
        ),
    )
    assert _kwarg_sent(monkeypatch, text_bound, "extra_body") is extra_body


def test_temperature_keeps_caller_extra_body_fields_by_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temperature combines with caller fields and retains their mapping by reference."""
    adapter = _adapter()
    caller_fields: dict[str, object] = {"top_k": 5}
    text_bound = _BoundAnthropicText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(
            _binding(
                system_prompt=None,
                tool_schemas=(),
                automatic_cache_breakpoints=True,
                extra_body=caller_fields,
                temperature=0.2,
            )
        ),
    )
    sent_extra_body = _kwarg_sent(monkeypatch, text_bound, "extra_body")
    assert isinstance(sent_extra_body, Mapping)
    assert dict(sent_extra_body) == {"temperature": 0.2, "top_k": 5}
    caller_fields["top_k"] = 6
    assert sent_extra_body["top_k"] == 6
    caller_fields["temperature"] = 0.7
    assert dict(sent_extra_body) == {"temperature": 0.2, "top_k": 6}


def test_structured_bind_validates_the_turns_text_into_the_instance() -> None:
    """The structured bound adapter validates the turn's text block into the response_format."""
    outcome = _structured_parse(_structured_message(_REPORT_JSON))
    assert isinstance(outcome, AdapterResult)
    assert outcome.output == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_output_may_inherit_no_output() -> None:
    """AdapterResult distinguishes successful output from NoOutput."""

    class ReportAlsoNoOutput(BaseModel, NoOutput):
        assistant_message: AssistantMessage = AssistantMessage(turn=())
        city: str
        celsius: int

    adapter = _adapter()
    bound = _BoundAnthropicStructured(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(
            _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=False)
        ),
        response_format=ReportAlsoNoOutput,
    )
    outcome = bound.interpret(_structured_message(_REPORT_JSON))
    assert isinstance(outcome, AdapterResult)
    assert outcome.output == ReportAlsoNoOutput(city="Nairobi", celsius=25)


@pytest.mark.parametrize(
    ("stop_reason", "expected_outcome_type"),
    [
        ("end_turn", EmptyTurn),
        ("refusal", Refusal),
        ("max_tokens", MaxCompletionTokensExceeded),
        ("model_context_window_exceeded", ContextWindowExceeded),
        (None, UnfinishedTurn),
    ],
    ids=[
        "end_turn",
        "refusal",
        "max_tokens",
        "model_context_window_exceeded",
        "no_stop_reason",
    ],
)
def test_structured_bind_reports_a_text_free_turn_by_its_stop_reason(
    stop_reason: at.StopReason | None, expected_outcome_type: type[NoOutputOutcome]
) -> None:
    """A turn with no text block parses no instance, so the stop reason is what names the outcome.

    A null stop reason is not a finished turn, so it is unfinished rather than empty.
    """
    outcome = _structured_parse(_structured_message(None, stop_reason=stop_reason))
    assert isinstance(outcome, expected_outcome_type)


def test_structured_bind_reports_schema_violation_on_text_the_model_rejects() -> None:
    """A finished turn whose text the response_format rejects is SchemaViolation.

    validation_error_json preserves the field, constraint, and rejected value.
    """
    outcome = _structured_parse(_structured_message('{"city": "Nairobi", "celsius": "SENTINEL"}'))
    assert isinstance(outcome, SchemaViolation)
    rejections = json.loads(outcome.validation_error_json)
    assert [rejection["loc"] for rejection in rejections] == [["celsius"]]
    assert rejections[0]["input"] == "SENTINEL"


def test_structured_bind_reports_max_completion_tokens_exceeded_on_text_cut_mid_json() -> None:
    """Truncated JSON at max_tokens returns MaxCompletionTokensExceeded."""
    outcome = _structured_parse(_structured_message('{"city": "Nair', stop_reason="max_tokens"))
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_bind_reports_a_tool_use_turn_as_none() -> None:
    """A tool_use turn parses no instance and nothing went wrong, so the output is None."""
    outcome = _structured_parse(_structured_message(None, stop_reason="tool_use"))
    assert isinstance(outcome, AdapterResult)
    assert outcome.output is None


def test_structured_bind_reports_a_tool_use_turn_whose_text_is_not_the_instance_as_none() -> None:
    """A tool_use turn whose text block is prose is the tool call, not a schema violation."""
    outcome = _structured_parse(_structured_message("let me look that up", stop_reason="tool_use"))
    assert isinstance(outcome, AdapterResult)
    assert outcome.output is None


def test_structured_bind_reports_a_paused_turn_as_unfinished_naming_the_stop_reason() -> None:
    """pause_turn is an unfinished turn, and the reason quotes anthropic's own word."""
    outcome = _structured_parse(_structured_message(None, stop_reason="pause_turn"))
    assert isinstance(outcome, UnfinishedTurn)
    assert "pause_turn" in outcome.reason


def test_structured_bind_reports_an_unfinished_turn_ahead_of_a_schema_violation() -> None:
    """A paused turn whose text is not the instance is the pause, which langchaint cannot continue."""
    outcome = _structured_parse(_structured_message("partial thought", stop_reason="pause_turn"))
    assert isinstance(outcome, UnfinishedTurn)


def _rate_limit_error(headers: dict[str, str]) -> anthropic.RateLimitError:
    """Build the SDK's 429 exception around a constructed httpx2 response."""
    response = httpx2.Response(
        429,
        headers=headers,
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def test_parse_anthropic_reads_retry_after_from_the_headers_without_letting_it_pick() -> None:
    """retry-after sets retry_after without changing the verdict type."""
    assert parse_anthropic(_rate_limit_error({"Retry-After-MS": "1500"})) == PauseAll(
        retry_after=1.5
    )
    assert parse_anthropic(_rate_limit_error({})) == PauseAll(retry_after=None)
    bad_request = _status_error(anthropic.BadRequestError, 400, {"retry-after": "7"})
    assert parse_anthropic(bad_request) == DoNotRetry()


def test_parse_anthropic_counts_a_fallthrough_and_a_listed_row_adds_nothing() -> None:
    """An unlisted status lands one tagged count. A listed status leaves the counter alone."""
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    assert parse_anthropic(_status_error(anthropic.RateLimitError, 429)) == PauseAll(
        retry_after=None
    )
    assert parse_anthropic(_status_error(anthropic.APIStatusError, 408)) == RetryThisOne(
        retry_after=None
    )
    assert dict(PARSE_FALLTHROUGH_COUNTS) == before
    assert parse_anthropic(_status_error(anthropic.APIStatusError, 599)) == RetryThisOne(
        retry_after=None
    )
    tag = "status=599 type=None"
    assert PARSE_FALLTHROUGH_COUNTS[tag] == before.get(tag, 0) + 1


def test_parse_anthropic_pauses_on_a_recognized_throttle_type_at_an_unlisted_status() -> None:
    """A rate-limit or overload error type pauses the rate-limit quota whatever status carried it."""
    overloaded = _status_error(anthropic.APIStatusError, 418, error_type="overloaded_error")
    assert parse_anthropic(overloaded) == PauseAll(retry_after=None)


def test_parse_anthropic_obeys_a_retry_directive_over_the_status_tables() -> None:
    """x-should-retry overrides status-table retry decisions."""
    final_500 = _status_error(anthropic.InternalServerError, 500, {"x-should-retry": "false"})
    assert parse_anthropic(final_500) == DoNotRetry()
    retryable_400 = _status_error(
        anthropic.BadRequestError, 400, {"x-should-retry": "true", "retry-after": "3"}
    )
    assert parse_anthropic(retryable_400) == RetryThisOne(retry_after=3.0)
    retryable_429 = _status_error(
        anthropic.RateLimitError, 429, {"x-should-retry": "true"}, "rate_limit_error"
    )
    assert parse_anthropic(retryable_429) == PauseAll(retry_after=None)


def test_false_retry_directive_stops_request_and_pauses_rate_limit_quota() -> None:
    """x-should-retry=false preserves a required SharedBackoff pause."""
    throttled = _status_error(
        anthropic.RateLimitError, 429, {"x-should-retry": "false", "retry-after": "7"}
    )
    assert parse_anthropic(throttled) == PauseAllDoNotRetry(retry_after=7.0)
    overloaded = _status_error(
        anthropic.APIStatusError, 418, {"x-should-retry": "false"}, "overloaded_error"
    )
    assert parse_anthropic(overloaded) == PauseAllDoNotRetry(retry_after=None)


def test_parse_anthropic_ignores_a_retry_directive_on_the_streams_200_status() -> None:
    """A mid-stream error ignores retry headers from status 200."""
    overloaded = _status_error(
        anthropic.APIStatusError, 200, {"x-should-retry": "false"}, "overloaded_error"
    )
    assert parse_anthropic(overloaded) == PauseAll(retry_after=None)
    rejected = _status_error(
        anthropic.APIStatusError, 200, {"x-should-retry": "true"}, "invalid_request_error"
    )
    assert parse_anthropic(rejected) == DoNotRetry()


def test_parse_anthropic_retries_a_transient_type_at_the_streams_200_status() -> None:
    """api_error and timeout_error retry by error type and count unlisted statuses."""
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    for transient_type in ("api_error", "timeout_error"):
        failed = _status_error(anthropic.APIStatusError, 200, error_type=transient_type)
        assert parse_anthropic(failed) == RetryThisOne(retry_after=None)
        tag = f"status=200 type={transient_type}"
        assert PARSE_FALLTHROUGH_COUNTS[tag] == before.get(tag, 0) + 1


def test_request_id_from_error_reads_the_sdk_errors_own_header_and_nothing_else() -> None:
    """The override reports the header the SDK read off the error response, None for any other error.

    Anthropic sends the request ID in request-id.
    Missing headers return None.
    """
    adapter = _adapter()
    assert adapter.request_id_from_error(_rate_limit_error({"request-id": "req_429"})) == "req_429"
    assert adapter.request_id_from_error(_rate_limit_error({})) is None
    assert adapter.request_id_from_error(ValueError("boom")) is None


def _status_error[ErrorT: anthropic.APIStatusError](
    error_class: type[ErrorT],
    status_code: int,
    headers: dict[str, str] | None = None,
    error_type: str | None = None,
) -> ErrorT:
    """Build one of the SDK's status exceptions around a constructed httpx2 response.

    error_type fills the SDK exception's error.type.
    None represents a non-JSON body.
    """
    response = httpx2.Response(
        status_code,
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
        headers=headers,
    )
    body = None if error_type is None else {"error": {"type": error_type, "message": "boom"}}
    return error_class("boom", response=response, body=body)


def _connection_error() -> anthropic.APIConnectionError:
    """Build the SDK's transport-failure exception, which carries a request and no response."""
    return anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )


def _anthropic_adapter_of(llm: LLM) -> AnthropicMessagesAdapter:
    """Narrow an LLM to its concrete adapter so tests read its client/model/pricing."""
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    return adapter


@pytest.mark.parametrize(
    ("model", "expected_client_class"),
    [
        ("anthropic.claude-fable-5", AsyncAnthropicBedrockMantle),
        ("anthropic.claude-opus-4-8", AsyncAnthropicBedrockMantle),
        ("anthropic.claude-haiku-4-5", AsyncAnthropicBedrockMantle),
        ("us.anthropic.claude-opus-4-6-v1", AsyncAnthropicBedrock),
        ("us.anthropic.claude-sonnet-4-6", AsyncAnthropicBedrock),
    ],
)
def test_bedrock_model_sends_the_id_verbatim_on_its_apis_client_class(
    model: AnthropicBedrockModelName,
    expected_client_class: type[AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle],
) -> None:
    """Each Bedrock wire model id reaches its API's client class unchanged, retries pinned off."""
    adapter = _anthropic_adapter_of(AnthropicBedrock(aws_region="us-east-1").model(model))
    assert adapter.model == model
    assert isinstance(adapter.client, expected_client_class)
    assert adapter.client.max_retries == 0


def test_bedrock_model_uses_its_bedrock_pricing_object() -> None:
    """The Bedrock catalog remains independent from direct Anthropic pricing."""
    adapter = _anthropic_adapter_of(
        AnthropicBedrock(aws_region="us-east-1").model("us.anthropic.claude-opus-4-6-v1")
    )
    assert adapter.pricing is ANTHROPIC_BEDROCK_PRICING["us.anthropic.claude-opus-4-6-v1"]


def test_bedrock_model_uses_a_matching_supplied_client() -> None:
    """Use a matching supplied client with SDK retries disabled."""
    adapter = _anthropic_adapter_of(
        AnthropicBedrock(client=AsyncAnthropicBedrockMantle(aws_region="eu-west-1")).model(
            "anthropic.claude-opus-4-8"
        )
    )
    assert isinstance(adapter.client, AsyncAnthropicBedrockMantle)
    assert adapter.model == "anthropic.claude-opus-4-8"
    assert adapter.client.max_retries == 0
    # The distinctive region proves the supplied client reaches the adapter.
    # `AnthropicBedrock.aws_region` is None.
    assert adapter.client.aws_region == "eu-west-1"


def test_bedrock_model_rejects_a_client_whose_class_does_not_serve_the_models_api() -> None:
    """Reject a legacy client for a mantle model."""
    legacy_client = AsyncAnthropicBedrock(aws_region="us-east-1")
    with pytest.raises(ValueError, match=re.escape("anthropic.claude-sonnet-5")) as excinfo:
        _ = AnthropicBedrock(client=legacy_client).model("anthropic.claude-sonnet-5")
    assert "AsyncAnthropicBedrockMantle" in str(excinfo.value)


def test_bedrock_model_accepts_custom_pricing_and_a_passed_client() -> None:
    """A passed client serves an uncataloged model with stated pricing."""
    model = "us.anthropic.claude-next"
    adapter = _anthropic_adapter_of(
        AnthropicBedrock(client=AsyncAnthropicBedrockMantle(aws_region="us-east-1")).model(
            model,
            pricing=_PRICING,
        )
    )
    assert adapter.model == model
    assert adapter.pricing is _PRICING


def test_uncataloged_bedrock_model_requires_a_passed_client() -> None:
    """An uncataloged model cannot select `api`."""
    with pytest.raises(ValueError, match="pass client="):
        _ = AnthropicBedrock(aws_region="us-east-1").model(
            "us.anthropic.claude-next",
            pricing=_PRICING,
        )


def test_uncataloged_bedrock_model_requires_pricing() -> None:
    """An uncataloged model has no default pricing."""
    bedrock = AnthropicBedrock(client=AsyncAnthropicBedrockMantle(aws_region="us-east-1"))
    with pytest.raises(ValueError, match="pass pricing="):
        _ = bedrock.model("us.anthropic.claude-next")


def test_bedrock_preferred_model_names_equal_anthropic_bedrock_keys() -> None:
    """`AnthropicBedrockModelName` literals equal `ANTHROPIC_BEDROCK` keys."""
    preferred_model_names, accepted_string_type = get_args(AnthropicBedrockModelName.__value__)
    assert accepted_string_type is str
    assert set(ANTHROPIC_BEDROCK) == set(get_args(preferred_model_names))


def test_wire_messages_marks_a_marked_user_part() -> None:
    """A user part with cache_breakpoint carries the marker on its own block. Unmarked siblings carry none."""
    messages = [
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                TextPart(text="question"),
            )
        ),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    blocks = _content_blocks(wire[0])
    assert blocks[0]["type"] == "text"
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]


def test_wire_messages_marks_a_marked_image_part() -> None:
    """An image part with cache_breakpoint carries the marker on its image block."""
    messages = [
        UserMessage(
            content=(ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),)
        ),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    image_block = _content_blocks(wire[0])[0]
    assert image_block["type"] == "image"
    assert image_block.get("cache_control") == {"type": "ephemeral"}


def test_wire_messages_marks_the_tool_result_block_for_a_marked_last_tool_part() -> None:
    """A marked last part of a ToolMessage marks the enclosing tool_result block, never a nested block."""
    messages = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="a"), TextPart(text="b", cache_breakpoint=True)),
        )
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    tool_result = _content_blocks(wire[0])[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("cache_control") == {"type": "ephemeral"}
    assert tool_result.get("content") == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_wire_messages_rejects_a_marked_non_last_tool_part() -> None:
    """A marked part before the ToolMessage's last is rejected instead of silently moving the boundary."""
    messages = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="a", cache_breakpoint=True), TextPart(text="b")),
        )
    ]
    with pytest.raises(_NotSendableError, match="last part"):
        _ = _wire_messages(
            messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
        )


def test_build_request_reports_an_unsendable_sequence_as_invalid_request() -> None:
    """An unsendable Sequence[Message] reaches build_request's caller as the InvalidRequest variant.

    Nothing is sent: the retry loop takes this answer before its first attempt.
    """
    messages = [UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),))]
    outcome = _structured_bound().build_request(messages)
    assert isinstance(outcome, InvalidRequest)
    assert "image/tiff" in outcome.reason


def test_build_request_reports_an_unparseable_args_json_as_invalid_request() -> None:
    """Malformed replayed args_json returns InvalidRequest."""
    messages = [
        AssistantMessage(turn=(ToolCall(id="c1", name="f", args_json="not json"),)),
        ToolMessage(tool_call_id="c1", content="ok"),
    ]
    outcome = _structured_bound().build_request(messages)
    assert isinstance(outcome, InvalidRequest)
    assert "args_json" in outcome.reason


def test_build_request_reports_a_stored_payload_naming_no_type_as_invalid_request() -> None:
    """RawPart.raw without type produces no content block."""
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    bound_adapter = _BoundAnthropicText(adapter=adapter, precomputed_fields=precomputed_fields)
    messages = [
        UserMessage(content="q"),
        AssistantMessage(turn=(RawPart(raw={"parts": [{"text": "from elsewhere"}]}),)),
    ]
    outcome = bound_adapter.build_request(messages)
    assert isinstance(outcome, InvalidRequest)
    assert "type key" in outcome.reason


def test_a_built_request_renders_as_json_carrying_the_prompt_and_no_omitted_field() -> None:
    """as_json holds the binding's precomputed fields and this call's converted messages.

    An unstated temperature is absent from the request.
    """
    request = _structured_bound().build_request([UserMessage(content="hi")])
    assert isinstance(request, _AnthropicRequestParams)
    rendered = json.loads(request.as_json())
    assert rendered["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert rendered["precomputed"]["model"] == "m"
    assert "temperature" not in rendered["precomputed"]


def test_wire_messages_writes_only_the_latest_four_marks_without_automatic_cache_breakpoints() -> (
    None
):
    """Five marks spend the 4-marker request budget on the latest four. The oldest goes unwritten."""
    messages = [
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(5))
        ),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    blocks = _content_blocks(wire[0])
    assert "cache_control" not in blocks[0]
    assert all(block["type"] == "text" for block in blocks)
    assert all(block.get("cache_control") == {"type": "ephemeral"} for block in blocks[1:])


def test_wire_messages_reserves_markers_for_automatic_cache_breakpoints() -> None:
    """`automatic_cache_breakpoints=True` leaves room for the latest explicit marks."""
    messages = [
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(3))
        ),
        UserMessage(content="question"),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=True, cache_ttl="5m", message_mark_budget=2
    )
    marked_blocks = _content_blocks(wire[0])
    assert "cache_control" not in marked_blocks[0]
    assert all(block["type"] == "text" for block in marked_blocks)
    assert all(block.get("cache_control") == {"type": "ephemeral"} for block in marked_blocks[1:])
    last_block = _content_blocks(wire[1])[-1]
    assert last_block["type"] == "text"
    assert last_block.get("cache_control") == {"type": "ephemeral"}


def test_request_renders_system_parts_with_marks_and_the_automatic_last_block_marker() -> None:
    """A parts system_prompt is one block per part. Marked parts and the automatic last block carry markers."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            ),
            tool_schemas=(),
            automatic_cache_breakpoints=True,
        )
    )
    assert precomputed_fields.system == [
        {"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "semi-stable context", "cache_control": {"type": "ephemeral"}},
    ]
    assert precomputed_fields.message_mark_budget == 1


def test_request_system_parts_without_automatic_cache_breakpoints_mark_only_marked_parts() -> None:
    """With `automatic_cache_breakpoints=False`, only `cache_breakpoint` writes a marker."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            system_prompt=(
                TextPart(text="stable", cache_breakpoint=True),
                TextPart(text="volatile"),
            ),
            tool_schemas=(),
            automatic_cache_breakpoints=False,
        )
    )
    assert precomputed_fields.system == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "volatile"},
    ]
    assert precomputed_fields.message_mark_budget == 3


def test_request_rejects_a_binding_whose_markers_exceed_the_request_limit() -> None:
    """Four marked system parts plus the automatic markers cannot fit the 4-marker limit."""
    with pytest.raises(ValueError, match="limit"):
        _ = _adapter()._precompute_fields(
            _binding(
                system_prompt=tuple(
                    TextPart(text=f"s{index}", cache_breakpoint=True) for index in range(4)
                ),
                tool_schemas=(),
                automatic_cache_breakpoints=True,
            )
        )


def test_request_str_system_leaves_a_message_mark_budget_of_two() -> None:
    """A str system prompt with `automatic_cache_breakpoints=True` leaves two message marks."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    assert precomputed_fields.message_mark_budget == 2
    uncached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=False)
    )
    assert uncached.message_mark_budget == 4


def test_wire_messages_budget_mixes_user_and_tool_result_marks_across_messages() -> None:
    """The latest-N budget counts marks across message kinds in message order."""
    messages = [
        UserMessage(content=(TextPart(text="oldest", cache_breakpoint=True),)),
        ToolMessage(tool_call_id="tu_1", content=(TextPart(text="mid", cache_breakpoint=True),)),
        UserMessage(content=(TextPart(text="latest", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=2
    )
    assert "cache_control" not in _content_blocks(wire[0])[0]
    tool_result = _content_blocks(wire[1])[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("cache_control") == {"type": "ephemeral"}
    latest_text = _content_blocks(wire[2])[0]
    assert latest_text["type"] == "text"
    assert latest_text.get("cache_control") == {"type": "ephemeral"}


def test_wire_messages_explicit_mark_on_the_last_block_coexists_with_the_automatic_marker() -> (
    None
):
    """An explicit mark on the last block and the automatic last-block marker write one identical marker."""
    messages = [
        UserMessage(content=(TextPart(text="q", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=True, cache_ttl="5m", message_mark_budget=2
    )
    assert _content_blocks(wire[0]) == [
        {"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}
    ]


def test_wire_messages_writes_no_marks_at_zero_budget() -> None:
    """A zero budget leaves every mark unwritten instead of slicing the whole list."""
    messages = [
        UserMessage(content=(TextPart(text="m", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=0
    )
    assert "cache_control" not in _content_blocks(wire[0])[0]


def _adapter_1h() -> AnthropicMessagesAdapter:
    """Build an adapter with the 1-hour cache TTL over a keyless client."""
    return AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
        cache_ttl="1h",
    )


def test_request_1h_ttl_writes_the_ttl_on_system_marks() -> None:
    """cache_ttl="1h" puts the explicit ttl key on the automatic system marker and flows into the request."""
    precomputed_fields = _adapter_1h()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    assert _block_list(precomputed_fields.system)[0].get("cache_control") == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert precomputed_fields.cache_ttl == "1h"


def test_request_1h_ttl_writes_the_ttl_on_the_last_tool_mark() -> None:
    """cache_ttl="1h" puts the explicit ttl key on the last-tool marker."""
    precomputed_fields = _adapter_1h()._precompute_fields(
        _binding(
            system_prompt=None, tool_schemas=_tool_schemas(), automatic_cache_breakpoints=True
        )
    )
    assert _block_list(precomputed_fields.tools)[-1].get("cache_control") == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_wire_messages_1h_ttl_writes_the_ttl_on_message_and_automatic_marks() -> None:
    """cache_ttl="1h" puts the explicit ttl key on cache_breakpoint marks and the automatic last-block marker."""
    messages = [
        UserMessage(content=(TextPart(text="context", cache_breakpoint=True),)),
        UserMessage(content="question"),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=True, cache_ttl="1h", message_mark_budget=2
    )
    first_block = _content_blocks(wire[0])[0]
    assert first_block["type"] == "text"
    assert first_block.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}
    last_block = _content_blocks(wire[1])[-1]
    assert last_block["type"] == "text"
    assert last_block.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_request_rejects_an_empty_tuple_system_prompt() -> None:
    """An empty parts tuple, reachable only via a directly constructed Binding, raises instead of IndexError."""
    with pytest.raises(ValueError, match="empty tuple"):
        _ = _adapter()._precompute_fields(
            _binding(system_prompt=(), tool_schemas=(), automatic_cache_breakpoints=True)
        )


def test_billing_reported_reports_nothing_until_the_first_event_and_the_snapshot_after() -> None:
    """billing_reported returns None before the first event and Billing after it."""

    async def scenario() -> tuple[ProviderBilling | None, ProviderBilling | None]:
        """Read billing before and after one stream item."""
        adapter_stream = _anthropic_stream(
            [_text_delta_event("he", 0)], _message_snapshot("end_turn")
        )
        before = adapter_stream.billing_reported()
        items = adapter_stream.items()
        await anext(items)
        return before, adapter_stream.billing_reported()

    before, after = asyncio.run(scenario())
    assert before is None
    assert after == _provider_billing_from_sdk_usage(
        at.Usage(input_tokens=1, output_tokens=1), _PRICING
    )


def _turn_content() -> list[at.ContentBlock]:
    """Build reasoning, server-tool-call, and text blocks.

    The reasoning block carries an extra raw field.
    The final text block receives the automatic cache marker.
    """
    return [
        at.ThinkingBlock.model_construct(
            type="thinking", thinking="check first", signature="sig-1", field_newer_than_sdk="x"
        ),
        at.ServerToolUseBlock(
            type="server_tool_use",
            id="srvtoolu_1",
            name="web_search",
            input={"query": "langchaint"},
        ),
        at.TextBlock(type="text", text="hello"),
    ]


def _turn_message(usage: at.Usage) -> at.Message:
    """Build a finished turn of _turn_content's blocks, billing the given usage."""
    return _message_with_content(_turn_content(), stop_reason="end_turn", usage=usage)


class TestAnthropicMessagesConformance(AdapterConformance):
    """The neutral invariants, over the Anthropic Messages adapter's own SDK objects."""

    @override
    def make_adapter(self) -> Adapter:
        """Build the adapter these invariants run against, priced for the standard tier alone."""
        return _adapter()

    @override
    def response_with_cache_writes(self) -> BaseModel:
        return _turn_message(_usage_with_cache_split())

    @override
    def response_without_usage(self) -> BaseModel:
        """Return a turn reporting zero everywhere, anthropic's Message requiring a usage object."""
        return _turn_message(at.Usage(input_tokens=0, output_tokens=0))

    @override
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return a turn served at priority, which _PRICING holds no table for."""
        return _turn_message(
            at.Usage(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=30,
                service_tier="priority",
            )
        )

    @override
    def response_with_impossible_counters(self) -> BaseModel:
        """Return a turn reporting a negative output counter."""
        return _turn_message(at.Usage(input_tokens=1, output_tokens=-1))

    @override
    def response_with_reasoning(self) -> BaseModel:
        """Return a turn whose thinking block carries the unnamed key."""
        return _turn_message(_usage_with_cache_split())

    @override
    def response_with_raw_part(self) -> BaseModel | None:
        """Return the turn whose middle block is a server tool call."""
        return _turn_message(_usage_with_cache_split())

    @override
    def assistant_wire_parts(self, request: RequestParams) -> Sequence[object]:
        """Read the content blocks of the assistant message this request ends with."""
        assert isinstance(request, _AnthropicRequestParams)
        return _content_blocks(request.messages[-1])

    @override
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return the same turn as the ParsedMessage a stream assembles into and as a Message."""
        whole = _turn_message(_usage_with_cache_split())
        return ParsedMessage[None].model_validate(whole.model_dump()), whole

    @override
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream whose accumulated message ends with no stop reason."""
        return _anthropic_stream([_text_delta_event("he", 0)], _message_snapshot(None))

    @override
    def sdk_errors_and_classifications(self) -> Mapping[Exception, ErrorClassification]:
        """Return Anthropic error classification cases."""
        return {
            _connection_error(): "transient",
            anthropic.APITimeoutError(
                httpx2.Request("POST", "https://api.anthropic.com")
            ): "transient",
            anthropic.RetryableError("middleware said retry"): "transient",
            _status_error(anthropic.RateLimitError, 429): "invalid_request",
            _status_error(anthropic.ConflictError, 409): "invalid_request",
            _status_error(anthropic.BadRequestError, 400): "invalid_request",
            _status_error(anthropic.AuthenticationError, 401): "invalid_request",
            _status_error(anthropic.PermissionDeniedError, 403): "invalid_request",
            _status_error(anthropic.NotFoundError, 404): "invalid_request",
            _status_error(anthropic.RequestTooLargeError, 413): "invalid_request",
            _status_error(anthropic.UnprocessableEntityError, 422): "invalid_request",
            _status_error(anthropic.APIStatusError, 402): "invalid_request",
            _status_error(anthropic.APIStatusError, 408): "invalid_request",
            _status_error(
                anthropic.BadRequestError, 400, {"x-should-retry": "false"}
            ): "invalid_request",
            _status_error(
                anthropic.InternalServerError, 500, {"x-should-retry": "false"}
            ): "declared_final",
            _status_error(anthropic.OverloadedError, 529): "unknown_exception",
            _status_error(anthropic.InternalServerError, 500): "unknown_exception",
            _status_error(anthropic.InternalServerError, 503): "unknown_exception",
            _status_error(anthropic.APIStatusError, 302): "unknown_exception",
            _status_error(
                anthropic.APIStatusError, 200, error_type="invalid_request_error"
            ): "declared_final",
            ValueError("boom"): "unknown_exception",
        }

    @override
    def sdk_errors_and_verdicts(self) -> Mapping[Exception, Verdict]:
        """Return Anthropic error verdict cases."""
        return {
            _status_error(
                anthropic.RateLimitError, 429, {"retry-after": "7"}, "rate_limit_error"
            ): PauseAll(retry_after=7.0),
            _status_error(anthropic.OverloadedError, 529, error_type="overloaded_error"): PauseAll(
                retry_after=None
            ),
            _status_error(
                anthropic.InternalServerError, 500, error_type="api_error"
            ): RetryThisOne(retry_after=None),
            _status_error(
                anthropic.InternalServerError, 504, error_type="timeout_error"
            ): RetryThisOne(retry_after=None),
            _status_error(anthropic.APIStatusError, 408): RetryThisOne(retry_after=None),
            _status_error(anthropic.ConflictError, 409): RetryThisOne(retry_after=None),
            _status_error(
                anthropic.BadRequestError, 400, error_type="invalid_request_error"
            ): DoNotRetry(),
            _status_error(
                anthropic.AuthenticationError, 401, error_type="authentication_error"
            ): DoNotRetry(),
            _status_error(anthropic.APIStatusError, 402, error_type="billing_error"): DoNotRetry(),
            _status_error(
                anthropic.PermissionDeniedError, 403, error_type="permission_error"
            ): DoNotRetry(),
            _status_error(
                anthropic.NotFoundError, 404, error_type="not_found_error"
            ): DoNotRetry(),
            _status_error(
                anthropic.RequestTooLargeError, 413, error_type="request_too_large"
            ): DoNotRetry(),
            _status_error(anthropic.UnprocessableEntityError, 422): DoNotRetry(),
            _status_error(anthropic.InternalServerError, 503): RetryThisOne(retry_after=None),
            _status_error(anthropic.APIStatusError, 451): DoNotRetry(),
            _status_error(anthropic.InternalServerError, 502): RetryThisOne(retry_after=None),
            _status_error(
                anthropic.RateLimitError, 429, {"x-should-retry": "false"}, "rate_limit_error"
            ): PauseAllDoNotRetry(retry_after=None),
            TransientError(
                "throttled body", retry_after_seconds=3.0, is_rate_limit=True
            ): PauseAll(retry_after=3.0),
            TransientError("failed body"): RetryThisOne(retry_after=None),
        }
