"""Anthropic Messages adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot: usage partition arithmetic, the 5-minute/1-hour cache-write cost split,
stop-reason mapping, tool_use extraction, cache-breakpoint placement, tool-choice translation,
and the request fields the binding precomputes.
"""

import asyncio
import base64
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import get_args, override

import anthropic
import anthropic.types as at
import httpx
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
from anthropic.types import MessageParam, ParsedMessage
from anthropic.types.parsed_message import ParsedTextBlock
from pydantic import BaseModel, TypeAdapter

from langchaint import (
    LLM,
    AssistantMessage,
    Billing,
    ImagePart,
    InferenceParams,
    Message,
    OpaqueElement,
    PydanticTool,
    ReasoningDelta,
    ReasoningTrace,
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
    AdapterStream,
    Binding,
    ContextWindowExceeded,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutputOutcome,
    Refusal,
    RequestParams,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.anthropic import (
    ANTHROPIC_BEDROCK,
    ANTHROPIC_PRICING,
    AnthropicBedrockModelName,
    AnthropicMessagesAdapter,
    AnthropicPricedServiceTier,
    AnthropicPricingTable,
    anthropic_bedrock_model,
    anthropic_model,
)
from langchaint.anthropic.messages_adapter import (
    PARSE_FALLTHROUGH_COUNTS,
    _adapter_result,
    _AnthropicRequestParams,
    _AnthropicStream,
    _assistant_content_blocks,
    _assistant_message_from,
    _billing_from_sdk_usage,
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

_STANDARD_RATES = AnthropicPricingTable(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_5m_usd_per_million_tokens=3.75,
    cache_write_1h_usd_per_million_tokens=6.0,
)

_PRICING: dict[AnthropicPricedServiceTier, AnthropicPricingTable] = {"standard": _STANDARD_RATES}
"""The standard tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = AnthropicPricingTable(
    input_cache_none_usd_per_million_tokens=6.0,
    output_usd_per_million_tokens=30.0,
    cache_read_usd_per_million_tokens=0.6,
    cache_write_5m_usd_per_million_tokens=7.5,
    cache_write_1h_usd_per_million_tokens=12.0,
)
"""Twice the standard rates, so a tier-selection test reads as a doubling."""


def _as_dict(value: object) -> dict[str, object]:
    """View one wire TypedDict as a plain dict for structural assertions."""
    assert isinstance(value, dict)
    mapping: dict[object, object] = value
    return {str(key): item for key, item in mapping.items()}


def _content_blocks(message: MessageParam) -> list[dict[str, object]]:
    """Return one wire message's content blocks as plain dicts."""
    content = _as_dict(message)["content"]
    assert isinstance(content, list)
    blocks: list[object] = content
    return [_as_dict(block) for block in blocks]


def _block_list(value: object) -> list[dict[str, object]]:
    """View a wire block list (never the omit sentinel here) as plain dicts."""
    assert isinstance(value, list)
    blocks: list[object] = value
    return [_as_dict(block) for block in blocks]


class _EchoArgs(BaseModel):
    """Argument model for the test tool."""

    city: str


def _tool_schemas() -> tuple[ToolSchema, ...]:
    """Return the schemas of one tool named get_weather."""

    async def function(args: _EchoArgs) -> str:
        """Return the city unchanged; never called in these tests."""
        return args.city

    tool = PydanticTool(
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


def test_billing_partitions_input_counters_and_prices() -> None:
    """input_tokens is the uncached counter, and the Billing's usage carries the priced cost."""
    usage = _billing_from_sdk_usage(_usage_with_cache_split(), _PRICING).usage
    assert usage.input_tokens_cache_read == 200
    assert usage.input_tokens_cache_write == 30
    assert usage.input_tokens_cache_none == 100
    assert usage.input_tokens_total == 330
    assert usage.cost_in_usd == (100 * 3.0 + 200 * 0.3 + 10 * 3.75 + 20 * 6.0 + 50 * 15.0) / 1e6


def test_billing_carries_the_sdk_usage_object_itself() -> None:
    """usage_raw is the SDK usage object by reference, not a copy."""
    usage = _usage_with_cache_split()
    billing = _billing_from_sdk_usage(usage, _PRICING)
    assert billing.usage_raw is usage


def test_billing_pins_the_served_tier_and_the_rates_that_applied() -> None:
    """The Billing holds the served tier and the rates the tier's table charged."""
    billing = _billing_from_sdk_usage(_usage_with_cache_split(), _PRICING)
    assert billing.service_tier == "standard"
    assert billing.input_cache_none_usd_per_million_tokens == 3.0
    assert billing.cache_read_usd_per_million_tokens == 0.3
    assert billing.output_usd_per_million_tokens == 15.0


def test_an_unpriced_tier_keeps_its_counters_and_its_name() -> None:
    """A response served at a tier no table prices still reports what it billed and who served it."""
    billing = _billing_from_sdk_usage(
        at.Usage(input_tokens=100, output_tokens=50, service_tier="priority"), _PRICING
    )
    assert billing.service_tier == "priority"
    assert billing.usage.input_tokens_total == 100
    assert billing.usage.output_tokens == 50


def test_billing_counts_the_writes_it_bills_for() -> None:
    """The cache-write counters read the same source the cost does, so the two cannot disagree.

    cache_creation_input_tokens and the cache_creation split are separate optional SDK fields with
    no documented relationship, so a response carrying only the split would otherwise report a
    cost covering 30 written tokens and a counter saying none were written.
    """
    usage = _billing_from_sdk_usage(
        at.Usage(
            input_tokens=100,
            output_tokens=0,
            cache_creation=at.CacheCreation(
                ephemeral_5m_input_tokens=10, ephemeral_1h_input_tokens=20
            ),
        ),
        _PRICING,
    ).usage
    assert usage.input_tokens_cache_write == 30
    assert abs(usage.cost_in_usd - (100 * 3.0 + 10 * 3.75 + 20 * 6.0) / 1e6) < 1e-12


def test_billing_treats_none_cache_counts_as_zero() -> None:
    """Absent cache counters normalize to zero, not None."""
    usage = _billing_from_sdk_usage(at.Usage(input_tokens=7, output_tokens=3), _PRICING).usage
    assert usage.input_tokens_cache_read == 0
    assert usage.input_tokens_cache_write == 0
    assert usage.input_tokens_cache_none == 7


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


def test_the_stored_write_price_is_the_blend_of_what_the_two_ttls_billed() -> None:
    """A response mixing both write TTLs stores one write price, and it reproduces the write cost.

    Usage collapses the two write counters into one, so neither TTL's own rate reproduces the cost;
    the blend does, which is what lets a stored record reprice itself without the split counts.
    """
    billing = _billing_from_sdk_usage(_usage_with_cache_split(), _PRICING)
    usage = billing.usage
    assert usage.input_tokens_cache_write == 30
    assert usage.input_tokens_cache_write_cost_in_usd == pytest.approx(
        usage.input_tokens_cache_write * billing.cache_write_usd_per_million_tokens / 1e6
    )
    assert 3.75 < billing.cache_write_usd_per_million_tokens < 6.0


def test_equal_write_rates_store_that_rate_as_the_write_price() -> None:
    """With both TTLs priced alike the blend is that rate, so blending adds no artifact."""
    equal_write_rates = AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=3.0,
        output_usd_per_million_tokens=15.0,
        cache_read_usd_per_million_tokens=0.3,
        cache_write_5m_usd_per_million_tokens=3.75,
        cache_write_1h_usd_per_million_tokens=3.75,
    )
    billing = _billing_from_sdk_usage(_usage_with_cache_split(), {"standard": equal_write_rates})
    assert billing.cache_write_usd_per_million_tokens == pytest.approx(3.75)


def test_a_response_that_wrote_no_cache_stores_the_five_minute_write_rate() -> None:
    """With nothing written there is nothing to blend, so the write price is the default TTL's rate."""
    billing = _billing_from_sdk_usage(at.Usage(input_tokens=7, output_tokens=3), _PRICING)
    assert billing.usage.input_tokens_cache_write == 0
    assert billing.cache_write_usd_per_million_tokens == 3.75


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response, standard rates a response reporting no tier."""
    counters = {"input_tokens": 100, "output_tokens": 50}
    pricing: dict[AnthropicPricedServiceTier, AnthropicPricingTable] = {
        "standard": _STANDARD_RATES,
        "priority": _PRIORITY_RATES,
    }
    at_priority = _billing_from_sdk_usage(at.Usage(**counters, service_tier="priority"), pricing)
    at_standard = _billing_from_sdk_usage(at.Usage(**counters, service_tier="standard"), pricing)
    reporting_none = _billing_from_sdk_usage(at.Usage(**counters), pricing)
    assert at_priority.usage.cost_in_usd == pytest.approx(2 * at_standard.usage.cost_in_usd)
    assert reporting_none.usage.cost_in_usd == at_standard.usage.cost_in_usd


def test_the_categories_are_priced_apart() -> None:
    """Each stored cost is its own category's product, and they sum to cost_in_usd."""
    usage = _billing_from_sdk_usage(_usage_with_cache_split(), _PRICING).usage
    assert usage.input_tokens_cache_none_cost_in_usd == 100 * 3.0 / 1e6
    assert usage.input_tokens_cache_read_cost_in_usd == 200 * 0.3 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == (10 * 3.75 + 20 * 6.0) / 1e6
    assert usage.output_tokens_cost_in_usd == 50 * 15.0 / 1e6
    assert usage.cost_in_usd == pytest.approx(
        (100 * 3.0 + 200 * 0.3 + 10 * 3.75 + 20 * 6.0 + 50 * 15.0) / 1e6
    )


def test_pricing_without_the_standard_key_raises_at_construction() -> None:
    """A pricing mapping missing "standard" fails before any request, naming the model."""
    priority_only: dict[AnthropicPricedServiceTier, AnthropicPricingTable] = {
        "priority": _PRIORITY_RATES
    }
    with pytest.raises(ValueError, match=re.escape("'standard'")):
        _ = AnthropicMessagesAdapter(
            client=AsyncAnthropic(api_key="test"),
            model="claude-sonnet-5",
            pricing=priority_only,
            provider_name="anthropic",
        )


def test_the_catalog_prices_the_standard_tier_and_a_caller_adds_others() -> None:
    """anthropic_model merges the caller's mapping over the catalog's standard-tier table."""
    adapter = _anthropic_adapter_of(
        anthropic_model(
            "claude-sonnet-5",
            client=AsyncAnthropic(api_key="test"),
            pricing={"priority": _PRIORITY_RATES},
        )
    )
    assert adapter.pricing["standard"] is ANTHROPIC_PRICING["claude-sonnet-5"]
    assert adapter.pricing["priority"] is _PRIORITY_RATES


def test_cache_ttl_is_stored_on_the_adapter() -> None:
    """Both constructors carry cache_ttl through to the adapter that writes the markers."""
    assert (
        _anthropic_adapter_of(
            anthropic_model(
                "claude-sonnet-5", client=AsyncAnthropic(api_key="test"), cache_ttl="1h"
            )
        ).cache_ttl
        == "1h"
    )
    assert (
        _anthropic_adapter_of(
            anthropic_bedrock_model(
                "anthropic.claude-sonnet-5", aws_region="us-east-1", cache_ttl="1h"
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
    """Recognized stop reasons map to their neutral name; everything else becomes other.

    The overflow is the one that is renamed rather than passed through, and it must not fall to
    "other": on a text binding, stop_reason is the only signal the caller gets for it.
    """
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
    assert reasoning_trace.raw == {
        "type": "thinking",
        "thinking": "check first",
        "signature": "sig-1",
    }
    assert reasoning_trace.text == "check first"
    assert assistant_message.text == "hello"
    assert assistant_message.tool_calls == (
        ToolCall(id="tu_1", name="get_weather", args_json='{"city": "Nairobi"}'),
    )
    blocks = _assistant_content_blocks(assistant_message)
    assert len(blocks) == len(message.content)
    assert blocks[0] == reasoning_trace.raw
    assert blocks[1] == {"type": "text", "text": "hello"}
    assert blocks[2] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "get_weather",
        "input": {"city": "Nairobi"},
    }


def test_empty_thinking_text_normalizes_to_none() -> None:
    """A thinking block whose text is "" yields text None, the single text-free condition.

    The openai adapter cannot produce "", so storing it here would make a text-free trace
    two values to test for, one of them provider-specific.
    """
    message = _message_with_content([
        at.ThinkingBlock(type="thinking", thinking="", signature="sig")
    ])
    reasoning_trace = _assistant_message_from(message).turn[0]
    assert isinstance(reasoning_trace, ReasoningTrace)
    assert reasoning_trace.text is None
    assert reasoning_trace.raw["thinking"] == ""


def test_redacted_thinking_round_trips_routed_by_its_type_key() -> None:
    """A redacted_thinking block round-trips as its own dump; the type key routes it on the wire.

    Its trace carries no text: the block holds an opaque string under data and nothing readable.
    """
    message = _message_with_content([
        at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes")
    ])
    assistant_message = _assistant_message_from(message)
    reasoning_trace = assistant_message.turn[0]
    assert isinstance(reasoning_trace, ReasoningTrace)
    assert reasoning_trace.text is None
    assert _assistant_content_blocks(assistant_message) == [
        {"type": "redacted_thinking", "data": "opaque-bytes"}
    ]


def test_a_server_tool_block_becomes_an_opaque_element_and_replays_as_itself() -> None:
    """A block this adapter has no variant for reaches the turn and goes back unchanged.

    The response was billed for that block, so dropping it would destroy output the caller paid for
    and leave a tool loop continuing from a turn the model did not produce.
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
    (opaque_element,) = assistant_message.turn
    assert isinstance(opaque_element, OpaqueElement)
    assert _assistant_content_blocks(assistant_message) == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "langchaint"},
        }
    ]


def test_produced_traces_survive_the_message_json_round_trip() -> None:
    """A produced turn holding thinking and redacted_thinking traces re-validates equal from JSON.

    Persist/resume serializes a Sequence[Message] with a TypeAdapter,
    and a raw that came back changed is a request the API rejects,
    so the round trip must restore each trace's dump exactly.
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
    """An openai-produced trace emits its dict as-is; the API rejects the unknown type key, not this adapter."""
    raw = {"type": "reasoning", "id": "rs_1"}
    assistant_message = AssistantMessage(turn=(ReasoningTrace(raw=raw), TextPart(text="hi")))
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
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    assert [message["role"] for message in wire] == ["user", "assistant", "user"]
    tool_results = _content_blocks(wire[2])
    assert len(tool_results) == 2
    assert tool_results[0]["is_error"] is False
    assert tool_results[1]["is_error"] is True


def test_wire_messages_marks_only_the_last_block_when_caching() -> None:
    """The per-request breakpoint lands on the last block of the last message."""
    messages = [
        ToolMessage(tool_call_id="tu_1", content="r1", is_error=False),
        ToolMessage(tool_call_id="tu_2", content="r2", is_error=True),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
    tool_results = _content_blocks(wire[0])
    assert tool_results[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tool_results[0]


def test_wire_messages_writes_no_breakpoint_on_a_thinking_last_block() -> None:
    """A Sequence[Message] ending on a thinking block writes no breakpoint that request.

    The thinking wire params carry no cache_control key, so the marker has nowhere valid to go.
    """
    messages = [
        AssistantMessage(
            turn=(
                TextPart(text="t"),
                ReasoningTrace(raw={"type": "thinking", "thinking": "x", "signature": "s"}),
            )
        )
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
    assert all("cache_control" not in block for block in _content_blocks(wire[0]))


def test_wire_messages_writes_no_breakpoint_when_caching_disabled() -> None:
    """With caching off, no block anywhere carries a cache_control marker."""
    messages = [
        UserMessage(content="hi"),
        ToolMessage(tool_call_id="tu_1", content="r1"),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
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
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
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
    """A tool_result image media type outside the accepted set is not sendable."""
    messages = [
        ToolMessage(tool_call_id="tu_1", content=(ImagePart(data=b"x", media_type="image/tiff"),))
    ]
    with pytest.raises(_NotSendableError, match="image/tiff"):
        _ = _wire_messages(
            messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
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

    disable_parallel_tool_use is the inverse of the neutral parallel_tool_calls, so a form that
    passed it through unchanged would ask for the opposite of what the caller stated.
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


def _binding(
    *,
    system_prompt: str | tuple[TextPart, ...] | None,
    tool_schemas: tuple[ToolSchema, ...],
    automatic_prompt_caching: bool,
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    """Assemble a binding with the fields these request tests vary."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        tool_choice="required",
        parallel_tool_calls=False,
        inference_params=InferenceParams(reasoning_effort="high"),
        automatic_prompt_caching=automatic_prompt_caching,
        extra_body=extra_body,
    )


def test_request_omits_tool_sentinels_without_tools() -> None:
    """No tools leaves both tools and tool_choice at the omit sentinel."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert precomputed_fields.max_tokens == 4096
    assert isinstance(precomputed_fields.tools, anthropic.Omit)
    assert isinstance(precomputed_fields.tool_choice, anthropic.Omit)
    assert precomputed_fields.output_config == {"effort": "high"}
    assert precomputed_fields.thinking == {"type": "adaptive"}


def test_request_passes_widened_reasoning_effort_through() -> None:
    """A value outside anthropic's own effort literal ("minimal") reaches the request unchanged."""
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(reasoning_effort="minimal"),
        automatic_prompt_caching=False,
    )
    precomputed_fields = _adapter()._precompute_fields(binding)
    assert precomputed_fields.output_config == {"effort": "minimal"}
    assert precomputed_fields.thinking == {"type": "adaptive"}


def test_request_omits_thinking_and_output_config_without_reasoning_effort() -> None:
    """A None reasoning_effort leaves both output_config and thinking at the omit sentinel."""
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(),
        automatic_prompt_caching=False,
    )
    precomputed_fields = _adapter()._precompute_fields(binding)
    assert isinstance(precomputed_fields.output_config, anthropic.Omit)
    assert isinstance(precomputed_fields.thinking, anthropic.Omit)


def test_request_maps_temperature_and_omits_it_when_unset() -> None:
    """A bound temperature lands on the request; None leaves the omit sentinel."""
    unset = _adapter()._precompute_fields(
        _binding(system_prompt=None, tool_schemas=(), automatic_prompt_caching=False)
    )
    assert isinstance(unset.temperature, anthropic.Omit)
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(temperature=0.2),
        automatic_prompt_caching=False,
    )
    assert _adapter()._precompute_fields(binding).temperature == 0.2


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request; None leaves the omit sentinel.

    The sentinel is what keeps an unstated tier off the wire: sending an explicit null would be a
    different request from omitting the key.
    """
    binding = _binding(system_prompt=None, tool_schemas=(), automatic_prompt_caching=False)
    assert isinstance(_adapter()._precompute_fields(binding).service_tier, anthropic.Omit)
    stated = AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
        service_tier="standard_only",
    )
    assert stated._precompute_fields(binding).service_tier == "standard_only"


def test_request_marks_the_system_block_only_when_caching() -> None:
    """The system block carries a breakpoint under caching and none without it."""
    cached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert _block_list(cached.system)[0]["cache_control"] == {"type": "ephemeral"}
    uncached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    assert "cache_control" not in _block_list(uncached.system)[0]


def test_request_marks_last_tool_only_without_a_system_prompt() -> None:
    """The prefix breakpoint sits on the last tool only when no system prompt follows."""
    schemas = _tool_schemas()
    without_system = _adapter()._precompute_fields(
        _binding(system_prompt=None, tool_schemas=schemas, automatic_prompt_caching=True)
    )
    assert _block_list(without_system.tools)[-1]["cache_control"] == {"type": "ephemeral"}
    with_system = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=schemas, automatic_prompt_caching=True)
    )
    assert "cache_control" not in _block_list(with_system.tools)[-1]


def test_user_content_blocks_rejects_unsupported_image_media_type() -> None:
    """An image media type outside the accepted set is not sendable."""
    message = UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),))
    with pytest.raises(_NotSendableError):
        _ = _user_content_blocks(message)


class _FakeSDKMessageStream(AsyncMessageStream[None]):
    """Replays constructed events without a connection.

    Overrides exactly the surface _AnthropicStream uses (iteration, close,
    the current_message_snapshot the stop-reason check reads, and the response its headers are read
    off); the base __init__ is deliberately not called,
    so the untouched base machinery stays unusable.
    The inherited request_id property is left in place, so a test of it exercises the SDK's own read.
    """

    def __init__(  # pyrefly: ignore[missing-super-call]
        self,
        replay_events: Sequence[ParsedMessageStreamEvent],
        message_snapshot: ParsedMessage[None],
        headers: dict[str, str] | None = None,
    ) -> None:
        self._replay_events = list(replay_events)
        self._message_snapshot = message_snapshot
        self._http_response = httpx.Response(
            200,
            headers=headers,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

    @property
    @override
    def response(self) -> httpx.Response:
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
    """The stream's own response is the only channel a streamed turn has for the header.

    The message the SDK assembles from the events never carries it, so a null here would leave every
    streaming call with no id to take to provider support.
    """
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
    """Drain the translated items into a list; None means a bare end_turn snapshot."""
    snapshot = message_snapshot if message_snapshot is not None else _message_snapshot("end_turn")

    async def scenario() -> list[StreamItem]:
        adapter_stream = _anthropic_stream(replay_events, snapshot)
        return [item async for item in adapter_stream.items()]

    return asyncio.run(scenario())


def test_stream_yields_bare_text_argument_fragments_and_one_complete_tool_call() -> None:
    """Text deltas pass through as the SDK's own strings.

    Each non-empty argument fragment yields a ToolCallDelta carrying the id and name the
    snapshot's tool_use block holds, and their concatenation is the completed call's args_json.
    A closing tool_use block yields one complete ToolCall whose args_json is the JSON text of the SDK-accumulated input;
    empty argument fragments and text block closes yield nothing.
    """

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
    """A block's stop event puts a blank line before the next block's first delta.

    A turn's reasoning arrives as one or more thinking blocks, and the break between two of them is
    a block boundary the API never sends as text, so deltas concatenated without it run the two
    blocks together.
    """
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

    Answer text following the block stop is untouched: only a thinking delta consumes a pending
    separator.
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
    """The separator armed before a dropped delta still falls before the next block's text."""
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

    The preceding block streamed nothing, so no reasoning text precedes the redacted one and the
    stream opens on the next block's own text.
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
        reasoning_trace = assistant_message.turn[0]
        assert isinstance(reasoning_trace, ReasoningTrace)
        assert reasoning_trace.raw == {
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
    """Build a structured-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    return _BoundAnthropicStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )


def _structured_parse(message: at.Message) -> _StructuredReport | None | NoOutputOutcome:
    """Run the structured binding's parse over one message, with the turn that message carries."""
    return _structured_bound()._parsed_output(message, _assistant_message_from(message))


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

    A message the SDK did not parse from an HTTP response body carries no request id, which is the
    state every streamed message is in.
    """
    identity = _structured_bound().identity_from_raw(_message_with_content([]))
    assert identity == ResponseIdentity(
        model_served="claude-sonnet-5", response_id="msg_1", request_id=None
    )


def test_identity_reads_the_request_id_the_sdk_attached_to_the_message() -> None:
    """A message parsed from a response body carries the request-id header, which identity reports.

    The assignment is what the SDK's own add_request_id does to every model it parses from a body
    (anthropic 0.120.0).
    """
    message = _message_with_content([])
    message._request_id = "req_anthropic"
    identity = _structured_bound().identity_from_raw(message)
    assert identity.request_id == "req_anthropic"


def _structured_message(
    text: str | None,
    stop_reason: at.StopReason | None = "end_turn",
) -> at.Message:
    """Build a message whose first text block carries the given text; None gives no text block."""
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
    """output_config carries the schema messages.parse(output_format=Model) would have sent, beside the bound effort.

    The adapter sends the schema itself so it can validate the response text in its own frame; this
    is what keeps the request unchanged by that move, the binding's own effort key included.
    """
    adapted_type: TypeAdapter[_StructuredReport] = TypeAdapter(_StructuredReport)
    assert _structured_bound()._precomputed_fields.output_config == {
        "effort": "high",
        "format": {"schema": transform_schema(adapted_type.json_schema()), "type": "json_schema"},
    }


def test_the_structured_request_sends_the_output_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_stream puts the precomputed output_config on the request.

    A request carrying the binding's own output_config instead would ask for no schema at all, and
    every turn would come back as prose the response_format rejects, reported as the caller's model
    being wrong rather than as the request that omitted its schema.
    """
    structured_bound = _structured_bound()
    output_config = _kwarg_sent(monkeypatch, structured_bound, "output_config")
    assert output_config == structured_bound._precomputed_fields.output_config


def test_request_rejects_an_extra_body_key_the_adapter_populates() -> None:
    """An extra_body key that open_stream passes as its own keyword raises at bind time.

    The SDK merges extra_body over the named request parameters with extra_body winning,
    so admitting the key would silently override the binding.
    """
    with pytest.raises(ValueError, match="max_tokens"):
        _ = _adapter()._precompute_fields(
            _binding(
                system_prompt=None,
                tool_schemas=(),
                automatic_prompt_caching=True,
                extra_body={"max_tokens": 10},
            )
        )


def test_the_request_sends_extra_body_by_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_stream passes the binding's extra_body to the SDK's extra_body parameter.

    A request that dropped it would silently go out without the caller's wire fields,
    which no offline round-trip test can catch.
    """
    adapter = _adapter()
    extra_body = {"top_k": 5}
    text_bound = _BoundAnthropicText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(
            _binding(
                system_prompt=None,
                tool_schemas=(),
                automatic_prompt_caching=True,
                extra_body=extra_body,
            )
        ),
    )
    assert _kwarg_sent(monkeypatch, text_bound, "extra_body") is extra_body


def test_structured_bind_validates_the_turns_text_into_the_instance() -> None:
    """The structured bound adapter validates the turn's text block into the response_format."""
    outcome = _structured_parse(_structured_message(_REPORT_JSON))
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)


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

    validation_error_json names the rejected field, what rejected it, and the value, which is what
    tells a caller whether to change the model or the prompt.
    """
    outcome = _structured_parse(_structured_message('{"city": "Nairobi", "celsius": "SENTINEL"}'))
    assert isinstance(outcome, SchemaViolation)
    rejections = json.loads(outcome.validation_error_json)
    assert [rejection["loc"] for rejection in rejections] == [["celsius"]]
    assert rejections[0]["input"] == "SENTINEL"


def test_structured_bind_reports_max_completion_tokens_exceeded_on_text_cut_mid_json() -> None:
    """A max_tokens turn whose JSON stopped mid-object is the truncation, not a schema violation.

    This is the response the SDK's own parse raised on; reporting it as a variant is what lets the
    retry loop fail the item with MaxCompletionTokensExceededError against the attempt it recorded.
    """
    outcome = _structured_parse(_structured_message('{"city": "Nair', stop_reason="max_tokens"))
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_bind_reports_a_tool_use_turn_as_none() -> None:
    """A tool_use turn parses no instance and nothing went wrong, so the output is None."""
    assert _structured_parse(_structured_message(None, stop_reason="tool_use")) is None


def test_structured_bind_reports_a_tool_use_turn_whose_text_is_not_the_instance_as_none() -> None:
    """A tool_use turn whose text block is prose is the tool call, not a schema violation."""
    outcome = _structured_parse(_structured_message("let me look that up", stop_reason="tool_use"))
    assert outcome is None


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
    """Build the SDK's 429 exception around a constructed httpx response."""
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def test_parse_anthropic_reads_retry_after_from_the_headers_without_letting_it_pick() -> None:
    """A retry-after header fills the verdict's retry_after and never changes which verdict.

    The header parsing itself is tested in tests/test_adapter.py against the shared function;
    what is provider-specific is where the headers are found. httpx.Headers is case-insensitive,
    so the lookup keeps working whatever case the server sent.
    """
    assert parse_anthropic(_rate_limit_error({"Retry-After-MS": "1500"})) == PauseAll(
        retry_after=1.5
    )
    assert parse_anthropic(_rate_limit_error({})) == PauseAll(retry_after=None)
    bad_request = _status_error(anthropic.BadRequestError, 400, {"retry-after": "7"})
    assert parse_anthropic(bad_request) == DoNotRetry()


def test_parse_anthropic_counts_a_fallthrough_and_a_listed_row_adds_nothing() -> None:
    """An unlisted status lands one tagged count; a listed status leaves the counter alone."""
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
    """A rate-limit or overload error type pauses the domain whatever status carried it."""
    overloaded = _status_error(anthropic.APIStatusError, 418, error_type="overloaded_error")
    assert parse_anthropic(overloaded) == PauseAll(retry_after=None)


def test_parse_anthropic_obeys_a_retry_directive_over_the_status_tables() -> None:
    """x-should-retry overrides the table verdict, which is what both SDK clients do with it.

    "false" on a 500 stops a status the table retries, and "true" on a 400 retries one it stops.
    "true" over a pausing verdict leaves it pausing: the directive speaks for this request, and
    dropping the pause would leave every sibling sending into the same rate limit.
    """
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


def test_parse_anthropic_pauses_the_domain_on_a_directive_that_gives_this_request_up() -> None:
    """A "false" directive over a pausing verdict stops this request and still pauses the domain.

    The 429 says the account is throttled and the directive says this request will not come back,
    which is the state PauseAllDoNotRetry carries and neither PauseAll nor DoNotRetry can.
    The second row is at status 418, in no table: the pause comes from the error type there, so a
    guard written on the status instead of the verdict would drop it.
    """
    throttled = _status_error(
        anthropic.RateLimitError, 429, {"x-should-retry": "false", "retry-after": "7"}
    )
    assert parse_anthropic(throttled) == PauseAllDoNotRetry(retry_after=7.0)
    overloaded = _status_error(
        anthropic.APIStatusError, 418, {"x-should-retry": "false"}, "overloaded_error"
    )
    assert parse_anthropic(overloaded) == PauseAllDoNotRetry(retry_after=None)


def test_parse_anthropic_ignores_a_retry_directive_on_the_streams_200_status() -> None:
    """A 200's headers belong to a request the provider accepted, so they judge no failure.

    The failure is a mid-stream error event raised on that live response, which the SDK never
    consults its retry predicate about, so the error type alone decides.
    """
    overloaded = _status_error(
        anthropic.APIStatusError, 200, {"x-should-retry": "false"}, "overloaded_error"
    )
    assert parse_anthropic(overloaded) == PauseAll(retry_after=None)
    rejected = _status_error(
        anthropic.APIStatusError, 200, {"x-should-retry": "true"}, "invalid_request_error"
    )
    assert parse_anthropic(rejected) == DoNotRetry()


def test_parse_anthropic_retries_a_transient_type_at_the_streams_200_status() -> None:
    """api_error and timeout_error retry at any status, counted where the status is unlisted.

    A mid-stream error event raises carrying the live response's 200 status, so the error type is
    the failure's one signal; the count is what keeps the odd status-type pair visible.
    """
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    for transient_type in ("api_error", "timeout_error"):
        failed = _status_error(anthropic.APIStatusError, 200, error_type=transient_type)
        assert parse_anthropic(failed) == RetryThisOne(retry_after=None)
        tag = f"status=200 type={transient_type}"
        assert PARSE_FALLTHROUGH_COUNTS[tag] == before.get(tag, 0) + 1


def test_request_id_from_error_reads_the_sdk_errors_own_header_and_nothing_else() -> None:
    """The override reports the header the SDK read off the error response, None for any other error.

    anthropic sends the id in request-id; a response without that header and an exception that never
    reached one both give None.
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
    """Build one of the SDK's status exceptions around a constructed httpx response.

    error_type fills the body's error.type the way the SDK reads it onto the exception; None
    builds the exception a non-JSON body produces, whose type attribute is None.
    """
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        headers=headers,
    )
    body = None if error_type is None else {"error": {"type": error_type, "message": "boom"}}
    return error_class("boom", response=response, body=body)


def _connection_error() -> anthropic.APIConnectionError:
    """Build the SDK's transport-failure exception, which carries a request and no response."""
    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
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
    adapter = _anthropic_adapter_of(anthropic_bedrock_model(model, aws_region="us-east-1"))
    assert adapter.model == model
    assert isinstance(adapter.client, expected_client_class)
    assert adapter.client.max_retries == 0


def test_bedrock_model_shares_the_first_party_pricing_object() -> None:
    """The Bedrock standard-tier table is the same object anthropic_model uses, not a copy."""
    adapter = _anthropic_adapter_of(
        anthropic_bedrock_model("us.anthropic.claude-opus-4-6-v1", aws_region="us-east-1")
    )
    assert adapter.pricing["standard"] is ANTHROPIC_PRICING["claude-opus-4-6"]


def test_bedrock_model_uses_a_matching_supplied_client() -> None:
    """A supplied client whose class serves the model's Bedrock API passes through, retries pinned off."""
    adapter = _anthropic_adapter_of(
        anthropic_bedrock_model(
            "anthropic.claude-opus-4-8", client=AsyncAnthropicBedrockMantle(aws_region="eu-west-1")
        )
    )
    assert isinstance(adapter.client, AsyncAnthropicBedrockMantle)
    assert adapter.model == "anthropic.claude-opus-4-8"
    assert adapter.client.max_retries == 0
    # aws_region survives with_options; the supplied client's distinctive region proves it is the one
    # used, not a default rebuilt from the constructor's own aws_region (which is None here).
    assert adapter.client.aws_region == "eu-west-1"


def test_bedrock_model_rejects_a_client_whose_class_does_not_serve_the_models_api() -> None:
    """A legacy client for a mantle-only model fails at construction, naming the model and required class."""
    legacy_client = AsyncAnthropicBedrock(aws_region="us-east-1")
    with pytest.raises(ValueError, match=re.escape("anthropic.claude-sonnet-5")) as excinfo:
        _ = anthropic_bedrock_model("anthropic.claude-sonnet-5", client=legacy_client)
    assert "AsyncAnthropicBedrockMantle" in str(excinfo.value)


def test_bedrock_table_is_total_over_the_bedrock_ids() -> None:
    """Every AnthropicBedrockModelName has a routing entry, so a new Bedrock wire model id must add one."""
    assert set(ANTHROPIC_BEDROCK) == set(get_args(AnthropicBedrockModelName.__value__))


def test_wire_messages_marks_a_marked_user_part() -> None:
    """A user part with cache_breakpoint carries the marker on its own block; unmarked siblings carry none."""
    messages = [
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                TextPart(text="question"),
            )
        ),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    blocks = _content_blocks(wire[0])
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]


def test_wire_messages_marks_a_marked_image_part() -> None:
    """An image part with cache_breakpoint carries the marker on its image block."""
    messages = [
        UserMessage(
            content=(ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),)
        ),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    assert _content_blocks(wire[0])[0]["cache_control"] == {"type": "ephemeral"}


def test_wire_messages_marks_the_tool_result_block_for_a_marked_last_tool_part() -> None:
    """A marked last part of a ToolMessage marks the enclosing tool_result block, never a nested block."""
    messages = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="a"), TextPart(text="b", cache_breakpoint=True)),
        )
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    tool_result = _content_blocks(wire[0])[0]
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert tool_result["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


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
            messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
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
    """A replayed tool call whose args_json is not JSON is an InvalidRequest, not a raise.

    args_json is caller data that nothing validates on the way in, so a batch item carrying one must
    fail on its own rather than escape the retry loop and cancel its siblings.
    """
    messages = [
        AssistantMessage(turn=(ToolCall(id="c1", name="f", args_json="not json"),)),
        ToolMessage(tool_call_id="c1", content="ok"),
    ]
    outcome = _structured_bound().build_request(messages)
    assert isinstance(outcome, InvalidRequest)
    assert "args_json" in outcome.reason


def test_build_request_reports_a_stored_payload_naming_no_type_as_invalid_request() -> None:
    """A stored payload with no type key is no content block, so nothing is sent.

    A turn replayed from a provider whose elements carry no type key is what gets here.
    Automatic caching is bound on, the case where the wire path reads that key to place the
    breakpoint: passing the payload through raises KeyError out of build_request, which names a
    langchaint defect for a request no defect produced.
    """
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    bound_adapter = _BoundAnthropicText(adapter=adapter, precomputed_fields=precomputed_fields)
    messages = [
        UserMessage(content="q"),
        AssistantMessage(turn=(OpaqueElement(raw={"parts": [{"text": "from elsewhere"}]}),)),
    ]
    outcome = bound_adapter.build_request(messages)
    assert isinstance(outcome, InvalidRequest)
    assert "type key" in outcome.reason


def test_a_built_request_renders_as_json_carrying_the_prompt_and_no_omitted_field() -> None:
    """as_json holds the binding's precomputed fields and this call's converted messages.

    temperature is absent rather than null, because the binding set none and the request body carries
    no such key.
    """
    request = _structured_bound().build_request([UserMessage(content="hi")])
    assert isinstance(request, _AnthropicRequestParams)
    rendered = json.loads(request.as_json())
    assert rendered["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert rendered["precomputed"]["model"] == "m"
    assert "temperature" not in rendered["precomputed"]


def test_wire_messages_writes_only_the_latest_four_marks_without_automatic_caching() -> None:
    """Five marks spend the 4-marker request budget on the latest four; the oldest goes unwritten."""
    messages = [
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(5))
        ),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    blocks = _content_blocks(wire[0])
    assert "cache_control" not in blocks[0]
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in blocks[1:])


def test_wire_messages_reserves_two_of_the_four_markers_for_automatic_caching() -> None:
    """With automatic caching, only the latest two marks are written beside the last-block marker."""
    messages = [
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(3))
        ),
        UserMessage(content="question"),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
    marked_blocks = _content_blocks(wire[0])
    assert "cache_control" not in marked_blocks[0]
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in marked_blocks[1:])
    assert _content_blocks(wire[1])[-1]["cache_control"] == {"type": "ephemeral"}


def test_request_renders_system_parts_with_marks_and_the_automatic_last_block_marker() -> None:
    """A parts system_prompt is one block per part; marked parts and the automatic last block carry markers."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            ),
            tool_schemas=(),
            automatic_prompt_caching=True,
        )
    )
    assert precomputed_fields.system == [
        {"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "semi-stable context", "cache_control": {"type": "ephemeral"}},
    ]
    assert precomputed_fields.message_mark_budget == 1


def test_request_system_parts_without_automatic_caching_mark_only_marked_parts() -> None:
    """Bound False, only the marked system part carries a marker; the budget spends only on it."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            system_prompt=(
                TextPart(text="stable", cache_breakpoint=True),
                TextPart(text="volatile"),
            ),
            tool_schemas=(),
            automatic_prompt_caching=False,
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
                automatic_prompt_caching=True,
            )
        )


def test_request_str_system_leaves_a_message_mark_budget_of_two() -> None:
    """A str system prompt under automatic caching leaves a message_mark_budget of two."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert precomputed_fields.message_mark_budget == 2
    uncached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
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
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=2
    )
    assert "cache_control" not in _content_blocks(wire[0])[0]
    assert _content_blocks(wire[1])[0]["cache_control"] == {"type": "ephemeral"}
    assert _content_blocks(wire[2])[0]["cache_control"] == {"type": "ephemeral"}


def test_wire_messages_explicit_mark_on_the_last_block_coexists_with_the_automatic_marker() -> (
    None
):
    """An explicit mark on the last block and the automatic last-block marker write one identical marker."""
    messages = [
        UserMessage(content=(TextPart(text="q", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        messages, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
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
        messages, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=0
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
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert _block_list(precomputed_fields.system)[0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert precomputed_fields.cache_ttl == "1h"


def test_request_1h_ttl_writes_the_ttl_on_the_last_tool_mark() -> None:
    """cache_ttl="1h" puts the explicit ttl key on the last-tool marker."""
    precomputed_fields = _adapter_1h()._precompute_fields(
        _binding(system_prompt=None, tool_schemas=_tool_schemas(), automatic_prompt_caching=True)
    )
    assert _block_list(precomputed_fields.tools)[-1]["cache_control"] == {
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
        messages, automatic_prompt_caching=True, cache_ttl="1h", message_mark_budget=2
    )
    assert _content_blocks(wire[0])[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert _content_blocks(wire[1])[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_request_rejects_an_empty_tuple_system_prompt() -> None:
    """An empty parts tuple, reachable only via a directly constructed Binding, raises instead of IndexError."""
    with pytest.raises(ValueError, match="empty tuple"):
        _ = _adapter()._precompute_fields(
            _binding(system_prompt=(), tool_schemas=(), automatic_prompt_caching=True)
        )


def test_billing_reported_reports_nothing_until_the_first_event_and_the_snapshot_after() -> None:
    """The running snapshot is readable only once an event has been accumulated.

    The SDK builds the snapshot from message_start and asserts on a read before that, so the
    adapter reports None until its first pull and prices the snapshot from then on.
    The snapshot carries input_tokens and output_tokens as required fields, so a mid-stream read
    always has counters to report.
    """

    async def scenario() -> tuple[Billing | None, Billing | None]:
        """Read the running billing before pulling anything, then after one item."""
        adapter_stream = _anthropic_stream(
            [_text_delta_event("he", 0)], _message_snapshot("end_turn")
        )
        before = adapter_stream.billing_reported()
        items = adapter_stream.items()
        await anext(items)
        return before, adapter_stream.billing_reported()

    before, after = asyncio.run(scenario())
    assert before is None
    assert after == _billing_from_sdk_usage(at.Usage(input_tokens=1, output_tokens=1), _PRICING)


def _turn_content() -> list[at.ContentBlock]:
    """Build a thinking block, a server tool call, then a text block.

    The thinking block carries a key the installed SDK does not name; model_construct rather than
    the constructor, because the extra key is the point and the constructor of a pinned SDK model
    has no field for a key that SDK does not name.
    The server tool call is the block this adapter has no turn-element variant for.
    The text block stays last, so the request's automatic cache marker lands on it rather than on a
    block whose stored dump an invariant compares against the wire.
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
        """Return a turn whose usage exercises every input counter and the write TTL split."""
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
    def response_with_a_block_the_adapter_does_not_model(self) -> BaseModel | None:
        """Return the turn whose middle block is a server tool call."""
        return _turn_message(_usage_with_cache_split())

    @override
    def assistant_wire_elements(self, request: RequestParams) -> Sequence[object]:
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
        """Return the adapter's whole exception table.

        Each status code is the one the SDK raises that class for, read from anthropic 0.120.2;
        the bare APIStatusError rows are the statuses the SDK maps to no class of its own,
        which is why the adapter reads the status rather than the exception class.
        APITimeoutError subclasses APIConnectionError, so timeouts reach transient through that
        isinstance, and RetryableError carries no response at all.
        A status row states the name a DoNotRetry failure takes, never whether it is retried:
        a 200, a mid-stream error event's raise on the live response, is declared_final;
        every 4xx is invalid_request whatever x-should-retry says, a 5xx marked final is
        declared_final, and any other status is unknown_exception.
        ValueError stands in for an exception the adapter cannot place.
        """
        return {
            _connection_error(): "transient",
            anthropic.APITimeoutError(
                httpx.Request("POST", "https://api.anthropic.com")
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
        """Return the parse rows: every listed status, both defaults, both TransientError forms.

        The statuses and error types come from the errors page parse_anthropic's docstring cites,
        plus the rows each table's docstring sources from the SDK;
        the rows without an error_type exercise the exception a non-JSON body produces.
        451 and 502 are the unlisted statuses, one per default: 451 takes the sub-500 DoNotRetry
        and 502 the 5xx RetryThisOne.
        The PauseAllDoNotRetry row is a 429 the provider's own x-should-retry marked final.
        """
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
