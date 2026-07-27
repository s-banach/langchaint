"""Anthropic Messages adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot: usage partition arithmetic, the 5-minute/1-hour cache-write cost split,
stop-reason mapping, tool_use extraction, cache-breakpoint placement, tool-choice translation,
and the precomputed request the binding determines.
"""

import asyncio
import base64
import json
import math
import re
from collections.abc import AsyncIterator, Sequence
from typing import get_args, override

import anthropic
import anthropic.types as at
import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from anthropic.lib.streaming import (
    AsyncMessageStream,
    ParsedContentBlockStopEvent,
    ParsedMessageStreamEvent,
)
from anthropic.types import MessageParam, ParsedMessage
from anthropic.types.parsed_message import ParsedTextBlock
from pydantic import BaseModel

from langchaint import (
    LLM,
    AssistantMessage,
    ImagePart,
    InferenceParams,
    PydanticTool,
    ReasoningTrace,
    SpecificToolChoice,
    StreamItem,
    TextPart,
    ToolCall,
    ToolManager,
    ToolMessage,
    UserMessage,
)
from langchaint.adapter import (
    AdapterResult,
    Binding,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    Refusal,
    ResponseOutcome,
    Unparsed,
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
    _adapter_result,
    _AnthropicStream,
    _assistant_content_blocks,
    _assistant_message_from,
    _BoundAnthropicStructured,
    _normalized_stop_reason,
    _normalized_usage,
    _NotSendableError,
    _user_content_blocks,
    _wire_messages,
    _wire_tool_choice,
)
from langchaint.exceptions import StreamProtocolError
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


def _assert_result[OutputT](outcome: ResponseOutcome[OutputT]) -> AdapterResult[OutputT]:
    """Narrow a ResponseOutcome to its success arm, failing the test on any other arm."""
    assert isinstance(outcome, AdapterResult)
    return outcome


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


def test_normalized_usage_partitions_input_counters_and_prices() -> None:
    """input_tokens is the uncached counter, and the normalized usage carries the priced cost."""
    usage = _normalized_usage(_usage_with_cache_split(), _PRICING)
    assert usage.input_tokens_cache_read == 200
    assert usage.input_tokens_cache_write == 30
    assert usage.input_tokens_cache_none == 100
    assert usage.input_tokens_total == 330
    assert usage.cost_in_usd == (100 * 3.0 + 200 * 0.3 + 10 * 3.75 + 20 * 6.0 + 50 * 15.0) / 1e6


def test_normalized_usage_counts_the_writes_it_bills_for() -> None:
    """The cache-write counter reads the same source the cost does, so the two cannot disagree.

    cache_creation_input_tokens and the cache_creation split are separate optional SDK fields with
    no documented relationship, so a response carrying only the split would otherwise report a
    cost covering 30 written tokens and a counter saying none were written.
    """
    usage = _normalized_usage(
        at.Usage(
            input_tokens=100,
            output_tokens=0,
            cache_creation=at.CacheCreation(
                ephemeral_5m_input_tokens=10, ephemeral_1h_input_tokens=20
            ),
        ),
        _PRICING,
    )
    assert usage.input_tokens_cache_write == 30
    assert abs(usage.cost_in_usd - (100 * 3.0 + 10 * 3.75 + 20 * 6.0) / 1e6) < 1e-12


def test_normalized_usage_treats_none_cache_counts_as_zero() -> None:
    """Absent cache counters normalize to zero, not None."""
    usage = _normalized_usage(at.Usage(input_tokens=7, output_tokens=3), _PRICING)
    assert usage.input_tokens_cache_read == 0
    assert usage.input_tokens_cache_write == 0
    assert usage.input_tokens_cache_none == 7


def test_normalized_usage_reads_reasoning_tokens_and_defaults_to_zero() -> None:
    """output_tokens_reasoning reads thinking_tokens, and is zero when output_tokens_details is absent."""
    with_details = _normalized_usage(
        at.Usage(
            input_tokens=1,
            output_tokens=9,
            output_tokens_details=at.OutputTokensDetails(thinking_tokens=4),
        ),
        _PRICING,
    )
    assert with_details.output_tokens_reasoning == 4
    without_details = _normalized_usage(at.Usage(input_tokens=1, output_tokens=9), _PRICING)
    assert without_details.output_tokens_reasoning == 0


def test_cost_splits_five_minute_and_one_hour_cache_writes() -> None:
    """The two cache-write tiers bill at their own rates from cache_creation."""
    cost = _normalized_usage(_usage_with_cache_split(), _PRICING).cost_in_usd
    expected = (100 * 3.0 + 200 * 0.3 + 10 * 3.75 + 20 * 6.0 + 50 * 15.0) / 1e6
    assert abs(cost - expected) < 1e-12


def test_cost_without_cache_creation_prices_all_writes_at_five_minute_rate() -> None:
    """With cache_creation absent, cache_creation_input_tokens bills as 5-minute writes."""
    usage = at.Usage(
        input_tokens=100,
        output_tokens=0,
        cache_creation_input_tokens=40,
    )
    cost = _normalized_usage(usage, _PRICING).cost_in_usd
    expected = (100 * 3.0 + 40 * 3.75) / 1e6
    assert abs(cost - expected) < 1e-12


def test_cost_is_nan_when_the_served_tier_has_no_table() -> None:
    """A response served at a tier the adapter holds no table for keeps its counters and costs NaN.

    The response was paid for, so the generation path keeps it and reports the cost as unknown.
    """
    usage = at.Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=30,
        service_tier="priority",
    )
    normalized = _normalized_usage(usage, _PRICING)
    assert math.isnan(normalized.cost_in_usd)
    assert math.isnan(normalized.output_tokens_cost_in_usd)
    assert normalized.input_tokens_total == 330
    assert normalized.output_tokens == 50


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response, standard rates a response reporting no tier."""
    counters = {"input_tokens": 100, "output_tokens": 50}
    pricing: dict[AnthropicPricedServiceTier, AnthropicPricingTable] = {
        "standard": _STANDARD_RATES,
        "priority": _PRIORITY_RATES,
    }
    at_priority = _normalized_usage(at.Usage(**counters, service_tier="priority"), pricing)
    at_standard = _normalized_usage(at.Usage(**counters, service_tier="standard"), pricing)
    reporting_none = _normalized_usage(at.Usage(**counters), pricing)
    assert at_priority.cost_in_usd == pytest.approx(2 * at_standard.cost_in_usd)
    assert reporting_none.cost_in_usd == at_standard.cost_in_usd


def test_the_categories_are_priced_apart() -> None:
    """Each stored cost is its own category's product, and they sum to cost_in_usd."""
    usage = _normalized_usage(_usage_with_cache_split(), _PRICING)
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
        AnthropicMessagesAdapter(
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
        ("pause_turn", "other"),
        (None, "other"),
    ],
)
def test_stop_reason_mapping(raw: str | None, expected: str) -> None:
    """Recognized stop reasons pass through; everything else becomes other."""
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
    result = _adapter_result(message=message, output="hello world", pricing=_PRICING)
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
    assert reasoning_trace.text == "check first"
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
    assert reasoning_trace.reasoning["thinking"] == ""


def test_redacted_thinking_round_trips_routed_by_its_type_key() -> None:
    """A redacted_thinking block round-trips as its own dump; the type key routes it on the wire.

    Its trace carries no text: the block holds an opaque string under data and nothing readable,
    which is the arm that exists because a redacted block has no text to carry.
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
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
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
    wire = _wire_messages(
        conversation, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
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
                    reasoning={"type": "thinking", "thinking": "x", "signature": "s"},
                ),
            )
        )
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
    assert all("cache_control" not in block for block in _content_blocks(wire[0]))


def test_wire_messages_writes_no_breakpoint_when_caching_disabled() -> None:
    """With caching off, no block anywhere carries a cache_control marker."""
    conversation = [
        UserMessage(content="hi"),
        ToolMessage(tool_call_id="tu_1", content="r1"),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    assert all(
        "cache_control" not in block for message in wire for block in _content_blocks(message)
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
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
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
    conversation = [
        ToolMessage(tool_call_id="tu_1", content=(ImagePart(data=b"x", media_type="image/tiff"),))
    ]
    with pytest.raises(_NotSendableError, match="image/tiff"):
        _wire_messages(
            conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
        )


def test_wire_tool_choice_required_becomes_any_and_inverts_parallel() -> None:
    """Neutral required maps to any; disable_parallel_tool_use is the inverse."""
    for parallel in (True, False):
        assert _wire_tool_choice("required", parallel_tool_calls=parallel) == {
            "type": "any",
            "disable_parallel_tool_use": not parallel,
        }


def test_wire_tool_choice_specific_tool_names_the_tool() -> None:
    """A SpecificToolChoice becomes the named-tool form."""
    assert _wire_tool_choice(SpecificToolChoice(tool_name="x"), parallel_tool_calls=True) == {
        "type": "tool",
        "name": "x",
        "disable_parallel_tool_use": False,
    }


def test_wire_tool_choice_none_forbids_calls() -> None:
    """Neutral none maps to the none form with no parallel flag."""
    assert _wire_tool_choice("none", parallel_tool_calls=True) == {"type": "none"}


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
    request = _adapter()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert request.max_tokens == 4096
    assert isinstance(request.tools, anthropic.Omit)
    assert isinstance(request.tool_choice, anthropic.Omit)
    assert request.output_config == {"effort": "high"}
    assert request.thinking == {"type": "adaptive"}


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
    request = _adapter()._request(binding)
    assert request.output_config == {"effort": "minimal"}
    assert request.thinking == {"type": "adaptive"}


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
    request = _adapter()._request(binding)
    assert isinstance(request.output_config, anthropic.Omit)
    assert isinstance(request.thinking, anthropic.Omit)


def test_request_maps_temperature_and_omits_it_when_unset() -> None:
    """A bound temperature lands on the request; None leaves the omit sentinel."""
    unset = _adapter()._request(
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
    assert _adapter()._request(binding).temperature == 0.2


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request; None leaves the omit sentinel.

    The sentinel is what keeps an unstated tier off the wire: sending an explicit null would be a
    different request from omitting the key.
    """
    binding = _binding(system_prompt=None, tool_schemas=(), automatic_prompt_caching=False)
    assert isinstance(_adapter()._request(binding).service_tier, anthropic.Omit)
    stated = AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
        service_tier="standard_only",
    )
    assert stated._request(binding).service_tier == "standard_only"


def test_request_marks_the_system_block_only_when_caching() -> None:
    """The system block carries a breakpoint under caching and none without it."""
    cached = _adapter()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert _block_list(cached.system)[0]["cache_control"] == {"type": "ephemeral"}
    uncached = _adapter()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    assert "cache_control" not in _block_list(uncached.system)[0]


def test_request_marks_last_tool_only_without_a_system_prompt() -> None:
    """The prefix breakpoint sits on the last tool only when no system prompt follows."""
    schemas = _tool_schemas()
    without_system = _adapter()._request(
        _binding(system_prompt=None, tool_schemas=schemas, automatic_prompt_caching=True)
    )
    assert _block_list(without_system.tools)[-1]["cache_control"] == {"type": "ephemeral"}
    with_system = _adapter()._request(
        _binding(system_prompt="sys", tool_schemas=schemas, automatic_prompt_caching=True)
    )
    assert "cache_control" not in _block_list(with_system.tools)[-1]


def test_user_content_blocks_rejects_unsupported_image_media_type() -> None:
    """An image media type outside the accepted set is not sendable."""
    message = UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),))
    with pytest.raises(_NotSendableError):
        _user_content_blocks(message)


class _FakeSDKMessageStream(AsyncMessageStream[None]):
    """Replays constructed events without a connection.

    Overrides exactly the surface _AnthropicStream uses (iteration, close,
    and the current_message_snapshot the stop-reason check reads); the base __init__ is deliberately not called,
    so the untouched base machinery stays unusable.
    """

    def __init__(
        self,
        replay_events: Sequence[ParsedMessageStreamEvent],
        message_snapshot: ParsedMessage[None],
    ) -> None:
        self._replay_events = list(replay_events)
        self._message_snapshot = message_snapshot

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
        result = _assert_result(await adapter_stream.final())
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
        adapter_stream = _anthropic_stream([_text_delta_event("he", 0)], _message_snapshot(None))
        with pytest.raises(StreamProtocolError):
            async for _item in adapter_stream.items():
                pass

    asyncio.run(scenario())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundAnthropicStructured[_StructuredReport]:
    """Build a structured-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    request = adapter._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    return _BoundAnthropicStructured(
        adapter=adapter, request=request, response_format=_StructuredReport
    )


def _parsed_message(
    parsed_output: _StructuredReport | None,
    stop_reason: at.StopReason = "end_turn",
) -> ParsedMessage[_StructuredReport]:
    """Build the SDK parse result whose first text block carries the given parsed output."""
    return ParsedMessage[_StructuredReport](
        id="msg_1",
        content=[
            ParsedTextBlock[_StructuredReport](type="text", text="{}", parsed_output=parsed_output)
        ],
        model="claude-sonnet-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=at.Usage(input_tokens=1, output_tokens=1),
    )


def test_structured_bind_returns_the_sdk_parsed_instance() -> None:
    """The structured bound adapter hands back the SDK-parsed response_format instance as content."""
    report = _StructuredReport(city="Nairobi", celsius=25)
    assert _structured_bound()._parsed_output(_parsed_message(report)) == report


def test_structured_bind_reports_unparsed_without_parsed_output() -> None:
    """A turn with no parsed output is Unparsed: a later attempt may still produce it."""
    outcome = _structured_bound()._parsed_output(_parsed_message(None))
    assert isinstance(outcome, Unparsed)
    # The rejected 200's billing rides on the arm so the retry record is not zero.
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_refusal_on_a_refusal_stop_reason() -> None:
    """A refusal stop_reason with no parsed output is Refusal, carrying its billing."""
    outcome = _structured_bound()._parsed_output(_parsed_message(None, stop_reason="refusal"))
    assert isinstance(outcome, Refusal)
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_max_completion_tokens_exceeded_on_a_max_tokens_stop_reason() -> (
    None
):
    """A max_tokens stop_reason with no parsed output is MaxCompletionTokensExceeded, carrying its billing."""
    outcome = _structured_bound()._parsed_output(_parsed_message(None, stop_reason="max_tokens"))
    assert isinstance(outcome, MaxCompletionTokensExceeded)
    assert outcome.usage.cost_in_usd > 0.0


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
    assert _adapter().retry_after_seconds(error) == 1.5


def test_retry_after_seconds_parses_the_seconds_header() -> None:
    """Without retry-after-ms, retry-after is parsed as float seconds."""
    error = _rate_limit_error({"retry-after": "49"})
    assert _adapter().retry_after_seconds(error) == 49.0


def test_retry_after_seconds_is_none_without_headers_or_status() -> None:
    """No headers, an unparseable value, and a non-SDK error all yield None."""
    adapter = _adapter()
    assert adapter.retry_after_seconds(_rate_limit_error({})) is None
    assert (
        adapter.retry_after_seconds(
            _rate_limit_error({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        )
        is None
    )
    assert adapter.retry_after_seconds(ValueError("boom")) is None


def _status_error[ErrorT: anthropic.APIStatusError](
    error_class: type[ErrorT],
    status_code: int,
    headers: dict[str, str] | None = None,
) -> ErrorT:
    """Build one of the SDK's status exceptions around a constructed httpx response."""
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        headers=headers,
    )
    return error_class("boom", response=response, body=None)


def _connection_error() -> anthropic.APIConnectionError:
    """Build the SDK's transport-failure exception, which carries a request and no response."""
    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_status_error(anthropic.RateLimitError, 429), "rate_limit"),
        (_status_error(anthropic.OverloadedError, 529), "rate_limit"),
        (_status_error(anthropic.InternalServerError, 500), "transient"),
        (_connection_error(), "transient"),
        (
            anthropic.APITimeoutError(httpx.Request("POST", "https://api.anthropic.com")),
            "transient",
        ),
        (_status_error(anthropic.ConflictError, 409), "transient"),
        (anthropic.RetryableError("middleware said retry"), "transient"),
        (_status_error(anthropic.BadRequestError, 400), "invalid_request"),
        (_status_error(anthropic.AuthenticationError, 401), "invalid_request"),
        (_status_error(anthropic.PermissionDeniedError, 403), "invalid_request"),
        (_status_error(anthropic.NotFoundError, 404), "invalid_request"),
        (_status_error(anthropic.RequestTooLargeError, 413), "invalid_request"),
        (_status_error(anthropic.UnprocessableEntityError, 422), "invalid_request"),
        (_status_error(anthropic.APIStatusError, 402), "invalid_request"),
        (_status_error(anthropic.APIStatusError, 408), "transient"),
        (_status_error(anthropic.InternalServerError, 503), "transient"),
        (_status_error(anthropic.APIStatusError, 302), "unrecognized"),
        (_status_error(anthropic.BadRequestError, 400, {"x-should-retry": "true"}), "transient"),
        (
            _status_error(anthropic.InternalServerError, 500, {"x-should-retry": "false"}),
            "unrecognized",
        ),
        (_status_error(anthropic.RateLimitError, 429, {"x-should-retry": "false"}), "rate_limit"),
        (_status_error(anthropic.RateLimitError, 429, {"x-should-retry": "true"}), "rate_limit"),
        (ValueError("boom"), "unrecognized"),
    ],
)
def test_classify_maps_each_sdk_exception_to_its_classification(
    error: Exception, expected: ErrorClassification
) -> None:
    """Each error lands on the classification the adapter's classify docstring names.

    Each status code is the one the SDK raises that class for, read from anthropic 0.120.0;
    the bare APIStatusError rows are the statuses the SDK maps to no class of its own,
    which is why the adapter reads the status rather than the exception class.
    OverloadedError is anthropic's own overload signal and shares the rate_limit class with RateLimitError.
    APITimeoutError subclasses APIConnectionError, so timeouts reach transient through that isinstance,
    and RetryableError carries no response at all.
    x-should-retry overrides the status in both directions, except on a rate-limit status, which
    stays rate_limit whatever the header says, so the limiter's account-wide pause is still armed.
    A 3xx and the non-SDK ValueError land on the unrecognized default,
    which fails the one item without a retry.
    """
    assert _adapter().classify(error) == expected


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
        anthropic_bedrock_model("anthropic.claude-sonnet-5", client=legacy_client)
    assert "AsyncAnthropicBedrockMantle" in str(excinfo.value)


def test_bedrock_table_is_total_over_the_bedrock_ids() -> None:
    """Every AnthropicBedrockModelName has a routing entry, so a new Bedrock wire model id must add one."""
    assert set(ANTHROPIC_BEDROCK) == set(get_args(AnthropicBedrockModelName.__value__))


def test_wire_messages_marks_a_marked_user_part() -> None:
    """A user part with cache_breakpoint carries the marker on its own block; unmarked siblings carry none."""
    conversation = [
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                TextPart(text="question"),
            )
        ),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    blocks = _content_blocks(wire[0])
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]


def test_wire_messages_marks_a_marked_image_part() -> None:
    """An image part with cache_breakpoint carries the marker on its image block."""
    conversation = [
        UserMessage(
            content=(ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),)
        ),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    assert _content_blocks(wire[0])[0]["cache_control"] == {"type": "ephemeral"}


def test_wire_messages_marks_the_tool_result_block_for_a_marked_last_tool_part() -> None:
    """A marked last part of a ToolMessage marks the enclosing tool_result block, never a nested block."""
    conversation = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="a"), TextPart(text="b", cache_breakpoint=True)),
        )
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    tool_result = _content_blocks(wire[0])[0]
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert tool_result["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


def test_wire_messages_rejects_a_marked_non_last_tool_part() -> None:
    """A marked part before the ToolMessage's last is rejected instead of silently moving the boundary."""
    conversation = [
        ToolMessage(
            tool_call_id="tu_1",
            content=(TextPart(text="a", cache_breakpoint=True), TextPart(text="b")),
        )
    ]
    with pytest.raises(_NotSendableError, match="last part"):
        _wire_messages(
            conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
        )


def test_send_reports_an_unsendable_conversation_as_invalid_request() -> None:
    """An unsendable conversation reaches send's caller as the InvalidRequest arm, with nothing sent.

    The client holds no API key, so reaching the wire would fail rather than return this arm.
    """

    async def scenario() -> None:
        conversation = [UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),))]
        outcome = await _structured_bound().send(conversation)
        assert isinstance(outcome, InvalidRequest)
        assert "image/tiff" in outcome.reason

    asyncio.run(scenario())


def test_wire_messages_writes_only_the_latest_four_marks_without_automatic_caching() -> None:
    """Five marks spend the 4-marker request budget on the latest four; the oldest goes unwritten."""
    conversation = [
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(5))
        ),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=4
    )
    blocks = _content_blocks(wire[0])
    assert "cache_control" not in blocks[0]
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in blocks[1:])


def test_wire_messages_reserves_two_slots_for_automatic_markers() -> None:
    """With automatic caching, only the latest two marks are written beside the last-block marker."""
    conversation = [
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(3))
        ),
        UserMessage(content="question"),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
    marked_blocks = _content_blocks(wire[0])
    assert "cache_control" not in marked_blocks[0]
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in marked_blocks[1:])
    assert _content_blocks(wire[1])[-1]["cache_control"] == {"type": "ephemeral"}


def test_request_renders_system_parts_with_marks_and_the_automatic_last_block_marker() -> None:
    """A parts system_prompt is one block per part; marked parts and the automatic last block carry markers."""
    request = _adapter()._request(
        _binding(
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            ),
            tool_schemas=(),
            automatic_prompt_caching=True,
        )
    )
    assert request.system == [
        {"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "semi-stable context", "cache_control": {"type": "ephemeral"}},
    ]
    assert request.message_mark_budget == 1


def test_request_system_parts_without_automatic_caching_mark_only_marked_parts() -> None:
    """Bound False, only the marked system part carries a marker; the budget spends only on it."""
    request = _adapter()._request(
        _binding(
            system_prompt=(
                TextPart(text="stable", cache_breakpoint=True),
                TextPart(text="volatile"),
            ),
            tool_schemas=(),
            automatic_prompt_caching=False,
        )
    )
    assert request.system == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "volatile"},
    ]
    assert request.message_mark_budget == 3


def test_request_rejects_a_binding_whose_markers_exceed_the_request_limit() -> None:
    """Four marked system parts plus the automatic markers cannot fit the 4-marker limit."""
    with pytest.raises(ValueError, match="limit"):
        _adapter()._request(
            _binding(
                system_prompt=tuple(
                    TextPart(text=f"s{index}", cache_breakpoint=True) for index in range(4)
                ),
                tool_schemas=(),
                automatic_prompt_caching=True,
            )
        )


def test_request_str_system_budget_leaves_two_slots_for_message_marks() -> None:
    """A str system prompt under automatic caching leaves two slots for message marks."""
    request = _adapter()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert request.message_mark_budget == 2
    uncached = _adapter()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=False)
    )
    assert uncached.message_mark_budget == 4


def test_wire_messages_budget_mixes_user_and_tool_result_marks_across_messages() -> None:
    """The latest-N budget counts marks across message kinds in conversation order."""
    conversation = [
        UserMessage(content=(TextPart(text="oldest", cache_breakpoint=True),)),
        ToolMessage(tool_call_id="tu_1", content=(TextPart(text="mid", cache_breakpoint=True),)),
        UserMessage(content=(TextPart(text="latest", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=2
    )
    assert "cache_control" not in _content_blocks(wire[0])[0]
    assert _content_blocks(wire[1])[0]["cache_control"] == {"type": "ephemeral"}
    assert _content_blocks(wire[2])[0]["cache_control"] == {"type": "ephemeral"}


def test_wire_messages_explicit_mark_on_the_last_block_coexists_with_the_automatic_marker() -> (
    None
):
    """An explicit mark on the last block and the automatic last-block marker write one identical marker."""
    conversation = [
        UserMessage(content=(TextPart(text="q", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=True, cache_ttl="5m", message_mark_budget=2
    )
    assert _content_blocks(wire[0]) == [
        {"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}
    ]


def test_wire_messages_writes_no_marks_at_zero_budget() -> None:
    """A zero budget leaves every mark unwritten instead of slicing the whole list."""
    conversation = [
        UserMessage(content=(TextPart(text="m", cache_breakpoint=True),)),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=False, cache_ttl="5m", message_mark_budget=0
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
    request = _adapter_1h()._request(
        _binding(system_prompt="sys", tool_schemas=(), automatic_prompt_caching=True)
    )
    assert _block_list(request.system)[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert request.cache_ttl == "1h"


def test_request_1h_ttl_writes_the_ttl_on_the_last_tool_mark() -> None:
    """cache_ttl="1h" puts the explicit ttl key on the last-tool marker."""
    request = _adapter_1h()._request(
        _binding(system_prompt=None, tool_schemas=_tool_schemas(), automatic_prompt_caching=True)
    )
    assert _block_list(request.tools)[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_wire_messages_1h_ttl_writes_the_ttl_on_message_and_automatic_marks() -> None:
    """cache_ttl="1h" puts the explicit ttl key on cache_breakpoint marks and the automatic last-block marker."""
    conversation = [
        UserMessage(content=(TextPart(text="context", cache_breakpoint=True),)),
        UserMessage(content="question"),
    ]
    wire = _wire_messages(
        conversation, automatic_prompt_caching=True, cache_ttl="1h", message_mark_budget=2
    )
    assert _content_blocks(wire[0])[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert _content_blocks(wire[1])[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_request_rejects_an_empty_tuple_system_prompt() -> None:
    """An empty parts tuple, reachable only via a directly constructed Binding, raises instead of IndexError."""
    with pytest.raises(ValueError, match="empty tuple"):
        _adapter()._request(
            _binding(system_prompt=(), tool_schemas=(), automatic_prompt_caching=True)
        )
