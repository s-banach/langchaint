"""OpenAI Responses adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot:
the usage partition derived by subtracting cache counts from input_tokens and its cross-check, cost arithmetic,
input-item placement, tool-choice translation, stop-reason derivation (the API reports no finish reason),
the zero-usage fallback when a response omits usage, and the precomputed request the binding determines.
"""

import asyncio
import base64
import json
import math
import re
from collections.abc import AsyncIterator, Sequence
from typing import override

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from openai._models import construct_type_unchecked
from openai.lib._parsing._responses import type_to_text_format_param
from openai.lib.streaming.responses import AsyncResponseStream, ResponseStreamEvent
from openai.lib.streaming.responses import (
    ResponseTextDeltaEvent as AccumulatedResponseTextDeltaEvent,
)
from openai.lib.streaming.responses._events import (
    ResponseCompletedEvent as AccumulatedResponseCompletedEvent,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseUsage,
)
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response import IncompleteDetails, ResponseStatus
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel, ValidationError

from langchaint import (
    AssistantMessage,
    ImagePart,
    InferenceParams,
    PricingTable,
    ReasoningEffort,
    ReasoningTrace,
    SpecificToolChoice,
    StreamItem,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.adapter import (
    AdapterResult,
    Binding,
    EmptyTurn,
    ErrorClassification,
    MaxCompletionTokensExceeded,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    Refusal,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.exceptions import StreamProtocolError
from langchaint.openai import (
    OpenAIPricedServiceTier,
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.responses_adapter import (
    _adapter_result,
    _assistant_items,
    _assistant_message_from,
    _BoundOpenAIStructured,
    _BoundOpenAIText,
    _normalized_stop_reason,
    _normalized_usage,
    _OpenAIStream,
    _wire_input,
    _wire_tool_choice,
)

_DEFAULT_RATES = PricingTable(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)

_PRICING: dict[OpenAIPricedServiceTier, PricingTable] = {"default": _DEFAULT_RATES}
"""The default tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = PricingTable(
    input_cache_none_usd_per_million_tokens=5.0,
    output_usd_per_million_tokens=20.0,
    cache_read_usd_per_million_tokens=2.5,
    cache_write_usd_per_million_tokens=6.25,
)
"""Twice the default rates, so a tier-selection test reads as a doubling."""

_TEXT_OUTPUT_ITEM: dict[str, object] = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hey", "annotations": []}],
}

_FUNCTION_CALL_OUTPUT_ITEM = {
    "type": "function_call",
    "id": "fc1",
    "call_id": "call1",
    "name": "lookup",
    "arguments": '{"q": 1}',
}

_REASONING_OUTPUT_ITEM: dict[str, object] = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "enc-1",
}


def _assert_result[OutputT](outcome: ResponseOutcome[OutputT]) -> AdapterResult[OutputT]:
    """Narrow a ResponseOutcome to its success arm, failing the test on any other arm."""
    assert isinstance(outcome, AdapterResult)
    return outcome


def _usage_with_cache() -> ResponseUsage:
    """Return usage whose input_tokens includes both cache counts."""
    return ResponseUsage(
        input_tokens=1000,
        input_tokens_details=InputTokensDetails(cached_tokens=600, cache_write_tokens=100),
        output_tokens=40,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=1040,
    )


_SERVER_ERROR = ResponseError(code="server_error", message="The server had an error.")
"""A failed response's error whose code the disposition table calls transient."""

_IMAGE_DOWNLOAD_ERROR = ResponseError(
    code="failed_to_download_image", message="Failed to download image from the URL."
)
"""A failed response's error whose code the disposition table calls terminal."""


def _response(
    *,
    usage: ResponseUsage | None,
    output: list[object] | None = None,
    status: str = "completed",
    incomplete_details: IncompleteDetails | None = None,
    service_tier: str | None = None,
    error: ResponseError | None = None,
) -> OpenAIResponse:
    """Build a response carrying the given usage, output items, status, tier, and failure error."""
    return OpenAIResponse.model_validate({
        "id": "r1",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": output if output is not None else [_TEXT_OUTPUT_ITEM],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "incomplete_details": incomplete_details,
        "usage": usage,
        "service_tier": service_tier,
        "error": error,
    })


def test_normalized_usage_subtracts_cache_from_input_tokens_and_prices() -> None:
    """The uncached counter is input_tokens minus both cache counts, and the priced cost rides on it."""
    usage = _normalized_usage(_response(usage=_usage_with_cache()), _PRICING)
    assert usage.input_tokens_cache_read == 600
    assert usage.input_tokens_cache_write == 100
    assert usage.input_tokens_cache_none == 300
    assert usage.input_tokens_total == 1000
    assert usage.cost_in_usd == pytest.approx(
        (300 * 2.5 + 600 * 1.25 + 100 * 3.125 + 40 * 10.0) / 1e6
    )


def test_normalized_usage_reads_reasoning_tokens() -> None:
    """output_tokens_reasoning reads the required reasoning_tokens counter."""
    usage = _normalized_usage(
        _response(
            usage=ResponseUsage(
                input_tokens=10,
                input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
                output_tokens=20,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
                total_tokens=30,
            )
        ),
        _PRICING,
    )
    assert usage.output_tokens_reasoning == 8


def test_normalized_usage_rejects_cache_counts_exceeding_input_tokens() -> None:
    """Cache counters summing past input_tokens raise instead of going negative.

    The subtraction derives input_tokens_cache_none, so the guard is its non-negativity constraint.
    """
    with pytest.raises(ValidationError):
        _normalized_usage(
            _response(
                usage=ResponseUsage(
                    input_tokens=1000,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=900, cache_write_tokens=200
                    ),
                    output_tokens=40,
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    total_tokens=1040,
                )
            ),
            _PRICING,
        )


def test_the_categories_are_priced_apart() -> None:
    """Cache reads, cache writes, the uncached remainder, and output each bill at their rate."""
    usage = _normalized_usage(_response(usage=_usage_with_cache()), _PRICING)
    assert usage.input_tokens_cache_none_cost_in_usd == 300 * 2.5 / 1e6
    assert usage.input_tokens_cache_read_cost_in_usd == 600 * 1.25 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == 100 * 3.125 / 1e6
    assert usage.output_tokens_cost_in_usd == 40 * 10.0 / 1e6


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response; "auto" and no tier both price at default."""
    pricing: dict[OpenAIPricedServiceTier, PricingTable] = {
        "default": _DEFAULT_RATES,
        "priority": _PRIORITY_RATES,
    }
    at_priority = _normalized_usage(
        _response(usage=_usage_with_cache(), service_tier="priority"), pricing
    )
    at_default = _normalized_usage(
        _response(usage=_usage_with_cache(), service_tier="default"), pricing
    )
    reporting_auto = _normalized_usage(
        _response(usage=_usage_with_cache(), service_tier="auto"), pricing
    )
    reporting_none = _normalized_usage(_response(usage=_usage_with_cache()), pricing)
    assert at_priority.cost_in_usd == pytest.approx(2 * at_default.cost_in_usd)
    assert reporting_auto.cost_in_usd == at_default.cost_in_usd
    assert reporting_none.cost_in_usd == at_default.cost_in_usd


def test_cost_is_nan_when_the_served_tier_has_no_table() -> None:
    """A response served at a tier the adapter holds no table for keeps its counters and costs NaN."""
    usage = _normalized_usage(_response(usage=_usage_with_cache(), service_tier="flex"), _PRICING)
    assert math.isnan(usage.cost_in_usd)
    assert usage.input_tokens_total == 1000
    assert usage.output_tokens == 40


def test_pricing_without_the_default_key_raises_at_construction() -> None:
    """A pricing mapping missing "default" fails before any request, naming the model."""
    priority_only: dict[OpenAIPricedServiceTier, PricingTable] = {"priority": _PRIORITY_RATES}
    with pytest.raises(ValueError, match=re.escape("'default'")):
        OpenAIResponsesAdapter(
            client=AsyncOpenAI(api_key="test"),
            model="gpt-5.6-terra",
            pricing=priority_only,
            provider_name="openai",
            supports_prompt_cache_options=True,
        )


def test_stop_reason_is_tool_use_with_a_function_call_item() -> None:
    """Any function_call output item derives tool_use, whatever the status."""
    response = _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM])
    assert _normalized_stop_reason(response) == "tool_use"


def test_stop_reason_completed_is_end_turn() -> None:
    """Status completed without tool calls derives end_turn."""
    assert _normalized_stop_reason(_response(usage=None)) == "end_turn"


_REFUSAL_MESSAGE_ITEM: dict[str, object] = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": "I can't help with that"}],
}


def test_stop_reason_refusal_block_is_refusal() -> None:
    """A refusal content block derives refusal, ahead of the status and tool-call checks."""
    assert (
        _normalized_stop_reason(_response(usage=None, output=[_REFUSAL_MESSAGE_ITEM])) == "refusal"
    )


def test_assistant_message_carries_the_refusal_text_and_replays_it() -> None:
    """A refusal content part becomes a TextPart, so the refused turn replays as the model wrote it.

    Dropped instead, the turn holds no elements and sends nothing back, which reopens the
    conversation at the point the model declined.
    """
    assistant_message = _assistant_message_from(
        _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM])
    )
    assert assistant_message.turn == (TextPart(text="I can't help with that"),)
    assert _assistant_items(assistant_message) == [
        {"role": "assistant", "content": "I can't help with that"}
    ]


def test_stop_reason_incomplete_for_max_output_tokens_is_max_tokens() -> None:
    """Status incomplete with reason max_output_tokens derives max_tokens."""
    response = _response(
        usage=None,
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="max_output_tokens"),
    )
    assert _normalized_stop_reason(response) == "max_tokens"


def test_stop_reason_incomplete_for_content_filter_is_refusal() -> None:
    """Status incomplete with reason content_filter derives refusal."""
    response = _response(
        usage=None,
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="content_filter"),
    )
    assert _normalized_stop_reason(response) == "refusal"


def test_stop_reason_other_statuses_are_other() -> None:
    """A failed status derives other."""
    assert _normalized_stop_reason(_response(usage=None, status="failed")) == "other"


def test_assistant_message_collects_text_and_tool_calls() -> None:
    """The assistant turn carries the concatenated text and every function call."""
    response = _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM])
    assistant_message = _assistant_message_from(response)
    assert assistant_message.text == "hey"
    assert assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )


def test_reasoning_round_trips_verbatim_in_position() -> None:
    """A reasoning item round-trips verbatim and in its original position.

    Produce yields one ReasoningTrace where the reasoning item sat.
    Consume re-emits the stored dict unchanged, in the same position, with one input item per modeled output item.
    """
    response = _response(
        usage=None,
        output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM],
    )
    assistant_message = _assistant_message_from(response)
    assert [type(element) for element in assistant_message.turn] == [
        ReasoningTrace,
        TextPart,
        ToolCall,
    ]
    reasoning_trace = assistant_message.turn[0]
    assert isinstance(reasoning_trace, ReasoningTrace)
    assert reasoning_trace.reasoning == _REASONING_OUTPUT_ITEM
    assert assistant_message.text == "hey"
    assert assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )
    items = _assistant_items(assistant_message)
    assert len(items) == len(response.output)
    assert items[0] == reasoning_trace.reasoning
    assert items[1] == {"role": "assistant", "content": "hey"}
    assert items[2] == {
        "type": "function_call",
        "call_id": "call1",
        "name": "lookup",
        "arguments": '{"q": 1}',
    }


def _reasoning_item(
    *, summary: tuple[str, ...] = (), content: tuple[str, ...] | None = None
) -> dict[str, object]:
    """Build a reasoning output item whose summary and content hold the given texts."""
    item: dict[str, object] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": text} for text in summary],
    }
    if content is not None:
        item["content"] = [{"type": "reasoning_text", "text": text} for text in content]
    return item


def test_one_summary_part_becomes_the_trace_text() -> None:
    """A single summary part lands on the trace verbatim."""
    response = _response(usage=None, output=[_reasoning_item(summary=("thought it over",))])
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == "thought it over"


def test_several_summary_parts_join_on_a_blank_line() -> None:
    """Summary parts join on a blank line, because each is a separately delimited unit.

    Joining on the empty string runs one part's last line into the next one's first.
    """
    response = _response(
        usage=None,
        output=[
            _reasoning_item(
                summary=("**Reading the question**\n\nFirst.", "**Answering**\n\nThen.")
            )
        ],
    )
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == "**Reading the question**\n\nFirst.\n\n**Answering**\n\nThen."


def test_summary_wins_over_content_when_both_hold_text() -> None:
    """Both lists populated take the summary, the one the request asks for.

    Reading content first, or concatenating the two, passes every other case here,
    so this is what pins the precedence.
    """
    response = _response(
        usage=None,
        output=[_reasoning_item(summary=("the summary",), content=("the content",))],
    )
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == "the summary"


def test_content_supplies_the_text_when_the_summary_is_empty() -> None:
    """An empty summary falls back to the content parts rather than dropping returned text."""
    response = _response(usage=None, output=[_reasoning_item(content=("worked it out",))])
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == "worked it out"


def test_a_reasoning_item_holding_no_text_leaves_the_trace_text_none() -> None:
    """Empty summary and content leave text None, the no-readable-text signal."""
    response = _response(usage=None, output=[_reasoning_item()])
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text is None


def test_parts_that_are_all_empty_leave_the_trace_text_none() -> None:
    r"""Several empty parts yield None, not the separator they would otherwise join into.

    Present-but-empty parts are the case a trailing falsy check alone misses:
    two of them join into "\n\n", which is truthy, so it would reach a span as a
    reasoning part carrying nothing.
    """
    response = _response(usage=None, output=[_reasoning_item(summary=("", ""))])
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text is None
    content_only = _response(usage=None, output=[_reasoning_item(content=("", ""))])
    content_trace = _assistant_message_from(content_only).turn[0]
    assert isinstance(content_trace, ReasoningTrace)
    assert content_trace.text is None


def test_an_empty_part_beside_a_real_one_drops_out_of_the_join() -> None:
    """An empty part contributes no leading or trailing blank line to the joined text."""
    response = _response(usage=None, output=[_reasoning_item(summary=("", "real"))])
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == "real"


def test_two_text_parts_stay_split_on_produce_and_rejoin_into_one_message_item() -> None:
    """A message item with two text parts yields two adjacent TextParts.

    On consume, the maximal adjacent run re-joins into one assistant message item.
    """
    two_part_message: dict[str, object] = {
        "type": "message",
        "id": "m1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "he", "annotations": []},
            {"type": "output_text", "text": "y", "annotations": []},
        ],
    }
    assistant_message = _assistant_message_from(_response(usage=None, output=[two_part_message]))
    assert assistant_message.turn == (TextPart(text="he"), TextPart(text="y"))
    assert _assistant_items(assistant_message) == [{"role": "assistant", "content": "hey"}]


def test_reasoning_with_a_key_the_installed_sdk_lacks_survives_the_wire_builder() -> None:
    """A stored dict carrying a field newer than the installed SDK param re-emits unchanged.

    A consume step that reshaped the dict to the pinned param keys would corrupt the payload
    the API re-reads across an SDK upgrade.
    """
    reasoning: dict[str, object] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "enc-1",
        "field_newer_than_sdk": "x",
    }
    assistant_message = AssistantMessage(turn=(ReasoningTrace(reasoning=reasoning),))
    assert _assistant_items(assistant_message) == [reasoning]


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """An anthropic-produced trace emits its dict as-is; the API rejects the unknown type key, not this adapter."""
    reasoning: dict[str, object] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(
        turn=(ReasoningTrace(reasoning=reasoning), TextPart(text="hi"))
    )
    assert _assistant_items(assistant_message) == [
        reasoning,
        {"role": "assistant", "content": "hi"},
    ]


def test_adapter_result_normalizes_a_response_with_usage() -> None:
    """A response with usage yields the normalized partition, cost, and stop reason.

    raw must be the SDK response object itself (identity, not equality):
    an equal copy would silently reintroduce the per-request deep copy the no-rewrap rule bans.
    """
    response = _response(usage=_usage_with_cache())
    result = _adapter_result(response=response, output="hey", pricing=_PRICING)
    assert result.output == "hey"
    assert result.usage.input_tokens_total == 1000
    assert (
        result.usage.cost_in_usd
        == _normalized_usage(_response(usage=_usage_with_cache()), _PRICING).cost_in_usd
    )
    assert result.usage_raw is response.usage
    assert result.stop_reason == "end_turn"
    assert result.raw is response


def test_adapter_result_falls_back_to_zero_usage_without_usage() -> None:
    """A response missing usage normalizes to zero counters and zero cost."""
    result = _adapter_result(response=_response(usage=None), output="hey", pricing=_PRICING)
    assert result.usage.input_tokens_total == 0
    assert result.usage.output_tokens == 0
    assert result.usage.cost_in_usd == 0.0
    assert result.usage_raw is None


def test_wire_input_converts_each_message_kind() -> None:
    """User, assistant (text plus tool calls), and tool messages each map to their items."""
    wire = _wire_input([
        UserMessage(content="q"),
        AssistantMessage(
            turn=(
                TextPart(text="thinking"),
                ToolCall(id="call1", name="lookup", args_json="{}"),
            ),
        ),
        ToolMessage(tool_call_id="call1", content="r"),
    ])
    assert wire == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking"},
        {"type": "function_call", "call_id": "call1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call1", "output": "r"},
    ]


def test_wire_input_converts_tool_result_parts_to_structured_output_content() -> None:
    """A ToolMessage carrying parts becomes a function_call_output whose output is the content list.

    The installed openai SDK's output field accepts input_text and input_image content params,
    so an image reaches the provider as a data: URI.
    A dropped part or mis-encoded image changes this list.
    """
    wire = _wire_input([
        ToolMessage(
            tool_call_id="call1",
            content=(TextPart(text="saw"), ImagePart(data=b"png", media_type="image/png")),
        )
    ])
    assert wire == [
        {
            "type": "function_call_output",
            "call_id": "call1",
            "output": [
                {"type": "input_text", "text": "saw"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}",
                    "detail": "auto",
                },
            ],
        }
    ]


def test_wire_tool_choice_passes_strings_through_and_names_specific_tools() -> None:
    """The neutral strings pass through unchanged; SpecificToolChoice becomes the function form."""
    assert _wire_tool_choice("auto") == "auto"
    assert _wire_tool_choice("required") == "required"
    assert _wire_tool_choice("none") == "none"
    assert _wire_tool_choice(SpecificToolChoice(tool_name="x")) == {
        "type": "function",
        "name": "x",
    }


def _adapter(
    *,
    reasoning_summary: ReasoningSummary | None = None,
    supports_prompt_cache_options: bool = True,
) -> OpenAIResponsesAdapter:
    """Build an adapter over a keyless client, valid because no request is sent.

    supports_prompt_cache_options defaults True, the gpt-5.6-and-later case, so every caller
    that does not name it exercises the path where a binding's caching value reaches the wire.
    """
    return OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=supports_prompt_cache_options,
        reasoning_summary=reasoning_summary,
    )


def _binding(
    *,
    automatic_prompt_caching: bool,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> Binding:
    """Assemble a toolless binding varying only caching, the system prompt, and reasoning effort."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(reasoning_effort=reasoning_effort),
        automatic_prompt_caching=automatic_prompt_caching,
    )


def test_request_sends_a_summary_alone_when_no_effort_is_bound() -> None:
    """reasoning_summary alone still sends the reasoning object, holding only the summary key.

    A summary is reached through the same object effort travels in, so gating that object on effort
    would silently ask for no summary whenever a caller set one without an effort.
    """
    request = _adapter(reasoning_summary="detailed")._request(
        _binding(automatic_prompt_caching=True)
    )
    assert request.reasoning == {"summary": "detailed"}


def test_request_sends_an_effort_alone_without_a_summary_key() -> None:
    """A bound effort with no summary sends the effort key and no summary key.

    An explicit null summary is a different request from omitting the key,
    which is what key-by-key assembly buys over one Reasoning(effort=..., summary=...) call.
    """
    request = _adapter()._request(_binding(automatic_prompt_caching=True, reasoning_effort="high"))
    assert request.reasoning == {"effort": "high"}


def test_request_sends_both_reasoning_keys_when_both_are_set() -> None:
    """Effort and summary set together travel in one reasoning object."""
    request = _adapter(reasoning_summary="auto")._request(
        _binding(automatic_prompt_caching=True, reasoning_effort="low")
    )
    assert request.reasoning == {"effort": "low", "summary": "auto"}


def test_request_omits_reasoning_when_neither_key_is_set() -> None:
    """Neither key set leaves reasoning at the omit sentinel, sending no reasoning object."""
    request = _adapter()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(request.reasoning, openai.Omit)


def test_request_omits_prompt_cache_options_under_automatic_caching() -> None:
    """Automatic caching leaves prompt_cache_options at the omit sentinel."""
    request = _adapter()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(request.prompt_cache_options, openai.Omit)


def test_request_requests_explicit_mode_when_caching_disabled() -> None:
    """Disabled caching sends explicit mode with no breakpoints."""
    request = _adapter()._request(_binding(automatic_prompt_caching=False))
    assert request.prompt_cache_options == {"mode": "explicit"}


def test_request_omits_prompt_cache_options_where_the_model_lacks_the_parameter() -> None:
    """A model not taking prompt_cache_options gets none under either binding value."""
    adapter = _adapter(supports_prompt_cache_options=False)
    disabled = adapter._request(_binding(automatic_prompt_caching=False))
    assert isinstance(disabled.prompt_cache_options, openai.Omit)
    enabled = adapter._request(_binding(automatic_prompt_caching=True))
    assert isinstance(enabled.prompt_cache_options, openai.Omit)


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request; None leaves the omit sentinel.

    The sentinel is what keeps an unstated tier off the wire: sending an explicit null would be a
    different request from omitting the key.
    """
    binding = _binding(automatic_prompt_caching=True)
    assert isinstance(_adapter()._request(binding).service_tier, openai.Omit)
    stated = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=True,
        service_tier="flex",
    )
    assert stated._request(binding).service_tier == "flex"


def test_request_maps_temperature_and_omits_it_when_unset() -> None:
    """A bound temperature lands on the request; None leaves the omit sentinel."""
    unset = _adapter()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(unset.temperature, openai.Omit)
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(temperature=0.2),
        automatic_prompt_caching=True,
    )
    assert _adapter()._request(binding).temperature == 0.2


def test_request_omits_tool_fields_without_tools() -> None:
    """No tools leaves tools, tool_choice, and parallel_tool_calls at the omit sentinel."""
    request = _adapter()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(request.tools, openai.Omit)
    assert isinstance(request.tool_choice, openai.Omit)
    assert isinstance(request.parallel_tool_calls, openai.Omit)


class _FakeSDKStream(AsyncResponseStream[None]):
    """Replays constructed events without a connection.

    Overrides exactly the surface _OpenAIStream uses (iteration and close);
    the base __init__ is deliberately not called, so the untouched base machinery stays unusable.
    """

    def __init__(self, replay_events: Sequence[ResponseStreamEvent]) -> None:
        self._replay_events = list(replay_events)

    @override
    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        for replay_event in self._replay_events:
            yield replay_event

    @override
    async def close(self) -> None:
        return


def _stream(replay_events: Sequence[ResponseStreamEvent]) -> _OpenAIStream[str]:
    """Build a text-content adapter stream over replayed events.

    The extractor is the shipped _BoundOpenAIText one, so these tests pin what a text stream
    reports for each terminal status.
    """
    adapter = _adapter()
    text_bound = _BoundOpenAIText(
        adapter=adapter, request=adapter._request(_binding(automatic_prompt_caching=True))
    )
    return _OpenAIStream(
        sdk_stream=_FakeSDKStream(replay_events),
        pricing=_PRICING,
        output_from_response=text_bound._text_outcome,
    )


def _collected_items(replay_events: Sequence[ResponseStreamEvent]) -> list[StreamItem]:
    """Drain the translated items into a list."""

    async def scenario() -> list[StreamItem]:
        return [item async for item in _stream(replay_events).items()]

    return asyncio.run(scenario())


def _text_delta_event(delta: str, sequence_number: int) -> AccumulatedResponseTextDeltaEvent:
    """Build one accumulated text-delta event."""
    return AccumulatedResponseTextDeltaEvent(
        type="response.output_text.delta",
        delta=delta,
        snapshot=delta,
        content_index=0,
        item_id="m1",
        output_index=0,
        logprobs=[],
        sequence_number=sequence_number,
    )


def _completed_event(
    response: OpenAIResponse, sequence_number: int
) -> AccumulatedResponseCompletedEvent[None]:
    """Wrap a response in the terminal completed event the SDK stream yields."""
    return AccumulatedResponseCompletedEvent[None](
        type="response.completed",
        response=ParsedResponse[None].model_validate(response.model_dump()),
        sequence_number=sequence_number,
    )


def test_stream_passes_text_deltas_through_as_bare_strings() -> None:
    """Text deltas pass through in order as the SDK's own strings; nothing follows them."""
    translated = _collected_items([
        _text_delta_event("he", 1),
        _text_delta_event("y", 2),
        _completed_event(_response(usage=_usage_with_cache()), 3),
    ])
    assert translated == ["he", "y"]


def test_stream_yields_one_complete_tool_call_and_ignores_message_items() -> None:
    """A function_call done event yields one complete ToolCall; message item lifecycles are dropped."""
    message_added = ResponseOutputItemAddedEvent.model_validate({
        "type": "response.output_item.added",
        "item": _TEXT_OUTPUT_ITEM,
        "output_index": 0,
        "sequence_number": 1,
    })
    function_call_added = ResponseOutputItemAddedEvent.model_validate({
        "type": "response.output_item.added",
        "item": _FUNCTION_CALL_OUTPUT_ITEM,
        "output_index": 1,
        "sequence_number": 2,
    })
    message_done = ResponseOutputItemDoneEvent.model_validate({
        "type": "response.output_item.done",
        "item": _TEXT_OUTPUT_ITEM,
        "output_index": 0,
        "sequence_number": 3,
    })
    function_call_done = ResponseOutputItemDoneEvent.model_validate({
        "type": "response.output_item.done",
        "item": _FUNCTION_CALL_OUTPUT_ITEM,
        "output_index": 1,
        "sequence_number": 4,
    })
    translated = _collected_items([
        message_added,
        function_call_added,
        message_done,
        function_call_done,
        _completed_event(
            _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]), 5
        ),
    ])
    assert translated == [ToolCall(id="call1", name="lookup", args_json='{"q": 1}')]


def test_stream_incomplete_terminal_still_assembles_final() -> None:
    """An incomplete terminal yields no item, and final() must not raise.

    The SDK's get_final_response() raises RuntimeError unless the terminal event is response.completed,
    so final() assembles from the captured terminal response instead.
    """

    async def scenario() -> None:
        incomplete_response = _response(
            usage=_usage_with_cache(),
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
        adapter_stream = _stream([
            _text_delta_event("he", 1),
            ResponseIncompleteEvent(
                type="response.incomplete", response=incomplete_response, sequence_number=2
            ),
        ])
        translated = [item async for item in adapter_stream.items()]
        assert translated == ["he"]
        result = _assert_result(await adapter_stream.final())
        assert result.output == "hey"
        assert result.stop_reason == "max_tokens"
        assert result.usage.input_tokens_total == 1000

    asyncio.run(scenario())


def test_final_after_completed_terminal_assembles_from_the_parsed_response() -> None:
    """The completed terminal already carries a ParsedResponse; final() assembles from it."""

    async def scenario() -> None:
        adapter_stream = _stream([
            _text_delta_event("he", 1),
            _completed_event(_response(usage=_usage_with_cache()), 2),
        ])
        translated = [item async for item in adapter_stream.items()]
        assert translated == ["he"]
        result = _assert_result(await adapter_stream.final())
        assert result.output == "hey"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens_total == 1000
        assert (
            result.usage.cost_in_usd
            == _normalized_usage(_response(usage=_usage_with_cache()), _PRICING).cost_in_usd
        )

    asyncio.run(scenario())


def test_stream_final_turn_carries_reasoning() -> None:
    """final()'s assistant turn includes the ReasoningTrace from the terminal response's output."""

    async def scenario() -> None:
        adapter_stream = _stream([
            _completed_event(
                _response(usage=None, output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM]), 1
            ),
        ])
        async for _item in adapter_stream.items():
            pass
        result = _assert_result(await adapter_stream.final())
        reasoning_trace = result.assistant_message.turn[0]
        assert isinstance(reasoning_trace, ReasoningTrace)
        assert reasoning_trace.reasoning == _REASONING_OUTPUT_ITEM

    asyncio.run(scenario())


def test_stream_failed_terminal_is_terminal_and_reports_the_provider_failure() -> None:
    """A failed terminal ends the stream without a StreamProtocolError, and final() reports the failure.

    The API reported the run as not finished, so whatever text had accumulated is a fragment;
    returning it as a Response would present that fragment as the turn. The member carries the
    response's billing, so the attempt is paid for.
    """

    async def scenario() -> None:
        adapter_stream = _stream([
            ResponseFailedEvent(
                type="response.failed",
                response=_response(
                    usage=_usage_with_cache(), status="failed", error=_SERVER_ERROR
                ),
                sequence_number=1,
            ),
        ])
        translated = [item async for item in adapter_stream.items()]
        assert translated == []
        outcome = await adapter_stream.final()
        assert isinstance(outcome, ProviderFailedTransiently)
        assert outcome.usage.input_tokens_total == 1000

    asyncio.run(scenario())


def test_stream_final_passes_a_leniently_built_terminal_through_unvalidated() -> None:
    """A terminal response holding an output item the strict model rejects still assembles.

    The SDK builds a non-completed terminal response leniently, tolerating an item type it does not
    model, so validating that response against the SDK's own strict model would raise
    ValidationError and destroy a partial answer the caller has already been billed for.
    """
    unmodelled_item: dict[str, object] = {"type": "quantum_tool_call", "id": "q1"}
    leniently_built = construct_type_unchecked(
        type_=OpenAIResponse,
        value={
            "id": "r1",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [_TEXT_OUTPUT_ITEM, unmodelled_item],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": None,
        },
    )

    async def scenario() -> None:
        adapter_stream = _stream([
            ResponseIncompleteEvent.construct(
                type="response.incomplete", response=leniently_built, sequence_number=1
            ),
        ])
        async for _item in adapter_stream.items():
            pass
        result = _assert_result(await adapter_stream.final())
        assert result.output == "hey"
        assert result.stop_reason == "max_tokens"
        assert result.raw is leniently_built

    asyncio.run(scenario())


def test_stream_without_terminal_raises() -> None:
    """Ending without any terminal event is a protocol violation."""
    with pytest.raises(StreamProtocolError):
        _collected_items([_text_delta_event("he", 1)])


def test_final_before_items_are_exhausted_raises() -> None:
    """final() needs the captured terminal response, so it demands drained items."""

    async def scenario() -> None:
        adapter_stream = _stream([_completed_event(_response(usage=None), 1)])
        with pytest.raises(StreamProtocolError):
            await adapter_stream.final()

    asyncio.run(scenario())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundOpenAIStructured[_StructuredReport]:
    """Build a structured-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    request = adapter._request(_binding(automatic_prompt_caching=False, system_prompt="sys"))
    return _BoundOpenAIStructured(
        adapter=adapter, request=request, response_format=_StructuredReport
    )


_REPORT_JSON = '{"city": "Nairobi", "celsius": 25}'
"""Text that validates into _StructuredReport."""


def _structured_response(
    text: str | None,
    *,
    refusal: bool = False,
    status: ResponseStatus = "completed",
    incomplete_details: IncompleteDetails | None = None,
    usage: ResponseUsage | None = None,
    tool_call: bool = False,
    error: ResponseError | None = None,
) -> OpenAIResponse:
    """Build a response whose message carries the given output text, or a refusal block.

    text None gives a message with no content at all, the turn that carried nothing to validate;
    refusal replaces the text part with a refusal part.
    tool_call appends a function_call output item, which is what makes the turn a tool call.
    """
    content: list[object] = []
    if refusal:
        content.append({"type": "refusal", "refusal": "I can't help with that"})
    elif text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
    message: dict[str, object] = {
        "id": "m1",
        "role": "assistant",
        "status": "completed",
        "type": "message",
        "content": content,
    }
    return _response(
        usage=usage,
        output=[message, *([_FUNCTION_CALL_OUTPUT_ITEM] if tool_call else [])],
        status=status,
        incomplete_details=incomplete_details,
        error=error,
    )


def test_structured_bind_builds_the_schema_the_sdk_parse_helper_builds() -> None:
    """The text parameter equals what responses.parse(text_format=Model) would have sent.

    The adapter sends the schema itself so it can validate the response text in its own frame; this
    is what keeps the request unchanged by that move.
    """
    assert _structured_bound()._text == {"format": type_to_text_format_param(_StructuredReport)}


def test_structured_bind_validates_the_turns_text_into_the_instance() -> None:
    """The structured bound adapter validates the turn's output text into the response_format."""
    outcome = _structured_bound()._parsed_output(_structured_response(_REPORT_JSON))
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_bind_reports_empty_turn_when_the_turn_carried_no_text() -> None:
    """A completed response with no text part and no tool call is EmptyTurn."""
    outcome = _structured_bound()._parsed_output(_structured_response(None))
    assert isinstance(outcome, EmptyTurn)


def test_structured_bind_reports_schema_violation_on_text_the_model_rejects() -> None:
    """A completed turn whose text the response_format rejects is SchemaViolation, carrying its billing.

    validation_error_json names the rejected field, what rejected it, and the value, which is what
    tells a caller whether to change the model or the prompt.
    """
    outcome = _structured_bound()._parsed_output(
        _structured_response(
            '{"city": "Nairobi", "celsius": "SENTINEL"}', usage=_usage_with_cache()
        )
    )
    assert isinstance(outcome, SchemaViolation)
    rejections = json.loads(outcome.validation_error_json)
    assert [rejection["loc"] for rejection in rejections] == [["celsius"]]
    assert rejections[0]["input"] == "SENTINEL"
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_max_completion_tokens_exceeded_on_text_cut_mid_json() -> None:
    """An incomplete turn whose JSON stopped mid-object is the truncation, not a schema violation.

    This is the response the SDK's own parse raised on, losing the 200 and its billing; the member
    carries both, and the item fails with MaxCompletionTokensExceededError.
    """
    outcome = _structured_bound()._parsed_output(
        _structured_response(
            '{"city": "Nair',
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
            usage=_usage_with_cache(),
        )
    )
    assert isinstance(outcome, MaxCompletionTokensExceeded)
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_a_tool_call_turn_as_none() -> None:
    """A completed turn whose output is a function call parses no instance and nothing went wrong."""
    assert _structured_bound()._parsed_output(_structured_response(None, tool_call=True)) is None


def test_structured_bind_reports_a_tool_call_turn_whose_text_is_not_the_instance_as_none() -> None:
    """A tool-call turn whose text is prose is the tool call, not a schema violation."""
    outcome = _structured_bound()._parsed_output(
        _structured_response("let me look that up", tool_call=True)
    )
    assert outcome is None


def test_structured_bind_sets_output_on_a_turn_that_also_called_a_tool() -> None:
    """The instance lands on output and the call still lands on tool_calls, so neither fact hides the other."""
    outcome = _structured_bound()._parsed_output(
        _structured_response(_REPORT_JSON, tool_call=True)
    )
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_stream_terminal_reports_max_completion_tokens_exceeded() -> None:
    """A structured stream's incomplete terminal is MaxCompletionTokensExceeded.

    The stream sends no text_format, so its terminal response is a plain Response on every status and
    one function reads them all.
    """
    outcome = _structured_bound()._parsed_output(
        _structured_response(
            None,
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
    )
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_stream_terminal_reports_a_failed_run_as_the_provider_failure() -> None:
    """A structured stream's failed terminal takes the same member the text binding reports."""
    outcome = _structured_bound()._parsed_output(
        _structured_response(None, status="failed", error=_SERVER_ERROR)
    )
    assert isinstance(outcome, ProviderFailedTransiently)


def _text_bound() -> _BoundOpenAIText:
    """Build a text-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    return _BoundOpenAIText(
        adapter=adapter, request=adapter._request(_binding(automatic_prompt_caching=False))
    )


def test_text_bind_reports_the_refusal_sentences_as_the_output() -> None:
    """A refused turn's output is the text the model wrote, not the empty output_text.

    The Response the caller reads then carries the same sentences in output and in
    assistant_message.text, which is what a refusal under the anthropic adapter carries too.
    """
    response = _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM])
    assert _text_bound()._text_outcome(response) == "I can't help with that"


def test_text_bind_send_reports_a_failed_status_as_the_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed status returns the failure member from send, carrying that run's billing."""
    text_bound = _text_bound()

    async def fake_create(**_request_kwargs: object) -> OpenAIResponse:
        """Return a response the API reported as failed, with real usage on it."""
        return _response(usage=_usage_with_cache(), status="failed", error=_SERVER_ERROR)

    monkeypatch.setattr(text_bound._adapter.client.responses, "create", fake_create)
    outcome = asyncio.run(text_bound.send([UserMessage(content="q")]))
    assert isinstance(outcome, ProviderFailedTransiently)
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_the_failure_on_a_failed_status_whose_text_validates() -> None:
    """A failed run is the failure member even when its fragment validates: it is not the answer."""
    outcome = _structured_bound()._parsed_output(
        _structured_response(
            _REPORT_JSON, status="failed", usage=_usage_with_cache(), error=_SERVER_ERROR
        )
    )
    assert isinstance(outcome, ProviderFailedTransiently)
    assert outcome.usage.cost_in_usd > 0.0


def test_a_failed_run_carrying_a_refusal_takes_the_failure_member_under_both_bindings() -> None:
    """A failed status wins over the refusal test, so one response does not split by binding.

    Were the refusal tested first, the structured binding would report Refusal (a terminal
    RefusalError) for a response the text binding retries.
    """
    structured_outcome = _structured_bound()._parsed_output(
        _structured_response(
            None, refusal=True, status="failed", usage=_usage_with_cache(), error=_SERVER_ERROR
        )
    )
    assert isinstance(structured_outcome, ProviderFailedTransiently)
    text_outcome = _text_bound()._text_outcome(
        _response(
            usage=_usage_with_cache(),
            output=[_REFUSAL_MESSAGE_ITEM],
            status="failed",
            error=_SERVER_ERROR,
        )
    )
    assert isinstance(text_outcome, ProviderFailedTransiently)


def test_a_transient_error_code_carries_the_providers_message_and_no_rate_limit_flag() -> None:
    """A server_error is transient, and its reason is openai's message verbatim."""
    outcome = _text_bound()._text_outcome(
        _response(usage=_usage_with_cache(), status="failed", error=_SERVER_ERROR)
    )
    assert isinstance(outcome, ProviderFailedTransiently)
    assert outcome.reason == _SERVER_ERROR.message
    assert outcome.is_rate_limit is False


def test_a_rate_limit_error_code_sets_the_rate_limit_flag() -> None:
    """rate_limit_exceeded is transient and flags the rate limit, which paces every sharing task."""
    outcome = _text_bound()._text_outcome(
        _response(
            usage=_usage_with_cache(),
            status="failed",
            error=ResponseError(code="rate_limit_exceeded", message="Rate limit reached."),
        )
    )
    assert isinstance(outcome, ProviderFailedTransiently)
    assert outcome.is_rate_limit is True


def test_a_terminal_error_code_carries_the_providers_message_without_retrying() -> None:
    """failed_to_download_image is terminal: the request names the image, so a resend fetches it again."""
    outcome = _text_bound()._text_outcome(
        _response(usage=_usage_with_cache(), status="failed", error=_IMAGE_DOWNLOAD_ERROR)
    )
    assert isinstance(outcome, ProviderFailedTerminally)
    assert outcome.reason == _IMAGE_DOWNLOAD_ERROR.message


def test_an_error_code_the_installed_sdk_does_not_name_is_terminal() -> None:
    """A code added after openai 2.45.0 fails the item once rather than spending the retry budget."""
    outcome = _text_bound()._text_outcome(
        _response(
            usage=_usage_with_cache(),
            status="failed",
            error=ResponseError.construct(code="a_code_from_a_later_sdk", message="Something."),
        )
    )
    assert isinstance(outcome, ProviderFailedTerminally)
    assert outcome.reason == "Something."


def test_a_failed_status_with_no_error_object_is_terminal() -> None:
    """A failed run naming nothing gives no ground to resend on, and says that in its reason."""
    outcome = _text_bound()._text_outcome(_response(usage=_usage_with_cache(), status="failed"))
    assert isinstance(outcome, ProviderFailedTerminally)
    assert outcome.reason == "openai reported status 'failed' and no error object"


def test_a_run_that_stopped_short_of_a_turn_is_unfinished_turn_naming_the_status() -> None:
    """A cancelled run is neither a failure openai described nor a turn, so it names its status."""
    outcome = _structured_bound()._parsed_output(
        _structured_response(None, status="cancelled", usage=_usage_with_cache())
    )
    assert isinstance(outcome, UnfinishedTurn)
    assert outcome.reason == "openai returned status 'cancelled'"


def test_structured_bind_reports_refusal_on_a_refusal_block() -> None:
    """A response carrying a refusal content block is Refusal, carrying its billing."""
    outcome = _structured_bound()._parsed_output(
        _structured_response(None, refusal=True, usage=_usage_with_cache())
    )
    assert isinstance(outcome, Refusal)
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_max_completion_tokens_exceeded_on_a_max_output_tokens_incomplete() -> (
    None
):
    """An incomplete response for max_output_tokens is MaxCompletionTokensExceeded, carrying its billing."""
    outcome = _structured_bound()._parsed_output(
        _structured_response(
            None,
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
            usage=_usage_with_cache(),
        )
    )
    assert isinstance(outcome, MaxCompletionTokensExceeded)
    assert outcome.usage.cost_in_usd > 0.0


def test_structured_bind_reports_refusal_on_a_content_filter_incomplete() -> None:
    """An incomplete response for content_filter is Refusal, so the item fails once.

    Retrying would send the same blocked request again for the whole retry budget
    and bill for each attempt.
    """
    outcome = _structured_bound()._parsed_output(
        _structured_response(
            None,
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="content_filter"),
            usage=_usage_with_cache(),
        )
    )
    assert isinstance(outcome, Refusal)
    assert outcome.usage.cost_in_usd > 0.0


def test_every_request_carries_the_reasoning_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create and stream, on both bindings, all send include=["reasoning.encrypted_content"].

    The offline round-trip tests cannot catch a dropped include:
    the SDK documents include as what populates encrypted_content,
    so without this parameter every replayed reasoning item could be silently empty.
    """
    adapter = _adapter()
    request = adapter._request(_binding(automatic_prompt_caching=True))
    includes: list[object] = []

    async def fake_create(**request_kwargs: object) -> OpenAIResponse:
        includes.append(request_kwargs["include"])
        return _response(usage=None)

    class _FakeStreamManager:
        async def __aenter__(self) -> _FakeSDKStream:
            return _FakeSDKStream([])

    def fake_stream(**request_kwargs: object) -> _FakeStreamManager:
        includes.append(request_kwargs["include"])
        return _FakeStreamManager()

    monkeypatch.setattr(adapter.client.responses, "create", fake_create)
    monkeypatch.setattr(adapter.client.responses, "stream", fake_stream)
    text_bound = _BoundOpenAIText(adapter=adapter, request=request)
    structured_bound = _BoundOpenAIStructured(
        adapter=adapter, request=request, response_format=_StructuredReport
    )

    async def scenario() -> None:
        conversation = [UserMessage(content="q")]
        await text_bound.send(conversation)
        await text_bound.open_stream(conversation)
        await structured_bound.send(conversation)
        await structured_bound.open_stream(conversation)

    asyncio.run(scenario())
    assert includes == [["reasoning.encrypted_content"]] * 4


def test_both_structured_request_paths_send_the_text_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send and open_stream both put the precomputed text parameter on the request.

    A path that dropped it would ask for no schema at all, and every turn would come back as prose
    the response_format rejects, reported as the caller's model being wrong rather than as the
    request that omitted its schema.
    """
    adapter = _adapter()
    request = adapter._request(_binding(automatic_prompt_caching=True))
    texts: list[object] = []

    async def fake_create(**request_kwargs: object) -> OpenAIResponse:
        texts.append(request_kwargs["text"])
        return _response(usage=None)

    class _FakeStreamManager:
        async def __aenter__(self) -> _FakeSDKStream:
            return _FakeSDKStream([])

    def fake_stream(**request_kwargs: object) -> _FakeStreamManager:
        texts.append(request_kwargs["text"])
        return _FakeStreamManager()

    monkeypatch.setattr(adapter.client.responses, "create", fake_create)
    monkeypatch.setattr(adapter.client.responses, "stream", fake_stream)
    structured_bound = _BoundOpenAIStructured(
        adapter=adapter, request=request, response_format=_StructuredReport
    )

    async def scenario() -> None:
        conversation = [UserMessage(content="q")]
        await structured_bound.send(conversation)
        await structured_bound.open_stream(conversation)

    asyncio.run(scenario())
    assert texts == [{"format": type_to_text_format_param(_StructuredReport)}] * 2


def _rate_limit_error(headers: dict[str, str]) -> openai.RateLimitError:
    """Build the SDK's 429 exception around a constructed httpx response."""
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


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


def _status_error[ErrorT: openai.APIStatusError](
    error_class: type[ErrorT],
    status_code: int,
    headers: dict[str, str] | None = None,
) -> ErrorT:
    """Build one of the SDK's status exceptions around a constructed httpx response."""
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        headers=headers,
    )
    return error_class("boom", response=response, body=None)


def _connection_error() -> openai.APIConnectionError:
    """Build the SDK's transport-failure exception, which carries a request and no response."""
    return openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_status_error(openai.RateLimitError, 429), "rate_limit"),
        (_status_error(openai.InternalServerError, 500), "transient"),
        (_connection_error(), "transient"),
        (openai.APITimeoutError(httpx.Request("POST", "https://api.openai.com")), "transient"),
        (_status_error(openai.ConflictError, 409), "transient"),
        (_status_error(openai.BadRequestError, 400), "invalid_request"),
        (_status_error(openai.AuthenticationError, 401), "invalid_request"),
        (_status_error(openai.PermissionDeniedError, 403), "invalid_request"),
        (_status_error(openai.NotFoundError, 404), "invalid_request"),
        (_status_error(openai.UnprocessableEntityError, 422), "invalid_request"),
        (_status_error(openai.APIStatusError, 413), "invalid_request"),
        (_status_error(openai.APIStatusError, 408), "transient"),
        (_status_error(openai.InternalServerError, 503), "transient"),
        (_status_error(openai.APIStatusError, 302), "unrecognized"),
        (_status_error(openai.BadRequestError, 400, {"x-should-retry": "true"}), "transient"),
        (
            _status_error(openai.InternalServerError, 500, {"x-should-retry": "false"}),
            "unrecognized",
        ),
        (_status_error(openai.RateLimitError, 429, {"x-should-retry": "false"}), "rate_limit"),
        (_status_error(openai.RateLimitError, 429, {"x-should-retry": "true"}), "rate_limit"),
        (ValueError("boom"), "unrecognized"),
    ],
)
def test_classify_maps_each_sdk_exception_to_its_classification(
    error: Exception, expected: ErrorClassification
) -> None:
    """Each error lands on the classification the adapter's classify docstring names.

    Each status code is the one the SDK raises that class for, read from openai 2.45.0;
    the bare APIStatusError rows are the statuses the SDK maps to no class of its own, 413 among
    them, which is why the adapter reads the status rather than the exception class.
    APITimeoutError subclasses APIConnectionError, so timeouts reach transient through that isinstance.
    x-should-retry overrides the status in both directions, except on a rate-limit status, which
    stays rate_limit whatever the header says, so the limiter's account-wide pause is still armed.
    A 3xx and the non-SDK ValueError land on the unrecognized default,
    which fails the one item without a retry.
    """
    assert _adapter().classify(error) == expected


def test_adapter_pins_sdk_retries_off() -> None:
    """The stored client copy carries max_retries=0 so only langchaint retries."""
    assert _adapter().client.max_retries == 0


def test_wire_input_marks_marked_user_and_tool_parts() -> None:
    """A marked part carries prompt_cache_breakpoint on its wire part; unmarked siblings carry none."""
    wire = _wire_input([
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                TextPart(text="question"),
            )
        ),
        ToolMessage(
            tool_call_id="c1",
            content=(
                TextPart(text="saw"),
                ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
            ),
        ),
    ])
    assert wire == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "shared context",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "input_text", "text": "question"},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [
                {"type": "input_text", "text": "saw"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}",
                    "detail": "auto",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
            ],
        },
    ]


def test_wire_input_sends_every_mark_without_a_client_side_cap() -> None:
    """The server keeps the latest breakpoints itself, so all five marks go to the wire."""
    wire = _wire_input([
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(5))
        ),
    ])
    assert wire == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"m{index}",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
                for index in range(5)
            ],
        }
    ]


def test_request_system_parts_become_a_developer_input_message() -> None:
    """A parts system_prompt travels as a developer-role input message; instructions stays unset."""
    request = _adapter()._request(
        _binding(
            automatic_prompt_caching=True,
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            ),
        )
    )
    assert request.instructions is None
    assert request.input_prefix == [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "stable instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "input_text", "text": "semi-stable context"},
            ],
        }
    ]


def test_request_str_system_travels_as_instructions_with_an_empty_prefix() -> None:
    """A str system_prompt keeps the instructions mapping and sends no prefix item."""
    request = _adapter()._request(_binding(automatic_prompt_caching=True, system_prompt="sys"))
    assert request.instructions == "sys"
    assert request.input_prefix == []
