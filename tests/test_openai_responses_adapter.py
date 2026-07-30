"""OpenAI Responses adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot:
the usage partition derived by subtracting cache counts from input_tokens and its cross-check, cost arithmetic,
input-item placement, tool-choice translation, stop-reason derivation (the API reports no finish reason),
the zero-usage fallback when a response omits usage, and the request fields the binding precomputes.
"""

import asyncio
import base64
import json
import math
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Literal, override

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
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response import IncompleteDetails, ResponseStatus
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from langchaint import (
    LLM,
    AssistantMessage,
    ImagePart,
    InferenceParams,
    ReasoningDelta,
    ReasoningEffort,
    ReasoningTrace,
    SpecificToolChoice,
    StopReason,
    StreamItem,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    EmptyTurn,
    ErrorClassification,
    MaxCompletionTokensExceeded,
    NoOutputOutcome,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.call import ResponseIdentity
from langchaint.conformance import AdapterConformance
from langchaint.exceptions import StreamProtocolError
from langchaint.openai import (
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.responses_adapter import (
    _assistant_items,
    _assistant_message_from,
    _billing_from_response,
    _BoundOpenAIStructured,
    _BoundOpenAIText,
    _normalized_stop_reason,
    _OpenAIRequestParams,
    _OpenAIStream,
    _wire_input,
    _wire_tool_choice,
)

_DEFAULT_RATES = OpenAIPricingTable(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)

_PRICING: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {"default": _DEFAULT_RATES}
"""The default tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = OpenAIPricingTable(
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
    model: str = "m",
) -> OpenAIResponse:
    """Build a response whose id is fixed at "r1"; every field a test varies is a parameter."""
    return OpenAIResponse.model_validate({
        "id": "r1",
        "created_at": 0,
        "model": model,
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


def test_billing_subtracts_cache_from_input_tokens_and_prices() -> None:
    """The uncached counter is input_tokens minus both cache counts, and the priced cost rides on it."""
    usage = _billing_from_response(_response(usage=_usage_with_cache()), _PRICING).usage
    assert usage.input_tokens_cache_read == 600
    assert usage.input_tokens_cache_write == 100
    assert usage.input_tokens_cache_none == 300
    assert usage.input_tokens_total == 1000
    assert usage.cost_in_usd == pytest.approx(
        (300 * 2.5 + 600 * 1.25 + 100 * 3.125 + 40 * 10.0) / 1e6
    )


def test_billing_carries_the_sdk_usage_object_itself() -> None:
    """usage_raw is the response's own ResponseUsage by reference, and None where it reported none."""
    raw = _response(usage=_usage_with_cache())
    assert _billing_from_response(raw, _PRICING).usage_raw is raw.usage
    assert _billing_from_response(_response(usage=None), _PRICING).usage_raw is None


def test_billing_pins_the_priced_tier_and_the_rates_that_applied() -> None:
    """The Billing holds the priced tier, not the reported value, and the four rates behind its costs.

    A response reporting "auto" prices at "default", and the reported value survives on raw.
    """
    raw = _response(usage=_usage_with_cache(), service_tier="auto")
    billing = _billing_from_response(raw, _PRICING)
    assert billing.service_tier == "default"
    assert billing.input_cache_none_usd_per_million_tokens == 2.5
    assert billing.cache_read_usd_per_million_tokens == 1.25
    assert billing.cache_write_usd_per_million_tokens == 3.125
    assert billing.output_usd_per_million_tokens == 10.0
    assert raw.service_tier == "auto"


def test_billing_reads_reasoning_tokens() -> None:
    """output_tokens_reasoning reads the required reasoning_tokens counter."""
    usage = _billing_from_response(
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
    ).usage
    assert usage.output_tokens_reasoning == 8


def test_billing_without_usage_pins_the_priced_tiers_rates() -> None:
    """A response missing usage still stores the rates the tier that served it would have spent."""
    billing = _billing_from_response(_response(usage=None), _PRICING)
    assert billing.output_usd_per_million_tokens == 10.0
    assert billing.service_tier == "default"


def test_a_response_that_billed_nothing_at_an_unpriced_tier_still_costs_zero() -> None:
    """Every counter zero leaves the total a number even where no rate priced the tier."""
    billing = _billing_from_response(_response(usage=None, service_tier="flex"), _PRICING)
    assert math.isnan(billing.output_usd_per_million_tokens)
    assert billing.usage.cost_in_usd == 0.0


def test_the_categories_are_priced_apart() -> None:
    """Cache reads, cache writes, the uncached remainder, and output each bill at their rate."""
    usage = _billing_from_response(_response(usage=_usage_with_cache()), _PRICING).usage
    assert usage.input_tokens_cache_none_cost_in_usd == 300 * 2.5 / 1e6
    assert usage.input_tokens_cache_read_cost_in_usd == 600 * 1.25 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == 100 * 3.125 / 1e6
    assert usage.output_tokens_cost_in_usd == 40 * 10.0 / 1e6


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response; "auto" and no tier both price at default."""
    pricing: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
        "default": _DEFAULT_RATES,
        "priority": _PRIORITY_RATES,
    }
    at_priority = _billing_from_response(
        _response(usage=_usage_with_cache(), service_tier="priority"), pricing
    ).usage
    at_default = _billing_from_response(
        _response(usage=_usage_with_cache(), service_tier="default"), pricing
    ).usage
    reporting_auto = _billing_from_response(
        _response(usage=_usage_with_cache(), service_tier="auto"), pricing
    ).usage
    reporting_none = _billing_from_response(_response(usage=_usage_with_cache()), pricing).usage
    assert at_priority.cost_in_usd == pytest.approx(2 * at_default.cost_in_usd)
    assert reporting_auto.cost_in_usd == at_default.cost_in_usd
    assert reporting_none.cost_in_usd == at_default.cost_in_usd


def test_an_unpriced_tier_keeps_its_counters_and_its_name() -> None:
    """A response served at a tier no table prices still reports what it billed and who served it."""
    billing = _billing_from_response(
        _response(usage=_usage_with_cache(), service_tier="flex"), _PRICING
    )
    assert billing.service_tier == "flex"
    assert billing.usage.input_tokens_total == 1000
    assert billing.usage.output_tokens == 40


def test_pricing_without_the_default_key_raises_at_construction() -> None:
    """A pricing mapping missing "default" fails before any request, naming the model."""
    priority_only: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
        "priority": _PRIORITY_RATES
    }
    with pytest.raises(ValueError, match=re.escape("'default'")):
        OpenAIResponsesAdapter(
            client=AsyncOpenAI(api_key="test"),
            model="gpt-5.6-terra",
            pricing=priority_only,
            provider_name="openai",
            supports_prompt_cache_options=True,
        )


_REFUSAL_MESSAGE_ITEM: dict[str, object] = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": "I can't help with that"}],
}


def _incomplete_response(reason: Literal["max_output_tokens", "content_filter"]) -> OpenAIResponse:
    """Build an incomplete response reporting one of the two reasons the API declares."""
    return _response(
        usage=None, status="incomplete", incomplete_details=IncompleteDetails(reason=reason)
    )


@pytest.mark.parametrize(
    ("build_response", "expected"),
    [
        (
            lambda: _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]),
            "tool_use",
        ),
        (lambda: _response(usage=None), "end_turn"),
        (lambda: _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM]), "refusal"),
        (
            lambda: _response(
                usage=None, output=[_REFUSAL_MESSAGE_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]
            ),
            "refusal",
        ),
        (lambda: _incomplete_response("max_output_tokens"), "max_tokens"),
        (lambda: _incomplete_response("content_filter"), "refusal"),
        (lambda: _response(usage=None, status="failed"), "other"),
    ],
    ids=[
        "function_call_item",
        "completed",
        "refusal_block",
        "refusal_block_beside_a_function_call",
        "incomplete_max_output_tokens",
        "incomplete_content_filter",
        "failed",
    ],
)
def test_stop_reason_mapping(
    build_response: Callable[[], OpenAIResponse], expected: StopReason
) -> None:
    """The API reports no finish reason field, so each stop reason is derived from the response.

    The refusal check runs ahead of the tool-call check, which the row carrying both pins:
    a filtered turn is a refusal whether or not the model also called a tool.
    """
    assert _normalized_stop_reason(build_response()) == expected


def test_assistant_message_carries_the_refusal_text_and_replays_it() -> None:
    """A refusal content part becomes a TextPart, so the refused turn replays as the model wrote it.

    Dropped instead, the turn holds no elements and sends nothing back, which reopens the
    Sequence[Message] at the point the model declined.
    """
    assistant_message = _assistant_message_from(
        _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM])
    )
    assert assistant_message.turn == (TextPart(text="I can't help with that"),)
    assert _assistant_items(assistant_message) == [
        {"role": "assistant", "content": "I can't help with that"}
    ]


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
    assert reasoning_trace.raw == _REASONING_OUTPUT_ITEM
    assert assistant_message.text == "hey"
    assert assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )
    items = _assistant_items(assistant_message)
    assert len(items) == len(response.output)
    assert items[0] == reasoning_trace.raw
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


@pytest.mark.parametrize(
    ("summary", "content", "expected_text"),
    [
        (("thought it over",), None, "thought it over"),
        (
            ("**Reading the question**\n\nFirst.", "**Answering**\n\nThen."),
            None,
            "**Reading the question**\n\nFirst.\n\n**Answering**\n\nThen.",
        ),
        (("the summary",), ("the content",), "the content"),
        (("thought it over",), ("",), "thought it over"),
        ((), ("worked it out",), "worked it out"),
        ((), None, None),
        (("", ""), None, None),
        ((), ("", ""), None),
        (("", "real"), None, "real"),
    ],
    ids=[
        "one_summary_part",
        "several_summary_parts",
        "content_beside_a_summary",
        "text_free_content_beside_a_summary",
        "content_with_an_empty_summary",
        "neither_list_holds_a_part",
        "every_summary_part_empty",
        "every_content_part_empty",
        "an_empty_part_beside_a_real_one",
    ],
)
def test_reasoning_trace_text_takes_the_content_over_the_summary_and_is_none_without_text(
    summary: tuple[str, ...], content: tuple[str, ...] | None, expected_text: str | None
) -> None:
    r"""Parts join on a blank line, and a join that holds no text leaves trace.text None.

    Content is the reasoning the summary is a rendering of, so it wins wherever it has text.
    Branching on whether the content list is present, rather than on the text it joins to, drops a
    summary into an unreportable None whenever content is present but empty.
    Parts join on a blank line because each is a separately delimited unit, and joining on the empty
    string runs one part's last line into the next one's first.
    Present-but-empty parts are what a trailing falsy check alone misses: two of them join into
    "\n\n", which is truthy, so it would reach a span as a reasoning part carrying nothing.
    """
    response = _response(usage=None, output=[_reasoning_item(summary=summary, content=content)])
    trace = _assistant_message_from(response).turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == expected_text


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


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """An anthropic-produced trace emits its dict as-is; the API rejects the unknown type key, not this adapter."""
    raw: dict[str, object] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(turn=(ReasoningTrace(raw=raw), TextPart(text="hi")))
    assert _assistant_items(assistant_message) == [
        raw,
        {"role": "assistant", "content": "hi"},
    ]


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


@pytest.mark.parametrize(
    ("reasoning_summary", "reasoning_effort", "expected_reasoning"),
    [
        ("detailed", None, {"summary": "detailed"}),
        (None, "high", {"effort": "high"}),
        ("auto", "low", {"effort": "low", "summary": "auto"}),
        (None, None, None),
    ],
    ids=["summary_alone", "effort_alone", "both_keys", "neither_key"],
)
def test_request_assembles_the_reasoning_object_key_by_key(
    reasoning_summary: ReasoningSummary | None,
    reasoning_effort: ReasoningEffort | None,
    expected_reasoning: dict[str, str] | None,
) -> None:
    """A key travels only where it is set, and neither set sends no reasoning object at all.

    A summary is reached through the same object effort travels in, so gating that object on effort
    would silently ask for no summary whenever a caller set one without an effort. An explicit null
    summary is a different request from omitting the key, which is what key-by-key assembly buys
    over one Reasoning(effort=..., summary=...) call.
    """
    precomputed_fields = _adapter(reasoning_summary=reasoning_summary)._precompute_fields(
        _binding(automatic_prompt_caching=True, reasoning_effort=reasoning_effort)
    )
    if expected_reasoning is None:
        assert isinstance(precomputed_fields.reasoning, openai.Omit)
    else:
        assert precomputed_fields.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("supports_prompt_cache_options", "automatic_prompt_caching", "expected_options"),
    [
        (True, False, {"mode": "explicit"}),
        (True, True, None),
        (False, True, None),
    ],
    ids=[
        "supported_and_caching_disabled",
        "supported_and_caching_automatic",
        "unsupported_and_caching_automatic",
    ],
)
def test_request_sends_explicit_mode_exactly_when_the_binding_declines_caching(
    expected_options: dict[str, str] | None,
    *,
    supports_prompt_cache_options: bool,
    automatic_prompt_caching: bool,
) -> None:
    """Explicit mode with no breakpoints is the one prompt_cache_options value langchaint sends.

    Bound True leaves the omit sentinel whatever the model takes, so the provider's implicit caching
    stays in place. The fourth combination, bound False on a model that takes no parameter, has no
    row here because it raises instead of building fields.
    """
    precomputed_fields = _adapter(
        supports_prompt_cache_options=supports_prompt_cache_options
    )._precompute_fields(_binding(automatic_prompt_caching=automatic_prompt_caching))
    if expected_options is None:
        assert isinstance(precomputed_fields.prompt_cache_options, openai.Omit)
    else:
        assert precomputed_fields.prompt_cache_options == expected_options


def test_declining_caching_on_a_model_without_the_parameter_raises() -> None:
    """A model taking no prompt_cache_options cannot be told to stop caching, so bind refuses."""
    with pytest.raises(ValueError, match="supports_prompt_cache_options"):
        _adapter(supports_prompt_cache_options=False)._precompute_fields(
            _binding(automatic_prompt_caching=False)
        )


def test_the_refusal_reaches_bind_before_any_request_is_built() -> None:
    """LLM.bind raises, so one bad configuration fails once rather than per item in a batch.

    bind converts eagerly, so this asserts the raise is not deferred to a request method, where a
    generate_many would send one doomed request per item and return one failure row per item.
    The match names the model, which is what tells a caller with several LLMs which one refused.
    """
    llm = LLM(_adapter(supports_prompt_cache_options=False))
    with pytest.raises(ValueError, match="model 'm'"):
        llm.bind(automatic_prompt_caching=False)


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request; None leaves the omit sentinel.

    The sentinel is what keeps an unstated tier off the wire: sending an explicit null would be a
    different request from omitting the key.
    """
    binding = _binding(automatic_prompt_caching=True)
    assert isinstance(_adapter()._precompute_fields(binding).service_tier, openai.Omit)
    stated = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=True,
        service_tier="flex",
    )
    assert stated._precompute_fields(binding).service_tier == "flex"


def test_request_maps_temperature_and_omits_it_when_unset() -> None:
    """A bound temperature lands on the request; None leaves the omit sentinel."""
    unset = _adapter()._precompute_fields(_binding(automatic_prompt_caching=True))
    assert isinstance(unset.temperature, openai.Omit)
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(temperature=0.2),
        automatic_prompt_caching=True,
    )
    assert _adapter()._precompute_fields(binding).temperature == 0.2


def test_request_omits_tool_fields_without_tools() -> None:
    """No tools leaves tools, tool_choice, and parallel_tool_calls at the omit sentinel."""
    precomputed_fields = _adapter()._precompute_fields(_binding(automatic_prompt_caching=True))
    assert isinstance(precomputed_fields.tools, openai.Omit)
    assert isinstance(precomputed_fields.tool_choice, openai.Omit)
    assert isinstance(precomputed_fields.parallel_tool_calls, openai.Omit)


class _FakeSDKStream(AsyncResponseStream[None]):
    """Replays constructed events without a connection.

    Overrides exactly the surface _OpenAIStream uses (iteration, close, and the _response its
    headers are read off);
    the base __init__ is deliberately not called, so the untouched base machinery stays unusable.
    """

    def __init__(
        self, replay_events: Sequence[ResponseStreamEvent], headers: dict[str, str] | None = None
    ) -> None:
        self._replay_events = list(replay_events)
        self._response = httpx.Response(
            200,
            headers=headers,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )

    @override
    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        for replay_event in self._replay_events:
            yield replay_event

    @override
    async def close(self) -> None:
        return


def _stream(
    replay_events: Sequence[ResponseStreamEvent], headers: dict[str, str] | None = None
) -> _OpenAIStream:
    """Build an adapter stream over replayed events, reading headers off a constructed response."""
    return _OpenAIStream(sdk_stream=_FakeSDKStream(replay_events, headers))


def test_a_stream_reports_the_request_id_header_of_the_response_it_reads() -> None:
    """The stream's own response is the only channel a streamed turn has for the header.

    The response the SDK assembles from the events never carries it, so a null here would leave
    every streaming call with no id to take to provider support.
    """
    assert _stream([], {"x-request-id": "req_stream"}).request_id() == "req_stream"
    assert _stream([]).request_id() is None


async def _text_final(adapter_stream: _OpenAIStream) -> ResponseOutcome[str]:
    """Read the stream's terminal response and interpret it as the shipped text binding does."""
    return _text_bound().interpret(await adapter_stream.final())


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


def _summary_delta_event(
    delta: str, summary_index: int, sequence_number: int
) -> ResponseReasoningSummaryTextDeltaEvent:
    """Build one summary-text delta event, belonging to the numbered summary part."""
    return ResponseReasoningSummaryTextDeltaEvent(
        type="response.reasoning_summary_text.delta",
        delta=delta,
        item_id="r1",
        output_index=0,
        summary_index=summary_index,
        sequence_number=sequence_number,
    )


def _summary_done_event(
    text: str, summary_index: int, sequence_number: int
) -> ResponseReasoningSummaryTextDoneEvent:
    """Build the done event closing the numbered summary part."""
    return ResponseReasoningSummaryTextDoneEvent(
        type="response.reasoning_summary_text.done",
        text=text,
        item_id="r1",
        output_index=0,
        summary_index=summary_index,
        sequence_number=sequence_number,
    )


def _reasoning_text_delta_event(
    delta: str, content_index: int, sequence_number: int
) -> ResponseReasoningTextDeltaEvent:
    """Build one reasoning-text delta event, belonging to the numbered content part."""
    return ResponseReasoningTextDeltaEvent(
        type="response.reasoning_text.delta",
        delta=delta,
        content_index=content_index,
        item_id="r1",
        output_index=0,
        sequence_number=sequence_number,
    )


def _reasoning_text_done_event(
    text: str, content_index: int, sequence_number: int
) -> ResponseReasoningTextDoneEvent:
    """Build the done event closing the numbered content part."""
    return ResponseReasoningTextDoneEvent(
        type="response.reasoning_text.done",
        text=text,
        content_index=content_index,
        item_id="r1",
        output_index=0,
        sequence_number=sequence_number,
    )


def _streamed_reasoning(translated: Sequence[StreamItem]) -> str:
    """Concatenate the reasoning deltas, as an application rendering them as flowing text would."""
    return "".join(item.text for item in translated if isinstance(item, ReasoningDelta))


def test_stream_passes_text_deltas_through_as_bare_strings() -> None:
    """Text deltas pass through in order as the SDK's own strings; nothing follows them."""
    translated = _collected_items([
        _text_delta_event("he", 1),
        _text_delta_event("y", 2),
        _completed_event(_response(usage=_usage_with_cache()), 3),
    ])
    assert translated == ["he", "y"]


def test_stream_yields_both_reasoning_channels_as_reasoning_deltas() -> None:
    """Summary deltas and reasoning-text deltas both become ReasoningDelta; answer text stays a bare string.

    Which channel a model fills is request-time behavior, so the adapter forwards whichever arrives.
    """
    translated = _collected_items([
        _summary_delta_event("weighing", 0, 1),
        _reasoning_text_delta_event("deciding", 0, 2),
        _text_delta_event("hey", 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert translated == [ReasoningDelta(text="weighing"), ReasoningDelta(text="deciding"), "hey"]


def test_a_summary_part_boundary_streams_the_separator_the_assembled_trace_uses() -> None:
    """A part's done event puts a blank line before the next part's first delta.

    The API breaks between two parts structurally and never sends it as text, so deltas concatenated
    without it run the parts together.

    One scenario drives both surfaces: the deltas spell out the two parts, and the terminal response
    carries the reasoning item holding those same parts, as a stream's completed event does. What an
    application prints while the stream runs then has to match the trace it holds once it ends.
    """
    parts = ("First, water evaporates.", "Then it condenses.")
    adapter_stream = _stream([
        _summary_delta_event("First, water ", 0, 1),
        _summary_delta_event("evaporates.", 0, 2),
        _summary_done_event(parts[0], 0, 3),
        _summary_delta_event("Then it ", 1, 4),
        _summary_delta_event("condenses.", 1, 5),
        _summary_done_event(parts[1], 1, 6),
        _completed_event(
            _response(usage=_usage_with_cache(), output=[_reasoning_item(summary=parts)]), 7
        ),
    ])

    async def scenario() -> tuple[str, AssistantMessage]:
        streamed = _streamed_reasoning([item async for item in adapter_stream.items()])
        return streamed, _assistant_message_from(await adapter_stream.final())

    streamed, assistant_message = asyncio.run(scenario())
    trace = assistant_message.turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert streamed == trace.text == "First, water evaporates.\n\nThen it condenses."


def test_the_reasoning_text_channel_separates_its_parts_the_same_way() -> None:
    """The content channel's own done event drives the separator, as the summary channel's does."""
    translated = _collected_items([
        _reasoning_text_delta_event("First.", 0, 1),
        _reasoning_text_done_event("First.", 0, 2),
        _reasoning_text_delta_event("Then.", 1, 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert _streamed_reasoning(translated) == "First.\n\nThen."


def test_a_pending_separator_crosses_from_one_reasoning_channel_to_the_other() -> None:
    """A summary part's done event separates it from a content delta, the next reasoning text.

    The pending separator belongs to the stream, not to the channel that armed it.
    """
    translated = _collected_items([
        _summary_delta_event("First.", 0, 1),
        _summary_done_event("First.", 0, 2),
        _reasoning_text_delta_event("Then.", 0, 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert _streamed_reasoning(translated) == "First.\n\nThen."


@pytest.mark.parametrize(
    "replay_events",
    [
        [
            _summary_done_event("", 0, 1),
            _summary_delta_event("thought it over", 1, 2),
            _completed_event(_response(usage=_usage_with_cache()), 3),
        ],
        [
            _summary_delta_event("", 0, 1),
            _summary_done_event("", 0, 2),
            _summary_delta_event("thought it over", 1, 3),
            _completed_event(_response(usage=_usage_with_cache()), 4),
        ],
    ],
    ids=["a_done_event_with_no_delta_before_it", "a_part_whose_only_delta_was_empty"],
)
def test_a_summary_part_that_streamed_no_text_leaves_the_next_part_unseparated(
    replay_events: Sequence[ResponseStreamEvent],
) -> None:
    """A part that streamed no text arms nothing, so the reasoning never opens on a blank line.

    A summary part holding no text is dropped from the assembled trace, and the stream owes the
    same: a separator falls between two reasoning deltas or not at all. An empty delta is not text
    either, so counting it as one would open the reasoning on a blank line, its part having
    contributed nothing for a separator to follow.
    """
    assert _collected_items(replay_events) == [ReasoningDelta(text="thought it over")]


def test_an_empty_delta_does_not_consume_the_pending_separator() -> None:
    """The dropped delta leaves the separator pending, so the reasoning never trails a blank line."""
    translated = _collected_items([
        _summary_delta_event("thought it over", 0, 1),
        _summary_done_event("thought it over", 0, 2),
        _summary_delta_event("", 1, 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert translated == [ReasoningDelta(text="thought it over")]


def test_an_empty_delta_keeps_the_separator_for_the_next_delta_that_carries_text() -> None:
    """The separator armed before a dropped delta still falls before the next part's text."""
    translated = _collected_items([
        _summary_delta_event("First.", 0, 1),
        _summary_done_event("First.", 0, 2),
        _summary_delta_event("", 1, 3),
        _summary_delta_event("Then.", 1, 4),
        _completed_event(_response(usage=_usage_with_cache()), 5),
    ])
    assert _streamed_reasoning(translated) == "First.\n\nThen."


def test_a_done_event_with_no_delta_after_it_streams_no_trailing_separator() -> None:
    """The separator precedes the next reasoning delta, so a last part contributes none.

    Answer text following the done event is untouched: only a reasoning delta consumes a pending
    separator.
    """
    translated = _collected_items([
        _summary_delta_event("thought it over", 0, 1),
        _summary_done_event("thought it over", 0, 2),
        _text_delta_event("hey", 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert translated == [ReasoningDelta(text="thought it over"), "hey"]


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
        result = _assert_result(await _text_final(adapter_stream))
        assert result.output == "hey"
        assert result.stop_reason == "max_tokens"

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
        result = _assert_result(await _text_final(adapter_stream))
        assert result.output == "hey"
        assert result.stop_reason == "end_turn"

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
        result = _assert_result(await _text_final(adapter_stream))
        reasoning_trace = result.assistant_message.turn[0]
        assert isinstance(reasoning_trace, ReasoningTrace)
        assert reasoning_trace.raw == _REASONING_OUTPUT_ITEM

    asyncio.run(scenario())


def test_stream_failed_terminal_is_terminal_and_reports_the_provider_failure() -> None:
    """A failed terminal ends the stream without a StreamProtocolError, and interpret reports the failure.

    The API reported the run as not finished, so whatever text had accumulated is a fragment;
    returning it as a Response would present that fragment as the turn. final() hands back the
    response the run billed for, which billing_from_raw prices.
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
        raw = await adapter_stream.final()
        assert isinstance(_text_bound().interpret(raw), ProviderFailedTransiently)
        assert _text_bound().billing_from_raw(raw).usage.input_tokens_total == 1000

    asyncio.run(scenario())


def test_stream_final_passes_a_leniently_built_terminal_through_unvalidated() -> None:
    """A terminal response holding an output item the strict model rejects still assembles.

    The SDK builds a non-completed terminal response leniently, tolerating an item type it does not
    model, so validating that response against the SDK's own strict model would raise
    ValidationError and destroy a partial answer the caller has already been billed for.
    final() hands back that object itself (identity, not equality): an equal copy would silently
    introduce the per-request deep copy the no-rewrap rule bans.
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
        assert await adapter_stream.final() is leniently_built
        result = _assert_result(await _text_final(adapter_stream))
        assert result.output == "hey"
        assert result.stop_reason == "max_tokens"

    asyncio.run(scenario())


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
    precomputed_fields = adapter._precompute_fields(
        _binding(automatic_prompt_caching=False, system_prompt="sys")
    )
    return _BoundOpenAIStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )


def _structured_parse(response: OpenAIResponse) -> _StructuredReport | None | NoOutputOutcome:
    """Run the structured binding's parse over one response, with the turn that response carries."""
    return _structured_bound()._parsed_output(response, _assistant_message_from(response))


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


def test_structured_bind_validates_the_turns_text_into_the_instance() -> None:
    """The structured bound adapter validates the turn's output text into the response_format."""
    outcome = _structured_parse(_structured_response(_REPORT_JSON))
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_bind_reports_empty_turn_when_the_turn_carried_no_text() -> None:
    """A completed response with no text part and no tool call is EmptyTurn."""
    outcome = _structured_parse(_structured_response(None))
    assert isinstance(outcome, EmptyTurn)


def test_structured_bind_reports_schema_violation_on_text_the_model_rejects() -> None:
    """A completed turn whose text the response_format rejects is SchemaViolation.

    validation_error_json names the rejected field, what rejected it, and the value, which is what
    tells a caller whether to change the model or the prompt.
    """
    outcome = _structured_parse(_structured_response('{"city": "Nairobi", "celsius": "SENTINEL"}'))
    assert isinstance(outcome, SchemaViolation)
    rejections = json.loads(outcome.validation_error_json)
    assert [rejection["loc"] for rejection in rejections] == [["celsius"]]
    assert rejections[0]["input"] == "SENTINEL"


def test_structured_bind_reports_max_completion_tokens_exceeded_on_text_cut_mid_json() -> None:
    """An incomplete turn whose JSON stopped mid-object is the truncation, not a schema violation.

    This is the response the SDK's own parse raised on; reporting it as a member is what lets the
    retry loop fail the item with MaxCompletionTokensExceededError against the attempt it recorded.
    """
    outcome = _structured_parse(
        _structured_response(
            '{"city": "Nair',
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
    )
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_bind_reports_a_tool_call_turn_as_none() -> None:
    """A completed turn whose output is a function call parses no instance and nothing went wrong."""
    assert _structured_parse(_structured_response(None, tool_call=True)) is None


def test_structured_bind_reports_a_tool_call_turn_whose_text_is_not_the_instance_as_none() -> None:
    """A tool-call turn whose text is prose is the tool call, not a schema violation."""
    outcome = _structured_parse(_structured_response("let me look that up", tool_call=True))
    assert outcome is None


def test_structured_bind_sets_output_on_a_turn_that_also_called_a_tool() -> None:
    """The instance lands on output and the call still lands on tool_calls, so neither fact hides the other."""
    outcome = _structured_parse(_structured_response(_REPORT_JSON, tool_call=True))
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)


def _text_bound() -> _BoundOpenAIText:
    """Build a text-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    return _BoundOpenAIText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(automatic_prompt_caching=False)),
    )


def test_text_bind_reports_the_refusal_sentences_as_the_output() -> None:
    """A refused turn's output is the text the model wrote, not the empty output_text.

    The Response the caller reads then carries the same sentences in output and in
    assistant_message.text, which is what a refusal under the anthropic adapter carries too.
    """
    response = _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM])
    result = _assert_result(_text_bound().interpret(response))
    assert result.output == "I can't help with that"


def test_text_bind_send_hands_back_a_failed_status_for_interpret_to_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed status comes back from send rather than raising, and interpret reports the failure member.

    The API answers 200 with a body saying the run failed, so the response reaches the retry loop,
    which records it and its price before interpret reads it.
    """
    text_bound = _text_bound()

    async def fake_create(**_request_kwargs: object) -> OpenAIResponse:
        """Return a response the API reported as failed, with real usage on it."""
        return _response(usage=_usage_with_cache(), status="failed", error=_SERVER_ERROR)

    monkeypatch.setattr(text_bound._adapter.client.responses, "create", fake_create)
    raw = asyncio.run(text_bound.send(text_bound.build_request([UserMessage(content="q")])))
    assert isinstance(raw, OpenAIResponse)
    assert isinstance(text_bound.interpret(raw), ProviderFailedTransiently)
    assert text_bound.billing_from_raw(raw).usage.cost_in_usd > 0.0


def test_identity_reads_the_responses_own_id_and_served_model() -> None:
    """Both values come off the response verbatim, neither from the id the binding sent.

    The response names a model other than the adapter's, so reading the sent id would fail here.
    A response the SDK did not parse from an HTTP response body carries no request id, which is the
    state every streamed response is in.
    """
    identity = _text_bound().identity_from_raw(_response(usage=None, model="m-2026-01-01"))
    assert identity == ResponseIdentity(
        model_served="m-2026-01-01", response_id="r1", request_id=None
    )


def test_identity_reads_the_request_id_the_sdk_attached_to_the_response() -> None:
    """A response parsed from a response body carries the request-id header, which identity reports.

    The assignment is what the SDK's own add_request_id does to every model it parses from a body
    (openai 2.48.0).
    """
    response = _response(usage=None)
    response._request_id = "req_openai"
    identity = _text_bound().identity_from_raw(response)
    assert identity.request_id == "req_openai"


def test_structured_bind_reports_the_failure_on_a_failed_status_whose_text_validates() -> None:
    """A failed run is the failure member even when its fragment validates: it is not the answer."""
    outcome = _structured_parse(
        _structured_response(_REPORT_JSON, status="failed", error=_SERVER_ERROR)
    )
    assert isinstance(outcome, ProviderFailedTransiently)


def test_a_failed_run_carrying_a_refusal_takes_the_failure_member_under_both_bindings() -> None:
    """A failed status wins over the refusal test, so one response does not split by binding.

    Were the refusal tested first, the structured binding would report Refusal (a terminal
    RefusalError) for a response the text binding retries.
    """
    structured_outcome = _structured_parse(
        _structured_response(None, refusal=True, status="failed", error=_SERVER_ERROR)
    )
    assert isinstance(structured_outcome, ProviderFailedTransiently)
    text_outcome = _text_bound().interpret(
        _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM], status="failed", error=_SERVER_ERROR)
    )
    assert isinstance(text_outcome, ProviderFailedTransiently)


def test_a_transient_error_code_carries_the_providers_message_and_no_rate_limit_flag() -> None:
    """A server_error is transient, and its reason is openai's message verbatim."""
    outcome = _text_bound().interpret(_response(usage=None, status="failed", error=_SERVER_ERROR))
    assert isinstance(outcome, ProviderFailedTransiently)
    assert outcome.reason == _SERVER_ERROR.message
    assert outcome.is_rate_limit is False


def test_a_rate_limit_error_code_sets_the_rate_limit_flag() -> None:
    """rate_limit_exceeded is transient and flags the rate limit, which paces every sharing task."""
    outcome = _text_bound().interpret(
        _response(
            usage=None,
            status="failed",
            error=ResponseError(code="rate_limit_exceeded", message="Rate limit reached."),
        )
    )
    assert isinstance(outcome, ProviderFailedTransiently)
    assert outcome.is_rate_limit is True


def test_a_terminal_error_code_carries_the_providers_message_without_retrying() -> None:
    """failed_to_download_image is terminal: the request names the image, so a resend fetches it again."""
    outcome = _text_bound().interpret(
        _response(usage=None, status="failed", error=_IMAGE_DOWNLOAD_ERROR)
    )
    assert isinstance(outcome, ProviderFailedTerminally)
    assert outcome.reason == _IMAGE_DOWNLOAD_ERROR.message


def test_an_error_code_the_installed_sdk_does_not_name_is_terminal() -> None:
    """A code added after openai 2.45.0 fails the item once rather than spending the retry budget."""
    outcome = _text_bound().interpret(
        _response(
            usage=None,
            status="failed",
            error=ResponseError.construct(code="a_code_from_a_later_sdk", message="Something."),
        )
    )
    assert isinstance(outcome, ProviderFailedTerminally)
    assert outcome.reason == "Something."


def test_a_failed_status_with_no_error_object_is_terminal() -> None:
    """A failed run naming nothing gives no ground to resend on, and says that in its reason."""
    outcome = _text_bound().interpret(_response(usage=None, status="failed"))
    assert isinstance(outcome, ProviderFailedTerminally)
    assert outcome.reason == "openai reported status 'failed' and no error object"


def test_a_run_that_stopped_short_of_a_turn_is_unfinished_turn_naming_the_status() -> None:
    """A cancelled run is neither a failure openai described nor a turn, so it names its status."""
    outcome = _structured_parse(_structured_response(None, status="cancelled"))
    assert isinstance(outcome, UnfinishedTurn)
    assert outcome.reason == "openai returned status 'cancelled'"


def test_structured_bind_reports_refusal_on_a_refusal_block() -> None:
    """A response carrying a refusal content block is Refusal."""
    outcome = _structured_parse(_structured_response(None, refusal=True))
    assert isinstance(outcome, Refusal)


def test_structured_bind_reports_max_completion_tokens_exceeded_on_a_max_output_tokens_incomplete() -> (
    None
):
    """An incomplete response for max_output_tokens is MaxCompletionTokensExceeded."""
    outcome = _structured_parse(
        _structured_response(
            None,
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
    )
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_bind_reports_refusal_on_a_content_filter_incomplete() -> None:
    """An incomplete response for content_filter is Refusal, so the item fails once.

    Retrying would send the same blocked request again for the whole retry budget
    and bill for each attempt.
    """
    outcome = _structured_parse(
        _structured_response(
            None,
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="content_filter"),
        )
    )
    assert isinstance(outcome, Refusal)


def test_every_request_carries_the_reasoning_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create and stream, on both bindings, all send include=["reasoning.encrypted_content"].

    The offline round-trip tests cannot catch a dropped include:
    the SDK documents include as what populates encrypted_content,
    so without this parameter every replayed reasoning item could be silently empty.
    """
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(_binding(automatic_prompt_caching=True))
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
    text_bound = _BoundOpenAIText(adapter=adapter, precomputed_fields=precomputed_fields)
    structured_bound = _BoundOpenAIStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )

    async def scenario() -> None:
        messages = [UserMessage(content="q")]
        await text_bound.send(text_bound.build_request(messages))
        await text_bound.open_stream(text_bound.build_request(messages))
        await structured_bound.send(structured_bound.build_request(messages))
        await structured_bound.open_stream(structured_bound.build_request(messages))

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
    precomputed_fields = adapter._precompute_fields(_binding(automatic_prompt_caching=True))
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
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )

    async def scenario() -> None:
        request = structured_bound.build_request([UserMessage(content="q")])
        await structured_bound.send(request)
        await structured_bound.open_stream(request)

    asyncio.run(scenario())
    assert texts == [{"format": type_to_text_format_param(_StructuredReport)}] * 2


def test_a_built_request_renders_as_json_carrying_the_prompt_and_no_omitted_field() -> None:
    """as_json holds the binding's precomputed fields and this call's converted input.

    temperature is absent rather than null, because the binding set none and the request body carries
    no such key.
    """
    request = _structured_bound().build_request([UserMessage(content="hi")])
    assert isinstance(request, _OpenAIRequestParams)
    rendered = json.loads(request.as_json())
    assert rendered["input"] == [{"role": "user", "content": "hi"}]
    assert rendered["precomputed"]["instructions"] == "sys"
    assert "temperature" not in rendered["precomputed"]


def _rate_limit_error(headers: dict[str, str]) -> openai.RateLimitError:
    """Build the SDK's 429 exception around a constructed httpx response."""
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_retry_after_seconds_reads_the_headers_of_an_sdk_error_and_nothing_else() -> None:
    """The override finds the headers on the SDK's own exception and yields None for any other.

    The parsing itself is tested in tests/test_adapter.py against the shared function; what is
    provider-specific is where the headers are found. httpx.Headers is case-insensitive, so the
    lookup keeps working whatever case the server sent.
    """
    adapter = _adapter()
    assert adapter.retry_after_seconds(_rate_limit_error({"Retry-After-MS": "1500"})) == 1.5
    assert adapter.retry_after_seconds(_rate_limit_error({})) is None
    assert adapter.retry_after_seconds(ValueError("boom")) is None


def test_request_id_from_error_reads_the_sdk_errors_own_header_and_nothing_else() -> None:
    """The override reports the header the SDK read off the error response, None for any other error.

    openai sends the id in x-request-id; a response without that header and an exception that never
    reached one both give None.
    """
    adapter = _adapter()
    assert (
        adapter.request_id_from_error(_rate_limit_error({"x-request-id": "req_429"})) == "req_429"
    )
    assert adapter.request_id_from_error(_rate_limit_error({})) is None
    assert adapter.request_id_from_error(ValueError("boom")) is None


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
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            automatic_prompt_caching=True,
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            ),
        )
    )
    assert precomputed_fields.instructions is None
    assert precomputed_fields.input_prefix == [
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
    precomputed_fields = _adapter()._precompute_fields(
        _binding(automatic_prompt_caching=True, system_prompt="sys")
    )
    assert precomputed_fields.instructions == "sys"
    assert precomputed_fields.input_prefix == []


def _conformance_output() -> list[object]:
    """Build one reasoning item then one message item, mapping one to one onto their wire items.

    The reasoning item carries a key the installed SDK does not name. An adapter that rebuilt the
    item from its own pinned model would drop that key, which the API rejects on replay because
    encrypted_content must arrive byte-identical.
    """
    return [_REASONING_OUTPUT_ITEM | {"field_newer_than_sdk": "x"}, dict(_TEXT_OUTPUT_ITEM)]


class TestOpenAIResponsesConformance(AdapterConformance):
    """The neutral invariants, over the OpenAI Responses adapter's own SDK objects."""

    @override
    def make_adapter(self) -> Adapter:
        """Build the adapter these invariants run against, priced for the default tier alone."""
        return _adapter()

    @override
    def response_with_cache_writes(self) -> BaseModel:
        """Return a turn whose input_tokens carries both a cache read and a cache write."""
        return _response(usage=_usage_with_cache(), output=_conformance_output())

    @override
    def response_without_usage(self) -> BaseModel:
        """Return a turn whose usage field is absent, which openai answers a run with."""
        return _response(usage=None, output=_conformance_output())

    @override
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return a turn served at flex, which _PRICING holds no table for."""
        return _response(
            usage=_usage_with_cache(), output=_conformance_output(), service_tier="flex"
        )

    @override
    def response_with_impossible_counters(self) -> BaseModel:
        """Return a turn whose cache counts sum past input_tokens.

        The uncached counter is the subtraction, so counts summing past the total drive it below
        zero.
        """
        return _response(
            usage=ResponseUsage(
                input_tokens=1000,
                input_tokens_details=InputTokensDetails(cached_tokens=900, cache_write_tokens=200),
                output_tokens=40,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=1040,
            ),
            output=_conformance_output(),
        )

    @override
    def response_with_reasoning(self) -> BaseModel:
        """Return a turn whose reasoning item carries the unnamed key."""
        return _response(usage=_usage_with_cache(), output=_conformance_output())

    @override
    def assistant_wire_elements(self, request: RequestParams) -> Sequence[object]:
        """Read the input items past the one the user message became."""
        assert isinstance(request, _OpenAIRequestParams)
        return request.input[1:]

    @override
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return the same turn as the ParsedResponse a stream assembles into and as a Response."""
        whole = _response(usage=_usage_with_cache(), output=_conformance_output())
        return ParsedResponse[None].model_validate(whole.model_dump()), whole

    @override
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream whose events end before any terminal event."""
        return _stream([_text_delta_event("he", 1)])

    @override
    def sdk_errors_and_classifications(self) -> Mapping[Exception, ErrorClassification]:
        """Return the adapter's whole exception table.

        Each status code is the one the SDK raises that class for, read from openai 2.45.0;
        the bare APIStatusError rows are the statuses the SDK maps to no class of its own, 413 among
        them, which is why the adapter reads the status rather than the exception class.
        APITimeoutError subclasses APIConnectionError, so timeouts reach transient through that
        isinstance.
        x-should-retry overrides the status in both directions, except on a rate-limit status, which
        stays rate_limit whatever the header says, so the limiter's account-wide pause is still
        armed, and on a 4xx marked final, which keeps the rejection name.
        A 3xx and the non-SDK ValueError land on the unknown_exception default, and a 5xx the
        provider marked final lands on declared_final; each fails the one item without a retry.
        """
        return {
            _status_error(openai.RateLimitError, 429): "rate_limit",
            _status_error(openai.InternalServerError, 500): "transient",
            _connection_error(): "transient",
            openai.APITimeoutError(httpx.Request("POST", "https://api.openai.com")): "transient",
            _status_error(openai.ConflictError, 409): "transient",
            _status_error(openai.BadRequestError, 400): "invalid_request",
            _status_error(openai.AuthenticationError, 401): "invalid_request",
            _status_error(openai.PermissionDeniedError, 403): "invalid_request",
            _status_error(openai.NotFoundError, 404): "invalid_request",
            _status_error(openai.UnprocessableEntityError, 422): "invalid_request",
            _status_error(openai.APIStatusError, 413): "invalid_request",
            _status_error(openai.APIStatusError, 408): "transient",
            _status_error(openai.InternalServerError, 503): "transient",
            _status_error(openai.APIStatusError, 302): "unknown_exception",
            _status_error(openai.BadRequestError, 400, {"x-should-retry": "true"}): "transient",
            _status_error(
                openai.BadRequestError, 400, {"x-should-retry": "false"}
            ): "invalid_request",
            _status_error(
                openai.InternalServerError, 500, {"x-should-retry": "false"}
            ): "declared_final",
            _status_error(openai.RateLimitError, 429, {"x-should-retry": "false"}): "rate_limit",
            _status_error(openai.RateLimitError, 429, {"x-should-retry": "true"}): "rate_limit",
            ValueError("boom"): "unknown_exception",
        }
