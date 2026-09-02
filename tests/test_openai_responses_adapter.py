"""Test OpenAI Responses with constructed SDK objects.

Tests cover Usage, input items, tool choice, stop reasons, streams, and requests.
"""

import asyncio
import base64
import inspect
import json
import math
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Literal, override

import httpx2
import openai
import pytest
from openai import AsyncOpenAI
from openai._models import construct_type_unchecked
from openai.lib._parsing._responses import type_to_text_format_param
from openai.lib.streaming.responses import (
    AsyncResponseStream,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseStreamEvent,
)
from openai.lib.streaming.responses import (
    ResponseTextDeltaEvent as AccumulatedResponseTextDeltaEvent,
)
from openai.lib.streaming.responses._events import (
    ResponseCompletedEvent as AccumulatedResponseCompletedEvent,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseUsage,
)
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel, TypeAdapter

from langchaint import (
    LLM,
    AllowedToolsChoice,
    AssistantMessage,
    AudioPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
    Message,
    RawPart,
    ReasoningDelta,
    ReasoningPart,
    SpecificToolChoice,
    StopReason,
    StreamItem,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
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
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutput,
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
    OpenAIPricingTable,
    OpenAIRates,
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.responses_adapter import (
    _assistant_items,
    _assistant_message_from,
    _BoundOpenAI,
    _BoundOpenAIStructured,
    _BoundOpenAIText,
    _normalized_stop_reason,
    _OpenAIRequestParams,
    _OpenAIStream,
    _wire_input,
    _wire_tool_choice,
)
from langchaint.openai.responses_adapter import (
    _billing_from_response as _provider_billing_from_response,
)
from langchaint.openai.shared import PARSE_FALLTHROUGH_COUNTS, parse_openai
from langchaint.pricing import Billing
from langchaint.shared_backoff import (
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    RetryThisOne,
    Verdict,
)
from langchaint.tools import ToolSchema
from tests.helpers import (
    openai_sdk_errors_and_classifications,
    openai_sdk_errors_and_verdicts,
    status_error,
)


def _billing_from_response(
    response: OpenAIResponse,
    pricing: OpenAIPricingTable,
    *,
    regional_processing: bool = False,
) -> Billing:
    return _provider_billing_from_response(
        response, pricing, regional_processing=regional_processing
    ).billing


_DEFAULT_RATES = OpenAIRates(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)

_PRICING = OpenAIPricingTable(
    default=_DEFAULT_RATES,
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)
"""The default tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = OpenAIRates(
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

_WEB_SEARCH_OUTPUT_ITEM: dict[str, object] = {
    "type": "web_search_call",
    "id": "ws_1",
    "status": "completed",
    "action": {"type": "search", "query": "langchaint"},
}
"""One built-in tool call without another TurnPart variant."""

_FILE_SEARCH_OUTPUT_ITEM: dict[str, object] = {
    "type": "file_search_call",
    "id": "fs_1",
    "status": "completed",
    "queries": ["first", "second"],
}


def _assert_result[OutputT](outcome: ResponseOutcome[OutputT]) -> AdapterResult[OutputT]:
    """Narrow a ResponseOutcome to its success variant, failing the test on any other variant."""
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
    """Build a response whose id is fixed at "r1". Every field a test varies is a parameter."""
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


def test_billing_partitions_and_prices_complete_usage() -> None:
    """Expose one complete SDK response's neutral counters, costs, tier, and applied rates."""
    raw = _response(usage=_usage_with_cache(), service_tier="auto")
    billing = _billing_from_response(raw, _PRICING)
    usage = billing.usage
    assert _provider_billing_from_response(raw, _PRICING).usage_raw is raw.usage
    assert billing.service_tier == "default"
    assert billing.input_cache_none_usd_per_million_tokens == 2.5
    assert billing.cache_read_usd_per_million_tokens == 1.25
    assert billing.cache_write_usd_per_million_tokens == 3.125
    assert billing.output_usd_per_million_tokens == 10.0
    assert raw.service_tier == "auto"
    assert usage.input_tokens_cache_read == 600
    assert usage.input_tokens_cache_write == 100
    assert usage.input_tokens_cache_none == 300
    assert usage.input_tokens_cache_none_cost_in_usd == 300 * 2.5 / 1e6
    assert usage.input_tokens_cache_read_cost_in_usd == 600 * 1.25 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == 100 * 3.125 / 1e6
    assert usage.output_tokens_cost_in_usd == 40 * 10.0 / 1e6


def test_web_search_output_adds_the_cataloged_provider_executed_tool_cost() -> None:
    """One search action adds one cataloged invocation."""
    response = _response(
        usage=_usage_with_cache(),
        output=[dict(_WEB_SEARCH_OUTPUT_ITEM), dict(_TEXT_OUTPUT_ITEM)],
    )
    usage = _billing_from_response(response, _PRICING).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(0.01)


def test_open_page_action_adds_no_provider_executed_tool_cost() -> None:
    """OpenAI's web-search guide prices `search` actions only."""
    open_page = dict(_WEB_SEARCH_OUTPUT_ITEM)
    open_page["action"] = {"type": "open_page", "url": "https://example.com"}
    response = _response(usage=_usage_with_cache(), output=[open_page])
    usage = _billing_from_response(response, _PRICING).usage
    assert usage.provider_executed_tool_cost_in_usd == 0.0


def test_find_in_page_action_adds_no_provider_executed_tool_cost() -> None:
    """OpenAI prices no separate fee for `find_in_page`."""
    find_in_page = dict(_WEB_SEARCH_OUTPUT_ITEM)
    find_in_page["action"] = {
        "type": "find_in_page",
        "url": "https://example.com",
        "pattern": "price",
    }
    usage = _billing_from_response(
        _response(usage=_usage_with_cache(), output=[find_in_page]), _PRICING
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == 0.0


def test_file_search_calls_and_web_search_have_separate_rates() -> None:
    """Each file-search output item costs once, regardless of its query count."""
    second_file_search = {**_FILE_SEARCH_OUTPUT_ITEM, "id": "fs_2", "queries": ["third"]}
    response = _response(
        usage=_usage_with_cache(),
        output=[_WEB_SEARCH_OUTPUT_ITEM, _FILE_SEARCH_OUTPUT_ITEM, second_file_search],
    )
    usage = _billing_from_response(response, _PRICING).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(0.015)


def test_unpriceable_charged_output_produces_nan() -> None:
    """Image generation lacks response evidence for exact pricing."""
    image_generation = {
        "type": "image_generation_call",
        "id": "image-1",
        "status": "completed",
        "result": "image-data",
    }
    usage = _billing_from_response(
        _response(usage=_usage_with_cache(), output=[image_generation]), _PRICING
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


def test_charged_output_at_an_unpriced_tier_keeps_tool_cost() -> None:
    """Missing token rates do not change provider-executed tool costs."""
    response = _response(
        usage=None,
        output=[dict(_WEB_SEARCH_OUTPUT_ITEM)],
        service_tier="flex",
    )
    usage = _billing_from_response(response, _PRICING).usage
    assert usage.provider_executed_tool_cost_in_usd == 0.01


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
    assert _provider_billing_from_response(_response(usage=None), _PRICING).usage_raw is None
    assert billing.output_usd_per_million_tokens == 10.0
    assert billing.service_tier == "default"


def test_a_response_that_billed_nothing_at_an_unpriced_tier_still_costs_zero() -> None:
    """Every counter zero leaves the total a number even where no rate priced the tier."""
    billing = _billing_from_response(_response(usage=None, service_tier="flex"), _PRICING)
    assert math.isnan(billing.output_usd_per_million_tokens)
    assert billing.usage.cost_in_usd == 0.0


def test_the_reported_tier_selects_the_table() -> None:
    """Use the response's `service_tier` because it may differ from the request's (openai 3.0.0)."""
    pricing = OpenAIPricingTable(default=_DEFAULT_RATES, fast=_PRIORITY_RATES)
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


_REFUSAL_MESSAGE_ITEM: dict[str, object] = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": "I can't help with that"}],
}


def _incomplete_response(reason: Literal["max_output_tokens", "content_filter"]) -> OpenAIResponse:
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

    The refusal remains in the turn for replay.
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

    Produce yields one ReasoningPart where the reasoning item sat.
    Consume re-emits the stored dict unchanged, in the same position, with one input item per modeled output item.
    """
    response = _response(
        usage=None,
        output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM],
    )
    assistant_message = _assistant_message_from(response)
    assert [type(part) for part in assistant_message.turn] == [
        ReasoningPart,
        TextPart,
        ToolCall,
    ]
    reasoning_part = assistant_message.turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.raw == _REASONING_OUTPUT_ITEM
    assert assistant_message.text == "hey"
    assert assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )
    items = _assistant_items(assistant_message)
    assert len(items) == len(response.output)
    assert items[0] == reasoning_part.raw
    assert items[1] == {"role": "assistant", "content": "hey"}
    assert items[2] == {
        "type": "function_call",
        "call_id": "call1",
        "name": "lookup",
        "arguments": '{"q": 1}',
    }


def test_a_built_in_tool_call_becomes_a_raw_part_and_replays_as_itself() -> None:
    """A built-in tool call becomes RawPart and returns unchanged.

    The billed raw item remains in the turn for replay.
    """
    response = _response(usage=None, output=[_WEB_SEARCH_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM])
    assistant_message = _assistant_message_from(response)
    assert [type(part) for part in assistant_message.turn] == [RawPart, TextPart]
    items = _assistant_items(assistant_message)
    assert items == [_WEB_SEARCH_OUTPUT_ITEM, {"role": "assistant", "content": "hey"}]


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
def test_reasoning_part_text_takes_the_content_over_the_summary_and_is_none_without_text(
    summary: tuple[str, ...], content: tuple[str, ...] | None, expected_text: str | None
) -> None:
    """Reasoning text prefers content, joins parts, and excludes empty text."""
    response = _response(usage=None, output=[_reasoning_item(summary=summary, content=content)])
    reasoning_part = _assistant_message_from(response).turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.text == expected_text


def test_two_text_parts_stay_split_on_produce_and_rejoin_into_one_message_item() -> None:
    """Output splits text parts and input rejoins adjacent text parts."""
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


def test_produced_reasoning_parts_survive_the_message_json_round_trip() -> None:
    """ReasoningPart.raw survives Message JSON serialization."""
    reasoning_item = _reasoning_item(summary=("thought it over",))
    reasoning_item["encrypted_content"] = "enc-1"
    response = _response(usage=None, output=[reasoning_item, _TEXT_OUTPUT_ITEM])
    messages_type_adapter: TypeAdapter[tuple[Message, ...]] = TypeAdapter(tuple[Message, ...])
    messages: tuple[Message, ...] = (_assistant_message_from(response),)
    restored = messages_type_adapter.validate_json(messages_type_adapter.dump_json(messages))
    assert restored == messages


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """A foreign ReasoningPart sends ReasoningPart.raw unchanged for provider validation."""
    raw: dict[str, JsonValue] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(turn=(ReasoningPart(raw=raw), TextPart(text="hi")))
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
    """A ToolMessage carrying parts becomes a function_call_output with structured content."""
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


def test_wire_input_sends_image_url_parts_unchanged() -> None:
    """ImageUrlPart reaches both Responses image_url fields unchanged."""
    image_url = ImageUrlPart(url="https://example.com/image.png", cache_breakpoint=True)
    wire = _wire_input([
        UserMessage(content=(image_url,)),
        ToolMessage(tool_call_id="call1", content=(image_url,)),
    ])
    expected_image = {
        "type": "input_image",
        "image_url": "https://example.com/image.png",
        "detail": "auto",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    assert wire == [
        {"role": "user", "content": [expected_image]},
        {
            "type": "function_call_output",
            "call_id": "call1",
            "output": [expected_image],
        },
    ]


@pytest.mark.parametrize(
    "message",
    [
        UserMessage(content=(AudioPart(data=b"wav", media_type="audio/wav"),)),
        ToolMessage(
            tool_call_id="call1",
            content=(AudioPart(data=b"wav", media_type="audio/wav"),),
        ),
    ],
)
def test_build_request_reports_audio_as_invalid_request(message: Message) -> None:
    """OpenAIResponsesAdapter returns InvalidRequest for AudioPart."""
    request = (
        _adapter().bind_text(_binding(automatic_cache_breakpoints=True)).build_request([message])
    )
    assert isinstance(request, InvalidRequest)
    assert "AudioPart" in request.reason
    assert type(message).__name__ in request.reason


def test_wire_tool_choice_passes_strings_through_and_names_specific_tools() -> None:
    """The neutral strings pass through unchanged. SpecificToolChoice becomes the function form."""
    assert _wire_tool_choice("auto") == "auto"
    assert _wire_tool_choice("required") == "required"
    assert _wire_tool_choice("none") == "none"
    assert _wire_tool_choice(SpecificToolChoice(tool_name="x")) == {
        "type": "function",
        "name": "x",
    }


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_wire_tool_choice_restricts_function_names_in_order(
    mode: Literal["auto", "required"],
) -> None:
    """AllowedToolsChoice maps to the Responses allowed_tools form."""
    assert _wire_tool_choice(AllowedToolsChoice(mode=mode, tool_names=("second", "first"))) == {
        "type": "allowed_tools",
        "mode": mode,
        "tools": [
            {"type": "function", "name": "second"},
            {"type": "function", "name": "first"},
        ],
    }


def _adapter(
    *,
    reasoning_summary: ReasoningSummary | None = None,
    supports_prompt_cache_options: bool = True,
) -> OpenAIResponsesAdapter:
    """Build an adapter over a keyless client, valid because no request is sent.

    supports_prompt_cache_options=True sends the binding's cache setting.
    """
    return OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=supports_prompt_cache_options,
        reasoning_summary=reasoning_summary,
    )


def test_config_fingerprint_data_contains_only_stored_request_configuration() -> None:
    """Fingerprint data includes constructor request settings and excludes billing settings."""
    adapter = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        regional_processing=True,
        supports_prompt_cache_options=False,
        reasoning_summary="detailed",
        service_tier="ultrafast",
    )
    assert adapter.config_fingerprint_data() == {
        "reasoning_summary": "detailed",
        "service_tier": "ultrafast",
        "supports_prompt_cache_options": False,
    }


def _binding(
    *,
    automatic_cache_breakpoints: bool,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    tool_schemas: tuple[ToolSchema, ...] = (),
    reasoning_level: str | None = None,
    provider_executed_tools: tuple[Mapping[str, object], ...] = (),
    tool_choice: ToolChoice = "auto",
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    """Assemble a binding with the fields these request tests vary."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        provider_executed_tools=provider_executed_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
        max_completion_tokens=None,
        reasoning_level=reasoning_level,
        temperature=None,
        automatic_cache_breakpoints=automatic_cache_breakpoints,
        extra_body=extra_body,
    )


@pytest.mark.parametrize(
    ("reasoning_summary", "reasoning_level", "expected_reasoning"),
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
    reasoning_level: str | None,
    expected_reasoning: dict[str, str] | None,
) -> None:
    """The request includes only stated reasoning fields."""
    precomputed_fields = _adapter(reasoning_summary=reasoning_summary)._precompute_fields(
        _binding(automatic_cache_breakpoints=True, reasoning_level=reasoning_level)
    )
    if expected_reasoning is None:
        assert isinstance(precomputed_fields.reasoning, openai.Omit)
    else:
        assert precomputed_fields.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("supports_prompt_cache_options", "automatic_cache_breakpoints", "expected_options"),
    [
        (True, False, {"mode": "explicit"}),
        (True, True, None),
        (False, True, None),
    ],
    ids=[
        "supported_and_automatic_cache_breakpoints_disabled",
        "supported_and_automatic_cache_breakpoints_enabled",
        "unsupported_and_automatic_cache_breakpoints_enabled",
    ],
)
def test_request_sends_explicit_mode_when_automatic_cache_breakpoints_are_disabled(
    expected_options: dict[str, str] | None,
    *,
    supports_prompt_cache_options: bool,
    automatic_cache_breakpoints: bool,
) -> None:
    """Explicit mode with no breakpoints is the one prompt_cache_options value langchaint sends.

    `automatic_cache_breakpoints=True` leaves the omit sentinel.
    `automatic_cache_breakpoints=False` without parameter support raises before building fields.
    """
    precomputed_fields = _adapter(
        supports_prompt_cache_options=supports_prompt_cache_options
    )._precompute_fields(_binding(automatic_cache_breakpoints=automatic_cache_breakpoints))
    if expected_options is None:
        assert isinstance(precomputed_fields.prompt_cache_options, openai.Omit)
    else:
        assert precomputed_fields.prompt_cache_options == expected_options


def test_disabling_automatic_cache_breakpoints_without_parameter_support_raises() -> None:
    """`automatic_cache_breakpoints=False` requires `prompt_cache_options`."""
    with pytest.raises(ValueError, match="supports_prompt_cache_options"):
        _ = _adapter(supports_prompt_cache_options=False)._precompute_fields(
            _binding(automatic_cache_breakpoints=False)
        )


def test_the_refusal_reaches_bind_before_any_request_is_built() -> None:
    """LLM.bind rejects unsupported cache configuration before requests."""
    llm = LLM(_adapter(supports_prompt_cache_options=False))
    with pytest.raises(ValueError, match="model 'm'"):
        _ = llm.bind(automatic_cache_breakpoints=False)


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request. None leaves the omit sentinel.

    The sentinel omits an unstated tier from the request.
    """
    binding = _binding(automatic_cache_breakpoints=True)
    assert isinstance(_adapter()._precompute_fields(binding).service_tier, openai.Omit)
    stated = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        regional_processing=False,
        supports_prompt_cache_options=True,
        service_tier="flex",
    )
    assert stated._precompute_fields(binding).service_tier == "flex"


def test_request_maps_temperature_and_omits_it_when_unset() -> None:
    """A bound temperature lands on the request. None leaves the omit sentinel."""
    unset = _adapter()._precompute_fields(_binding(automatic_cache_breakpoints=True))
    assert isinstance(unset.temperature, openai.Omit)
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        provider_executed_tools=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        max_completion_tokens=None,
        reasoning_level=None,
        temperature=0.2,
        automatic_cache_breakpoints=True,
    )
    assert _adapter()._precompute_fields(binding).temperature == 0.2


def test_request_omits_tool_fields_without_tools() -> None:
    """No tools leaves tools, tool_choice, and parallel_tool_calls at the omit sentinel."""
    precomputed_fields = _adapter()._precompute_fields(_binding(automatic_cache_breakpoints=True))
    assert isinstance(precomputed_fields.tools, openai.Omit)
    assert isinstance(precomputed_fields.tool_choice, openai.Omit)
    assert isinstance(precomputed_fields.parallel_tool_calls, openai.Omit)


def test_provider_executed_tools_reach_responses_tools_unchanged() -> None:
    """Responses receives provider-executed tool mappings through its tools parameter."""
    provider_tool: dict[str, object] = {"type": "web_search", "search_context_size": "low"}
    precomputed = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True,
            provider_executed_tools=(provider_tool,),
        )
    )
    assert precomputed.tools == [provider_tool]
    assert precomputed.tool_choice == "auto"
    assert precomputed.parallel_tool_calls is True


@pytest.mark.parametrize(
    "tool_type",
    [
        "file_search",
        "web_search",
        "web_search_2025_08_26",
        "web_search_preview",
        "web_search_preview_2025_03_11",
    ],
)
def test_every_supported_provider_executed_type_reaches_responses(tool_type: str) -> None:
    """Each reviewed OpenAI provider-executed `type` reaches Responses unchanged."""
    provider_tool: dict[str, object] = {"type": tool_type}
    precomputed = _adapter()._precompute_fields(
        _binding(automatic_cache_breakpoints=True, provider_executed_tools=(provider_tool,))
    )
    assert precomputed.tools == [provider_tool]


@pytest.mark.parametrize(
    "tool_type",
    [
        "apply_patch",
        "code_interpreter",
        "computer",
        "computer_use_preview",
        "custom",
        "function",
        "image_generation",
        "local_shell",
        "mcp",
        "namespace",
        "programmatic_tool_calling",
        "shell",
        "tool_search",
    ],
)
def test_every_unlisted_provider_executed_type_is_rejected(tool_type: str) -> None:
    """Responses rejects every installed `ToolParam` type outside the reviewed set."""
    with pytest.raises(ValueError, match="supported string type"):
        _ = _adapter()._precompute_fields(
            _binding(
                automatic_cache_breakpoints=True,
                provider_executed_tools=({"type": tool_type},),
            )
        )


def test_provider_executed_tools_require_direct_openai_billing() -> None:
    """A non-OpenAI provider lacks verified OpenAI tool billing."""
    adapter = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="azure.ai.openai",
        regional_processing=False,
        supports_prompt_cache_options=True,
    )
    with pytest.raises(ValueError, match="provider_name='openai'"):
        _ = adapter._precompute_fields(
            _binding(
                automatic_cache_breakpoints=True,
                provider_executed_tools=({"type": "web_search"},),
            )
        )


@pytest.mark.parametrize("tool_type", ["web_search", "file_search"])
@pytest.mark.parametrize("rate", [None, True, math.nan, math.inf, -0.01])
def test_configured_openai_tool_rates_must_be_finite_and_nonnegative(
    tool_type: str, rate: float | None
) -> None:
    """A configured charged tool rejects an unusable caller rate before requests."""
    pricing = OpenAIPricingTable(
        default=OpenAIRates(
            input_cache_none_usd_per_million_tokens=2.5,
            output_usd_per_million_tokens=10.0,
            cache_read_usd_per_million_tokens=1.25,
            cache_write_usd_per_million_tokens=3.125,
        ),
        web_search_usd_per_invocation=rate if tool_type == "web_search" else 0.01,
        file_search_usd_per_invocation=rate if tool_type == "file_search" else 0.0025,
    )
    adapter = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=pricing,
        provider_name="openai",
        regional_processing=False,
        supports_prompt_cache_options=True,
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _ = adapter._precompute_fields(
            _binding(
                automatic_cache_breakpoints=True,
                provider_executed_tools=({"type": tool_type},),
            )
        )


def test_openai_provider_rates_default_to_unavailable() -> None:
    """Ordinary custom pricing requires no unused provider-tool rates."""
    parameters = inspect.signature(OpenAIPricingTable).parameters
    assert parameters["web_search_usd_per_invocation"].default is None
    assert parameters["file_search_usd_per_invocation"].default is None


def test_provider_executed_tools_follow_function_tools_in_responses() -> None:
    """Responses preserves each collection's order and places functions first."""
    schema = ToolSchema(
        name="echo",
        description="Echo the input.",
        args_schema={"type": "object"},
    )
    provider_tool: dict[str, object] = {"type": "web_search"}
    precomputed = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True,
            tool_schemas=(schema,),
            provider_executed_tools=(provider_tool,),
        )
    )
    assert not isinstance(precomputed.tools, openai.Omit)
    assert precomputed.tools[0]["type"] == "function"
    assert precomputed.tools[1] is provider_tool


def test_allowed_tools_choice_keeps_complete_responses_tool_definitions() -> None:
    """AllowedToolsChoice changes tool_choice without removing tools."""
    schemas = (
        ToolSchema(name="first", description="First.", args_schema={"type": "object"}),
        ToolSchema(name="second", description="Second.", args_schema={"type": "object"}),
    )
    provider_tool: dict[str, object] = {"type": "web_search"}
    precomputed = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True,
            tool_schemas=schemas,
            provider_executed_tools=(provider_tool,),
            tool_choice=AllowedToolsChoice(mode="auto", tool_names=("second",)),
        )
    )
    assert not isinstance(precomputed.tools, openai.Omit)
    assert precomputed.tools == [
        {
            "type": "function",
            "name": "first",
            "description": "First.",
            "parameters": {"type": "object"},
            "strict": None,
        },
        {
            "type": "function",
            "name": "second",
            "description": "Second.",
            "parameters": {"type": "object"},
            "strict": None,
        },
        provider_tool,
    ]
    assert precomputed.tool_choice == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "second"}],
    }


def test_request_rejects_an_extra_body_key_the_adapter_populates() -> None:
    """An extra_body key that open_stream passes as its own keyword raises at bind time.

    Rejecting the duplicate key prevents extra_body from overriding the binding.
    """
    with pytest.raises(ValueError, match="temperature"):
        _ = _adapter()._precompute_fields(
            _binding(automatic_cache_breakpoints=True, extra_body={"temperature": 0.5})
        )


def test_the_request_sends_extra_body_by_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_stream passes the binding's extra_body to the SDK's extra_body parameter."""
    adapter = _adapter()
    extra_body = {"safety_identifier": "user-7"}
    text_bound = _BoundOpenAIText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(
            _binding(automatic_cache_breakpoints=True, extra_body=extra_body)
        ),
    )
    assert _kwarg_sent(monkeypatch, text_bound, "extra_body") is extra_body


class _FakeSDKStream(AsyncResponseStream[None]):
    """Replays constructed events without a connection.

    Overrides iteration, close, and _response for _OpenAIStream.
    the base __init__ is deliberately not called, so the untouched base machinery stays unusable.
    """

    def __init__(  # pyrefly: ignore[missing-super-call]
        self, replay_events: Sequence[ResponseStreamEvent], headers: dict[str, str] | None = None
    ) -> None:
        self._replay_events = list(replay_events)
        self._response = httpx2.Response(
            200,
            headers=headers,
            request=httpx2.Request("POST", "https://api.openai.com/v1/responses"),
        )

    @override
    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        for replay_event in self._replay_events:
            yield replay_event

    @override
    async def close(self) -> None:
        return


def _stream(
    replay_events: Sequence[ResponseStreamEvent],
    headers: dict[str, str] | None = None,
    *,
    charged_provider_tools: bool = False,
) -> _OpenAIStream:
    """Build an adapter stream over replayed events, reading headers off a constructed response."""
    return _OpenAIStream(
        sdk_stream=_FakeSDKStream(replay_events, headers),
        pricing=_PRICING,
        regional_processing=False,
        charged_provider_tools=charged_provider_tools,
    )


def test_cutoff_openai_provider_tool_billing_is_nan() -> None:
    """A charged binding cannot report zero before terminal usage arrives."""
    billing = _stream([], charged_provider_tools=True).billing_reported()
    assert billing is not None
    assert math.isnan(billing.billing.usage.provider_executed_tool_cost_in_usd)
    assert _stream([]).billing_reported() is None


def _kwarg_sent[OutputT](
    monkeypatch: pytest.MonkeyPatch, bound: _BoundOpenAI[OutputT], key: str
) -> object:
    """Open one stream through a fake, capturing the request kwarg key it was passed."""
    captured: list[object] = []

    class _FakeStreamManager:
        async def __aenter__(self) -> _FakeSDKStream:
            return _FakeSDKStream([])

    def fake_stream(**request_kwargs: object) -> _FakeStreamManager:
        captured.append(request_kwargs[key])
        return _FakeStreamManager()

    monkeypatch.setattr(bound._adapter.client.responses, "stream", fake_stream)
    request = bound.build_request([UserMessage(content="q")])
    assert isinstance(request, RequestParams)
    _ = asyncio.run(bound.open_stream(request))
    (kwarg,) = captured
    return kwarg


def test_a_stream_reports_the_request_id_header_of_the_response_it_reads() -> None:
    """The stream's own response is the only channel a streamed turn has for the header.

    _response supplies the streamed request ID for provider support.
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
    """Text deltas pass through in order as the SDK's own strings. Nothing follows them."""
    translated = _collected_items([
        _text_delta_event("he", 1),
        _text_delta_event("y", 2),
        _completed_event(_response(usage=_usage_with_cache()), 3),
    ])
    assert translated == ["he", "y"]


def test_stream_yields_both_reasoning_channels_as_reasoning_deltas() -> None:
    """Summary deltas and reasoning-text deltas both become ReasoningDelta. Answer text stays a bare string.

    Which channel a model fills is request-time behavior, so the adapter forwards whichever arrives.
    """
    translated = _collected_items([
        _summary_delta_event("weighing", 0, 1),
        _reasoning_text_delta_event("deciding", 0, 2),
        _text_delta_event("hey", 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert translated == [ReasoningDelta(text="weighing"), ReasoningDelta(text="deciding"), "hey"]


def test_a_summary_part_boundary_streams_the_assembled_reasoning_part_separator() -> None:
    """A part's done event puts a blank line before the next part's first delta.

    A structural part boundary inserts a blank line between reasoning parts.

    The streamed text must match the completed ReasoningPart.text.
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
    reasoning_part = assistant_message.turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert streamed == reasoning_part.text == "First, water evaporates.\n\nThen it condenses."


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

    The pending separator belongs to the stream, not to the channel that created it.
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
    """An empty summary part does not add a separator."""
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
    """A pending separator before a dropped delta remains before the next part's text."""
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

    Only a reasoning delta consumes a pending separator.
    """
    translated = _collected_items([
        _summary_delta_event("thought it over", 0, 1),
        _summary_done_event("thought it over", 0, 2),
        _text_delta_event("hey", 3),
        _completed_event(_response(usage=_usage_with_cache()), 4),
    ])
    assert translated == [ReasoningDelta(text="thought it over"), "hey"]


def test_stream_yields_argument_fragments_then_one_complete_tool_call() -> None:
    """A function_call's argument deltas yield ToolCallDelta items named through its added event.

    Argument fragments concatenate into args_json.
    Empty fragments and message lifecycles yield nothing.
    """
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

    def args_fragment(
        delta: str, snapshot: str, sequence_number: int
    ) -> ResponseFunctionCallArgumentsDeltaEvent:
        return ResponseFunctionCallArgumentsDeltaEvent.model_validate({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc1",
            "output_index": 1,
            "sequence_number": sequence_number,
            "delta": delta,
            "snapshot": snapshot,
        })

    message_done = ResponseOutputItemDoneEvent.model_validate({
        "type": "response.output_item.done",
        "item": _TEXT_OUTPUT_ITEM,
        "output_index": 0,
        "sequence_number": 6,
    })
    function_call_done = ResponseOutputItemDoneEvent.model_validate({
        "type": "response.output_item.done",
        "item": _FUNCTION_CALL_OUTPUT_ITEM,
        "output_index": 1,
        "sequence_number": 7,
    })
    translated = _collected_items([
        message_added,
        function_call_added,
        args_fragment("", "", 3),
        args_fragment('{"q"', '{"q"', 4),
        args_fragment(": 1}", '{"q": 1}', 5),
        message_done,
        function_call_done,
        _completed_event(
            _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]), 8
        ),
    ])
    assert translated == [
        ToolCallDelta(id="call1", name="lookup", partial_args_json='{"q"'),
        ToolCallDelta(id="call1", name="lookup", partial_args_json=": 1}"),
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    ]


def test_stream_incomplete_terminal_still_assembles_final() -> None:
    """final() returns the captured incomplete terminal response."""

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
    """The completed terminal already carries a ParsedResponse. final() assembles from it."""

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
    """final()'s assistant turn includes the ReasoningPart from the terminal response's output."""

    async def scenario() -> None:
        adapter_stream = _stream([
            _completed_event(
                _response(usage=None, output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM]), 1
            ),
        ])
        async for _item in adapter_stream.items():
            pass
        result = _assert_result(await _text_final(adapter_stream))
        reasoning_part = result.assistant_message.turn[0]
        assert isinstance(reasoning_part, ReasoningPart)
        assert reasoning_part.raw == _REASONING_OUTPUT_ITEM

    asyncio.run(scenario())


def test_stream_failed_terminal_is_terminal_and_reports_the_provider_failure() -> None:
    """A failed terminal returns raw for failure interpretation and Billing."""

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
        assert _text_bound().billing_from_raw(raw).billing.usage.input_tokens_total == 1000

    asyncio.run(scenario())


def test_stream_final_passes_a_leniently_built_terminal_through_unvalidated() -> None:
    """Final preserves a leniently constructed terminal response by identity."""
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


def test_stream_error_event_raises_a_status_error_carrying_the_events_fields() -> None:
    """An error event without a terminal response raises APIStatusError."""

    async def scenario() -> None:
        adapter_stream = _stream([
            ResponseErrorEvent(
                type="error",
                code="server_error",
                message="The server had an error.",
                param=None,
                sequence_number=1,
            ),
        ])
        with pytest.raises(openai.APIStatusError) as caught:
            async for _item in adapter_stream.items():
                pass
        assert caught.value.status_code == 200
        assert caught.value.code == "server_error"
        assert "The server had an error." in str(caught.value)
        assert parse_openai(caught.value) == RetryThisOne(retry_after=None)

    asyncio.run(scenario())


def test_stream_ending_with_no_terminal_and_no_error_event_raises() -> None:
    """A stream that ends before any terminal event is a protocol failure, not an empty turn."""

    async def scenario() -> None:
        adapter_stream = _stream([_text_delta_event("he", 1)])
        with pytest.raises(StreamProtocolError):
            async for _item in adapter_stream.items():
                pass

    asyncio.run(scenario())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundOpenAIStructured[_StructuredReport]:
    """Build a structured-bound adapter over a keyless client. No request is sent."""
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(
        _binding(automatic_cache_breakpoints=False, system_prompt="sys")
    )
    return _BoundOpenAIStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )


def _structured_parse(response: OpenAIResponse) -> ResponseOutcome[_StructuredReport | None]:
    """Run the structured binding's parse over one response, with the turn that response carries."""
    return _structured_bound()._parsed_outcome(response, _assistant_message_from(response))


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
    """Build a response with optional text, refusal, and tool call."""
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
    assert _assert_result(outcome).output == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_output_may_inherit_no_output() -> None:
    """AdapterResult distinguishes successful output from NoOutput."""

    class ReportAlsoNoOutput(BaseModel, NoOutput):
        assistant_message: AssistantMessage = AssistantMessage(turn=())
        city: str
        celsius: int

    adapter = _adapter()
    bound = _BoundOpenAIStructured(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(
            _binding(system_prompt="sys", automatic_cache_breakpoints=False)
        ),
        response_format=ReportAlsoNoOutput,
    )
    outcome = bound.interpret(_structured_response(_REPORT_JSON))
    assert isinstance(outcome, AdapterResult)
    assert outcome.output == ReportAlsoNoOutput(city="Nairobi", celsius=25)


def test_structured_bind_reports_empty_turn_when_the_turn_carried_no_text() -> None:
    """A completed response with no text part and no tool call is EmptyTurn."""
    outcome = _structured_parse(_structured_response(None))
    assert isinstance(outcome, EmptyTurn)


def test_structured_bind_reports_schema_violation_on_text_the_model_rejects() -> None:
    """A completed turn whose text the response_format rejects is SchemaViolation.

    validation_error_json preserves the field, constraint, and rejected value.
    """
    outcome = _structured_parse(_structured_response('{"city": "Nairobi", "celsius": "SENTINEL"}'))
    assert isinstance(outcome, SchemaViolation)
    rejections = json.loads(outcome.validation_error_json)
    assert [rejection["loc"] for rejection in rejections] == [["celsius"]]
    assert rejections[0]["input"] == "SENTINEL"


def test_structured_bind_reports_max_completion_tokens_exceeded_on_text_cut_mid_json() -> None:
    """An incomplete turn whose JSON stopped mid-object is the truncation, not a schema violation.

    Truncated JSON at max_output_tokens returns MaxCompletionTokensExceeded.
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
    outcome = _structured_parse(_structured_response(None, tool_call=True))
    assert _assert_result(outcome).output is None


def test_structured_bind_reports_a_tool_call_turn_whose_text_is_not_the_instance_as_none() -> None:
    """A tool-call turn whose text is prose is the tool call, not a schema violation."""
    outcome = _structured_parse(_structured_response("let me look that up", tool_call=True))
    assert _assert_result(outcome).output is None


def test_structured_bind_sets_output_on_a_turn_that_also_called_a_tool() -> None:
    """The instance lands on output and the call still lands on tool_calls, so neither fact hides the other."""
    outcome = _structured_parse(_structured_response(_REPORT_JSON, tool_call=True))
    assert _assert_result(outcome).output == _StructuredReport(city="Nairobi", celsius=25)


def _text_bound() -> _BoundOpenAIText:
    """Build a text-bound adapter over a keyless client. No request is sent."""
    adapter = _adapter()
    return _BoundOpenAIText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(automatic_cache_breakpoints=False)),
    )


def test_text_bind_reports_the_refusal_sentences_as_the_output() -> None:
    """A refused turn's output is the text the model wrote, not the empty output_text.

    Response.output and assistant_message.text carry the same refusal text.
    """
    response = _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM])
    result = _assert_result(_text_bound().interpret(response))
    assert result.output == "I can't help with that"


def test_identity_reads_the_responses_own_id_and_served_model() -> None:
    """ResponseIdentity uses the served model and optional request ID."""
    identity = _text_bound().identity_from_raw(
        _response(usage=None, model="m-2026-01-01"), request_id=None
    )
    assert identity == ResponseIdentity(
        model_served="m-2026-01-01", response_id="r1", request_id=None
    )


def test_identity_reads_the_adapter_stream_request_id() -> None:
    """The AdapterStream request id reaches ResponseIdentity unchanged."""
    response = _response(usage=None)
    identity = _text_bound().identity_from_raw(response, request_id="req_openai")
    assert identity.request_id == "req_openai"


def test_structured_bind_reports_the_failure_on_a_failed_status_whose_text_validates() -> None:
    """A failed run is the failure variant even when its fragment validates: it is not the answer."""
    outcome = _structured_parse(
        _structured_response(_REPORT_JSON, status="failed", error=_SERVER_ERROR)
    )
    assert isinstance(outcome, ProviderFailedTransiently)


def test_a_failed_run_carrying_a_refusal_takes_the_failure_variant_under_both_bindings() -> None:
    """A failed status wins over the refusal test, so one response does not split by binding.

    Structured binding classifies incomplete before refusal.
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

    A blocked request returns a terminal failure without retries.
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
    """Both bindings send include=["reasoning.encrypted_content"] on the request."""
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(_binding(automatic_cache_breakpoints=True))
    text_bound = _BoundOpenAIText(adapter=adapter, precomputed_fields=precomputed_fields)
    structured_bound = _BoundOpenAIStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )
    includes = [
        _kwarg_sent(monkeypatch, text_bound, "include"),
        _kwarg_sent(monkeypatch, structured_bound, "include"),
    ]
    assert includes == [["reasoning.encrypted_content"]] * 2


def test_the_structured_request_sends_the_text_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_stream puts the precomputed text parameter on the request.

    The request preserves the response schema in text.
    """
    adapter = _adapter()
    structured_bound = _BoundOpenAIStructured(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(automatic_cache_breakpoints=True)),
        response_format=_StructuredReport,
    )
    assert _kwarg_sent(monkeypatch, structured_bound, "text") == {
        "format": type_to_text_format_param(_StructuredReport)
    }


def test_a_built_request_renders_as_json_carrying_the_prompt_and_no_omitted_field() -> None:
    """as_json holds the binding's precomputed fields and this call's converted input.

    An unstated temperature is absent from the request.
    """
    request = _structured_bound().build_request([UserMessage(content="hi")])
    assert isinstance(request, _OpenAIRequestParams)
    rendered = json.loads(request.as_json())
    assert rendered["input"] == [{"role": "user", "content": "hi"}]
    assert rendered["precomputed"]["instructions"] == "sys"
    assert "temperature" not in rendered["precomputed"]


def _rate_limit_error(headers: dict[str, str]) -> openai.RateLimitError:
    """Build the SDK's 429 exception around a constructed httpx2.Response."""
    response = httpx2.Response(
        429,
        headers=headers,
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_parse_openai_reads_retry_after_from_the_headers_without_letting_it_pick() -> None:
    """Read mixed-case Retry-After-MS without changing the verdict."""
    assert parse_openai(_rate_limit_error({"Retry-After-MS": "1500"})) == PauseAll(retry_after=1.5)
    assert parse_openai(_rate_limit_error({})) == PauseAll(retry_after=None)
    bad_request = status_error(openai.BadRequestError, 400, {"retry-after": "7"})
    assert parse_openai(bad_request) == DoNotRetry()


def test_parse_openai_does_not_retry_a_429_naming_a_spend_limit_code() -> None:
    """error.code separates the 429 no wait restores from the rate limit a pause absorbs."""
    exhausted = status_error(openai.RateLimitError, 429, error_code="credit_balance_exhausted")
    assert parse_openai(exhausted) == DoNotRetry()
    throttled = status_error(openai.RateLimitError, 429, error_code="rate_limit_exceeded")
    assert parse_openai(throttled) == PauseAll(retry_after=None)


def test_parse_openai_obeys_a_retry_directive_over_the_status_tables() -> None:
    """x-should-retry overrides the table verdict, which is what the SDK client does with it.

    "false" on a 500 stops a status the table retries, and "true" on a 400 retries one it stops.
    """
    final_500 = status_error(openai.InternalServerError, 500, {"x-should-retry": "false"})
    assert parse_openai(final_500) == DoNotRetry()
    retryable_400 = status_error(
        openai.BadRequestError, 400, {"x-should-retry": "true", "retry-after": "3"}
    )
    assert parse_openai(retryable_400) == RetryThisOne(retry_after=3.0)


def test_false_retry_directive_stops_request_and_pauses_rate_limit_quota() -> None:
    """A "false" directive stops this request and pauses the rate-limit quota."""
    throttled = status_error(
        openai.RateLimitError, 429, {"x-should-retry": "false", "retry-after": "7"}
    )
    assert parse_openai(throttled) == PauseAllDoNotRetry(retry_after=7.0)


def test_parse_openai_applies_a_directive_to_a_spend_limit_429_by_its_verdict() -> None:
    """x-should-retry overrides a spend-limit DoNotRetry verdict."""
    exhausted = status_error(
        openai.RateLimitError, 429, {"x-should-retry": "false"}, "credit_balance_exhausted"
    )
    assert parse_openai(exhausted) == DoNotRetry()
    retryable = status_error(
        openai.RateLimitError, 429, {"x-should-retry": "true"}, "credit_balance_exhausted"
    )
    assert parse_openai(retryable) == RetryThisOne(retry_after=None)


def test_parse_openai_ignores_a_retry_directive_on_the_streams_200_status() -> None:
    """A 200's headers belong to a request the provider accepted, so they judge no failure.

    A mid-stream error at status 200 uses its error code.
    """
    throttled = status_error(
        openai.APIStatusError, 200, {"x-should-retry": "false"}, "rate_limit_exceeded"
    )
    assert parse_openai(throttled) == PauseAll(retry_after=None)
    blocked = status_error(
        openai.APIStatusError, 200, {"x-should-retry": "true"}, "invalid_prompt"
    )
    assert parse_openai(blocked) == DoNotRetry()


def test_parse_openai_counts_a_fallthrough_and_a_listed_row_adds_nothing() -> None:
    """An unlisted status lands one tagged count. A listed status leaves the counter alone."""
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    assert parse_openai(status_error(openai.InternalServerError, 500)) == RetryThisOne(
        retry_after=None
    )
    assert parse_openai(status_error(openai.NotFoundError, 404)) == DoNotRetry()
    assert dict(PARSE_FALLTHROUGH_COUNTS) == before
    assert parse_openai(status_error(openai.APIStatusError, 599)) == RetryThisOne(retry_after=None)
    tag = "status=599 type=None"
    assert PARSE_FALLTHROUGH_COUNTS[tag] == before.get(tag, 0) + 1


def test_parse_openai_verdicts_a_status_200_by_the_errors_code() -> None:
    """A 200 is a mid-stream error event's raise, so the code picks the verdict the status cannot.

    server_error retries one request.
    rate_limit_exceeded pauses SharedBackoff.
    Terminal codes do not retry.
    """
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    server_error = status_error(openai.APIStatusError, 200, error_code="server_error")
    assert parse_openai(server_error) == RetryThisOne(retry_after=None)
    throttled = status_error(openai.APIStatusError, 200, error_code="rate_limit_exceeded")
    assert parse_openai(throttled) == PauseAll(retry_after=None)
    blocked = status_error(openai.APIStatusError, 200, error_code="invalid_prompt")
    assert parse_openai(blocked) == DoNotRetry()
    assert dict(PARSE_FALLTHROUGH_COUNTS) == before


def test_parse_openai_counts_a_status_200_code_outside_the_table_as_a_fallthrough() -> None:
    """An unknown code and an absent code on a 200 each land DoNotRetry and one tagged count."""
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    unknown = status_error(openai.APIStatusError, 200, error_code="brand_new_code")
    assert parse_openai(unknown) == DoNotRetry()
    assert parse_openai(status_error(openai.APIStatusError, 200)) == DoNotRetry()
    for tag in ("status=200 type=brand_new_code", "status=200 type=None"):
        assert PARSE_FALLTHROUGH_COUNTS[tag] == before.get(tag, 0) + 1


def test_request_id_from_error_reads_the_sdk_errors_own_header_and_nothing_else() -> None:
    """The override reports the header the SDK read off the error response, None for any other error.

    OpenAI sends the request ID in x-request-id.
    Missing headers return None.
    """
    adapter = _adapter()
    assert (
        adapter.request_id_from_error(_rate_limit_error({"x-request-id": "req_429"})) == "req_429"
    )
    assert adapter.request_id_from_error(_rate_limit_error({})) is None
    assert adapter.request_id_from_error(ValueError("boom")) is None


def test_adapter_pins_sdk_retries_off() -> None:
    """The stored client copy carries max_retries=0 so only langchaint retries."""
    assert _adapter().client.max_retries == 0


def test_wire_input_marks_marked_user_and_tool_parts() -> None:
    """A marked part carries prompt_cache_breakpoint on its wire part. Unmarked siblings carry none."""
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
    """A parts system_prompt travels as a developer-role input message. instructions stays unset."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True,
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
        _binding(automatic_cache_breakpoints=True, system_prompt="sys")
    )
    assert precomputed_fields.instructions == "sys"
    assert precomputed_fields.input_prefix == []


def _conformance_output() -> list[object]:
    """Build reasoning, web-search, and message items.

    The reasoning item carries an extra raw field.
    """
    return [
        _REASONING_OUTPUT_ITEM | {"field_newer_than_sdk": "x"},
        dict(_WEB_SEARCH_OUTPUT_ITEM),
        dict(_TEXT_OUTPUT_ITEM),
    ]


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
        return _response(usage=None, output=[dict(_TEXT_OUTPUT_ITEM)])

    @override
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return a turn served at flex, which _PRICING holds no table for."""
        return _response(
            usage=_usage_with_cache(), output=_conformance_output(), service_tier="flex"
        )

    @override
    def response_with_impossible_counters(self) -> BaseModel:
        """Return a turn whose cache counts sum past input_tokens.

        Excess cache counts make the derived uncached counter negative.
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
    def response_with_raw_part(self) -> BaseModel | None:
        """Return the turn whose middle item is the built-in web search call."""
        return _response(usage=_usage_with_cache(), output=_conformance_output())

    @override
    def assistant_wire_parts(self, request: RequestParams) -> Sequence[object]:
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
        """Return the shared OpenAI classification table."""
        return openai_sdk_errors_and_classifications()

    @override
    def sdk_errors_and_verdicts(self) -> Mapping[Exception, Verdict]:
        """Return the shared OpenAI verdict table."""
        return openai_sdk_errors_and_verdicts()
