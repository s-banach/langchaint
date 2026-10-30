import json
import math
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import override

import httpx2
import openai
import pytest
from openai import AsyncOpenAI
from openai._models import construct_type_unchecked
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
    ResponseOutputMessage,
    ResponseOutputText,
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
from pydantic import BaseModel

from langchaint import (
    LLM,
    AllowedToolsChoice,
    AssistantMessage,
    AudioPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
    Message,
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
    AdapterStream,
    Binding,
    BoundAdapter,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    RejectedMessages,
    RequestFailure,
    RequestParams,
    ResponseIdentity,
    ResponseOutcome,
    UsableResponse,
)
from langchaint.billing.pricing import Billing, LongContextPricing, TokenRates
from langchaint.common.exceptions import StreamProtocolError
from langchaint.conformance import AdapterConformance
from langchaint.openai import (
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
    OpenAIResponsesServiceTier,
    ReasoningSummary,
)
from langchaint.openai.responses_adapter import (
    _assistant_items,
    _assistant_message_from,
    _BoundOpenAIResponsesStructured,
    _BoundOpenAIResponsesText,
    _OpenAIResponsesRequestParams,
    _OpenAIResponsesStream,
    _wire_input,
    _wire_tool_choice,
)
from langchaint.openai.responses_adapter import (
    _billing_from_response as _provider_billing_from_response,
)
from langchaint.openai.shared import REQUEST_FAILURE_FALLTHROUGH_COUNTS, openai_request_failure
from langchaint.tools import ToolSchema
from tests.helpers import (
    openai_sdk_errors_and_request_failures,
    run_with_timeout,
    status_error,
    terminal,
    transient,
)


def _billing_from_response(response: OpenAIResponse, pricing: OpenAIPricingTable) -> Billing:
    return _provider_billing_from_response(response, pricing).billing


_DEFAULT_RATES = TokenRates(
    input_tokens_cache_none=2.5,
    output_tokens=10.0,
    input_tokens_cache_read=1.25,
    input_tokens_cache_write=3.125,
)

_PRICING = OpenAIPricingTable(
    default=_DEFAULT_RATES,
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)
"""The default tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = TokenRates(
    input_tokens_cache_none=5.0,
    output_tokens=20.0,
    input_tokens_cache_read=2.5,
    input_tokens_cache_write=6.25,
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
"""One built-in tool call without another AssistantPart variant."""

_FILE_SEARCH_OUTPUT_ITEM: dict[str, object] = {
    "type": "file_search_call",
    "id": "fs_1",
    "status": "completed",
    "queries": ["first", "second"],
}


def _assert_usable[OutputT](outcome: ResponseOutcome[OutputT]) -> UsableResponse[OutputT]:
    assert outcome.kind == "usable_response"
    return outcome


def _usage_with_cache() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=1000,
        input_tokens_details=InputTokensDetails(cached_tokens=600, cache_write_tokens=100),
        output_tokens=40,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
        total_tokens=1040,
    )


_SERVER_ERROR = ResponseError(code="server_error", message="The server had an error.")
"""A failed response's error whose code the disposition table calls transient."""


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
    raw = _response(usage=_usage_with_cache())
    provider_billing = _provider_billing_from_response(raw, _PRICING)
    billing = provider_billing.billing
    usage = billing.usage
    assert provider_billing.usage_raw is raw.usage
    assert billing.usd_per_million_tokens.input_tokens_cache_none == 2.5
    assert billing.usd_per_million_tokens.input_tokens_cache_read == 1.25
    assert billing.usd_per_million_tokens.input_tokens_cache_write == 3.125
    assert usage.input_tokens_cache_read == 600
    assert usage.input_tokens_cache_write == 100
    assert usage.input_tokens_cache_none == 300
    assert usage.output_tokens_reasoning == 8
    assert usage.input_tokens_cache_none_cost_in_usd == 300 * 2.5 / 1e6
    assert usage.input_tokens_cache_read_cost_in_usd == 600 * 1.25 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == 100 * 3.125 / 1e6
    assert usage.output_tokens_cost_in_usd == 40 * 10.0 / 1e6


_COST_AT_DEFAULT_RATES = (300 * 2.5 + 600 * 1.25 + 100 * 3.125 + 40 * 10.0) / 1e6
"""The cost of `_usage_with_cache` at `_DEFAULT_RATES`."""


@pytest.mark.parametrize(
    ("service_tier", "usage", "expected_service_tier", "expected_output_rate", "expected_cost"),
    [
        ("priority", _usage_with_cache(), "priority", 20.0, 2 * _COST_AT_DEFAULT_RATES),
        ("default", _usage_with_cache(), "default", 10.0, _COST_AT_DEFAULT_RATES),
        ("auto", _usage_with_cache(), "default", 10.0, _COST_AT_DEFAULT_RATES),
        (None, _usage_with_cache(), "default", 10.0, _COST_AT_DEFAULT_RATES),
        (None, None, "default", 10.0, 0.0),
        ("flex", _usage_with_cache(), "flex", float("nan"), float("nan")),
        ("flex", None, "flex", float("nan"), 0.0),
    ],
    ids=[
        "priority",
        "default",
        "auto_reports_default",
        "no_tier_reports_default",
        "no_usage_keeps_the_tiers_rates",
        "unpriced_tier",
        "unpriced_tier_that_billed_nothing_costs_zero",
    ],
)
def test_the_reported_tier_selects_the_rates_and_names_the_billing(
    service_tier: str | None,
    usage: ResponseUsage | None,
    expected_service_tier: str,
    expected_output_rate: float,
    expected_cost: float,
) -> None:
    pricing = OpenAIPricingTable(default=_DEFAULT_RATES, fast=_PRIORITY_RATES)
    billing = _billing_from_response(_response(usage=usage, service_tier=service_tier), pricing)
    assert billing.service_tier == expected_service_tier
    assert billing.usd_per_million_tokens.output_tokens == pytest.approx(
        expected_output_rate, nan_ok=True
    )
    assert billing.usage.cost_in_usd == pytest.approx(expected_cost, nan_ok=True)


def _web_search_with_action(action: dict[str, object]) -> dict[str, object]:
    return _WEB_SEARCH_OUTPUT_ITEM | {"action": action}


@pytest.mark.parametrize(
    ("output", "service_tier", "expected_cost"),
    [
        (
            [
                _WEB_SEARCH_OUTPUT_ITEM,
                _FILE_SEARCH_OUTPUT_ITEM,
                _FILE_SEARCH_OUTPUT_ITEM | {"id": "fs_2", "queries": ["third"]},
            ],
            None,
            0.01 + 2 * 0.0025,
        ),
        (
            [_web_search_with_action({"type": "open_page", "url": "https://example.com"})],
            None,
            0.0,
        ),
        (
            [
                _web_search_with_action({
                    "type": "find_in_page",
                    "url": "https://example.com",
                    "pattern": "price",
                })
            ],
            None,
            0.0,
        ),
        (
            [
                {
                    "type": "image_generation_call",
                    "id": "image-1",
                    "status": "completed",
                    "result": "image-data",
                }
            ],
            None,
            float("nan"),
        ),
        ([_WEB_SEARCH_OUTPUT_ITEM], "flex", 0.01),
    ],
    ids=[
        "one_search_and_one_fee_per_file_search_item",
        "open_page_action_is_free",
        "find_in_page_action_is_free",
        "image_generation_is_unpriceable",
        "unpriced_token_tier_keeps_the_tool_cost",
    ],
)
def test_provider_executed_tool_cost_counts_priced_output_items(
    output: list[object], service_tier: str | None, expected_cost: float
) -> None:
    usage = _billing_from_response(
        _response(usage=None, output=output, service_tier=service_tier), _PRICING
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(expected_cost, nan_ok=True)


def test_billing_at_a_tier_the_table_has_no_field_for_is_nan() -> None:
    raw = _response(usage=_usage_with_cache()).model_copy(update={"service_tier": "decisions"})
    usage = _billing_from_response(raw, _PRICING).usage
    assert math.isnan(usage.input_tokens_cache_none_cost_in_usd)
    assert math.isnan(usage.output_tokens_cost_in_usd)


def test_pricing_table_scaled_scales_every_tier_and_keeps_modifiers() -> None:
    long_context = LongContextPricing(
        prompt_tokens_above=272_000, input_multiplier=2.0, output_multiplier=1.5
    )
    table = OpenAIPricingTable(
        default=_DEFAULT_RATES,
        flex=_DEFAULT_RATES,
        scale=_DEFAULT_RATES,
        long_context=long_context,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    )
    assert table.scaled(2.0) == OpenAIPricingTable(
        default=_PRIORITY_RATES,
        flex=_PRIORITY_RATES,
        scale=_PRIORITY_RATES,
        long_context=long_context,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    )


_REFUSAL_MESSAGE_ITEM: dict[str, object] = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": "I can't help with that"}],
}


def _incomplete_response(reason: str) -> OpenAIResponse:
    return _response(
        usage=None,
        status="incomplete",
        incomplete_details=IncompleteDetails.model_construct(reason=reason),
    )


@pytest.mark.parametrize(
    ("response", "expected", "expected_output"),
    [
        (_response(usage=None), "stop", "hey"),
        (
            _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]),
            "tool_call",
            None,
        ),
        (
            _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM]),
            "refusal",
            "I can't help with that",
        ),
        (
            _response(usage=None, output=[_REFUSAL_MESSAGE_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]),
            "refusal",
            None,
        ),
        (_incomplete_response("max_output_tokens"), "max_completion_tokens", "hey"),
        (_incomplete_response("content_filter"), "refusal", "hey"),
        (_incomplete_response("max_messages"), "other", "hey"),
        (_incomplete_response("steered"), "other", "hey"),
    ],
    ids=[
        "completed",
        "function_call_item",
        "refusal_block",
        "refusal_block_beside_a_function_call",
        "incomplete_max_output_tokens",
        "incomplete_content_filter",
        "incomplete_max_messages",
        "incomplete_steered",
    ],
)
def test_stop_reason_mapping(
    response: OpenAIResponse, expected: StopReason, expected_output: str
) -> None:
    usable = _assert_usable(_text_bound().interpret(response))
    assert usable.stop_reason == expected
    assert usable.output == expected_output


def test_every_output_item_replays_as_one_input_item_in_position() -> None:
    assistant_message = _assistant_message_from(
        _response(
            usage=None,
            output=[
                _REASONING_OUTPUT_ITEM,
                _WEB_SEARCH_OUTPUT_ITEM,
                _TEXT_OUTPUT_ITEM,
                _FUNCTION_CALL_OUTPUT_ITEM,
            ],
        )
    )
    assert _assistant_items(assistant_message) == [
        _REASONING_OUTPUT_ITEM,
        _WEB_SEARCH_OUTPUT_ITEM,
        {"role": "assistant", "content": "hey", "phase": "commentary"},
        {"type": "function_call", "call_id": "call1", "name": "lookup", "arguments": '{"q": 1}'},
    ]


def _reasoning_item(
    *, summary: tuple[str, ...] = (), content: tuple[str, ...] | None = None
) -> dict[str, object]:
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
    response = _response(usage=None, output=[_reasoning_item(summary=summary, content=content)])
    assistant_message = _assistant_message_from(response)
    assert isinstance(assistant_message.content, tuple)
    reasoning_part = assistant_message.content[0]
    assert reasoning_part.kind == "reasoning"
    assert reasoning_part.text == expected_text


def test_text_parts_replay_as_separate_messages() -> None:
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
    assert assistant_message.content == (TextPart(text="he"), TextPart(text="y"))
    assert _assistant_items(assistant_message) == [
        {"role": "assistant", "content": "he", "phase": "commentary"},
        {"role": "assistant", "content": "y", "phase": "final_answer"},
    ]


@pytest.mark.parametrize("content", ["answer", (TextPart(text="answer"),)])
def test_text_binding_replays_plain_text_and_one_text_part_as_a_final_answer(
    content: str | tuple[TextPart, ...],
) -> None:
    request = _text_bound().build_request_params([AssistantMessage(content=content)])
    assert isinstance(request, _OpenAIResponsesRequestParams)
    assert request.precomputed.text is openai.omit
    assert request.input == [
        {"role": "assistant", "content": "answer", "phase": "final_answer"},
    ]


def test_replay_uses_edited_content_to_reconstruct_text_and_phases() -> None:
    response = _response(
        usage=None,
        output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM],
    )
    assistant_message = _assistant_message_from(response)
    assert isinstance(assistant_message.content, tuple)
    tool_call = assistant_message.content[-1]
    assert tool_call.kind == "tool_call"
    edited = assistant_message.model_copy(
        update={
            "content": (
                TextPart(text="Checking"),
                tool_call.model_copy(update={"args_json": '{"q": 2}'}),
                TextPart(text="Waiting"),
            ),
        }
    )
    assert _assistant_items(edited) == [
        {"role": "assistant", "content": "Checking", "phase": "commentary"},
        {"type": "function_call", "call_id": "call1", "name": "lookup", "arguments": '{"q": 2}'},
        {"role": "assistant", "content": "Waiting", "phase": "commentary"},
    ]
    assert isinstance(edited.content, tuple)
    without_tool_calls = edited.model_copy(
        update={
            "content": tuple(part for part in edited.content if part.kind != "tool_call"),
        }
    )
    assert _assistant_items(without_tool_calls) == [
        {"role": "assistant", "content": "Checking", "phase": "commentary"},
        {"role": "assistant", "content": "Waiting", "phase": "final_answer"},
    ]


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    raw: dict[str, JsonValue] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(content=(ReasoningPart(raw=raw), TextPart(text="hi")))
    assert _assistant_items(assistant_message) == [
        raw,
        {"role": "assistant", "content": "hi", "phase": "final_answer"},
    ]


def test_wire_input_converts_each_message_kind() -> None:
    wire = _wire_input(
        [
            UserMessage(content="q"),
            AssistantMessage(
                content=(
                    TextPart(text="thinking"),
                    ToolCall(id="call1", name="lookup", args_json="{}"),
                ),
            ),
            ToolMessage(tool_call_id="call1", content="r"),
        ],
        supports_cache_breakpoints=True,
    )
    assert wire == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking", "phase": "commentary"},
        {"type": "function_call", "call_id": "call1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call1", "output": "r"},
    ]


_IMAGE_URL = "https://example.com/image.png"
_EXPLICIT_BREAKPOINT: dict[str, object] = {"prompt_cache_breakpoint": {"mode": "explicit"}}


def test_wire_input_converts_content_parts_and_only_their_cache_breakpoints() -> None:
    wire = _wire_input(
        [
            UserMessage(
                content=(
                    TextPart(text="shared context", cache_breakpoint=True),
                    TextPart(text="question"),
                    ImagePart(data=b"png", media_type="image/png"),
                    ImageUrlPart(url=_IMAGE_URL, cache_breakpoint=True),
                )
            ),
            ToolMessage(
                tool_call_id="call1",
                content=(
                    TextPart(text="saw"),
                    ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
                    ImageUrlPart(url=_IMAGE_URL),
                ),
            ),
        ],
        supports_cache_breakpoints=True,
    )
    data_image = {
        "type": "input_image",
        "image_url": "data:image/png;base64,cG5n",
        "detail": "auto",
    }
    url_image = {"type": "input_image", "image_url": _IMAGE_URL, "detail": "auto"}
    assert wire == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "shared context"} | _EXPLICIT_BREAKPOINT,
                {"type": "input_text", "text": "question"},
                data_image,
                url_image | _EXPLICIT_BREAKPOINT,
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call1",
            "output": [
                {"type": "input_text", "text": "saw"},
                data_image | _EXPLICIT_BREAKPOINT,
                url_image,
            ],
        },
    ]


def test_wire_input_sends_every_cache_breakpoint_without_a_client_side_cap() -> None:
    wire = _wire_input(
        [
            UserMessage(
                content=tuple(
                    TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(5)
                )
            ),
        ],
        supports_cache_breakpoints=True,
    )
    assert wire == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"m{index}"} | _EXPLICIT_BREAKPOINT
                for index in range(5)
            ],
        }
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
def test_build_request_reports_audio_as_rejected_messages(message: Message) -> None:
    request = (
        _adapter()
        .bind_text(_binding(automatic_cache_breakpoints=True))
        .build_request_params([message])
    )
    assert isinstance(request, RejectedMessages)
    assert "AudioPart" in request.error_text
    assert type(message).__name__ in request.error_text


def test_wire_tool_choice_passes_strings_through_and_names_specific_tools() -> None:
    assert _wire_tool_choice("auto") == "auto"
    assert _wire_tool_choice("required") == "required"
    assert _wire_tool_choice("none") == "none"
    assert _wire_tool_choice(SpecificToolChoice(tool_name="x")) == {
        "type": "function",
        "name": "x",
    }


def _adapter(
    *,
    client: AsyncOpenAI | None = None,
    pricing: OpenAIPricingTable = _PRICING,
    provider_name: str = "openai",
    reasoning_summary: ReasoningSummary | None = None,
    supports_cache_breakpoints: bool = True,
    service_tier: OpenAIResponsesServiceTier | None = None,
) -> OpenAIResponsesAdapter:
    """Use a placeholder API key; request tests supply an offline HTTP transport."""
    return OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test") if client is None else client,
        model="m",
        pricing=pricing,
        provider_name=provider_name,
        supports_cache_breakpoints=supports_cache_breakpoints,
        reasoning_summary=reasoning_summary,
        service_tier=service_tier,
    )


def test_config_fingerprint_data_contains_only_stored_request_configuration() -> None:
    adapter = OpenAIResponsesAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_cache_breakpoints=False,
        reasoning_summary="detailed",
        service_tier="ultrafast",
    )
    assert adapter.config_fingerprint_data() == {
        "reasoning_summary": "detailed",
        "service_tier": "ultrafast",
        "supports_cache_breakpoints": False,
    }


def _binding(
    *,
    automatic_cache_breakpoints: bool,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    tool_schemas: tuple[ToolSchema, ...] = (),
    reasoning_level: str | None = None,
    temperature: float | None = None,
    provider_executed_tools: tuple[Mapping[str, object], ...] = (),
    tool_choice: ToolChoice = "auto",
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        provider_executed_tools=provider_executed_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
        max_completion_tokens=None,
        reasoning_level=reasoning_level,
        temperature=temperature,
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
    precomputed_fields = _adapter(reasoning_summary=reasoning_summary)._precompute_fields(
        _binding(automatic_cache_breakpoints=True, reasoning_level=reasoning_level)
    )
    if expected_reasoning is None:
        assert isinstance(precomputed_fields.reasoning, openai.Omit)
    else:
        assert precomputed_fields.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("supports_cache_breakpoints", "automatic_cache_breakpoints", "expected_options"),
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
    supports_cache_breakpoints: bool,
    automatic_cache_breakpoints: bool,
) -> None:
    precomputed_fields = _adapter(
        supports_cache_breakpoints=supports_cache_breakpoints
    )._precompute_fields(_binding(automatic_cache_breakpoints=automatic_cache_breakpoints))
    if expected_options is None:
        assert isinstance(precomputed_fields.prompt_cache_options, openai.Omit)
    else:
        assert precomputed_fields.prompt_cache_options == expected_options


def test_the_refusal_reaches_bind_before_any_request_is_built() -> None:
    llm = LLM(_adapter(supports_cache_breakpoints=False))
    with pytest.raises(ValueError, match="model 'm'"):
        _ = llm.bind(automatic_cache_breakpoints=False)


def test_a_request_under_a_default_binding_omits_every_unstated_field() -> None:
    request = (
        _adapter()
        .bind_text(_binding(automatic_cache_breakpoints=True, system_prompt="sys"))
        .build_request_params([UserMessage(content="hi")])
    )
    assert isinstance(request, _OpenAIResponsesRequestParams)
    assert json.loads(request.as_json()) == {
        "precomputed": {
            "model": "m",
            "instructions": "sys",
            "input_prefix": [],
            "include": ["reasoning.encrypted_content"],
            "extra_body": None,
            "charged_provider_executed_tools": False,
        },
        "input": [{"role": "user", "content": "hi"}],
    }


def test_request_sends_a_stated_service_tier_and_temperature() -> None:
    precomputed_fields = _adapter(service_tier="flex")._precompute_fields(
        _binding(automatic_cache_breakpoints=True, temperature=0.2)
    )
    assert precomputed_fields.service_tier == "flex"
    assert precomputed_fields.temperature == 0.2


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
def test_every_supported_provider_executed_type_reaches_responses_unchanged(
    tool_type: str,
) -> None:
    provider_executed_tool: dict[str, object] = {"type": tool_type, "search_context_size": "low"}
    precomputed = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True, provider_executed_tools=(provider_executed_tool,)
        )
    )
    assert precomputed.tools == [provider_executed_tool]


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
    with pytest.raises(ValueError, match="supported string type"):
        _ = _adapter()._precompute_fields(
            _binding(
                automatic_cache_breakpoints=True,
                provider_executed_tools=({"type": tool_type},),
            )
        )


def test_provider_executed_tools_require_direct_openai_billing() -> None:
    with pytest.raises(ValueError, match="provider_name='openai'"):
        _ = _adapter(provider_name="azure.ai.openai")._precompute_fields(
            _binding(
                automatic_cache_breakpoints=True,
                provider_executed_tools=({"type": "web_search"},),
            )
        )


@pytest.mark.parametrize("tool_type", ["web_search", "file_search"])
def test_a_bound_openai_tool_requires_its_price(tool_type: str) -> None:
    pricing = OpenAIPricingTable(
        default=_DEFAULT_RATES,
        web_search_usd_per_invocation=None if tool_type == "web_search" else 0.01,
        file_search_usd_per_invocation=None if tool_type == "file_search" else 0.0025,
    )
    with pytest.raises(ValueError, match="is None"):
        _ = _adapter(pricing=pricing)._precompute_fields(
            _binding(
                automatic_cache_breakpoints=True,
                provider_executed_tools=({"type": tool_type},),
            )
        )


def test_tool_without_description_omits_the_description_key() -> None:
    precomputed = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True,
            tool_schemas=(ToolSchema(name="lookup", args_schema={"type": "object"}),),
        )
    )
    assert precomputed.tools == [
        {"type": "function", "name": "lookup", "parameters": {"type": "object"}, "strict": None}
    ]


def test_allowed_tools_choice_keeps_complete_responses_tool_definitions() -> None:
    schemas = (
        ToolSchema(name="first", description="First.", args_schema={"type": "object"}),
        ToolSchema(name="second", description="Second.", args_schema={"type": "object"}),
    )
    provider_executed_tool: dict[str, object] = {"type": "web_search"}
    precomputed = _adapter()._precompute_fields(
        _binding(
            automatic_cache_breakpoints=True,
            tool_schemas=schemas,
            provider_executed_tools=(provider_executed_tool,),
            tool_choice=AllowedToolsChoice(mode="required", tool_names=("second", "first")),
        )
    )
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
        provider_executed_tool,
    ]
    assert precomputed.tool_choice == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "second"}, {"type": "function", "name": "first"}],
    }
    assert precomputed.parallel_tool_calls is True


def test_request_rejects_an_extra_body_key_the_adapter_populates() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _ = _adapter()._precompute_fields(
            _binding(automatic_cache_breakpoints=True, extra_body={"temperature": 0.5})
        )


def _request_body_sent[OutputT](
    bind: Callable[[OpenAIResponsesAdapter], BoundAdapter[OutputT]],
) -> object:
    bodies: list[bytes] = []

    def record_request(request: httpx2.Request) -> httpx2.Response:
        bodies.append(request.content)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"})

    bound = bind(
        _adapter(
            client=AsyncOpenAI(
                api_key="offline",
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record_request)),
            )
        )
    )
    request = bound.build_request_params([UserMessage(content="q")])
    assert isinstance(request, RequestParams)
    _ = run_with_timeout(bound.open_stream(request))
    (body,) = bodies
    decoded_body: object = json.loads(body)
    return decoded_body


def test_the_request_body_carries_the_reasoning_include_and_extra_body_by_reference() -> None:
    extra_body = {"safety_identifier": "user-7"}

    def bind_then_edit_extra_body(adapter: OpenAIResponsesAdapter) -> BoundAdapter[str | None]:
        bound = adapter.bind_text(
            _binding(automatic_cache_breakpoints=True, extra_body=extra_body)
        )
        extra_body["safety_identifier"] = "user-8"
        return bound

    assert _request_body_sent(bind_then_edit_extra_body) == {
        "model": "m",
        "instructions": None,
        "input": [{"role": "user", "content": "q"}],
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": True,
        "safety_identifier": "user-8",
    }


class _StructuredReport(BaseModel):
    city: str
    celsius: int


def test_the_structured_request_sends_a_strict_response_schema() -> None:
    assert _request_body_sent(
        lambda adapter: adapter.bind_structured(
            _binding(automatic_cache_breakpoints=True), _StructuredReport
        )
    ) == {
        "model": "m",
        "instructions": None,
        "input": [{"role": "user", "content": "q"}],
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": True,
        "text": {
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "_StructuredReport",
                "schema": _StructuredReport.model_json_schema() | {"additionalProperties": False},
            }
        },
    }


@pytest.mark.parametrize(
    ("supports_cache_breakpoints", "automatic_cache_breakpoints"),
    [(True, False), (True, True), (False, True)],
)
def test_request_cache_breakpoints_follow_capability(
    *, supports_cache_breakpoints: bool, automatic_cache_breakpoints: bool
) -> None:
    bound = _adapter(supports_cache_breakpoints=supports_cache_breakpoints).bind_text(
        _binding(
            automatic_cache_breakpoints=automatic_cache_breakpoints,
            system_prompt=(TextPart(text="", cache_breakpoint=True),),
        )
    )
    parts = (
        TextPart(text="context", cache_breakpoint=True),
        ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
        ImageUrlPart(url=_IMAGE_URL, cache_breakpoint=True),
    )
    params = bound.build_request_params((
        UserMessage(content=parts),
        ToolMessage(tool_call_id="c", content=parts),
    ))
    assert isinstance(params, _OpenAIResponsesRequestParams)
    request_json = json.dumps(params.input)
    assert request_json.count('"prompt_cache_breakpoint"') == (
        7 if supports_cache_breakpoints else 0
    )
    assert ('"prompt_cache_options"' in params.as_json()) == (not automatic_cache_breakpoints)
    assert request_json.count('"context"') == 2


def test_request_system_parts_become_a_developer_input_message() -> None:
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
                {"type": "input_text", "text": "stable instructions"} | _EXPLICIT_BREAKPOINT,
                {"type": "input_text", "text": "semi-stable context"},
            ],
        }
    ]


class _FakeSDKStream(AsyncResponseStream[None]):
    """Skip the SDK initializer because iteration uses constructed events without a connection."""

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
    charged_provider_executed_tools: bool = False,
) -> _OpenAIResponsesStream:
    return _OpenAIResponsesStream(
        sdk_stream=_FakeSDKStream(replay_events, headers),
        pricing=_PRICING,
        charged_provider_executed_tools=charged_provider_executed_tools,
    )


def test_cutoff_openai_provider_executed_tool_billing_is_nan() -> None:
    billing = _stream([], charged_provider_executed_tools=True).provider_billing()
    assert billing is not None
    assert math.isnan(billing.billing.usage.provider_executed_tool_cost_in_usd)
    assert _stream([]).provider_billing() is None


def test_a_stream_reports_the_request_id_header_of_the_response_it_reads() -> None:
    assert _stream([], {"x-request-id": "req_stream"}).request_id() == "req_stream"
    assert _stream([]).request_id() is None


def _text_delta_event(delta: str, sequence_number: int) -> AccumulatedResponseTextDeltaEvent:
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
    return AccumulatedResponseCompletedEvent[None](
        type="response.completed",
        response=ParsedResponse[None].model_validate(response.model_dump()),
        sequence_number=sequence_number,
    )


def _collected_items(replay_events: Sequence[ResponseStreamEvent]) -> list[StreamItem]:
    terminal_event = _completed_event(_response(usage=None), len(replay_events) + 1)

    async def scenario() -> list[StreamItem]:
        return [item async for item in _stream([*replay_events, terminal_event]).items()]

    return run_with_timeout(scenario())


def _summary_delta_event(
    delta: str, summary_index: int, sequence_number: int
) -> ResponseReasoningSummaryTextDeltaEvent:
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
    return ResponseReasoningTextDoneEvent(
        type="response.reasoning_text.done",
        text=text,
        content_index=content_index,
        item_id="r1",
        output_index=0,
        sequence_number=sequence_number,
    )


_SEPARATOR_DELTA = ReasoningDelta(text="\n\n")
"""The delta that precedes a reasoning part after the first, matching ReasoningPart.text."""


@pytest.mark.parametrize(
    ("replay_events", "expected_items"),
    [
        (
            [
                _summary_delta_event("weighing", 0, 1),
                _reasoning_text_delta_event("deciding", 0, 2),
                _text_delta_event("hey", 3),
            ],
            [ReasoningDelta(text="weighing"), ReasoningDelta(text="deciding"), "hey"],
        ),
        (
            [
                _reasoning_text_delta_event("First.", 0, 1),
                _reasoning_text_done_event("First.", 0, 2),
                _reasoning_text_delta_event("Then.", 1, 3),
            ],
            [ReasoningDelta(text="First."), _SEPARATOR_DELTA, ReasoningDelta(text="Then.")],
        ),
        (
            [
                _summary_delta_event("First.", 0, 1),
                _summary_done_event("First.", 0, 2),
                _reasoning_text_delta_event("Then.", 0, 3),
            ],
            [ReasoningDelta(text="First."), _SEPARATOR_DELTA, ReasoningDelta(text="Then.")],
        ),
        (
            [
                _summary_done_event("", 0, 1),
                _summary_delta_event("thought it over", 1, 2),
            ],
            [ReasoningDelta(text="thought it over")],
        ),
        (
            [
                _summary_delta_event("", 0, 1),
                _summary_done_event("", 0, 2),
                _summary_delta_event("thought it over", 1, 3),
            ],
            [ReasoningDelta(text="thought it over")],
        ),
        (
            [
                _summary_delta_event("First.", 0, 1),
                _summary_done_event("First.", 0, 2),
                _summary_delta_event("", 1, 3),
                _summary_delta_event("Then.", 1, 4),
            ],
            [ReasoningDelta(text="First."), _SEPARATOR_DELTA, ReasoningDelta(text="Then.")],
        ),
        (
            [
                _summary_delta_event("thought it over", 0, 1),
                _summary_done_event("thought it over", 0, 2),
                _text_delta_event("hey", 3),
            ],
            [ReasoningDelta(text="thought it over"), "hey"],
        ),
        (
            [
                _summary_delta_event("thought it over", 0, 1),
                _summary_done_event("thought it over", 0, 2),
                _summary_delta_event("", 1, 3),
            ],
            [ReasoningDelta(text="thought it over")],
        ),
    ],
    ids=[
        "both_channels_stream_as_reasoning_deltas_and_text_stays_a_string",
        "the_reasoning_text_channel_separates_its_parts",
        "a_pending_separator_crosses_channels",
        "a_done_event_with_no_delta_before_it",
        "a_part_whose_only_delta_was_empty",
        "an_empty_delta_keeps_the_separator_for_the_next_text",
        "a_last_part_streams_no_trailing_separator",
        "a_trailing_empty_delta_streams_no_trailing_separator",
    ],
)
def test_reasoning_deltas_separate_parts_that_streamed_text(
    replay_events: Sequence[ResponseStreamEvent], expected_items: list[StreamItem]
) -> None:
    assert _collected_items(replay_events) == expected_items


def test_a_summary_part_boundary_streams_the_assembled_reasoning_part_separator() -> None:
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
        streamed = "".join([
            item.text async for item in adapter_stream.items() if isinstance(item, ReasoningDelta)
        ])
        return streamed, _assistant_message_from(await adapter_stream.final())

    streamed, assistant_message = run_with_timeout(scenario())
    assert isinstance(assistant_message.content, tuple)
    reasoning_part = assistant_message.content[0]
    assert reasoning_part.kind == "reasoning"
    assert streamed == reasoning_part.text == "First, water evaporates.\n\nThen it condenses."


def test_stream_yields_argument_fragments_then_one_complete_tool_call() -> None:
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
    ])
    assert translated == [
        ToolCallDelta(id="call1", name="lookup", partial_args_json='{"q"'),
        ToolCallDelta(id="call1", name="lookup", partial_args_json=": 1}"),
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    ]


def test_stream_failed_terminal_is_the_final_response() -> None:
    failed_response = _response(usage=_usage_with_cache(), status="failed", error=_SERVER_ERROR)

    async def scenario() -> None:
        adapter_stream = _stream([
            ResponseFailedEvent(
                type="response.failed", response=failed_response, sequence_number=1
            )
        ])
        assert [item async for item in adapter_stream.items()] == []
        assert await adapter_stream.final() is failed_response

    run_with_timeout(scenario())


def test_stream_final_passes_a_leniently_built_terminal_through_unvalidated() -> None:
    """Final preserves a leniently constructed incomplete terminal response by identity."""
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
        usable = _assert_usable(_text_bound().interpret(leniently_built))
        assert usable.output == "hey"
        assert usable.stop_reason == "max_completion_tokens"

    run_with_timeout(scenario())


def test_final_before_items_are_exhausted_raises() -> None:
    async def scenario() -> None:
        adapter_stream = _stream([_completed_event(_response(usage=None), 1)])
        with pytest.raises(StreamProtocolError):
            await adapter_stream.final()

    run_with_timeout(scenario())


def test_stream_error_event_raises_a_status_error_carrying_the_events_fields() -> None:
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
        assert openai_request_failure(caught.value) == transient(pauses_quota=False)

    run_with_timeout(scenario())


def _structured_bound() -> _BoundOpenAIResponsesStructured[_StructuredReport]:
    adapter = _adapter()
    precomputed_fields = adapter._precompute_fields(
        _binding(automatic_cache_breakpoints=False, system_prompt="sys")
    )
    return _BoundOpenAIResponsesStructured(
        adapter=adapter, precomputed_fields=precomputed_fields, response_format=_StructuredReport
    )


def _structured_parse(response: OpenAIResponse) -> ResponseOutcome[_StructuredReport | None]:
    return _structured_bound()._parsed_outcome(response, _assistant_message_from(response))


class _CapturedAnswer(BaseModel):
    value: int


@pytest.mark.parametrize("name", ["final_commentary_no_tools", "final_provisional_no_tools"])
def test_structured_binding_selects_the_final_answer_in_live_luna_captures(name: str) -> None:
    """Earlier commentary and a schema-valid provisional answer precede the final answer."""
    path = Path(__file__).parent / "fixtures" / "responses_structured" / f"{name}.json"
    response = OpenAIResponse.model_validate_json(path.read_text())
    adapter = _adapter()
    bound = _BoundOpenAIResponsesStructured(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(automatic_cache_breakpoints=False)),
        response_format=_CapturedAnswer,
    )
    billing = bound.billing_from_raw(response)
    outcome = _assert_usable(bound.interpret(response))
    assert outcome.output == _CapturedAnswer(value=17)
    assert outcome.assistant_message == _assistant_message_from(response)
    assert billing.usage_raw is response.usage

    expected: list[object] = []
    for item in response.output:
        assert item.type == "message"
        assert len(item.content) == 1
        block = item.content[0]
        assert block.type == "output_text"
        expected.append({"role": "assistant", "content": block.text, "phase": item.phase})
    assert _assistant_items(outcome.assistant_message) == expected

    async def streamed() -> OpenAIResponse:
        stream = _stream([_completed_event(response, 1)])
        _ = [item async for item in stream.items()]
        return await stream.final()

    assert bound.interpret(run_with_timeout(streamed())) == outcome


@pytest.mark.parametrize(
    ("earlier", "final", "expected_kind"),
    [
        ("Checking the temperature", '{"city":"Lima","celsius":19}', "usable_response"),
        ('{"city":"Nairobi","celsius":25}', '{"city":"Lima","celsius":19}', "usable_response"),
        ('{"city":"Nairobi","celsius":25}', "Invalid final answer", "schema_violation"),
        ('{"city":"Nairobi","celsius":25}', "", "schema_violation"),
    ],
)
def test_structured_binding_validates_the_final_complete_block_without_a_phase(
    earlier: str, final: str, expected_kind: str
) -> None:
    message = ResponseOutputMessage(
        id="message",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(type="output_text", text=text, annotations=[])
            for text in (earlier, final)
        ],
    )
    response = _response(usage=_usage_with_cache(), output=[message])
    outcome = _structured_parse(response)
    assert outcome.kind == expected_kind
    assert outcome.assistant_message == _assistant_message_from(response)
    if outcome.kind == "usable_response":
        assert outcome.output == _StructuredReport(city="Lima", celsius=19)


_REPORT_JSON = '{"city": "Nairobi", "celsius": 25}'
"""Text that validates into _StructuredReport."""


def _structured_response(
    text: str | None,
    *,
    refusal: bool = False,
    status: ResponseStatus = "completed",
    incomplete_details: IncompleteDetails | None = None,
    tool_call: bool = False,
    error: ResponseError | None = None,
) -> OpenAIResponse:
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
        usage=None,
        output=[message, *([_FUNCTION_CALL_OUTPUT_ITEM] if tool_call else [])],
        status=status,
        incomplete_details=incomplete_details,
        error=error,
    )


def test_structured_bind_preserves_schema_valid_commentary_beside_a_tool_call() -> None:
    outcome = _assert_usable(
        _structured_bound().interpret(_structured_response(_REPORT_JSON, tool_call=True))
    )
    assert outcome.output is None
    assert outcome.assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )


@pytest.mark.parametrize("refusal", ["I can't help with that", _REPORT_JSON])
def test_final_refusal_cannot_use_earlier_structured_text(refusal: str) -> None:
    response = _response(
        usage=None,
        output=[
            *_structured_response(_REPORT_JSON).output,
            {**_REFUSAL_MESSAGE_ITEM, "content": [{"type": "refusal", "refusal": refusal}]},
        ],
    )
    outcome = _structured_bound().interpret(response)
    assert outcome.kind == "refusal"
    assert outcome.assistant_message.content == (
        TextPart(text=_REPORT_JSON),
        TextPart(text=refusal),
    )


@pytest.mark.parametrize("text", [None, "let me look that up"])
def test_structured_bind_reports_an_assistant_message_with_tool_calls_as_none(
    text: str | None,
) -> None:
    outcome = _structured_parse(_structured_response(text, tool_call=True))
    assert _assert_usable(outcome).output is None


@pytest.mark.parametrize(
    ("response", "expected_kind"),
    [
        (_structured_response(None), "empty_assistant_message"),
        (
            _structured_response(
                '{"city": "Nair',
                status="incomplete",
                incomplete_details=IncompleteDetails(reason="max_output_tokens"),
            ),
            "max_completion_tokens_exceeded",
        ),
        (_structured_response(None, refusal=True), "refusal"),
        (
            _structured_response(
                None,
                status="incomplete",
                incomplete_details=IncompleteDetails(reason="content_filter"),
            ),
            "refusal",
        ),
        (
            _structured_response(_REPORT_JSON, status="failed", error=_SERVER_ERROR),
            "provider_failed_transiently",
        ),
        (
            _structured_response(None, refusal=True, status="failed", error=_SERVER_ERROR),
            "provider_failed_transiently",
        ),
    ],
    ids=[
        "completed_without_text_is_an_empty_assistant_message",
        "json_cut_at_max_output_tokens_is_truncation",
        "refusal_block",
        "content_filter_blocks_without_retry",
        "failed_status_wins_over_text_that_validates",
        "failed_status_wins_over_a_refusal",
    ],
)
def test_structured_bind_reports_why_an_assistant_message_has_no_instance(
    response: OpenAIResponse, expected_kind: str
) -> None:
    assert _structured_parse(response).kind == expected_kind


def test_a_cancelled_run_is_an_unfinished_assistant_message_naming_the_status() -> None:
    outcome = _structured_parse(_structured_response(None, status="cancelled"))
    assert outcome.kind == "unfinished_assistant_message"
    assert outcome.error_text == "openai returned status 'cancelled'"


def _text_bound() -> _BoundOpenAIResponsesText:
    adapter = _adapter()
    return _BoundOpenAIResponsesText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(automatic_cache_breakpoints=False)),
    )


def test_identity_reads_the_responses_own_id_and_served_model() -> None:
    identity = _text_bound().identity_from_raw(
        _response(usage=None, model="m-2026-01-01"), request_id="req_openai"
    )
    assert identity == ResponseIdentity(
        model_served="m-2026-01-01", response_id="r1", request_id="req_openai"
    )


@pytest.mark.parametrize(
    ("error", "expected_kind", "expected_reason"),
    [
        (_SERVER_ERROR, "provider_failed_transiently", "The server had an error."),
        (
            ResponseError(
                code="failed_to_download_image", message="Failed to download image from the URL."
            ),
            "provider_failed_terminally",
            "Failed to download image from the URL.",
        ),
        (
            ResponseError.model_construct(
                code="misalignment_policy_violation", message="The response violated policy."
            ),
            "provider_failed_terminally",
            "The response violated policy.",
        ),
        (
            ResponseError.construct(code="a_code_from_a_later_sdk", message="Something."),
            "provider_failed_terminally",
            "Something.",
        ),
        (
            None,
            "provider_failed_terminally",
            "openai reported status 'failed' and no error object",
        ),
    ],
    ids=[
        "transient_code",
        "terminal_code",
        "terminal_code_the_sdk_literal_omits",
        "code_the_installed_sdk_does_not_name",
        "no_error_object",
    ],
)
def test_a_failed_run_takes_its_variant_from_the_error_code(
    error: ResponseError | None, expected_kind: str, expected_reason: str
) -> None:
    outcome = _text_bound().interpret(_response(usage=None, status="failed", error=error))
    assert outcome.kind == expected_kind
    assert isinstance(outcome, ProviderFailedTransiently | ProviderFailedTerminally)
    assert outcome.error_text == expected_reason
    assert outcome.assistant_message.content == (TextPart(text="hey"),)


@pytest.mark.parametrize(
    ("error", "expected_pauses_quota"),
    [
        (_SERVER_ERROR, False),
        (ResponseError(code="rate_limit_exceeded", message="Rate limit reached."), True),
    ],
)
def test_only_a_rate_limit_error_code_pauses_the_quota(
    error: ResponseError, *, expected_pauses_quota: bool
) -> None:
    outcome = _text_bound().interpret(_response(usage=None, status="failed", error=error))
    assert outcome.kind == "provider_failed_transiently"
    assert outcome.pauses_quota is expected_pauses_quota


@pytest.mark.parametrize(
    ("failure", "expected_request_failure", "fallthrough_tag"),
    [
        (
            status_error(openai.RateLimitError, 429, {"Retry-After-MS": "1500"}),
            transient(pauses_quota=True, retry_after_seconds=1.5),
            None,
        ),
        (status_error(openai.RateLimitError, 429), transient(pauses_quota=True), None),
        (
            status_error(openai.BadRequestError, 400, {"retry-after": "7"}),
            terminal("rejected"),
            None,
        ),
        (
            status_error(openai.RateLimitError, 429, error_code="credit_balance_exhausted"),
            terminal("rejected"),
            None,
        ),
        (
            status_error(openai.RateLimitError, 429, error_code="rate_limit_exceeded"),
            transient(pauses_quota=True),
            None,
        ),
        (
            status_error(openai.InternalServerError, 500, {"x-should-retry": "false"}),
            terminal("provider_failed_terminally"),
            None,
        ),
        (
            status_error(
                openai.BadRequestError, 400, {"x-should-retry": "true", "retry-after": "3"}
            ),
            transient(pauses_quota=False, retry_after_seconds=3.0),
            None,
        ),
        (
            status_error(
                openai.RateLimitError, 429, {"x-should-retry": "false", "retry-after": "7"}
            ),
            RequestFailure(
                kind="provider_failed_terminally", pauses_quota=True, retry_after_seconds=7.0
            ),
            None,
        ),
        (
            status_error(
                openai.RateLimitError, 429, {"x-should-retry": "false"}, "credit_balance_exhausted"
            ),
            terminal("rejected"),
            None,
        ),
        (
            status_error(
                openai.RateLimitError, 429, {"x-should-retry": "true"}, "credit_balance_exhausted"
            ),
            transient(pauses_quota=False),
            None,
        ),
        (
            status_error(
                openai.APIStatusError, 200, {"x-should-retry": "false"}, "rate_limit_exceeded"
            ),
            transient(pauses_quota=True),
            None,
        ),
        (
            status_error(openai.APIStatusError, 200, {"x-should-retry": "true"}, "invalid_prompt"),
            terminal("provider_failed_terminally"),
            None,
        ),
        (
            status_error(openai.APIStatusError, 200, error_code="server_error"),
            transient(pauses_quota=False),
            None,
        ),
        (
            status_error(openai.APIStatusError, 200, error_code="misalignment_policy_violation"),
            terminal("provider_failed_terminally"),
            None,
        ),
        (
            status_error(openai.APIStatusError, 599),
            transient(pauses_quota=False),
            "status=599 type=None",
        ),
        (
            status_error(openai.APIStatusError, 200, error_code="brand_new_code"),
            terminal("provider_failed_terminally"),
            "status=200 type=brand_new_code",
        ),
        (
            status_error(openai.APIStatusError, 200),
            terminal("provider_failed_terminally"),
            "status=200 type=None",
        ),
    ],
    ids=[
        "mixed_case_retry_after_ms",
        "rate_limit_without_a_stated_wait",
        "retry_after_does_not_pick_the_kind",
        "spend_limit_429",
        "throttled_429",
        "false_directive_stops_a_retried_status",
        "true_directive_retries_a_stopped_status",
        "false_directive_on_a_rate_limit_pauses_and_stops",
        "false_directive_on_a_spend_limit",
        "true_directive_on_a_spend_limit",
        "status_200_ignores_a_false_directive",
        "status_200_ignores_a_true_directive",
        "status_200_transient_code",
        "status_200_terminal_code",
        "unlisted_5xx_falls_through",
        "status_200_unknown_code_falls_through",
        "status_200_without_a_code_falls_through",
    ],
)
def test_openai_request_failure_and_counts_only_fallthroughs(
    failure: openai.APIStatusError,
    expected_request_failure: RequestFailure,
    fallthrough_tag: str | None,
) -> None:
    before = REQUEST_FAILURE_FALLTHROUGH_COUNTS.copy()
    assert openai_request_failure(failure) == expected_request_failure
    if fallthrough_tag is not None:
        before[fallthrough_tag] += 1
    assert before == REQUEST_FAILURE_FALLTHROUGH_COUNTS


def test_request_id_from_error_reads_the_sdk_errors_own_header_and_nothing_else() -> None:
    adapter = _adapter()
    rate_limited = status_error(openai.RateLimitError, 429, {"x-request-id": "req_429"})
    assert adapter.request_id_from_error(rate_limited) == "req_429"
    assert adapter.request_id_from_error(status_error(openai.RateLimitError, 429)) is None
    assert adapter.request_id_from_error(ValueError("boom")) is None


def test_adapter_pins_sdk_retries_off() -> None:
    assert _adapter().client.max_retries == 0


def _conformance_output() -> list[object]:
    return [
        _REASONING_OUTPUT_ITEM | {"field_newer_than_sdk": "x"},
        dict(_WEB_SEARCH_OUTPUT_ITEM),
        dict(_TEXT_OUTPUT_ITEM),
    ]


class TestOpenAIResponsesConformance(AdapterConformance):
    @override
    def make_adapter(self) -> Adapter:
        return _adapter()

    @override
    def response_with_cache_writes(self) -> BaseModel:
        return _response(usage=_usage_with_cache(), output=_conformance_output())

    @override
    def response_without_usage(self) -> BaseModel:
        return _response(usage=None, output=[dict(_TEXT_OUTPUT_ITEM)])

    @override
    def response_at_an_unpriced_tier(self) -> BaseModel:
        return _response(
            usage=_usage_with_cache(), output=_conformance_output(), service_tier="flex"
        )

    @override
    def response_with_impossible_counters(self) -> BaseModel:
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
    def response_with_text(self, text: str) -> BaseModel:
        return _structured_response(text)

    @override
    def response_with_text_parts(self, texts: Sequence[str]) -> BaseModel:
        return _response(
            usage=None,
            output=[
                ResponseOutputMessage(
                    id=f"message_{index}",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
                )
                for index, text in enumerate(texts)
            ],
        )

    @override
    def response_with_tool_call(self, text: str) -> BaseModel:
        return _structured_response(text, tool_call=True)

    @override
    def response_with_reasoning(self) -> BaseModel:
        return _response(usage=_usage_with_cache(), output=_conformance_output())

    @override
    def response_with_raw_part(self) -> BaseModel | None:
        return _response(usage=_usage_with_cache(), output=_conformance_output())

    @override
    def assistant_wire_parts(self, request_params: RequestParams) -> Sequence[object]:
        assert isinstance(request_params, _OpenAIResponsesRequestParams)
        return request_params.input[1:]

    @override
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        whole = _response(usage=_usage_with_cache(), output=_conformance_output())
        return ParsedResponse[None].model_validate(whole.model_dump()), whole

    @override
    def stream_without_its_terminal_event(self) -> AdapterStream:
        return _stream([_text_delta_event("he", 1)])

    @override
    def sdk_errors_and_request_failures(self) -> Mapping[Exception, RequestFailure]:
        return openai_sdk_errors_and_request_failures()
