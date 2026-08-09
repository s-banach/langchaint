"""OpenAI Chat Completions adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot:
the usage partition derived by subtracting cache counters from prompt_tokens,
the pluggable cache-read counter both readers exercise,
the one-assistant-param turn round trip with reasoning_content merged beside content,
finish_reason normalization, the runtime-None finish_reason a lenient snapshot admits,
stream event translation with the SDK's own state, the mid-stream bare-APIError rewrap,
and the request fields the binding precomputes.
"""

import asyncio
import base64
import json
import math
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import override

import httpx
import openai
import pytest
from openai import AsyncOpenAI, AsyncStream
from openai._models import construct_type_unchecked
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from langchaint import (
    AssistantMessage,
    AudioPart,
    ContentPart,
    ImagePart,
    ImageUrlPart,
    InferenceParams,
    RawPart,
    ReasoningDelta,
    ReasoningPart,
    SpecificToolChoice,
    StopReason,
    StreamItem,
    TextPart,
    ToolCall,
    ToolCallDelta,
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
    NoOutputOutcome,
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.conformance import AdapterConformance
from langchaint.deepseek import cache_read_tokens_from_usage_deepseek
from langchaint.exceptions import StreamProtocolError
from langchaint.openai import (
    OpenAIChatCompletionsAdapter,
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
)
from langchaint.openai.chat_completions_adapter import (
    _assistant_message_from,
    _assistant_message_param,
    _billing_from_chat_completion,
    _BoundChatCompletions,
    _BoundChatCompletionsStructured,
    _BoundChatCompletionsText,
    _ChatCompletionsRequestParams,
    _ChatCompletionsStream,
    _finished_turn_or_unfinished,
    _FinishedTurn,
    _normalized_stop_reason,
    _wire_messages,
    _wire_tool_choice,
    cache_read_tokens_from_usage_openai,
)
from langchaint.pricing import Billing
from langchaint.shared_backoff import RetryThisOne, Verdict
from langchaint.tools import ToolSchema
from tests.helpers import openai_sdk_errors_and_classifications, openai_sdk_errors_and_verdicts

_DEFAULT_RATES = OpenAIPricingTable(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_PRICING: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {"default": _DEFAULT_RATES}
"""The default tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = OpenAIPricingTable(
    input_cache_none_usd_per_million_tokens=5.0,
    output_usd_per_million_tokens=20.0,
    cache_read_usd_per_million_tokens=2.5,
    cache_write_usd_per_million_tokens=6.25,
    web_search_usd_per_invocation=0.02,
    file_search_usd_per_invocation=0.005,
)
"""Twice the default rates, so a tier-selection test reads as a doubling."""

_TOOL_CALL_WIRE: dict[str, object] = {
    "id": "call1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"q": 1}'},
}
"""One function tool call as the API returns it on an assistant message."""

_CUSTOM_TOOL_CALL_WIRE: dict[str, object] = {
    "id": "c9",
    "type": "custom",
    "custom": {"name": "n", "input": "i"},
}
"""One custom tool call as the API returns it."""

_FUNCTION_CALL_WIRE: dict[str, object] = {
    "name": "legacy_lookup",
    "arguments": "{}",
}
"""One deprecated function_call as the API returns it."""


def _assert_result[OutputT](outcome: ResponseOutcome[OutputT]) -> AdapterResult[OutputT]:
    """Narrow a ResponseOutcome to its success variant, failing the test on any other variant."""
    assert isinstance(outcome, AdapterResult)
    return outcome


def _usage_with_cache() -> CompletionUsage:
    """Return usage whose prompt_tokens includes both cache counters."""
    return CompletionUsage.model_validate({
        "prompt_tokens": 1000,
        "completion_tokens": 40,
        "total_tokens": 1040,
        "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 0},
    })


def _completion(
    *,
    usage: CompletionUsage | None,
    message: Mapping[str, object] | None = None,
    finish_reason: str = "stop",
    service_tier: str | None = None,
    choices: list[object] | None = None,
    model: str = "m",
) -> ChatCompletion:
    """Build a completion whose id is fixed at "r1"; every field a test varies is a parameter.

    message holds the assistant message's fields minus role; None gives content "hey".
    choices overrides the single built choice, [] being the no-choices response.
    """
    if choices is None:
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", **(message or {"content": "hey"})},
                "finish_reason": finish_reason,
            }
        ]
    return ChatCompletion.model_validate({
        "id": "r1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": choices,
        "usage": usage,
        "service_tier": service_tier,
    })


def _lenient_completion(finish_reason: str | None) -> ChatCompletion:
    """Build the runtime shape strict validation rejects: a finish_reason outside the SDK's Literal.

    The SDK's stream state constructs its snapshot leniently, so None and unknown values are
    runtime states interpret meets.
    """
    completion = construct_type_unchecked(
        value={
            "id": "r1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hey"},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": None,
        },
        type_=ChatCompletion,
    )
    assert isinstance(completion, ChatCompletion)
    return completion


def _billing(
    completion: ChatCompletion,
    pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable] = _PRICING,
) -> Billing:
    """Price one completion with the openai cache-read reader, the adapter's default."""
    return _billing_from_chat_completion(
        completion,
        pricing=pricing,
        cache_read_tokens_from_usage=cache_read_tokens_from_usage_openai,
    )


def test_billing_subtracts_cache_from_prompt_tokens_and_prices() -> None:
    """The uncached counter is prompt_tokens minus both cache counters, and the priced cost rides on it."""
    usage = _billing(_completion(usage=_usage_with_cache())).usage
    assert usage.input_tokens_cache_read == 600
    assert usage.input_tokens_cache_write == 100
    assert usage.input_tokens_cache_none == 300
    assert usage.input_tokens_total == 1000
    assert usage.cost_in_usd == pytest.approx(
        (300 * 2.5 + 600 * 1.25 + 100 * 3.125 + 40 * 10.0) / 1e6
    )


def test_billing_carries_the_sdk_usage_object_itself() -> None:
    """usage_raw is the completion's own CompletionUsage by reference, and None where it reported none."""
    raw = _completion(usage=_usage_with_cache())
    assert _billing(raw).usage_raw is raw.usage
    assert _billing(_completion(usage=None)).usage_raw is None


def test_search_annotations_produce_unknown_provider_executed_tool_cost() -> None:
    """Search annotations lack the invocation count required for exact billing."""
    message: dict[str, object] = {
        "content": "source",
        "annotations": [
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 0,
                    "end_index": 6,
                    "title": "Source",
                    "url": "https://example.com",
                },
            }
        ],
    }
    usage = _billing(_completion(usage=_usage_with_cache(), message=message)).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


def test_billing_reads_reasoning_tokens() -> None:
    """output_tokens_reasoning reads completion_tokens_details.reasoning_tokens."""
    usage = _billing(
        _completion(
            usage=CompletionUsage.model_validate({
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 8},
            })
        )
    ).usage
    assert usage.output_tokens_reasoning == 8


def test_billing_reads_zero_cache_counters_where_the_details_objects_are_absent() -> None:
    """Bare required counters partition as all-uncached input and no reasoning output."""
    usage = _billing(
        _completion(
            usage=CompletionUsage.model_validate({
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            })
        )
    ).usage
    assert usage.input_tokens_cache_read == 0
    assert usage.input_tokens_cache_write == 0
    assert usage.input_tokens_cache_none == 10
    assert usage.output_tokens_reasoning == 0


def test_billing_without_usage_pins_the_priced_tiers_rates() -> None:
    """A completion missing usage still stores the rates the tier that served it would have spent."""
    billing = _billing(_completion(usage=None))
    assert billing.output_usd_per_million_tokens == 10.0
    assert billing.service_tier == "default"


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response; "auto" and no tier both price at default."""
    pricing: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
        "default": _DEFAULT_RATES,
        "priority": _PRIORITY_RATES,
    }
    at_priority = _billing(
        _completion(usage=_usage_with_cache(), service_tier="priority"), pricing
    ).usage
    at_default = _billing(
        _completion(usage=_usage_with_cache(), service_tier="default"), pricing
    ).usage
    reporting_auto = _billing(
        _completion(usage=_usage_with_cache(), service_tier="auto"), pricing
    ).usage
    reporting_none = _billing(_completion(usage=_usage_with_cache()), pricing).usage
    assert at_priority.cost_in_usd == pytest.approx(2 * at_default.cost_in_usd)
    assert reporting_auto.cost_in_usd == at_default.cost_in_usd
    assert reporting_none.cost_in_usd == at_default.cost_in_usd


def _deepseek_usage() -> CompletionUsage:
    """Return usage as DeepSeek reports it: the cache partition in extra fields, no details objects."""
    return CompletionUsage.model_validate({
        "prompt_tokens": 1000,
        "completion_tokens": 40,
        "total_tokens": 1040,
        "prompt_cache_hit_tokens": 600,
        "prompt_cache_miss_tokens": 400,
    })


def test_the_deepseek_reader_prices_cache_hits_as_cache_reads() -> None:
    """prompt_cache_hit_tokens becomes the cache-read counter and the miss remainder stays uncached.

    Through the openai reader the same usage reads as all-uncached, which is the 50x over-report
    cache_read_tokens_from_usage exists to prevent.
    """
    billing = _billing_from_chat_completion(
        _completion(usage=_deepseek_usage()),
        pricing=_PRICING,
        cache_read_tokens_from_usage=cache_read_tokens_from_usage_deepseek,
    )
    assert billing.usage.input_tokens_cache_read == 600
    assert billing.usage.input_tokens_cache_write == 0
    assert billing.usage.input_tokens_cache_none == 400
    assert _billing(_completion(usage=_deepseek_usage())).usage.input_tokens_cache_read == 0


def test_the_deepseek_reader_returns_zero_without_the_extra_field() -> None:
    """The openai usage shape carries no prompt_cache_hit_tokens, so the reader reads zero."""
    assert cache_read_tokens_from_usage_deepseek(_usage_with_cache()) == 0


def test_pricing_without_the_default_key_raises_at_construction() -> None:
    """A pricing mapping missing "default" fails before any request, naming the model."""
    priority_only: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
        "priority": _PRIORITY_RATES
    }
    with pytest.raises(ValueError, match=re.escape("'default'")):
        _ = OpenAIChatCompletionsAdapter(
            client=AsyncOpenAI(api_key="test"),
            model="deepseek-v4-flash",
            pricing=priority_only,
            provider_name="deepseek",
            supports_prompt_cache_options=False,
        )


def _finished(completion: ChatCompletion) -> _FinishedTurn:
    """Read the completion's first choice as a finished turn, failing the test where none exists."""
    finished_turn = _finished_turn_or_unfinished(completion)
    assert isinstance(finished_turn, _FinishedTurn)
    return finished_turn


@pytest.mark.parametrize(
    ("build_completion", "expected"),
    [
        (lambda: _completion(usage=None), "end_turn"),
        (
            lambda: _completion(
                usage=None, message={"content": "on it", "tool_calls": [_TOOL_CALL_WIRE]}
            ),
            "tool_use",
        ),
        (
            lambda: _completion(
                usage=None,
                message={"tool_calls": [_TOOL_CALL_WIRE]},
                finish_reason="tool_calls",
            ),
            "tool_use",
        ),
        (lambda: _completion(usage=None, finish_reason="length"), "max_tokens"),
        (lambda: _completion(usage=None, finish_reason="content_filter"), "refusal"),
        (
            lambda: _completion(usage=None, message={"refusal": "I can't help with that"}),
            "refusal",
        ),
        (lambda: _completion(usage=None, finish_reason="function_call"), "other"),
        (lambda: _lenient_completion("weird"), "other"),
    ],
    ids=[
        "stop",
        "stop_with_tool_calls",
        "tool_calls",
        "length",
        "content_filter",
        "refusal_field_beside_stop",
        "function_call",
        "unknown_value",
    ],
)
def test_stop_reason_mapping(
    build_completion: Callable[[], ChatCompletion], expected: StopReason
) -> None:
    """Each finish_reason row maps as the module docstring states.

    The refusal field is tested ahead of the rows, which the refusal-beside-stop row pins.
    """
    assert _normalized_stop_reason(_finished(build_completion())) == expected


def test_the_turn_orders_reasoning_then_text_then_refusal_then_tool_calls() -> None:
    """One message decomposes into the turn, reasoning_content read off the extra fields."""
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": "hey",
        "refusal": "but no more",
        "tool_calls": [_TOOL_CALL_WIRE],
        "reasoning_content": "thought it over",
    })
    assert _assistant_message_from(message).turn == (
        ReasoningPart(raw={"reasoning_content": "thought it over"}, text="thought it over"),
        TextPart(text="hey"),
        TextPart(text="but no more"),
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )


def test_a_custom_tool_call_becomes_a_raw_part_and_replays_in_order() -> None:
    """The custom.input and ToolCall.args_json fields name different provider concepts."""
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": None,
        "tool_calls": [_CUSTOM_TOOL_CALL_WIRE, _TOOL_CALL_WIRE],
    })
    assistant_message = _assistant_message_from(message)
    assert assistant_message.turn == (
        RawPart(raw=_CUSTOM_TOOL_CALL_WIRE),
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "tool_calls": [_CUSTOM_TOOL_CALL_WIRE, _TOOL_CALL_WIRE],
    }


def test_a_function_call_becomes_a_raw_part_and_replays_unchanged() -> None:
    """The function_call field lacks the id ToolCall requires."""
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "function_call": _FUNCTION_CALL_WIRE,
    })
    assistant_message = _assistant_message_from(message)
    assert assistant_message.turn == (RawPart(raw={"function_call": _FUNCTION_CALL_WIRE}),)
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "function_call": _FUNCTION_CALL_WIRE,
    }


@pytest.mark.parametrize("extra_type", ["custom", "future"])
def test_a_function_call_keeps_its_field_when_an_extra_type_is_present(extra_type: str) -> None:
    """The function_call field identifies the replay position independently from its value."""
    function_call = {**_FUNCTION_CALL_WIRE, "type": extra_type}
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "function_call": function_call,
    })
    assistant_message = _assistant_message_from(message)
    assert assistant_message.turn == (RawPart(raw={"function_call": function_call}),)
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "function_call": function_call,
    }


def test_build_request_rejects_two_deprecated_function_calls() -> None:
    """One assistant message has one function_call field, so a second value cannot fit."""
    bound = _adapter().bind_text(_binding())
    request = bound.build_request([
        AssistantMessage(
            turn=(
                RawPart(raw={"function_call": _FUNCTION_CALL_WIRE}),
                RawPart(raw={"function_call": {"name": "second_lookup", "arguments": "{}"}}),
            )
        )
    ])
    assert isinstance(request, InvalidRequest)
    assert "more than one function_call" in request.reason


def test_assistant_message_carries_the_refusal_text_and_replays_it() -> None:
    """The refusal becomes a TextPart, so the refused turn replays as the model wrote it.

    Dropped instead, the turn holds no TurnPart values and sends nothing back, which reopens the
    Sequence[Message] at the point the model declined.
    """
    assistant_message = _assistant_message_from(
        ChatCompletionMessage.model_validate({"role": "assistant", "refusal": "I can't help"})
    )
    assert assistant_message.turn == (TextPart(text="I can't help"),)
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "content": "I can't help",
    }


def test_the_turn_replays_as_one_assistant_param_with_reasoning_merged_beside_content() -> None:
    """Texts join into content. ToolCall values and ReasoningPart.raw keep their fields."""
    assistant_message = AssistantMessage(
        turn=(
            ReasoningPart(raw={"reasoning_content": "thought it over"}, text="thought it over"),
            TextPart(text="he"),
            TextPart(text="y"),
            ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
        )
    )
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "reasoning_content": "thought it over",
        "content": "hey",
        "tool_calls": [_TOOL_CALL_WIRE],
    }


def test_foreign_reasoning_merges_its_keys_into_the_param_unchanged() -> None:
    """A foreign ReasoningPart sends ReasoningPart.raw unchanged for provider validation."""
    raw: dict[str, object] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(turn=(ReasoningPart(raw=raw), TextPart(text="hi")))
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "content": "hi",
        **raw,
    }


def test_wire_messages_converts_each_message_kind() -> None:
    """User, assistant, and tool messages each map to their message param."""
    wire = _wire_messages([
        UserMessage(content="q"),
        AssistantMessage(
            turn=(
                TextPart(text="thinking"),
                ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
            ),
        ),
        ToolMessage(tool_call_id="call1", content="r"),
    ])
    assert wire == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking", "tool_calls": [_TOOL_CALL_WIRE]},
        {"role": "tool", "tool_call_id": "call1", "content": "r"},
    ]


def test_wire_messages_marks_marked_user_and_tool_parts() -> None:
    """A marked part carries prompt_cache_breakpoint on its wire part; unmarked siblings carry none."""
    wire = _wire_messages([
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
                TextPart(text="question"),
            )
        ),
        ToolMessage(
            tool_call_id="c1",
            content=(TextPart(text="saw", cache_breakpoint=True), TextPart(text="more")),
        ),
    ])
    assert wire == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "shared context",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}",
                        "detail": "auto",
                    },
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "text", "text": "question"},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "saw", "prompt_cache_breakpoint": {"mode": "explicit"}},
                {"type": "text", "text": "more"},
            ],
        },
    ]


def test_wire_messages_maps_image_url_part_and_audio_part_in_user_message() -> None:
    """UserMessage maps ImageUrlPart and supported AudioPart.media_type values."""
    wire = _wire_messages([
        UserMessage(
            content=(
                ImageUrlPart(url="https://example.com/image.png", cache_breakpoint=True),
                AudioPart(data=b"wav", media_type="audio/wav", cache_breakpoint=True),
                AudioPart(data=b"mp3", media_type="audio/mpeg"),
            )
        )
    ])
    assert wire == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png",
                        "detail": "auto",
                    },
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(b"wav").decode("ascii"),
                        "format": "wav",
                    },
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(b"mp3").decode("ascii"),
                        "format": "mp3",
                    },
                },
            ],
        }
    ]


def test_build_request_rejects_audio_part_media_type_without_input_audio_format() -> None:
    """AudioPart.media_type without an input_audio.format mapping returns InvalidRequest."""
    request = (
        _adapter()
        .bind_text(_binding())
        .build_request([UserMessage(content=(AudioPart(data=b"audio", media_type="audio/ogg"),))])
    )
    assert isinstance(request, InvalidRequest)
    assert "AudioPart" in request.reason
    assert "UserMessage" in request.reason
    assert "audio/ogg" in request.reason


def test_wire_tool_choice_passes_strings_through_and_names_specific_tools() -> None:
    """The neutral strings pass through unchanged; SpecificToolChoice becomes the function form."""
    assert _wire_tool_choice("auto") == "auto"
    assert _wire_tool_choice("required") == "required"
    assert _wire_tool_choice("none") == "none"
    assert _wire_tool_choice(SpecificToolChoice(tool_name="x")) == {
        "type": "function",
        "function": {"name": "x"},
    }


def _adapter(*, supports_prompt_cache_options: bool = True) -> OpenAIChatCompletionsAdapter:
    """Build an adapter over a keyless client, valid because no request is sent.

    supports_prompt_cache_options defaults True, the gpt-5.6-and-later case, so every caller
    that does not name it exercises the path where a binding's caching value reaches the wire.
    """
    return OpenAIChatCompletionsAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=supports_prompt_cache_options,
    )


def _binding(
    *,
    automatic_prompt_caching: bool = True,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    tool_schemas: tuple[ToolSchema, ...] = (),
    provider_executed_tools: tuple[Mapping[str, object], ...] = (),
    inference_params: InferenceParams | None = None,
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    """Assemble a binding with the fields these request tests vary."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        provider_executed_tools=provider_executed_tools,
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=inference_params if inference_params is not None else InferenceParams(),
        automatic_prompt_caching=automatic_prompt_caching,
        extra_body=extra_body,
    )


def test_provider_executed_tools_raise_with_the_responses_interface() -> None:
    """Chat Completions directs provider-executed tools to Responses."""
    with pytest.raises(ValueError, match="OpenAIResponsesAdapter"):
        _ = _adapter().bind_text(_binding(provider_executed_tools=({"type": "web_search"},)))


def test_web_search_options_raise_with_the_responses_interface() -> None:
    """Chat Completions rejects web search because its billing evidence is incomplete."""
    with pytest.raises(ValueError, match="OpenAIResponsesAdapter"):
        _ = _adapter().bind_text(_binding(extra_body={"web_search_options": {}}))


def test_request_str_system_becomes_one_system_message_first() -> None:
    """A str system_prompt is the messages_prefix's one system-role message."""
    precomputed_fields = _adapter()._precompute_fields(_binding(system_prompt="sys"))
    assert precomputed_fields.messages_prefix == [{"role": "system", "content": "sys"}]


def test_request_system_parts_become_one_system_message_of_marked_parts() -> None:
    """A parts system_prompt travels as one system message whose marked parts carry breakpoints."""
    precomputed_fields = _adapter()._precompute_fields(
        _binding(
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            )
        )
    )
    assert precomputed_fields.messages_prefix == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "stable instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "text", "text": "semi-stable context"},
            ],
        }
    ]


def test_build_request_places_the_prefix_ahead_of_the_converted_messages() -> None:
    """Every request's messages are the binding's prefix followed by this call's Sequence[Message]."""
    bound = _adapter().bind_text(_binding(system_prompt="sys"))
    request = bound.build_request([UserMessage(content="q")])
    assert isinstance(request, _ChatCompletionsRequestParams)
    assert request.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]


@pytest.mark.parametrize(
    "part",
    [
        ImagePart(data=b"png", media_type="image/png"),
        ImageUrlPart(url="https://example.com/image.png"),
        AudioPart(data=b"wav", media_type="audio/wav"),
    ],
)
def test_build_request_reports_image_part_image_url_part_and_audio_part_in_tool_message(
    part: ContentPart,
) -> None:
    """ImagePart, ImageUrlPart, and AudioPart return InvalidRequest before sending."""
    bound = _adapter().bind_text(_binding())
    request = bound.build_request([
        ToolMessage(
            tool_call_id="c1",
            content=(TextPart(text="saw"), part),
        )
    ])
    assert isinstance(request, InvalidRequest)
    assert "text-only" in request.reason
    assert type(part).__name__ in request.reason
    assert "ToolMessage" in request.reason


def test_build_request_reports_a_raw_part_in_a_turn_as_invalid_request() -> None:
    """Unsupported RawPart.raw produces InvalidRequest before sending."""
    bound = _adapter().bind_text(_binding())
    request = bound.build_request([
        UserMessage(content="q"),
        AssistantMessage(
            turn=(
                RawPart(raw={"type": "server_tool_use", "name": "web_search"}),
                TextPart(text="searching"),
            )
        ),
    ])
    assert isinstance(request, InvalidRequest)
    assert "no Chat Completions wire form" in request.reason


def test_request_maps_the_inference_params_and_omits_the_unset() -> None:
    """Each set parameter lands on its wire field; unset ones leave the omit sentinel."""
    fields_set = _adapter()._precompute_fields(
        _binding(
            inference_params=InferenceParams(
                max_completion_tokens=5, temperature=0.2, reasoning_effort="high"
            )
        )
    )
    assert fields_set.max_completion_tokens == 5
    assert fields_set.temperature == 0.2
    assert fields_set.reasoning_effort == "high"
    fields_unset = _adapter()._precompute_fields(_binding())
    assert isinstance(fields_unset.max_completion_tokens, openai.Omit)
    assert isinstance(fields_unset.temperature, openai.Omit)
    assert isinstance(fields_unset.reasoning_effort, openai.Omit)


def test_request_omits_tool_fields_without_tools_and_sends_all_three_with_them() -> None:
    """Tools bring tool_choice and parallel_tool_calls with them; toolless bindings send none of the three."""
    toolless = _adapter()._precompute_fields(_binding())
    assert isinstance(toolless.tools, openai.Omit)
    assert isinstance(toolless.tool_choice, openai.Omit)
    assert isinstance(toolless.parallel_tool_calls, openai.Omit)
    with_tool = _adapter()._precompute_fields(
        _binding(
            tool_schemas=(
                ToolSchema(
                    name="lookup",
                    description="d",
                    args_schema={"type": "object", "properties": {}},
                ),
            )
        )
    )
    assert with_tool.tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert with_tool.tool_choice == "auto"
    assert with_tool.parallel_tool_calls is True


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
        _ = _adapter(supports_prompt_cache_options=False)._precompute_fields(
            _binding(automatic_prompt_caching=False)
        )


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request; None leaves the omit sentinel."""
    assert isinstance(_adapter()._precompute_fields(_binding()).service_tier, openai.Omit)
    stated = OpenAIChatCompletionsAdapter(
        client=AsyncOpenAI(api_key="test"),
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=True,
        service_tier="flex",
    )
    assert stated._precompute_fields(_binding()).service_tier == "flex"


def test_request_rejects_an_extra_body_key_the_adapter_populates() -> None:
    """An extra_body key that open_stream passes as its own keyword raises at bind time.

    stream is in the rejected set: the SDK merges extra_body over the named parameters with
    extra_body winning, and the stream path depends on it.
    """
    with pytest.raises(ValueError, match="temperature"):
        _ = _adapter()._precompute_fields(_binding(extra_body={"temperature": 0.5}))
    with pytest.raises(ValueError, match="stream"):
        _ = _adapter()._precompute_fields(_binding(extra_body={"stream": False}))


def test_adapter_pins_sdk_retries_off() -> None:
    """The stored client copy carries max_retries=0 so only langchaint retries."""
    assert _adapter().client.max_retries == 0


def _text_bound() -> _BoundChatCompletionsText:
    """Build a text-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    return _BoundChatCompletionsText(
        adapter=adapter, precomputed_fields=adapter._precompute_fields(_binding())
    )


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundChatCompletionsStructured[_StructuredReport]:
    """Build a structured-bound adapter over a keyless client; no request is sent."""
    adapter = _adapter()
    return _BoundChatCompletionsStructured(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding()),
        response_format=_StructuredReport,
    )


_REPORT_JSON = '{"city": "Nairobi", "celsius": 25}'
"""Text that validates into _StructuredReport."""


def _structured_completion(
    text: str | None,
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
    tool_call: bool = False,
) -> ChatCompletion:
    """Build a completion whose assistant message carries the given content, refusal, and tool call."""
    message: dict[str, object] = {"content": text, "refusal": refusal}
    if tool_call:
        message["tool_calls"] = [_TOOL_CALL_WIRE]
    return _completion(usage=None, message=message, finish_reason=finish_reason)


def _structured_parse(completion: ChatCompletion) -> _StructuredReport | None | NoOutputOutcome:
    """Run the structured binding's parse over one completion's finished turn."""
    return _structured_bound()._parsed_output(_finished(completion))


def test_structured_bind_validates_the_turns_text_into_the_instance() -> None:
    """The structured bound adapter validates the message's content into the response_format."""
    outcome = _structured_parse(_structured_completion(_REPORT_JSON))
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)
    interpreted = _assert_result(
        _structured_bound().interpret(_structured_completion(_REPORT_JSON))
    )
    assert interpreted.output == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_bind_reports_empty_turn_when_the_turn_carried_no_text() -> None:
    """A stop-finished completion with no content and no tool call is EmptyTurn."""
    assert isinstance(_structured_parse(_structured_completion(None)), EmptyTurn)


def test_structured_bind_reports_schema_violation_on_text_the_model_rejects() -> None:
    """A finished turn whose text the response_format rejects is SchemaViolation.

    validation_error_json names the rejected field, what rejected it, and the value, which is what
    tells a caller whether to change the model or the prompt.
    """
    outcome = _structured_parse(
        _structured_completion('{"city": "Nairobi", "celsius": "SENTINEL"}')
    )
    assert isinstance(outcome, SchemaViolation)
    rejections = json.loads(outcome.validation_error_json)
    assert [rejection["loc"] for rejection in rejections] == [["celsius"]]
    assert rejections[0]["input"] == "SENTINEL"


def test_structured_bind_reports_max_completion_tokens_exceeded_on_text_cut_mid_json() -> None:
    """A length-finished turn whose JSON stopped mid-object is the truncation, not a schema violation."""
    outcome = _structured_parse(_structured_completion('{"city": "Nair', finish_reason="length"))
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_bind_reports_the_truncation_on_a_tool_call_cut_by_the_token_cap() -> None:
    """A length-finished turn carrying tool calls is the truncation, never a dispatchable turn.

    The ordinary shape is arguments cut mid-JSON, so returning None here would hand the
    application a ToolCall whose args_json does not parse.
    """
    outcome = _structured_parse(
        _structured_completion(None, finish_reason="length", tool_call=True)
    )
    assert isinstance(outcome, MaxCompletionTokensExceeded)


def test_structured_bind_reports_refusal_on_a_refusal_beside_a_tool_call() -> None:
    """A refusal arriving with a tool call is the model declining, not a dispatchable turn."""
    outcome = _structured_parse(
        _structured_completion(None, refusal="I can't help", tool_call=True)
    )
    assert isinstance(outcome, Refusal)


def test_structured_bind_reports_empty_turn_and_preserves_a_custom_tool_call() -> None:
    """EmptyTurn.assistant_message preserves a custom tool call langchaint cannot dispatch."""
    completion = _completion(
        usage=None,
        message={"tool_calls": [_CUSTOM_TOOL_CALL_WIRE]},
        finish_reason="tool_calls",
    )
    outcome = _structured_parse(completion)
    assert isinstance(outcome, EmptyTurn)
    assert outcome.assistant_message.turn == (RawPart(raw=_CUSTOM_TOOL_CALL_WIRE),)


def test_structured_bind_reports_a_tool_call_turn_as_none() -> None:
    """A tool-call turn parses no instance and nothing went wrong, its prose text included."""
    assert _structured_parse(_structured_completion(None, tool_call=True)) is None
    assert _structured_parse(_structured_completion("let me look that up", tool_call=True)) is None


def test_structured_bind_sets_output_on_a_turn_that_also_called_a_tool() -> None:
    """The instance lands on output and the call still lands on tool_calls, so neither fact hides the other."""
    outcome = _structured_parse(_structured_completion(_REPORT_JSON, tool_call=True))
    assert outcome == _StructuredReport(city="Nairobi", celsius=25)


def test_structured_bind_reports_refusal_and_never_validates_the_refusal_text() -> None:
    """A refusal is the model declining, so its sentences are never a candidate instance."""
    outcome = _structured_parse(_structured_completion(None, refusal=_REPORT_JSON))
    assert isinstance(outcome, Refusal)
    assert outcome.assistant_message.turn == (TextPart(text=_REPORT_JSON),)


def test_structured_bind_reports_refusal_on_a_content_filter_finish() -> None:
    """A content_filter finish with no text is Refusal, not EmptyTurn."""
    outcome = _structured_parse(_structured_completion(None, finish_reason="content_filter"))
    assert isinstance(outcome, Refusal)


def test_structured_request_replaces_the_omitted_response_format() -> None:
    """The structured binding's precomputed fields carry the non-strict JSON-schema format."""
    response_format = _structured_bound()._precomputed_fields.response_format
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "_StructuredReport",
            "schema": _StructuredReport.model_json_schema(),
            "strict": False,
        },
    }
    assert isinstance(_text_bound()._precomputed_fields.response_format, openai.Omit)


def test_text_bind_reports_the_refusal_sentences_as_the_output() -> None:
    """The refusal is the turn's text under the text binding, its condition named by the stop reason."""
    result = _assert_result(
        _text_bound().interpret(_structured_completion(None, refusal="I can't help"))
    )
    assert result.output == "I can't help"
    assert result.stop_reason == "refusal"


def test_a_completion_with_no_choices_is_unfinished_turn() -> None:
    """No choices is a response langchaint cannot read a turn from, with an empty partial turn."""
    outcome = _text_bound().interpret(_completion(usage=None, choices=[]))
    assert isinstance(outcome, UnfinishedTurn)
    assert outcome.assistant_message.turn == ()


def test_a_choice_with_no_finish_reason_is_unfinished_turn_carrying_the_partial_turn() -> None:
    """finish_reason reads None at runtime on a lenient snapshot, which is not a finished turn."""
    outcome = _text_bound().interpret(_lenient_completion(None))
    assert isinstance(outcome, UnfinishedTurn)
    assert outcome.assistant_message.text == "hey"


def test_identity_reads_the_completions_own_id_and_served_model() -> None:
    """model_served reports what served the request, and request_id is None on the model itself."""
    identity = _text_bound().identity_from_raw(_completion(usage=None, model="m-2026-01-01"))
    assert identity.response_id == "r1"
    assert identity.model_served == "m-2026-01-01"
    assert identity.request_id is None


class _FakeSDKStream(AsyncStream[ChatCompletionChunk]):
    """Replays constructed chunks, raising where the replay holds an exception, without a connection.

    Overrides exactly the surface _ChatCompletionsStream uses (iteration, close, and the response
    its headers are read off); the base __init__ is deliberately not called, so the untouched base
    machinery stays unusable.
    """

    def __init__(  # pyrefly: ignore[missing-super-call]
        self,
        replay: Sequence[ChatCompletionChunk | Exception],
        headers: dict[str, str] | None = None,
    ) -> None:
        self._replay = list(replay)
        self.response = httpx.Response(
            200,
            headers=headers,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

    @override
    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk_or_error in self._replay:
            if isinstance(chunk_or_error, Exception):
                raise chunk_or_error
            yield chunk_or_error

    @override
    async def close(self) -> None:
        return


def _stream(
    replay: Sequence[ChatCompletionChunk | Exception], headers: dict[str, str] | None = None
) -> _ChatCompletionsStream:
    """Build an adapter stream over replayed chunks, reading headers off a constructed response."""
    return _ChatCompletionsStream(
        sdk_stream=_FakeSDKStream(replay, headers),
        pricing=_PRICING,
        cache_read_tokens_from_usage=cache_read_tokens_from_usage_openai,
    )


def _chunk(
    *,
    delta: Mapping[str, object] | None = None,
    finish_reason: str | None = None,
    usage: CompletionUsage | None = None,
    choices: list[object] | None = None,
) -> ChatCompletionChunk:
    """Build one chunk with a single choice; choices=[] is the usage-only trailing chunk."""
    if choices is None:
        choices = [{"index": 0, "delta": dict(delta or {}), "finish_reason": finish_reason}]
    return ChatCompletionChunk.model_validate({
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": choices,
        "usage": usage,
    })


def _collected_items(replay: Sequence[ChatCompletionChunk | Exception]) -> list[StreamItem]:
    """Drain the translated items into a list."""

    async def scenario() -> list[StreamItem]:
        return [item async for item in _stream(replay).items()]

    return asyncio.run(scenario())


def _text_stream_chunks() -> list[ChatCompletionChunk]:
    """Chunks in the provider's order: role and text deltas, the finish, then the usage-only chunk."""
    return [
        _chunk(delta={"role": "assistant", "content": "he"}),
        _chunk(delta={"content": "y"}),
        _chunk(finish_reason="stop"),
        _chunk(choices=[], usage=_usage_with_cache()),
    ]


def test_stream_passes_text_deltas_through_as_bare_strings() -> None:
    """Answer text streams as the SDK's own delta strings, and the terminal chunks add no items."""
    assert _collected_items(_text_stream_chunks()) == ["he", "y"]


def test_stream_yields_reasoning_deltas_and_the_final_turn_carries_their_concatenation() -> None:
    """Each reasoning_content delta streams as one ReasoningDelta and the snapshot joins them."""
    stream = _stream([
        _chunk(delta={"role": "assistant", "reasoning_content": "part a"}),
        _chunk(delta={"reasoning_content": " part b", "content": "hey"}),
        _chunk(finish_reason="stop"),
    ])

    async def scenario() -> tuple[list[StreamItem], ChatCompletion]:
        items = [item async for item in stream.items()]
        return items, await stream.final()

    items, final = asyncio.run(scenario())
    assert items == [
        ReasoningDelta(text="part a"),
        ReasoningDelta(text=" part b"),
        "hey",
    ]
    result = _assert_result(_text_bound().interpret(final))
    assert result.assistant_message.turn[0] == ReasoningPart(
        raw={"reasoning_content": "part a part b"}, text="part a part b"
    )


@pytest.mark.parametrize(
    ("id_carrying_fragment_index", "expected_deltas"),
    [
        (
            0,
            [
                ToolCallDelta(id="call1", name="lookup", partial_args_json='{"q"'),
                ToolCallDelta(id="call1", name="lookup", partial_args_json=": 1"),
                ToolCallDelta(id="call1", name="lookup", partial_args_json=', "r": 2}'),
            ],
        ),
        (
            1,
            [
                ToolCallDelta(id="call1", name="lookup", partial_args_json='{"q": 1'),
                ToolCallDelta(id="call1", name="lookup", partial_args_json=', "r": 2}'),
            ],
        ),
    ],
    ids=["id_on_first_fragment", "id_on_second_fragment"],
)
def test_stream_yields_argument_fragments_then_one_complete_tool_call(
    id_carrying_fragment_index: int, expected_deltas: list[ToolCallDelta]
) -> None:
    """Fragment merging stays in the SDK, and ids come off the assembled snapshot.

    A fragment with the id known yields a ToolCallDelta at once.
    A fragment before the id arrives is held back and prefixed to the next fragment that yields.
    An OpenAI-compatible provider may omit the id on early fragments.
    Either way the concatenated deltas are the completed call's args_json.
    """
    fragments: list[dict[str, object]] = [
        {"index": 0, "type": "function", "function": {"name": "lookup", "arguments": '{"q"'}},
        {"index": 0, "function": {"arguments": ": 1"}},
        {"index": 0, "function": {"arguments": ', "r": 2}'}},
    ]
    fragments[id_carrying_fragment_index]["id"] = "call1"
    items = _collected_items([
        _chunk(delta={"role": "assistant"}),
        _chunk(delta={"tool_calls": [fragments[0]]}),
        _chunk(delta={"tool_calls": [fragments[1]]}),
        _chunk(delta={"tool_calls": [fragments[2]]}),
        _chunk(finish_reason="tool_calls"),
    ])
    expected_call = ToolCall(id="call1", name="lookup", args_json='{"q": 1, "r": 2}')
    assert items == [*expected_deltas, expected_call]


def test_a_sparse_tool_call_fragment_index_is_a_stream_protocol_error() -> None:
    """The SDK's state cannot place a fragment whose index skips its predecessors."""
    replay = [
        _chunk(delta={"role": "assistant"}),
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            }
        ),
    ]
    with pytest.raises(StreamProtocolError, match="index"):
        _ = _collected_items(replay)


def test_billing_reported_is_none_until_the_usage_chunk_arrives() -> None:
    """A stream cut off before the usage-bearing chunk reports None, and the full drain reports it."""
    stream = _stream(_text_stream_chunks())

    async def scenario() -> tuple[Billing | None, Billing | None]:
        before = stream.billing_reported()
        async for _item in stream.items():
            pass
        return before, stream.billing_reported()

    before, after = asyncio.run(scenario())
    assert before is None
    assert after is not None
    assert after.usage.input_tokens_total == 1000


def test_final_patches_the_tracked_usage_over_a_trailing_chunks_reset() -> None:
    """A usage-less chunk after the usage chunk resets the snapshot's usage; final() restores it."""
    stream = _stream([*_text_stream_chunks(), _chunk(choices=[])])

    async def scenario() -> ChatCompletion:
        async for _item in stream.items():
            pass
        return await stream.final()

    assert asyncio.run(scenario()).usage == _usage_with_cache()


def test_final_before_items_are_exhausted_raises() -> None:
    """final() without a drained stream has nothing assembled to return."""
    with pytest.raises(StreamProtocolError, match="items"):
        _ = asyncio.run(_stream(_text_stream_chunks()).final())


def test_a_stream_reports_the_request_id_header_of_the_response_it_reads() -> None:
    """The stream's own response is the only channel a streamed turn has for the header.

    The snapshot the SDK's state assembles never carries it, so a null here would leave every
    streaming call with no id to take to provider support.
    """
    assert _stream([], {"x-request-id": "req_stream"}).request_id() == "req_stream"
    assert _stream([]).request_id() is None


def _bare_api_error() -> openai.APIError:
    """Build the bare APIError the SDK raises for a mid-stream SSE error payload."""
    return openai.APIError(
        "provider mid-stream error",
        httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        body={"code": "server_error", "type": "insufficient_quota", "message": "boom"},
    )


def test_a_mid_stream_bare_api_error_rewraps_as_a_status_error_on_the_live_response() -> None:
    """The rewrap carries the 200 status and the error's code, so parse_openai verdicts it.

    server_error is a transient code, so the mid-stream failure retries rather than failing the item.
    """
    replay = [_chunk(delta={"role": "assistant", "content": "he"}), _bare_api_error()]
    with pytest.raises(openai.APIStatusError) as raised:
        _ = _collected_items(replay)
    assert raised.value.status_code == 200
    assert raised.value.code == "server_error"
    assert "provider mid-stream error" in raised.value.message
    assert isinstance(raised.value.__cause__, openai.APIError)
    assert _adapter().parse(raised.value) == RetryThisOne(retry_after=None)


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
        openai.APIResponseValidationError(
            response=httpx.Response(
                200, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            ),
            body=None,
        ),
    ],
    ids=["connection_error", "response_validation_error"],
)
def test_an_api_error_subclass_raised_mid_stream_propagates_untouched(error: Exception) -> None:
    """Only the bare APIError is the SSE error payload; every subclass keeps its own meaning."""
    with pytest.raises(type(error)) as raised:
        _ = _collected_items([_chunk(delta={"role": "assistant"}), error])
    assert raised.value is error


def _kwarg_sent[OutputT](
    monkeypatch: pytest.MonkeyPatch, bound: _BoundChatCompletions[OutputT], key: str
) -> object:
    """Open one stream through a fake create, capturing the request kwarg key it was passed."""
    captured: list[object] = []

    async def fake_create(**request_kwargs: object) -> _FakeSDKStream:
        captured.append(request_kwargs[key])
        return _FakeSDKStream([])

    monkeypatch.setattr(bound._adapter.client.chat.completions, "create", fake_create)
    request = bound.build_request([UserMessage(content="q")])
    assert not isinstance(request, InvalidRequest)
    _ = asyncio.run(bound.open_stream(request))
    (kwarg,) = captured
    return kwarg


def test_the_request_sends_extra_body_by_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_stream passes the binding's extra_body to the SDK's extra_body parameter.

    A request that dropped it would silently go out without the caller's wire fields,
    which no offline round-trip test can catch.
    """
    adapter = _adapter()
    extra_body = {"safety_identifier": "user-7"}
    text_bound = _BoundChatCompletionsText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(extra_body=extra_body)),
    )
    assert _kwarg_sent(monkeypatch, text_bound, "extra_body") is extra_body


def test_every_request_streams_and_asks_for_the_usage_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stream=True is the request path and stream_options is what produces the trailing usage."""
    assert _kwarg_sent(monkeypatch, _text_bound(), "stream") is True
    assert _kwarg_sent(monkeypatch, _text_bound(), "stream_options") == {"include_usage": True}


def test_a_built_request_renders_as_json_carrying_the_messages_and_no_omitted_field() -> None:
    """as_json holds the binding's precomputed fields and this call's converted messages.

    temperature is absent rather than null, because the binding set none and the request body
    carries no such key.
    """
    adapter = _adapter()
    bound = _BoundChatCompletionsText(
        adapter=adapter,
        precomputed_fields=adapter._precompute_fields(_binding(system_prompt="sys")),
    )
    request = bound.build_request([UserMessage(content="hi")])
    assert isinstance(request, _ChatCompletionsRequestParams)
    rendered = json.loads(request.as_json())
    assert rendered["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert "temperature" not in rendered["precomputed"]


def _conformance_message() -> dict[str, object]:
    """Return an assistant message whose reasoning_content is the key the installed SDK does not name.

    An adapter that rebuilt the message from the SDK's pinned model would drop that key, which
    DeepSeek requires byte-identical on replay inside a tool loop.
    """
    return {"content": "hey", "reasoning_content": "thought it over"}


class TestOpenAIChatCompletionsConformance(AdapterConformance):
    """The neutral invariants, over the Chat Completions adapter's own SDK objects."""

    @override
    def make_adapter(self) -> Adapter:
        """Build the adapter these invariants run against, priced for the default tier alone."""
        return _adapter()

    @override
    def response_with_cache_writes(self) -> BaseModel:
        """Return a turn whose prompt_tokens carries both a cache read and a cache write."""
        return _completion(usage=_usage_with_cache())

    @override
    def response_without_usage(self) -> BaseModel:
        """Return a turn whose usage field is absent, the runtime state a cut-off stream leaves."""
        return _completion(usage=None)

    @override
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return a turn served at flex, which _PRICING holds no table for."""
        return _completion(usage=_usage_with_cache(), service_tier="flex")

    @override
    def response_with_impossible_counters(self) -> BaseModel:
        """Return a turn whose cache counters sum past prompt_tokens.

        The uncached counter is the subtraction, so counters summing past the total drive it below
        zero.
        """
        return _completion(
            usage=CompletionUsage.model_validate({
                "prompt_tokens": 1000,
                "completion_tokens": 40,
                "total_tokens": 1040,
                "prompt_tokens_details": {"cached_tokens": 900, "cache_write_tokens": 200},
            })
        )

    @override
    def response_with_reasoning(self) -> BaseModel:
        """Return a turn carrying reasoning_content beside its one text part."""
        return _completion(usage=_usage_with_cache(), message=_conformance_message())

    @override
    def response_with_raw_part(self) -> BaseModel:
        """Return message.content beside one custom message.tool_calls entry.

        openai 2.51.0 defines both fields on ChatCompletionMessage.
        """
        return _completion(
            usage=_usage_with_cache(),
            message={"content": "hello", "tool_calls": [_CUSTOM_TOOL_CALL_WIRE]},
        )

    @override
    def assistant_wire_parts(self, request: RequestParams) -> Sequence[object]:
        """Decompose one assistant param into wire-order parts.

        This wire stores the turn in one message param.
        Split ReasoningPart.raw, content, and ToolCall values in TurnPart order.
        """
        assert isinstance(request, _ChatCompletionsRequestParams)
        (assistant_param,) = request.messages[1:]
        payload = dict(assistant_param)
        parts: list[object] = []
        if "reasoning_content" in payload:
            parts.append({"reasoning_content": payload["reasoning_content"]})
        if "content" in payload:
            parts.append(payload["content"])
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            parts.extend(tool_calls)
        return parts

    @override
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return the same turn as the snapshot ChatCompletionStreamState assembles and whole."""
        whole = _completion(usage=_usage_with_cache(), message=_conformance_message())
        state = ChatCompletionStreamState()
        for chunk in (
            _chunk(
                delta={
                    "role": "assistant",
                    "content": "hey",
                    "reasoning_content": "thought it over",
                },
                finish_reason="stop",
            ),
            _chunk(choices=[], usage=_usage_with_cache()),
        ):
            _ = state.handle_chunk(chunk)
        return state.current_completion_snapshot, whole

    @override
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream whose chunks end before any finish_reason."""
        return _stream([_chunk(delta={"role": "assistant", "content": "he"})])

    @override
    def sdk_errors_and_classifications(self) -> Mapping[Exception, ErrorClassification]:
        """Return the table both openai adapters share; its builder's docstring states each row."""
        return openai_sdk_errors_and_classifications()

    @override
    def sdk_errors_and_verdicts(self) -> Mapping[Exception, Verdict]:
        """Return the parse rows both openai adapters share; the builder's docstring names their sources."""
        return openai_sdk_errors_and_verdicts()
