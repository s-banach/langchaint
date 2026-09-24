"""Test OpenAI Chat Completions with constructed SDK objects.

Tests cover Usage, assistant messages, stop reasons, streams, errors, and requests.
"""

import json
import math
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import override

import httpx2
import openai
import pytest
from openai import AsyncOpenAI, AsyncStream
from openai._models import construct_type_unchecked
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from langchaint import (
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
    BoundAdapter,
    ErrorClassification,
    InvalidRequest,
    RequestParams,
    ResponseOutcome,
)
from langchaint.billing.pricing import Billing, ProviderBilling
from langchaint.common.exceptions import StreamProtocolError
from langchaint.concurrency.shared_backoff import RetryThisOne, Verdict
from langchaint.conformance import AdapterConformance
from langchaint.deepseek import cache_read_tokens_from_usage_deepseek
from langchaint.openai import (
    OpenAIChatCompletionsAdapter,
    OpenAIPricingTable,
    OpenAIRates,
    OpenAIServiceTier,
)
from langchaint.openai.chat_completions_adapter import (
    _assistant_message_from,
    _assistant_message_param,
    _billing_from_chat_completion,
    _ChatCompletionsRequestParams,
    _ChatCompletionsStream,
    _wire_messages,
    _wire_tool_choice,
    cache_read_tokens_from_usage_openai,
)
from langchaint.tools import ToolSchema
from tests.helpers import (
    connection_error,
    openai_sdk_errors_and_classifications,
    openai_sdk_errors_and_verdicts,
    run_with_timeout,
)

_DEFAULT_RATES = OpenAIRates(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)

_PRICING = OpenAIPricingTable(default=_DEFAULT_RATES)
"""The default tier alone, so a response reporting another tier prices NaN."""

_PRIORITY_RATES = OpenAIRates(
    input_cache_none_usd_per_million_tokens=5.0,
    output_usd_per_million_tokens=20.0,
    cache_read_usd_per_million_tokens=2.5,
    cache_write_usd_per_million_tokens=6.25,
)
"""Twice the default rates, so a tier-selection test reads as a doubling."""

_TOOL_CALL_WIRE: dict[str, object] = {
    "id": "call1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"q": 1}'},
}
"""One function tool call as the API returns it on an assistant message."""

_TOOL_CALL = ToolCall(id="call1", name="lookup", args_json='{"q": 1}')
"""The ToolCall that _TOOL_CALL_WIRE converts to."""

_CUSTOM_TOOL_CALL_WIRE: dict[str, JsonValue] = {
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
    assert outcome.kind == "adapter_result"
    return outcome


def _usage(
    prompt_tokens_details: Mapping[str, int] | None = None,
    completion_tokens_details: Mapping[str, int] | None = None,
) -> CompletionUsage:
    """Return usage of 1000 prompt tokens and 40 completion tokens with the given details."""
    return CompletionUsage.model_validate({
        "prompt_tokens": 1000,
        "completion_tokens": 40,
        "total_tokens": 1040,
        "prompt_tokens_details": prompt_tokens_details,
        "completion_tokens_details": completion_tokens_details,
    })


def _usage_with_cache() -> CompletionUsage:
    """Return usage whose prompt_tokens includes both cache counters."""
    return _usage(
        prompt_tokens_details={"cached_tokens": 600, "cache_write_tokens": 100},
        completion_tokens_details={"reasoning_tokens": 0},
    )


def _deepseek_usage() -> CompletionUsage:
    """Return usage as DeepSeek reports it: the cache partition in extra fields, no details objects."""
    return CompletionUsage.model_validate({
        "prompt_tokens": 1000,
        "completion_tokens": 40,
        "total_tokens": 1040,
        "prompt_cache_hit_tokens": 600,
        "prompt_cache_miss_tokens": 400,
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
    """Build a completion whose id is fixed at "r1". Every field a test varies is a parameter.

    message holds the assistant message's fields minus role. None gives content "hey".
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

    interpret handles None and unknown stop reasons from lenient SDK snapshots.
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


def _provider_billing(
    completion: ChatCompletion,
    pricing: OpenAIPricingTable = _PRICING,
    cache_read_tokens_from_usage: Callable[
        [CompletionUsage], int
    ] = cache_read_tokens_from_usage_openai,
) -> ProviderBilling:
    """Price one completion, with the openai cache-read reader unless another is given."""
    return _billing_from_chat_completion(
        completion, pricing=pricing, cache_read_tokens_from_usage=cache_read_tokens_from_usage
    )


def _billing(completion: ChatCompletion, pricing: OpenAIPricingTable = _PRICING) -> Billing:
    """Price one completion with the openai cache-read reader, the adapter's default."""
    return _provider_billing(completion, pricing).billing


@pytest.mark.parametrize(
    ("usage_raw", "cache_read_tokens_from_usage", "expected_counters"),
    [
        (_usage_with_cache(), cache_read_tokens_from_usage_openai, (600, 100, 300, 0)),
        (
            _usage(completion_tokens_details={"reasoning_tokens": 8}),
            cache_read_tokens_from_usage_openai,
            (0, 0, 1000, 8),
        ),
        (_deepseek_usage(), cache_read_tokens_from_usage_openai, (0, 0, 1000, 0)),
        (_deepseek_usage(), cache_read_tokens_from_usage_deepseek, (600, 0, 400, 0)),
        (_usage_with_cache(), cache_read_tokens_from_usage_deepseek, (0, 100, 900, 0)),
    ],
    ids=[
        "openai_cache_details",
        "openai_reasoning_details_only",
        "openai_reader_without_details_objects",
        "deepseek_reader_on_deepseek_counters",
        "deepseek_reader_without_its_extra_field",
    ],
)
def test_billing_partitions_prompt_tokens_by_the_cache_read_reader(
    usage_raw: CompletionUsage,
    cache_read_tokens_from_usage: Callable[[CompletionUsage], int],
    expected_counters: tuple[int, int, int, int],
) -> None:
    """The uncached counter is prompt_tokens minus the cache-read and cache-write counters.

    cache_read_tokens_from_usage prevents treating DeepSeek's cache hits as uncached tokens.
    An absent details object reads as zero.
    expected_counters lists the cache-read, cache-write, and uncached input counters, then reasoning.
    """
    usage = _provider_billing(
        _completion(usage=usage_raw), cache_read_tokens_from_usage=cache_read_tokens_from_usage
    ).billing.usage
    assert (
        usage.input_tokens_cache_read,
        usage.input_tokens_cache_write,
        usage.input_tokens_cache_none,
        usage.output_tokens_reasoning,
    ) == expected_counters


def test_billing_carries_the_sdk_usage_object_itself() -> None:
    """usage_raw is the completion's own CompletionUsage by reference, and None where it reported none."""
    raw = _completion(usage=_usage_with_cache())
    assert _provider_billing(raw).usage_raw is raw.usage
    assert _provider_billing(_completion(usage=None)).usage_raw is None


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


def test_billing_without_usage_pins_the_priced_tiers_rates() -> None:
    """A completion missing usage still stores the rates the tier that served it would have spent."""
    billing = _billing(_completion(usage=None))
    assert billing.output_usd_per_million_tokens == 10.0
    assert billing.service_tier == "default"


def test_the_reported_tier_selects_the_table() -> None:
    """Priority rates price a priority response. "auto" and no tier both price at default."""
    pricing = OpenAIPricingTable(default=_DEFAULT_RATES, fast=_PRIORITY_RATES)

    def cost_at(service_tier: str | None) -> float:
        """Price the cached usage as served at service_tier."""
        completion = _completion(usage=_usage_with_cache(), service_tier=service_tier)
        return _billing(completion, pricing).usage.cost_in_usd

    default_cost = cost_at("default")
    assert default_cost == pytest.approx((300 * 2.5 + 600 * 1.25 + 100 * 3.125 + 40 * 10.0) / 1e6)
    assert cost_at("priority") == pytest.approx(2 * default_cost)
    assert cost_at("auto") == default_cost
    assert cost_at(None) == default_cost


@pytest.mark.parametrize(
    ("build_completion", "expected", "expected_output"),
    [
        (lambda: _completion(usage=None), "end_turn", "hey"),
        (
            lambda: _completion(
                usage=None, message={"content": "on it", "tool_calls": [_TOOL_CALL_WIRE]}
            ),
            "tool_use",
            "on it",
        ),
        (
            lambda: _completion(
                usage=None,
                message={"tool_calls": [_TOOL_CALL_WIRE]},
                finish_reason="tool_calls",
            ),
            "tool_use",
            "",
        ),
        (lambda: _completion(usage=None, finish_reason="length"), "max_tokens", "hey"),
        (lambda: _completion(usage=None, finish_reason="content_filter"), "refusal", "hey"),
        (
            lambda: _completion(usage=None, message={"refusal": "I can't help with that"}),
            "refusal",
            "I can't help with that",
        ),
        (lambda: _completion(usage=None, finish_reason="function_call"), "other", "hey"),
        (lambda: _lenient_completion("weird"), "other", "hey"),
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
    build_completion: Callable[[], ChatCompletion], expected: StopReason, expected_output: str
) -> None:
    """Check translated stop reasons alongside the preserved output.

    Under the text binding, a refusal's sentences are the output and the stop reason names the refusal.
    """
    result = _assert_result(_text_bound().interpret(build_completion()))
    assert result.stop_reason == expected
    assert result.output == expected_output


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
        _TOOL_CALL,
    )


def test_a_custom_tool_call_becomes_a_raw_part_and_replays_in_order() -> None:
    """The custom.input and ToolCall.args_json fields name different provider concepts."""
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": None,
        "tool_calls": [_CUSTOM_TOOL_CALL_WIRE, _TOOL_CALL_WIRE],
    })
    assistant_message = _assistant_message_from(message)
    assert assistant_message.turn == (RawPart(raw=_CUSTOM_TOOL_CALL_WIRE), _TOOL_CALL)
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "tool_calls": [_CUSTOM_TOOL_CALL_WIRE, _TOOL_CALL_WIRE],
    }


@pytest.mark.parametrize(
    "function_call",
    [
        _FUNCTION_CALL_WIRE,
        {**_FUNCTION_CALL_WIRE, "type": "custom"},
        {**_FUNCTION_CALL_WIRE, "type": "future"},
    ],
    ids=["plain", "extra_type_custom", "extra_type_future"],
)
def test_a_function_call_becomes_a_raw_part_and_replays_unchanged(
    function_call: dict[str, object],
) -> None:
    """The function_call field lacks the id ToolCall requires.

    The function_call field identifies the replay position independently from its value.
    """
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


def test_foreign_reasoning_merges_its_keys_into_the_param_unchanged() -> None:
    """A foreign ReasoningPart sends ReasoningPart.raw unchanged for provider validation."""
    raw: dict[str, JsonValue] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(turn=(ReasoningPart(raw=raw), TextPart(text="hi")))
    assert _assistant_message_param(assistant_message) == {
        "role": "assistant",
        "content": "hi",
        **raw,
    }


def test_wire_messages_converts_each_message_kind() -> None:
    """User, assistant, and tool messages each map to their message param.

    The assistant turn becomes one param whose texts join into content.
    ToolCall values and ReasoningPart.raw keep their fields.
    """
    wire = _wire_messages([
        UserMessage(content="q"),
        AssistantMessage(
            turn=(
                ReasoningPart(
                    raw={"reasoning_content": "thought it over"}, text="thought it over"
                ),
                TextPart(text="he"),
                TextPart(text="y"),
                _TOOL_CALL,
            )
        ),
        ToolMessage(tool_call_id="call1", content="r"),
    ])
    assert wire == [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "reasoning_content": "thought it over",
            "content": "hey",
            "tool_calls": [_TOOL_CALL_WIRE],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "r"},
    ]


def test_wire_messages_maps_user_and_tool_parts_and_marks_marked_ones() -> None:
    """Each content part maps to its wire part, and a marked part carries prompt_cache_breakpoint.

    AudioPart.media_type "audio/wav" and "audio/mpeg" map to input_audio.format "wav" and "mp3".
    """
    marked = {"prompt_cache_breakpoint": {"mode": "explicit"}}
    wire = _wire_messages([
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
                ImageUrlPart(url="https://example.com/image.png", cache_breakpoint=True),
                AudioPart(data=b"wav", media_type="audio/wav", cache_breakpoint=True),
                AudioPart(data=b"mp3", media_type="audio/mpeg"),
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
                {"type": "text", "text": "shared context", **marked},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,cG5n", "detail": "auto"},
                    **marked,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png", "detail": "auto"},
                    **marked,
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": "d2F2", "format": "wav"},
                    **marked,
                },
                {"type": "input_audio", "input_audio": {"data": "bXAz", "format": "mp3"}},
                {"type": "text", "text": "question"},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "saw", **marked},
                {"type": "text", "text": "more"},
            ],
        },
    ]


@pytest.mark.parametrize(
    ("messages", "reason_fragments"),
    [
        (
            [
                AssistantMessage(
                    turn=(
                        RawPart(raw={"function_call": _FUNCTION_CALL_WIRE}),
                        RawPart(raw={"function_call": {"name": "second", "arguments": "{}"}}),
                    )
                )
            ],
            ("more than one function_call",),
        ),
        (
            [
                AssistantMessage(
                    turn=(RawPart(raw={"type": "server_tool_use"}), TextPart(text="searching"))
                )
            ],
            ("no Chat Completions wire form",),
        ),
        (
            [UserMessage(content=(AudioPart(data=b"audio", media_type="audio/ogg"),))],
            ("AudioPart", "UserMessage", "audio/ogg"),
        ),
        *[
            (
                [ToolMessage(tool_call_id="c1", content=(part,))],
                ("text-only", type(part).__name__, "ToolMessage"),
            )
            for part in (
                ImagePart(data=b"png", media_type="image/png"),
                ImageUrlPart(url="https://example.com/image.png"),
                AudioPart(data=b"wav", media_type="audio/wav"),
            )
        ],
    ],
    ids=[
        "two_function_calls",
        "raw_part_without_wire_form",
        "audio_part_media_type_without_input_audio_format",
        "image_part_in_tool_message",
        "image_url_part_in_tool_message",
        "audio_part_in_tool_message",
    ],
)
def test_build_request_reports_unsendable_messages_as_invalid_request(
    messages: list[Message], reason_fragments: tuple[str, ...]
) -> None:
    """A message with no Chat Completions wire form returns InvalidRequest before sending.

    One assistant message has one function_call field, and the tool message param's content is text-only.
    """
    request = _adapter().bind_text(_binding()).build_request(messages)
    assert isinstance(request, InvalidRequest)
    for reason_fragment in reason_fragments:
        assert reason_fragment in request.reason


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("required", "required"),
        (SpecificToolChoice(tool_name="x"), {"type": "function", "function": {"name": "x"}}),
        *[
            (
                AllowedToolsChoice(mode=mode, tool_names=("second", "first")),
                {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": mode,
                        "tools": [
                            {"type": "function", "function": {"name": "second"}},
                            {"type": "function", "function": {"name": "first"}},
                        ],
                    },
                },
            )
            for mode in ("auto", "required")
        ],
    ],
    ids=["string", "specific_tool", "allowed_tools_auto", "allowed_tools_required"],
)
def test_wire_tool_choice(tool_choice: ToolChoice, expected: object) -> None:
    """Map the neutral tool choice to the Chat Completions tool_choice form.

    Neutral strings pass through unchanged.
    SpecificToolChoice becomes the function form.
    AllowedToolsChoice becomes the allowed_tools form listing function names in order.
    """
    assert _wire_tool_choice(tool_choice) == expected


def _adapter(
    *,
    client: AsyncOpenAI | None = None,
    supports_prompt_cache_options: bool = True,
    service_tier: OpenAIServiceTier | None = None,
) -> OpenAIChatCompletionsAdapter:
    """Build an adapter over a keyless client unless a test supplies one, valid because no request leaves.

    supports_prompt_cache_options=True sends the binding's cache setting.
    """
    return OpenAIChatCompletionsAdapter(
        client=AsyncOpenAI(api_key="test") if client is None else client,
        model="m",
        pricing=_PRICING,
        provider_name="openai",
        supports_prompt_cache_options=supports_prompt_cache_options,
        service_tier=service_tier,
    )


def test_config_fingerprint_data_contains_only_stored_request_configuration() -> None:
    """Fingerprint data includes constructor request settings and excludes billing settings."""
    adapter = _adapter(supports_prompt_cache_options=False, service_tier="priority")
    assert adapter.config_fingerprint_data() == {
        "service_tier": "priority",
        "supports_prompt_cache_options": False,
    }


def _binding(
    *,
    automatic_cache_breakpoints: bool = True,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    tool_schemas: tuple[ToolSchema, ...] = (),
    provider_executed_tools: tuple[Mapping[str, object], ...] = (),
    tool_choice: ToolChoice = "auto",
    max_completion_tokens: int | None = None,
    reasoning_level: str | None = None,
    temperature: float | None = None,
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    """Assemble a binding with the fields these request tests vary."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        provider_executed_tools=provider_executed_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
        max_completion_tokens=max_completion_tokens,
        reasoning_level=reasoning_level,
        temperature=temperature,
        automatic_cache_breakpoints=automatic_cache_breakpoints,
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


def test_request_maps_generation_fields_and_omits_the_unset() -> None:
    """Each set parameter lands on its wire field. Unset ones leave the omit sentinel."""
    fields_set = _adapter()._precompute_fields(
        _binding(max_completion_tokens=5, temperature=0.2, reasoning_level="high")
    )
    assert fields_set.max_completion_tokens == 5
    assert fields_set.temperature == 0.2
    assert fields_set.reasoning_effort == "high"
    fields_unset = _adapter()._precompute_fields(_binding())
    assert isinstance(fields_unset.max_completion_tokens, openai.Omit)
    assert isinstance(fields_unset.temperature, openai.Omit)
    assert isinstance(fields_unset.reasoning_effort, openai.Omit)


def test_request_omits_tool_fields_without_tools_and_sends_all_three_with_them() -> None:
    """Tools bring tool_choice and parallel_tool_calls with them. Toolless bindings send none."""
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


def test_allowed_tools_choice_keeps_complete_chat_completions_tool_definitions() -> None:
    """AllowedToolsChoice changes tool_choice without removing tools."""
    schemas = (
        ToolSchema(name="first", description="First.", args_schema={"type": "object"}),
        ToolSchema(name="second", description="Second.", args_schema={"type": "object"}),
    )
    tool_choice = AllowedToolsChoice(mode="required", tool_names=("second",))
    precomputed = _adapter()._precompute_fields(
        _binding(tool_schemas=schemas, tool_choice=tool_choice)
    )
    assert not isinstance(precomputed.tools, openai.Omit)
    assert [tool["function"]["name"] for tool in precomputed.tools] == ["first", "second"]
    assert precomputed.tool_choice == _wire_tool_choice(tool_choice)


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


def test_request_sends_service_tier_only_when_the_adapter_states_one() -> None:
    """A stated service_tier lands on the request. None leaves the omit sentinel."""
    assert isinstance(_adapter()._precompute_fields(_binding()).service_tier, openai.Omit)
    stated = _adapter(service_tier="flex")._precompute_fields(_binding())
    assert stated.service_tier == "flex"


def test_request_rejects_an_extra_body_key_the_adapter_populates() -> None:
    """An extra_body key that open_stream passes as its own keyword raises at bind time.

    extra_body cannot override stream.
    """
    with pytest.raises(ValueError, match="temperature"):
        _ = _adapter()._precompute_fields(_binding(extra_body={"temperature": 0.5}))
    with pytest.raises(ValueError, match="stream"):
        _ = _adapter()._precompute_fields(_binding(extra_body={"stream": False}))


def test_adapter_pins_sdk_retries_off() -> None:
    """The stored client copy carries max_retries=0 so only langchaint retries."""
    assert _adapter().client.max_retries == 0


def _text_bound() -> BoundAdapter[str]:
    """Bind a keyless adapter for text. No request is sent."""
    return _adapter().bind_text(_binding())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> BoundAdapter[_StructuredReport | None]:
    """Bind a keyless adapter for _StructuredReport. No request is sent."""
    return _adapter().bind_structured(_binding(), _StructuredReport)


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


def test_structured_bind_sets_output_on_a_turn_that_also_called_a_tool() -> None:
    """The message's content validates into the response_format beside the extracted tool call."""
    outcome = _assert_result(
        _structured_bound().interpret(_structured_completion(_REPORT_JSON, tool_call=True))
    )
    assert outcome.output == _StructuredReport(city="Nairobi", celsius=25)
    assert outcome.assistant_message.tool_calls == (_TOOL_CALL,)


@pytest.mark.parametrize(
    ("completion", "expected_kind"),
    [
        (_structured_completion(None), "empty_turn"),
        (
            _completion(
                usage=None,
                message={"tool_calls": [_CUSTOM_TOOL_CALL_WIRE]},
                finish_reason="tool_calls",
            ),
            "empty_turn",
        ),
        (
            _structured_completion('{"city": "Nair', finish_reason="length"),
            "max_completion_tokens_exceeded",
        ),
        (
            _structured_completion(None, finish_reason="length", tool_call=True),
            "max_completion_tokens_exceeded",
        ),
        (_structured_completion(None, refusal="I can't help", tool_call=True), "refusal"),
        (_structured_completion(None, refusal=_REPORT_JSON), "refusal"),
        (_structured_completion(None, finish_reason="content_filter"), "refusal"),
        (_structured_completion(None, tool_call=True), "adapter_result"),
        (_structured_completion("let me look that up", tool_call=True), "adapter_result"),
    ],
    ids=[
        "no_text",
        "custom_tool_call_only",
        "length_cut_mid_json",
        "length_cut_a_tool_call",
        "refusal_beside_a_tool_call",
        "refusal_text_that_would_validate",
        "content_filter",
        "tool_call",
        "tool_call_beside_prose",
    ],
)
def test_structured_bind_reports_why_a_turn_produced_no_instance(
    completion: ChatCompletion, expected_kind: str
) -> None:
    """Every outcome carries the converted turn, including tool calls langchaint cannot dispatch.

    A refusal is the model declining, so its sentences never enter validation.
    A length finish is the truncation, never a dispatchable turn, even with tool calls.
    A tool-call turn parses no instance and returns an AdapterResult whose output is None.
    """
    outcome = _structured_bound().interpret(completion)
    assert outcome.kind == expected_kind
    assert outcome.assistant_message == _assistant_message_from(completion.choices[0].message)


def test_a_completion_with_no_choices_is_unfinished_turn() -> None:
    """No choices is a response langchaint cannot read a turn from, with an empty partial turn."""
    outcome = _text_bound().interpret(_completion(usage=None, choices=[]))
    assert outcome.kind == "unfinished_turn"
    assert outcome.assistant_message.turn == ()


def test_a_choice_with_no_finish_reason_is_unfinished_turn_carrying_the_partial_turn() -> None:
    """finish_reason reads None at runtime on a lenient snapshot, which is not a finished turn."""
    outcome = _text_bound().interpret(_lenient_completion(None))
    assert outcome.kind == "unfinished_turn"
    assert outcome.assistant_message.text == "hey"


def test_identity_reads_response_fields_and_adapter_stream_request_id() -> None:
    """ResponseIdentity combines response fields with AdapterStream.request_id()."""
    identity = _text_bound().identity_from_raw(
        _completion(usage=None, model="m-2026-01-01"), request_id="req-chat"
    )
    assert identity.response_id == "r1"
    assert identity.model_served == "m-2026-01-01"
    assert identity.request_id == "req-chat"


class _FakeSDKStream(AsyncStream[ChatCompletionChunk]):
    """Replay constructed chunks and exceptions without a connection."""

    def __init__(  # pyrefly: ignore[missing-super-call]
        self,
        replay: Sequence[ChatCompletionChunk | Exception],
        headers: dict[str, str] | None = None,
    ) -> None:
        self._replay = list(replay)
        self.response = httpx2.Response(
            200,
            headers=headers,
            request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
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
    """Build one chunk with a single choice. choices=[] is the usage-only trailing chunk."""
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

    return run_with_timeout(scenario())


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

    items, final = run_with_timeout(scenario())
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
    """Tool-call fragments yield deltas and one assembled ToolCall."""
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

    async def scenario() -> tuple[ProviderBilling | None, ProviderBilling | None]:
        before = stream.billing_reported()
        async for _item in stream.items():
            pass
        return before, stream.billing_reported()

    before, after = run_with_timeout(scenario())
    assert before is None
    assert after is not None
    assert after.billing.usage.input_tokens_total == 1000


def test_final_patches_the_tracked_usage_over_a_trailing_chunks_reset() -> None:
    """A usage-less chunk after the usage chunk resets the snapshot's usage. final() restores it."""
    stream = _stream([*_text_stream_chunks(), _chunk(choices=[])])

    async def scenario() -> ChatCompletion:
        async for _item in stream.items():
            pass
        return await stream.final()

    assert run_with_timeout(scenario()).usage == _usage_with_cache()


def test_final_before_items_are_exhausted_raises() -> None:
    """final() without a drained stream has nothing assembled to return."""
    with pytest.raises(StreamProtocolError, match="items"):
        _ = run_with_timeout(_stream(_text_stream_chunks()).final())


def test_a_stream_reports_the_request_id_header_of_the_response_it_reads() -> None:
    """Read a streamed request ID from the stream response."""
    assert _stream([], {"x-request-id": "req_stream"}).request_id() == "req_stream"
    assert _stream([]).request_id() is None


def test_a_mid_stream_bare_api_error_rewraps_as_a_status_error_on_the_live_response() -> None:
    """The rewrap carries the 200 status and the error's code, so parse_openai verdicts it.

    server_error is a transient code, so the mid-stream failure retries rather than failing the item.
    """
    bare_api_error = openai.APIError(
        "provider mid-stream error",
        httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
        body={"code": "server_error", "type": "insufficient_quota", "message": "boom"},
    )
    replay = [_chunk(delta={"role": "assistant", "content": "he"}), bare_api_error]
    with pytest.raises(openai.APIStatusError) as raised:
        _ = _collected_items(replay)
    assert raised.value.status_code == 200
    assert raised.value.code == "server_error"
    assert "provider mid-stream error" in raised.value.message
    assert raised.value.__cause__ is bare_api_error
    assert _adapter().parse(raised.value) == RetryThisOne(retry_after=None)


@pytest.mark.parametrize(
    "error",
    [
        connection_error(),
        openai.APIResponseValidationError(
            response=httpx2.Response(
                200, request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
            ),
            body=None,
        ),
    ],
    ids=["connection_error", "response_validation_error"],
)
def test_an_api_error_subclass_raised_mid_stream_propagates_untouched(error: Exception) -> None:
    """Only the bare APIError is the SSE error payload. Every subclass keeps its own meaning.

    APIResponseValidationError carries a 200 response like the rewrapped bare error, and still passes through.
    """
    with pytest.raises(type(error)) as raised:
        _ = _collected_items([_chunk(delta={"role": "assistant"}), error])
    assert raised.value is error


def _request_body_sent[OutputT](
    bind: Callable[[OpenAIChatCompletionsAdapter], BoundAdapter[OutputT]],
) -> object:
    """Open one stream through an offline transport and return the JSON body the SDK sent."""
    bodies: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(request.content)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"})

    client = AsyncOpenAI(
        api_key="offline",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    bound = bind(_adapter(client=client))
    request = bound.build_request([UserMessage(content="q")])
    assert not isinstance(request, InvalidRequest)
    _ = run_with_timeout(bound.open_stream(request))
    (body,) = bodies
    decoded_body: object = json.loads(body)
    return decoded_body


def test_the_text_request_streams_usage_and_sends_extra_body_by_reference() -> None:
    """stream=True is the request path and stream_options is what produces the trailing usage.

    The request carries the binding's extra_body as it is when sent, and a text binding sends no response_format.
    """
    extra_body = {"safety_identifier": "user-7"}

    def bind_then_edit_extra_body(adapter: OpenAIChatCompletionsAdapter) -> BoundAdapter[str]:
        bound = adapter.bind_text(_binding(extra_body=extra_body))
        extra_body["safety_identifier"] = "user-8"
        return bound

    assert _request_body_sent(bind_then_edit_extra_body) == {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "safety_identifier": "user-8",
    }


def test_structured_request_sends_the_non_strict_json_schema_response_format() -> None:
    """The structured binding asks for the caller's model schema and validates the text itself."""
    assert _request_body_sent(
        lambda adapter: adapter.bind_structured(_binding(), _StructuredReport)
    ) == {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "_StructuredReport",
                "schema": _StructuredReport.model_json_schema(),
                "strict": False,
            },
        },
    }


def test_a_built_request_renders_as_json_carrying_the_messages_and_no_omitted_field() -> None:
    """as_json holds the binding's precomputed fields and this call's messages after the prefix.

    An unstated temperature is absent from the request.
    """
    request = (
        _adapter()
        .bind_text(_binding(system_prompt="sys"))
        .build_request([UserMessage(content="hi")])
    )
    assert isinstance(request, _ChatCompletionsRequestParams)
    rendered = json.loads(request.as_json())
    assert rendered["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert "temperature" not in rendered["precomputed"]


def _conformance_message() -> dict[str, object]:
    """Return an assistant message whose reasoning_content is the key the installed SDK does not name.

    Replay preserves the extra reasoning_content key byte-for-byte.
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

        Excess cache counters make the derived uncached counter negative.
        """
        return _completion(
            usage=_usage(prompt_tokens_details={"cached_tokens": 900, "cache_write_tokens": 200})
        )

    @override
    def response_with_text(self, text: str) -> BaseModel:
        return _structured_completion(text)

    @override
    def response_with_reasoning(self) -> BaseModel:
        """Return a turn carrying reasoning_content beside its one text part."""
        return _completion(usage=_usage_with_cache(), message=_conformance_message())

    @override
    def response_with_raw_part(self) -> BaseModel:
        """Return message.content beside one custom message.tool_calls entry.

        openai 3.0.0 defines both fields on ChatCompletionMessage.
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
        """Return the shared OpenAI classification table."""
        return openai_sdk_errors_and_classifications()

    @override
    def sdk_errors_and_verdicts(self) -> Mapping[Exception, Verdict]:
        """Return the shared OpenAI verdict table."""
        return openai_sdk_errors_and_verdicts()
