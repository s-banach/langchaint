"""Test Anthropic Messages adapters with constructed SDK objects."""

import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from typing import NamedTuple, TypeIs, override

import anthropic
import anthropic.types as at
import httpx2
import pytest
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
)
from anthropic.lib.streaming import (
    AsyncMessageStream,
    ParsedContentBlockStopEvent,
    ParsedMessageStreamEvent,
)
from anthropic.types import ContentBlockParam, MessageParam, ParsedMessage
from anthropic.types.parsed_message import ParsedTextBlock
from pydantic import BaseModel

from langchaint import (
    LLM,
    AllowedToolsChoice,
    AssistantMessage,
    AudioPart,
    Billing,
    ImagePart,
    ImageUrlPart,
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
    Adapter,
    AdapterStream,
    Binding,
    ErrorClassification,
    InvalidRequest,
    ProviderBilling,
    RequestParams,
    ResponseIdentity,
    ResponseOutcome,
)
from langchaint.anthropic import (
    ANTHROPIC_BEDROCK_PRICING,
    BEDROCK_CROSS_REGION_MULTIPLIER,
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
    CacheTTL,
    _AnthropicProviderTools,
    _AnthropicRequestParams,
    _AnthropicStream,
    _assistant_content_blocks,
    _assistant_message_from,
    _BoundAnthropicStructured,
    _extra_body_with_temperature,
    _wire_messages,
    _wire_tool_choice,
    parse_anthropic,
)
from langchaint.anthropic.messages_adapter import (
    _billing_from_sdk_usage as _provider_billing_from_sdk_usage,
)
from langchaint.common.exceptions import TransientError
from langchaint.concurrency.shared_backoff import (
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    RetryThisOne,
    Verdict,
)
from langchaint.conformance import AdapterConformance
from langchaint.tools import ToolSchema
from tests.helpers import run_with_timeout


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

_MARK = {"type": "ephemeral"}
"""The cache_control marker every 5-minute breakpoint writes."""

_MARK_1H = {"type": "ephemeral", "ttl": "1h"}
"""The cache_control marker every 1-hour breakpoint writes."""


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


def _cache_marks(wire: Sequence[MessageParam]) -> list[list[object]]:
    """Return each wire block's cache_control marker, None for an unmarked block, grouped by message."""
    return [[block.get("cache_control") for block in _content_blocks(message)] for message in wire]


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


def _adapter(
    *,
    client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle | None = None,
    provider_name: str = "anthropic",
    pricing: AnthropicPricingTable = _PRICING,
    cache_ttl: CacheTTL = "5m",
) -> AnthropicMessagesAdapter:
    """Build an adapter over a keyless client unless a test supplies one, valid because no request leaves."""
    return AnthropicMessagesAdapter(
        client=AsyncAnthropic(api_key="test") if client is None else client,
        model="m",
        pricing=pricing,
        provider_name=provider_name,
        cache_ttl=cache_ttl,
    )


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
        max_completion_tokens=None,
        reasoning_level="high",
        temperature=temperature,
        automatic_cache_breakpoints=automatic_cache_breakpoints,
        extra_body=extra_body,
    )


_UNSET_BINDING = Binding(
    system_prompt=None,
    tool_schemas=(),
    provider_executed_tools=(),
    tool_choice="auto",
    parallel_tool_calls=True,
    max_completion_tokens=None,
    reasoning_level=None,
    temperature=None,
    automatic_cache_breakpoints=False,
)
"""A binding that states no optional request field."""


def _precomputed_with_provider_tools(*tool_types: str) -> _AnthropicProviderTools:
    """Validate provider-executed tools of the given types under the default adapter."""
    return (
        _adapter()
        ._precompute_fields(
            _binding(
                system_prompt=None,
                tool_schemas=(),
                provider_executed_tools=tuple({"type": tool_type} for tool_type in tool_types),
                automatic_cache_breakpoints=False,
            )
        )
        .provider_tools
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
    assert billing.cache_write_usd_per_million_tokens == pytest.approx(5.25)
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


class _ToolCostCase(NamedTuple):
    """One provider-executed tool configuration, the server-tool counters a response reports, and the cost."""

    provider_executed_tool_types: tuple[str, ...]
    server_tool_counters: Mapping[str, int]
    expected_cost_in_usd: float
    billing_complete: bool = True


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _ToolCostCase((), {"web_search_requests": 2}, 0.02),
            id="web_search_at_the_cataloged_rate",
        ),
        pytest.param(
            _ToolCostCase(
                ("web_fetch_20260209", "tool_search_tool_bm25", "code_execution_20260120"),
                {"web_fetch_requests": 2, "tool_search_requests": 3, "code_execution_requests": 4},
                0.0,
            ),
            id="configured_zero_fee_tools",
        ),
        pytest.param(
            _ToolCostCase((), {"web_fetch_requests": 1}, float("nan")),
            id="unconfigured_web_fetch",
        ),
        pytest.param(
            _ToolCostCase((), {"tool_search_requests": 1}, float("nan")),
            id="unconfigured_tool_search",
        ),
        pytest.param(
            _ToolCostCase((), {"code_execution_requests": 1}, float("nan")),
            id="unconfigured_code_execution",
        ),
        pytest.param(
            _ToolCostCase((), {"future_requests": 0}, 0.0),
            id="unexpected_counter_at_zero",
        ),
        pytest.param(
            _ToolCostCase((), {"future_requests": 2}, float("nan")),
            id="unexpected_counter_fired",
        ),
        pytest.param(
            _ToolCostCase(("web_search_20260318",), {}, float("nan"), billing_complete=False),
            id="truncated_web_search_snapshot",
        ),
    ],
)
def test_provider_executed_tool_cost(case: _ToolCostCase) -> None:
    """Web search prices per invocation, and web fetch, tool search, and exempt code execution add no fee.

    A nonzero counter no configured tool accounts for is an unpriced charge, so the cost is NaN.
    A partial usage snapshot cannot prove the final web-search count, so the cost is NaN.
    """
    usage_raw = at.Usage(
        input_tokens=1,
        output_tokens=1,
        server_tool_use=at.ServerToolUsage.model_validate({
            "web_search_requests": 0,
            "web_fetch_requests": 0,
            **case.server_tool_counters,
        }),
    )
    usage = _billing_from_sdk_usage(
        usage_raw,
        _PRICING,
        provider_tools=_precomputed_with_provider_tools(*case.provider_executed_tool_types),
        billing_complete=case.billing_complete,
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(
        case.expected_cost_in_usd, nan_ok=True
    )


@pytest.mark.parametrize("rate", [None, True, float("nan"), float("inf"), -0.01])
def test_configured_anthropic_web_search_rate_must_be_usable(rate: float | None) -> None:
    """A configured search rejects an unusable caller rate before requests."""
    adapter = _adapter(pricing=replace(_PRICING, web_search_usd_per_invocation=rate))
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


def test_model_cache_ttl_reaches_the_system_cache_marker() -> None:
    """Check the system cache marker built through each model factory."""
    for llm in (
        Anthropic(client=AsyncAnthropic(api_key="test")).model("claude-sonnet-5", cache_ttl="1h"),
        AnthropicBedrock(aws_region="us-east-1").model(
            "anthropic.claude-sonnet-5", cache_ttl="1h"
        ),
    ):
        bound = llm.adapter.bind_text(
            _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
        )
        request = bound.build_request([UserMessage(content="q")])
        assert isinstance(request, _AnthropicRequestParams)
        assert json.loads(request.as_json())["precomputed"]["system"][0]["cache_control"] == (
            _MARK_1H
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("model_context_window_exceeded", "context_window_exceeded"),
        ("pause_turn", "other"),
        (None, "other"),
    ],
)
def test_stop_reason_mapping(raw: at.StopReason | None, expected: str) -> None:
    """Check translated stop reasons alongside the preserved partial text."""
    message = _message_snapshot(raw, [at.TextBlock(type="text", text="partial text")])
    result = (
        _adapter()
        .bind_text(_binding(system_prompt=None, tool_schemas=(), automatic_cache_breakpoints=True))
        .interpret(message)
    )
    assert result.kind == "adapter_result"
    assert result.stop_reason == expected
    assert result.output == "partial text"


def test_text_output_concatenates_the_text_blocks() -> None:
    """The text binding's output joins every text block, and tool_use passes as the stop reason."""
    message = _message_with_content([
        at.TextBlock(type="text", text="hello "),
        at.TextBlock(type="text", text="world"),
    ])
    result = (
        _adapter()
        .bind_text(_binding(system_prompt=None, tool_schemas=(), automatic_cache_breakpoints=True))
        .interpret(message)
    )
    assert result.kind == "adapter_result"
    assert result.output == "hello world"
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


def test_assistant_blocks_convert_to_parts_and_back() -> None:
    """Each SDK block becomes one TurnPart in order, and each TurnPart converts back to its wire block.

    A server tool block becomes RawPart and replays without the SDK model's unset fields.
    """
    thinking_raw = {"type": "thinking", "thinking": "check first", "signature": "sig-1"}
    server_tool_raw = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "langchaint"},
    }
    assistant_message = _assistant_message_from(
        _message_with_content([
            at.ThinkingBlock(type="thinking", thinking="check first", signature="sig-1"),
            at.TextBlock(type="text", text="hello"),
            at.ToolUseBlock(
                type="tool_use", id="tu_1", name="get_weather", input={"city": "Nairobi"}
            ),
            at.ServerToolUseBlock(
                type="server_tool_use",
                id="srvtoolu_1",
                name="web_search",
                input={"query": "langchaint"},
            ),
        ])
    )
    assert assistant_message.turn == (
        ReasoningPart(raw=thinking_raw, text="check first"),
        TextPart(text="hello"),
        ToolCall(id="tu_1", name="get_weather", args_json='{"city": "Nairobi"}'),
        RawPart(raw=server_tool_raw),
    )
    assert _assistant_content_blocks(assistant_message) == [
        thinking_raw,
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Nairobi"}},
        server_tool_raw,
    ]


@pytest.mark.parametrize(
    ("block", "expected_raw"),
    [
        pytest.param(
            at.ThinkingBlock(type="thinking", thinking="", signature="sig"),
            {"type": "thinking", "thinking": "", "signature": "sig"},
            id="empty_thinking",
        ),
        pytest.param(
            at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes"),
            {"type": "redacted_thinking", "data": "opaque-bytes"},
            id="redacted_thinking",
        ),
    ],
)
def test_text_free_reasoning_reads_as_none_and_replays_by_its_type_key(
    block: at.ContentBlock, expected_raw: dict[str, str]
) -> None:
    """Empty and redacted thinking have text None, the one text-free value across adapters.

    The dump replays unchanged, and its type key routes it on the wire.
    """
    assistant_message = _assistant_message_from(_message_with_content([block]))
    assert assistant_message.turn == (ReasoningPart(raw=expected_raw, text=None),)
    assert _assistant_content_blocks(assistant_message) == [expected_raw]


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """A foreign ReasoningPart sends ReasoningPart.raw unchanged for provider validation."""
    raw = {"type": "reasoning", "id": "rs_1"}
    assistant_message = AssistantMessage(turn=(ReasoningPart(raw=raw), TextPart(text="hi")))
    assert _assistant_content_blocks(assistant_message) == [
        raw,
        {"type": "text", "text": "hi"},
    ]


def test_wire_messages_groups_consecutive_tool_results() -> None:
    """Consecutive ToolMessages collapse into one user message of tool_result blocks.

    The group goes out before the next user or assistant message, so each tool_use is answered next.
    """
    messages = [
        UserMessage(content="hi"),
        AssistantMessage(
            turn=(
                TextPart(text="checking"),
                ToolCall(id="tu_1", name="t", args_json='{"a": 1}'),
                ToolCall(id="tu_2", name="t", args_json='{"a": 2}'),
            ),
        ),
        ToolMessage(tool_call_id="tu_1", content="r1", is_error=False),
        ToolMessage(tool_call_id="tu_2", content="r2", is_error=True),
        UserMessage(content="and then?"),
        AssistantMessage(turn=(ToolCall(id="tu_3", name="t", args_json='{"a": 3}'),)),
        ToolMessage(tool_call_id="tu_3", content="r3"),
        AssistantMessage(turn=(TextPart(text="done"),)),
    ]
    wire = _wire_messages(
        messages, automatic_cache_breakpoints=False, cache_ttl="5m", message_mark_budget=4
    )
    assert [
        (
            message["role"],
            [(block["type"], block.get("is_error")) for block in _content_blocks(message)],
        )
        for message in wire
    ] == [
        ("user", [("text", None)]),
        ("assistant", [("text", None), ("tool_use", None), ("tool_use", None)]),
        ("user", [("tool_result", False), ("tool_result", True)]),
        ("user", [("text", None)]),
        ("assistant", [("tool_use", None)]),
        ("user", [("tool_result", False)]),
        ("assistant", [("text", None)]),
    ]


class _MarkCase(NamedTuple):
    """One message sequence, the caching parameters it converts under, and the markers each block gets."""

    messages: tuple[Message, ...]
    message_mark_budget: int
    expected_marks: list[list[object]]
    automatic_cache_breakpoints: bool = False
    cache_ttl: CacheTTL = "5m"


def _marked_texts(count: int) -> tuple[TextPart, ...]:
    """Return text parts that each set cache_breakpoint."""
    return tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(count))


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _MarkCase(
                (
                    ToolMessage(tool_call_id="tu_1", content="r1"),
                    ToolMessage(tool_call_id="tu_2", content="r2", is_error=True),
                ),
                2,
                [[None, _MARK]],
                automatic_cache_breakpoints=True,
            ),
            id="automatic_marker_on_the_last_block_only",
        ),
        pytest.param(
            _MarkCase(
                (
                    AssistantMessage(
                        turn=(
                            TextPart(text="t"),
                            ReasoningPart(
                                raw={"type": "thinking", "thinking": "x", "signature": "s"}
                            ),
                        )
                    ),
                ),
                2,
                [[None, None]],
                automatic_cache_breakpoints=True,
            ),
            id="no_automatic_marker_on_a_thinking_last_block",
        ),
        pytest.param(
            _MarkCase(
                (UserMessage(content="hi"), ToolMessage(tool_call_id="tu_1", content="r1")),
                4,
                [[None], [None]],
            ),
            id="no_automatic_marker_when_disabled",
        ),
        pytest.param(
            _MarkCase(
                (
                    UserMessage(
                        content=(
                            TextPart(text="shared context", cache_breakpoint=True),
                            TextPart(text="question"),
                        )
                    ),
                ),
                4,
                [[_MARK, None]],
            ),
            id="marked_user_text_part",
        ),
        pytest.param(
            _MarkCase(
                (
                    UserMessage(
                        content=(
                            ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
                        )
                    ),
                ),
                4,
                [[_MARK]],
            ),
            id="marked_user_image_part",
        ),
        pytest.param(
            _MarkCase(
                (
                    ToolMessage(
                        tool_call_id="tu_1",
                        content=(TextPart(text="a"), TextPart(text="b", cache_breakpoint=True)),
                    ),
                ),
                4,
                [[_MARK]],
            ),
            id="marked_last_tool_part_marks_its_tool_result",
        ),
        pytest.param(
            _MarkCase(
                (UserMessage(content=_marked_texts(5)),), 4, [[None, _MARK, _MARK, _MARK, _MARK]]
            ),
            id="budget_keeps_the_latest_marks",
        ),
        pytest.param(
            _MarkCase(
                (UserMessage(content=_marked_texts(3)), UserMessage(content="question")),
                2,
                [[None, _MARK, _MARK], [_MARK]],
                automatic_cache_breakpoints=True,
            ),
            id="automatic_marker_beside_the_latest_marks",
        ),
        pytest.param(
            _MarkCase(
                (
                    UserMessage(content=(TextPart(text="oldest", cache_breakpoint=True),)),
                    ToolMessage(
                        tool_call_id="tu_1",
                        content=(TextPart(text="mid", cache_breakpoint=True),),
                    ),
                    UserMessage(content=(TextPart(text="latest", cache_breakpoint=True),)),
                ),
                2,
                [[None], [_MARK], [_MARK]],
            ),
            id="budget_counts_across_message_kinds",
        ),
        pytest.param(
            _MarkCase(
                (UserMessage(content=_marked_texts(1)),),
                2,
                [[_MARK]],
                automatic_cache_breakpoints=True,
            ),
            id="explicit_and_automatic_marks_coincide",
        ),
        pytest.param(
            _MarkCase((UserMessage(content=_marked_texts(1)),), 0, [[None]]),
            id="zero_budget_writes_no_marks",
        ),
        pytest.param(
            _MarkCase(
                (UserMessage(content=_marked_texts(1)), UserMessage(content="question")),
                2,
                [[_MARK_1H], [_MARK_1H]],
                automatic_cache_breakpoints=True,
                cache_ttl="1h",
            ),
            id="one_hour_ttl",
        ),
    ],
)
def test_wire_messages_places_cache_markers(case: _MarkCase) -> None:
    """The latest explicit marks within message_mark_budget and the automatic marker carry cache_control.

    A user part marks its own block, and a ToolMessage's marked last part marks its tool_result block.
    The automatic marker goes on the last block unless that block is thinking, which carries no cache_control.
    """
    wire = _wire_messages(
        case.messages,
        automatic_cache_breakpoints=case.automatic_cache_breakpoints,
        cache_ttl=case.cache_ttl,
        message_mark_budget=case.message_mark_budget,
    )
    assert _cache_marks(wire) == case.expected_marks


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
                "data": "cG5n",
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
    assert _content_blocks(wire[0]) == [{**expected_unmarked_block, "cache_control": _MARK}]
    tool_result = _content_blocks(wire[1])[0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("content") == [expected_unmarked_block]
    assert tool_result.get("cache_control") == _MARK


@pytest.mark.parametrize(
    ("messages", "reason_fragment"),
    [
        pytest.param(
            (UserMessage(content=(AudioPart(data=b"wav", media_type="audio/wav"),)),),
            "AudioPart inside UserMessage",
            id="user_audio",
        ),
        pytest.param(
            (
                ToolMessage(
                    tool_call_id="tu_1",
                    content=(AudioPart(data=b"wav", media_type="audio/wav"),),
                ),
            ),
            "AudioPart inside ToolMessage",
            id="tool_audio",
        ),
        pytest.param(
            (UserMessage(content=(ImagePart(data=b"x", media_type="image/tiff"),)),),
            "image/tiff",
            id="user_image_media_type",
        ),
        pytest.param(
            (
                ToolMessage(
                    tool_call_id="tu_1",
                    content=(ImagePart(data=b"x", media_type="image/tiff"),),
                ),
            ),
            "image/tiff",
            id="tool_image_media_type",
        ),
        pytest.param(
            (
                ToolMessage(
                    tool_call_id="tu_1",
                    content=(TextPart(text="a", cache_breakpoint=True), TextPart(text="b")),
                ),
            ),
            "last part",
            id="marked_non_last_tool_part",
        ),
        pytest.param(
            (
                AssistantMessage(turn=(ToolCall(id="c1", name="f", args_json="not json"),)),
                ToolMessage(tool_call_id="c1", content="ok"),
            ),
            "args_json",
            id="unparseable_args_json",
        ),
        pytest.param(
            (
                UserMessage(content="q"),
                AssistantMessage(turn=(RawPart(raw={"parts": [{"text": "from elsewhere"}]}),)),
            ),
            "type key",
            id="stored_payload_naming_no_type",
        ),
    ],
)
def test_build_request_reports_an_unsendable_sequence_as_invalid_request(
    messages: tuple[Message, ...], reason_fragment: str
) -> None:
    """An unsendable Sequence[Message] reaches build_request's caller as the InvalidRequest variant.

    Nothing is sent: the retry loop takes this answer before its first attempt.
    A marked non-last ToolMessage part is rejected instead of silently moving the cache boundary.
    """
    request = _structured_bound().build_request(messages)
    assert isinstance(request, InvalidRequest)
    assert reason_fragment in request.reason


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
        "uses_top_level_cache_control": True,
        "cache_ttl": "1h",
        "default_max_completion_tokens": 8192,
        "inference_geo": "us",
        "service_tier": "standard_only",
    }


def test_unset_binding_fields_stay_at_the_omit_sentinel() -> None:
    """An unstated optional field keeps the SDK's omit sentinel, which leaves the provider default."""
    precomputed_fields = _adapter()._precompute_fields(_UNSET_BINDING)
    for field_value in (
        precomputed_fields.system,
        precomputed_fields.tools,
        precomputed_fields.tool_choice,
        precomputed_fields.output_config,
        precomputed_fields.thinking,
        precomputed_fields.temperature,
        precomputed_fields.service_tier,
        precomputed_fields.inference_geo,
        precomputed_fields.cache_control,
    ):
        assert isinstance(field_value, anthropic.Omit)


def test_request_passes_reasoning_level_through() -> None:
    """A value outside anthropic's own effort literal ("minimal") reaches the request unchanged."""
    precomputed_fields = _adapter()._precompute_fields(
        replace(_UNSET_BINDING, reasoning_level="minimal")
    )
    assert precomputed_fields.output_config == {"effort": "minimal"}
    assert precomputed_fields.thinking == {"type": "adaptive"}


def test_open_stream_sends_the_built_request() -> None:
    """The request body holds the adapter's settings, the binding's precomputed fields, and the messages.

    A bound temperature travels in extra_body because the SDK stream takes it there.
    """
    sent_bodies: list[object] = []

    def reject_after_recording(request: httpx2.Request) -> httpx2.Response:
        """Record the request body, then end the call with a 400 so no stream is read."""
        sent_bodies.append(json.loads(request.content))
        return httpx2.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error"}}
        )

    adapter = AnthropicMessagesAdapter(
        client=AsyncAnthropic(
            api_key="test",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reject_after_recording)),
        ),
        model="m",
        pricing=_PRICING,
        provider_name="anthropic",
        cache_ttl="1h",
        service_tier="standard_only",
        inference_geo="us",
    )
    bound = adapter.bind_text(
        _binding(
            system_prompt="sys",
            tool_schemas=(),
            automatic_cache_breakpoints=True,
            temperature=0.2,
        )
    )
    request = bound.build_request([UserMessage(content="q")])
    assert not isinstance(request, InvalidRequest)
    with pytest.raises(anthropic.BadRequestError):
        _ = run_with_timeout(bound.open_stream(request))
    assert sent_bodies == [
        {
            "model": "m",
            "max_tokens": 4096,
            "system": [{"type": "text", "text": "sys", "cache_control": _MARK_1H}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
            "service_tier": "standard_only",
            "inference_geo": "us",
            "cache_control": _MARK_1H,
            "temperature": 0.2,
            "stream": True,
        }
    ]


def test_extra_body_keeps_caller_fields_by_reference_beside_a_temperature() -> None:
    """Without a temperature the caller's mapping is sent itself, and with one it is read through."""
    caller_fields: dict[str, object] = {"top_k": 5}

    def extra_body_bound_with(temperature: float | None) -> Mapping[str, object] | None:
        return _extra_body_with_temperature(
            _adapter()._precompute_fields(
                _binding(
                    system_prompt=None,
                    tool_schemas=(),
                    automatic_cache_breakpoints=True,
                    extra_body=caller_fields,
                    temperature=temperature,
                )
            )
        )

    assert extra_body_bound_with(None) is caller_fields
    with_temperature = extra_body_bound_with(0.2)
    assert with_temperature is not None
    assert dict(with_temperature) == {"temperature": 0.2, "top_k": 5}
    caller_fields["top_k"] = 6
    caller_fields["temperature"] = 0.7
    assert dict(with_temperature) == {"temperature": 0.2, "top_k": 6}


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
    assert tools[1].get("cache_control") == _MARK
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
    provider_tools = _precomputed_with_provider_tools(web_tool_type, code_execution_type)
    assert provider_tools.code_execution_exempt


@pytest.mark.parametrize(
    "provider_tool",
    [
        {"type": tool_type}
        for tool_type in (
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
        )
    ]
    + [{}, {"type": 1}],
)
def test_every_unlisted_anthropic_provider_type_is_rejected(
    provider_tool: Mapping[str, object],
) -> None:
    """Messages rejects every reviewed client-executed or unaudited `type`, and a missing or non-string one."""
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
        _ = _precomputed_with_provider_tools(code_execution_type)


def test_anthropic_bedrock_rejects_provider_executed_tools() -> None:
    """Anthropic pricing does not establish Bedrock provider-tool billing."""
    adapter = _adapter(
        client=AsyncAnthropicBedrock(aws_region="us-east-1"), provider_name="aws.bedrock"
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


def test_provider_executed_cache_markers_reduce_the_message_budget() -> None:
    """Provider-executed cache markers count toward Anthropic's request limit."""
    provider_tools = tuple(
        {
            "type": "web_search_20250305",
            "name": f"web_search_{index}",
            "cache_control": _MARK,
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


def test_the_system_block_marker_follows_automatic_cache_breakpoints() -> None:
    """The automatic system marker is one of the four request markers, as is the automatic message marker."""
    cached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    assert _block_list(cached.system)[0].get("cache_control") == _MARK
    assert cached.message_mark_budget == 2
    uncached = _adapter()._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=False)
    )
    assert "cache_control" not in _block_list(uncached.system)[0]
    assert uncached.message_mark_budget == 4


def test_request_marks_last_tool_only_without_a_system_prompt() -> None:
    """The prefix breakpoint sits on the last tool only when no system prompt follows."""
    schemas = _tool_schemas()
    adapter = _adapter(cache_ttl="1h")
    without_system = adapter._precompute_fields(
        _binding(system_prompt=None, tool_schemas=schemas, automatic_cache_breakpoints=True)
    )
    assert _block_list(without_system.tools)[-1].get("cache_control") == _MARK_1H
    with_system = adapter._precompute_fields(
        _binding(system_prompt="sys", tool_schemas=schemas, automatic_cache_breakpoints=True)
    )
    assert "cache_control" not in _block_list(with_system.tools)[-1]


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


_REDACTED_BLOCK_STOP_EVENT = ParsedContentBlockStopEvent(
    type="content_block_stop",
    index=1,
    content_block=at.RedactedThinkingBlock(type="redacted_thinking", data="opaque-bytes"),
)
"""The stop event closing a redacted thinking block at index 1."""


def _collected_items(
    replay_events: Sequence[ParsedMessageStreamEvent],
    message_snapshot: ParsedMessage[None] | None = None,
) -> list[StreamItem]:
    """Drain the translated items into a list. None means a bare end_turn snapshot."""
    snapshot = message_snapshot if message_snapshot is not None else _message_snapshot("end_turn")

    async def scenario() -> list[StreamItem]:
        adapter_stream = _anthropic_stream(replay_events, snapshot)
        return [item async for item in adapter_stream.items()]

    return run_with_timeout(scenario())


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


@pytest.mark.parametrize(
    ("replay_events", "expected_items"),
    [
        pytest.param(
            [
                _thinking_delta_event("First, ", 0),
                _thinking_delta_event("water evaporates.", 0),
                _thinking_block_stop_event("First, water evaporates.", 0),
                _thinking_delta_event("Then it ", 1),
                _thinking_delta_event("condenses.", 1),
                _thinking_block_stop_event("Then it condenses.", 1),
            ],
            [
                ReasoningDelta(text="First, "),
                ReasoningDelta(text="water evaporates."),
                ReasoningDelta(text="\n\n"),
                ReasoningDelta(text="Then it "),
                ReasoningDelta(text="condenses."),
            ],
            id="blank_line_between_thinking_blocks",
        ),
        pytest.param(
            [
                _thinking_delta_event("thought it over", 0),
                _thinking_block_stop_event("thought it over", 0),
                _text_delta_event("hey", 1),
            ],
            [ReasoningDelta(text="thought it over"), "hey"],
            id="no_separator_after_the_last_thinking_block",
        ),
        pytest.param(
            [
                _thinking_delta_event("", 0),
                _thinking_block_stop_event("", 0),
                _thinking_delta_event("thought it over", 1),
            ],
            [ReasoningDelta(text="thought it over")],
            id="empty_delta_dropped_and_arms_no_separator",
        ),
        pytest.param(
            [
                _thinking_delta_event("First.", 0),
                _thinking_block_stop_event("First.", 0),
                _thinking_delta_event("", 1),
                _thinking_delta_event("Then.", 1),
            ],
            [
                ReasoningDelta(text="First."),
                ReasoningDelta(text="\n\n"),
                ReasoningDelta(text="Then."),
            ],
            id="empty_delta_keeps_the_pending_separator",
        ),
        pytest.param(
            [
                _thinking_delta_event("First.", 0),
                _thinking_block_stop_event("First.", 0),
                _REDACTED_BLOCK_STOP_EVENT,
                _thinking_delta_event("Then.", 2),
            ],
            [
                ReasoningDelta(text="First."),
                ReasoningDelta(text="\n\n"),
                ReasoningDelta(text="Then."),
            ],
            id="redacted_block_adds_no_text_or_separator",
        ),
        pytest.param(
            [
                _thinking_delta_event("", 0),
                _thinking_block_stop_event("", 0),
                _REDACTED_BLOCK_STOP_EVENT,
                _thinking_delta_event("Then.", 2),
            ],
            [ReasoningDelta(text="Then.")],
            id="redacted_block_after_a_text_free_block_adds_no_separator",
        ),
    ],
)
def test_thinking_streams_as_reasoning_deltas_separated_by_a_blank_line(
    replay_events: list[ParsedMessageStreamEvent], expected_items: list[StreamItem]
) -> None:
    """A blank line separates thinking blocks that streamed text, placed before the next block's text.

    Empty deltas and redacted blocks stream nothing, so they neither add nor consume a separator.
    """
    assert _collected_items(replay_events) == expected_items


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


@pytest.mark.parametrize(
    ("client", "uses_top_level_cache_control"),
    [
        (AsyncAnthropic(api_key="test"), True),
        (AsyncAnthropicBedrockMantle(aws_region="us-east-1"), True),
        (AsyncAnthropicBedrock(aws_region="us-east-1"), False),
    ],
    ids=["anthropic", "bedrock_mantle", "bedrock_legacy"],
)
def test_automatic_cache_breakpoints_select_final_caching_by_client(
    client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
    *,
    uses_top_level_cache_control: bool,
) -> None:
    """Direct and Mantle clients use top-level caching, while legacy Bedrock marks the final block."""
    adapter = _adapter(
        client=client,
        provider_name=("anthropic" if isinstance(client, AsyncAnthropic) else "aws.bedrock"),
        cache_ttl="1h",
    )
    bound = adapter.bind_text(
        _binding(system_prompt="sys", tool_schemas=(), automatic_cache_breakpoints=True)
    )
    request = bound.build_request([
        UserMessage(
            content=tuple(
                TextPart(text=str(index), cache_breakpoint=index < 4) for index in range(5)
            )
        )
    ])
    assert isinstance(request, _AnthropicRequestParams)
    final_block_mark = None if uses_top_level_cache_control else _MARK_1H
    assert _cache_marks(request.messages) == [[None, None, _MARK_1H, _MARK_1H, final_block_mark]]
    if uses_top_level_cache_control:
        assert request.precomputed.cache_control == _MARK_1H
    else:
        assert isinstance(request.precomputed.cache_control, anthropic.Omit)


def test_identity_reads_the_messages_own_id_and_served_model_beside_the_request_id() -> None:
    """The message supplies its id and served model, and the AdapterStream supplies the request id."""
    identity = _structured_bound().identity_from_raw(
        _message_with_content([]), request_id="req_anthropic"
    )
    assert identity == ResponseIdentity(
        model_served="claude-sonnet-5", response_id="msg_1", request_id="req_anthropic"
    )


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


def test_the_structured_request_merges_the_schema_into_the_output_config() -> None:
    """The response_format's JSON schema, as anthropic's SDK transforms it, joins the binding's effort."""
    request = _structured_bound().build_request([UserMessage(content="q")])
    assert isinstance(request, _AnthropicRequestParams)
    assert request.precomputed.output_config == {
        "effort": "high",
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "description": "The response_format the structured bind path parses into.",
                "title": "_StructuredReport",
                "properties": {
                    "city": {"type": "string", "title": "City"},
                    "celsius": {"type": "integer", "title": "Celsius"},
                },
                "additionalProperties": False,
                "required": ["city", "celsius"],
            },
        },
    }


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


@pytest.mark.parametrize(
    ("text", "stop_reason", "expected_kind", "expected_output"),
    [
        pytest.param(
            _REPORT_JSON,
            "end_turn",
            "adapter_result",
            _StructuredReport(city="Nairobi", celsius=25),
            id="valid_text",
        ),
        pytest.param(None, "end_turn", "empty_turn", None, id="end_turn_without_text"),
        pytest.param(None, "refusal", "refusal", None, id="refusal"),
        pytest.param(None, "max_tokens", "max_completion_tokens_exceeded", None, id="max_tokens"),
        pytest.param(
            '{"city": "Nair',
            "max_tokens",
            "max_completion_tokens_exceeded",
            None,
            id="text_cut_mid_json_at_max_tokens",
        ),
        pytest.param(
            None,
            "model_context_window_exceeded",
            "context_window_exceeded",
            None,
            id="context_window_exceeded",
        ),
        pytest.param(None, None, "unfinished_turn", None, id="no_stop_reason"),
        pytest.param(None, "pause_turn", "unfinished_turn", None, id="pause_turn"),
        pytest.param(
            "partial thought",
            "pause_turn",
            "unfinished_turn",
            None,
            id="pause_turn_ahead_of_schema_violation",
        ),
        pytest.param(None, "tool_use", "adapter_result", None, id="tool_use"),
        pytest.param(
            "let me look that up", "tool_use", "adapter_result", None, id="tool_use_with_prose"
        ),
    ],
)
def test_structured_outcome_by_text_and_stop_reason(
    text: str | None,
    stop_reason: at.StopReason | None,
    expected_kind: str,
    expected_output: _StructuredReport | None,
) -> None:
    """Valid text validates into the instance, and otherwise the stop reason names the outcome.

    A null stop reason or pause_turn is not a finished turn, so it is unfinished even when text failed validation.
    A tool_use turn parses no instance and nothing went wrong, so its output is None even beside prose.
    """
    outcome = _structured_parse(_structured_message(text, stop_reason=stop_reason))
    assert outcome.kind == expected_kind
    if outcome.kind == "adapter_result":
        assert outcome.output == expected_output


def test_an_unfinished_structured_turn_names_the_stop_reason() -> None:
    """The reason quotes anthropic's own stop reason."""
    outcome = _structured_parse(_structured_message(None, stop_reason="pause_turn"))
    assert outcome.kind == "unfinished_turn"
    assert "pause_turn" in outcome.reason


def test_parse_anthropic_counts_each_fallthrough_and_no_listed_row() -> None:
    """An unlisted status lands one tagged count, even when its error type picks the verdict.

    A listed status leaves the counter alone.
    """
    before = dict(PARSE_FALLTHROUGH_COUNTS)
    _ = parse_anthropic(_status_error(anthropic.RateLimitError, 429))
    _ = parse_anthropic(_status_error(anthropic.APIStatusError, 408))
    assert dict(PARSE_FALLTHROUGH_COUNTS) == before
    for failure, tag in (
        (_status_error(anthropic.APIStatusError, 599), "status=599 type=None"),
        (
            _status_error(anthropic.APIStatusError, 200, error_type="api_error"),
            "status=200 type=api_error",
        ),
    ):
        _ = parse_anthropic(failure)
        assert PARSE_FALLTHROUGH_COUNTS[tag] == before.get(tag, 0) + 1


def test_request_id_from_error_reads_the_sdk_errors_own_header_and_nothing_else() -> None:
    """The override reports the header the SDK read off the error response, None for any other error.

    Anthropic sends the request ID in request-id.
    Missing headers return None.
    """
    adapter = _adapter()
    with_header = _status_error(anthropic.RateLimitError, 429, {"request-id": "req_429"})
    assert adapter.request_id_from_error(with_header) == "req_429"
    assert adapter.request_id_from_error(_status_error(anthropic.RateLimitError, 429)) is None
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
        ("anthropic.claude-opus-5", AsyncAnthropicBedrockMantle),
        ("anthropic.claude-opus-4-8", AsyncAnthropicBedrockMantle),
        ("anthropic.claude-haiku-4-5", AsyncAnthropicBedrockMantle),
        ("us.anthropic.claude-opus-5", AsyncAnthropicBedrockMantle),
        ("us-gov.anthropic.claude-sonnet-5", AsyncAnthropicBedrockMantle),
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


def test_exact_prefixed_catalog_entry_uses_its_own_bedrock_pricing_object() -> None:
    """A verbatim catalog entry already states its regional rates, so no premium applies.

    The Bedrock catalog remains independent from direct Anthropic pricing.
    """
    adapter = _anthropic_adapter_of(
        AnthropicBedrock(aws_region="us-east-1").model("us.anthropic.claude-opus-4-6-v1")
    )
    assert adapter.pricing is ANTHROPIC_BEDROCK_PRICING["us.anthropic.claude-opus-4-6-v1"]


@pytest.mark.parametrize("prefix", sorted(BEDROCK_CROSS_REGION_MULTIPLIER))
def test_prefixed_bedrock_model_applies_its_premium_by_default(prefix: str) -> None:
    """Every cross-region prefix multiplies the unprefixed token rates by its multiplier."""
    adapter = _anthropic_adapter_of(
        AnthropicBedrock(aws_region="us-east-1").model(f"{prefix}.anthropic.claude-sonnet-5")
    )
    base = ANTHROPIC_BEDROCK_PRICING["anthropic.claude-sonnet-5"]
    multiplier = BEDROCK_CROSS_REGION_MULTIPLIER[prefix]
    assert adapter.pricing == replace(base, standard=base.standard.multiplied(multiplier))


@pytest.mark.parametrize("prefix", sorted(BEDROCK_CROSS_REGION_MULTIPLIER))
def test_prefixed_bedrock_model_uses_the_unprefixed_pricing_object_when_disabled(
    prefix: str,
) -> None:
    """`apply_cross_region_premium=False` resolves to the unprefixed catalog table itself."""
    adapter = _anthropic_adapter_of(
        AnthropicBedrock(aws_region="us-east-1", apply_cross_region_premium=False).model(
            f"{prefix}.anthropic.claude-sonnet-5"
        )
    )
    assert adapter.pricing is ANTHROPIC_BEDROCK_PRICING["anthropic.claude-sonnet-5"]


def test_pricing_table_multiplied_scales_every_tier_and_keeps_modifiers() -> None:
    """`multiplied` scales standard, priority, and batch rates and preserves the other fields."""
    table = AnthropicPricingTable(
        standard=_PRICING.standard,
        priority=_PRICING.standard,
        batch=_PRICING.standard,
        inference_geo_us_multiplier=1.1,
        web_search_usd_per_invocation=0.01,
    )
    scaled = table.multiplied(2.0)
    doubled = _PRICING.standard.multiplied(2.0)
    assert scaled == replace(table, standard=doubled, priority=doubled, batch=doubled)


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
        {"type": "text", "text": "stable instructions", "cache_control": _MARK},
        {"type": "text", "text": "semi-stable context", "cache_control": _MARK},
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
        {"type": "text", "text": "stable", "cache_control": _MARK},
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

    before, after = run_with_timeout(scenario())
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
    def response_with_text(self, text: str) -> BaseModel:
        return _structured_message(text)

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
            _status_error(anthropic.AuthenticationError, 401): "auth",
            _status_error(anthropic.PermissionDeniedError, 403): "auth",
            _status_error(anthropic.AuthenticationError, 401, {"x-should-retry": "false"}): "auth",
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
        """Return Anthropic error verdict cases.

        A retry-after header fills retry_after without choosing the verdict.
        A rate-limit or overload error type pauses the rate-limit quota at any status.
        x-should-retry overrides the status tables, but x-should-retry=false keeps a required pause.
        A mid-stream error carries status 200, whose retry headers do not apply.
        """
        return {
            _status_error(
                anthropic.RateLimitError, 429, {"retry-after": "7"}, "rate_limit_error"
            ): PauseAll(retry_after=7.0),
            _status_error(anthropic.RateLimitError, 429, {"retry-after-ms": "1500"}): PauseAll(
                retry_after=1.5
            ),
            _status_error(anthropic.BadRequestError, 400, {"retry-after": "7"}): DoNotRetry(),
            _status_error(anthropic.OverloadedError, 529, error_type="overloaded_error"): PauseAll(
                retry_after=None
            ),
            _status_error(anthropic.APIStatusError, 418, error_type="overloaded_error"): PauseAll(
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
            _status_error(anthropic.APIStatusError, 599): RetryThisOne(retry_after=None),
            _status_error(
                anthropic.InternalServerError, 500, {"x-should-retry": "false"}
            ): DoNotRetry(),
            _status_error(
                anthropic.BadRequestError, 400, {"x-should-retry": "true", "retry-after": "3"}
            ): RetryThisOne(retry_after=3.0),
            _status_error(
                anthropic.RateLimitError, 429, {"x-should-retry": "true"}, "rate_limit_error"
            ): PauseAll(retry_after=None),
            _status_error(
                anthropic.RateLimitError,
                429,
                {"x-should-retry": "false", "retry-after": "7"},
                "rate_limit_error",
            ): PauseAllDoNotRetry(retry_after=7.0),
            _status_error(
                anthropic.APIStatusError, 418, {"x-should-retry": "false"}, "overloaded_error"
            ): PauseAllDoNotRetry(retry_after=None),
            _status_error(
                anthropic.APIStatusError, 200, {"x-should-retry": "false"}, "overloaded_error"
            ): PauseAll(retry_after=None),
            _status_error(
                anthropic.APIStatusError, 200, {"x-should-retry": "true"}, "invalid_request_error"
            ): DoNotRetry(),
            _status_error(anthropic.APIStatusError, 200, error_type="api_error"): RetryThisOne(
                retry_after=None
            ),
            _status_error(anthropic.APIStatusError, 200, error_type="timeout_error"): RetryThisOne(
                retry_after=None
            ),
            TransientError(
                "throttled body", retry_after_seconds=3.0, is_rate_limit=True
            ): PauseAll(retry_after=3.0),
            TransientError("failed body"): RetryThisOne(retry_after=None),
        }
