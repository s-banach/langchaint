"""Implement the Anthropic Messages API through the official anthropic SDK.

The following SDK facts were verified against anthropic 1.0.0.
A structured binding sends the same `output_config.format` as `messages.parse(output_format=Model)`.
Its schema uses `transform_schema(TypeAdapter(Model).json_schema())` and type `"json_schema"`.
The adapter validates response text after the SDK exposes the complete message and billing.
SDK parsing may reject text before final output and cache counters arrive.
`messages.stream` assembles deltas, and `get_final_message()` returns the message.
The SDK reports no all-inclusive input total.

The API requires an unchanged replay of the latest assistant message's thinking during tool use.
The API filters earlier thinking blocks.
The API rejects consecutive thinking blocks outside their original order.
The adapter replays every `ReasoningPart` in `parts` order.

`automatic_cache_breakpoints=True` places a cache breakpoint at the end of the frozen prefix.
For `AsyncAnthropic` and `AsyncAnthropicBedrockMantle`, it also sends top-level `cache_control`.
Top-level `cache_control` selects the final cacheable block.
For `AsyncAnthropicBedrock`, it instead places a cache breakpoint on the last message block.
The frozen prefix ends at the system prompt or at the last tool when no system prompt exists.
`automatic_cache_breakpoints=False` places no automatic cache breakpoint.
A user part with `cache_breakpoint=True` adds `cache_control` to its text or image block.
A final `ToolMessage` part with `cache_breakpoint=True` adds `cache_control` to its enclosing `tool_result` block.
A non-final `ToolMessage` part with `cache_breakpoint=True` returns `RejectedMessages` because the boundary would move.
A parts `system_prompt` produces one system block per part and preserves its cache breakpoints.

The API accepts at most four cache breakpoints per request.
Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching, read 2026-07-25.
Top-level `cache_control` and the binding's cache breakpoints reduce `message_cache_breakpoint_budget`.
Binding fails with `ValueError` when its cache breakpoints exceed the limit.
The adapter sends only the latest message cache breakpoints that fit `message_cache_breakpoint_budget`.
Keeping only the latest cache breakpoints rarely costs a cache hit, because the latest prefix includes the earlier ones.
An older cache breakpoint matters only when the latest one finds nothing in the cache.
That happens when this conversation's entries have expired but a shared prefix is still cached.

To find an earlier request's cache entry, each cache breakpoint checks its own block and at most 19 blocks before it.
On the Claude API, consecutive `tool_use` blocks count as one block, and so do consecutive `tool_result` blocks.
A request adding 20 or more blocks misses the previous request's cache unless a cache breakpoint is in its first 19.
Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching, read 2026-09-24.

Top-level `cache_control` and each cache breakpoint use `cache_ttl`.
The default `"5m"` omits the API-default `ttl` key.
`"1h"` sends `ttl="1h"` and uses the `input_tokens_cache_write_1h` rate.

Content mappings were verified against anthropic 0.121.0.
- `ImagePart` becomes `Base64ImageSourceParam`.
- `ImageUrlPart` becomes `URLImageSourceParam`.
- `AudioPart` returns `RejectedMessages` inside `UserMessage` and `ToolMessage`.
- `Usage.server_tool_use` reports web-search invocation counts.

Request and response mappings:
- `ToolMessage` becomes `tool_result` inside a user message.
- Consecutive `ToolMessage` values share one user message because the API requires alternating roles.
- `stop_reason` `end_turn`, `tool_use`, and `max_tokens` map to `"stop"`, `"tool_call"`, and `"max_completion_tokens"`.
- `refusal` maps to `"refusal"`, and `model_context_window_exceeded` maps to `"context_window_exceeded"`.
- Other `stop_reason` values map to `"other"`.
- `reasoning_level` sends `output_config.effort` with `thinking={"type": "adaptive"}`.
- The adapter sends neither field alone because effort applies only to adaptive thinking.
- `reasoning_level` accepts values wider than the SDK literal and sends each value unchanged.
- The provider reports unsupported values through its own error.
- The adapter never sends `thinking.display`.
- The SDK default `"summarized"` returns thinking text, while `"omitted"` redacts it.
"""

import base64
import json
import math
from abc import ABC
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast, override

import anthropic
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    Omit,
    omit,
    transform_schema,
)
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    ImageBlockParam,
    MessageParam,
    OutputConfigParam,
    RedactedThinkingBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlockParam,
    URLImageSourceParam,
)
from anthropic.types.json_output_format_param import JSONOutputFormatParam
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint.adapter import (
    REASONING_PART_SEPARATOR,
    Adapter,
    AdapterStream,
    AllowedToolsChoice,
    Binding,
    BoundAdapter,
    ContextWindowExceeded,
    EmptyAssistantMessage,
    MaxCompletionTokensExceeded,
    ReasoningDelta,
    Refusal,
    RejectedMessages,
    RequestFailure,
    RequestParams,
    ResponseIdentity,
    ResponseOutcome,
    SchemaViolation,
    SpecificToolChoice,
    StreamItem,
    ToolCallDelta,
    ToolChoice,
    UnfinishedAssistantMessage,
    UsableResponse,
    _RejectedMessagesError,
    narrowed_request_params,
    record_request_failure_fallthrough,
    reject_extra_body_keys_the_adapter_populates,
    request_failure_from_response,
    request_params_json,
    validated_provider_executed_tool_types,
)
from langchaint.billing.pricing import (
    Billing,
    ProviderBilling,
    TokenRates,
    category_cost_in_usd,
    invocation_cost_in_usd,
    require_finite_nonnegative_rate,
)
from langchaint.billing.usage import Usage
from langchaint.common.exceptions import StreamProtocolError
from langchaint.common.messages import (
    AssistantMessage,
    AssistantPart,
    ContentPart,
    Message,
    RawPart,
    ReasoningPart,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.tools import ToolSchema

type _ContentBlockParam = (
    TextBlockParam
    | ImageBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | ThinkingBlockParam
    | RedactedThinkingBlockParam
)

type _AnthropicImageMediaType = Literal["image/gif", "image/jpeg", "image/png", "image/webp"]

_ANTHROPIC_IMAGE_MEDIA_TYPES: tuple[_AnthropicImageMediaType, ...] = (
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
)


_PAUSE_STATUSES = frozenset({429, 529})
"""429 rate_limit_error and 529 overloaded_error pause every request sharing the rate-limit quota."""

_PAUSE_ERROR_TYPES = frozenset({"rate_limit_error", "overloaded_error"})
"""The two _PAUSE_STATUSES error types, which pause at any status carrying them."""

_RETRY_THIS_ONE_STATUSES = frozenset({408, 409, 500, 503, 504})
"""One request's failure, retried without pausing siblings.

500 api_error and 504 timeout_error come from the errors page.
408 and 409 are the request and lock timeouts anthropic's own SDK retries (anthropic 0.120.2 _should_retry).
503 is the status AsyncAnthropicBedrock raises ServiceUnavailableError for (anthropic 0.120.2).
No errors page states that a Bedrock 503 throttles the rate-limit quota.
It therefore retries without pausing every request.
"""

_RETRY_THIS_ONE_ERROR_TYPES = frozenset({"api_error", "timeout_error"})
"""The 500 and 504 error types, which retry at any status carrying them."""

_DO_NOT_RETRY_STATUSES = frozenset({400, 401, 402, 403, 404, 413, 422})
"""The statuses that reject this request. A resend fails again."""

REQUEST_FAILURE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""`record_request_failure_fallthrough` increments this counter for each status-family default."""

_MAX_CACHE_BREAKPOINTS_PER_REQUEST = 4
"""The API allows at most 4 cache breakpoints per request, the binding's own included."""

type CacheTTL = Literal["5m", "1h"]
"""Cache TTLs with write rates of 1.25x ("5m") or 2x ("1h") base input."""

type AnthropicServiceTier = Literal["auto", "standard_only"]
"""What a request may ask for (anthropic 0.120.0).

"auto" permits priority capacity when available or standard capacity otherwise.
No request value selects priority capacity.
"standard_only" is the one value that pins a tier.
"""

type _AnthropicReportedServiceTier = Literal["standard", "priority", "batch"]
"""What an Anthropic response reports having served."""

_STANDARD_TIER: _AnthropicReportedServiceTier = "standard"

type AnthropicClient = AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle


def client_without_retries[ClientT: AnthropicClient](client: ClientT) -> ClientT:
    """Return one client whose SDK retries are disabled."""
    if client.max_retries == 0:
        return client
    # Bedrock copy() drops custom transports unless http_client is passed again.
    return client.with_options(max_retries=0, http_client=client._client)


@dataclass(frozen=True, kw_only=True)
class AnthropicRates:
    """Anthropic token rates for one service tier.

    Each rate is in USD per million tokens and has the name of the `Usage` counter it prices.
    The `_5m` and `_1h` cache-write rates price the five-minute and one-hour shares of `input_tokens_cache_write`.
    Pass NaN for an unknown rate.
    A nonzero counter in that category then costs NaN, and a zero counter costs zero.
    """

    input_tokens_cache_none: float
    output_tokens: float
    input_tokens_cache_read: float
    input_tokens_cache_write_5m: float
    input_tokens_cache_write_1h: float

    def price(
        self,
        *,
        service_tier: str,
        usage_raw: BaseModel | None,
        input_tokens_cache_read: int,
        input_tokens_cache_write_5m: int,
        input_tokens_cache_write_1h: int,
        input_tokens_cache_none: int,
        output_tokens: int,
        output_tokens_reasoning: int,
        provider_executed_tool_cost_in_usd: float,
    ) -> ProviderBilling:
        """Price one response's counters, the two cache-write TTLs each at their own rate.

        `Usage.input_tokens_cache_write` sums both write counters.
        `input_tokens_cache_write_cost_in_usd` sums both write costs.
        The reported cache-write price blends the rates to reproduce that cost.
        Zero writes use Anthropic's default five-minute rate.

        Raises:
            pydantic.ValidationError: a counter is negative.
        """
        cache_write_5m_cost_in_usd = category_cost_in_usd(
            input_tokens_cache_write_5m,
            usd_per_million_tokens=self.input_tokens_cache_write_5m,
        )
        cache_write_1h_cost_in_usd = category_cost_in_usd(
            input_tokens_cache_write_1h,
            usd_per_million_tokens=self.input_tokens_cache_write_1h,
        )
        input_tokens_cache_write = input_tokens_cache_write_5m + input_tokens_cache_write_1h
        input_tokens_cache_write_cost_in_usd = (
            cache_write_5m_cost_in_usd + cache_write_1h_cost_in_usd
        )
        return ProviderBilling(
            billing=Billing(
                usage=Usage(
                    input_tokens_cache_read=input_tokens_cache_read,
                    input_tokens_cache_write=input_tokens_cache_write,
                    input_tokens_cache_none=input_tokens_cache_none,
                    output_tokens=output_tokens,
                    output_tokens_reasoning=output_tokens_reasoning,
                    input_tokens_cache_read_cost_in_usd=category_cost_in_usd(
                        input_tokens_cache_read,
                        usd_per_million_tokens=self.input_tokens_cache_read,
                    ),
                    input_tokens_cache_write_cost_in_usd=input_tokens_cache_write_cost_in_usd,
                    input_tokens_cache_none_cost_in_usd=category_cost_in_usd(
                        input_tokens_cache_none,
                        usd_per_million_tokens=self.input_tokens_cache_none,
                    ),
                    output_tokens_cost_in_usd=category_cost_in_usd(
                        output_tokens,
                        usd_per_million_tokens=self.output_tokens,
                    ),
                    provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
                ),
                service_tier=service_tier,
                usd_per_million_tokens=TokenRates(
                    input_tokens_cache_read=self.input_tokens_cache_read,
                    input_tokens_cache_write=(
                        input_tokens_cache_write_cost_in_usd * 1_000_000 / input_tokens_cache_write
                        if input_tokens_cache_write
                        else self.input_tokens_cache_write_5m
                    ),
                    input_tokens_cache_none=self.input_tokens_cache_none,
                    output_tokens=self.output_tokens,
                ),
            ),
            usage_raw=usage_raw,
        )

    def multiplied(self, multiplier: float) -> "AnthropicRates":
        """Return token rates multiplied by one value."""
        return AnthropicRates(
            input_tokens_cache_none=self.input_tokens_cache_none * multiplier,
            output_tokens=self.output_tokens * multiplier,
            input_tokens_cache_read=self.input_tokens_cache_read * multiplier,
            input_tokens_cache_write_5m=self.input_tokens_cache_write_5m * multiplier,
            input_tokens_cache_write_1h=self.input_tokens_cache_write_1h * multiplier,
        )


_UNPRICED_RATES = AnthropicRates(
    input_tokens_cache_none=float("nan"),
    output_tokens=float("nan"),
    input_tokens_cache_read=float("nan"),
    input_tokens_cache_write_5m=float("nan"),
    input_tokens_cache_write_1h=float("nan"),
)


@dataclass(frozen=True, kw_only=True)
class AnthropicPricingTable:
    """Anthropic rates and modifiers for one model."""

    standard: AnthropicRates
    priority: AnthropicRates | None = None
    batch: AnthropicRates | None = None
    inference_geo_us_multiplier: float | None = None
    web_search_usd_per_invocation: float | None = None

    def __post_init__(self) -> None:
        """Reject an invalid regional multiplier.

        Raises:
            ValueError: The regional multiplier is invalid.
        """
        multiplier = self.inference_geo_us_multiplier
        if multiplier is None:
            return
        if isinstance(multiplier, bool) or not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("inference_geo_us_multiplier must be finite and positive")

    def rates_for(
        self,
        *,
        service_tier: _AnthropicReportedServiceTier | None,
        inference_geo: str | None,
    ) -> AnthropicRates:
        """Select token rates using reported response metadata."""
        priced_tier = _priced_tier(service_tier)
        if priced_tier == "standard":
            rates = self.standard
        elif priced_tier == "priority":
            rates = self.priority
        else:
            rates = self.batch
        if rates is None:
            return _UNPRICED_RATES
        if inference_geo != "us":
            return rates
        multiplier = self.inference_geo_us_multiplier
        if multiplier is None:
            return _UNPRICED_RATES
        return rates.multiplied(multiplier)

    def multiplied(self, multiplier: float) -> "AnthropicPricingTable":
        """Return the table with every token rate multiplied by one value.

        Modifiers and per-invocation prices are unchanged.
        """
        return AnthropicPricingTable(
            standard=self.standard.multiplied(multiplier),
            priority=None if self.priority is None else self.priority.multiplied(multiplier),
            batch=None if self.batch is None else self.batch.multiplied(multiplier),
            inference_geo_us_multiplier=self.inference_geo_us_multiplier,
            web_search_usd_per_invocation=self.web_search_usd_per_invocation,
        )


def _cache_control_param(cache_ttl: CacheTTL) -> CacheControlEphemeralParam:
    """Build one cache_control value.

    "5m" omits the ttl key because it is the API default.

    The "5m" wire form must stay byte-stable across releases.
    A changed wire form would invalidate a caller's live cache entry.
    """
    if cache_ttl == "5m":
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": "1h"}


_WEB_SEARCH_TOOL_TYPES = frozenset({
    "web_search_20250305",
    "web_search_20260209",
    "web_search_20260318",
})
_WEB_FETCH_TOOL_TYPES = frozenset({
    "web_fetch_20250910",
    "web_fetch_20260209",
    "web_fetch_20260309",
    "web_fetch_20260318",
})
_TOOL_SEARCH_TOOL_TYPES = frozenset({
    "tool_search_tool_bm25",
    "tool_search_tool_bm25_20251119",
    "tool_search_tool_regex",
    "tool_search_tool_regex_20251119",
})
_CODE_EXECUTION_TOOL_TYPES = frozenset({
    "code_execution_20260120",
    "code_execution_20260521",
})
_CODE_EXECUTION_EXEMPTING_WEB_TOOL_TYPES = frozenset({
    "web_search_20260209",
    "web_search_20260318",
    "web_fetch_20260209",
    "web_fetch_20260309",
    "web_fetch_20260318",
})
_SUPPORTED_PROVIDER_EXECUTED_TOOL_TYPES = (
    _WEB_SEARCH_TOOL_TYPES
    | _WEB_FETCH_TOOL_TYPES
    | _TOOL_SEARCH_TOOL_TYPES
    | _CODE_EXECUTION_TOOL_TYPES
)


@dataclass(frozen=True)
class _AnthropicProviderExecutedTools:
    """Validated provider-executed tool categories needed for billing."""

    web_search: bool = False
    web_fetch: bool = False
    tool_search: bool = False
    code_execution_exempt: bool = False


_NO_ANTHROPIC_PROVIDER_EXECUTED_TOOLS = _AnthropicProviderExecutedTools()


def _provider_executed_tools(
    provider_executed_tools: tuple[Mapping[str, object], ...],
) -> _AnthropicProviderExecutedTools:
    """Validate supported `type` values while preserving each mapping by reference.

    Raises:
        ValueError: a mapping lacks a supported string `type` value.
            Also raised when code execution lacks a qualifying web tool.
    """
    tool_types = validated_provider_executed_tool_types(
        provider_executed_tools,
        supported_types=_SUPPORTED_PROVIDER_EXECUTED_TOOL_TYPES,
        adapter_name="Anthropic",
    )
    code_execution = bool(tool_types & _CODE_EXECUTION_TOOL_TYPES)
    code_execution_exempt = bool(tool_types & _CODE_EXECUTION_EXEMPTING_WEB_TOOL_TYPES)
    if code_execution and not code_execution_exempt:
        raise ValueError("Anthropic code execution requires a qualifying web tool")
    return _AnthropicProviderExecutedTools(
        web_search=bool(tool_types & _WEB_SEARCH_TOOL_TYPES),
        web_fetch=bool(tool_types & _WEB_FETCH_TOOL_TYPES),
        tool_search=bool(tool_types & _TOOL_SEARCH_TOOL_TYPES),
        code_execution_exempt=code_execution and code_execution_exempt,
    )


@dataclass(frozen=True, kw_only=True)
class _AnthropicPrecomputedFields:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place.
    Passing them as explicit keywords keeps the SDK's overload resolution intact.
    """

    model: str
    max_tokens: int
    temperature: float | Omit
    system: list[TextBlockParam] | Omit
    tools: list[ToolUnionParam] | Omit
    tool_choice: ToolChoiceParam | Omit
    output_config: OutputConfigParam | Omit
    thinking: ThinkingConfigParam | Omit
    service_tier: AnthropicServiceTier | Omit
    inference_geo: str | Omit
    cache_control: CacheControlEphemeralParam | Omit
    automatic_message_cache_breakpoint: bool
    cache_ttl: CacheTTL
    message_cache_breakpoint_budget: int
    """The cache breakpoints left for message parts under the API's limit of four per request.

    Cache breakpoints in system blocks and tools reduce this value.
    So does the automatic message cache breakpoint or top-level `cache_control`.
    """

    extra_body: Mapping[str, object] | None
    provider_executed_tools: _AnthropicProviderExecutedTools


@dataclass(frozen=True)
class _TemperatureExtraBody(Mapping[str, object]):
    """Expose temperature with caller fields while retaining caller_fields by reference."""

    temperature: float
    caller_fields: Mapping[str, object]

    @override
    def __getitem__(self, key: str) -> object:
        if key == "temperature":
            return self.temperature
        return self.caller_fields[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield "temperature"
        yield from (key for key in self.caller_fields if key != "temperature")

    @override
    def __len__(self) -> int:
        caller_field_count = len(self.caller_fields)
        if "temperature" in self.caller_fields:
            return caller_field_count
        return caller_field_count + 1


def _extra_body_with_temperature(
    precomputed: _AnthropicPrecomputedFields,
) -> Mapping[str, object] | None:
    if isinstance(precomputed.temperature, Omit):
        return precomputed.extra_body
    if precomputed.extra_body is None:
        return {"temperature": precomputed.temperature}
    return _TemperatureExtraBody(
        temperature=precomputed.temperature,
        caller_fields=precomputed.extra_body,
    )


_ADAPTER_POPULATED_WIRE_KEYS = frozenset({
    "cache_control",
    "model",
    "max_tokens",
    "temperature",
    "system",
    "tools",
    "tool_choice",
    "output_config",
    "thinking",
    "service_tier",
    "inference_geo",
    "messages",
    "stream",
})
"""The wire keys an extra_body must not hold because the adapter owns their values."""


@dataclass(frozen=True, kw_only=True)
class _AnthropicRequestParams(RequestParams):
    """One messages request: the binding's precomputed fields and one input's converted messages."""

    precomputed: _AnthropicPrecomputedFields
    messages: list[MessageParam]

    @override
    def as_json(self) -> str:
        """Render the request as a JSON object, dropping every field left to the provider's default."""
        return request_params_json(self, omitted_class=Omit)


def _part_block(
    part: ContentPart, *, message_class: type[UserMessage] | type[ToolMessage]
) -> TextBlockParam | ImageBlockParam:
    """Convert one ContentPart to its wire block.

    Raises:
        _RejectedMessagesError: ContentPart has no wire form for message_class.
    """
    match part.kind:
        case "text":
            return {"type": "text", "text": part.text}
        case "image":
            if part.media_type not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
                raise _RejectedMessagesError(
                    f"AnthropicMessagesAdapter cannot send ImagePart inside "
                    f"{message_class.__name__}.content: the Anthropic API accepts media types "
                    f"{_ANTHROPIC_IMAGE_MEDIA_TYPES}, not {part.media_type!r}"
                )
            image_source: Base64ImageSourceParam = {
                "type": "base64",
                "media_type": part.media_type,
                "data": base64.b64encode(part.data).decode("ascii"),
            }
            return {"type": "image", "source": image_source}
        case "image_url":
            url_image_source: URLImageSourceParam = {"type": "url", "url": part.url}
            return {"type": "image", "source": url_image_source}
        case "audio":
            missing_audio_type = (
                "the Anthropic API has no audio input content type"
                if message_class is UserMessage
                else "ToolResultBlockParam.content has no audio variant"
            )
            raise _RejectedMessagesError(
                f"AnthropicMessagesAdapter cannot send AudioPart inside "
                f"{message_class.__name__}.content: {missing_audio_type}"
            )


def _user_content_blocks(
    user_message: UserMessage,
) -> tuple[list[_ContentBlockParam], list[TextBlockParam | ImageBlockParam]]:
    """Convert one UserMessage.content value to wire blocks.

    The second element holds the blocks whose part sets cache_breakpoint in content order.
    The caller applies `message_cache_breakpoint_budget`, so this function writes no `cache_control`.

    Raises:
        _RejectedMessagesError: _part_block rejects one ContentPart.
    """
    blocks: list[_ContentBlockParam] = []
    part_cache_breakpoint_blocks: list[TextBlockParam | ImageBlockParam] = []
    if isinstance(user_message.content, str):
        blocks.append({"type": "text", "text": user_message.content})
        return blocks, part_cache_breakpoint_blocks
    for part in user_message.content:
        block = _part_block(part, message_class=UserMessage)
        blocks.append(block)
        if part.cache_breakpoint:
            part_cache_breakpoint_blocks.append(block)
    return blocks, part_cache_breakpoint_blocks


def _tool_result_content(
    content: str | tuple[ContentPart, ...],
) -> str | list[TextBlockParam | ImageBlockParam]:
    """Convert one ToolMessage's content to the tool_result content field.

    A bare string passes through.
    A ContentPart tuple becomes TextBlockParam and ImageBlockParam values.

    Raises:
        _RejectedMessagesError: _part_block rejects one ContentPart.
    """
    if isinstance(content, str):
        return content
    return [_part_block(part, message_class=ToolMessage) for part in content]


def _replayed_block(raw: Mapping[str, object]) -> _ContentBlockParam:
    """Copy a stored SDK block without reading or changing its fields.

    The copy prevents `cache_control` writes from changing the stored block.
    A block from another provider passes through when it has a `type` key.

    Raises:
        _RejectedMessagesError: `raw` lacks the `type` key required by anthropic 0.120.2 block parameters.
    """
    if "type" not in raw:
        raise _RejectedMessagesError(
            "ReasoningPart.raw or RawPart.raw names no type key, "
            "so anthropic has no content block to send it as"
        )
    # cast: a deliberately-opaque value re-enters the typed API whose own serialization produced it.
    return cast("_ContentBlockParam", dict(raw))


def _assistant_content_blocks(assistant_message: AssistantMessage) -> list[_ContentBlockParam]:
    """Convert one AssistantMessage to wire blocks in `parts` order.

    `ReasoningPart.raw` and `RawPart.raw` pass through unchanged by their `type` keys.
    The API rejects modified thinking blocks and unknown `type` values.

    Raises:
        json.JSONDecodeError: `ToolCall.args_json` is invalid JSON.
        _RejectedMessagesError: A stored block lacks a `type` key.
    """
    blocks: list[_ContentBlockParam] = []
    for part in assistant_message.parts:
        if part.kind == "text":
            blocks.append(TextBlockParam(type="text", text=part.text))
        elif part.kind == "tool_call":
            blocks.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=part.id,
                    name=part.name,
                    input=json.loads(part.args_json),
                )
            )
        else:
            blocks.append(_replayed_block(part.raw))
    return blocks


def _tool_message_has_cache_breakpoint(tool_message: ToolMessage) -> bool:
    """Return whether the last part places a cache breakpoint on the enclosing `tool_result` block.

    Raises:
        _RejectedMessagesError: A non-final part sets `cache_breakpoint`, because the API caches up to the block end.
    """
    if isinstance(tool_message.content, str):
        return False
    cache_breakpoint_indexes = [
        index for index, part in enumerate(tool_message.content) if part.cache_breakpoint
    ]
    if not cache_breakpoint_indexes:
        return False
    if cache_breakpoint_indexes != [len(tool_message.content) - 1]:
        raise _RejectedMessagesError(
            "cache_breakpoint on a ToolMessage part is honored only on the message's last part: "
            "the cache breakpoint goes on the enclosing tool_result block, whose span ends at the last part"
        )
    return True


def _wire_messages(
    messages: Sequence[Message],
    *,
    automatic_cache_breakpoints: bool,
    cache_ttl: CacheTTL,
    message_cache_breakpoint_budget: int,
) -> list[MessageParam]:
    """Convert messages and apply the permitted cache breakpoints.

    `automatic_cache_breakpoints` places a cache breakpoint on the last block unless it is a thinking block.
    A user part's cache breakpoint goes on its own block, and a tool part's on its enclosing `tool_result`.
    The latest cache breakpoints up to `message_cache_breakpoint_budget` are sent.

    Raises:
        _RejectedMessagesError: A `ContentPart` lacks a wire form, or raw lacks `type`.
            A non-final tool part that sets `cache_breakpoint` also raises it.
        json.JSONDecodeError: `ToolCall.args_json` is invalid JSON.
    """
    wire: list[tuple[Literal["user", "assistant"], list[_ContentBlockParam]]] = []
    pending_tool_results: list[_ContentBlockParam] = []
    cache_breakpoint_blocks: list[TextBlockParam | ImageBlockParam | ToolResultBlockParam] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            wire.append(("user", list(pending_tool_results)))
            pending_tool_results.clear()

    for message in messages:
        match message.kind:
            case "tool":
                tool_result_block: ToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": _tool_result_content(message.content),
                    "is_error": message.is_error,
                }
                if _tool_message_has_cache_breakpoint(message):
                    cache_breakpoint_blocks.append(tool_result_block)
                pending_tool_results.append(tool_result_block)
            case "user":
                flush_tool_results()
                blocks, part_cache_breakpoint_blocks = _user_content_blocks(message)
                cache_breakpoint_blocks.extend(part_cache_breakpoint_blocks)
                wire.append(("user", blocks))
            case "assistant":
                flush_tool_results()
                wire.append(("assistant", _assistant_content_blocks(message)))
    flush_tool_results()
    if message_cache_breakpoint_budget > 0:
        for block in cache_breakpoint_blocks[-message_cache_breakpoint_budget:]:
            block["cache_control"] = _cache_control_param(cache_ttl)
    if automatic_cache_breakpoints and wire:
        last_blocks = wire[-1][1]
        if last_blocks:
            last_block = last_blocks[-1]
            if last_block["type"] != "thinking" and last_block["type"] != "redacted_thinking":
                last_block["cache_control"] = _cache_control_param(cache_ttl)
    return [MessageParam(role=role, content=blocks) for role, blocks in wire]


def _request_messages(
    messages: Sequence[Message], precomputed_fields: _AnthropicPrecomputedFields
) -> list[MessageParam] | RejectedMessages:
    """Convert messages under the binding's caching parameters, or report them unsendable.

    The one place a Sequence[Message] this adapter will not put on the wire becomes RejectedMessages.
    The wire block holds parsed `tool_call.args_json`.
    Text that is not JSON has no wire block.
    """
    try:
        return _wire_messages(
            messages,
            automatic_cache_breakpoints=precomputed_fields.automatic_message_cache_breakpoint,
            cache_ttl=precomputed_fields.cache_ttl,
            message_cache_breakpoint_budget=precomputed_fields.message_cache_breakpoint_budget,
        )
    except _RejectedMessagesError as rejected:
        return RejectedMessages(error_text=str(rejected))
    except json.JSONDecodeError as not_json:
        return RejectedMessages(
            error_text=f"a tool call's args_json is not valid JSON: {not_json}"
        )


def _wire_tool_choice(tool_choice: ToolChoice, *, parallel_tool_calls: bool) -> ToolChoiceParam:
    """Convert the neutral tool choice.

    Neutral "required" is Anthropic "any".

    Raises:
        TypeError: `tool_choice` is `AllowedToolsChoice`, which Anthropic does not support.
    """
    disable_parallel_tool_use = not parallel_tool_calls
    if isinstance(tool_choice, SpecificToolChoice):
        return {
            "type": "tool",
            "name": tool_choice.tool_name,
            "disable_parallel_tool_use": disable_parallel_tool_use,
        }
    if isinstance(tool_choice, AllowedToolsChoice):
        raise TypeError("AnthropicMessagesAdapter does not support AllowedToolsChoice")
    if tool_choice == "auto":
        return {"type": "auto", "disable_parallel_tool_use": disable_parallel_tool_use}
    if tool_choice == "required":
        return {"type": "any", "disable_parallel_tool_use": disable_parallel_tool_use}
    return {"type": "none"}


def _wire_tools(
    tool_schemas: tuple[ToolSchema, ...],
    provider_executed_tools: tuple[Mapping[str, object], ...],
    *,
    cache_breakpoint_on_last_tool: bool,
    cache_ttl: CacheTTL,
) -> list[ToolUnionParam]:
    """Convert every bound tool to one ordered wire list.

    `cache_breakpoint_on_last_tool` puts the frozen-prefix cache breakpoint on the last tool.
    The last tool carries the cache breakpoint when no system prompt follows the tools.
    """
    tools: list[ToolUnionParam] = [
        {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "input_schema": dict(tool_schema.args_schema),
        }
        for tool_schema in tool_schemas
    ]
    # cast: the neutral Mapping type exceeds the SDK TypedDict union.
    # The adapter validated each mapping's type discriminator.
    tools.extend(cast("ToolUnionParam", tool) for tool in provider_executed_tools)
    if cache_breakpoint_on_last_tool and tools and "cache_control" not in tools[-1]:
        # Copying preserves the caller's mapping.
        last_tool = dict(tools[-1])
        last_tool["cache_control"] = _cache_control_param(cache_ttl)
        # cast: the copied mapping remains wider than the SDK TypedDict union.
        tools[-1] = cast("ToolUnionParam", last_tool)
    return tools


_STOP_REASON_BY_ANTHROPIC_STOP_REASON: Mapping[str, StopReason] = {
    "end_turn": "stop",
    "tool_use": "tool_call",
    "max_tokens": "max_completion_tokens",
    "refusal": "refusal",
    "model_context_window_exceeded": "context_window_exceeded",
}
"""The langchaint `StopReason` of each Anthropic `stop_reason` with a counterpart."""


def _normalized_stop_reason(stop_reason: str | None) -> StopReason:
    if stop_reason is None:
        return "other"
    return _STOP_REASON_BY_ANTHROPIC_STOP_REASON.get(stop_reason, "other")


def _unfinished_message_or_none(
    message: anthropic.types.Message, *, assistant_message: AssistantMessage
) -> UnfinishedAssistantMessage | None:
    """Return `UnfinishedAssistantMessage` for `pause_turn`, a null stop reason, or an unknown stop reason."""
    stop_reason = message.stop_reason
    if stop_reason in (
        "end_turn",
        "tool_use",
        "max_tokens",
        "refusal",
        "stop_sequence",
        "model_context_window_exceeded",
    ):
        return None
    return UnfinishedAssistantMessage(
        error_text=f"anthropic returned stop_reason {stop_reason!r}, which langchaint cannot continue",
        assistant_message=assistant_message,
    )


def _as_message(raw: BaseModel) -> anthropic.types.Message:
    """Narrow a raw response to the SDK message this adapter produces.

    `BoundAdapter` accepts `BaseModel` because the neutral core imports no SDK.
    This adapter's stream produces every valid value.

    Raises:
        TypeError: `raw` is not an anthropic `Message`.
    """
    if not isinstance(raw, anthropic.types.Message):
        raise TypeError(f"expected an anthropic Message, got {type(raw).__name__}")
    return raw


def _first_text_block_text(message: anthropic.types.Message) -> str | None:
    """Return the text of the assistant message's first text block, None when it holds none.

    Structured output validation uses this block.
    SDK parsing validates every text block and returns the first instance.
    """
    for block in message.content:
        if block.type == "text":
            return block.text
    return None


def _assistant_message_from(message: anthropic.types.Message) -> AssistantMessage:
    """Build `AssistantMessage` from SDK blocks in order.

    Empty text blocks are dropped because the API rejects them on replay.
    Thinking blocks become replayable `ReasoningPart` values.
    Redacted thinking has `text=None`.
    Unmodeled blocks become replayable `RawPart` values.
    """
    parts: list[AssistantPart] = []
    for block in message.content:
        if block.type == "text":
            if block.text:
                parts.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            parts.append(ToolCall(id=block.id, name=block.name, args_json=json.dumps(block.input)))
        elif block.type == "thinking":
            parts.append(
                ReasoningPart(
                    raw=block.model_dump(mode="json", exclude_none=True),
                    text=block.thinking or None,
                )
            )
        elif block.type == "redacted_thinking":
            parts.append(ReasoningPart(raw=block.model_dump(mode="json", exclude_none=True)))
        else:
            parts.append(RawPart(raw=block.model_dump(mode="json", exclude_none=True)))
    return AssistantMessage(parts=tuple(parts))


def _priced_tier(
    service_tier: _AnthropicReportedServiceTier | None,
) -> _AnthropicReportedServiceTier:
    """Normalize a missing reported tier to `standard`.

    Bedrock responses need this default because Anthropic service tiers do not apply.
    """
    return service_tier if service_tier is not None else _STANDARD_TIER


def _billing_from_sdk_usage(
    usage_raw: anthropic.types.Usage,
    pricing: AnthropicPricingTable,
    *,
    provider_executed_tools: _AnthropicProviderExecutedTools = _NO_ANTHROPIC_PROVIDER_EXECUTED_TOOLS,
    billing_complete: bool = True,
) -> ProviderBilling:
    """Price SDK counters by the reported service tier.

    `usage.input_tokens` excludes cache reads and writes.
    Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching, read 2026-07-25.
    `usage.cache_creation` separates five-minute and one-hour writes.
    Missing `usage.cache_creation` makes `cache_creation_input_tokens` five-minute writes.
    `usage.output_tokens_details` is optional.

    Raises:
        pydantic.ValidationError: A reported token counter is negative.
        ValueError: A provider-executed request counter is boolean or negative.
    """
    output_tokens_details = usage_raw.output_tokens_details
    input_tokens_cache_write_5m = usage_raw.cache_creation_input_tokens or 0
    input_tokens_cache_write_1h = 0
    if usage_raw.cache_creation is not None:
        input_tokens_cache_write_5m = usage_raw.cache_creation.ephemeral_5m_input_tokens
        input_tokens_cache_write_1h = usage_raw.cache_creation.ephemeral_1h_input_tokens
    service_tier = _priced_tier(usage_raw.service_tier)
    rates = pricing.rates_for(
        service_tier=usage_raw.service_tier,
        inference_geo=usage_raw.inference_geo,
    )
    server_tool_use = usage_raw.server_tool_use
    web_search_requests = 0 if server_tool_use is None else server_tool_use.web_search_requests
    provider_executed_tool_cost_in_usd = invocation_cost_in_usd(
        web_search_requests,
        usd_per_invocation=pricing.web_search_usd_per_invocation,
    )
    if provider_executed_tools.web_search and not billing_complete:
        provider_executed_tool_cost_in_usd = float("nan")
    if server_tool_use is not None:
        accounted_counters = {"web_search_requests"}
        if provider_executed_tools.web_fetch:
            accounted_counters.add("web_fetch_requests")
        if provider_executed_tools.tool_search:
            accounted_counters.add("tool_search_requests")
        if provider_executed_tools.code_execution_exempt:
            accounted_counters.add("code_execution_requests")
        unaccounted_counter_fired = any(
            counter_name.endswith("_requests") and counter
            for counter_name, counter in server_tool_use.model_dump().items()
            if counter_name not in accounted_counters
        )
        if unaccounted_counter_fired:
            provider_executed_tool_cost_in_usd = float("nan")
    return rates.price(
        service_tier=service_tier,
        usage_raw=usage_raw,
        input_tokens_cache_read=usage_raw.cache_read_input_tokens or 0,
        input_tokens_cache_write_5m=input_tokens_cache_write_5m,
        input_tokens_cache_write_1h=input_tokens_cache_write_1h,
        input_tokens_cache_none=usage_raw.input_tokens,
        output_tokens=usage_raw.output_tokens,
        output_tokens_reasoning=(
            output_tokens_details.thinking_tokens if output_tokens_details is not None else 0
        ),
        provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
    )


def _usable_response[OutputT](
    message: anthropic.types.Message, output: OutputT, assistant_message: AssistantMessage
) -> UsableResponse[OutputT]:
    """Normalize one completed message around already-extracted output and its assistant message."""
    return UsableResponse(
        output=output,
        assistant_message=assistant_message,
        stop_reason=_normalized_stop_reason(message.stop_reason),
    )


def anthropic_request_failure(error: Exception) -> RequestFailure:
    """Return the `RequestFailure` of one exception an Anthropic request raised.

    `APIConnectionError` and `RetryableError` are transient transport failures that pause nothing.
    `APITimeoutError` is an `APIConnectionError` subclass.
    `request_failure_from_response` builds an `APIStatusError`'s failure from `_anthropic_table_answers`.
    It applies `x-should-retry` on every status except 200, which identifies a mid-stream error.
    Other exceptions are `unknown_exception`.
    """
    if isinstance(error, (anthropic.APIConnectionError, anthropic.RetryableError)):
        return RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None)
    if not isinstance(error, anthropic.APIStatusError):
        return RequestFailure(
            kind="unknown_exception", pauses_quota=False, retry_after_seconds=None
        )
    retries, pauses_quota = _anthropic_table_answers(error)
    return request_failure_from_response(
        status_code=error.status_code,
        headers=error.response.headers,
        retries=retries,
        pauses_quota=pauses_quota,
    )


def _anthropic_table_answers(error: anthropic.APIStatusError) -> tuple[bool, bool]:
    """Return whether one `APIStatusError` retries and whether it pauses the quota, by status and error type.

    Source: https://platform.claude.com/docs/en/api/errors, read 2026-08-01.
    Error types override status because stream errors may carry the live response's 200 status.
    `_PAUSE_STATUSES` and `_PAUSE_ERROR_TYPES` retry and pause.
    `_RETRY_THIS_ONE_STATUSES` and `_RETRY_THIS_ONE_ERROR_TYPES` retry without a pause.
    Unlisted 5xx statuses retry without a pause.
    Other unlisted statuses neither retry nor pause.
    """
    for error_types, statuses, pauses_quota in (
        (_PAUSE_ERROR_TYPES, _PAUSE_STATUSES, True),
        (_RETRY_THIS_ONE_ERROR_TYPES, _RETRY_THIS_ONE_STATUSES, False),
    ):
        if error.type in error_types or error.status_code in statuses:
            if error.status_code not in statuses:
                record_request_failure_fallthrough(
                    REQUEST_FAILURE_FALLTHROUGH_COUNTS,
                    function_name="anthropic_request_failure",
                    status_code=error.status_code,
                    error_type=error.type,
                )
            return True, pauses_quota
    if error.status_code in _DO_NOT_RETRY_STATUSES:
        return False, False
    record_request_failure_fallthrough(
        REQUEST_FAILURE_FALLTHROUGH_COUNTS,
        function_name="anthropic_request_failure",
        status_code=error.status_code,
        error_type=error.type,
    )
    return error.status_code >= 500, False


class AnthropicMessagesAdapter(Adapter):
    """Adapter over an AsyncAnthropic, AsyncAnthropicBedrock, or AsyncAnthropicBedrockMantle client.

    All three clients expose `messages.stream` and `with_options`.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = {
        AsyncAnthropicBedrock: "aws.bedrock",
        AsyncAnthropicBedrockMantle: "aws.bedrock",
    }
    """AsyncAnthropic is deliberately absent: it reaches whatever its base_url points at.

    Both classes here use Bedrock authentication and URLs.
    The caller states `provider_name` for other clients.
    """

    def __init__(
        self,
        *,
        client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
        model: str,
        pricing: AnthropicPricingTable,
        provider_name: str,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
        inference_geo: str | None = None,
    ) -> None:
        """Store request, caching, and pricing configuration without sending a request.

        `provider_name` is `"anthropic"` for `AsyncAnthropic` and `"aws.bedrock"` for Bedrock clients.
        The stored client disables SDK retries and preserves custom Bedrock transports.
        `cache_ttl` applies to top-level `cache_control` and every automatic and explicit cache breakpoint.
        `"5m"` writes bill 1.25 times base input, and `"1h"` writes bill twice base input.
        Mixed TTLs require one-hour cache breakpoints before five-minute ones.
        Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching.
        `pricing` supplies rates and modifiers.
        `inference_geo` requests an inference geography.
        `service_tier` requests a tier, while the reported tier selects pricing.

        Raises:
            ValueError: `provider_name` contradicts a Bedrock client class.
        """
        super().__init__(
            client=client,
            model=model,
            provider_name=provider_name,
            automatic_cache_breakpoints_default=False,
        )
        self.client: AnthropicClient = client_without_retries(client)
        self.pricing: AnthropicPricingTable = pricing
        self.cache_ttl: CacheTTL = cache_ttl
        self._uses_top_level_cache_control: bool = not isinstance(
            self.client, AsyncAnthropicBedrock
        )
        self.service_tier: AnthropicServiceTier | None = service_tier
        self.inference_geo: str | None = inference_geo

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return stored request configuration outside `Binding`."""
        return {
            "uses_top_level_cache_control": self._uses_top_level_cache_control,
            "cache_ttl": self.cache_ttl,
            "inference_geo": self.inference_geo,
            "service_tier": self.service_tier,
        }

    def _precompute_fields(self, binding: Binding) -> _AnthropicPrecomputedFields:
        """Precompute request fields and the remaining `message_cache_breakpoint_budget`.

        A string `system_prompt` becomes one system block.
        A parts `system_prompt` becomes one block per part and preserves its cache breakpoints.
        `automatic_cache_breakpoints` places a cache breakpoint on the last system block or the last tool.
        Binding cache breakpoints count toward the limit of four before message cache breakpoints.

        Raises:
            ValueError: `max_completion_tokens` is `None`, because the Messages API requires `max_tokens`.
            ValueError: Binding cache breakpoints exceed four, `extra_body` conflicts, or `system_prompt` is empty.
            ValueError: A provider-executed tool type is unsupported or code execution lacks a qualifying web tool.
            ValueError: Provider-executed tools use another provider or web-search rates are invalid.
            TypeError: `tool_choice` is `AllowedToolsChoice`, which Anthropic does not support.
        """
        if binding.max_completion_tokens is None:
            raise ValueError(
                "Anthropic requires max_completion_tokens; pass max_completion_tokens= to bind"
            )
        reject_extra_body_keys_the_adapter_populates(
            binding.extra_body, populated_keys=_ADAPTER_POPULATED_WIRE_KEYS
        )
        provider_executed_tools = _provider_executed_tools(binding.provider_executed_tools)
        if binding.provider_executed_tools and self.provider_name != "anthropic":
            raise ValueError("Anthropic provider_executed_tools require provider_name='anthropic'")
        if provider_executed_tools.web_search:
            require_finite_nonnegative_rate(
                rate_name="web_search_usd_per_invocation",
                rate=self.pricing.web_search_usd_per_invocation,
            )
        system: list[TextBlockParam] | Omit = omit
        bind_cache_breakpoint_count = 0
        if binding.system_prompt is not None:
            system_blocks: list[TextBlockParam] = []
            if isinstance(binding.system_prompt, str):
                system_blocks.append({"type": "text", "text": binding.system_prompt})
            else:
                if not binding.system_prompt:
                    raise ValueError(
                        "system_prompt is an empty tuple of parts; bind rejects this, "
                        "so it can only come from a directly constructed Binding"
                    )
                for part in binding.system_prompt:
                    system_block: TextBlockParam = {"type": "text", "text": part.text}
                    if part.cache_breakpoint:
                        system_block["cache_control"] = _cache_control_param(self.cache_ttl)
                    system_blocks.append(system_block)
            if binding.automatic_cache_breakpoints:
                system_blocks[-1]["cache_control"] = _cache_control_param(self.cache_ttl)
            bind_cache_breakpoint_count = sum(
                1 for block in system_blocks if "cache_control" in block
            )
            system = system_blocks
        tools: list[ToolUnionParam] | Omit = omit
        tool_choice: ToolChoiceParam | Omit = omit
        if binding.tool_schemas or binding.provider_executed_tools:
            cache_breakpoint_on_last_tool = (
                binding.automatic_cache_breakpoints and binding.system_prompt is None
            )
            tools = _wire_tools(
                binding.tool_schemas,
                binding.provider_executed_tools,
                cache_breakpoint_on_last_tool=cache_breakpoint_on_last_tool,
                cache_ttl=self.cache_ttl,
            )
            bind_cache_breakpoint_count += sum(1 for tool in tools if "cache_control" in tool)
            tool_choice = _wire_tool_choice(
                binding.tool_choice, parallel_tool_calls=binding.parallel_tool_calls
            )
        automatic_cache_breakpoint_count = 1 if binding.automatic_cache_breakpoints else 0
        message_cache_breakpoint_budget = (
            _MAX_CACHE_BREAKPOINTS_PER_REQUEST
            - bind_cache_breakpoint_count
            - automatic_cache_breakpoint_count
        )
        if message_cache_breakpoint_budget < 0:
            raise ValueError(
                f"the binding writes {bind_cache_breakpoint_count + automatic_cache_breakpoint_count} cache breakpoints, "
                f"over the API's limit of {_MAX_CACHE_BREAKPOINTS_PER_REQUEST} per request; "
                "set cache_breakpoint=False on some system parts or remove cache_control from provider_executed_tools"
            )
        output_config: OutputConfigParam | Omit = omit
        thinking: ThinkingConfigParam | Omit = omit
        if binding.reasoning_level is not None:
            # cast: `reasoning_level` deliberately exceeds the SDK effort literal.
            output_config = cast("OutputConfigParam", {"effort": binding.reasoning_level})
            thinking = {"type": "adaptive"}
        return _AnthropicPrecomputedFields(
            model=self.model,
            max_tokens=binding.max_completion_tokens,
            temperature=(binding.temperature if binding.temperature is not None else omit),
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            output_config=output_config,
            thinking=thinking,
            service_tier=self.service_tier if self.service_tier is not None else omit,
            inference_geo=self.inference_geo if self.inference_geo is not None else omit,
            cache_control=(
                _cache_control_param(self.cache_ttl)
                if binding.automatic_cache_breakpoints and self._uses_top_level_cache_control
                else omit
            ),
            automatic_message_cache_breakpoint=(
                binding.automatic_cache_breakpoints and not self._uses_top_level_cache_control
            ),
            cache_ttl=self.cache_ttl,
            message_cache_breakpoint_budget=message_cache_breakpoint_budget,
            extra_body=binding.extra_body,
            provider_executed_tools=provider_executed_tools,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output without I/O.

        Raises:
            ValueError: `binding.max_completion_tokens` is `None`, or `binding` contains other unsupported values.
            TypeError: `binding.tool_choice` is `AllowedToolsChoice`.
        """
        return _BoundAnthropicText(
            adapter=self, precomputed_fields=self._precompute_fields(binding)
        )

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format without I/O.

        Raises:
            ValueError: `binding.max_completion_tokens` is `None`, or `binding` contains other unsupported values.
            TypeError: `binding.tool_choice` is `AllowedToolsChoice`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` cannot produce a JSON schema.
            pydantic.PydanticUserError: `response_format` is not fully defined.
        """
        return _BoundAnthropicStructured(
            adapter=self,
            precomputed_fields=self._precompute_fields(binding),
            response_format=response_format,
        )

    @override
    def request_failure(self, error: Exception) -> RequestFailure:
        """Delegate to anthropic_request_failure, whose docstring names the tables and the defaults."""
        return anthropic_request_failure(error)

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        """Read the request-id header off the SDK exception.

        `APIStatusError` alone carries `request_id` from response headers in anthropic 0.120.0.
        """
        if isinstance(error, anthropic.APIStatusError):
            return error.request_id
        return None


class _AnthropicStream(AdapterStream):
    """One open Messages stream, backed by the SDK's AsyncMessageStream."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncMessageStream[Any],
        pricing: AnthropicPricingTable,
        provider_executed_tools: _AnthropicProviderExecutedTools = _NO_ANTHROPIC_PROVIDER_EXECUTED_TOOLS,
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._provider_executed_tools = provider_executed_tools
        self._snapshot_started = False
        self._billing_complete = False
        """Whether an event has been accumulated, which is what makes current_message_snapshot readable."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate SDK events into `StreamItem` values.

        `input_json_delta` yields `ToolCallDelta` only for `tool_use` blocks.
        The adapter reads the id and name from the SDK message snapshot.
        `server_tool_use` streams no delta and becomes `RawPart` in `final()`.
        `REASONING_PART_SEPARATOR` separates thinking blocks that emitted text.
        Empty deltas and redacted thinking emit no reasoning text or separator.

        Yields:
            Stream items for SDK events langchaint models.

        Raises:
            StreamProtocolError: The stream ends without a stop reason.
        """
        reasoning_delta_yielded = False
        separator_pending = False
        async for event in self._sdk_stream:
            self._snapshot_started = True
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield event.delta.text
                elif event.delta.type == "thinking_delta" and event.delta.thinking:
                    if separator_pending:
                        separator_pending = False
                        yield ReasoningDelta(text=REASONING_PART_SEPARATOR)
                    reasoning_delta_yielded = True
                    yield ReasoningDelta(text=event.delta.thinking)
                elif event.delta.type == "input_json_delta" and event.delta.partial_json:
                    block = self._sdk_stream.current_message_snapshot.content[event.index]
                    if block.type == "tool_use":
                        yield ToolCallDelta(
                            id=block.id,
                            name=block.name,
                            partial_args_json=event.delta.partial_json,
                        )
            elif event.type == "content_block_stop":
                if event.content_block.type == "tool_use":
                    yield ToolCall(
                        id=event.content_block.id,
                        name=event.content_block.name,
                        args_json=json.dumps(event.content_block.input),
                    )
                elif event.content_block.type == "thinking":
                    separator_pending = reasoning_delta_yielded
        if self._sdk_stream.current_message_snapshot.stop_reason is None:
            raise StreamProtocolError("stream ended without a stop reason")
        self._billing_complete = True

    @override
    async def final(self) -> anthropic.types.Message:
        """Return the message the SDK assembled from the stream's events, after the stream ends."""
        return await self._sdk_stream.get_final_message()

    @override
    def provider_billing(self) -> ProviderBilling | None:
        """Return snapshot billing after the first event, or `None` before it.

        Anthropic 0.120.0 provides required input tokens from `message_start`.
        Optional cache counters arrive with `message_delta`.
        """
        if not self._snapshot_started:
            return None
        return _billing_from_sdk_usage(
            self._sdk_stream.current_message_snapshot.usage,
            self._pricing,
            provider_executed_tools=self._provider_executed_tools,
            billing_complete=self._billing_complete,
        )

    @override
    def request_id(self) -> str | None:
        """Read the request-id header off the response the SDK stream is reading.

        `AsyncMessageStream.request_id` is readable when the stream opens in anthropic 0.120.0.
        """
        return self._sdk_stream.request_id

    @override
    async def close(self) -> None:
        await self._sdk_stream.close()


class _BoundAnthropic[OutputT](BoundAdapter[OutputT], ABC):
    """What both anthropic bindings share: the request path, and what a response says about itself."""

    def __init__(
        self, *, adapter: AnthropicMessagesAdapter, precomputed_fields: _AnthropicPrecomputedFields
    ) -> None:
        self._adapter = adapter
        self._precomputed_fields = precomputed_fields

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Price counters using reported response metadata.

        Raises:
            TypeError: raw is not an anthropic Message.
            pydantic.ValidationError: the message reports a negative counter.
        """
        return _billing_from_sdk_usage(
            _as_message(raw).usage,
            pricing=self._adapter.pricing,
            provider_executed_tools=self._precomputed_fields.provider_executed_tools,
        )

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Combine the message's id and model with request_id.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        message = _as_message(raw)
        return ResponseIdentity(
            model_served=message.model,
            response_id=message.id,
            request_id=request_id,
        )

    @override
    def build_request_params(
        self, messages: Sequence[Message]
    ) -> RequestParams | RejectedMessages:
        """Convert messages under the binding's precomputed fields."""
        wire_messages = _request_messages(messages, self._precomputed_fields)
        if isinstance(wire_messages, RejectedMessages):
            return wire_messages
        return _AnthropicRequestParams(
            precomputed=self._precomputed_fields, messages=wire_messages
        )

    @override
    async def open_stream(self, request_params: RequestParams) -> AdapterStream:
        """Open one messages.stream and return the live stream.

        Raises:
            TypeError: request was built by another adapter.
            Exception: The SDK fails to open the stream.
        """
        params = narrowed_request_params(request_params, _AnthropicRequestParams)
        precomputed = params.precomputed
        manager = self._adapter.client.messages.stream(
            model=precomputed.model,
            max_tokens=precomputed.max_tokens,
            system=precomputed.system,
            tools=precomputed.tools,
            tool_choice=precomputed.tool_choice,
            output_config=precomputed.output_config,
            thinking=precomputed.thinking,
            service_tier=precomputed.service_tier,
            inference_geo=precomputed.inference_geo,
            cache_control=precomputed.cache_control,
            messages=params.messages,
            extra_body=_extra_body_with_temperature(precomputed),
        )
        return _AnthropicStream(
            sdk_stream=await manager.__aenter__(),
            pricing=self._adapter.pricing,
            provider_executed_tools=precomputed.provider_executed_tools,
        )


class _BoundAnthropicText(_BoundAnthropic[str]):
    """Text-bound adapter: output is the concatenated text of the assistant message."""

    @override
    def interpret(self, raw: BaseModel) -> UsableResponse[str]:
        """Read the assistant message, whose concatenated text is this binding's output.

        Every message supplies its text despite an unhandled stop reason.

        Raises:
            TypeError: `raw` is not an anthropic `Message`.
        """
        message = _as_message(raw)
        assistant_message = _assistant_message_from(message)
        return _usable_response(message, assistant_message.text, assistant_message)


class _BoundAnthropicStructured[ModelT: BaseModel](_BoundAnthropic[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the assistant message's text."""

    def __init__(
        self,
        *,
        adapter: AnthropicMessagesAdapter,
        precomputed_fields: _AnthropicPrecomputedFields,
        response_format: type[ModelT],
    ) -> None:
        """Precompute the request's output_config, the JSON-schema format merged into the binding's.

        The format matches `messages.parse(output_format=...)`.
        The merge is what keeps a reasoning effort the binding set: output_config carries both keys.
        The merged value replaces the binding value in every request.
        """
        self._output_type_adapter: TypeAdapter[ModelT] = TypeAdapter(response_format)
        output_format = JSONOutputFormatParam(
            schema=transform_schema(self._output_type_adapter.json_schema()), type="json_schema"
        )
        bound_output_config = precomputed_fields.output_config
        output_config: OutputConfigParam = (
            {"format": output_format}
            if isinstance(bound_output_config, Omit)
            else {**bound_output_config, "format": output_format}
        )
        super().__init__(
            adapter=adapter,
            precomputed_fields=replace(precomputed_fields, output_config=output_config),
        )

    def _parsed_outcome(
        self, message: anthropic.types.Message, assistant_message: AssistantMessage
    ) -> ResponseOutcome[ModelT | None]:
        """Validate text, return `None` for an assistant message with tool calls, or return a failure variant.

        Validation occurs after the request records its message and billing.
        Stop reasons take precedence over schema validation.
        Every failure variant carries `assistant_message`.
        """
        validation_error: ValidationError | None = None
        text = _first_text_block_text(message)
        if text is not None:
            try:
                output = self._output_type_adapter.validate_json(text)
                return _usable_response(message, output, assistant_message)
            except ValidationError as rejection:
                validation_error = rejection
        unfinished_message = _unfinished_message_or_none(
            message, assistant_message=assistant_message
        )
        if unfinished_message is not None:
            return unfinished_message
        if message.stop_reason == "tool_use":
            return _usable_response(message, None, assistant_message)
        if message.stop_reason == "refusal":
            return Refusal(assistant_message=assistant_message)
        if message.stop_reason == "max_tokens":
            return MaxCompletionTokensExceeded(assistant_message=assistant_message)
        if message.stop_reason == "model_context_window_exceeded":
            return ContextWindowExceeded(assistant_message=assistant_message)
        if validation_error is not None:
            return SchemaViolation(
                validation_error_json=validation_error.json(include_url=False),
                assistant_message=assistant_message,
            )
        return EmptyAssistantMessage(assistant_message=assistant_message)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        """Validate the assistant message's text into the instance, or report why the message produced none.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        message = _as_message(raw)
        assistant_message = _assistant_message_from(message)
        return self._parsed_outcome(message, assistant_message)
