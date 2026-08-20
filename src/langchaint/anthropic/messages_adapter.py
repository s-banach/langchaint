"""Implement the Anthropic Messages API through the official anthropic SDK.

The following SDK facts were verified against anthropic 1.0.0.
A structured binding sends the same `output_config.format` as `messages.parse(output_format=Model)`.
Its schema uses `transform_schema(TypeAdapter(Model).json_schema())` and type `"json_schema"`.
The adapter validates response text after the SDK exposes the complete message and billing.
SDK parsing may reject text before final output and cache counters arrive.
`messages.stream` assembles deltas, and `get_final_message()` returns the message.
The SDK reports no all-inclusive input total.

The API requires an unchanged replay of the latest assistant turn's thinking during tool use.
The API filters earlier thinking blocks.
The API rejects consecutive thinking blocks outside their original order.
The adapter replays every `ReasoningPart` in turn order.

`automatic_cache_breakpoints=True` marks the frozen prefix and the last message block.
The frozen prefix ends at the system prompt or at the last tool when no system prompt exists.
`automatic_cache_breakpoints=False` adds no automatic marker.
The adapter never sends top-level automatic `cache_control` because Bedrock does not support it.
A marked user part adds `cache_control` to its text or image block.
A marked final `ToolMessage` part adds `cache_control` to its enclosing `tool_result` block.
A marked non-final `ToolMessage` part returns `InvalidRequest` because the boundary would move.
A parts `system_prompt` produces one system block per part and preserves marked boundaries.

The API accepts at most four `cache_control` markers per request.
Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching, read 2026-07-25.
System marks and automatic marks reduce `message_mark_budget`.
Binding fails with `ValueError` when its marks exceed the limit.
The adapter marks only the latest message parts that fit `message_mark_budget`.
Each marker uses `cache_ttl`.
The default `"5m"` omits the API-default `ttl` key.
`"1h"` sends `ttl="1h"` and uses `cache_write_1h_usd_per_million_tokens`.

Content mappings were verified against anthropic 0.121.0.
- `ImagePart` becomes `Base64ImageSourceParam`.
- `ImageUrlPart` becomes `URLImageSourceParam`.
- `AudioPart` returns `InvalidRequest` inside `UserMessage` and `ToolMessage`.
- `Usage.server_tool_use` reports web-search invocation counts.

Request and response mappings:
- `ToolMessage` becomes `tool_result` inside a user message.
- Consecutive `ToolMessage` values share one user message because the API requires alternating roles.
- `end_turn`, `tool_use`, `max_tokens`, and `refusal` preserve their `stop_reason` values.
- Other `stop_reason` values map to `"other"`.
- `reasoning_effort` sends `output_config.effort` with `thinking={"type": "adaptive"}`.
- The adapter sends neither field alone because effort applies only to adaptive thinking.
- `ReasoningEffort` accepts values wider than the SDK literal and sends each value unchanged.
- The provider reports unsupported values through its own error.
- The adapter never sends `thinking.display`.
- The SDK default `"summarized"` returns thinking text, while `"omitted"` redacts it.
"""

import base64
import json
from abc import ABC
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite, nan
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
    AdapterResult,
    AdapterStream,
    AllowedToolsChoice,
    Binding,
    BoundAdapter,
    ContextWindowExceeded,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    ReasoningDelta,
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    SpecificToolChoice,
    StreamItem,
    ToolCallDelta,
    ToolChoice,
    UnfinishedTurn,
    narrowed_request,
    record_parse_fallthrough,
    reject_extra_body_keys_the_adapter_populates,
    request_json,
    retry_after_seconds_from_headers,
    terminal_classification_from_response,
    validated_provider_executed_tool_types,
    verdict_from_transient_error,
    verdict_under_retry_directive,
)
from langchaint.call import ResponseIdentity
from langchaint.exceptions import StreamProtocolError, TransientError
from langchaint.messages import (
    AssistantMessage,
    ContentPart,
    Message,
    RawPart,
    ReasoningPart,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnPart,
    UserMessage,
)
from langchaint.pricing import (
    Billing,
    category_cost,
    invocation_cost_in_usd,
    require_finite_nonnegative_rate,
)
from langchaint.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict
from langchaint.tools import ToolSchema
from langchaint.usage import Usage

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
"""The statuses that reject this request; a resend fails again.

Every one but 422 is on anthropic's errors page.
422 is not: every client the adapter supports raises UnprocessableEntityError for it
(anthropic 0.120.2).
"""

PARSE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""How often parse_anthropic fell to a status-family default, keyed by status and error type.

A diagnostic surface, read by no decision: a growing key names a status or error type the
tables above should learn.
"""

_CACHE_MARKER_REQUEST_LIMIT = 4
"""The API allows at most 4 cache_control markers per request, the binding's own included."""

type CacheTTL = Literal["5m", "1h"]
"""A cache entry's time to live, the two tiers the API offers; writes bill 1.25x ("5m") or 2x ("1h") base input."""

type AnthropicServiceTier = Literal["auto", "standard_only"]
"""What a request may ask for (anthropic 0.120.0).

"auto" is a ceiling, not a selector: the SDK documents the parameter as whether to use priority
capacity if available or standard capacity, so no request value names priority.
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
    return client.with_options(max_retries=0, http_client=client._client)  # noqa: SLF001


@dataclass(frozen=True, kw_only=True)
class AnthropicRates:
    """Anthropic token rates for one service tier."""

    input_cache_none_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_5m_usd_per_million_tokens: float
    cache_write_1h_usd_per_million_tokens: float

    def price(  # noqa: PLR0913 (anthropic splits the cache-write counter that other providers report as one)
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
    ) -> Billing:
        """Price one response's counters, the two cache-write TTLs each at their own rate.

        `Usage.input_tokens_cache_write` sums both write counters.
        `input_tokens_cache_write_cost_in_usd` sums both write costs.
        The reported cache-write price blends the rates to reproduce that cost.
        Zero writes use Anthropic's default five-minute rate.

        Raises:
            pydantic.ValidationError: a counter is negative.
        """
        cache_write_5m_cost_in_usd = category_cost(
            input_tokens_cache_write_5m,
            usd_per_million_tokens=self.cache_write_5m_usd_per_million_tokens,
        )
        cache_write_1h_cost_in_usd = category_cost(
            input_tokens_cache_write_1h,
            usd_per_million_tokens=self.cache_write_1h_usd_per_million_tokens,
        )
        input_tokens_cache_write = input_tokens_cache_write_5m + input_tokens_cache_write_1h
        input_tokens_cache_write_cost_in_usd = (
            cache_write_5m_cost_in_usd + cache_write_1h_cost_in_usd
        )
        return Billing(
            usage=Usage(
                input_tokens_cache_read=input_tokens_cache_read,
                input_tokens_cache_write=input_tokens_cache_write,
                input_tokens_cache_none=input_tokens_cache_none,
                output_tokens=output_tokens,
                output_tokens_reasoning=output_tokens_reasoning,
                input_tokens_cache_read_cost_in_usd=category_cost(
                    input_tokens_cache_read,
                    usd_per_million_tokens=self.cache_read_usd_per_million_tokens,
                ),
                input_tokens_cache_write_cost_in_usd=input_tokens_cache_write_cost_in_usd,
                input_tokens_cache_none_cost_in_usd=category_cost(
                    input_tokens_cache_none,
                    usd_per_million_tokens=self.input_cache_none_usd_per_million_tokens,
                ),
                output_tokens_cost_in_usd=category_cost(
                    output_tokens,
                    usd_per_million_tokens=self.output_usd_per_million_tokens,
                ),
                provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
            ),
            service_tier=service_tier,
            usage_raw=usage_raw,
            input_cache_none_usd_per_million_tokens=self.input_cache_none_usd_per_million_tokens,
            cache_read_usd_per_million_tokens=self.cache_read_usd_per_million_tokens,
            cache_write_usd_per_million_tokens=(
                input_tokens_cache_write_cost_in_usd * 1_000_000 / input_tokens_cache_write
                if input_tokens_cache_write
                else self.cache_write_5m_usd_per_million_tokens
            ),
            output_usd_per_million_tokens=self.output_usd_per_million_tokens,
        )

    def multiplied(self, multiplier: float) -> "AnthropicRates":
        """Return token rates multiplied by one value."""
        return AnthropicRates(
            input_cache_none_usd_per_million_tokens=(
                self.input_cache_none_usd_per_million_tokens * multiplier
            ),
            output_usd_per_million_tokens=self.output_usd_per_million_tokens * multiplier,
            cache_read_usd_per_million_tokens=(
                self.cache_read_usd_per_million_tokens * multiplier
            ),
            cache_write_5m_usd_per_million_tokens=(
                self.cache_write_5m_usd_per_million_tokens * multiplier
            ),
            cache_write_1h_usd_per_million_tokens=(
                self.cache_write_1h_usd_per_million_tokens * multiplier
            ),
        )


_UNPRICED_RATES = AnthropicRates(
    input_cache_none_usd_per_million_tokens=float("nan"),
    output_usd_per_million_tokens=float("nan"),
    cache_read_usd_per_million_tokens=float("nan"),
    cache_write_5m_usd_per_million_tokens=float("nan"),
    cache_write_1h_usd_per_million_tokens=float("nan"),
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
        if isinstance(multiplier, bool) or not isfinite(multiplier) or multiplier <= 0:
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


def _cache_control_param(cache_ttl: CacheTTL) -> CacheControlEphemeralParam:
    """Build one cache_control marker; "5m" omits the ttl key because it is the API default.

    The "5m" wire form must stay byte-stable across releases,
    so that upgrading langchaint alone cannot invalidate a caller's live cache entry.
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
class _AnthropicProviderTools:
    """Validated provider-executed tool categories needed for billing."""

    web_search: bool = False
    web_fetch: bool = False
    tool_search: bool = False
    code_execution_exempt: bool = False


_NO_ANTHROPIC_PROVIDER_TOOLS = _AnthropicProviderTools()


def _provider_tools(
    provider_executed_tools: tuple[Mapping[str, object], ...],
) -> _AnthropicProviderTools:
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
    return _AnthropicProviderTools(
        web_search=bool(tool_types & _WEB_SEARCH_TOOL_TYPES),
        web_fetch=bool(tool_types & _WEB_FETCH_TOOL_TYPES),
        tool_search=bool(tool_types & _TOOL_SEARCH_TOOL_TYPES),
        code_execution_exempt=code_execution and code_execution_exempt,
    )


@dataclass(frozen=True, kw_only=True)
class _AnthropicPrecomputedFields:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place;
    passing them as explicit keywords (never **kwargs) keeps the SDK's overload resolution intact.
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
    automatic_cache_breakpoints: bool
    cache_ttl: CacheTTL
    message_mark_budget: int
    """What the binding's own markers (system marks, the frozen-prefix and last-message markers) leave
    of the API's 4-marker request limit for per-request marked parts."""

    extra_body: Mapping[str, object] | None
    provider_tools: _AnthropicProviderTools


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
    """One messages request: the binding's precomputed fields and this call's converted messages."""

    precomputed: _AnthropicPrecomputedFields
    messages: list[MessageParam]

    @override
    def as_json(self) -> str:
        """Render the request as a JSON object, dropping every field left to the provider's default."""
        return request_json(self, omitted_class=Omit)


class _NotSendableError(Exception):
    """A Sequence[Message] this adapter will not put on the wire, raised by a conversion helper.

    `_request_messages` converts it to the `InvalidRequest` returned by `build_request`.
    Per-part converters raise it when a `Sequence[Message]` is unsendable.
    """

    def __init__(self, reason: str) -> None:
        """Store what cannot be sent; it becomes the InvalidRequest reason."""
        super().__init__(reason)
        self.reason = reason


def _part_block(
    part: ContentPart, *, message_class: type[UserMessage] | type[ToolMessage]
) -> TextBlockParam | ImageBlockParam:
    """Convert one ContentPart to its wire block.

    Raises:
        _NotSendableError: ContentPart has no wire form for message_class.
    """
    match part.kind:
        case "text":
            return {"type": "text", "text": part.text}
        case "image":
            if part.media_type not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
                raise _NotSendableError(
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
            raise _NotSendableError(
                f"AnthropicMessagesAdapter cannot send AudioPart inside "
                f"{message_class.__name__}.content: {missing_audio_type}"
            )


def _user_content_blocks(
    user_message: UserMessage,
) -> tuple[list[_ContentBlockParam], list[TextBlockParam | ImageBlockParam]]:
    """Convert one UserMessage.content value to wire blocks.

    The second element holds the blocks whose part sets cache_breakpoint, in content order;
    the caller applies the request-wide marker budget, so no marker is written here.

    Raises:
        _NotSendableError: _part_block rejects one ContentPart.
    """
    blocks: list[_ContentBlockParam] = []
    marked: list[TextBlockParam | ImageBlockParam] = []
    if isinstance(user_message.content, str):
        blocks.append({"type": "text", "text": user_message.content})
        return blocks, marked
    for part in user_message.content:
        block = _part_block(part, message_class=UserMessage)
        blocks.append(block)
        if part.cache_breakpoint:
            marked.append(block)
    return blocks, marked


def _tool_result_content(
    content: str | tuple[ContentPart, ...],
) -> str | list[TextBlockParam | ImageBlockParam]:
    """Convert one ToolMessage's content to the tool_result content field.

    A bare string passes through.
    A ContentPart tuple becomes TextBlockParam and ImageBlockParam values.

    Raises:
        _NotSendableError: _part_block rejects one ContentPart.
    """
    if isinstance(content, str):
        return content
    return [_part_block(part, message_class=ToolMessage) for part in content]


def _replayed_block(raw: Mapping[str, object]) -> _ContentBlockParam:
    """Copy a stored SDK block without reading or changing its fields.

    The copy prevents cache-marker writes from changing the stored block.
    A block from another provider passes through when it has a `type` key.

    Raises:
        _NotSendableError: `raw` lacks the `type` key required by anthropic 0.120.2 block parameters.
    """
    if "type" not in raw:
        raise _NotSendableError(
            "ReasoningPart.raw or RawPart.raw names no type key, "
            "so anthropic has no content block to send it as"
        )
    # cast: a deliberately-opaque value re-enters the typed API whose own serialization produced it.
    return cast("_ContentBlockParam", dict(raw))


def _assistant_content_blocks(assistant_message: AssistantMessage) -> list[_ContentBlockParam]:
    """Convert one AssistantMessage to wire blocks in turn order.

    `ReasoningPart.raw` and `RawPart.raw` pass through unchanged by their `type` keys.
    The API rejects modified thinking blocks and unknown `type` values.
    Empty `TextPart` values are omitted because the API rejects them.

    Raises:
        json.JSONDecodeError: `ToolCall.args_json` is invalid JSON.
        _NotSendableError: A stored block lacks a `type` key.
    """
    blocks: list[_ContentBlockParam] = []
    for part in assistant_message.turn:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append(TextBlockParam(type="text", text=part.text))
        elif isinstance(part, ToolCall):
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


def _tool_message_is_marked(tool_message: ToolMessage) -> bool:
    """Return whether the last part marks the enclosing `tool_result` block.

    Raises:
        _NotSendableError: A non-final part sets `cache_breakpoint` because the API marks the block end.
    """
    if isinstance(tool_message.content, str):
        return False
    marked_indexes = [
        index for index, part in enumerate(tool_message.content) if part.cache_breakpoint
    ]
    if not marked_indexes:
        return False
    if marked_indexes != [len(tool_message.content) - 1]:
        raise _NotSendableError(
            "cache_breakpoint on a ToolMessage part is honored only on the message's last part: "
            "the marker goes on the enclosing tool_result block, whose span ends at the last part"
        )
    return True


def _wire_messages(
    messages: Sequence[Message],
    *,
    automatic_cache_breakpoints: bool,
    cache_ttl: CacheTTL,
    message_mark_budget: int,
) -> list[MessageParam]:
    """Convert messages and apply the permitted cache markers.

    `automatic_cache_breakpoints` marks the last block unless it is a thinking block.
    A user part marks its own block, while a tool part marks its enclosing `tool_result`.
    The latest marks up to `message_mark_budget` are sent.

    Raises:
        _NotSendableError: A `ContentPart` lacks a wire form, a non-final tool part is marked, or raw lacks `type`.
        json.JSONDecodeError: `ToolCall.args_json` is invalid JSON.
    """
    wire: list[tuple[Literal["user", "assistant"], list[_ContentBlockParam]]] = []
    pending_tool_results: list[_ContentBlockParam] = []
    marked_blocks: list[TextBlockParam | ImageBlockParam | ToolResultBlockParam] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            wire.append(("user", list(pending_tool_results)))
            pending_tool_results.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_result_block: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": _tool_result_content(message.content),
                "is_error": message.is_error,
            }
            if _tool_message_is_marked(message):
                marked_blocks.append(tool_result_block)
            pending_tool_results.append(tool_result_block)
        elif isinstance(message, UserMessage):
            flush_tool_results()
            blocks, marked = _user_content_blocks(message)
            marked_blocks.extend(marked)
            wire.append(("user", blocks))
        else:
            flush_tool_results()
            wire.append(("assistant", _assistant_content_blocks(message)))
    flush_tool_results()
    if message_mark_budget > 0:
        for block in marked_blocks[-message_mark_budget:]:
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
) -> list[MessageParam] | InvalidRequest:
    """Convert messages under the binding's caching parameters, or report them unsendable.

    The one place a Sequence[Message] this adapter will not put on the wire becomes an InvalidRequest.
    An unparseable tool_call.args_json is one of those: the wire block holds the parsed arguments,
    so text that is not JSON has no block to go in.
    """
    try:
        return _wire_messages(
            messages,
            automatic_cache_breakpoints=precomputed_fields.automatic_cache_breakpoints,
            cache_ttl=precomputed_fields.cache_ttl,
            message_mark_budget=precomputed_fields.message_mark_budget,
        )
    except _NotSendableError as not_sendable:
        return InvalidRequest(reason=not_sendable.reason)
    except json.JSONDecodeError as not_json:
        return InvalidRequest(reason=f"a tool call's args_json is not valid JSON: {not_json}")


def _wire_tool_choice(tool_choice: ToolChoice, *, parallel_tool_calls: bool) -> ToolChoiceParam:
    """Convert the neutral tool choice; neutral "required" is Anthropic "any".

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

    cache_breakpoint_on_last_tool puts the frozen-prefix cache breakpoint on the last tool,
    used when no system prompt follows the tools to carry it.
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
        # Copying preserves caller state; mutating it would alter the binding's mapping.
        last_tool = dict(tools[-1])
        last_tool["cache_control"] = _cache_control_param(cache_ttl)
        # cast: the copied mapping remains wider than the SDK TypedDict union.
        tools[-1] = cast("ToolUnionParam", last_tool)
    return tools


def _normalized_stop_reason(stop_reason: str | None) -> StopReason:
    if stop_reason in ("end_turn", "tool_use", "max_tokens", "refusal"):
        return stop_reason
    if stop_reason == "model_context_window_exceeded":
        return "context_window_exceeded"
    return "other"


def _unfinished_turn_or_none(
    message: anthropic.types.Message, *, assistant_message: AssistantMessage
) -> UnfinishedTurn | None:
    """Return `UnfinishedTurn` for `pause_turn`, a null stop reason, or an unknown stop reason."""
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
    return UnfinishedTurn(
        reason=f"anthropic returned stop_reason {stop_reason!r}, which langchaint cannot continue",
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
    """Return the text of the turn's first text block, None when the turn holds none.

    Structured output validation uses this block.
    SDK parsing validates every text block and returns the first instance.
    """
    for block in message.content:
        if block.type == "text":
            return block.text
    return None


def _assistant_message_from(message: anthropic.types.Message) -> AssistantMessage:
    """Build `AssistantMessage` from SDK blocks in order.

    Thinking blocks become replayable `ReasoningPart` values.
    Redacted thinking has `text=None`.
    Unmodeled blocks become replayable `RawPart` values.
    """
    turn: list[TurnPart] = []
    for block in message.content:
        if block.type == "text":
            turn.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            turn.append(ToolCall(id=block.id, name=block.name, args_json=json.dumps(block.input)))
        elif block.type == "thinking":
            turn.append(
                ReasoningPart(
                    raw=block.model_dump(mode="python", exclude_none=True),
                    text=block.thinking or None,
                )
            )
        elif block.type == "redacted_thinking":
            turn.append(ReasoningPart(raw=block.model_dump(mode="python", exclude_none=True)))
        else:
            turn.append(RawPart(raw=block.model_dump(mode="python", exclude_none=True)))
    return AssistantMessage(turn=tuple(turn))


def _priced_tier(
    service_tier: _AnthropicReportedServiceTier | None,
) -> _AnthropicReportedServiceTier:
    """Normalize a missing reported tier to `standard`.

    Bedrock responses need this default because Anthropic service tiers do not apply.
    """
    return service_tier if service_tier is not None else _STANDARD_TIER


def _billing_from_sdk_usage(
    usage: anthropic.types.Usage,
    pricing: AnthropicPricingTable,
    *,
    provider_tools: _AnthropicProviderTools = _NO_ANTHROPIC_PROVIDER_TOOLS,
    billing_complete: bool = True,
) -> Billing:
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
    output_tokens_details = usage.output_tokens_details
    input_tokens_cache_write_5m = usage.cache_creation_input_tokens or 0
    input_tokens_cache_write_1h = 0
    if usage.cache_creation is not None:
        input_tokens_cache_write_5m = usage.cache_creation.ephemeral_5m_input_tokens
        input_tokens_cache_write_1h = usage.cache_creation.ephemeral_1h_input_tokens
    service_tier = _priced_tier(usage.service_tier)
    rates = pricing.rates_for(
        service_tier=usage.service_tier,
        inference_geo=usage.inference_geo,
    )
    server_tool_use = usage.server_tool_use
    web_search_requests = 0 if server_tool_use is None else server_tool_use.web_search_requests
    provider_executed_tool_cost_in_usd = invocation_cost_in_usd(
        web_search_requests,
        usd_per_invocation=pricing.web_search_usd_per_invocation,
    )
    if provider_tools.web_search and not billing_complete:
        provider_executed_tool_cost_in_usd = nan
    if server_tool_use is not None:
        known_zero_fee_counters: set[str] = set()
        if provider_tools.web_fetch:
            known_zero_fee_counters.add("web_fetch_requests")
        if provider_tools.tool_search:
            known_zero_fee_counters.add("tool_search_requests")
        if provider_tools.code_execution_exempt:
            known_zero_fee_counters.add("code_execution_requests")
        for counter_name, counter in server_tool_use.model_dump().items():
            if counter_name == "web_search_requests" or counter_name in known_zero_fee_counters:
                continue
            if counter_name.endswith("_requests") and counter:
                provider_executed_tool_cost_in_usd = nan
    return rates.price(
        service_tier=service_tier,
        usage_raw=usage,
        input_tokens_cache_read=usage.cache_read_input_tokens or 0,
        input_tokens_cache_write_5m=input_tokens_cache_write_5m,
        input_tokens_cache_write_1h=input_tokens_cache_write_1h,
        input_tokens_cache_none=usage.input_tokens,
        output_tokens=usage.output_tokens,
        output_tokens_reasoning=(
            output_tokens_details.thinking_tokens if output_tokens_details is not None else 0
        ),
        provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
    )


def _adapter_result[OutputT](
    message: anthropic.types.Message, output: OutputT, assistant_message: AssistantMessage
) -> AdapterResult[OutputT]:
    """Normalize one completed message around already-extracted output and its turn."""
    return AdapterResult(
        output=output,
        assistant_message=assistant_message,
        stop_reason=_normalized_stop_reason(message.stop_reason),
    )


def parse_anthropic(failure: Exception) -> Verdict:
    """Map one AnthropicMessagesAdapter.failure_types exception to its verdict.

    _verdict_from_anthropic_tables reads the status and the error type.
    On every status but 200 the provider's own x-should-retry directive then overrides that verdict,
    through verdict_under_retry_directive, which states the rule.
    Status 200 identifies a mid-stream error, so its headers do not override the table verdict.
    A retry-after header only fills a verdict's retry_after.
    A TransientError takes verdict_from_transient_error's shared mapping.
    Never raises: an Exception outside failure_types is DoNotRetry, counted as a fallthrough.
    """
    if isinstance(failure, TransientError):
        return verdict_from_transient_error(failure)
    if not isinstance(failure, anthropic.APIStatusError):
        record_parse_fallthrough(
            PARSE_FALLTHROUGH_COUNTS,
            parse_name="parse_anthropic",
            status_code=None,
            error_type=type(failure).__name__,
        )
        return DoNotRetry()
    retry_after = retry_after_seconds_from_headers(failure.response.headers)
    verdict = _verdict_from_anthropic_tables(failure, retry_after)
    if failure.status_code == 200:
        return verdict
    return verdict_under_retry_directive(
        verdict, headers=failure.response.headers, retry_after=retry_after
    )


def _verdict_from_anthropic_tables(
    failure: anthropic.APIStatusError, retry_after: float | None
) -> Verdict:
    """Classify one `APIStatusError` by status and error type.

    Source: https://platform.claude.com/docs/en/api/errors, read 2026-08-01.
    Error types override status because stream errors may carry the live response's 200 status.
    Unlisted 5xx statuses return `RetryThisOne`.
    Other unlisted statuses return `DoNotRetry`.
    """
    for error_types, statuses, verdict_class in (
        (_PAUSE_ERROR_TYPES, _PAUSE_STATUSES, PauseAll),
        (_RETRY_THIS_ONE_ERROR_TYPES, _RETRY_THIS_ONE_STATUSES, RetryThisOne),
    ):
        if failure.type in error_types or failure.status_code in statuses:
            if failure.status_code not in statuses:
                record_parse_fallthrough(
                    PARSE_FALLTHROUGH_COUNTS,
                    parse_name="parse_anthropic",
                    status_code=failure.status_code,
                    error_type=failure.type,
                )
            return verdict_class(retry_after=retry_after)
    if failure.status_code in _DO_NOT_RETRY_STATUSES:
        return DoNotRetry()
    record_parse_fallthrough(
        PARSE_FALLTHROUGH_COUNTS,
        parse_name="parse_anthropic",
        status_code=failure.status_code,
        error_type=failure.type,
    )
    if failure.status_code >= 500:
        return RetryThisOne(retry_after=retry_after)
    return DoNotRetry()


class AnthropicMessagesAdapter(Adapter):
    """Adapter over an AsyncAnthropic, AsyncAnthropicBedrock, or AsyncAnthropicBedrockMantle client.

    All three clients expose `messages.stream` and `with_options`.
    `default_max_completion_tokens` supplies required `max_tokens` when the binding omits it.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = {
        AsyncAnthropicBedrock: "aws.bedrock",
        AsyncAnthropicBedrockMantle: "aws.bedrock",
    }
    """AsyncAnthropic is deliberately absent: it reaches whatever its base_url points at.

    Both classes here speak Bedrock's auth and URL scheme, so the class fixes the provider,
    and the caller's stated value stands for anything else.
    """

    def __init__(  # noqa: PLR0913 (each request and billing parameter remains explicit)
        self,
        *,
        client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
        model: str,
        pricing: AnthropicPricingTable,
        provider_name: str,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
        inference_geo: str | None = None,
    ) -> None:
        """Store request, caching, and pricing configuration without sending a request.

        `provider_name` is `"anthropic"` for `AsyncAnthropic` and `"aws.bedrock"` for Bedrock clients.
        The stored client disables SDK retries and preserves custom Bedrock transports.
        `cache_ttl` applies to every automatic and explicit cache marker.
        `"5m"` writes bill 1.25 times base input, and `"1h"` writes bill twice base input.
        Mixed TTLs require one-hour markers before five-minute markers.
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
        self.default_max_completion_tokens: int = default_max_completion_tokens
        self.cache_ttl: CacheTTL = cache_ttl
        self.service_tier: AnthropicServiceTier | None = service_tier
        self.inference_geo: str | None = inference_geo

    def _precompute_fields(self, binding: Binding) -> _AnthropicPrecomputedFields:
        """Precompute request fields and the remaining `message_mark_budget`.

        A string `system_prompt` becomes one system block.
        A parts `system_prompt` becomes one block per part and preserves marks.
        `automatic_cache_breakpoints` marks the last system block or the last tool.
        Binding marks use the four-marker limit before message marks.

        Raises:
            ValueError: Binding marks exceed four, `extra_body` conflicts, or `system_prompt` is empty.
            ValueError: A provider-executed tool type is unsupported or code execution lacks a qualifying web tool.
            ValueError: Provider-executed tools use another provider or web-search rates are invalid.
            TypeError: `tool_choice` is `AllowedToolsChoice`, which Anthropic does not support.
        """
        reject_extra_body_keys_the_adapter_populates(
            binding.extra_body, populated_keys=_ADAPTER_POPULATED_WIRE_KEYS
        )
        provider_tools = _provider_tools(binding.provider_executed_tools)
        if binding.provider_executed_tools and self.provider_name != "anthropic":
            raise ValueError("Anthropic provider_executed_tools require provider_name='anthropic'")
        if provider_tools.web_search:
            require_finite_nonnegative_rate(
                rate_name="web_search_usd_per_invocation",
                rate=self.pricing.web_search_usd_per_invocation,
            )
        max_tokens = binding.inference_params.max_completion_tokens
        system: list[TextBlockParam] | Omit = omit
        bind_marker_count = 0
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
            bind_marker_count = sum(1 for block in system_blocks if "cache_control" in block)
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
            bind_marker_count += sum(1 for tool in tools if "cache_control" in tool)
            tool_choice = _wire_tool_choice(
                binding.tool_choice, parallel_tool_calls=binding.parallel_tool_calls
            )
        last_message_marker_count = 1 if binding.automatic_cache_breakpoints else 0
        message_mark_budget = (
            _CACHE_MARKER_REQUEST_LIMIT - bind_marker_count - last_message_marker_count
        )
        if message_mark_budget < 0:
            raise ValueError(
                f"the binding writes {bind_marker_count + last_message_marker_count} cache markers, "
                f"over the API's limit of {_CACHE_MARKER_REQUEST_LIMIT} per request; "
                f"unmark some system parts"
            )
        output_config: OutputConfigParam | Omit = omit
        thinking: ThinkingConfigParam | Omit = omit
        if binding.inference_params.reasoning_effort is not None:
            # cast: `ReasoningEffort` deliberately exceeds the SDK effort literal.
            output_config = cast(
                "OutputConfigParam", {"effort": binding.inference_params.reasoning_effort}
            )
            thinking = {"type": "adaptive"}
        return _AnthropicPrecomputedFields(
            model=self.model,
            max_tokens=(
                max_tokens if max_tokens is not None else self.default_max_completion_tokens
            ),
            temperature=(
                binding.inference_params.temperature
                if binding.inference_params.temperature is not None
                else omit
            ),
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            output_config=output_config,
            thinking=thinking,
            service_tier=self.service_tier if self.service_tier is not None else omit,
            inference_geo=self.inference_geo if self.inference_geo is not None else omit,
            automatic_cache_breakpoints=binding.automatic_cache_breakpoints,
            cache_ttl=self.cache_ttl,
            message_mark_budget=message_mark_budget,
            extra_body=binding.extra_body,
            provider_tools=provider_tools,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output; pure conversion, no I/O.

        Propagates _precompute_fields' ValueError and TypeError.
        """
        return _BoundAnthropicText(
            adapter=self, precomputed_fields=self._precompute_fields(binding)
        )

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format; pure conversion, no I/O.

        Propagates _precompute_fields' ValueError and TypeError.
        """
        return _BoundAnthropicStructured(
            adapter=self,
            precomputed_fields=self._precompute_fields(binding),
            response_format=response_format,
        )

    failure_types: ClassVar[tuple[type[Exception], ...]] = (
        anthropic.APIStatusError,
        TransientError,
    )
    """The exceptions parse_anthropic maps to a verdict.

    APIStatusError catches every error status, not only the ones with their own subclass:
    _make_status_error returns a specific subclass only for the statuses it lists and the bare
    APIStatusError for every other one (anthropic 0.120.2, Bedrock clients included), so a
    subclass list would silently drop whatever status the provider adds next.
    """

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Delegate to parse_anthropic, whose docstring names the table and the defaults."""
        return parse_anthropic(failure)

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Sort an exception parse gave no verdict, or name the terminal error for a DoNotRetry.

        `APIConnectionError` and `RetryableError` are transient transport failures.
        `APITimeoutError` is an `APIConnectionError` subclass.
        `APIStatusError` reaches this function only after `parse` returns `DoNotRetry`.
        Other exceptions return `unknown_exception`.
        """
        if isinstance(error, (anthropic.APIConnectionError, anthropic.RetryableError)):
            return "transient"
        if not isinstance(error, anthropic.APIStatusError):
            return "unknown_exception"
        return terminal_classification_from_response(
            status_code=error.response.status_code,
            headers=error.response.headers,
        )

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
        provider_tools: _AnthropicProviderTools = _NO_ANTHROPIC_PROVIDER_TOOLS,
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._provider_tools = provider_tools
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
    def billing_reported(self) -> Billing | None:
        """Return snapshot billing after the first event, or `None` before it.

        Anthropic 0.120.0 provides required input tokens from `message_start`.
        Optional cache counters arrive with `message_delta`.
        """
        if not self._snapshot_started:
            return None
        return _billing_from_sdk_usage(
            self._sdk_stream.current_message_snapshot.usage,
            self._pricing,
            provider_tools=self._provider_tools,
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
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundAnthropic[OutputT](BoundAdapter[OutputT], ABC):
    """What both anthropic bindings share: the request path, and what a response says about itself.

    A subclass sets _adapter and _precomputed_fields in its own __init__ and implements interpret.
    """

    _adapter: AnthropicMessagesAdapter
    _precomputed_fields: _AnthropicPrecomputedFields

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Price counters using reported response metadata.

        Raises:
            TypeError: raw is not an anthropic Message.
            pydantic.ValidationError: the message reports a negative counter.
        """
        return _billing_from_sdk_usage(
            _as_message(raw).usage,
            pricing=self._adapter.pricing,
            provider_tools=self._precomputed_fields.provider_tools,
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
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Convert messages under the binding's precomputed fields."""
        wire_messages = _request_messages(messages, self._precomputed_fields)
        if isinstance(wire_messages, InvalidRequest):
            return wire_messages
        return _AnthropicRequestParams(
            precomputed=self._precomputed_fields, messages=wire_messages
        )

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open one messages.stream and return the live stream; connection failures raise here.

        Raises:
            TypeError: request was built by another adapter.
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
        """
        params = narrowed_request(request, _AnthropicRequestParams)
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
            messages=params.messages,
            extra_body=_extra_body_with_temperature(precomputed),
        )
        return _AnthropicStream(
            sdk_stream=await manager.__aenter__(),
            pricing=self._adapter.pricing,
            provider_tools=precomputed.provider_tools,
        )


class _BoundAnthropicText(_BoundAnthropic[str]):
    """Text-bound adapter: output is the concatenated text of the turn."""

    def __init__(
        self, *, adapter: AnthropicMessagesAdapter, precomputed_fields: _AnthropicPrecomputedFields
    ) -> None:
        self._adapter = adapter
        self._precomputed_fields = precomputed_fields

    @override
    def interpret(self, raw: BaseModel) -> AdapterResult[str]:
        """Read the turn, whose concatenated text is this binding's output.

        Every message supplies its text despite an unhandled stop reason.

        Raises:
            TypeError: `raw` is not an anthropic `Message`.
        """
        message = _as_message(raw)
        assistant_message = _assistant_message_from(message)
        return _adapter_result(message, assistant_message.text, assistant_message)


class _BoundAnthropicStructured[ModelT: BaseModel](_BoundAnthropic[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the turn's text."""

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
        self._adapter = adapter
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
        self._precomputed_fields = replace(precomputed_fields, output_config=output_config)

    def _parsed_outcome(
        self, message: anthropic.types.Message, assistant_message: AssistantMessage
    ) -> ResponseOutcome[ModelT | None]:
        """Validate text, return `None` for a tool-call turn, or return a failure variant.

        Validation occurs after the attempt records its message and billing.
        Stop reasons take precedence over schema validation.
        Every failure variant carries `assistant_message`.
        """
        validation_error: ValidationError | None = None
        text = _first_text_block_text(message)
        if text is not None:
            try:
                output = self._output_type_adapter.validate_json(text)
                return _adapter_result(message, output, assistant_message)
            except ValidationError as rejection:
                validation_error = rejection
        unfinished_turn = _unfinished_turn_or_none(message, assistant_message=assistant_message)
        if unfinished_turn is not None:
            return unfinished_turn
        if message.stop_reason == "tool_use":
            return _adapter_result(message, None, assistant_message)
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
        return EmptyTurn(assistant_message=assistant_message)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, or report why the message produced none.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        message = _as_message(raw)
        assistant_message = _assistant_message_from(message)
        return self._parsed_outcome(message, assistant_message)
