"""Implement Gemini `generateContent` through the google-genai SDK.

The following SDK facts were verified against google-genai 2.16.0.
`retry_args(None)` permits one attempt.
Every request sets `HttpRetryOptions(attempts=1)` through per-request `http_options`.
The per-request value replaces the client value and preserves `max_attempts` for passed clients.
`generate_content_stream` yields `GenerateContentResponse` chunks without assembling them.
`assembled_response` assembles the chunks.
A mid-stream error raises `errors.APIError` with the response body code.
`errors.APIError` has `code`, `status`, `message`, `details`, and an optional `httpx.Response`.
It has no request-id attribute, and its response has no request-id field.

`usage_metadata.prompt_token_count` includes `cached_content_token_count`.
`total_token_count` sums prompt, candidate, tool-use prompt, and thought tokens.
Gemini reports no cache-write counter and bills no cache writes.
Implicit cache reads appear in `cached_content_token_count`.
Source: https://ai.google.dev/gemini-api/docs/caching, read 2026-08-03.

`Part.thought` marks reasoning, and `Part.thought_signature` contains bytes.
JSON-mode SDK dumps encode bytes as base64 and validate them back.
`ReasoningPart.raw` stores that JSON-mode dump.
SDK models reject unknown keys, so another provider's dump returns `InvalidRequest`.
SDK enums preserve unknown effort and tier values as synthetic enum values with `UserWarning`.
`FunctionResponse.response` uses `"output"` and `"error"` for `ToolMessage.content` and `ToolMessage.is_error`.

Thought signatures require exact replay on thought and built-in tool steps.
Source: https://ai.google.dev/gemini-api/docs/thinking, read 2026-08-03.
The adapter emits `ReasoningPart` for each part with `thought` or `thought_signature`.
`Part.model_validate_json` restores each part unchanged.
A signed part may also emit `TextPart` or `ToolCall` for its answer text or function call.
Replay skips a matching following `TurnPart` because `ReasoningPart.raw` already contains its data.

`generateContent` has no request field for implicit caching.
Both `automatic_cache_breakpoints` values produce the same request and cache-read billing.
Gemini never bills a cache write.
A marked system part raises `ValueError`, and a marked message part returns `InvalidRequest`.
`extra_body={"cachedContent": ...}` selects an explicit cache resource.

Content mappings were verified against google-genai 2.17.0.
- `ImagePart` and `AudioPart` become `Part.inline_data`.
- `ImageUrlPart` becomes `Part.file_data`.
- Inside `ToolMessage`, `ImagePart` and `AudioPart` become `FunctionResponsePart.inline_data`.
- Inside `ToolMessage`, `ImageUrlPart` becomes `FunctionResponsePart.file_data`.
- `Part.tool_call` and `Part.tool_response` carry provider-executed tool evidence.

Request and response mappings:
- `ToolMessage` becomes `function_response` inside user-role `Content`.
- Consecutive `ToolMessage` values share one `Content`.
- `FunctionResponse.name` comes from the earlier `ToolCall` matching `tool_call_id`.
- A missing match returns `InvalidRequest`.
- `FunctionCall.id` is optional, while `ToolCall.id` is required.
- The adapter uses the function name when the provider omits the id.
- Replay sends the id only when it differs from the function name.
- Same-name calls without provider ids share an id and match results by order.
- `reasoning_level` sends the exact `thinking_level` string with `include_thoughts=True`.
- The provider reports unsupported values through its own error.
- `parallel_tool_calls=False` raises at bind because Gemini has no disabling field.
- `MAX_TOKENS` maps to `"max_tokens"`, and refusal reasons map to `"refusal"`.
- `STOP` maps by `ToolCall` presence to `"tool_use"` or `"end_turn"`.
- Other finish reasons map to `"other"`.
- Gemini reports terminal conditions through statuses or stream errors instead of a completed 200 failure body.
- The adapter does not construct `ContextWindowExceeded`, `ProviderFailedTransiently`, or `ProviderFailedTerminally`.
"""

import json
from abc import ABC
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import nan
from typing import ClassVar, Literal, override

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint.adapter import (
    REASONING_PART_SEPARATOR,
    Adapter,
    AdapterResult,
    AdapterStream,
    AllowedToolsChoice,
    Binding,
    BoundAdapter,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutput,
    ReasoningDelta,
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    SpecificToolChoice,
    StreamItem,
    ToolChoice,
    UnfinishedTurn,
    _NotSendableError,
    narrowed_request,
    record_parse_fallthrough,
    reject_extra_body_keys_the_adapter_populates,
    retry_after_seconds_from_headers,
    verdict_from_transient_error,
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
    ProviderBilling,
    category_cost,
    invocation_cost_in_usd,
    require_finite_nonnegative_rate,
    require_pricing_key,
)
from langchaint.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict
from langchaint.usage import Usage

_PAUSE_STATUSES = frozenset({429, 503})
"""429 RESOURCE_EXHAUSTED and 503 UNAVAILABLE pause every request sharing the rate-limit quota."""

_RETRY_THIS_ONE_STATUSES = frozenset({408, 500, 502, 504})
"""One request's failure, retried without pausing siblings.

500 INTERNAL and 504 DEADLINE_EXCEEDED come from the troubleshooting page.
408 and 502 are unlisted there.
The google-genai 2.19.0 retryable set is `(408, 429, 500, 502, 503, 504)`.
"""

_DO_NOT_RETRY_STATUSES = frozenset({400, 403, 404})
"""The request-rejection statuses from the troubleshooting page. A resend fails again."""

PARSE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""`record_parse_fallthrough` increments this counter for each status-family default."""

type GeminiServiceTier = Literal["flex", "standard", "priority"]
"""What a request may ask for: the SDK ServiceTier wire values (google-genai 2.16.0)."""

type GeminiPricedServiceTier = Literal[
    "ON_DEMAND", "ON_DEMAND_PRIORITY", "ON_DEMAND_FLEX", "PROVISIONED_THROUGHPUT"
]
"""What a response's usage_metadata.traffic_type reports having been served at (google-genai 2.16.0).

The pricing mapping uses these values as keys.
Its key type is `str` because the SDK's open enum constructs unknown values.
The request and response tier vocabularies share no value.
"""

_ON_DEMAND_TIER: GeminiPricedServiceTier = "ON_DEMAND"

_VERTEX_PROVIDER_NAME = "gcp.vertex_ai"

_REFUSAL_FINISH_REASONS = frozenset({
    types.FinishReason.SAFETY,
    types.FinishReason.RECITATION,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.IMAGE_SAFETY,
    types.FinishReason.IMAGE_PROHIBITED_CONTENT,
    types.FinishReason.IMAGE_RECITATION,
})
"""The finish reasons where the model or a filter declined to answer: the turn is a refusal."""

_FINISHED_FINISH_REASONS = (
    frozenset({types.FinishReason.STOP, types.FinishReason.MAX_TOKENS}) | _REFUSAL_FINISH_REASONS
)
"""The finish reasons langchaint can read a completed turn from.

All other values produce `UnfinishedTurn` for a structured binding.
"""

_NO_CACHE_BREAKPOINT_WIRE_FORM = (
    "cache_breakpoint has no Gemini wire form: generateContent has no request field marking a "
    "prompt-cache boundary, and dropping the mark would silently misstate the request"
)

_SUPPORTED_PROVIDER_TOOL_FIELDS = frozenset({
    "code_execution",
    "file_search",
    "google_maps",
    "google_search",
    "url_context",
})
_CHARGED_PROVIDER_TOOL_FIELDS = frozenset({"google_maps", "google_search"})


def _populated_provider_tool_fields(tool: types.Tool) -> frozenset[str]:
    """Return every populated field after `types.Tool` normalization.

    Raises:
        ValueError: no field is populated or an unsupported field is populated.
            Also raised when Google Search enables image search.
    """
    populated_fields = frozenset(
        field_name for field_name, field_value in tool if field_value is not None
    )
    if not populated_fields or not populated_fields <= _SUPPORTED_PROVIDER_TOOL_FIELDS:
        raise ValueError("Gemini provider_executed_tools contain an unsupported field")
    google_search = tool.google_search
    if (
        google_search is not None
        and google_search.search_types is not None
        and google_search.search_types.image_search is not None
    ):
        raise ValueError("Gemini provider_executed_tools do not support image search")
    return populated_fields


def _normalize_provider_tools(
    provider_executed_tools: tuple[Mapping[str, object], ...],
) -> tuple[tuple[types.Tool, ...], frozenset[str]]:
    """Normalize every mapping and return every populated provider tool field.

    Raises:
        pydantic.ValidationError: a mapping fails `types.Tool` validation.
        ValueError: `_populated_provider_tool_fields` rejects a normalized tool.
    """
    normalized_tools: list[types.Tool] = []
    populated_fields: set[str] = set()
    for provider_tool in provider_executed_tools:
        normalized_tool = types.Tool.model_validate(provider_tool)
        normalized_tools.append(normalized_tool)
        populated_fields.update(_populated_provider_tool_fields(normalized_tool))
    return tuple(normalized_tools), frozenset(populated_fields)


@dataclass(frozen=True, kw_only=True)
class GeminiRates:
    """One rate per category Gemini bills on a request.

    No cache-write rate: no cache write is ever billed, so none exists to state.
    """

    input_cache_none_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


_NO_CACHE_WRITE_RATE = float("nan")
"""The cache-write price every Gemini Billing reports.

Gemini bills no cache writes, so no rate exists to state.
The write counter is always zero, and `category_cost(0, usd_per_million_tokens=float("nan"))` is `0.0`.
"""


@dataclass(frozen=True, kw_only=True)
class GeminiPricingTable:
    """Store one model's rates for one served tier.

    Above long_prompt_threshold_tokens, long_prompt_rates prices input, cache reads, and output.
    Models without this pricing leave both long-prompt fields None.
    """

    rates: GeminiRates
    google_search_usd_per_query: float | None = None
    google_maps_usd_per_query: float | None = None
    long_prompt_threshold_tokens: int | None = None
    long_prompt_rates: GeminiRates | None = None

    def __post_init__(self) -> None:
        """Require the two long-prompt fields together.

        Raises:
            ValueError: Exactly one of long_prompt_threshold_tokens and long_prompt_rates is set.
        """
        if (self.long_prompt_threshold_tokens is None) != (self.long_prompt_rates is None):
            raise ValueError(
                "long_prompt_threshold_tokens and long_prompt_rates must be set together"
            )

    def price(  # noqa: PLR0913 (each normalized category arrives separately)
        self,
        *,
        service_tier: str,
        usage_raw: BaseModel | None,
        prompt_token_count: int,
        input_tokens_cache_read: int,
        input_tokens_cache_none: int,
        output_tokens: int,
        output_tokens_reasoning: int,
        provider_executed_tool_cost_in_usd: float,
    ) -> ProviderBilling:
        """Price one response's counters, at the long-prompt rates when the prompt crosses the threshold.

        prompt_token_count excludes the tool-execution input the categories include.
        The zero cache-write counter costs zero despite its NaN price.

        Raises:
            pydantic.ValidationError: a counter is negative.
        """
        rates = self.rates
        if (
            self.long_prompt_threshold_tokens is not None
            and self.long_prompt_rates is not None
            and prompt_token_count > self.long_prompt_threshold_tokens
        ):
            rates = self.long_prompt_rates
        return ProviderBilling(
            billing=Billing(
                usage=Usage(
                    input_tokens_cache_read=input_tokens_cache_read,
                    input_tokens_cache_write=0,
                    input_tokens_cache_none=input_tokens_cache_none,
                    output_tokens=output_tokens,
                    output_tokens_reasoning=output_tokens_reasoning,
                    input_tokens_cache_read_cost_in_usd=category_cost(
                        input_tokens_cache_read,
                        usd_per_million_tokens=rates.cache_read_usd_per_million_tokens,
                    ),
                    input_tokens_cache_write_cost_in_usd=0.0,
                    input_tokens_cache_none_cost_in_usd=category_cost(
                        input_tokens_cache_none,
                        usd_per_million_tokens=rates.input_cache_none_usd_per_million_tokens,
                    ),
                    output_tokens_cost_in_usd=category_cost(
                        output_tokens,
                        usd_per_million_tokens=rates.output_usd_per_million_tokens,
                    ),
                    provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
                ),
                service_tier=service_tier,
                input_cache_none_usd_per_million_tokens=rates.input_cache_none_usd_per_million_tokens,
                cache_read_usd_per_million_tokens=rates.cache_read_usd_per_million_tokens,
                cache_write_usd_per_million_tokens=_NO_CACHE_WRITE_RATE,
                output_usd_per_million_tokens=rates.output_usd_per_million_tokens,
            ),
            usage_raw=usage_raw,
        )


_UNPRICED = GeminiPricingTable(
    rates=GeminiRates(
        input_cache_none_usd_per_million_tokens=float("nan"),
        cache_read_usd_per_million_tokens=float("nan"),
        output_usd_per_million_tokens=float("nan"),
    ),
    google_search_usd_per_query=float("nan"),
    google_maps_usd_per_query=float("nan"),
)
"""What prices a response reporting a traffic_type the adapter holds no table for.

Every nonzero counter costs NaN.
Every zero counter costs zero.
"""


def _require_provider_tool_support(
    *,
    model: str,
    pricing: Mapping[str, GeminiPricingTable],
    provider_tool_fields: frozenset[str],
) -> None:
    """Require Gemini 3 and finite configured rates across supplied pricing tables.

    Raises:
        ValueError: the model is not Gemini 3.
            Also raised for unavailable, negative, infinite, or NaN configured rates.
    """
    model_id = model.rsplit("/", 1)[-1]
    if provider_tool_fields and not model_id.startswith("gemini-3"):
        raise ValueError("Gemini provider_executed_tools require a Gemini 3 model")
    for pricing_table in pricing.values():
        if "google_search" in provider_tool_fields:
            google_search_rate = pricing_table.google_search_usd_per_query
            require_finite_nonnegative_rate(
                rate_name="google_search_usd_per_query", rate=google_search_rate
            )
        if "google_maps" in provider_tool_fields:
            google_maps_rate = pricing_table.google_maps_usd_per_query
            require_finite_nonnegative_rate(
                rate_name="google_maps_usd_per_query", rate=google_maps_rate
            )


def _priced_tier(traffic_type: types.TrafficType | None) -> str:
    """Return the tier that selects the table: what the response reports, "ON_DEMAND" where it reports none.

    TRAFFIC_TYPE_UNSPECIFIED counts as reporting none: it is the proto zero value, not a tier.
    """
    if traffic_type is None or traffic_type == types.TrafficType.TRAFFIC_TYPE_UNSPECIFIED:
        return _ON_DEMAND_TIER
    return traffic_type.value


def _billing_from_usage(
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
    pricing: Mapping[str, GeminiPricingTable],
    *,
    provider_executed_tool_cost_in_usd: float,
) -> ProviderBilling:
    """Price the reported counters at the table the served tier selects.

    prompt_token_count includes cached_content_token_count (google-genai 2.16.0 field description).
    This includes implicit cache reads (https://ai.google.dev/gemini-api/docs/caching, read 2026-08-03).
    Their difference is input_tokens_cache_none.
    thoughts_token_count is outside candidates_token_count.
    output_tokens is their sum, and output_tokens_reasoning is thoughts_token_count.
    Source: google-genai 2.16.0 total_token_count field description.
    tool_use_prompt_token_count is outside prompt_token_count and uses the uncached input rate.
    None usage_metadata bills zero counters at the "ON_DEMAND" table's prices.

    Raises:
        pydantic.ValidationError: the counters leave a category negative.
    """
    if usage_metadata is None:
        return pricing.get(_ON_DEMAND_TIER, _UNPRICED).price(
            service_tier=_ON_DEMAND_TIER,
            usage_raw=None,
            prompt_token_count=0,
            input_tokens_cache_read=0,
            input_tokens_cache_none=0,
            output_tokens=0,
            output_tokens_reasoning=0,
            provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
        )
    service_tier = _priced_tier(usage_metadata.traffic_type)
    prompt_token_count = usage_metadata.prompt_token_count or 0
    input_tokens_cache_read = usage_metadata.cached_content_token_count or 0
    output_tokens_reasoning = usage_metadata.thoughts_token_count or 0
    return pricing.get(service_tier, _UNPRICED).price(
        service_tier=service_tier,
        usage_raw=usage_metadata,
        prompt_token_count=prompt_token_count,
        input_tokens_cache_read=input_tokens_cache_read,
        input_tokens_cache_none=prompt_token_count
        - input_tokens_cache_read
        + (usage_metadata.tool_use_prompt_token_count or 0),
        output_tokens=(usage_metadata.candidates_token_count or 0) + output_tokens_reasoning,
        output_tokens_reasoning=output_tokens_reasoning,
        provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
    )


_PROVIDER_FIELD_BY_TOOL_TYPE = {
    types.ToolType.GOOGLE_SEARCH_WEB: "google_search",
    types.ToolType.GOOGLE_MAPS: "google_maps",
    types.ToolType.URL_CONTEXT: "url_context",
    types.ToolType.FILE_SEARCH: "file_search",
}


def _all_candidate_parts(response: types.GenerateContentResponse) -> list[types.Part]:
    """Return every part from every response candidate."""
    parts: list[types.Part] = []
    for candidate in response.candidates or []:
        if candidate.content is not None and candidate.content.parts is not None:
            parts.extend(candidate.content.parts)
    return parts


def _queries_from_tool_call(tool_call: types.ToolCall, *, maps: bool) -> list[str] | None:
    """Read validated query evidence, or return None for malformed evidence."""
    if tool_call.args is None:
        return None
    queries = tool_call.args.get("queries")
    if not isinstance(queries, list):
        return None
    if any(not isinstance(query, str) for query in queries):
        return None
    if maps and any(not query for query in queries):
        return None
    return queries


def _provider_executed_tool_cost_in_usd(
    provider_tool_parts: Sequence[types.Part],
    *,
    table: GeminiPricingTable,
    configured_fields: frozenset[str],
    billing_complete: bool,
) -> float:
    """Price paired Search and Maps calls from one assembled response."""
    if not billing_complete and configured_fields & _CHARGED_PROVIDER_TOOL_FIELDS:
        return nan
    tool_calls: list[types.ToolCall] = []
    tool_responses: list[types.ToolResponse] = []
    for part in provider_tool_parts:
        if part.tool_call is not None:
            tool_calls.append(part.tool_call)
        if part.tool_response is not None:
            tool_responses.append(part.tool_response)

    call_keys: Counter[tuple[str, types.ToolType]] = Counter()
    response_keys: Counter[tuple[str, types.ToolType]] = Counter()
    typed_tool_calls: list[tuple[types.ToolCall, types.ToolType]] = []
    for tool_call in tool_calls:
        tool_type = tool_call.tool_type
        if tool_call.id is None or tool_type is None:
            return nan
        call_keys[(tool_call.id, tool_type)] += 1
        typed_tool_calls.append((tool_call, tool_type))
    for tool_response in tool_responses:
        if tool_response.id is None or tool_response.tool_type is None:
            return nan
        response_keys[(tool_response.id, tool_response.tool_type)] += 1
    if call_keys != response_keys:
        return nan

    search_queries: set[str] = set()
    maps_query_count = 0
    for tool_call, tool_type in typed_tool_calls:
        provider_field = _PROVIDER_FIELD_BY_TOOL_TYPE.get(tool_type)
        if provider_field is None or provider_field not in configured_fields:
            return nan
        if provider_field == "google_search":
            queries = _queries_from_tool_call(tool_call, maps=False)
            if queries is None:
                return nan
            search_queries.update(query for query in queries if query)
        elif provider_field == "google_maps":
            queries = _queries_from_tool_call(tool_call, maps=True)
            if queries is None:
                return nan
            maps_query_count += len(queries)

    google_search_rate = table.google_search_usd_per_query
    google_maps_rate = table.google_maps_usd_per_query
    return invocation_cost_in_usd(
        len(search_queries),
        usd_per_invocation=google_search_rate,
    ) + invocation_cost_in_usd(
        maps_query_count,
        usd_per_invocation=google_maps_rate,
    )


def _billing_from_provider_evidence(
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
    provider_tool_parts: Sequence[types.Part],
    pricing: Mapping[str, GeminiPricingTable],
    *,
    configured_fields: frozenset[str] = frozenset(),
    billing_complete: bool = True,
) -> ProviderBilling:
    """Price token counters and assembled provider-executed tool evidence."""
    service_tier = _ON_DEMAND_TIER
    if usage_metadata is not None:
        service_tier = _priced_tier(usage_metadata.traffic_type)
    table = pricing.get(service_tier, _UNPRICED)
    return _billing_from_usage(
        usage_metadata,
        pricing,
        provider_executed_tool_cost_in_usd=_provider_executed_tool_cost_in_usd(
            provider_tool_parts,
            table=table,
            configured_fields=configured_fields,
            billing_complete=billing_complete,
        ),
    )


def _billing_from_response(
    response: types.GenerateContentResponse,
    pricing: Mapping[str, GeminiPricingTable],
    *,
    configured_fields: frozenset[str] = frozenset(),
    billing_complete: bool = True,
) -> ProviderBilling:
    """Price token counters and provider evidence from every candidate."""
    return _billing_from_provider_evidence(
        response.usage_metadata,
        _all_candidate_parts(response),
        pricing,
        configured_fields=configured_fields,
        billing_complete=billing_complete,
    )


def _tool_call_from(function_call: types.FunctionCall) -> ToolCall:
    """Build the neutral ToolCall, synthesizing the id from the name when the provider sent none.

    An absent FunctionCall.name becomes an empty string and fails during dispatch or replay.
    """
    name = function_call.name or ""
    return ToolCall(
        id=function_call.id or name,
        name=name,
        args_json=json.dumps(function_call.args or {}),
    )


def _assistant_message_from(content: types.Content | None) -> AssistantMessage:
    """Build `AssistantMessage` from candidate parts in order.

    Thought or signed parts become replayable `ReasoningPart` values.
    Signed answer text and function calls also produce `TextPart` or `ToolCall`.
    Unmodeled parts become replayable `RawPart` values.
    Empty text becomes `RawPart` only when the part has no other modeled value.
    Missing content or parts produces an empty turn.
    """
    parts = content.parts if content is not None and content.parts is not None else []
    turn: list[TurnPart] = []
    for part in parts:
        carries_reasoning = part.thought or part.thought_signature is not None
        if carries_reasoning:
            turn.append(
                ReasoningPart(
                    raw=part.model_dump(mode="json", exclude_none=True),
                    text=(part.text or None) if part.thought else None,
                )
            )
        if part.function_call is not None:
            turn.append(_tool_call_from(part.function_call))
        elif part.text:
            if not part.thought:
                turn.append(TextPart(text=part.text))
        elif not carries_reasoning:
            turn.append(RawPart(raw=part.model_dump(mode="json", exclude_none=True)))
    return AssistantMessage(turn=tuple(turn))


def _function_call_from(tool_call: ToolCall) -> types.FunctionCall:
    """Convert one ToolCall back to its wire form, omitting an id that was synthesized from the name.

    Raises:
        _NotSendableError: args_json does not contain a JSON object.
        json.JSONDecodeError: args_json is not valid JSON.
    """
    args = json.loads(tool_call.args_json)
    if not isinstance(args, dict):
        raise _NotSendableError(
            f"a tool call's args_json must hold a JSON object to go on the Gemini wire, "
            f"not {type(args).__name__}"
        )
    return types.FunctionCall(
        id=None if tool_call.id == tool_call.name else tool_call.id,
        name=tool_call.name,
        args=args,
    )


def _part_from_dump(raw: Mapping[str, object], *, part_description: str) -> types.Part:
    """Restore one stored raw dump to the Part that produced it, byte-identical.

    `model_validate_json` reverses the JSON-mode dump stored by `_assistant_message_from`.
    `part_description` names the `ReasoningPart` or `RawPart`.

    Raises:
        _NotSendableError: raw does not restore to a Part.
    """
    try:
        return types.Part.model_validate_json(json.dumps(raw))
    except (TypeError, ValueError) as not_a_part:
        raise _NotSendableError(
            f"{part_description} does not restore to a Gemini Part, so it cannot be sent: "
            f"{not_a_part}"
        ) from not_a_part


def _matches_replayed_part_payload(part: TurnPart, replayed_part: types.Part) -> bool:
    """Whether TurnPart repeats payload already carried by replayed_part.

    _assistant_message_from pairs a signature-carrying part with the ToolCall or TextPart that holds its payload.
    A part without a match goes on the wire itself.
    """
    function_call = replayed_part.function_call
    if isinstance(part, ToolCall) and function_call is not None:
        name = function_call.name or ""
        try:
            args: object = json.loads(part.args_json)
        except json.JSONDecodeError:
            return False
        return (
            part.name == name
            and part.id == (function_call.id or name)
            and args == (function_call.args or {})
        )
    if isinstance(part, TextPart) and function_call is None:
        return (
            bool(replayed_part.text)
            and not replayed_part.thought
            and part.text == replayed_part.text
        )
    return False


def _assistant_wire_parts(assistant_message: AssistantMessage) -> list[types.Part]:
    """Convert one AssistantMessage to wire parts in turn order.

    A ReasoningPart restores its source Part.
    A following TurnPart with the same payload is skipped.
    An empty TextPart yields nothing.
    A RawPart restores the same way and carries no such pair.

    Raises:
        _NotSendableError: Stored raw data does not restore to a Gemini Part, or args_json is not an object.
        json.JSONDecodeError: a tool call's args_json is not valid JSON.
    """
    parts: list[types.Part] = []
    replayed_part_with_paired_payload: types.Part | None = None
    for part in assistant_message.turn:
        if replayed_part_with_paired_payload is not None:
            replayed_part = replayed_part_with_paired_payload
            replayed_part_with_paired_payload = None
            if _matches_replayed_part_payload(part, replayed_part):
                continue
        if isinstance(part, TextPart):
            if part.text:
                parts.append(types.Part(text=part.text))
        elif isinstance(part, ToolCall):
            parts.append(types.Part(function_call=_function_call_from(part)))
        elif isinstance(part, ReasoningPart):
            replayed_part = _part_from_dump(part.raw, part_description="a ReasoningPart")
            parts.append(replayed_part)
            if replayed_part.function_call is not None or (
                replayed_part.text and not replayed_part.thought
            ):
                replayed_part_with_paired_payload = replayed_part
        else:
            parts.append(_part_from_dump(part.raw, part_description="a RawPart"))
    return parts


def _cache_breakpoint_reason(
    part: ContentPart, *, message_class: type[UserMessage] | type[ToolMessage]
) -> str:
    """Name the marked ContentPart and message_class Gemini cannot send."""
    return (
        f"GeminiGenerateContentAdapter cannot send {type(part).__name__} inside "
        f"{message_class.__name__}.content: cache_breakpoint has no Gemini wire form"
    )


def _inline_part(data: bytes, media_type: str) -> types.Part:
    return types.Part(inline_data=types.Blob(data=data, mime_type=media_type))


def _inline_function_response_part(data: bytes, media_type: str) -> types.FunctionResponsePart:
    return types.FunctionResponsePart(
        inline_data=types.FunctionResponseBlob(data=data, mime_type=media_type)
    )


def _user_parts(content: str | tuple[ContentPart, ...]) -> list[types.Part]:
    """Convert one UserMessage's content to wire parts.

    ImagePart and AudioPart use Part.inline_data.
    ImageUrlPart uses Part.file_data.

    Raises:
        _NotSendableError: A ContentPart sets cache_breakpoint.
    """
    if isinstance(content, str):
        return [types.Part(text=content)]
    parts: list[types.Part] = []
    for part in content:
        if part.cache_breakpoint:
            raise _NotSendableError(_cache_breakpoint_reason(part, message_class=UserMessage))
        match part.kind:
            case "text":
                parts.append(types.Part(text=part.text))
            case "image":
                parts.append(_inline_part(part.data, part.media_type))
            case "image_url":
                parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=part.url,
                            mime_type=part.media_type,
                        )
                    )
                )
            case "audio":
                parts.append(_inline_part(part.data, part.media_type))
    return parts


def _function_response_part(
    tool_message: ToolMessage, tool_call_names: Mapping[str, str]
) -> types.Part:
    """Convert one ToolMessage to its function_response part.

    FunctionResponse.response stores concatenated TextPart.text under "output" or "error".
    ImagePart and AudioPart use FunctionResponsePart.inline_data.
    ImageUrlPart uses FunctionResponsePart.file_data.
    FunctionResponse.id is omitted when it equals the recovered ToolCall.name.

    Raises:
        _NotSendableError: ToolMessage.tool_call_id matches no earlier ToolCall.id.
            FunctionResponse.name cannot be recovered in that case.
            cache_breakpoint on any ContentPart also raises.
    """
    name = tool_call_names.get(tool_message.tool_call_id)
    if name is None:
        raise _NotSendableError(
            f"tool_call_id {tool_message.tool_call_id!r} matches no ToolCall in an earlier "
            f"assistant turn, so the FunctionResponse name the wire requires cannot be recovered"
        )
    function_response_parts: list[types.FunctionResponsePart] = []
    if isinstance(tool_message.content, str):
        text = tool_message.content
    else:
        texts: list[str] = []
        for part in tool_message.content:
            if part.cache_breakpoint:
                raise _NotSendableError(_cache_breakpoint_reason(part, message_class=ToolMessage))
            match part.kind:
                case "text":
                    texts.append(part.text)
                case "image":
                    function_response_parts.append(
                        _inline_function_response_part(part.data, part.media_type)
                    )
                case "image_url":
                    function_response_parts.append(
                        types.FunctionResponsePart(
                            file_data=types.FunctionResponseFileData(
                                file_uri=part.url,
                                mime_type=part.media_type,
                            )
                        )
                    )
                case "audio":
                    function_response_parts.append(
                        _inline_function_response_part(part.data, part.media_type)
                    )
        text = "".join(texts)
    return types.Part(
        function_response=types.FunctionResponse(
            id=None if tool_message.tool_call_id == name else tool_message.tool_call_id,
            name=name,
            response={"error" if tool_message.is_error else "output": text},
            parts=function_response_parts or None,
        )
    )


def _wire_contents(messages: Sequence[Message]) -> list[types.Content]:
    """Convert messages to wire contents.

    Consecutive ToolMessage values form one user-role Content.
    Assistant turns supply ToolCall names for later FunctionResponse values.

    Raises:
        _NotSendableError: A message is unsendable.
        json.JSONDecodeError: a tool call's args_json is not valid JSON.
    """
    contents: list[types.Content] = []
    pending_function_responses: list[types.Part] = []
    tool_call_names: dict[str, str] = {}

    def flush_function_responses() -> None:
        if pending_function_responses:
            contents.append(types.Content(role="user", parts=list(pending_function_responses)))
            pending_function_responses.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            pending_function_responses.append(_function_response_part(message, tool_call_names))
        elif isinstance(message, UserMessage):
            flush_function_responses()
            contents.append(types.Content(role="user", parts=_user_parts(message.content)))
        else:
            flush_function_responses()
            for tool_call in message.tool_calls:
                tool_call_names[tool_call.id] = tool_call.name
            contents.append(types.Content(role="model", parts=_assistant_wire_parts(message)))
    flush_function_responses()
    return contents


def _request_contents(messages: Sequence[Message]) -> list[types.Content] | InvalidRequest:
    """Convert messages, or report them unsendable.

    An unsendable Sequence[Message] becomes InvalidRequest. This includes unparseable tool_call.args_json.
    """
    try:
        return _wire_contents(messages)
    except _NotSendableError as not_sendable:
        return InvalidRequest(reason=str(not_sendable))
    except json.JSONDecodeError as not_json:
        return InvalidRequest(reason=f"a tool call's args_json is not valid JSON: {not_json}")


@dataclass(frozen=True, kw_only=True)
class _GeminiRequestParams(RequestParams):
    """One generateContent request: the binding's config and this call's converted contents."""

    model: str
    config: types.GenerateContentConfig
    contents: list[types.Content]

    @override
    def as_json(self) -> str:
        """Render the request as a JSON object, dropping every field left to the provider's default.

        exclude_none omits absent SDK fields.
        http_options is transport configuration, but extra_body enters the request body.
        An unsupported JSON value becomes str(value) so table construction cannot fail on one cell.
        """
        body: dict[str, object] = {
            "model": self.model,
            "contents": [
                content.model_dump(mode="json", exclude_none=True) for content in self.contents
            ],
            "config": self.config.model_dump(
                mode="json", exclude_none=True, exclude={"http_options"}
            ),
        }
        http_options = self.config.http_options
        if isinstance(http_options, types.HttpOptions) and http_options.extra_body:
            body["extra_body"] = http_options.extra_body
        return json.dumps(body, default=str)


def _wire_tool_config(tool_choice: ToolChoice) -> types.ToolConfig:
    """Convert the neutral tool choice.

    Neutral "required" is Gemini mode ANY.
    """
    if isinstance(tool_choice, SpecificToolChoice):
        function_calling_config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=[tool_choice.tool_name],
        )
    elif isinstance(tool_choice, AllowedToolsChoice):
        function_calling_config = types.FunctionCallingConfig(
            mode=(
                types.FunctionCallingConfigMode.VALIDATED
                if tool_choice.mode == "auto"
                else types.FunctionCallingConfigMode.ANY
            ),
            allowed_function_names=list(tool_choice.tool_names),
        )
    elif tool_choice == "auto":
        function_calling_config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.AUTO
        )
    elif tool_choice == "required":
        function_calling_config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY
        )
    else:
        function_calling_config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.NONE
        )
    return types.ToolConfig(function_calling_config=function_calling_config)


_ADAPTER_POPULATED_TOP_LEVEL_WIRE_KEYS = frozenset({
    "contents",
    "systeminstruction",
    "tools",
    "toolconfig",
    "servicetier",
})
"""The top-level wire-body keys the adapter populates, stored as _normalized_wire_key spells them."""

_ADAPTER_POPULATED_GENERATION_CONFIG_KEYS = frozenset({
    "temperature",
    "maxoutputtokens",
    "thinkingconfig",
    "responsemimetype",
    "responsejsonschema",
})
"""The generationConfig terminal keys the adapter populates, normalized by _normalized_wire_key.

Unlisted terminal keys such as topK, seed, and stopSequences pass through.
"""


def _normalized_wire_key(key: str) -> str:
    """Normalize a wire-body key the way the SDK matches extra_body keys to wire keys.

    google-genai 2.16.0 _normalize_key_for_matching ignores case and underscores.
    """
    return key.replace("_", "").lower()


def _reject_extra_body_keys(extra_body: Mapping[str, object] | None) -> None:
    """Refuse the extra_body keys whose SDK merge would silently override the binding.

    The SDK recursively merges extra_body into the wire body.
    extra_body wins matched terminal keys.
    _normalized_wire_key compares both mappings.
    Populated top-level keys and generationConfig terminal keys are refused.

    Raises:
        ValueError: extra_body has a refused key, or generationConfig has an invalid shape or key.
    """
    if extra_body is None:
        return
    generation_configs: list[Mapping[str, object]] = []
    for key, nested in extra_body.items():
        if _normalized_wire_key(key) != "generationconfig":
            continue
        if not isinstance(nested, Mapping):
            # `Adapter.bind_text` requires `ValueError` for binding defects.
            raise ValueError(  # noqa: TRY004
                f"extra_body key {key!r} must map to an object: the SDK merge would substitute "
                f"a non-object value for the generationConfig fields the adapter populates"
            )
        for nested_key in nested:
            if not isinstance(nested_key, str):
                raise ValueError(  # noqa: TRY004
                    f"extra_body key {key!r} must map to an object with only string keys"
                )
        generation_configs.append(nested)
    reject_extra_body_keys_the_adapter_populates(
        extra_body,
        populated_keys=_ADAPTER_POPULATED_TOP_LEVEL_WIRE_KEYS,
        normalized_key=_normalized_wire_key,
    )
    for generation_config in generation_configs:
        reject_extra_body_keys_the_adapter_populates(
            generation_config,
            populated_keys=_ADAPTER_POPULATED_GENERATION_CONFIG_KEYS,
            normalized_key=_normalized_wire_key,
        )


def _retry_after_seconds_from_retry_info(details: object) -> float | None:
    """Read the RetryInfo-stated wait from an APIError's details, None on any shape mismatch.

    google-genai 2.16.0 errors.py supplies the error envelope or its inner error dict.
    RetryInfo uses the google/rpc/error_details.proto shape.
    retryDelay is a protobuf Duration encoded as decimal seconds plus "s".
    """
    if not isinstance(details, Mapping):
        return None
    error_body: object = details.get("error", details)
    if not isinstance(error_body, Mapping):
        return None
    entries: object = error_body.get("details")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        type_url: object = entry.get("@type", "")
        if not (isinstance(type_url, str) and type_url.endswith("RetryInfo")):
            continue
        delay: object = entry.get("retryDelay")
        if not isinstance(delay, str) or not delay.endswith("s"):
            return None
        try:
            retry_after_seconds = float(delay[:-1])
        except ValueError:
            return None
        return retry_after_seconds if retry_after_seconds > 0 else None
    return None


def _retry_after_seconds_from_error(failure: errors.APIError) -> float | None:
    """Read the server-stated wait: response headers first, then the body's RetryInfo detail."""
    if failure.response is not None:
        header_stated = retry_after_seconds_from_headers(failure.response.headers)
        if header_stated is not None:
            return header_stated
    return _retry_after_seconds_from_retry_info(failure.details)


def parse_gemini(failure: Exception) -> Verdict:
    """Classify one failure by the documented status table.

    Source: https://ai.google.dev/gemini-api/docs/troubleshooting, read 2026-08-03.
    Statuses 408 and 502 also come from the SDK retryable set.
    Unlisted statuses of 500 or above return `RetryThisOne`.
    Other unlisted statuses return `DoNotRetry`.
    Response headers take precedence over body `RetryInfo` for `retry_after`.
    `TransientError` uses `verdict_from_transient_error`.
    Exceptions outside `failure_types` return `DoNotRetry`.
    """
    if isinstance(failure, TransientError):
        return verdict_from_transient_error(failure)
    if not isinstance(failure, errors.APIError):
        record_parse_fallthrough(
            PARSE_FALLTHROUGH_COUNTS,
            parse_name="parse_gemini",
            status_code=None,
            error_type=type(failure).__name__,
        )
        return DoNotRetry()
    retry_after = _retry_after_seconds_from_error(failure)
    if failure.code in _PAUSE_STATUSES:
        return PauseAll(retry_after=retry_after)
    if failure.code in _RETRY_THIS_ONE_STATUSES:
        return RetryThisOne(retry_after=retry_after)
    if failure.code in _DO_NOT_RETRY_STATUSES:
        return DoNotRetry()
    record_parse_fallthrough(
        PARSE_FALLTHROUGH_COUNTS,
        parse_name="parse_gemini",
        status_code=failure.code,
        error_type=failure.status,
    )
    if failure.code is not None and failure.code >= 500:
        return RetryThisOne(retry_after=retry_after)
    return DoNotRetry()


class GeminiGenerateContentAdapter(Adapter):
    """Adapter over a genai.Client, reaching the Gemini Developer API or Vertex AI.

    client.vertexai selects the platform, so __init__ validates provider_name.
    """

    def __init__(
        self,
        *,
        client: genai.Client,
        model: str,
        pricing: Mapping[str, GeminiPricingTable],
        provider_name: str,
        service_tier: GeminiServiceTier | None = None,
    ) -> None:
        """Store the SDK client and request pricing without sending a request.

        `provider_name` records the provider reached by `client`.
        A Vertex AI client requires `"gcp.vertex_ai"`.
        Per-request options disable retries without copying the client.
        `pricing` maps each `GeminiPricedServiceTier` to rates.
        Missing reported tiers cost NaN.
        Missing `traffic_type` uses `"ON_DEMAND"`.
        `service_tier` requests a tier, while reported `traffic_type` selects pricing.

        Raises:
            ValueError: `pricing` lacks `"ON_DEMAND"` or a Vertex client uses another `provider_name`.
        """
        require_pricing_key(pricing, key=_ON_DEMAND_TIER, model=model)
        if client.vertexai and provider_name != _VERTEX_PROVIDER_NAME:
            raise ValueError(
                f"provider_name={provider_name!r} contradicts the client: "
                f"client.vertexai is True, which reaches {_VERTEX_PROVIDER_NAME!r}"
            )
        super().__init__(
            client=client,
            model=model,
            provider_name=provider_name,
            automatic_cache_breakpoints_default=False,
        )
        self.client: genai.Client = client
        self.pricing: Mapping[str, GeminiPricingTable] = pricing
        self.service_tier: GeminiServiceTier | None = service_tier

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return stored request configuration outside `Binding`."""
        return {"service_tier": self.service_tier}

    def _bound_config(
        self, binding: Binding, *, response_json_schema: dict[str, object] | None
    ) -> tuple[types.GenerateContentConfig, frozenset[str]]:
        """Convert one binding to its shared `GenerateContentConfig`.

        Every config disables SDK retries.
        Structured configs send `response_json_schema` with `response_mime_type="application/json"`.

        Raises:
            pydantic.ValidationError: A provider-executed tool fails `types.Tool` validation.
            ValueError: A marked system part, disabled parallel calls, empty `system_prompt`, or reserved key is used.
            ValueError: Provider-executed tools use a choice other than "auto", a pre-Gemini-3 model, or Vertex AI.
            ValueError: A configured charged rate is not finite and nonnegative.
        """
        _reject_extra_body_keys(binding.extra_body)
        if not binding.parallel_tool_calls:
            raise ValueError(
                "parallel_tool_calls False has no Gemini wire form: generateContent has no "
                "parameter disabling parallel function calls"
            )
        if binding.provider_executed_tools and binding.tool_choice != "auto":
            raise ValueError(
                "Gemini provider_executed_tools require tool_choice='auto'. "
                "Gemini ToolConfig selects function declarations only."
            )
        if binding.provider_executed_tools and self.provider_name == _VERTEX_PROVIDER_NAME:
            raise ValueError("Gemini provider_executed_tools require the Gemini Developer API")
        normalized_provider_tools, provider_tool_fields = _normalize_provider_tools(
            binding.provider_executed_tools
        )
        _require_provider_tool_support(
            model=self.model,
            pricing=self.pricing,
            provider_tool_fields=provider_tool_fields,
        )
        system_instruction: str | types.Content | None = None
        if binding.system_prompt is not None:
            if isinstance(binding.system_prompt, str):
                system_instruction = binding.system_prompt
            else:
                if not binding.system_prompt:
                    raise ValueError(
                        "system_prompt is an empty tuple of parts; bind rejects this, "
                        "so it can only come from a directly constructed Binding"
                    )
                for part in binding.system_prompt:
                    if part.cache_breakpoint:
                        raise ValueError(
                            f"cache_breakpoint on a system part is unsendable: "
                            f"{_NO_CACHE_BREAKPOINT_WIRE_FORM}"
                        )
                system_instruction = types.Content(
                    parts=[types.Part(text=part.text) for part in binding.system_prompt]
                )
        tools: list[types.Tool] | None = None
        tool_config: types.ToolConfig | None = None
        if binding.tool_schemas:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool_schema.name,
                            description=tool_schema.description,
                            parameters_json_schema=dict(tool_schema.args_schema),
                        )
                        for tool_schema in binding.tool_schemas
                    ]
                )
            ]
            tool_config = _wire_tool_config(binding.tool_choice)
        if binding.provider_executed_tools:
            if tools is None:
                tools = []
            tools.extend(normalized_provider_tools)
            tool_config = types.ToolConfig(
                function_calling_config=(
                    types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.VALIDATED)
                    if binding.tool_schemas
                    else None
                ),
                include_server_side_tool_invocations=True,
            )
        thinking_config: types.ThinkingConfig | None = None
        if binding.reasoning_level is not None:
            thinking_level = types.ThinkingLevel(binding.reasoning_level)
            if thinking_level.value != binding.reasoning_level:
                raise ValueError(
                    f"reasoning_level {binding.reasoning_level!r} is not an exact Gemini value; "
                    f"the Gemini SDK normalizes it to {thinking_level.value!r}"
                )
            thinking_config = types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=True,
            )
        # `HttpOptions.extra_body` requires a `dict`, and the copied values retain their identities.
        return (
            types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                tool_config=tool_config,
                temperature=binding.temperature,
                max_output_tokens=binding.max_completion_tokens,
                thinking_config=thinking_config,
                service_tier=(
                    types.ServiceTier(self.service_tier) if self.service_tier is not None else None
                ),
                response_mime_type="application/json"
                if response_json_schema is not None
                else None,
                response_json_schema=response_json_schema,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1),
                    extra_body=dict(binding.extra_body)
                    if binding.extra_body is not None
                    else None,
                ),
            ),
            provider_tool_fields,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output without I/O.

        Raises:
            pydantic.ValidationError: A provider-executed tool is invalid.
            ValueError: `binding` contains unsupported values.
        """
        config, provider_tool_fields = self._bound_config(binding, response_json_schema=None)
        return _BoundGeminiText(
            adapter=self, config=config, provider_tool_fields=provider_tool_fields
        )

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format without I/O.

        Raises:
            pydantic.ValidationError: A provider-executed tool is invalid.
            ValueError: `binding` contains unsupported values.
            pydantic.PydanticInvalidForJsonSchema: `response_format` cannot produce a JSON schema.
            pydantic.PydanticUserError: `response_format` is not fully defined.
        """
        output_type_adapter: TypeAdapter[ModelT] = TypeAdapter(response_format)
        config, provider_tool_fields = self._bound_config(
            binding, response_json_schema=output_type_adapter.json_schema()
        )
        return _BoundGeminiStructured(
            adapter=self,
            config=config,
            provider_tool_fields=provider_tool_fields,
            output_type_adapter=output_type_adapter,
        )

    failure_types: ClassVar[tuple[type[Exception], ...]] = (errors.APIError, TransientError)
    """The exceptions parse_gemini maps to a verdict.

    `ClientError` and `ServerError` subclass `APIError`.
    A mid-stream error chunk also raises `APIError`.
    """

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Delegate to parse_gemini, whose docstring names the table and the defaults."""
        return parse_gemini(failure)

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Sort an exception parse gave no verdict, or name the terminal error for a DoNotRetry.

        google-genai 2.16.0 retry_args classifies both httpx exceptions as retryable transport failures.
        A 4xx APIError is an invalid request. Any other APIError is unknown_exception.
        Anything else the SDK raises is unknown_exception, which fails this item without a retry.
        """
        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
            return "transient"
        if not isinstance(error, errors.APIError):
            return "unknown_exception"
        if error.code is not None and 400 <= error.code < 500:
            return "invalid_request"
        return "unknown_exception"


class _ResponseAccumulator:
    """Assemble streamed chunks into one GenerateContentResponse.

    `assembled_response` uses this class.

    _GeminiStream reads the public accumulated values before response() is built.
    """

    def __init__(self) -> None:
        self.parts: list[types.Part] = []
        self.provider_tool_parts: list[types.Part] = []
        self.finish_reason: types.FinishReason | None = None
        self.usage_metadata: types.GenerateContentResponseUsageMetadata | None = None
        self.model_version: str | None = None
        self.response_id: str | None = None
        self.prompt_feedback: types.GenerateContentResponsePromptFeedback | None = None
        self.grounding_metadata: types.GroundingMetadata | None = None

    def add(self, chunk: types.GenerateContentResponse) -> None:
        """Fold one chunk in: scalars take the last non-None value, parts append or merge.

        Only the first part can continue the last accumulated part.
        Continuation requires text-only parts with equal thought values and no prior thought_signature.
        Their text concatenates, and the incoming thought_signature ends the merged part.
        """
        if chunk.usage_metadata is not None:
            self.usage_metadata = chunk.usage_metadata
        if chunk.model_version is not None:
            self.model_version = chunk.model_version
        if chunk.response_id is not None:
            self.response_id = chunk.response_id
        if chunk.prompt_feedback is not None:
            self.prompt_feedback = chunk.prompt_feedback
        candidates = chunk.candidates
        if not candidates:
            return
        for streamed_candidate in candidates:
            if streamed_candidate.content is None or streamed_candidate.content.parts is None:
                continue
            self.provider_tool_parts.extend(
                part
                for part in streamed_candidate.content.parts
                if part.tool_call is not None or part.tool_response is not None
            )
        candidate = candidates[0]
        if candidate.finish_reason is not None:
            self.finish_reason = candidate.finish_reason
        if candidate.grounding_metadata is not None:
            self.grounding_metadata = candidate.grounding_metadata
        for index, part in enumerate(_candidate_parts(chunk)):
            last = self.parts[-1] if self.parts else None
            if index == 0 and last is not None and _continues_text(last, part):
                self.parts[-1] = last.model_copy(
                    update={
                        "text": (last.text or "") + (part.text or ""),
                        "thought_signature": part.thought_signature,
                    }
                )
            else:
                self.parts.append(part)

    def blocked(self) -> bool:
        """Whether a prompt_feedback with a block_reason arrived, which ends a stream without candidates."""
        return self.prompt_feedback is not None and self.prompt_feedback.block_reason is not None

    def response(self) -> types.GenerateContentResponse:
        """Build the assembled response."""
        candidates: list[types.Candidate] | None = None
        if self.parts or self.finish_reason is not None:
            candidates = [
                types.Candidate(
                    content=(
                        types.Content(role="model", parts=list(self.parts)) if self.parts else None
                    ),
                    finish_reason=self.finish_reason,
                    grounding_metadata=self.grounding_metadata,
                )
            ]
        return types.GenerateContentResponse(
            candidates=candidates,
            prompt_feedback=self.prompt_feedback,
            usage_metadata=self.usage_metadata,
            model_version=self.model_version,
            response_id=self.response_id,
        )


def _is_text_only(part: types.Part) -> bool:
    """Whether text is the part's only payload, the precondition for merging it into a neighbor.

    Inspect the dump so an unmodeled payload field prevents merging.
    """
    payload_keys = part.model_dump(exclude_none=True).keys() - {"thought", "thought_signature"}
    return payload_keys == {"text"}


def _continues_text(last: types.Part, incoming: types.Part) -> bool:
    """Whether incoming is the next slice of last: both text-only, same thought flag, last unsigned."""
    return (
        _is_text_only(last)
        and _is_text_only(incoming)
        and bool(last.thought) == bool(incoming.thought)
        and last.thought_signature is None
    )


def _candidate_parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    """Read the first candidate's parts, empty wherever the walk finds None."""
    candidates = chunk.candidates
    if not candidates:
        return []
    content = candidates[0].content
    if content is None or content.parts is None:
        return []
    return content.parts


def assembled_response(
    chunks: Iterable[types.GenerateContentResponse],
) -> types.GenerateContentResponse:
    """Assemble streamed chunks into the one response a whole call would have returned.

    _ResponseAccumulator.add defines the merge rule.
    Adapter conformance tests compare this result with a whole response.
    """
    accumulator = _ResponseAccumulator()
    for chunk in chunks:
        accumulator.add(chunk)
    return accumulator.response()


class _GeminiStream(AdapterStream):
    """One open generateContent stream, wrapping the SDK's chunk iterator."""

    def __init__(
        self,
        *,
        chunks: AsyncIterator[types.GenerateContentResponse],
        pricing: Mapping[str, GeminiPricingTable],
        provider_tool_fields: frozenset[str] = frozenset(),
        first_chunk: types.GenerateContentResponse | None = None,
    ) -> None:
        """Store the SDK iterator and the chunk open_stream pulled ahead of it.

        first_chunk precedes chunks. None means open_stream pulled no chunk.
        """
        self._chunks = chunks
        self._pricing = pricing
        self._provider_tool_fields = provider_tool_fields
        self._first_chunk = first_chunk
        self._accumulator = _ResponseAccumulator()
        self._billing_complete = False

    async def _first_then_rest(self) -> AsyncIterator[types.GenerateContentResponse]:
        """Chain the pulled-ahead first_chunk onto the SDK iterator.

        Yields:
            first_chunk when one was pulled ahead, then every chunk of the SDK iterator.
        """
        if self._first_chunk is not None:
            yield self._first_chunk
        async for chunk in self._chunks:
            yield chunk

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate chunks into answer text, reasoning deltas, and complete tool calls.

        Each function call arrives whole and yields no `ToolCallDelta`.
        A thought signature or following part ends a reasoning part.
        `REASONING_PART_SEPARATOR` separates reasoning parts that emitted text.

        Yields:
            Stream items for parts langchaint models.

        Raises:
            StreamProtocolError: The stream ends without `finish_reason` or `block_reason`.
            errors.APIError: The SDK iterator receives a mid-stream error.
        """
        reasoning_delta_yielded = False
        separator_pending = False
        async for chunk in self._first_then_rest():
            self._accumulator.add(chunk)
            parts = _candidate_parts(chunk)
            for index, part in enumerate(parts):
                if part.function_call is not None:
                    yield _tool_call_from(part.function_call)
                if part.text:
                    if part.thought:
                        if separator_pending:
                            separator_pending = False
                            yield ReasoningDelta(text=REASONING_PART_SEPARATOR)
                        reasoning_delta_yielded = True
                        yield ReasoningDelta(text=part.text)
                    else:
                        yield part.text
                part_ended = part.thought_signature is not None or index < len(parts) - 1
                if part.thought and part_ended:
                    separator_pending = reasoning_delta_yielded
        if self._accumulator.finish_reason is None and not self._accumulator.blocked():
            raise StreamProtocolError("stream ended without a finish reason or a block reason")
        self._billing_complete = True

    @override
    async def final(self) -> types.GenerateContentResponse:
        """Return the response assembled from the stream's chunks, after the stream ends."""
        return self._accumulator.response()

    @override
    def billing_reported(self) -> ProviderBilling | None:
        """Return available billing from accumulated stream evidence.

        Missing usage returns None when no charged provider tool was configured.
        A charged cutoff returns NaN for the provider-tool category.
        """
        charged_fields = self._provider_tool_fields & _CHARGED_PROVIDER_TOOL_FIELDS
        if self._accumulator.usage_metadata is None and not charged_fields:
            return None
        return _billing_from_provider_evidence(
            self._accumulator.usage_metadata,
            self._accumulator.provider_tool_parts,
            self._pricing,
            configured_fields=self._provider_tool_fields,
            billing_complete=self._billing_complete,
        )

    @override
    def request_id(self) -> str | None:
        """Return None: the SDK models no request-id header on a streamed response (google-genai 2.16.0)."""
        return None

    @override
    async def close(self) -> None:
        """Close the SDK's chunk iterator.

        An async generator's aclose is idempotent.

        A test iterator without aclose has no connection to close.
        """
        if isinstance(self._chunks, AsyncGenerator):
            await self._chunks.aclose()


def _as_response(raw: BaseModel) -> types.GenerateContentResponse:
    """Narrow a raw response to the SDK response this adapter produces.

    BoundAdapter accepts BaseModel because the neutral core imports no SDK.
    This adapter produces GenerateContentResponse.

    Raises:
        TypeError: raw is not a genai GenerateContentResponse.
    """
    if not isinstance(raw, types.GenerateContentResponse):
        raise TypeError(f"expected a genai GenerateContentResponse, got {type(raw).__name__}")
    return raw


@dataclass(frozen=True, kw_only=True)
class _FinishedTurn:
    """A candidate langchaint can read a turn from: its finish reason and its converted turn."""

    finish_reason: types.FinishReason
    assistant_message: AssistantMessage


def _finished_turn_or_no_output(
    response: types.GenerateContentResponse,
) -> _FinishedTurn | Refusal | UnfinishedTurn:
    """Read the first candidate as a finished turn, or report why no turn can be read.

    No candidates with block_reason becomes Refusal with an empty turn.
    No candidates without block_reason becomes UnfinishedTurn.
    A candidate without finish_reason also becomes UnfinishedTurn and preserves its partial turn.
    """
    candidates = response.candidates
    if not candidates:
        if (
            response.prompt_feedback is not None
            and response.prompt_feedback.block_reason is not None
        ):
            return Refusal(assistant_message=AssistantMessage(turn=()))
        return UnfinishedTurn(
            reason="gemini returned no candidates and no block reason, so there is no turn to read",
            assistant_message=AssistantMessage(turn=()),
        )
    candidate = candidates[0]
    assistant_message = _assistant_message_from(candidate.content)
    if candidate.finish_reason is None:
        return UnfinishedTurn(
            reason="gemini returned a candidate with no finish_reason, "
            "which langchaint cannot call finished",
            assistant_message=assistant_message,
        )
    return _FinishedTurn(
        finish_reason=candidate.finish_reason, assistant_message=assistant_message
    )


def _normalized_stop_reason(finished_turn: _FinishedTurn) -> StopReason:
    """Map the finish reason to the neutral vocabulary.

    The module docstring states each mapping.
    """
    finish_reason = finished_turn.finish_reason
    if finish_reason == types.FinishReason.MAX_TOKENS:
        return "max_tokens"
    if finish_reason in _REFUSAL_FINISH_REASONS:
        return "refusal"
    if finish_reason == types.FinishReason.STOP:
        return "tool_use" if finished_turn.assistant_message.tool_calls else "end_turn"
    return "other"


def _adapter_result[OutputT](
    finished_turn: _FinishedTurn, output: OutputT
) -> AdapterResult[OutputT]:
    return AdapterResult(
        output=output,
        assistant_message=finished_turn.assistant_message,
        stop_reason=_normalized_stop_reason(finished_turn),
    )


class _BoundGemini[OutputT](BoundAdapter[OutputT], ABC):
    """What both gemini bindings share: the request path, and what a response says about itself.

    A subclass sets _adapter and _config in its own __init__ and implements interpret.
    """

    _adapter: GeminiGenerateContentAdapter
    _config: types.GenerateContentConfig
    _provider_tool_fields: frozenset[str]

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Price the response's counters at the table its reported traffic_type selects.

        Raises:
            TypeError: raw is not a genai GenerateContentResponse.
            pydantic.ValidationError: the response's counters leave a category negative.
        """
        return _billing_from_response(
            _as_response(raw),
            self._adapter.pricing,
            configured_fields=self._provider_tool_fields,
        )

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Combine the response's id and model with request_id.

        Both fields are optional on the SDK response.
        An absent field reports as the empty string.

        Raises:
            TypeError: raw is not a genai GenerateContentResponse.
        """
        response = _as_response(raw)
        return ResponseIdentity(
            model_served=response.model_version or "",
            response_id=response.response_id or "",
            request_id=request_id,
        )

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Convert messages under the binding's config."""
        contents = _request_contents(messages)
        if isinstance(contents, InvalidRequest):
            return contents
        return _GeminiRequestParams(
            model=self._adapter.model, config=self._config, contents=contents
        )

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open one generate_content_stream and return the live stream.

        In google-genai 2.16.0, awaiting generate_content_stream returns an unstarted async generator.
        Pulling first_chunk performs connection I/O before any event is yielded.
        _GeminiStream yields first_chunk first.

        Raises:
            TypeError: request was built by another adapter.
            Exception: The SDK fails to open the stream.
        """
        params = narrowed_request(request, _GeminiRequestParams)
        chunks = await self._adapter.client.aio.models.generate_content_stream(
            model=params.model, contents=params.contents, config=params.config
        )
        try:
            first_chunk = await anext(chunks)
        except StopAsyncIteration:
            first_chunk = None
        return _GeminiStream(
            chunks=chunks,
            pricing=self._adapter.pricing,
            provider_tool_fields=self._provider_tool_fields,
            first_chunk=first_chunk,
        )


class _BoundGeminiText(_BoundGemini[str]):
    """Text-bound adapter: output is the concatenated text of the turn."""

    def __init__(
        self,
        *,
        adapter: GeminiGenerateContentAdapter,
        config: types.GenerateContentConfig,
        provider_tool_fields: frozenset[str],
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._provider_tool_fields = provider_tool_fields

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Read the turn, whose concatenated text is this binding's output.

        Every finished turn returns its text. stop_reason reports refusal or truncation.

        Raises:
            TypeError: raw is not a genai GenerateContentResponse.
        """
        finished_turn = _finished_turn_or_no_output(_as_response(raw))
        if isinstance(finished_turn, NoOutput):
            return finished_turn
        return _adapter_result(finished_turn, finished_turn.assistant_message.text)


class _BoundGeminiStructured[ModelT: BaseModel](_BoundGemini[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the turn's text."""

    def __init__(
        self,
        *,
        adapter: GeminiGenerateContentAdapter,
        config: types.GenerateContentConfig,
        provider_tool_fields: frozenset[str],
        output_type_adapter: TypeAdapter[ModelT],
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._provider_tool_fields = provider_tool_fields
        self._output_type_adapter = output_type_adapter

    def _parsed_outcome(self, finished_turn: _FinishedTurn) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, report a tool-call turn as None, or report why neither exists.

        Local validation preserves the response and rejected text.
        A non-finished reason takes precedence over schema validation.
        MAX_TOKENS takes precedence over SchemaViolation.
        """
        validation_error: ValidationError | None = None
        text = finished_turn.assistant_message.text
        if text:
            try:
                output = self._output_type_adapter.validate_json(text)
                return _adapter_result(finished_turn, output)
            except ValidationError as rejection:
                validation_error = rejection
        finish_reason = finished_turn.finish_reason
        assistant_message = finished_turn.assistant_message
        if finish_reason not in _FINISHED_FINISH_REASONS:
            return UnfinishedTurn(
                reason=f"gemini returned finish_reason {finish_reason.value!r}, "
                f"which langchaint cannot continue",
                assistant_message=assistant_message,
            )
        if assistant_message.tool_calls:
            return _adapter_result(finished_turn, None)
        if finish_reason in _REFUSAL_FINISH_REASONS:
            return Refusal(assistant_message=assistant_message)
        if finish_reason == types.FinishReason.MAX_TOKENS:
            return MaxCompletionTokensExceeded(assistant_message=assistant_message)
        if validation_error is not None:
            return SchemaViolation(
                validation_error_json=validation_error.json(include_url=False),
                assistant_message=assistant_message,
            )
        return EmptyTurn(assistant_message=assistant_message)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, or report why the response produced none.

        Raises:
            TypeError: raw is not a genai GenerateContentResponse.
        """
        finished_turn = _finished_turn_or_no_output(_as_response(raw))
        if isinstance(finished_turn, NoOutput):
            return finished_turn
        return self._parsed_outcome(finished_turn)
