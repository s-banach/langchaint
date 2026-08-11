"""Adapter for Google's generateContent API over the google-genai SDK.

Verified against google-genai 2.16.0:
- The client never retries unless retry options are set: `retry_args(None)` stops after one attempt.
  Every request this adapter builds carries `HttpOptions(retry_options=HttpRetryOptions(attempts=1))`,
  and a per-request `http_options` patches over the client's own field by field,
  so `max_attempts` stays true for any client the caller passes.
- `generate_content_stream` yields `GenerateContentResponse` chunks and the SDK has no stream
  assembler, so `assembled_response` here is the assembly; see its docstring for the merge rule.
  A mid-stream error chunk raises `errors.APIError` from the iterator, carrying the body's code.
- `errors.APIError` carries `code`, `status`, `message`, `details`, and an optional httpx `response`;
  no request-id attribute exists, and no response field models one.
- `usage_metadata.prompt_token_count` includes `cached_content_token_count`, and
  `total_token_count` is prompt + candidates + tool_use_prompt + thoughts, so thoughts sit outside
  candidates. No cache-write counter exists: Gemini bills no cache writes.
  That implicit cache reads land in `cached_content_token_count` is stated on
  https://ai.google.dev/gemini-api/docs/caching (read 2026-08-03).
- `Part.thought` marks reasoning and `Part.thought_signature` is bytes; the SDK models serialize
  bytes as base64 in JSON mode and validate them back, so a `mode="json"` dump is the
  round-trippable form ReasoningPart.raw carries.
- The SDK models forbid unknown keys, so a stored dump another provider produced has no wire route
  and build_request reports it as InvalidRequest.
- The SDK enums construct unknown values as synthetic members with a UserWarning, so an effort or
  tier word the SDK does not know still reaches the wire as given.
- `FunctionResponse.response` documents the "output" and "error" keys as function output and error
  details, which is where ToolMessage.content and is_error go.

Reasoning replay, from https://ai.google.dev/gemini-api/docs/thinking (read 2026-08-03): thought
signatures must be resent exactly as received, and they appear on thought steps and built-in tool
steps. The adapter therefore emits a ReasoningPart for every part carrying `thought` or a
`thought_signature` and restores it verbatim with `Part.model_validate_json`. A signature-carrying
part that also holds answer text or a function call additionally yields its TextPart or ToolCall,
so consumers see the turn's content; on replay ReasoningPart.raw already carries that
payload, and the following TurnPart is skipped when it matches, keeping the wire form
byte-identical to what arrived.

Caching: generateContent has no request field enabling, disabling, or marking implicit caching, so
`automatic_prompt_caching` True and False build identical requests, implicit cache reads bill at
the cache-read rate under either value, and no cache write is ever billed. A user-placed
`cache_breakpoint` mark has no wire form either; silently dropping it would misstate the request,
so a marked system part raises ValueError at bind and a marked message part returns InvalidRequest.
An explicit cache resource is reachable through extra_body as `{"cachedContent": ...}`.

ContentPart mappings verified against google-genai 2.17.0:
- `ImagePart` becomes `Part.inline_data`.
- `ImageUrlPart` becomes `Part.file_data`.
- `AudioPart` becomes `Part.inline_data`.
- `ImagePart` and `AudioPart` become `FunctionResponsePart.inline_data` inside `ToolMessage`.
- `ImageUrlPart` becomes `FunctionResponsePart.file_data` inside `ToolMessage`.

Provider-tool mappings were verified against google-genai 2.17.0.
`Part.tool_call` and `Part.tool_response` carry server invocation evidence.

Mapping decisions:
- ToolMessage becomes a `function_response` part inside a user-role Content; consecutive tool
  messages group into one Content. The FunctionResponse `name` is recovered from the ToolCall whose
  id matches `tool_call_id`, so a tool_call_id matching no earlier call is InvalidRequest.
- `FunctionCall.id` is optional on the wire; ToolCall.id is required, so the adapter uses the
  provider's id when present and the function name otherwise, and on replay sends the id only when
  it differs from the name. When the provider issues no ids, two same-name calls in one turn share
  an id and their results align by order.
- `reasoning_effort` "none" sends `thinking_budget=0`, Gemini's disable form. Every other value
  sends `thinking_level` as the word upper-cased plus `include_thoughts=True` (which is what
  returns readable thought summaries), passed through as given so a value or model the provider
  rejects surfaces as its own error.
- `parallel_tool_calls` False raises at bind: no wire form disables parallel function calls.
- `stop_reason`: MAX_TOKENS is "max_tokens"; the refusal finish reasons are "refusal"; STOP is
  "tool_use" when the turn holds a ToolCall and "end_turn" otherwise, because Gemini has no tool
  finish reason; everything else is "other".
- ContextWindowExceeded, ProviderFailedTransiently, and ProviderFailedTerminally are never
  constructed: Gemini reports those conditions as error statuses or mid-stream error chunks, never
  as a parseable failure body on a completed 200.
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

# The types suppression: the SDK publishes this exact import in its own docs.
from google.genai import errors, types  # pyrefly: ignore[implicit-reexport]
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint.adapter import (
    REASONING_PART_SEPARATOR,
    Adapter,
    AdapterResult,
    AdapterStream,
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
408 and 502 are unlisted there; the evidence is the SDK's own default retryable set,
`_RETRY_HTTP_STATUS_CODES == (408, 429, 500, 502, 503, 504)` (google-genai 2.16.0).
"""

_DO_NOT_RETRY_STATUSES = frozenset({400, 403, 404})
"""The statuses the troubleshooting page lists as this request's rejection; a resend fails again."""

PARSE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""How often parse_gemini fell to a status-family default, keyed by status and error status word.

A diagnostic surface, read by no decision: a growing key names a status the tables above should learn.
"""

type GeminiServiceTier = Literal["flex", "standard", "priority"]
"""What a request may ask for: the SDK ServiceTier wire values (google-genai 2.16.0)."""

type GeminiPricedServiceTier = Literal[
    "ON_DEMAND", "ON_DEMAND_PRIORITY", "ON_DEMAND_FLEX", "PROVISIONED_THROUGHPUT"
]
"""What a response's usage_metadata.traffic_type reports having been served at (google-genai 2.16.0).

The pricing mapping's keys are these words, but its key type is str: the SDK types traffic_type as
an open enum that constructs unknown values, so the tier read off a response is a str and the
mapping's docstrings name this vocabulary instead of its type narrowing to it.
Disjoint from GeminiServiceTier: the request and response vocabularies share no word, so the tier
is read off each response.
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

Everything else (LANGUAGE, OTHER, MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, the remaining
image members, UNSPECIFIED, and any value the provider adds later) is a turn langchaint cannot
call finished, reported as UnfinishedTurn by the structured binding.
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

Gemini bills no cache writes, so no rate exists to state; the write counter is always zero, and
category_cost(0, NaN) is 0.0, so nothing NaN-poisons.
"""


@dataclass(frozen=True, kw_only=True)
class GeminiPricingTable:
    """One model's rates at one served tier, with the long-prompt repricing some models carry.

    Google prices some models by prompt length: above long_prompt_threshold_tokens prompt tokens,
    input, cache read, and output all reprice at long_prompt_rates. Models without the split leave
    both long fields None.
    """

    rates: GeminiRates
    google_search_usd_per_query: float | None = None
    google_maps_usd_per_query: float | None = None
    long_prompt_threshold_tokens: int | None = None
    long_prompt_rates: GeminiRates | None = None

    def __post_init__(self) -> None:
        """Require the two long-prompt fields together.

        Raises:
            ValueError: exactly one of long_prompt_threshold_tokens and long_prompt_rates is set;
                a threshold without rates prices nothing, and rates without a threshold never apply.
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
    ) -> Billing:
        """Price one response's counters, at the long-prompt rates when the prompt crosses the threshold.

        prompt_token_count excludes the tool-execution input the categories include.
        The cache-write counter is zero with a NaN price: no write is ever billed, and
        _NO_CACHE_WRITE_RATE says why that poisons nothing.

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
        return Billing(
            usage=Usage(
                input_tokens_cache_read=input_tokens_cache_read,
                input_tokens_cache_write=0,
                input_tokens_cache_none=input_tokens_cache_none,
                output_tokens=output_tokens,
                output_tokens_reasoning=output_tokens_reasoning,
                input_tokens_cache_read_cost_in_usd=category_cost(
                    input_tokens_cache_read, rates.cache_read_usd_per_million_tokens
                ),
                input_tokens_cache_write_cost_in_usd=0.0,
                input_tokens_cache_none_cost_in_usd=category_cost(
                    input_tokens_cache_none, rates.input_cache_none_usd_per_million_tokens
                ),
                output_tokens_cost_in_usd=category_cost(
                    output_tokens, rates.output_usd_per_million_tokens
                ),
                provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
            ),
            service_tier=service_tier,
            usage_raw=usage_raw,
            input_cache_none_usd_per_million_tokens=rates.input_cache_none_usd_per_million_tokens,
            cache_read_usd_per_million_tokens=rates.cache_read_usd_per_million_tokens,
            cache_write_usd_per_million_tokens=_NO_CACHE_WRITE_RATE,
            output_usd_per_million_tokens=rates.output_usd_per_million_tokens,
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

Every nonzero counter costs NaN and every zero counter costs zero, so the paid response survives
carrying a cost that says it is unknown.
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
) -> Billing:
    """Price the reported counters at the table the served tier selects.

    prompt_token_count includes cached_content_token_count (google-genai 2.16.0 field description
    for explicit caches; https://ai.google.dev/gemini-api/docs/caching, read 2026-08-03, for
    implicit reads landing in the same counter), so cache_none is their difference.
    thoughts_token_count sits outside candidates_token_count (the SDK's total_token_count
    description sums them beside prompt), so output is their sum and reasoning is the thoughts
    share. tool_use_prompt_token_count counts tool-execution results fed back as input and sits
    outside prompt_token_count (same total_token_count description); it joins cache_none, where the
    uncached input rate prices it.
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
        google_search_rate,
    ) + invocation_cost_in_usd(
        maps_query_count,
        google_maps_rate,
    )


def _billing_from_provider_evidence(
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
    provider_tool_parts: Sequence[types.Part],
    pricing: Mapping[str, GeminiPricingTable],
    *,
    configured_fields: frozenset[str] = frozenset(),
    billing_complete: bool = True,
) -> Billing:
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
) -> Billing:
    """Price token counters and provider evidence from every candidate."""
    return _billing_from_provider_evidence(
        response.usage_metadata,
        _all_candidate_parts(response),
        pricing,
        configured_fields=configured_fields,
        billing_complete=billing_complete,
    )


class _NotSendableError(Exception):
    """A Sequence[Message] this adapter will not put on the wire, raised by a conversion helper.

    Never leaves this module: _request_contents turns it into the InvalidRequest that build_request
    returns. It exists because a Sequence[Message] is found unsendable several frames below
    build_request, in per-part converters whose callers would each have to thread a union outward
    otherwise.
    """

    def __init__(self, reason: str) -> None:
        """Store what cannot be sent; it becomes the InvalidRequest reason."""
        super().__init__(reason)
        self.reason = reason


def _tool_call_from(function_call: types.FunctionCall) -> ToolCall:
    """Build the neutral ToolCall, synthesizing the id from the name when the provider sent none.

    FunctionCall.name is optional on the SDK model; a call without one is unusable, and inventing a
    name would misstate the turn, so it passes through as the empty string and fails loudly at
    dispatch or on replay.
    """
    name = function_call.name or ""
    return ToolCall(
        id=function_call.id or name,
        name=name,
        args_json=json.dumps(function_call.args or {}),
    )


def _assistant_message_from(content: types.Content | None) -> AssistantMessage:
    """Build the langchaint assistant turn from a candidate's content, part order preserved.

    A part carrying `thought` or a `thought_signature` becomes a ReasoningPart holding the part's
    JSON-mode dump, the round-trippable form for the signature bytes; the module docstring names
    the resend-exactly source. A signature-carrying part that also holds a function call or
    non-thought text additionally yields its ToolCall or TextPart, so the turn's consumers see the
    content; _assistant_wire_parts skips that TurnPart during replay.
    Thought text goes on ReasoningPart.text and nowhere else.
    A part that yields none of those three, executable_code and code_execution_result among them,
    becomes a RawPart holding the same dump, so the turn carries what the response was billed
    for. Empty text is no text, so a part whose text is the empty string yields no TextPart; where
    it carries no function call and no reasoning it becomes a RawPart, and one carrying
    nothing at all replays as the empty-text part that arrived.
    Absent content or parts is an empty turn (a SAFETY candidate can carry a finish_reason and no
    content).
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
        _NotSendableError: args_json parses to something other than a JSON object; the wire field
            holds the parsed arguments object, so nothing else has a wire form.
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

    model_validate_json is the inverse of the JSON-mode dump _assistant_message_from stored,
    restoring the signature bytes from base64.
    part_description names the ReasoningPart or RawPart.

    Raises:
        _NotSendableError: raw does not restore to a Part. The SDK models forbid unknown keys, so a
            dump another provider produced has no Gemini wire route; a raw holding a value JSON
            cannot represent fails the same way.
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

    _assistant_message_from pairs a signature-carrying part with the ToolCall or TextPart holding
    its consumer-facing payload; this is the replay-side match that keeps the pair one wire part.
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

    A ReasoningPart restores to the Part that produced it; when that part carries a function call
    or non-thought text, the paired TurnPart follows it and
    is skipped, so the pair goes back as the one part that arrived. An empty TextPart yields nothing.
    A RawPart restores the same way and carries no such pair.

    Raises:
        _NotSendableError: ReasoningPart.raw or RawPart.raw does not restore to a Gemini Part (from
            _part_from_dump), or a tool call's args_json is not a JSON object (from
            _function_call_from).
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

    Consecutive tool messages group into one user-role Content. Assistant turns record their
    ToolCall ids and names as they pass, which is where a later ToolMessage's FunctionResponse
    name comes from.

    Raises:
        _NotSendableError: a message is unsendable; each per-part converter names its condition.
        json.JSONDecodeError: a tool call's args_json is not valid JSON
            (from _assistant_wire_parts).
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

    The one place a Sequence[Message] this adapter will not put on the wire becomes an
    InvalidRequest. An unparseable tool_call.args_json is one of those: the wire field holds the
    parsed arguments object, so text that is not JSON has nowhere to go.
    """
    try:
        return _wire_contents(messages)
    except _NotSendableError as not_sendable:
        return InvalidRequest(reason=not_sendable.reason)
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

        genai has no omit sentinel (absence is None), so instead of request_json this dumps the SDK
        models with exclude_none. http_options is excluded as transport configuration rather than
        request body, except its extra_body, which the SDK merges into the wire body and so is part
        of the request; a value json cannot render becomes its str(), because a table build must not
        fail over one cell of one row.
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
    """Convert the neutral tool choice; neutral "required" is Gemini mode ANY."""
    if isinstance(tool_choice, SpecificToolChoice):
        function_calling_config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=[tool_choice.tool_name],
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
"""The generationConfig leaves the adapter populates, stored as _normalized_wire_key spells them.

Unlisted leaves (topK, seed, stopSequences) pass through, which is what keeps unmapped generation
parameters reachable.
"""


def _normalized_wire_key(key: str) -> str:
    """Normalize a wire-body key the way the SDK matches extra_body keys to wire keys.

    The SDK's recursive_dict_update matches keys ignoring case and underscores
    (google-genai 2.16.0 _normalize_key_for_matching), so a collision check comparing spellings
    exactly would let "systeminstruction" merge over systemInstruction.
    """
    return key.replace("_", "").lower()


def _reject_extra_body_keys(extra_body: Mapping[str, object] | None) -> None:
    """Refuse the extra_body keys whose SDK merge would silently override the binding.

    The SDK merges extra_body into the wire body recursively with extra_body winning on a matched
    leaf, matching keys as _normalized_wire_key states, so the comparison normalizes both sides:
    the top-level keys the adapter populates are refused, and inside a generationConfig entry the
    leaves the adapter populates are refused while the rest pass.

    Raises:
        ValueError: extra_body holds a refused key, or a key matching generationConfig holds
            something other than an object, which the merge would substitute wholesale for the
            generationConfig fields the adapter populates.
            Also raised when generationConfig contains a non-string key.
    """
    if extra_body is None:
        return
    generation_configs: list[Mapping[str, object]] = []
    for key, nested in extra_body.items():
        if _normalized_wire_key(key) != "generationconfig":
            continue
        if not isinstance(nested, Mapping):
            # TRY004 is suppressed because a bind defect is ValueError by the Adapter.bind_text contract.
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

    details is the error body, which is the whole {"error": {...}} envelope or the inner error dict
    depending on the raise path (google-genai 2.16.0 errors.py), so the read starts at
    details.get("error", details). The RetryInfo entry's shape comes from
    google/rpc/error_details.proto, its "@type" the type URL and its retryDelay a protobuf
    Duration, whose JSON encoding is decimal seconds suffixed "s" ("32s"); the wire shape is not
    introspectable, so the read is defensive throughout.
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
    """Map one GeminiGenerateContentAdapter.failure_types exception to its verdict.

    The listed rows come from the troubleshooting page,
    https://ai.google.dev/gemini-api/docs/troubleshooting (read 2026-08-03):
    _PAUSE_STATUSES are PauseAll, _RETRY_THIS_ONE_STATUSES are RetryThisOne (408 and 502 from the
    SDK's own retryable set, as that constant's docstring states), and _DO_NOT_RETRY_STATUSES are
    DoNotRetry.
    Failures outside the rows take a default, counted in PARSE_FALLTHROUGH_COUNTS and logged: an
    unlisted status of 500 or above is RetryThisOne, one attempt's server-side failure; anything
    else is DoNotRetry, matching the page's rule that 4xx names this request's defect.
    The status picks the verdict; a retry-after only fills its retry_after, read from the response
    headers and then from the body's RetryInfo detail.
    A TransientError takes verdict_from_transient_error's shared mapping.
    Never raises: an Exception outside failure_types is DoNotRetry, counted as a fallthrough.
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

    One client class serves both platforms; client.vertexai says which one a client reaches, so
    provider_name_by_client_class stays empty and __init__ checks the flag itself.
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
        """Store the SDK client, which owns credentials and endpoints.

        provider_name says which platform the client reaches: "gcp.gemini" for the Developer API,
        "gcp.vertex_ai" for Vertex AI (both from the OTel gen_ai.provider.name value set).
        The client is stored as constructed: retry suppression is per-request (the module docstring
        names the mechanism), so no client copy is needed.

        pricing holds one table per traffic_type word a response can report
        (GeminiPricedServiceTier names the vocabulary), and a response reporting a word absent from
        it costs NaN. The "ON_DEMAND" key is required because every response that reports no
        traffic_type prices there.
        service_tier is what the request asks for, None sending nothing. It cannot decide the
        price: the request and response vocabularies are disjoint, and the traffic_type the
        response reports is what selects the table, so a "flex" request needs a pricing entry for
        the tier its responses report.

        Raises:
            ValueError: pricing has no "ON_DEMAND" key; or client.vertexai is True under a
                provider_name other than "gcp.vertex_ai", a request that succeeds and bills
                normally while every span it produces carries the wrong provider.
                A non-Vertex client takes the caller's value, since its base_url decides what it
                reaches.
        """
        require_pricing_key(pricing, key=_ON_DEMAND_TIER, model=model)
        if client.vertexai and provider_name != _VERTEX_PROVIDER_NAME:
            raise ValueError(
                f"provider_name={provider_name!r} contradicts the client: "
                f"client.vertexai is True, which reaches {_VERTEX_PROVIDER_NAME!r}"
            )
        super().__init__(client=client, model=model, provider_name=provider_name)
        self.client = client
        self.pricing = pricing
        self.service_tier: GeminiServiceTier | None = service_tier

    def _bound_config(
        self, binding: Binding, *, response_json_schema: dict[str, object] | None
    ) -> tuple[types.GenerateContentConfig, frozenset[str]]:
        """Convert the binding to the one GenerateContentConfig every request of this binding sends.

        The SDK's own config type is the container for the frozen prefix, so no precomputed-fields
        dataclass stands beside it. Every config carries the retry-suppressing http_options and, for
        a structured binding, response_json_schema with response_mime_type "application/json".

        Raises:
            pydantic.ValidationError: a provider-executed mapping fails `types.Tool` validation.
            ValueError: the binding asks for something this adapter cannot send: a marked system
                part or parallel_tool_calls False (neither has a Gemini wire form), an empty
                system_prompt parts tuple, or a refused extra_body key (from
                _reject_extra_body_keys). Also raised for provider_executed_tools with non-auto
                tool_choice.
                Provider-executed tools require a Gemini 3 model.
                `provider_executed_tools` require the Gemini Developer API.
                google-genai 2.17.0 lacks `include_server_side_tool_invocations` on Vertex AI.
                Configured charged rates must be finite and nonnegative.
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
        reasoning_effort = binding.inference_params.reasoning_effort
        if reasoning_effort == "none":
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        elif reasoning_effort is not None:
            thinking_config = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(reasoning_effort.upper()),
                include_thoughts=True,
            )
        # dict(binding.extra_body): HttpOptions.extra_body is declared dict, so the Mapping is
        # shallow-copied to satisfy it; the values pass through by reference.
        return (
            types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                tool_config=tool_config,
                temperature=binding.inference_params.temperature,
                max_output_tokens=binding.inference_params.max_completion_tokens,
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
        """Bind for plain-text output; pure conversion, no I/O.

        Propagates _bound_config's ValueError.
        """
        config, provider_tool_fields = self._bound_config(binding, response_json_schema=None)
        return _BoundGeminiText(
            adapter=self, config=config, provider_tool_fields=provider_tool_fields
        )

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format; pure conversion, no I/O.

        Propagates _bound_config's ValueError.
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

    APIError is the SDK's one status-error class (ClientError and ServerError subclass it), and a
    mid-stream error chunk raises it too, so every provider-stated failure lands in parse.
    """

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Delegate to parse_gemini, whose docstring names the table and the defaults."""
        return parse_gemini(failure)

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Sort an exception parse gave no verdict, or name the terminal error for a DoNotRetry.

        httpx.TimeoutException and httpx.ConnectError are the transport failures the SDK's own
        retry predicate names retryable (google-genai 2.16.0 retry_args), arriving unparsed:
        transient.
        An APIError only arrives here after parse verdicted DoNotRetry; a 4xx code is this
        request's rejection, and any other code is one langchaint has no account of.
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
    """Assembles streamed chunks into one GenerateContentResponse; assembled_response's engine.

    The attributes are the last-seen scalars and the merged parts, public because _GeminiStream
    reads them mid-stream (billing_reported, the terminal-event check) before response() is built.
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

        Merge rule: only a chunk's first part can continue the last accumulated part, because parts
        arriving as separate entries of one chunk's list are distinct parts. It continues it when
        both are text-only with the same thought flag and the accumulated one carries no
        thought_signature: the text concatenates and the incoming signature is adopted, so a
        signature always ends the part carrying it.
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
        """Build the assembled response; no candidate is emitted when nothing candidate-borne arrived."""
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

    Read off the dump rather than off named fields, so a payload field this module does not
    enumerate (a new SDK media kind) makes the part unmergeable instead of silently dropped.
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

    The SDK has no stream assembler, so this is the adapter's own; _ResponseAccumulator.add states
    the merge rule. Public for adapter-author tests, which run a chunk list through it beside an
    identically shaped whole response.
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

        first_chunk is the chunk open_stream pulled to perform the connection I/O; items() and
        the accumulator process it ahead of chunks. None means no chunk was pulled ahead: a
        stream the provider ended empty, or a directly constructed stream whose chunks hold
        everything.
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
        """Translate chunks into answer text chunks, reasoning text deltas, and completed tool calls.

        Function calls arrive whole in one part, so each yields one complete ToolCall and no ToolCallDelta.
        A turn's reasoning arrives as thought-part text slices, and a reasoning part ends where its
        thought_signature arrives or where another part follows it in the same chunk's list, the
        two boundaries _ResponseAccumulator.add's merge rule keeps. Each boundary between two
        reasoning parts that both streamed text reaches the caller as a REASONING_PART_SEPARATOR
        delta before the next part's first delta.

        Yields:
            Stream items; a part langchaint does not model streams nothing and reaches the caller
            in the turn final()'s assembled response carries.

        Raises:
            StreamProtocolError: the stream ended having seen neither a finish_reason nor a
                block_reason.
            errors.APIError: a mid-stream error chunk raises from the SDK iterator; it propagates
                to the retry loop, which parses it.
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
    def billing_reported(self) -> Billing | None:
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
        """Close the SDK's chunk iterator; an async generator's aclose is idempotent.

        The SDK returns an async generator; an iterator without aclose (a test double) holds no
        connection of its own, so there is nothing to close.
        """
        if isinstance(self._chunks, AsyncGenerator):
            await self._chunks.aclose()


def _as_response(raw: BaseModel) -> types.GenerateContentResponse:
    """Narrow a raw response to the SDK response this adapter produces.

    The BoundAdapter methods that read a response take BaseModel, because BoundLLM holds them and
    the neutral core imports no SDK. Every value reaching them came from this adapter's own stream,
    so another type is a defect in langchaint and not a provider behavior.

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

    No candidates with a block_reason is the provider declining the prompt: Refusal with an empty
    turn. No candidates without one is a response langchaint cannot read a turn from, and so is a
    candidate without a finish_reason, the in-progress state of a turn that never closed: both are
    UnfinishedTurn, carrying whatever partial turn the candidate held.
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
    """Map the finish reason to the neutral vocabulary; the module docstring states each row."""
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
    def billing_from_raw(self, raw: BaseModel) -> Billing:
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

        Both fields are optional on the SDK response; an absent one reports as the empty string,
        which states the response named none without inventing a value.
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
        """Open one generate_content_stream and return the live stream; connection failures raise here.

        The SDK's generate_content_stream returns an unstarted async generator, so its await does
        no connection I/O (google-genai 2.16.0); pulling the first chunk here is what performs it,
        and the stream yields that chunk back first, so a connection failure raises before any
        event is yielded, as the BoundAdapter.open_stream contract states.

        Raises:
            TypeError: request was built by another adapter.
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
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

        Every finished turn is a result: a refusal or a truncation still carries whatever text the
        model wrote, its condition named by the stop reason, and no schema stands between that text
        and the output.

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

        Validating here rather than in the SDK is what puts the response and its text in scope when
        the text is rejected; the anthropic adapter's _parsed_outcome states the shared reasoning,
        and the ordering matches it: the finish reason is read before the rejection, so text the
        token cap cut mid-object is reported as the truncation and not as a violation of the schema
        it was closing.
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
