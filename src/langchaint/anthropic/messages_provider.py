"""Adapter for the Anthropic Messages API over the official SDK.

Verified against anthropic 0.116.0:
- `messages.parse(output_format=Model)` returns `ParsedMessage[Model]` with a `parsed_output` property;
  the SDK builds the JSON-schema output format and parses the response text.
- `messages.stream(...)` returns a manager whose entered stream assembles deltas into a `ParsedMessage` snapshot;
  `get_final_message()` returns it.
- `Usage.input_tokens` excludes cache reads and writes,
  so the three package counters map directly and no all-inclusive provider total exists to cross-check.
- `Usage.cache_creation` splits cache writes into `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`,
  which bill at different rates.

Reasoning replay, verified by docs and live runs because it is request-time behavior SDK introspection cannot show:
the API 400s a tool-use continuation unless the latest assistant turn's thinking blocks are re-sent unmodified.
It filters prior turns' thinking blocks itself, so re-emitting every ReasoningTrace unconditionally is safe.
It rejects consecutive thinking blocks re-sent out of their emission order, which turn order preserves.
It rejects thinking re-fed on a request whose binding enables no reasoning;
the adapter surfaces that provider error rather than silencing it.

Cache breakpoints: with automatic_prompt_caching bound True,
the bound adapter puts one `cache_control` marker at the end of the frozen prefix (the system prompt,
or the last tool when no system prompt is bound) at bind time, and one on the last block of each request's messages,
so the cached span grows with the conversation.
Bound False, the adapter writes no marker of its own.
A part with cache_breakpoint True adds a marker under either binding: on the part's own text or image block
in a user message, and on the enclosing tool_result block for the last part of a ToolMessage
(the API documents cache_control on the tool_result block itself; a marked part that is not
the message's last would silently move the boundary to the block's end, so it aborts instead).
A system_prompt bound as parts renders one system block per part, marked parts carrying cache_control,
so a breakpoint can sit inside the frozen prefix (stable instructions marked, semi-stable context after).
The API allows at most 4 cache_control markers per request.
The binding's own markers (marked system parts, the automatic frozen-prefix and last-message markers)
spend slots first; a binding whose markers alone exceed the limit fails at bind with ValueError.
The remainder is the per-request budget for marked message parts:
the latest marks up to that budget are written and older ones left unwritten,
mirroring openai's documented latest-N rule so a conversation that accrues one mark per turn keeps working.
Every marker carries the adapter's cache_ttl ("5m" by default, omitting the ttl key since it is the API default,
so the wire form matches markers written before cache_ttl existed; "1h" writes ttl "1h",
whose writes bill at the PricingTable's cache_write_1h_usd_per_million_tokens).

Mapping decisions:
- ToolMessage becomes a `tool_result` block inside a user message;
  consecutive tool results group into one user message because the API requires alternating roles.
- `stop_reason` maps end_turn/tool_use/max_tokens/refusal to themselves and every other value to "other".
- `reasoning_effort` maps to `output_config.effort`.
"""

import base64
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, override

import anthropic
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    Omit,
    omit,
)
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    ImageBlockParam,
    MessageParam,
    OutputConfigParam,
    ParsedMessage,
    RedactedThinkingBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolChoiceParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from pydantic import BaseModel

from langchaint.exceptions import (
    AbortBatchError,
    ExceededMaxCompletionTokensError,
    RefusalError,
    StreamProtocolError,
    TransientError,
)
from langchaint.messages import (
    AssistantMessage,
    Message,
    Part,
    ReasoningTrace,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnElement,
    UserMessage,
)
from langchaint.pricing import CostBreakdown, PriceableCounts, price
from langchaint.provider import (
    Binding,
    BoundProvider,
    ErrorClass,
    PricingTable,
    Provider,
    ProviderResult,
    ProviderStream,
    SpecificTool,
    StreamItem,
    ToolChoice,
    retry_after_seconds_from_headers,
)
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


_CACHE_MARKER_REQUEST_LIMIT = 4
"""The API allows at most 4 cache_control markers per request; bind-time markers spend slots first."""

type CacheTtl = Literal["5m", "1h"]
"""A cache entry's time to live, the two tiers the API offers; writes bill 1.25x ("5m") or 2x ("1h") base input."""


def _cache_control_param(cache_ttl: CacheTtl) -> CacheControlEphemeralParam:
    """Build one cache_control marker; "5m" omits the ttl key because it is the API default.

    The omission keeps the "5m" wire form byte-identical to a marker written before cache_ttl existed,
    so upgrading the package alone cannot invalidate a live cache entry.
    """
    if cache_ttl == "5m":
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": "1h"}


@dataclass(frozen=True, kw_only=True)
class _AnthropicRequest:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place;
    passing them as explicit keywords (never **kwargs) keeps the SDK's overload resolution intact.
    """

    model: str
    max_tokens: int
    temperature: float | Omit
    system: list[TextBlockParam] | Omit
    tools: list[ToolParam] | Omit
    tool_choice: ToolChoiceParam | Omit
    output_config: OutputConfigParam | Omit
    automatic_prompt_caching: bool
    cache_ttl: CacheTtl
    message_mark_budget: int
    """What the binding's own markers (system marks, the frozen-prefix and last-message markers) leave
    of the API's 4-marker request limit for per-request marked parts."""


def _part_block(part: Part) -> TextBlockParam | ImageBlockParam:
    """Convert one content Part to its wire block.

    Raises:
        AbortBatchError: an ImagePart's media_type is outside the API's accepted set;
            the same request would be rejected again.
    """
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if part.media_type not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
        raise AbortBatchError(
            f"the Anthropic API accepts image media types "
            f"{_ANTHROPIC_IMAGE_MEDIA_TYPES}, not {part.media_type!r}"
        )
    image_source: Base64ImageSourceParam = {
        "type": "base64",
        "media_type": part.media_type,
        "data": base64.b64encode(part.data).decode("ascii"),
    }
    return {"type": "image", "source": image_source}


def _user_content_blocks(
    user_message: UserMessage,
) -> tuple[list[_ContentBlockParam], list[TextBlockParam | ImageBlockParam]]:
    """Convert one UserMessage's content to wire blocks; an image part propagates _part_block's AbortBatchError.

    The second element holds the blocks whose part sets cache_breakpoint, in content order;
    the caller applies the request-wide marker budget, so no marker is written here.
    """
    blocks: list[_ContentBlockParam] = []
    marked: list[TextBlockParam | ImageBlockParam] = []
    if isinstance(user_message.content, str):
        blocks.append({"type": "text", "text": user_message.content})
        return blocks, marked
    for part in user_message.content:
        block = _part_block(part)
        blocks.append(block)
        if part.cache_breakpoint:
            marked.append(block)
    return blocks, marked


def _tool_result_content(
    content: str | tuple[Part, ...],
) -> str | list[TextBlockParam | ImageBlockParam]:
    """Convert one ToolMessage's content to the tool_result content field.

    A bare string passes through; a sequence of parts becomes wire text and image blocks,
    an image part propagating _part_block's AbortBatchError.
    """
    if isinstance(content, str):
        return content
    return [_part_block(part) for part in content]


def _assistant_content_blocks(assistant_message: AssistantMessage) -> list[_ContentBlockParam]:
    """Convert one AssistantMessage to wire blocks in turn order.

    A ReasoningTrace's reasoning dict goes to the wire unchanged, routed by its own type key,
    because the API rejects a tool-use continuation whose latest thinking block was modified.
    A trace another provider produced goes to the wire the same way and the API rejects its
    unknown type key, so a conversation replayed through the wrong provider fails loudly;
    switching providers means first rebuilding concluded assistant turns without their traces.
    An empty TextPart is skipped because the API rejects empty text blocks.

    Raises:
        json.JSONDecodeError: a tool_call.args_json is not valid JSON.
    """
    blocks: list[_ContentBlockParam] = []
    for element in assistant_message.turn:
        if isinstance(element, TextPart):
            if element.text:
                blocks.append(TextBlockParam(type="text", text=element.text))
        elif isinstance(element, ToolCall):
            blocks.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=element.id,
                    name=element.name,
                    input=json.loads(element.args_json),
                )
            )
        elif isinstance(element, ReasoningTrace):
            # The dict is the producing SDK block's model_dump; when this adapter produced it,
            # its shape is the wire param's by construction, and when another provider did,
            # the API rejects the unknown type key (the loud failure the docstring states).
            # Reconstructing it field by field would risk the exact
            # byte-level change the API rejects. The shallow copy keeps the wire path
            # (which mutates blocks to place cache breakpoints) from ever writing into the
            # frozen message's stored payload.
            blocks.append(
                cast("ThinkingBlockParam | RedactedThinkingBlockParam", dict(element.reasoning))
            )
    return blocks


def _tool_message_is_marked(tool_message: ToolMessage) -> bool:
    """Whether the tool message's last part sets cache_breakpoint, marking the enclosing tool_result block.

    The marker goes on the tool_result block itself, the placement the API documents;
    for the message's last part that is equivalent, because the block's span ends where that part ends.

    Raises:
        AbortBatchError: a part other than the message's last sets cache_breakpoint;
            the enclosing block's marker would silently move the boundary to the block's end,
            and the same request would abort again.
    """
    if isinstance(tool_message.content, str):
        return False
    marked_indexes = [
        index for index, part in enumerate(tool_message.content) if part.cache_breakpoint
    ]
    if not marked_indexes:
        return False
    if marked_indexes != [len(tool_message.content) - 1]:
        raise AbortBatchError(
            "cache_breakpoint on a ToolMessage part is honored only on the message's last part: "
            "the marker goes on the enclosing tool_result block, whose span ends at the last part"
        )
    return True


def _wire_messages(
    conversation: Sequence[Message],
    *,
    automatic_prompt_caching: bool,
    cache_ttl: CacheTtl,
    message_mark_budget: int,
) -> list[MessageParam]:
    """Convert a conversation to wire messages.

    With automatic_prompt_caching, places the per-request cache breakpoint on the last content block,
    so the cached span grows with the conversation.
    A thinking or redacted_thinking last block gets no breakpoint (its wire param has no cache_control key),
    so that request writes none.
    A part with cache_breakpoint marks its own block in a user message
    and the enclosing tool_result block in a tool message;
    the latest marks up to message_mark_budget are written and older ones left unwritten.
    message_mark_budget is what the binding's markers leave of the request limit,
    computed once in _request; at 0, every mark goes unwritten.

    Raises:
        AbortBatchError: an image part's media_type is outside the API's set (from _part_block),
            or a ToolMessage part other than the last sets cache_breakpoint (from the tool_result marking).
        json.JSONDecodeError: a tool_call.args_json is not valid JSON (from _assistant_content_blocks).
    """
    wire: list[tuple[Literal["user", "assistant"], list[_ContentBlockParam]]] = []
    pending_tool_results: list[_ContentBlockParam] = []
    marked_blocks: list[TextBlockParam | ImageBlockParam | ToolResultBlockParam] = []

    def flush_tool_results() -> None:
        """Group buffered consecutive tool results into one user message."""
        if pending_tool_results:
            wire.append(("user", list(pending_tool_results)))
            pending_tool_results.clear()

    for message in conversation:
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
    if automatic_prompt_caching and wire:
        last_blocks = wire[-1][1]
        if last_blocks:
            last_block = last_blocks[-1]
            if last_block["type"] != "thinking" and last_block["type"] != "redacted_thinking":
                last_block["cache_control"] = _cache_control_param(cache_ttl)
    return [MessageParam(role=role, content=blocks) for role, blocks in wire]


def _wire_tool_choice(tool_choice: ToolChoice, *, parallel_tool_calls: bool) -> ToolChoiceParam:
    """Convert the neutral tool choice; neutral "required" is Anthropic "any"."""
    disable_parallel_tool_use = not parallel_tool_calls
    if isinstance(tool_choice, SpecificTool):
        return {
            "type": "tool",
            "name": tool_choice.tool_name,
            "disable_parallel_tool_use": disable_parallel_tool_use,
        }
    if tool_choice == "auto":
        return {"type": "auto", "disable_parallel_tool_use": disable_parallel_tool_use}
    if tool_choice == "required":
        return {"type": "any", "disable_parallel_tool_use": disable_parallel_tool_use}
    return {"type": "none"}


def _wire_tools(
    tool_schemas: tuple[ToolSchema, ...],
    *,
    cache_breakpoint_on_last_tool: bool,
    cache_ttl: CacheTtl,
) -> list[ToolParam]:
    """Convert tool schemas to wire tools.

    cache_breakpoint_on_last_tool puts the frozen-prefix cache breakpoint on the last tool,
    used when no system prompt follows the tools to carry it.
    """
    tools: list[ToolParam] = [
        {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "input_schema": dict(tool_schema.args_schema),
        }
        for tool_schema in tool_schemas
    ]
    if cache_breakpoint_on_last_tool and tools:
        tools[-1]["cache_control"] = _cache_control_param(cache_ttl)
    return tools


def _normalized_stop_reason(stop_reason: str | None) -> StopReason:
    """Map the provider stop reason into the package vocabulary."""
    if stop_reason in ("end_turn", "tool_use", "max_tokens", "refusal"):
        return stop_reason
    return "other"


def _assistant_message_from(message: anthropic.types.Message) -> AssistantMessage:
    """Build the package assistant turn from the SDK message, block order preserved.

    A thinking or redacted_thinking block becomes a ReasoningTrace carrying the block's own
    model_dump for verbatim replay; server tool blocks are dropped (built-in tools are out of scope).
    """
    turn: list[TurnElement] = []
    for block in message.content:
        if block.type == "text":
            turn.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            turn.append(
                ToolCall(id=block.id, name=block.name, args_json=json.dumps(block.input))
            )
        elif block.type in ("thinking", "redacted_thinking"):
            turn.append(
                ReasoningTrace(reasoning=block.model_dump(mode="python", exclude_none=True))
            )
    return AssistantMessage(turn=tuple(turn))


def _normalized_usage(usage: anthropic.types.Usage, pricing: PricingTable) -> Usage:
    """Map the raw counters onto the package's disjoint partition and price them.

    `usage.input_tokens` excludes cache reads and writes (verified against anthropic 0.116.0),
    so it is exactly the uncached-input counter.
    output_tokens_details is optional on the SDK Usage; its thinking_tokens counter is 0 when it is absent.

    Raises:
        AbortBatchError: propagated from _cost_in_usd when the response reports 1-hour cache writes
            but the PricingTable has no cache_write_1h_usd_per_million_tokens.
    """
    output_tokens_details = usage.output_tokens_details
    return Usage(
        input_tokens_cache_read=usage.cache_read_input_tokens or 0,
        input_tokens_cache_write=usage.cache_creation_input_tokens or 0,
        input_tokens_cache_none=usage.input_tokens,
        output_tokens=usage.output_tokens,
        output_tokens_reasoning=(
            output_tokens_details.thinking_tokens if output_tokens_details is not None else 0
        ),
        cost_in_usd=_cost_in_usd(usage=usage, pricing=pricing),
    )


def cost_breakdown(usage_raw: anthropic.types.Usage, pricing: PricingTable) -> CostBreakdown:
    """Exact per-category cost of one response, computed from its raw SDK usage.

    The raw usage is consumed (not the neutral Usage) because only it keeps the 5-minute / 1-hour
    cache-write split the two rates need, and the arithmetic is the same price() call that produced
    the stored Usage.cost_in_usd for the same response, so total_cost_in_usd equals it.

    Raises:
        ValueError: usage_raw reports 1-hour cache writes but pricing has no
            cache_write_1h_usd_per_million_tokens (propagated from price;
            a standalone reporting call has no batch, so this is never AbortBatchError).
    """
    return price(counts=_priceable_counts(usage_raw), pricing=pricing)


def _priceable_counts(usage: anthropic.types.Usage) -> PriceableCounts:
    """Split the raw counters into pricing categories, keeping the two cache-write tiers apart.

    usage.cache_creation splits writes into 5-minute and 1-hour tokens; when it is absent,
    cache_creation_input_tokens bills entirely at the base cache_write_usd_per_million_tokens rate.
    usage.input_tokens excludes cache reads and writes (verified against anthropic 0.116.0),
    so it is exactly the uncached count.
    """
    input_tokens_cache_write = usage.cache_creation_input_tokens or 0
    input_tokens_cache_write_1h = 0
    if usage.cache_creation is not None:
        input_tokens_cache_write = usage.cache_creation.ephemeral_5m_input_tokens
        input_tokens_cache_write_1h = usage.cache_creation.ephemeral_1h_input_tokens
    return PriceableCounts(
        input_tokens_cache_none=usage.input_tokens,
        input_tokens_cache_read=usage.cache_read_input_tokens or 0,
        input_tokens_cache_write=input_tokens_cache_write,
        input_tokens_cache_write_1h=input_tokens_cache_write_1h,
        output_tokens=usage.output_tokens,
    )


def _cost_in_usd(usage: anthropic.types.Usage, pricing: PricingTable) -> float:
    """Price the raw counts for the generation path, where a batch exists.

    Shares price() and _priceable_counts with the public cost_breakdown,
    so the stored Usage.cost_in_usd and a reported breakdown cannot disagree.

    Raises:
        AbortBatchError: the response reports 1-hour cache writes but the PricingTable has no
            cache_write_1h_usd_per_million_tokens. A pricing-table defect dooms every sibling
            sharing it, so it aborts the batch, carrying the billed 200's raw usage as evidence.
    """
    try:
        return price(counts=_priceable_counts(usage), pricing=pricing).total_cost_in_usd
    except ValueError as exc:
        raise AbortBatchError(str(exc), usage_raw=usage) from exc


def _provider_result[OutputT](
    message: anthropic.types.Message, output: OutputT, pricing: PricingTable
) -> ProviderResult[OutputT]:
    """Normalize one completed message around already-extracted output.

    Raises:
        AbortBatchError: propagated from _normalized_usage when the response reports 1-hour cache writes
            but the PricingTable has no cache_write_1h_usd_per_million_tokens.
    """
    return ProviderResult(
        output=output,
        assistant_message=_assistant_message_from(message),
        usage=_normalized_usage(message.usage, pricing=pricing),
        usage_raw=message.usage,
        stop_reason=_normalized_stop_reason(message.stop_reason),
        raw=message,
    )


class AnthropicMessagesProvider(Provider):
    """Adapter over an AsyncAnthropic, AsyncAnthropicBedrock, or AsyncAnthropicBedrockMantle client.

    The three clients expose the same messages.create/parse/stream methods and with_options,
    so the adapter logic is identical across the first-party API and both Bedrock surfaces.
    default_max_completion_tokens fills the API-required max_tokens
    when the binding's inference_params leave max_completion_tokens None.
    """

    name = "anthropic_messages"

    def __init__(
        self,
        *,
        client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
        model: str,
        pricing: PricingTable,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTtl = "5m",
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        The stored client is a with_options(max_retries=0) copy: the package's retry loop owns all retrying,
        counts every request as an attempt, and feeds rate-limit errors to the RateLimiter,
        so the SDK must never retry beneath it.
        The copy re-feeds client._client (the caller's httpx.AsyncClient) explicitly:
        the two Bedrock client classes override copy() without the "http_client or self._client" reuse the
        base AsyncAnthropic.copy has (anthropic 0.116.0), so a plain with_options rebuilds a fresh default
        transport and drops a custom transport (loaded certs, proxy). Passing it back keeps it; the value is
        the SDK client's own httpx client, re-entering the same SDK's copy, so the private read is known-true.
        cache_ttl applies uniformly to every cache_control marker this adapter writes,
        automatic and cache_breakpoint alike; "5m" is the API default and writes bill 1.25x base input,
        "1h" holds entries across longer gaps and writes bill 2x
        (priced by the PricingTable's cache_write_1h_usd_per_million_tokens).
        A uniform TTL per adapter also sidesteps the API's rules for mixing TTLs within one request.

        Raises:
            ValueError: cache_ttl is "1h" but pricing has no cache_write_1h_usd_per_million_tokens.
                Every 1-hour marker this adapter would write produces 1-hour cache writes the table
                cannot price, so the first cached response would abort its batch; failing here turns
                that mid-batch abort into an immediate config error before any request is sent.
        """
        if cache_ttl == "1h" and pricing.cache_write_1h_usd_per_million_tokens is None:
            raise ValueError(
                f"cache_ttl='1h' for model {model!r} requires a PricingTable with "
                f"cache_write_1h_usd_per_million_tokens, which prices the 2x 1-hour cache writes"
            )
        super().__init__(model=model, pricing=pricing)
        # client._client is the SDK client's own httpx transport, re-fed to the same SDK's copy to keep
        # a custom transport the Bedrock copy() override would otherwise drop (see the docstring above).
        self.client = client.with_options(max_retries=0, http_client=client._client)  # noqa: SLF001
        self.default_max_completion_tokens = default_max_completion_tokens
        self.cache_ttl: CacheTtl = cache_ttl

    def _request(self, binding: Binding) -> _AnthropicRequest:
        """Precompute the typed request fields the binding determines.

        A str system_prompt is one system block; a parts system_prompt is one block per part,
        each marked part carrying cache_control.
        automatic_prompt_caching marks the last system block (idempotent when it is already marked)
        or, with no system prompt, the last tool.
        The binding's markers spend the API's 4-marker request limit first;
        message_mark_budget carries the remainder to _wire_messages.

        Raises:
            ValueError: the binding's markers alone (marked system parts plus the automatic markers)
                exceed the API's 4-marker request limit; unmark some system parts.
                Also raised on an empty tuple system_prompt,
                which bind rejects and only a directly constructed Binding can carry.
        """
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
            if binding.automatic_prompt_caching:
                system_blocks[-1]["cache_control"] = _cache_control_param(self.cache_ttl)
            bind_marker_count = sum(1 for block in system_blocks if "cache_control" in block)
            system = system_blocks
        tools: list[ToolParam] | Omit = omit
        tool_choice: ToolChoiceParam | Omit = omit
        if binding.tool_schemas:
            cache_breakpoint_on_last_tool = (
                binding.automatic_prompt_caching and binding.system_prompt is None
            )
            tools = _wire_tools(
                binding.tool_schemas,
                cache_breakpoint_on_last_tool=cache_breakpoint_on_last_tool,
                cache_ttl=self.cache_ttl,
            )
            if cache_breakpoint_on_last_tool:
                bind_marker_count += 1
            tool_choice = _wire_tool_choice(
                binding.tool_choice, parallel_tool_calls=binding.parallel_tool_calls
            )
        last_message_marker_count = 1 if binding.automatic_prompt_caching else 0
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
        if binding.inference_params.reasoning_effort is not None:
            output_config = {"effort": binding.inference_params.reasoning_effort}
        return _AnthropicRequest(
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
            automatic_prompt_caching=binding.automatic_prompt_caching,
            cache_ttl=self.cache_ttl,
            message_mark_budget=message_mark_budget,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundProvider[str]:
        """Bind for plain-text output; pure conversion, no I/O."""
        return _BoundAnthropicText(adapter=self, request=self._request(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundProvider[ModelT]:
        """Bind for structured output parsed by the SDK; pure conversion, no I/O."""
        return _BoundAnthropicStructured(
            adapter=self,
            request=self._request(binding),
            response_format=response_format,
        )

    @override
    def classify(self, error: Exception) -> ErrorClass:
        """Map the SDK exception to rate_limit, transient, or abort.

        Rate limit: RateLimitError (429) and OverloadedError (529);
        both mean further requests from this account fail the same way right now,
        so admission should pause account-wide.
        Transient: other 5xx, timeouts, connection failures.
        Everything unrecognized is abort so bugs are not retried.
        """
        if isinstance(error, (anthropic.RateLimitError, anthropic.OverloadedError)):
            return "rate_limit"
        if isinstance(error, (anthropic.InternalServerError, anthropic.APIConnectionError)):
            return "transient"
        return "abort"

    @override
    def retry_after_seconds(self, error: Exception) -> float | None:
        """Read the server-stated wait from the SDK exception's response headers."""
        if isinstance(error, anthropic.APIStatusError):
            return retry_after_seconds_from_headers(error.response.headers)
        return None


class _AnthropicStream[OutputT](ProviderStream[OutputT]):
    """One open Messages stream, backed by the SDK's AsyncMessageStream."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncMessageStream[Any],
        pricing: PricingTable,
        output_from_message: Callable[[ParsedMessage[Any]], OutputT],
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._output_from_message = output_from_message

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate the SDK stream into text chunks and completed tool calls.

        Text chunks are the SDK deltas' own strings, passed through without wrapping.
        A tool call is yielded once, when its content block closes,
        built from the SDK-accumulated block exactly like the non-streaming path.

        Yields:
            Stream items; SDK events the package does not model are dropped.

        Raises:
            StreamProtocolError: the stream ended without a stop reason.
        """
        async for event in self._sdk_stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield event.delta.text
            elif event.type == "content_block_stop" and event.content_block.type == "tool_use":
                yield ToolCall(
                    id=event.content_block.id,
                    name=event.content_block.name,
                    args_json=json.dumps(event.content_block.input),
                )
        if self._sdk_stream.current_message_snapshot.stop_reason is None:
            raise StreamProtocolError("stream ended without a stop reason")

    @override
    async def final(self) -> ProviderResult[OutputT]:
        """Return the SDK-assembled result after the stream ends."""
        message = await self._sdk_stream.get_final_message()
        return _provider_result(
            message=message, output=self._output_from_message(message), pricing=self._pricing
        )

    @override
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundAnthropicText(BoundProvider[str]):
    """Text-bound provider: output is the concatenated text of the turn."""

    def __init__(self, *, adapter: AnthropicMessagesProvider, request: _AnthropicRequest) -> None:
        self._adapter = adapter
        self._request = request

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[str]:
        """Send one non-streaming request via messages.create."""
        message = await self._adapter.client.messages.create(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation,
                automatic_prompt_caching=self._request.automatic_prompt_caching,
                cache_ttl=self._request.cache_ttl,
                message_mark_budget=self._request.message_mark_budget,
            ),
        )
        return _provider_result(
            message=message,
            output=_assistant_message_from(message).text,
            pricing=self._adapter.pricing,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[str]:
        """Open one streaming request; connection failures raise here."""
        manager = self._adapter.client.messages.stream(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation,
                automatic_prompt_caching=self._request.automatic_prompt_caching,
                cache_ttl=self._request.cache_ttl,
                message_mark_budget=self._request.message_mark_budget,
            ),
        )
        sdk_stream = await manager.__aenter__()
        return _AnthropicStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            output_from_message=lambda message: _assistant_message_from(message).text,
        )


class _BoundAnthropicStructured[ModelT: BaseModel](BoundProvider[ModelT]):
    """Structured-bound provider: output is the SDK-parsed response_format instance."""

    def __init__(
        self,
        *,
        adapter: AnthropicMessagesProvider,
        request: _AnthropicRequest,
        response_format: type[ModelT],
    ) -> None:
        self._adapter = adapter
        self._request = request
        self._response_format = response_format

    def _parsed_output(self, message: ParsedMessage[ModelT]) -> ModelT:
        """Extract the parsed instance, or raise the error that classifies why the turn produced none.

        Each raised error carries this attempt's billing (usage with cost_in_usd, usage_raw,
        stop_reason) so a rejected 200's cost is not lost.

        Raises:
            RefusalError: the model refused (stop_reason "refusal"); terminal per-item, not retried.
            ExceededMaxCompletionTokensError: the response hit the token cap (stop_reason "max_tokens");
                terminal per-item, not retried.
            TransientError: the turn completed but carried no parsed output for another reason,
                which a later attempt may fix.
            AbortBatchError: propagated from _normalized_usage when the response reports 1-hour cache writes
                but the PricingTable has no cache_write_1h_usd_per_million_tokens.
        """
        parsed_output = message.parsed_output
        if parsed_output is None:
            usage = _normalized_usage(message.usage, pricing=self._adapter.pricing)
            stop_reason = _normalized_stop_reason(message.stop_reason)
            if message.stop_reason == "refusal":
                raise RefusalError.for_rejected_200(
                    usage=usage, usage_raw=message.usage, stop_reason=stop_reason
                )
            if message.stop_reason == "max_tokens":
                raise ExceededMaxCompletionTokensError.for_rejected_200(
                    usage=usage, usage_raw=message.usage, stop_reason=stop_reason
                )
            raise TransientError(
                "structured response contained no parsed output",
                usage=usage,
                usage_raw=message.usage,
                stop_reason=stop_reason,
            )
        return parsed_output

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[ModelT]:
        """Send one non-streaming request via messages.parse."""
        message = await self._adapter.client.messages.parse(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation,
                automatic_prompt_caching=self._request.automatic_prompt_caching,
                cache_ttl=self._request.cache_ttl,
                message_mark_budget=self._request.message_mark_budget,
            ),
            output_format=self._response_format,
        )
        return _provider_result(
            message=message,
            output=self._parsed_output(message),
            pricing=self._adapter.pricing,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[ModelT]:
        """Open one streaming request; connection failures raise here."""
        manager = self._adapter.client.messages.stream(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation,
                automatic_prompt_caching=self._request.automatic_prompt_caching,
                cache_ttl=self._request.cache_ttl,
                message_mark_budget=self._request.message_mark_budget,
            ),
            output_format=self._response_format,
        )
        sdk_stream = await manager.__aenter__()
        return _AnthropicStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            output_from_message=self._parsed_output,
        )
