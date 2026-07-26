"""Adapter for the Anthropic Messages API over the official SDK.

Verified against anthropic 0.120.0:
- A structured binding sends `output_config.format` built as
  `{"schema": transform_schema(TypeAdapter(Model).json_schema()), "type": "json_schema"}`,
  the same `output_config` `messages.parse(output_format=Model)` builds, and validates the response
  text itself. The SDK validates inside `parse` and inside the stream's `content_block_stop`
  handling, one event before the final `output_tokens` and the cache counters arrive, so a rejection
  raised there reaches langchaint with neither the message nor its billing attached.
- `messages.stream(...)` returns a manager whose entered stream assembles deltas into a message
  snapshot; `get_final_message()` returns it.
- `Usage.cache_creation` splits cache writes into `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`,
  which bill at different rates.

`Usage.input_tokens` excludes cache reads and writes, so the three langchaint counters map directly
and no all-inclusive provider total exists to cross-check. That one is verified by docs rather than
by introspection: the SDK documents no relationship among the input counters,
so `_normalized_usage` carries the page that does.

Reasoning replay, verified by docs and live runs because it is request-time behavior SDK introspection cannot show:
the API 400s a tool-use continuation unless the latest assistant turn's thinking blocks are re-sent unmodified.
It filters prior turns' thinking blocks itself, so re-emitting every ReasoningTrace unconditionally is safe.
It rejects consecutive thinking blocks re-sent out of their emission order, which turn order preserves.

Cache breakpoints: with automatic_prompt_caching bound True,
the bound adapter puts one `cache_control` marker at the end of the frozen prefix (the system prompt,
or the last tool when no system prompt is bound) at bind time, and one on the last block of each request's messages,
so the cached span grows with the conversation.
Bound False, the adapter writes no marker of its own.
The adapter never sends the API's top-level automatic cache_control request parameter:
it is unavailable on Bedrock, which this adapter also serves.
A part with cache_breakpoint True adds a marker under either binding: on the part's own text or image block
in a user message, and on the enclosing tool_result block for the last part of a ToolMessage
(the API documents cache_control on the tool_result block itself; a marked part that is not
the message's last would silently move the boundary to the block's end, so it is rejected instead).
A system_prompt bound as parts renders one system block per part, marked parts carrying cache_control,
so a breakpoint can sit inside the frozen prefix (stable instructions marked, semi-stable context after).
The API allows at most 4 cache_control markers per request.
The SDK documents no such limit, so the source is the provider's prompt-caching page, read 2026-07-25:
https://platform.claude.com/docs/en/build-with-claude/prompt-caching
The binding's own markers (marked system parts, the automatic frozen-prefix and last-message markers)
spend slots first; a binding whose markers alone exceed the limit fails at bind with ValueError.
The remainder is the per-request budget for marked message parts:
the latest marks up to that budget are written and older ones left unwritten,
mirroring openai's documented latest-N rule so a conversation that accrues one mark per turn keeps working.
Every marker carries the adapter's cache_ttl ("5m" by default, omitting the ttl key since it is the API default;
"1h" writes ttl "1h", whose writes bill at each table's cache_write_1h_usd_per_million_tokens).

Mapping decisions:
- ToolMessage becomes a `tool_result` block inside a user message;
  consecutive tool results group into one user message because the API requires alternating roles.
- `stop_reason` maps end_turn/tool_use/max_tokens/refusal to themselves and every other value to "other".
- `reasoning_effort` maps to `output_config.effort` plus `thinking={"type": "adaptive"}`:
  effort steers reasoning depth only under adaptive thinking (request-time behavior SDK introspection cannot show),
  so the adapter sends the pair together and never one alone.
  The value passes through as given, wider than the SDK's effort literal
  (ReasoningEffort is the union of both providers' vocabularies, and openai's "none"/"minimal" are outside it),
  and a model or value the API rejects surfaces as the provider's own error.
  `thinking.display` is never sent: it takes `"summarized"` or `"omitted"` and chooses whether thinking
  text is returned or redacted, and the SDK documents the default as `"summarized"`, which returns it,
  so the adapter leaves the default in place rather than opting into redaction.
"""

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
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
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.json_output_format_param import JSONOutputFormatParam
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    BoundAdapter,
    ContextWindowExceeded,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutput,
    NoOutputOutcome,
    Refusal,
    ResponseOutcome,
    SchemaViolation,
    SpecificToolChoice,
    StreamItem,
    ToolChoice,
    UnfinishedTurn,
    classification_from_response,
    retry_after_seconds_from_headers,
)
from langchaint.exceptions import StreamProtocolError
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
from langchaint.pricing import category_cost
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


_RATE_LIMIT_STATUSES = frozenset({429, 529})
"""529 is the SDK's overloaded status (anthropic 0.120.0), which pauses admission like a 429."""

_CACHE_MARKER_REQUEST_LIMIT = 4
"""The API allows at most 4 cache_control markers per request; bind-time markers spend slots first."""

type CacheTTL = Literal["5m", "1h"]
"""A cache entry's time to live, the two tiers the API offers; writes bill 1.25x ("5m") or 2x ("1h") base input."""

type AnthropicServiceTier = Literal["auto", "standard_only"]
"""What a request may ask for (anthropic 0.120.0).

"auto" is a ceiling, not a selector: the SDK documents the parameter as whether to use priority
capacity if available or standard capacity, so no request value names priority.
"standard_only" is the one value that pins a tier.
"""

type AnthropicPricedServiceTier = Literal["standard", "priority", "batch"]
"""What a response reports having been served at (anthropic 0.120.0), and the pricing mapping's key.

Disjoint from AnthropicServiceTier: the request and response vocabularies share no word, and the
response field carries no precondition on the request, so the tier is read off each response.
"""

_STANDARD_TIER: AnthropicPricedServiceTier = "standard"


@dataclass(frozen=True, kw_only=True)
class AnthropicPricingTable:
    """USD prices per one million tokens for one model at one Anthropic service tier.

    Two cache-write rates, because Anthropic bills 5-minute and 1-hour writes differently and one
    response can report tokens written at both TTLs. Both are required: a table that priced only the
    TTL its adapter marks would price the other half of such a response at the wrong rate.
    price() spends both and reports their sum as one cache-write cost, so what reaches Usage is
    the neutral four categories.

    input_cache_none_usd_per_million_tokens prices only the uncached input, the partition's
    input_tokens_cache_none; cache reads and writes bill at their own rates.
    """

    input_cache_none_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_5m_usd_per_million_tokens: float
    cache_write_1h_usd_per_million_tokens: float

    def price(
        self,
        *,
        input_tokens_cache_read: int,
        input_tokens_cache_write_5m: int,
        input_tokens_cache_write_1h: int,
        input_tokens_cache_none: int,
        output_tokens: int,
        output_tokens_reasoning: int,
    ) -> Usage:
        """Price one response's counters, the two cache-write TTLs each at their own rate.

        The two write counts arrive apart and leave together: Usage.input_tokens_cache_write is
        their sum, and input_tokens_cache_write_cost_in_usd is the sum of what each cost.

        Raises:
            pydantic.ValidationError: a counter is negative.
        """
        return Usage(
            input_tokens_cache_read=input_tokens_cache_read,
            input_tokens_cache_write=input_tokens_cache_write_5m + input_tokens_cache_write_1h,
            input_tokens_cache_none=input_tokens_cache_none,
            output_tokens=output_tokens,
            output_tokens_reasoning=output_tokens_reasoning,
            input_tokens_cache_read_cost_in_usd=category_cost(
                input_tokens_cache_read, self.cache_read_usd_per_million_tokens
            ),
            input_tokens_cache_write_cost_in_usd=(
                category_cost(
                    input_tokens_cache_write_5m, self.cache_write_5m_usd_per_million_tokens
                )
                + category_cost(
                    input_tokens_cache_write_1h, self.cache_write_1h_usd_per_million_tokens
                )
            ),
            input_tokens_cache_none_cost_in_usd=category_cost(
                input_tokens_cache_none, self.input_cache_none_usd_per_million_tokens
            ),
            output_tokens_cost_in_usd=category_cost(
                output_tokens, self.output_usd_per_million_tokens
            ),
        )


_UNPRICED = AnthropicPricingTable(
    input_cache_none_usd_per_million_tokens=float("nan"),
    output_usd_per_million_tokens=float("nan"),
    cache_read_usd_per_million_tokens=float("nan"),
    cache_write_5m_usd_per_million_tokens=float("nan"),
    cache_write_1h_usd_per_million_tokens=float("nan"),
)
"""What prices a response reporting a service tier the adapter holds no table for.

Every nonzero counter costs NaN and every zero counter costs zero, so the paid response survives
carrying a cost that says it is unknown.
"""


def _cache_control_param(cache_ttl: CacheTTL) -> CacheControlEphemeralParam:
    """Build one cache_control marker; "5m" omits the ttl key because it is the API default.

    The "5m" wire form must stay byte-stable across releases,
    so that upgrading langchaint alone cannot invalidate a caller's live cache entry.
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
    thinking: ThinkingConfigParam | Omit
    service_tier: AnthropicServiceTier | Omit
    automatic_prompt_caching: bool
    cache_ttl: CacheTTL
    message_mark_budget: int
    """What the binding's own markers (system marks, the frozen-prefix and last-message markers) leave
    of the API's 4-marker request limit for per-request marked parts."""


class _NotSendableError(Exception):
    """A conversation this adapter will not put on the wire, raised by a conversion helper.

    Never leaves this module: _request_messages turns it into the InvalidRequest arm that
    send and open_stream return. It exists because the conversation is found unsendable several
    frames below them, in per-part converters whose callers would each have to thread a union
    outward otherwise.
    """

    def __init__(self, reason: str) -> None:
        """Store what cannot be sent; it becomes the InvalidRequest reason."""
        super().__init__(reason)
        self.reason = reason


def _part_block(part: Part) -> TextBlockParam | ImageBlockParam:
    """Convert one content Part to its wire block.

    Raises:
        _NotSendableError: an ImagePart's media_type is outside the API's accepted set,
            so the API would reject this request; the item fails its own row.
    """
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if part.media_type not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
        raise _NotSendableError(
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
    """Convert one UserMessage's content to wire blocks; an image part propagates _part_block's _NotSendableError.

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
    an image part propagating _part_block's _NotSendableError.
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
            # its shape is the wire param's by construction, so the cast holds. A trace another
            # provider produced is not this shape; it is passed through unchanged, never dropped
            # or neutralized here (trimming is the app's job), and left to the API.
            # Reconstructing it field by field would risk the exact
            # byte-level change replay cannot tolerate. The shallow copy keeps the wire path
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
        _NotSendableError: a part other than the message's last sets cache_breakpoint.
            The API accepts such a request, and the enclosing block's marker silently moves the
            boundary to the block's end, so the wire form would not mean what the message says;
            the adapter reports the conversation InvalidRequest and the item fails its own row.
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
    conversation: Sequence[Message],
    *,
    automatic_prompt_caching: bool,
    cache_ttl: CacheTTL,
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
        _NotSendableError: an image part's media_type is outside the API's set (from _part_block),
            or a ToolMessage part other than the last sets cache_breakpoint (from the tool_result marking).
        json.JSONDecodeError: a tool_call.args_json is not valid JSON (from _assistant_content_blocks).
    """
    wire: list[tuple[Literal["user", "assistant"], list[_ContentBlockParam]]] = []
    pending_tool_results: list[_ContentBlockParam] = []
    marked_blocks: list[TextBlockParam | ImageBlockParam | ToolResultBlockParam] = []

    def flush_tool_results() -> None:
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


def _request_messages(
    conversation: Sequence[Message], request: _AnthropicRequest
) -> list[MessageParam] | InvalidRequest:
    """Convert a conversation under this request's caching parameters, or report it unsendable.

    The one place a _NotSendableError becomes an AttemptOutcome arm.
    Every send and open_stream starts here and returns the InvalidRequest unchanged when it gets one.

    Raises:
        json.JSONDecodeError: a tool_call.args_json is not valid JSON (from _wire_messages).
    """
    try:
        return _wire_messages(
            conversation,
            automatic_prompt_caching=request.automatic_prompt_caching,
            cache_ttl=request.cache_ttl,
            message_mark_budget=request.message_mark_budget,
        )
    except _NotSendableError as not_sendable:
        return InvalidRequest(reason=not_sendable.reason)


def _wire_tool_choice(tool_choice: ToolChoice, *, parallel_tool_calls: bool) -> ToolChoiceParam:
    """Convert the neutral tool choice; neutral "required" is Anthropic "any"."""
    disable_parallel_tool_use = not parallel_tool_calls
    if isinstance(tool_choice, SpecificToolChoice):
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
    cache_ttl: CacheTTL,
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
    if stop_reason in ("end_turn", "tool_use", "max_tokens", "refusal"):
        return stop_reason
    if stop_reason == "model_context_window_exceeded":
        return "context_window_exceeded"
    return "other"


def _unfinished_turn_or_none(
    message: anthropic.types.Message, *, assistant_message: AssistantMessage
) -> UnfinishedTurn | None:
    """Report a 200 that is not a finished turn, or None when the turn finished.

    Continuing an unfinished turn means sending its content back for the provider to resume, which
    this adapter does not do, so returning that content as the result would be silently wrong data.
    Two known cases reach this. "pause_turn" pauses a turn for the caller to continue, which the API
    emits for server-side tool execution this adapter never requests. A null stop reason is the
    in-progress state of a message that is not finished. A stop reason outside the SDK's literal
    lands here too, because a value this adapter cannot place is one it cannot call finished.
    """
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

    The BoundAdapter methods that read a response take BaseModel, because BoundLLM holds them and
    the neutral core imports no SDK. Every value reaching them came from this adapter's own send or
    stream, so another type is a defect in langchaint and not a provider behavior.

    Raises:
        TypeError: raw is not an anthropic Message.
    """
    if not isinstance(raw, anthropic.types.Message):
        raise TypeError(f"expected an anthropic Message, got {type(raw).__name__}")
    return raw


def _first_text_block_text(message: anthropic.types.Message) -> str | None:
    """Return the text of the turn's first text block, None when the turn holds none.

    The block a structured turn's instance is validated from. Reading the first block matches what
    the SDK's own parse yields for every message it does not raise on: it validates every text block
    and returns the first instance, so a message whose first block is not the instance raises there.
    """
    for block in message.content:
        if block.type == "text":
            return block.text
    return None


def _assistant_message_from(message: anthropic.types.Message) -> AssistantMessage:
    """Build the langchaint assistant turn from the SDK message, block order preserved.

    A thinking or redacted_thinking block becomes a ReasoningTrace carrying the block's own
    model_dump for verbatim replay; server tool blocks are dropped (built-in tools are out of scope).
    The two reasoning block types get an arm each because only thinking carries readable text:
    a redacted_thinking block holds an opaque string under data and nothing a reader can display,
    so its trace has no text.
    """
    turn: list[TurnElement] = []
    for block in message.content:
        if block.type == "text":
            turn.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            turn.append(ToolCall(id=block.id, name=block.name, args_json=json.dumps(block.input)))
        elif block.type == "thinking":
            turn.append(
                ReasoningTrace(
                    reasoning=block.model_dump(mode="python", exclude_none=True),
                    text=block.thinking or None,
                )
            )
        elif block.type == "redacted_thinking":
            turn.append(
                ReasoningTrace(reasoning=block.model_dump(mode="python", exclude_none=True))
            )
    return AssistantMessage(turn=tuple(turn))


def _normalized_usage(
    usage: anthropic.types.Usage,
    pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable],
) -> Usage:
    """Map the raw counters onto langchaint's disjoint partition and price them at the served tier.

    `usage.input_tokens` excludes cache reads and writes, so it is exactly the uncached-input counter.
    The SDK documents no relationship among the input counters, so the source is the provider's
    prompt-caching page, which gives the total as
    cache_read_input_tokens + cache_creation_input_tokens + input_tokens, read 2026-07-25:
    https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    output_tokens_details is optional on the SDK Usage.

    usage.cache_creation splits the writes into 5-minute and 1-hour tokens, and one response can
    report both nonzero; when it is absent, cache_creation_input_tokens is read as 5-minute writes.
    Reading the split here and the collapsed count elsewhere would report a write the cost included
    and the counter did not, so both come off the same read and go into one price() call.

    A response reporting no tier prices at the standard rates, which is what a Bedrock response
    needs: the field is unlikely to be populated there and Anthropic's service tiers do not apply.
    """
    output_tokens_details = usage.output_tokens_details
    input_tokens_cache_write_5m = usage.cache_creation_input_tokens or 0
    input_tokens_cache_write_1h = 0
    if usage.cache_creation is not None:
        input_tokens_cache_write_5m = usage.cache_creation.ephemeral_5m_input_tokens
        input_tokens_cache_write_1h = usage.cache_creation.ephemeral_1h_input_tokens
    served_tier = usage.service_tier if usage.service_tier is not None else _STANDARD_TIER
    return pricing.get(served_tier, _UNPRICED).price(
        input_tokens_cache_read=usage.cache_read_input_tokens or 0,
        input_tokens_cache_write_5m=input_tokens_cache_write_5m,
        input_tokens_cache_write_1h=input_tokens_cache_write_1h,
        input_tokens_cache_none=usage.input_tokens,
        output_tokens=usage.output_tokens,
        output_tokens_reasoning=(
            output_tokens_details.thinking_tokens if output_tokens_details is not None else 0
        ),
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


class AnthropicMessagesAdapter(Adapter):
    """Adapter over an AsyncAnthropic, AsyncAnthropicBedrock, or AsyncAnthropicBedrockMantle client.

    The three clients expose the same messages.create/parse/stream methods and with_options,
    so the adapter logic is identical across the first-party API and both Bedrock APIs.
    default_max_completion_tokens fills the API-required max_tokens
    when the binding's inference_params leave max_completion_tokens None.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = {
        AsyncAnthropicBedrock: "aws.bedrock",
        AsyncAnthropicBedrockMantle: "aws.bedrock",
    }
    """AsyncAnthropic is deliberately absent: it reaches whatever its base_url points at.

    Both classes here speak Bedrock's auth and URL scheme, so the class fixes the provider,
    and the caller's stated value stands for anything else.
    """

    def __init__(
        self,
        *,
        client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
        model: str,
        pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable],
        provider_name: str,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        provider_name says which provider the client reaches: "anthropic" for AsyncAnthropic,
        "aws.bedrock" for either Bedrock client class.
        anthropic_model and anthropic_bedrock_model each pass the one value their client serves.
        Both Bedrock classes are in provider_name_by_client_class, so a value contradicting either
        makes Adapter.__init__ raise; an AsyncAnthropic takes the value its caller states, since
        its base_url decides what it reaches.

        The stored client is a with_options(max_retries=0) copy: langchaint's retry loop owns all retrying,
        counts every request as an attempt, and feeds rate-limit errors to the RateLimiter,
        so the SDK must never retry beneath it.
        The copy re-feeds client._client (the caller's httpx.AsyncClient) explicitly:
        the two Bedrock client classes override copy() without the "http_client or self._client" reuse the
        base AsyncAnthropic.copy has (anthropic 0.120.0), so a plain with_options rebuilds a fresh default
        transport and drops a custom transport (loaded certs, proxy). Passing it back keeps it; the value is
        the SDK client's own httpx client, re-entering the same SDK's copy, so the private read is known-true.
        cache_ttl applies uniformly to every cache_control marker this adapter writes,
        automatic and cache_breakpoint alike; "5m" is the API default and writes bill 1.25x base input,
        "1h" holds entries across longer gaps and writes bill 2x
        (priced by each table's cache_write_1h_usd_per_million_tokens).
        A uniform TTL per adapter also sidesteps the API's rules for mixing TTLs within one request:
        mixing is allowed but requires the 1h markers before the 5m markers.
        That ordering rule is docs-only, not SDK-introspectable, stated under "Mixing different TTLs" at
        https://platform.claude.com/docs/en/build-with-claude/prompt-caching.
        cache_ttl is taken here, not by LLM.bind, whose parameters are provider-neutral.
        The openai Responses adapter has no TTL to configure, so no neutral parameter fits.
        A different TTL is therefore a different adapter.

        pricing holds one table per service tier this adapter can price, keyed by what a response
        reports, and a response reporting a tier absent from it costs NaN. The "standard" key is
        required because every response that reports no tier at all prices there.
        service_tier is what the request asks for, None sending nothing. It cannot decide the price:
        the request and response vocabularies are disjoint, and the tier the response reports is
        what selects the table.

        Raises:
            ValueError: pricing has no "standard" key, which would price every response reporting
                no tier, and every standard-tier response, as NaN, with nothing said until the cost
                comes back unknown.
                Also raised by Adapter.__init__ when provider_name contradicts the client's class.
        """
        if _STANDARD_TIER not in pricing:
            raise ValueError(
                f"pricing for model {model!r} has no {_STANDARD_TIER!r} key; "
                f"it prices every response that reports no service tier, so it is required"
            )
        super().__init__(client=client, model=model, provider_name=provider_name)
        # client._client is the SDK client's own httpx transport, re-fed to the same SDK's copy to keep
        # a custom transport the Bedrock copy() override would otherwise drop (see the docstring above).
        self.client = client.with_options(max_retries=0, http_client=client._client)  # noqa: SLF001
        self.pricing = pricing
        self.default_max_completion_tokens = default_max_completion_tokens
        self.cache_ttl: CacheTTL = cache_ttl
        self.service_tier: AnthropicServiceTier | None = service_tier

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
        thinking: ThinkingConfigParam | Omit = omit
        if binding.inference_params.reasoning_effort is not None:
            # cast: deliberate pass-through of the union ReasoningEffort vocabulary, wider than the
            # SDK's effort literal; a value the API rejects surfaces as the provider's own error.
            output_config = cast(
                "OutputConfigParam", {"effort": binding.inference_params.reasoning_effort}
            )
            thinking = {"type": "adaptive"}
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
            thinking=thinking,
            service_tier=self.service_tier if self.service_tier is not None else omit,
            automatic_prompt_caching=binding.automatic_prompt_caching,
            cache_ttl=self.cache_ttl,
            message_mark_budget=message_mark_budget,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output; pure conversion, no I/O."""
        return _BoundAnthropicText(adapter=self, request=self._request(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format; pure conversion, no I/O."""
        return _BoundAnthropicStructured(
            adapter=self,
            request=self._request(binding),
            response_format=response_format,
        )

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Map the SDK exception to rate_limit, transient, invalid_request, or unrecognized.

        A response's status decides, not the SDK exception class: _make_status_error returns a
        specific subclass only for the statuses it lists and the bare APIStatusError for every
        other one (verified against anthropic 0.120.0, Bedrock clients included), so a class list
        would silently drop whatever status the provider adds next.
        classification_from_response holds the shared rule; 529 joins 429 as a rate limit here
        because the SDK reserves it for an overloaded service.

        APIConnectionError, which APITimeoutError subclasses, and RetryableError, the marker the
        SDK's own retry policy honors from middleware, are transient.
        Anything else the SDK raises is unrecognized, which fails this item without a retry.
        """
        if isinstance(error, (anthropic.APIConnectionError, anthropic.RetryableError)):
            return "transient"
        if not isinstance(error, anthropic.APIStatusError):
            return "unrecognized"
        return classification_from_response(
            status_code=error.response.status_code,
            headers=error.response.headers,
            rate_limit_statuses=_RATE_LIMIT_STATUSES,
        )

    @override
    def retry_after_seconds(self, error: Exception) -> float | None:
        """Read the server-stated wait from the SDK exception's response headers."""
        if isinstance(error, anthropic.APIStatusError):
            return retry_after_seconds_from_headers(error.response.headers)
        return None


class _AnthropicStream(AdapterStream):
    """One open Messages stream, backed by the SDK's AsyncMessageStream."""

    def __init__(self, *, sdk_stream: AsyncMessageStream[Any]) -> None:
        self._sdk_stream = sdk_stream

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate the SDK stream into text chunks and completed tool calls.

        A tool call is built from the SDK-accumulated block exactly like the non-streaming path.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.

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
    async def final(self) -> anthropic.types.Message:
        """Return the message the SDK assembled from the stream's events, after the stream ends."""
        return await self._sdk_stream.get_final_message()

    @override
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundAnthropicText(BoundAdapter[str]):
    """Text-bound adapter: output is the concatenated text of the turn."""

    def __init__(self, *, adapter: AnthropicMessagesAdapter, request: _AnthropicRequest) -> None:
        self._adapter = adapter
        self._request = request

    @override
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Price the message's counters at the tier it reports.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        return _normalized_usage(_as_message(raw).usage, pricing=self._adapter.pricing)

    @override
    def interpret(self, raw: BaseModel) -> AdapterResult[str]:
        """Read the turn, whose concatenated text is this binding's output.

        Every message a text binding receives is a result: a stop reason langchaint cannot continue
        still carries the text the model wrote, and no schema stands between that text and the output.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        message = _as_message(raw)
        assistant_message = _assistant_message_from(message)
        return _adapter_result(message, assistant_message.text, assistant_message)

    @override
    async def send(
        self, conversation: Sequence[Message]
    ) -> anthropic.types.Message | InvalidRequest:
        """Send one non-streaming request via messages.create."""
        messages = _request_messages(conversation, self._request)
        if isinstance(messages, InvalidRequest):
            return messages
        return await self._adapter.client.messages.create(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            thinking=self._request.thinking,
            service_tier=self._request.service_tier,
            messages=messages,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream | InvalidRequest:
        """Open one streaming request; connection failures raise here."""
        messages = _request_messages(conversation, self._request)
        if isinstance(messages, InvalidRequest):
            return messages
        manager = self._adapter.client.messages.stream(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            thinking=self._request.thinking,
            service_tier=self._request.service_tier,
            messages=messages,
        )
        sdk_stream = await manager.__aenter__()
        return _AnthropicStream(sdk_stream=sdk_stream)


class _BoundAnthropicStructured[ModelT: BaseModel](BoundAdapter[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the turn's text."""

    def __init__(
        self,
        *,
        adapter: AnthropicMessagesAdapter,
        request: _AnthropicRequest,
        response_format: type[ModelT],
    ) -> None:
        """Precompute the request's output_config, the JSON-schema format merged into the binding's.

        The format is built from the same transform_schema(TypeAdapter(...).json_schema()) call
        messages.parse makes, so the request carries what passing output_format would have sent.
        The merge is what keeps a reasoning effort the binding set: output_config carries both keys.
        """
        self._adapter = adapter
        self._request = request
        self._output_type_adapter: TypeAdapter[ModelT] = TypeAdapter(response_format)
        output_format = JSONOutputFormatParam(
            schema=transform_schema(self._output_type_adapter.json_schema()), type="json_schema"
        )
        bound_output_config = request.output_config
        self._output_config: OutputConfigParam = (
            {"format": output_format}
            if isinstance(bound_output_config, Omit)
            else {**bound_output_config, "format": output_format}
        )

    def _parsed_output(
        self, message: anthropic.types.Message, assistant_message: AssistantMessage
    ) -> ModelT | None | NoOutputOutcome:
        """Validate the turn's text into the instance, report a tool-call turn as None, or report why neither exists.

        Validating here rather than in the SDK is what puts the message and its text in scope when
        the text is rejected: the member returned for a rejection is one the retry loop can place
        against the attempt it already recorded, where a raise from inside the SDK is not.

        None is the tool-call turn and nothing else: the turn is the tool calls, which the assistant
        message carries, so a turn whose text is not the instance yields no instance without anything
        having gone wrong.

        Every other stop reason has a named member, and none of them is retried, because no stop reason
        states an error: the model finished on the terms it reports, so a resend is a fresh sample.
        The stop reason is read before the rejection, so text the token cap cut mid-object is
        reported as the truncation and not as a violation of the schema it was closing.

        Each member carries assistant_message, so the turn a rejected 200 did produce reaches the
        caller on the failure.
        The stop reason chooses the member and is not carried on it: what such a 200 reports is fixed
        by its GenerationError subclass.
        """
        validation_error: ValidationError | None = None
        text = _first_text_block_text(message)
        if text is not None:
            try:
                return self._output_type_adapter.validate_json(text)
            except ValidationError as rejection:
                validation_error = rejection
        unfinished_turn = _unfinished_turn_or_none(message, assistant_message=assistant_message)
        if unfinished_turn is not None:
            return unfinished_turn
        if message.stop_reason == "tool_use":
            return None
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
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Price the message's counters at the tier it reports.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        return _normalized_usage(_as_message(raw).usage, pricing=self._adapter.pricing)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, or report why the message produced none.

        Raises:
            TypeError: raw is not an anthropic Message.
        """
        message = _as_message(raw)
        assistant_message = _assistant_message_from(message)
        output = self._parsed_output(message, assistant_message)
        if isinstance(output, NoOutput):
            return output
        return _adapter_result(message, output, assistant_message)

    @override
    async def send(
        self, conversation: Sequence[Message]
    ) -> anthropic.types.Message | InvalidRequest:
        """Send one non-streaming request via messages.create."""
        messages = _request_messages(conversation, self._request)
        if isinstance(messages, InvalidRequest):
            return messages
        return await self._adapter.client.messages.create(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._output_config,
            thinking=self._request.thinking,
            service_tier=self._request.service_tier,
            messages=messages,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream | InvalidRequest:
        """Open one streaming request; connection failures raise here."""
        messages = _request_messages(conversation, self._request)
        if isinstance(messages, InvalidRequest):
            return messages
        manager = self._adapter.client.messages.stream(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            temperature=self._request.temperature,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._output_config,
            thinking=self._request.thinking,
            service_tier=self._request.service_tier,
            messages=messages,
        )
        sdk_stream = await manager.__aenter__()
        return _AnthropicStream(sdk_stream=sdk_stream)
