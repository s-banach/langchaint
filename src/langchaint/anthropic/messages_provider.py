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

Cache breakpoints: with automatic_prompt_caching bound True,
the bound adapter puts one `cache_control` marker at the end of the frozen prefix (the system prompt,
or the last tool when no system prompt is bound) at bind time, and one on the last block of each request's messages,
so the cached span grows with the conversation.
Bound False, no marker is written and nothing is cached.

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
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, Omit, omit
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    Base64ImageSourceParam,
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


@dataclass(frozen=True, kw_only=True)
class _AnthropicRequest:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place;
    passing them as explicit keywords (never **kwargs) keeps the SDK's overload resolution intact.
    """

    model: str
    max_tokens: int
    system: list[TextBlockParam] | Omit
    tools: list[ToolParam] | Omit
    tool_choice: ToolChoiceParam | Omit
    output_config: OutputConfigParam | Omit
    automatic_prompt_caching: bool


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


def _user_content_blocks(user_message: UserMessage) -> list[_ContentBlockParam]:
    """Convert one UserMessage's content to wire blocks; an image part propagates _part_block's AbortBatchError."""
    if isinstance(user_message.content, str):
        return [{"type": "text", "text": user_message.content}]
    return [_part_block(part) for part in user_message.content]


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


def _is_native_reasoning_trace(reasoning_trace: ReasoningTrace) -> bool:
    """Whether this adapter produced the trace and can re-feed it; a foreign trace is dropped on consume."""
    return reasoning_trace.provider_name == AnthropicMessagesProvider.name


def _assistant_content_blocks(assistant_message: AssistantMessage) -> list[_ContentBlockParam]:
    """Convert one AssistantMessage to wire blocks in turn order.

    A native ReasoningTrace's reasoning dict goes to the wire unchanged, routed by its own type key,
    because the API rejects a tool-use continuation whose latest thinking block was modified;
    a foreign trace is dropped so a conversation replayed through another provider degrades instead of erroring.
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
        elif _is_native_reasoning_trace(element):
            # The dict is this adapter's own SDK block's model_dump, so its shape is the wire
            # param's by construction; reconstructing it field by field would risk the exact
            # byte-level change the API rejects. The shallow copy keeps the wire path
            # (which mutates blocks to place cache breakpoints) from ever writing into the
            # frozen message's stored payload.
            blocks.append(
                cast("ThinkingBlockParam | RedactedThinkingBlockParam", dict(element.reasoning))
            )
    return blocks


def _wire_messages(
    conversation: Sequence[Message], *, automatic_prompt_caching: bool
) -> list[MessageParam]:
    """Convert a conversation to wire messages.

    With automatic_prompt_caching, places the per-request cache breakpoint on the last content block,
    so the cached span grows with the conversation.
    A thinking or redacted_thinking last block gets no breakpoint (its wire param has no cache_control key),
    so that request writes none.

    Raises:
        AbortBatchError: an image part's media_type is outside the API's set (from _part_block).
        json.JSONDecodeError: a tool_call.args_json is not valid JSON (from _assistant_content_blocks).
    """
    wire: list[tuple[Literal["user", "assistant"], list[_ContentBlockParam]]] = []
    pending_tool_results: list[_ContentBlockParam] = []

    def flush_tool_results() -> None:
        """Group buffered consecutive tool results into one user message."""
        if pending_tool_results:
            wire.append(("user", list(pending_tool_results)))
            pending_tool_results.clear()

    for message in conversation:
        if isinstance(message, ToolMessage):
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": _tool_result_content(message.content),
                "is_error": message.is_error,
            })
        elif isinstance(message, UserMessage):
            flush_tool_results()
            wire.append(("user", _user_content_blocks(message)))
        else:
            flush_tool_results()
            wire.append(("assistant", _assistant_content_blocks(message)))
    flush_tool_results()
    if automatic_prompt_caching and wire:
        last_blocks = wire[-1][1]
        if last_blocks:
            last_block = last_blocks[-1]
            if last_block["type"] != "thinking" and last_block["type"] != "redacted_thinking":
                last_block["cache_control"] = {"type": "ephemeral"}
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
    tool_schemas: tuple[ToolSchema, ...], *, cache_breakpoint_on_last_tool: bool
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
        tools[-1]["cache_control"] = {"type": "ephemeral"}
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
                ReasoningTrace(
                    provider_name=AnthropicMessagesProvider.name,
                    reasoning=block.model_dump(mode="python", exclude_none=True),
                )
            )
    return AssistantMessage(turn=tuple(turn))


def _normalized_usage(usage: anthropic.types.Usage) -> Usage:
    """Map the raw counters onto the package's disjoint partition.

    `usage.input_tokens` excludes cache reads and writes (verified against anthropic 0.116.0),
    so it is exactly the uncached-input counter and no provider-reported all-inclusive total exists.
    """
    return Usage(
        input_tokens_cache_read=usage.cache_read_input_tokens or 0,
        input_tokens_cache_write=usage.cache_creation_input_tokens or 0,
        input_tokens_cache_none=usage.input_tokens,
        output_tokens=usage.output_tokens,
        input_tokens_total_provider_reported=None,
    )


def _cost_in_usd(usage: anthropic.types.Usage, pricing: PricingTable) -> float:
    """Price the raw counts.

    5-minute and 1-hour cache writes bill at different rates, split by usage.cache_creation.

    Raises:
        AbortBatchError:
            the response reports 1-hour cache writes but the PricingTable has no cache_write_1h_usd_per_million_tokens.
    """
    cache_write_5m_tokens = usage.cache_creation_input_tokens or 0
    cache_write_1h_tokens = 0
    if usage.cache_creation is not None:
        cache_write_5m_tokens = usage.cache_creation.ephemeral_5m_input_tokens
        cache_write_1h_tokens = usage.cache_creation.ephemeral_1h_input_tokens
    cost_in_usd = (
        usage.input_tokens * pricing.input_cache_none_usd_per_million_tokens
        + (usage.cache_read_input_tokens or 0) * pricing.cache_read_usd_per_million_tokens
        + cache_write_5m_tokens * pricing.cache_write_usd_per_million_tokens
        + usage.output_tokens * pricing.output_usd_per_million_tokens
    ) / 1_000_000
    if cache_write_1h_tokens:
        if pricing.cache_write_1h_usd_per_million_tokens is None:
            raise AbortBatchError(
                "the response reports 1-hour cache writes but the PricingTable "
                "has no cache_write_1h_usd_per_million_tokens"
            )
        cost_in_usd += (
            cache_write_1h_tokens * pricing.cache_write_1h_usd_per_million_tokens / 1_000_000
        )
    return cost_in_usd


def _provider_result[OutputT](
    message: anthropic.types.Message, output: OutputT, pricing: PricingTable
) -> ProviderResult[OutputT]:
    """Normalize one completed message around already-extracted output."""
    return ProviderResult(
        output=output,
        assistant_message=_assistant_message_from(message),
        usage=_normalized_usage(message.usage),
        cost_in_usd=_cost_in_usd(usage=message.usage, pricing=pricing),
        stop_reason=_normalized_stop_reason(message.stop_reason),
        raw=message,
    )


class AnthropicMessagesProvider(Provider):
    """Adapter over an AsyncAnthropic or AsyncAnthropicBedrock client.

    default_max_completion_tokens fills the API-required max_tokens
    when the binding's inference_params leave max_completion_tokens None.
    """

    name = "anthropic_messages"

    def __init__(
        self,
        *,
        client: AsyncAnthropic | AsyncAnthropicBedrock,
        model: str,
        pricing: PricingTable,
        default_max_completion_tokens: int = 4096,
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        The stored client is a with_options(max_retries=0) copy: the package's retry loop owns all retrying,
        counts every request as an attempt, and feeds rate-limit errors to the RateLimiter,
        so the SDK must never retry beneath it.
        """
        super().__init__(model=model, pricing=pricing)
        self.client = client.with_options(max_retries=0)
        self.default_max_completion_tokens = default_max_completion_tokens

    def _request(self, binding: Binding) -> _AnthropicRequest:
        """Precompute the typed request fields the binding determines."""
        max_tokens = binding.inference_params.max_completion_tokens
        system: list[TextBlockParam] | Omit = omit
        if binding.system_prompt is not None:
            system_block: TextBlockParam = {"type": "text", "text": binding.system_prompt}
            if binding.automatic_prompt_caching:
                system_block["cache_control"] = {"type": "ephemeral"}
            system = [system_block]
        tools: list[ToolParam] | Omit = omit
        tool_choice: ToolChoiceParam | Omit = omit
        if binding.tool_schemas:
            tools = _wire_tools(
                binding.tool_schemas,
                cache_breakpoint_on_last_tool=(
                    binding.automatic_prompt_caching and binding.system_prompt is None
                ),
            )
            tool_choice = _wire_tool_choice(
                binding.tool_choice, parallel_tool_calls=binding.parallel_tool_calls
            )
        output_config: OutputConfigParam | Omit = omit
        if binding.inference_params.reasoning_effort is not None:
            output_config = {"effort": binding.inference_params.reasoning_effort}
        return _AnthropicRequest(
            model=self.model,
            max_tokens=(
                max_tokens if max_tokens is not None else self.default_max_completion_tokens
            ),
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            output_config=output_config,
            automatic_prompt_caching=binding.automatic_prompt_caching,
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
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation, automatic_prompt_caching=self._request.automatic_prompt_caching
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
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation, automatic_prompt_caching=self._request.automatic_prompt_caching
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

        Each raised error carries this attempt's billing (usage, cost_in_usd,
        stop_reason) so a rejected 200's cost is not lost.

        Raises:
            RefusalError: the model refused (stop_reason "refusal"); terminal per-item, not retried.
            ExceededMaxCompletionTokensError: the response hit the token cap (stop_reason "max_tokens");
                terminal per-item, not retried.
            TransientError: the turn completed but carried no parsed output for another reason,
                which a later attempt may fix.
        """
        parsed_output = message.parsed_output
        if parsed_output is None:
            usage = _normalized_usage(message.usage)
            cost_in_usd = _cost_in_usd(usage=message.usage, pricing=self._adapter.pricing)
            stop_reason = _normalized_stop_reason(message.stop_reason)
            if message.stop_reason == "refusal":
                raise RefusalError.for_rejected_200(
                    usage=usage, cost_in_usd=cost_in_usd, stop_reason=stop_reason
                )
            if message.stop_reason == "max_tokens":
                raise ExceededMaxCompletionTokensError.for_rejected_200(
                    usage=usage, cost_in_usd=cost_in_usd, stop_reason=stop_reason
                )
            raise TransientError(
                "structured response contained no parsed output",
                usage=usage,
                cost_in_usd=cost_in_usd,
                stop_reason=stop_reason,
            )
        return parsed_output

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[ModelT]:
        """Send one non-streaming request via messages.parse."""
        message = await self._adapter.client.messages.parse(
            model=self._request.model,
            max_tokens=self._request.max_tokens,
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation, automatic_prompt_caching=self._request.automatic_prompt_caching
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
            system=self._request.system,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            output_config=self._request.output_config,
            messages=_wire_messages(
                conversation, automatic_prompt_caching=self._request.automatic_prompt_caching
            ),
            output_format=self._response_format,
        )
        sdk_stream = await manager.__aenter__()
        return _AnthropicStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            output_from_message=self._parsed_output,
        )
