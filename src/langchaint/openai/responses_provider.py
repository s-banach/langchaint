"""Adapter for the OpenAI Responses API over the official SDK.

Verified against openai 2.45.0:
- `responses.parse(text_format=Model)` returns `ParsedResponse[Model]`;
  `output_parsed` is the instance from the last message output.
- `responses.stream(...)` returns a manager whose entered stream yields typed events and assembles the response.
  Usage and status arrive on the terminal `response.completed`, `response.incomplete`,
  or `response.failed` event's response;
  the adapter captures that response itself because the SDK's `get_final_response()`
  raises RuntimeError unless the terminal event is `response.completed`.
  Only the `response.completed` event carries a `ParsedResponse`;
  the other terminal responses are re-validated into one,
  whose absent parsed output makes the structured path classify the empty parse (refusal,
  token-cap truncation, or transient).
- `usage.input_tokens` includes `input_tokens_details.cached_tokens` and `input_tokens_details.cache_write_tokens`,
  so it is the provider-reported all-inclusive input total the Usage partition is checked against.
  Cache writes bill starting with gpt-5.6, so the PricingTable's cache-write rate applies here too.
- `prompt_cache_options` (supported on gpt-5.6 and later) controls caching per request;
  `{"mode": "explicit"}` with no explicit breakpoints disables it.
  The adapter sends it only when the binding sets automatic_prompt_caching False; bound True,
  the provider's implicit caching is left in place and nothing is sent.
  The adapter sends it regardless of model (it keeps no model-version table),
  so binding False with a pre-gpt-5.6 model may be rejected by the API.
  `prompt_cache_options.ttl` takes "30m" as its only value,
  so there is no TTL to configure and this adapter has no counterpart to the anthropic adapter's `cache_ttl`.
- A part with cache_breakpoint True becomes `prompt_cache_breakpoint: {"mode": "explicit"}` on its wire part,
  under either binding value: implicit mode writes up to the latest three explicit breakpoints,
  explicit mode up to the latest four, and older marks are read-only for matching,
  so the adapter sends every mark and caps nothing.
  With automatic_prompt_caching False, marked parts are what re-enables caching at exactly those boundaries.
- The API stores responses server-side for later retrieval by default;
  the adapter always sends `store=False` because conversation state is the caller's conversation argument,
  and a stored copy would be an unused side effect.
- The adapter sends `include=["reasoning.encrypted_content"]` on every request,
  so reasoning items come back with `encrypted_content` populated and round-trip statelessly under `store=False`.
  The SDK documents `include` as what populates `encrypted_content`;
  a live run on 2026-07-17 saw it populated without the flag, undocumented behavior the adapter does not rely on.

Mapping decisions:
- A str system_prompt travels as the `instructions` parameter, not as an input item;
  a parts system_prompt travels as a developer-role input message first in every request's input,
  the message the SDK documents `instructions` as inserting, because only input message parts
  carry prompt_cache_breakpoint.
- An AssistantMessage re-feeds its turn elements in emission order,
  which the API requires for replay under store=False:
  a ReasoningTrace is its reasoning item re-sent unchanged, a ToolCall one `function_call` item,
  and a maximal run of adjacent TextParts one assistant message item;
  ToolMessage becomes a `function_call_output` item keyed by call_id.
  The API has no is_error flag, so the error text in output is the only error signal.
- ImagePart becomes an `input_image` item with a data: URI and `detail="auto"`.
- The API reports no finish reason; stop_reason is derived: a `ResponseOutputRefusal` content block means refusal,
  else any `function_call` output item means tool_use, otherwise status "completed" means end_turn,
  status "incomplete" with reason "max_output_tokens" means max_tokens, and anything else is "other".
- Streaming yields the SDK's own delta strings unwrapped and each tool call once, complete,
  from its `response.output_item.done` event; argument fragments are never surfaced.
  Usage, cost, and stop reason arrive only on final()'s ProviderResult.
"""

import base64
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, override

import openai
from openai import AsyncBedrockOpenAI, AsyncOpenAI, Omit, omit
from openai.lib.streaming.responses import AsyncResponseStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCallParam,
    ResponseIncludable,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseUsage,
    ToolChoiceFunctionParam,
)
from openai.types.responses import (
    Response as OpenAIResponse,
)
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response_create_params import PromptCacheOptions
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseFunctionCallOutputItemListParam,
    ResponseInputItemParam,
)
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam

from langchaint.exceptions import (
    MaxCompletionTokensExceededError,
    RefusalError,
    StreamProtocolError,
    TransientError,
)
from langchaint.messages import (
    AssistantMessage,
    ImagePart,
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
    SpecificToolChoice,
    StreamItem,
    ToolChoice,
    retry_after_seconds_from_headers,
)
from langchaint.tools import ToolSchema
from langchaint.usage import ZERO_USAGE, Usage

type _WireToolChoice = Literal["none", "auto", "required"] | ToolChoiceFunctionParam
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""


@dataclass(frozen=True, kw_only=True)
class _OpenAIRequest:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place; passing them as explicit keywords
    (never **kwargs) keeps the SDK's overload resolution intact.
    instructions is the bound str system prompt; a parts system prompt travels in input_prefix instead.
    tool_choice and parallel_tool_calls are omitted without tools because the API rejects them otherwise.
    include is always ["reasoning.encrypted_content"]:
    the adapter re-feeds the whole conversation every turn, so every response's reasoning items
    must carry the payload a later request replays.
    """

    model: str
    instructions: str | None
    input_prefix: list[ResponseInputItemParam]
    """Items sent ahead of the conversation every request: a system_prompt bound as parts becomes
    one developer-role input message here (its parts carry prompt_cache_breakpoint marks,
    which the instructions string cannot), and a str or absent system_prompt leaves it empty."""

    max_output_tokens: int | Omit
    temperature: float | Omit
    reasoning: Reasoning | Omit
    tools: list[FunctionToolParam] | Omit
    tool_choice: _WireToolChoice | Omit
    parallel_tool_calls: bool | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    include: list[ResponseIncludable]


def _image_data_uri(image_part: ImagePart) -> str:
    """Encode an ImagePart as a base64 data: URI carrying its media type."""
    encoded_data = base64.b64encode(image_part.data).decode("ascii")
    return f"data:{image_part.media_type};base64,{encoded_data}"


def _user_item(user_message: UserMessage) -> EasyInputMessageParam:
    """Convert one UserMessage to a user message item.

    A part with cache_breakpoint carries prompt_cache_breakpoint on its wire part;
    the API writes up to the latest four breakpoints per request (three in implicit mode)
    and treats older ones as read-only, so every mark is sent and no client-side cap applies.
    """
    if isinstance(user_message.content, str):
        return {"role": "user", "content": user_message.content}
    parts: ResponseInputMessageContentListParam = []
    for part in user_message.content:
        if isinstance(part, TextPart):
            wire_text: ResponseInputTextParam = {"type": "input_text", "text": part.text}
            if part.cache_breakpoint:
                wire_text["prompt_cache_breakpoint"] = {"mode": "explicit"}
            parts.append(wire_text)
        else:
            wire_image: ResponseInputImageParam = {
                "type": "input_image",
                "image_url": _image_data_uri(part),
                "detail": "auto",
            }
            if part.cache_breakpoint:
                wire_image["prompt_cache_breakpoint"] = {"mode": "explicit"}
            parts.append(wire_image)
    return {"role": "user", "content": parts}


def _function_call_output(content: str | tuple[Part, ...]) -> str | ResponseFunctionCallOutputItemListParam:
    """Convert one ToolMessage's content to the function_call_output output field.

    The installed openai SDK's function_call_output output field is `str | ResponseFunctionCallOutputItemListParam`,
    a list of input_text and input_image content params, so parts carry images to this provider.
    A bare string passes through; a sequence of parts becomes that structured content list.
    The image content param is a distinct wire type from the user-message input_image param,
    so this builds its own dict rather than reusing _user_item's list, sharing only the data: URI encoding.
    A part with cache_breakpoint carries prompt_cache_breakpoint on its wire part,
    under the same latest-N server rule _user_item's docstring states.
    """
    if isinstance(content, str):
        return content
    output_content: ResponseFunctionCallOutputItemListParam = []
    for part in content:
        if isinstance(part, TextPart):
            output_text: ResponseInputTextContentParam = {"type": "input_text", "text": part.text}
            if part.cache_breakpoint:
                output_text["prompt_cache_breakpoint"] = {"mode": "explicit"}
            output_content.append(output_text)
        else:
            output_image: ResponseInputImageContentParam = {
                "type": "input_image",
                "image_url": _image_data_uri(part),
                "detail": "auto",
            }
            if part.cache_breakpoint:
                output_image["prompt_cache_breakpoint"] = {"mode": "explicit"}
            output_content.append(output_image)
    return output_content


def _assistant_items(assistant_message: AssistantMessage) -> list[ResponseInputItemParam]:
    """Convert one AssistantMessage to its input items in turn order.

    The API requires the original item order for replay under store=False.
    A maximal run of adjacent TextParts becomes one assistant message item whose content joins their texts
    (turn carries no message-item boundary, so the run is the inverse of the produce rule's per-part split);
    each ToolCall becomes a function_call item keyed by call_id,
    which the paired ToolMessage's function_call_output references.
    A ReasoningTrace's reasoning dict goes to the wire unchanged, routed by its own type key,
    so encrypted_content replays byte-identical.
    A trace another provider produced goes to the wire the same way and the API rejects its
    unknown type key, so a conversation replayed through the wrong provider fails loudly;
    switching providers means first rebuilding concluded assistant turns without their traces.
    """
    items: list[ResponseInputItemParam] = []
    pending_texts: list[str] = []

    def flush_text_run() -> None:
        """Emit the buffered adjacent TextParts as one assistant message item."""
        if pending_texts:
            items.append({"role": "assistant", "content": "".join(pending_texts)})
            pending_texts.clear()

    for element in assistant_message.turn:
        if isinstance(element, TextPart):
            if element.text:
                pending_texts.append(element.text)
        elif isinstance(element, ToolCall):
            flush_text_run()
            function_call_item: ResponseFunctionToolCallParam = {
                "type": "function_call",
                "call_id": element.id,
                "name": element.name,
                "arguments": element.args_json,
            }
            items.append(function_call_item)
        elif isinstance(element, ReasoningTrace):
            flush_text_run()
            # The dict is the producing SDK item's model_dump; when this adapter produced it,
            # its shape is the wire param's by construction, so the cast holds. A trace another
            # provider produced is not this shape; it is passed through unchanged, never dropped
            # or neutralized here (trimming is the app's job), and left to the API.
            # Reconstructing it field by field would risk changing the
            # payload the API re-reads. The shallow copy keeps the wire path from ever aliasing
            # the frozen message's stored payload into a mutable request structure.
            items.append(cast("ResponseReasoningItemParam", dict(element.reasoning)))
    flush_text_run()
    return items


def _wire_input(conversation: Sequence[Message]) -> list[ResponseInputItemParam]:
    """Convert a conversation to input items; the system prompt is not one."""
    wire: list[ResponseInputItemParam] = []
    for message in conversation:
        if isinstance(message, ToolMessage):
            function_call_output: FunctionCallOutput = {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": _function_call_output(message.content),
            }
            wire.append(function_call_output)
        elif isinstance(message, UserMessage):
            wire.append(_user_item(message))
        else:
            wire.extend(_assistant_items(message))
    return wire


def _wire_tool_choice(tool_choice: ToolChoice) -> _WireToolChoice:
    """Convert the neutral tool choice.

    "auto", "required", and "none" map to the same strings; SpecificToolChoice becomes the named-function form.
    """
    if isinstance(tool_choice, SpecificToolChoice):
        return {"type": "function", "name": tool_choice.tool_name}
    return tool_choice


def _wire_tools(tool_schemas: tuple[ToolSchema, ...]) -> list[FunctionToolParam]:
    """Convert tool schemas to function tools.

    strict is a required key of FunctionToolParam; None leaves the provider's non-strict default in place,
    matching the schemas the ToolManager generates, which are not written to strict mode's restrictions.
    """
    return [
        {
            "type": "function",
            "name": tool_schema.name,
            "description": tool_schema.description,
            "parameters": dict(tool_schema.args_schema),
            "strict": None,
        }
        for tool_schema in tool_schemas
    ]


def _has_refusal(response: OpenAIResponse) -> bool:
    """Whether any output message carries a ResponseOutputRefusal content block."""
    return any(
        content_part.type == "refusal"
        for item in response.output
        if item.type == "message"
        for content_part in item.content
    )


def _normalized_stop_reason(response: OpenAIResponse) -> StopReason:
    """Derive the stop reason; the API reports no finish reason field."""
    if _has_refusal(response):
        return "refusal"
    if any(item.type == "function_call" for item in response.output):
        return "tool_use"
    match response.status:
        case "completed":
            return "end_turn"
        case "incomplete" if (
            response.incomplete_details is not None
            and response.incomplete_details.reason == "max_output_tokens"
        ):
            return "max_tokens"
        case _:
            return "other"


def _assistant_message_from(response: OpenAIResponse) -> AssistantMessage:
    """Build the package assistant turn from the output items, item order preserved.

    A reasoning item becomes a ReasoningTrace carrying the item's own model_dump for verbatim replay;
    a message item becomes one TextPart per output_text content part it holds, in their order
    (a refusal content part is not a TextPart and is not captured);
    built-in tool call items are dropped (built-in tools are out of scope).
    """
    turn: list[TurnElement] = []
    for item in response.output:
        if item.type == "reasoning":
            turn.append(
                ReasoningTrace(reasoning=item.model_dump(mode="python", exclude_none=True))
            )
        elif item.type == "function_call":
            turn.append(ToolCall(id=item.call_id, name=item.name, args_json=item.arguments))
        elif item.type == "message":
            turn.extend(
                TextPart(text=content_part.text)
                for content_part in item.content
                if content_part.type == "output_text"
            )
    return AssistantMessage(turn=tuple(turn))


def _normalized_usage(usage: ResponseUsage, pricing: PricingTable) -> Usage:
    """Map the raw counters onto the package's disjoint partition and price them.

    input_tokens includes cached and cache-write tokens (verified against openai 2.45.0),
    so the uncached counter is the remainder after subtracting them.
    output_tokens_details and its reasoning_tokens counter are both required on the SDK Usage.
    """
    details = usage.input_tokens_details
    return Usage(
        input_tokens_cache_read=details.cached_tokens,
        input_tokens_cache_write=details.cache_write_tokens,
        input_tokens_cache_none=(
            usage.input_tokens - details.cached_tokens - details.cache_write_tokens
        ),
        output_tokens=usage.output_tokens,
        output_tokens_reasoning=usage.output_tokens_details.reasoning_tokens,
        cost_in_usd=_cost_in_usd(usage=usage, pricing=pricing),
    )


def cost_breakdown(usage_raw: ResponseUsage, pricing: PricingTable) -> CostBreakdown:
    """Exact per-category cost of one response, computed from its raw SDK usage.

    The arithmetic is the same price() call that produced the stored Usage.cost_in_usd
    for the same response, so total_cost_in_usd equals it.
    OpenAI has one cache-write tier, priced at cache_write_usd_per_million_tokens,
    so input_tokens_cache_write_1h is always 0 and price's missing-1h-rate ValueError cannot fire here.
    """
    return price(counts=_priceable_counts(usage_raw), pricing=pricing)


def _priceable_counts(usage: ResponseUsage) -> PriceableCounts:
    """Split the raw counters into pricing categories.

    usage.input_tokens includes cached and cache-write tokens (verified against openai 2.45.0),
    so the uncached count is the remainder after subtracting them.
    OpenAI has no 1-hour write tier: every write lands in the base input_tokens_cache_write slot
    and input_tokens_cache_write_1h is always 0.
    """
    details = usage.input_tokens_details
    return PriceableCounts(
        input_tokens_cache_none=(
            usage.input_tokens - details.cached_tokens - details.cache_write_tokens
        ),
        input_tokens_cache_read=details.cached_tokens,
        input_tokens_cache_write=details.cache_write_tokens,
        input_tokens_cache_write_1h=0,
        output_tokens=usage.output_tokens,
    )


def _cost_in_usd(usage: ResponseUsage, pricing: PricingTable) -> float:
    """Price the raw counts through the same price() call cost_breakdown uses.

    Sharing the one arithmetic path keeps the stored Usage.cost_in_usd and a reported breakdown
    from disagreeing. _priceable_counts always fills input_tokens_cache_write_1h with 0,
    so price's missing-1h-rate ValueError cannot fire and no translation to a batch error is needed.
    """
    return price(counts=_priceable_counts(usage), pricing=pricing).total_cost_in_usd


def _provider_result[OutputT](
    response: OpenAIResponse, output: OutputT, pricing: PricingTable
) -> ProviderResult[OutputT]:
    """Normalize one completed request around already-extracted output.

    response.usage is typed optional; a response without it normalizes to ZERO_USAGE (zero cost) and
    usage_raw None.
    """
    return ProviderResult(
        output=output,
        assistant_message=_assistant_message_from(response),
        usage=_normalized_usage(response.usage, pricing=pricing) if response.usage else ZERO_USAGE,
        usage_raw=response.usage,
        stop_reason=_normalized_stop_reason(response),
        raw=response,
    )


class OpenAIResponsesProvider(Provider):
    """Adapter over an AsyncOpenAI or AsyncBedrockOpenAI client."""

    name = "openai_responses"

    def __init__(
        self,
        *,
        client: AsyncOpenAI | AsyncBedrockOpenAI,
        model: str,
        pricing: PricingTable,
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        The stored client is a with_options(max_retries=0) copy: the package's retry loop owns all retrying,
        counts every request as an attempt, and feeds rate-limit errors to the RateLimiter,
        so the SDK must never retry beneath it.
        """
        super().__init__(model=model, pricing=pricing)
        self.client = client.with_options(max_retries=0)

    def _request(self, binding: Binding) -> _OpenAIRequest:
        """Precompute the typed request fields the binding determines.

        A str system_prompt travels as the instructions parameter,
        which the SDK documents as "a system (or developer) message inserted into the model's context".
        A parts system_prompt travels as that message itself, a developer-role input message
        first in every request's input, because only input message parts carry prompt_cache_breakpoint.
        """
        instructions: str | None = None
        input_prefix: list[ResponseInputItemParam] = []
        if isinstance(binding.system_prompt, str):
            instructions = binding.system_prompt
        elif binding.system_prompt is not None:
            system_parts: ResponseInputMessageContentListParam = []
            for part in binding.system_prompt:
                system_text: ResponseInputTextParam = {"type": "input_text", "text": part.text}
                if part.cache_breakpoint:
                    system_text["prompt_cache_breakpoint"] = {"mode": "explicit"}
                system_parts.append(system_text)
            input_prefix.append({"role": "developer", "content": system_parts})
        tools: list[FunctionToolParam] | Omit = omit
        tool_choice: _WireToolChoice | Omit = omit
        parallel_tool_calls: bool | Omit = omit
        if binding.tool_schemas:
            tools = _wire_tools(binding.tool_schemas)
            tool_choice = _wire_tool_choice(binding.tool_choice)
            parallel_tool_calls = binding.parallel_tool_calls
        return _OpenAIRequest(
            model=self.model,
            instructions=instructions,
            input_prefix=input_prefix,
            max_output_tokens=(
                binding.inference_params.max_completion_tokens
                if binding.inference_params.max_completion_tokens is not None
                else omit
            ),
            temperature=(
                binding.inference_params.temperature
                if binding.inference_params.temperature is not None
                else omit
            ),
            reasoning=(
                Reasoning(effort=binding.inference_params.reasoning_effort)
                if binding.inference_params.reasoning_effort is not None
                else omit
            ),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_options=(
                omit if binding.automatic_prompt_caching else PromptCacheOptions(mode="explicit")
            ),
            include=["reasoning.encrypted_content"],
        )

    @override
    def bind_text(self, binding: Binding) -> BoundProvider[str]:
        """Bind for plain-text output; pure conversion, no I/O."""
        return _BoundOpenAIText(adapter=self, request=self._request(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundProvider[ModelT]:
        """Bind for structured output parsed by the SDK; pure conversion, no I/O."""
        return _BoundOpenAIStructured(
            adapter=self,
            request=self._request(binding),
            response_format=response_format,
        )

    @override
    def classify(self, error: Exception) -> ErrorClass:
        """Map the SDK exception to rate_limit, transient, or abort.

        Rate limit: RateLimitError (429), meaning further requests from this account fail the same way right now,
        so admission should pause account-wide.
        Transient: 5xx, timeouts, connection failures.
        Everything unrecognized is abort so bugs are not retried.
        """
        if isinstance(error, openai.RateLimitError):
            return "rate_limit"
        if isinstance(error, (openai.InternalServerError, openai.APIConnectionError)):
            return "transient"
        return "abort"

    @override
    def retry_after_seconds(self, error: Exception) -> float | None:
        """Read the server-stated wait from the SDK exception's response headers."""
        if isinstance(error, openai.APIStatusError):
            return retry_after_seconds_from_headers(error.response.headers)
        return None


class _OpenAIStream[OutputT](ProviderStream[OutputT]):
    """One open Responses stream, backed by the SDK's stream helper."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncResponseStream[Any],
        pricing: PricingTable,
        output_from_response: Callable[[ParsedResponse[Any]], OutputT],
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._output_from_response = output_from_response
        self._terminal_response: ParsedResponse[Any] | OpenAIResponse | None = None

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate the SDK stream into text chunks and completed tool calls.

        Text chunks are the SDK deltas' own strings, passed through without wrapping.
        A tool call is yielded once, complete, from the output_item.done event that carries its finished item.
        The terminal event's response is kept for final(), which must not call the SDK's get_final_response():
        that raises RuntimeError unless the terminal event is response.completed.

        Yields:
            Stream items; SDK events the package does not model (reasoning, built-in tool activity) are dropped.

        Raises:
            StreamProtocolError: the stream ended without a terminal response.
        """
        async for sdk_event in self._sdk_stream:
            if sdk_event.type == "response.output_text.delta":
                yield sdk_event.delta
            elif (
                sdk_event.type == "response.output_item.done"
                and sdk_event.item.type == "function_call"
            ):
                yield ToolCall(
                    id=sdk_event.item.call_id,
                    name=sdk_event.item.name,
                    args_json=sdk_event.item.arguments,
                )
            elif sdk_event.type in ("response.completed", "response.incomplete", "response.failed"):
                self._terminal_response = sdk_event.response
        if self._terminal_response is None:
            raise StreamProtocolError("stream ended without a terminal response")

    @override
    async def final(self) -> ProviderResult[OutputT]:
        """Return the result assembled from the captured terminal response.

        Only response.completed carries a ParsedResponse;
        an incomplete or failed terminal response is re-validated into one whose parsed output is absent,
        so the structured output extractor classifies the empty parse (a RefusalError,
        a MaxCompletionTokensExceededError,
        or a TransientError) while the text extractor returns the partial output_text.

        Raises:
            StreamProtocolError: items() was not exhausted first, so no terminal response was captured.
        """
        if self._terminal_response is None:
            raise StreamProtocolError("final() requires items() to be exhausted first")
        parsed_response = (
            self._terminal_response
            if isinstance(self._terminal_response, ParsedResponse)
            else ParsedResponse[None].model_validate(self._terminal_response.model_dump())
        )
        return _provider_result(
            response=parsed_response,
            output=self._output_from_response(parsed_response),
            pricing=self._pricing,
        )

    @override
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundOpenAIText(BoundProvider[str]):
    """Text-bound provider: output is the concatenated output text."""

    def __init__(self, *, adapter: OpenAIResponsesProvider, request: _OpenAIRequest) -> None:
        self._adapter = adapter
        self._request = request

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[str]:
        """Send one non-streaming request via responses.create."""
        response = await self._adapter.client.responses.create(
            model=self._request.model,
            instructions=self._request.instructions,
            max_output_tokens=self._request.max_output_tokens,
            temperature=self._request.temperature,
            reasoning=self._request.reasoning,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            parallel_tool_calls=self._request.parallel_tool_calls,
            prompt_cache_options=self._request.prompt_cache_options,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
        )
        return _provider_result(
            response=response,
            output=response.output_text,
            pricing=self._adapter.pricing,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[str]:
        """Open one streaming request; connection failures raise here."""
        manager = self._adapter.client.responses.stream(
            model=self._request.model,
            instructions=self._request.instructions,
            max_output_tokens=self._request.max_output_tokens,
            temperature=self._request.temperature,
            reasoning=self._request.reasoning,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            parallel_tool_calls=self._request.parallel_tool_calls,
            prompt_cache_options=self._request.prompt_cache_options,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
        )
        sdk_stream = await manager.__aenter__()
        return _OpenAIStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            output_from_response=lambda response: response.output_text,
        )


class _BoundOpenAIStructured[ModelT: BaseModel](BoundProvider[ModelT]):
    """Structured-bound provider: output is the SDK-parsed response_format instance."""

    def __init__(
        self,
        *,
        adapter: OpenAIResponsesProvider,
        request: _OpenAIRequest,
        response_format: type[ModelT],
    ) -> None:
        self._adapter = adapter
        self._request = request
        self._response_format = response_format

    def _parsed_output(self, response: ParsedResponse[ModelT]) -> ModelT:
        """Extract the parsed instance, or raise the error that classifies why the turn produced none.

        Each raised error carries this attempt's billing (usage with cost_in_usd, usage_raw,
        stop_reason) so a rejected 200's cost is not lost.

        Raises:
            RefusalError: the response carried a ResponseOutputRefusal block; terminal per-item, not retried.
            MaxCompletionTokensExceededError: the response was incomplete for max_output_tokens;
                terminal per-item, not retried.
            TransientError: the turn completed but carried no parsed output for another reason,
                which a later attempt may fix.
        """
        if response.output_parsed is None:
            usage = (
                _normalized_usage(response.usage, pricing=self._adapter.pricing)
                if response.usage
                else ZERO_USAGE
            )
            stop_reason = _normalized_stop_reason(response)
            if stop_reason == "refusal":
                raise RefusalError.for_rejected_200(
                    usage=usage, usage_raw=response.usage, stop_reason=stop_reason
                )
            if (
                response.status == "incomplete"
                and response.incomplete_details is not None
                and response.incomplete_details.reason == "max_output_tokens"
            ):
                raise MaxCompletionTokensExceededError.for_rejected_200(
                    usage=usage, usage_raw=response.usage, stop_reason=stop_reason
                )
            raise TransientError(
                "structured response contained no parsed output",
                usage=usage,
                usage_raw=response.usage,
                stop_reason=stop_reason,
            )
        return response.output_parsed

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[ModelT]:
        """Send one non-streaming request via responses.parse.

        Raises:
            RefusalError, MaxCompletionTokensExceededError, or TransientError: the parse yielded no instance;
                propagated from _parsed_output, which names the condition for each.
        """
        response = await self._adapter.client.responses.parse(
            model=self._request.model,
            instructions=self._request.instructions,
            max_output_tokens=self._request.max_output_tokens,
            temperature=self._request.temperature,
            reasoning=self._request.reasoning,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            parallel_tool_calls=self._request.parallel_tool_calls,
            prompt_cache_options=self._request.prompt_cache_options,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
            text_format=self._response_format,
        )
        return _provider_result(
            response=response,
            output=self._parsed_output(response),
            pricing=self._adapter.pricing,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[ModelT]:
        """Open one streaming request; connection failures raise here."""
        manager = self._adapter.client.responses.stream(
            model=self._request.model,
            instructions=self._request.instructions,
            max_output_tokens=self._request.max_output_tokens,
            temperature=self._request.temperature,
            reasoning=self._request.reasoning,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            parallel_tool_calls=self._request.parallel_tool_calls,
            prompt_cache_options=self._request.prompt_cache_options,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
            text_format=self._response_format,
        )
        sdk_stream = await manager.__aenter__()
        return _OpenAIStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            output_from_response=self._parsed_output,
        )
