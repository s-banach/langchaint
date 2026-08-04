"""Adapter for the Chat Completions API over the official openai SDK.

Chat Completions is the API OpenAI-compatible providers serve, so this adapter is the route to
DeepSeek, Groq, and xAI through an AsyncOpenAI whose base_url points at them;
langchaint.deepseek wraps it for DeepSeek.

Verified against openai 2.51.0:
- `create(stream=True)` is the request path.
  It returns `AsyncStream[ChatCompletionChunk]` and performs the HTTP send inside the awaited call.
  The SDK's `chat.completions.stream()` helper is not usable here:
  it raises ValueError for any input tool whose `strict` is not True,
  and the schemas the ToolManager generates are non-strict.
- `ChatCompletionStreamState` assembles chunks into the snapshot `final()` returns.
  Constructed bare (no input_tools, no response_format), its `has_parseable_input` is False,
  which gates off the `LengthFinishReasonError` and `ContentFilterFinishReasonError` raises
  inside `handle_chunk`.
  `get_final_completion` is never called: it raises those two unconditionally on finish_reason
  "length" or "content_filter", which would destroy a truncated response and its billing.
  The structured binding validates the response text itself and sends its response_format with
  `"strict": False`, so no SDK frame validates where the response is out of scope.
- `handle_chunk` returns the events it assembled, so fragment merging stays in the SDK:
  a `content.delta` event carries answer text, and a `tool_calls.function.arguments.done` event
  carries the completed name, the accumulated arguments, and the call's index, whose id is read
  off `current_completion_snapshot` at that index.
  A sparse or out-of-order tool-call fragment index raises IndexError inside `handle_chunk`.
- The state's accumulation assigns `usage` from every chunk, resetting it to None on a chunk
  carrying none, and everything else, `service_tier` included, only from the first chunk.
  The stream therefore tracks the last non-None usage itself, and `final()` patches it onto a
  snapshot whose own usage a trailing chunk reset.
  `stream_options={"include_usage": True}` is what produces the usage-bearing final chunk.
- The SDK base model allows extra fields and `model_dump` includes them, so a provider's
  `reasoning_content` survives on `ChatCompletionMessage`, on `ChoiceDelta`, and through stream
  assembly, where string extras concatenate across deltas.
- A mid-stream SSE payload carrying `error` makes the chunk iterator raise the bare
  `openai.APIError`; the four raise sites in `openai._streaming` are the only bare-APIError
  constructions in the SDK, so `type(error) is openai.APIError` selects exactly them and every
  subclass (APIStatusError, APIConnectionError, APIResponseValidationError) propagates untouched.
  `items()` re-raises the bare error as an APIStatusError on the live 200 response, so
  `parse_openai` verdicts it by its error code: a transient code retries, and a terminal or
  unlisted code fails the item.
- `Choice.finish_reason` is statically a required Literal, but a snapshot is built leniently,
  so it reads None at runtime on a stream that never closed.
  `choices` is a required list that can be empty.
- `CompletionUsage` requires `prompt_tokens`, `completion_tokens`, and `total_tokens`;
  `prompt_tokens_details` (`cached_tokens`, `cache_write_tokens`) and
  `completion_tokens_details` (`reasoning_tokens`) are Optional, each counter included.
- `ChatCompletionToolMessageParam.content` is text-only,
  so an ImagePart inside ToolMessage.content is an InvalidRequest.
- `prompt_cache_options` and part-level `prompt_cache_breakpoint` take the same values as on the
  Responses API, and the adapter maps `automatic_prompt_caching` and marked parts exactly as
  Binding.automatic_prompt_caching's docstring states for the openai adapters.

Mapping decisions:
- A str system_prompt becomes one system-role message first in every request;
  a parts system_prompt becomes one system-role message of text parts,
  which carry prompt_cache_breakpoint marks.
- An AssistantMessage replays as one assistant message param, because this wire holds a turn as
  one object rather than as items: the turn's texts join into its `content`, each ToolCall
  becomes one entry of its `tool_calls`, and a ReasoningTrace's raw dict merges into the param,
  putting `reasoning_content` beside `content` in the one message DeepSeek requires it on.
  Replaying the trace verbatim is correct on both DeepSeek paths: outside a tool loop the API
  ignores a replayed `reasoning_content`, and inside one omitting it is a 400
  (https://api-docs.deepseek.com/guides/thinking_mode, read 2026-08-03).
  openai's own Chat Completions returns no reasoning field, so the trace path never fires there.
- `message.refusal` becomes a TextPart in the turn, so the sentences the model wrote to refuse
  are the turn's text and replay as assistant content; the stop reason is "refusal", tested
  ahead of the finish_reason rows, so a refusal arriving with finish_reason "stop" reports it.
- The finish_reason rows: "stop" is "end_turn" or "tool_use" by the turn's calls,
  "tool_calls" is "tool_use", "length" is "max_tokens", "content_filter" is "refusal",
  and "function_call" or an unknown value is "other".
- Streaming yields the SDK's own answer delta strings unwrapped, each `reasoning_content` delta
  in a ReasoningDelta, and each tool call once, complete.
  `reasoning_content` is one concatenated string, so no part separator ever applies.
  Usage, cost, and stop reason arrive only on final()'s AdapterResult.
"""

from abc import ABC
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, cast, override

import openai
from openai import AsyncOpenAI, AsyncStream, Omit, omit
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import PromptCacheOptions
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel, ValidationError

from langchaint.adapter import (
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
    NoOutputOutcome,
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
    reject_extra_body_keys_the_adapter_populates,
    request_id_from_raw,
    request_json,
)
from langchaint.call import ResponseIdentity
from langchaint.exceptions import StreamProtocolError
from langchaint.inference_params import ReasoningEffort
from langchaint.messages import (
    AssistantMessage,
    ImagePart,
    Message,
    ReasoningTrace,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnElement,
    UserMessage,
)
from langchaint.openai.shared import (
    _DEFAULT_TIER,
    _UNPRICED,
    OPENAI_FAILURE_TYPES,
    PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS,
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIServiceTier,
    _image_data_uri,
    _priced_tier,
    classify_openai,
    parse_openai,
    request_id_from_openai_error,
    require_prompt_cache_options_support,
)
from langchaint.pricing import Billing, require_pricing_key
from langchaint.shared_backoff import Verdict
from langchaint.tools import ToolSchema

type _WireToolChoice = Literal["none", "auto", "required"] | ChatCompletionNamedToolChoiceParam
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""


@dataclass(frozen=True, kw_only=True)
class _ChatCompletionsPrecomputedFields:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place;
    passing them as explicit keywords (never **kwargs) keeps the SDK's overload resolution intact.
    tool_choice and parallel_tool_calls are omitted without tools because the API rejects them otherwise.
    """

    model: str
    messages_prefix: list[ChatCompletionMessageParam]
    """Messages sent ahead of the Sequence[Message] every request:
    a bound system_prompt becomes one system-role message here, and an absent one leaves it empty."""

    max_completion_tokens: int | Omit
    temperature: float | Omit
    reasoning_effort: ReasoningEffort | Omit
    tools: list[ChatCompletionFunctionToolParam] | Omit
    tool_choice: _WireToolChoice | Omit
    parallel_tool_calls: bool | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    service_tier: OpenAIServiceTier | Omit
    response_format: ResponseFormatJSONSchema | Omit
    """The structured binding's JSON-schema format, omitted by the text binding, which asks for none."""

    extra_body: Mapping[str, object] | None


_ADAPTER_POPULATED_WIRE_KEYS = frozenset({
    "model",
    "messages",
    "max_completion_tokens",
    "temperature",
    "reasoning_effort",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "prompt_cache_options",
    "service_tier",
    "response_format",
    "stream",
    "stream_options",
})
"""The wire keys an extra_body must not hold: every keyword open_stream passes,
including stream and stream_options, which the stream path depends on."""


@dataclass(frozen=True, kw_only=True)
class _ChatCompletionsRequestParams(RequestParams):
    """One Chat Completions request: the binding's precomputed fields and this call's messages."""

    precomputed: _ChatCompletionsPrecomputedFields
    messages: list[ChatCompletionMessageParam]
    """What goes on the wire as messages: the binding's messages_prefix followed by the Sequence[Message]."""

    @override
    def as_json(self) -> str:
        """Render the request as a JSON object, dropping every field left to the provider's default."""
        return request_json(self, omitted_class=Omit)


class _NotSendableError(Exception):
    """A Sequence[Message] this adapter will not put on the wire, raised by a conversion helper.

    Never leaves this module: build_request turns it into the InvalidRequest it returns.
    It exists because a Sequence[Message] is found unsendable below build_request, in a per-part
    converter whose callers would each have to thread a union outward otherwise.
    """

    def __init__(self, reason: str) -> None:
        """Store what cannot be sent; it becomes the InvalidRequest reason."""
        super().__init__(reason)
        self.reason = reason


def _reasoning_content_extra(model: BaseModel) -> str | None:
    """Read a non-empty reasoning_content string off a model's extra fields, None where there is none.

    reasoning_content is no field of the installed SDK's models, so a provider that returns it
    (DeepSeek) lands it in model_extra, on the whole message and on each stream delta alike.
    """
    extra = model.model_extra
    if extra is None:
        return None
    reasoning_content = extra.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content
    return None


def _text_part_param(part: TextPart) -> ChatCompletionContentPartTextParam:
    """Convert one TextPart to a text content part, marked where the part carries a breakpoint.

    Every mark is sent and no client-side cap applies, the per-request write limits being the API's.
    """
    wire_text: ChatCompletionContentPartTextParam = {"type": "text", "text": part.text}
    if part.cache_breakpoint:
        wire_text["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return wire_text


def _user_message(user_message: UserMessage) -> ChatCompletionUserMessageParam:
    """Convert one UserMessage to a user-role message param.

    A part with cache_breakpoint carries prompt_cache_breakpoint on its wire part.
    """
    if isinstance(user_message.content, str):
        return {"role": "user", "content": user_message.content}
    parts: list[ChatCompletionContentPartParam] = []
    for part in user_message.content:
        if isinstance(part, TextPart):
            parts.append(_text_part_param(part))
        else:
            wire_image: ChatCompletionContentPartImageParam = {
                "type": "image_url",
                "image_url": {"url": _image_data_uri(part), "detail": "auto"},
            }
            if part.cache_breakpoint:
                wire_image["prompt_cache_breakpoint"] = {"mode": "explicit"}
            parts.append(wire_image)
    return {"role": "user", "content": parts}


def _tool_message(tool_message: ToolMessage) -> ChatCompletionToolMessageParam:
    """Convert one ToolMessage to a tool-role message param.

    The API has no is_error flag, so the error text in content is the only error signal.

    Raises:
        _NotSendableError: content holds an ImagePart, which the text-only tool message param
            cannot carry; dropping it silently would misstate the request.
    """
    if isinstance(tool_message.content, str):
        return {
            "role": "tool",
            "tool_call_id": tool_message.tool_call_id,
            "content": tool_message.content,
        }
    parts: list[ChatCompletionContentPartTextParam] = []
    for part in tool_message.content:
        if isinstance(part, ImagePart):
            raise _NotSendableError(
                "an ImagePart inside ToolMessage.content has no Chat Completions wire form: "
                "the tool message param's content is text-only"
            )
        parts.append(_text_part_param(part))
    return {
        "role": "tool",
        "tool_call_id": tool_message.tool_call_id,
        "content": parts,
    }


def _assistant_message_param(assistant_message: AssistantMessage) -> ChatCompletionMessageParam:
    """Convert one AssistantMessage to the one assistant-role message param this wire holds a turn as.

    The turn's texts join into content (the wire has no per-part boundary to keep), each ToolCall
    becomes one tool_calls entry keyed by its id, and each ReasoningTrace's raw dict merges into
    the param, so a reasoning_content that arrived replays byte-identical beside content.
    A trace another provider produced merges the same way and the API rejects or ignores its
    unknown keys, so a Sequence[Message] replayed through the wrong provider is the provider's to
    refuse, never silently dropped here.
    """
    param: dict[str, object] = {"role": "assistant"}
    for element in assistant_message.turn:
        if isinstance(element, ReasoningTrace):
            param.update(element.raw)
    texts = [
        element.text
        for element in assistant_message.turn
        if isinstance(element, TextPart) and element.text
    ]
    if texts:
        param["content"] = "".join(texts)
    tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = [
        {
            "id": element.id,
            "type": "function",
            "function": {"name": element.name, "arguments": element.args_json},
        }
        for element in assistant_message.turn
        if isinstance(element, ToolCall)
    ]
    if tool_calls:
        param["tool_calls"] = tool_calls
    # The trace's raw keys are deliberately wider than the assistant param TypedDict, under
    # "Honor user inputs faithfully"; the cast is that boundary.
    return cast("ChatCompletionMessageParam", param)


def _wire_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    """Convert messages to wire message params; the system prompt is not one.

    Propagates _tool_message's _NotSendableError for an ImagePart inside ToolMessage.content.
    """
    wire: list[ChatCompletionMessageParam] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            wire.append(_tool_message(message))
        elif isinstance(message, UserMessage):
            wire.append(_user_message(message))
        else:
            wire.append(_assistant_message_param(message))
    return wire


def _wire_tool_choice(tool_choice: ToolChoice) -> _WireToolChoice:
    """Convert the neutral tool choice."""
    if isinstance(tool_choice, SpecificToolChoice):
        return {"type": "function", "function": {"name": tool_choice.tool_name}}
    return tool_choice


def _wire_tools(tool_schemas: tuple[ToolSchema, ...]) -> list[ChatCompletionFunctionToolParam]:
    """Convert tool schemas to function tools.

    strict is left unset, leaving the provider's non-strict default in place,
    matching the schemas the ToolManager generates, which are not written to strict mode's restrictions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": dict(tool_schema.args_schema),
            },
        }
        for tool_schema in tool_schemas
    ]


def _as_chat_completion(raw: BaseModel) -> ChatCompletion:
    """Narrow a raw response to the SDK response this adapter produces.

    The BoundAdapter methods that read a response take BaseModel, because BoundLLM holds them and
    the neutral core imports no SDK. Every value reaching them came from this adapter's own stream,
    so another type is a defect in langchaint and not a provider behavior.

    Raises:
        TypeError: raw is not an openai ChatCompletion.
    """
    if not isinstance(raw, ChatCompletion):
        raise TypeError(f"expected an openai ChatCompletion, got {type(raw).__name__}")
    return raw


def _assistant_message_from(message: ChatCompletionMessage) -> AssistantMessage:
    """Build the langchaint assistant turn from the one message this wire holds a turn as.

    A non-empty reasoning_content extra becomes a ReasoningTrace first, carrying that one field as
    its raw for verbatim replay and the same string as its text;
    content becomes one TextPart, and a refusal becomes one TextPart after it, because the
    sentences the model wrote to refuse are the turn's text and a turn built without them replays
    as nothing;
    each function tool call becomes a ToolCall, its arguments passed through unparsed.
    A tool call whose type is "custom" is dropped: this adapter sends only function tools, so a
    custom call back is a provider defect no request of this adapter can elicit.
    """
    turn: list[TurnElement] = []
    reasoning_content = _reasoning_content_extra(message)
    if reasoning_content is not None:
        turn.append(
            ReasoningTrace(raw={"reasoning_content": reasoning_content}, text=reasoning_content)
        )
    if message.content:
        turn.append(TextPart(text=message.content))
    if message.refusal:
        turn.append(TextPart(text=message.refusal))
    turn.extend(
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            args_json=tool_call.function.arguments,
        )
        for tool_call in message.tool_calls or ()
        if tool_call.type == "function"
    )
    return AssistantMessage(turn=tuple(turn))


@dataclass(frozen=True, kw_only=True)
class _FinishedTurn:
    """A choice langchaint can read a turn from: its finish reason, its message, and its converted turn."""

    finish_reason: str
    message: ChatCompletionMessage
    assistant_message: AssistantMessage


def _finished_turn_or_unfinished(completion: ChatCompletion) -> _FinishedTurn | UnfinishedTurn:
    """Read the first choice as a finished turn, or report why no turn can be read.

    No choices at all is a response langchaint cannot read a turn from, and so is a choice whose
    finish_reason reads None at runtime, the in-progress state of a stream that never closed:
    both are UnfinishedTurn, carrying whatever partial turn the choice held.
    """
    if not completion.choices:
        return UnfinishedTurn(
            reason="openai returned no choices, so there is no turn to read",
            assistant_message=AssistantMessage(turn=()),
        )
    choice = completion.choices[0]
    assistant_message = _assistant_message_from(choice.message)
    finish_reason: str | None = choice.finish_reason
    if finish_reason is None:
        return UnfinishedTurn(
            reason="openai returned a choice with no finish_reason, "
            "which langchaint cannot call finished",
            assistant_message=assistant_message,
        )
    return _FinishedTurn(
        finish_reason=finish_reason, message=choice.message, assistant_message=assistant_message
    )


def _normalized_stop_reason(finished_turn: _FinishedTurn) -> StopReason:
    """Map the finish reason to the neutral vocabulary; the module docstring states each row.

    The refusal field is tested ahead of the rows, as the module docstring states.
    """
    if finished_turn.message.refusal:
        return "refusal"
    match finished_turn.finish_reason:
        case "stop":
            return "tool_use" if finished_turn.assistant_message.tool_calls else "end_turn"
        case "tool_calls":
            return "tool_use"
        case "length":
            return "max_tokens"
        case "content_filter":
            return "refusal"
        case _:
            return "other"


def cache_read_tokens_from_usage_openai(usage: CompletionUsage) -> int:
    """Read the cache-read counter openai reports: prompt_tokens_details.cached_tokens, 0 absent.

    The default cache_read_tokens_from_usage of OpenAIChatCompletionsAdapter; an OpenAI-compatible
    provider reporting cache reads through an extra usage field supplies its own reader, as
    langchaint.deepseek does.
    """
    details = usage.prompt_tokens_details
    if details is None:
        return 0
    return details.cached_tokens or 0


def _billing_from_chat_completion(
    completion: ChatCompletion,
    *,
    pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable],
    cache_read_tokens_from_usage: Callable[[CompletionUsage], int],
) -> Billing:
    """Price one response's raw counters at the table its priced tier selects.

    The whole response is the argument, not its usage: the tier that selects the rates is on the
    response and the counters are on the usage, and pricing one response's counters at another
    response's tier is the mistake worth making impossible.

    prompt_tokens is the all-inclusive input total, so the uncached counter is the remainder after
    subtracting the cache-read and cache-write counters.
    The SDK documents no relationship among the input counters, so the source is the provider's
    prompt-caching page, whose worked example reports 1920 cached tokens inside a 2006-token
    prompt total, read 2026-07-25:
    https://developers.openai.com/api/docs/guides/prompt-caching
    DeepSeek partitions the same total: its prompt_cache_hit_tokens and prompt_cache_miss_tokens
    sum to prompt_tokens (https://api-docs.deepseek.com/guides/kv_cache, read 2026-08-03).
    cache_read_tokens_from_usage reads the cache-read counter, because an OpenAI-compatible
    provider may report it through an extra usage field; the write counter is
    prompt_tokens_details.cache_write_tokens, 0 where the details object is absent.

    A response with no usage at all bills zero counters, at the priced tier's rates.

    Raises:
        pydantic.ValidationError: the counters leave input_tokens_cache_none negative, a response
            over-reporting its cache counters, so the priced Usage rejects it.
    """
    service_tier = _priced_tier(completion.service_tier)
    table = pricing.get(service_tier, _UNPRICED)
    usage = completion.usage
    if usage is None:
        return table.price(
            service_tier=service_tier,
            usage_raw=None,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            input_tokens_cache_none=0,
            output_tokens=0,
            output_tokens_reasoning=0,
        )
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    cache_read_tokens = cache_read_tokens_from_usage(usage)
    cache_write_tokens = (
        prompt_details.cache_write_tokens or 0 if prompt_details is not None else 0
    )
    return table.price(
        service_tier=service_tier,
        usage_raw=usage,
        input_tokens_cache_read=cache_read_tokens,
        input_tokens_cache_write=cache_write_tokens,
        input_tokens_cache_none=usage.prompt_tokens - cache_read_tokens - cache_write_tokens,
        output_tokens=usage.completion_tokens,
        output_tokens_reasoning=(
            completion_details.reasoning_tokens or 0 if completion_details is not None else 0
        ),
    )


def _wire_response_format(response_format: type[BaseModel]) -> ResponseFormatJSONSchema:
    """Build the response_format the structured binding sends for the caller's model.

    strict is False because the adapter validates the response text itself; the module docstring
    states what that keeps out of the SDK's frames.
    """
    json_schema: JSONSchema = {
        "name": response_format.__name__,
        "schema": response_format.model_json_schema(),
        "strict": False,
    }
    return {"type": "json_schema", "json_schema": json_schema}


class OpenAIChatCompletionsAdapter(Adapter):
    """Adapter over an AsyncOpenAI, AsyncBedrockOpenAI, or AsyncAzureOpenAI client.

    All three expose the same chat.completions.create method and with_options,
    so the adapter logic is identical across the first-party API, Bedrock, Azure, and every
    OpenAI-compatible endpoint a base AsyncOpenAI's base_url reaches.
    The client parameter is annotated AsyncOpenAI because the other two subclass it;
    provider_name_by_client_class is what tells them apart.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = (
        PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS
    )
    """The shared openai-SDK map, whose docstring states why AsyncOpenAI is absent from it."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable],
        provider_name: str,
        supports_prompt_cache_options: bool,
        cache_read_tokens_from_usage: Callable[
            [CompletionUsage], int
        ] = cache_read_tokens_from_usage_openai,
        service_tier: OpenAIServiceTier | None = None,
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        The stored client is a with_options(max_retries=0) copy: langchaint's retry loop owns all retrying,
        counts every request as an attempt, and feeds each failure to SharedBackoff through parse,
        so the SDK must never retry beneath it.

        provider_name says which provider the client reaches: "openai" for OpenAI's own endpoint,
        "aws.bedrock" for AsyncBedrockOpenAI, "azure.ai.openai" for AsyncAzureOpenAI, and the
        provider an AsyncOpenAI's base_url reaches for a compatible endpoint ("deepseek", "groq").
        The two platform classes are in provider_name_by_client_class, so a value contradicting
        either makes Adapter.__init__ raise; an AsyncOpenAI takes the provider_name its caller
        states, since its base_url decides what it reaches.

        supports_prompt_cache_options says whether the model accepts the prompt_cache_options
        request parameter, which openai documents as gpt-5.6-and-later (openai 2.45.0).
        That parameter is the only thing that carries automatic_prompt_caching False to the wire,
        so False here makes bind raise for a binding that declines caching, rather than cache
        anyway at whatever that model charges for it.
        It has no default because a wrong value fails either way: True on a model without the
        parameter risks a rejected request, and False on one with it refuses a binding the model
        would have accepted.
        It is a parameter here rather than a lookup on model because model is a str whose namespace
        this adapter cannot know: it serves every OpenAI-compatible endpoint.

        cache_read_tokens_from_usage reads the cache-read counter off a response's usage.
        The default reads what openai reports, prompt_tokens_details.cached_tokens; it is a
        parameter because an OpenAI-compatible provider may report the counter through an extra
        usage field, as DeepSeek does, and pricing those reads at the uncached rate would
        over-report the cost of every cached request.
        langchaint.deepseek passes cache_read_tokens_from_usage_deepseek.

        pricing holds one table per service tier this adapter can price, keyed by what a response
        reports, and a response served at a tier absent from it costs NaN. The "default" key is
        required because every response reporting no tier, and every response reporting "auto",
        prices there.
        service_tier is what the request asks for, None sending nothing. It cannot decide the price:
        the API documents the reported mode as possibly different from the requested one.

        Raises:
            ValueError: pricing has no "default" key, which would price every response reporting no
                tier, and every default-tier response, as NaN, with nothing said until the cost
                comes back unknown.
                Also raised by Adapter.__init__ when provider_name contradicts the client's class.
        """
        require_pricing_key(pricing, key=_DEFAULT_TIER, model=model)
        super().__init__(client=client, model=model, provider_name=provider_name)
        self.client = client.with_options(max_retries=0)
        self.pricing = pricing
        self.supports_prompt_cache_options = supports_prompt_cache_options
        self.cache_read_tokens_from_usage = cache_read_tokens_from_usage
        self.service_tier: OpenAIServiceTier | None = service_tier

    def _precompute_fields(self, binding: Binding) -> _ChatCompletionsPrecomputedFields:
        """Precompute the typed request fields the binding determines.

        Raises:
            ValueError: the binding declines automatic caching on a model built with
                supports_prompt_cache_options False, or its extra_body holds a key in
                _ADAPTER_POPULATED_WIRE_KEYS.
        """
        reject_extra_body_keys_the_adapter_populates(
            binding.extra_body, populated_keys=_ADAPTER_POPULATED_WIRE_KEYS
        )
        require_prompt_cache_options_support(
            model=self.model,
            automatic_prompt_caching=binding.automatic_prompt_caching,
            supports_prompt_cache_options=self.supports_prompt_cache_options,
        )
        messages_prefix: list[ChatCompletionMessageParam] = []
        if isinstance(binding.system_prompt, str):
            messages_prefix.append({"role": "system", "content": binding.system_prompt})
        elif binding.system_prompt is not None:
            messages_prefix.append({
                "role": "system",
                "content": [_text_part_param(part) for part in binding.system_prompt],
            })
        tools: list[ChatCompletionFunctionToolParam] | Omit = omit
        tool_choice: _WireToolChoice | Omit = omit
        parallel_tool_calls: bool | Omit = omit
        if binding.tool_schemas:
            tools = _wire_tools(binding.tool_schemas)
            tool_choice = _wire_tool_choice(binding.tool_choice)
            parallel_tool_calls = binding.parallel_tool_calls
        return _ChatCompletionsPrecomputedFields(
            model=self.model,
            messages_prefix=messages_prefix,
            max_completion_tokens=(
                binding.inference_params.max_completion_tokens
                if binding.inference_params.max_completion_tokens is not None
                else omit
            ),
            temperature=(
                binding.inference_params.temperature
                if binding.inference_params.temperature is not None
                else omit
            ),
            reasoning_effort=(
                binding.inference_params.reasoning_effort
                if binding.inference_params.reasoning_effort is not None
                else omit
            ),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_options=(
                omit if binding.automatic_prompt_caching else PromptCacheOptions(mode="explicit")
            ),
            service_tier=self.service_tier if self.service_tier is not None else omit,
            response_format=omit,
            extra_body=binding.extra_body,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output; pure conversion, no I/O.

        Propagates _precompute_fields' ValueError.
        """
        return _BoundChatCompletionsText(
            adapter=self, precomputed_fields=self._precompute_fields(binding)
        )

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format; pure conversion, no I/O.

        Propagates _precompute_fields' ValueError.
        """
        return _BoundChatCompletionsStructured(
            adapter=self,
            precomputed_fields=self._precompute_fields(binding),
            response_format=response_format,
        )

    failure_types: ClassVar[tuple[type[Exception], ...]] = OPENAI_FAILURE_TYPES
    """The shared tuple, whose docstring states why the bare APIStatusError covers every status."""

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Delegate to parse_openai, whose docstring names the table and the defaults."""
        return parse_openai(failure)

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Delegate to classify_openai, whose docstring names each row."""
        return classify_openai(error)

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        """Delegate to request_id_from_openai_error, which reads the SDK exception's header."""
        return request_id_from_openai_error(error)


def _snapshot_tool_call_id(state: ChatCompletionStreamState, index: int) -> str:
    """Read the id of the assembled tool call at index, which the state's events do not carry.

    A tool_calls.function.arguments.done event for index means the state assembled a call there,
    and its id came from that call's first fragment, the only fragment carrying one.
    """
    tool_calls = state.current_completion_snapshot.choices[0].message.tool_calls or ()
    return tool_calls[index].id


class _ChatCompletionsStream(AdapterStream):
    """One open Chat Completions stream, assembled by the SDK's ChatCompletionStreamState."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncStream[ChatCompletionChunk],
        pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable],
        cache_read_tokens_from_usage: Callable[[CompletionUsage], int],
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._cache_read_tokens_from_usage = cache_read_tokens_from_usage
        self._state = ChatCompletionStreamState()
        self._last_usage: CompletionUsage | None = None
        self._chunk_received = False

    async def _chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        """Iterate the SDK stream, rewrapping its mid-stream error raise for parse_openai.

        Yields:
            Every chunk the SDK stream yields.

        Raises:
            openai.APIStatusError: an SSE payload carried an error object, which the SDK raises as
                the bare openai.APIError; raised on the live response, so it carries the 200
                status and the open request's headers, with the SDK error's own body.
                The identity test selects exactly the bare class, the module docstring naming why,
                so every APIError subclass propagates untouched.
        """
        chunk_iterator = aiter(self._sdk_stream)
        while True:
            try:
                chunk = await anext(chunk_iterator)
            except StopAsyncIteration:
                return
            except openai.APIError as error:
                if type(error) is openai.APIError:
                    raise openai.APIStatusError(
                        error.message,
                        response=self._sdk_stream.response,
                        body=error.body,
                    ) from error
                raise
            yield chunk

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate chunks into answer text chunks, reasoning text deltas, and completed tool calls.

        Each chunk feeds the SDK's stream state, whose returned events carry the answer deltas and
        the completed tool calls; a reasoning_content delta is read off the chunk in hand, which
        the events do not carry, and yielded ahead of that chunk's events.
        The last non-None usage is tracked here for final() and billing_reported(), because the
        state's accumulation resets usage on every chunk that carries none.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.

        Raises:
            openai.APIStatusError: _chunks rewrapped the SDK's mid-stream error raise.
            StreamProtocolError: the SDK's state rejected a tool-call fragment index, or the
                stream ended without any choice reporting a finish_reason, so no turn closed.
        """
        finish_reason_seen = False
        async for chunk in self._chunks():
            self._chunk_received = True
            if chunk.usage is not None:
                self._last_usage = chunk.usage
            if any(choice.finish_reason for choice in chunk.choices):
                finish_reason_seen = True
            reasoning_delta = (
                _reasoning_content_extra(chunk.choices[0].delta) if chunk.choices else None
            )
            if reasoning_delta is not None:
                yield ReasoningDelta(text=reasoning_delta)
            try:
                events = self._state.handle_chunk(chunk)
            except IndexError as defect:
                raise StreamProtocolError(
                    "openai sent a tool-call fragment whose index does not follow its "
                    "predecessors, which the SDK's stream state cannot place"
                ) from defect
            for event in events:
                if event.type == "content.delta":
                    yield event.delta
                elif event.type == "tool_calls.function.arguments.done":
                    yield ToolCall(
                        id=_snapshot_tool_call_id(self._state, event.index),
                        name=event.name,
                        args_json=event.arguments,
                    )
        if not finish_reason_seen:
            raise StreamProtocolError("stream ended without a finish reason")

    def _snapshot_with_tracked_usage(self) -> ChatCompletion:
        """Return the assembled snapshot, patching in the tracked usage where a trailing chunk reset it."""
        snapshot = self._state.current_completion_snapshot
        if snapshot.usage is None and self._last_usage is not None:
            return snapshot.model_copy(update={"usage": self._last_usage})
        return snapshot

    @override
    async def final(self) -> ChatCompletion:
        """Return the response the SDK's state assembled.

        It is never re-validated into another model: the state constructs its snapshot leniently,
        and validating that snapshot against the SDK's own strict model could raise over a
        response whose partial output and billing the caller is owed.

        Raises:
            StreamProtocolError: items() was not exhausted first, so there is nothing assembled.
        """
        if not self._chunk_received:
            raise StreamProtocolError("final() requires items() to be exhausted first")
        return self._snapshot_with_tracked_usage()

    @override
    def billing_reported(self) -> Billing | None:
        """Return what the tracked usage bills at the snapshot's tier, or None before one arrives.

        The provider sends usage on the trailing chunk stream_options asks for, so a stream cut
        off early reports None and the caller records what it knows: nothing yet.
        """
        if self._last_usage is None:
            return None
        return _billing_from_chat_completion(
            self._snapshot_with_tracked_usage(),
            pricing=self._pricing,
            cache_read_tokens_from_usage=self._cache_read_tokens_from_usage,
        )

    @override
    def request_id(self) -> str | None:
        """Read the request-id header off the response the SDK stream is reading.

        AsyncStream sets its httpx response as the public response attribute in its constructor
        (openai 2.51.0), so this is readable from the moment the stream opens.
        """
        request_id: str | None = self._sdk_stream.response.headers.get("x-request-id")
        return request_id

    @override
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundChatCompletions[OutputT](BoundAdapter[OutputT], ABC):
    """What both Chat Completions bindings share: the request path, and what a response says about itself.

    A subclass sets _adapter and _precomputed_fields in its own __init__ and implements interpret.
    """

    _adapter: OpenAIChatCompletionsAdapter
    _precomputed_fields: _ChatCompletionsPrecomputedFields

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Price the response's counters at the table its priced tier selects.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
            pydantic.ValidationError: the response over-reports its cache counters, leaving the
                derived uncached-input counter negative.
        """
        return _billing_from_chat_completion(
            _as_chat_completion(raw),
            pricing=self._adapter.pricing,
            cache_read_tokens_from_usage=self._adapter.cache_read_tokens_from_usage,
        )

    @override
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Read the response's own id, the model it reports serving the request, and the request id.

        id and model are both required str fields on the SDK's ChatCompletion (openai 2.51.0),
        so neither is absent and neither needs converting.
        request_id_from_raw returns None on a snapshot the stream state assembled, which carries
        no request-id of its own; the stream handle's request_id() is where the header is read.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
        """
        completion = _as_chat_completion(raw)
        return ResponseIdentity(
            model_served=completion.model,
            response_id=completion.id,
            request_id=request_id_from_raw(completion),
        )

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Convert messages into the wire messages every attempt of this call sends."""
        try:
            wire_messages = _wire_messages(messages)
        except _NotSendableError as not_sendable:
            return InvalidRequest(reason=not_sendable.reason)
        return _ChatCompletionsRequestParams(
            precomputed=self._precomputed_fields,
            messages=[*self._precomputed_fields.messages_prefix, *wire_messages],
        )

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open one streaming create call and return the live stream; connection failures raise here.

        Raises:
            TypeError: request was built by another adapter.
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
        """
        params = narrowed_request(request, _ChatCompletionsRequestParams)
        precomputed = params.precomputed
        sdk_stream = await self._adapter.client.chat.completions.create(
            model=precomputed.model,
            messages=params.messages,
            max_completion_tokens=precomputed.max_completion_tokens,
            temperature=precomputed.temperature,
            reasoning_effort=precomputed.reasoning_effort,
            tools=precomputed.tools,
            tool_choice=precomputed.tool_choice,
            parallel_tool_calls=precomputed.parallel_tool_calls,
            prompt_cache_options=precomputed.prompt_cache_options,
            service_tier=precomputed.service_tier,
            response_format=precomputed.response_format,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=precomputed.extra_body,
        )
        return _ChatCompletionsStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            cache_read_tokens_from_usage=self._adapter.cache_read_tokens_from_usage,
        )


class _BoundChatCompletionsText(_BoundChatCompletions[str]):
    """Text-bound adapter: output is the concatenated text of the turn."""

    def __init__(
        self,
        *,
        adapter: OpenAIChatCompletionsAdapter,
        precomputed_fields: _ChatCompletionsPrecomputedFields,
    ) -> None:
        self._adapter = adapter
        self._precomputed_fields = precomputed_fields

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Read the turn, whose concatenated text is this binding's output.

        Every finished turn is a result: a refusal or a truncation still carries whatever text the
        model wrote, its condition named by the stop reason, and no schema stands between that text
        and the output.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
        """
        finished_turn = _finished_turn_or_unfinished(_as_chat_completion(raw))
        if isinstance(finished_turn, NoOutput):
            return finished_turn
        return AdapterResult(
            output=finished_turn.assistant_message.text,
            assistant_message=finished_turn.assistant_message,
            stop_reason=_normalized_stop_reason(finished_turn),
        )


class _BoundChatCompletionsStructured[ModelT: BaseModel](_BoundChatCompletions[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the turn's text."""

    def __init__(
        self,
        *,
        adapter: OpenAIChatCompletionsAdapter,
        precomputed_fields: _ChatCompletionsPrecomputedFields,
        response_format: type[ModelT],
    ) -> None:
        """Precompute the request's response_format parameter, the JSON-schema format this binding asks for.

        It replaces the binding's omitted response_format field, so every request this binding
        builds carries it and the two bindings send the same fields.
        """
        self._adapter = adapter
        self._response_format = response_format
        self._precomputed_fields = replace(
            precomputed_fields, response_format=_wire_response_format(response_format)
        )

    def _parsed_output(self, finished_turn: _FinishedTurn) -> ModelT | None | NoOutputOutcome:
        """Validate the turn's text into the instance, report a tool-call turn as None, or report why neither exists.

        Validating here rather than in the SDK is what puts the response and its text in scope when
        the text is rejected: the member returned for a rejection is one the retry loop can place
        against the attempt it already recorded, where a raise from inside the SDK is not.

        The text validated is message.content alone: a refusal is the model declining, so its
        sentences are never a candidate instance, and the turn carrying them reaches the caller on
        the Refusal member.
        The finish reason is read before the rejection, so text the token cap cut mid-object is
        reported as the truncation and not as a violation of the schema it was closing.

        None is the tool-call turn and nothing else: a turn _normalized_stop_reason calls
        tool_use, whose calls the turn kept, selects it. A refusal beside a call, a call the
        token cap cut mid-arguments, and a "tool_calls" finish leaving no call in the turn each
        fall to their own member below rather than dispatch as a completed turn.
        The instance wins where a turn carries both, because a turn that produced the instance
        answered the request whether or not it also called a tool.
        """
        validation_error: ValidationError | None = None
        text = finished_turn.message.content
        if text:
            try:
                return self._response_format.model_validate_json(text)
            except ValidationError as rejection:
                validation_error = rejection
        assistant_message = finished_turn.assistant_message
        if _normalized_stop_reason(finished_turn) == "tool_use" and assistant_message.tool_calls:
            return None
        if finished_turn.message.refusal or finished_turn.finish_reason == "content_filter":
            return Refusal(assistant_message=assistant_message)
        if finished_turn.finish_reason == "length":
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
            TypeError: raw is not an openai ChatCompletion.
        """
        finished_turn = _finished_turn_or_unfinished(_as_chat_completion(raw))
        if isinstance(finished_turn, NoOutput):
            return finished_turn
        output = self._parsed_output(finished_turn)
        if isinstance(output, NoOutput):
            return output
        return AdapterResult(
            output=output,
            assistant_message=finished_turn.assistant_message,
            stop_reason=_normalized_stop_reason(finished_turn),
        )
