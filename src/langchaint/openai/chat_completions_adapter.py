"""Implement Chat Completions through the official openai SDK.

Streaming behavior was verified against openai 2.51.0.

OpenAI-compatible providers serve Chat Completions through `AsyncOpenAI.base_url`.
`langchaint.deepseek` configures this adapter for DeepSeek.

`create(stream=True)` sends the request and returns `AsyncStream[ChatCompletionChunk]`.
`chat.completions.stream()` rejects any input tool whose `strict` is not `True`.
`ToolManager` produces non-strict schemas, so this adapter uses `create(stream=True)`.
`ChatCompletionStreamState` assembles chunks for `final()`.
Without `input_tools` or `response_format`, `handle_chunk` preserves length and content-filter responses.
`get_final_completion` raises on those responses, so this adapter does not call it.
The structured binding sends `"strict": False` and validates response text after assembly.

`handle_chunk` returns assembled events.
`content.delta` carries answer text.
`tool_calls.function.arguments.done` carries the name, arguments, and index.
The adapter reads the call id from `current_completion_snapshot` at that index.
A sparse or out-of-order index makes `handle_chunk` raise `IndexError`.
`ChatCompletionStreamState` replaces `usage` with every chunk and takes other fields from the first chunk.
The adapter preserves the last reported `usage` when a trailing chunk resets it.
`stream_options={"include_usage": True}` requests the final usage chunk.

The SDK preserves extra fields through `model_dump` and stream assembly.
String extras concatenate across deltas.
This preserves `reasoning_content` on `ChatCompletionMessage` and `ChoiceDelta`.
A mid-stream SSE `error` makes the iterator raise `openai.APIError`.
Only `openai._streaming` constructs that exact type in openai 2.51.0.
`items()` converts that error to `APIStatusError` on the live response for `parse_openai`.
Subclasses of `openai.APIError` propagate unchanged.
An unclosed stream may produce `Choice.finish_reason=None` despite its required `Literal` type.
The required `choices` list may be empty.

`CompletionUsage` requires `prompt_tokens`, `completion_tokens`, and `total_tokens`.
Its optional details contain cached, cache-write, and reasoning counters.
`ChatCompletionToolMessageParam.content` accepts only text.
Cache parameters match the Responses API values and follow `Binding.automatic_cache_breakpoints`.

Content mappings were verified against openai 2.53.0.
- `ImagePart` becomes a data URL in `image_url.url`.
- `ImageUrlPart.url` becomes `image_url.url` unchanged.
- `AudioPart` accepts `audio/wav` and `audio/mpeg` inside `UserMessage`.
- `ImagePart`, `ImageUrlPart`, and `AudioPart` inside `ToolMessage` return `InvalidRequest`.
- `ChatCompletionMessage.audio` remains available through `Response.raw`.

Request and response mappings:
- A string `system_prompt` becomes the first system-role message.
- A parts `system_prompt` becomes one system-role message.
- `AssistantMessage` becomes one assistant message parameter.
- Its texts fill `content`, each `ToolCall` fills `tool_calls`, and `ReasoningPart.raw` supplies extra fields.
- DeepSeek requires replayed `reasoning_content` during a tool loop.
- Source: https://api-docs.deepseek.com/guides/thinking_mode, read 2026-08-03.
- `message.refusal` becomes `TextPart` and sets `stop_reason="refusal"`, including with `finish_reason="stop"`.
- `finish_reason="stop"` maps by tool calls to `"end_turn"` or `"tool_use"`.
- `"tool_calls"`, `"length"`, and `"content_filter"` map to `"tool_use"`, `"max_tokens"`, and `"refusal"`.
- `"function_call"` and unknown values map to `"other"`.
- Streaming yields answer text, `ReasoningDelta`, `ToolCallDelta`, and one complete `ToolCall`.
- The adapter delays fragments until their call id exists and prepends them to the next emitted fragment.
- OpenAI-compatible providers may omit ids on early fragments.
- `reasoning_content` forms one string without separators.
- `final()` supplies usage, cost, and stop reason.
"""

import base64
from abc import ABC
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import nan
from typing import ClassVar, Literal, cast, override

import openai
from openai import AsyncOpenAI, AsyncStream, Omit, omit
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
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
    reject_extra_body_keys_the_adapter_populates,
    request_json,
)
from langchaint.call import ResponseIdentity
from langchaint.exceptions import StreamProtocolError
from langchaint.inference_params import ReasoningEffort
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
from langchaint.openai.shared import (
    OPENAI_FAILURE_TYPES,
    PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS,
    OpenAIPricingTable,
    OpenAIServiceTier,
    _image_data_uri,
    _priced_tier,
    classify_openai,
    client_without_retries,
    parse_openai,
    request_id_from_openai_error,
    require_prompt_cache_options_support,
)
from langchaint.pricing import Billing
from langchaint.shared_backoff import Verdict
from langchaint.tools import ToolSchema

type _WireToolChoice = Literal["none", "auto", "required"] | ChatCompletionNamedToolChoiceParam
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""

_AUDIO_FORMAT_BY_MEDIA_TYPE: Mapping[str, Literal["wav", "mp3"]] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
}
"""AudioPart.media_type values mapped to input_audio.format."""


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
    Per-part converters raise it when a `Sequence[Message]` is unsendable.
    """

    def __init__(self, reason: str) -> None:
        """Store what cannot be sent; it becomes the InvalidRequest reason."""
        super().__init__(reason)
        self.reason = reason


def _reasoning_content_extra(model: BaseModel) -> str | None:
    """Read a non-empty reasoning_content string off a model's extra fields, None where there is none.

    The SDK models omit `reasoning_content`.
    DeepSeek returns it through `model_extra` on messages and stream deltas.
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


def _image_part_param(
    image_url: str, *, cache_breakpoint: bool
) -> ChatCompletionContentPartImageParam:
    wire_image: ChatCompletionContentPartImageParam = {
        "type": "image_url",
        "image_url": {"url": image_url, "detail": "auto"},
    }
    if cache_breakpoint:
        wire_image["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return wire_image


def _text_only_tool_message_error(part: ContentPart) -> _NotSendableError:
    return _NotSendableError(
        f"OpenAIChatCompletionsAdapter cannot send {type(part).__name__} inside "
        "ToolMessage.content: the tool message param's content is text-only"
    )


def _user_message(user_message: UserMessage) -> ChatCompletionUserMessageParam:
    """Convert one UserMessage to a user-role message param.

    A part with cache_breakpoint carries prompt_cache_breakpoint on its wire part.

    Raises:
        _NotSendableError: AudioPart.media_type has no input_audio.format mapping.
    """
    if isinstance(user_message.content, str):
        return {"role": "user", "content": user_message.content}
    parts: list[ChatCompletionContentPartParam] = []
    for part in user_message.content:
        match part.kind:
            case "text":
                parts.append(_text_part_param(part))
            case "image":
                parts.append(
                    _image_part_param(
                        _image_data_uri(part), cache_breakpoint=part.cache_breakpoint
                    )
                )
            case "image_url":
                parts.append(_image_part_param(part.url, cache_breakpoint=part.cache_breakpoint))
            case "audio":
                audio_format = _AUDIO_FORMAT_BY_MEDIA_TYPE.get(part.media_type)
                if audio_format is None:
                    raise _NotSendableError(
                        "OpenAIChatCompletionsAdapter cannot send AudioPart inside "
                        f"UserMessage.content: AudioPart.media_type must be 'audio/wav' or "
                        f"'audio/mpeg', not {part.media_type!r}"
                    )
                wire_audio: ChatCompletionContentPartInputAudioParam = {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(part.data).decode("ascii"),
                        "format": audio_format,
                    },
                }
                if part.cache_breakpoint:
                    wire_audio["prompt_cache_breakpoint"] = {"mode": "explicit"}
                parts.append(wire_audio)
    return {"role": "user", "content": parts}


def _tool_message(tool_message: ToolMessage) -> ChatCompletionToolMessageParam:
    """Convert one ToolMessage to a tool-role message param.

    The API has no is_error flag, so the error text in content is the only error signal.

    Raises:
        _NotSendableError: content holds ImagePart, ImageUrlPart, or AudioPart.
    """
    if isinstance(tool_message.content, str):
        return {
            "role": "tool",
            "tool_call_id": tool_message.tool_call_id,
            "content": tool_message.content,
        }
    parts: list[ChatCompletionContentPartTextParam] = []
    for part in tool_message.content:
        match part.kind:
            case "text":
                parts.append(_text_part_param(part))
            case "image":
                raise _text_only_tool_message_error(part)
            case "image_url":
                raise _text_only_tool_message_error(part)
            case "audio":
                raise _text_only_tool_message_error(part)
    return {
        "role": "tool",
        "tool_call_id": tool_message.tool_call_id,
        "content": parts,
    }


def _assistant_message_param(assistant_message: AssistantMessage) -> ChatCompletionMessageParam:
    """TextPart values concatenate into content.

    ToolCall values become function tool_calls entries.
    RawPart.raw with type "custom" becomes a custom tool_calls entry unchanged.
    RawPart.raw holding only function_call merges into the message unchanged.
    These tool_calls entries retain their emission order.
    Each ReasoningPart.raw merges into the message fields.
    This matches openai 2.51.0's ChatCompletionAssistantMessageParam variants.

    Raises:
        _NotSendableError: RawPart.raw matches no supported shape, or repeats function_call.
    """
    param: dict[str, object] = {"role": "assistant"}
    texts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCallUnionParam] = []
    for part in assistant_message.turn:
        if isinstance(part, ReasoningPart):
            param.update(part.raw)
        elif isinstance(part, TextPart):
            if part.text:
                texts.append(part.text)
        elif isinstance(part, ToolCall):
            tool_calls.append({
                "id": part.id,
                "type": "function",
                "function": {"name": part.name, "arguments": part.args_json},
            })
        elif part.raw.get("type") == "custom":
            # cast: a deliberately-opaque value re-enters the typed API that serialized it.
            tool_calls.append(cast("ChatCompletionMessageToolCallUnionParam", part.raw))
        elif len(part.raw) == 1 and "function_call" in part.raw:
            if "function_call" in param:
                raise _NotSendableError(
                    "an assistant turn contains more than one function_call, but Chat Completions "
                    "has one function_call field"
                )
            param.update(part.raw)
        else:
            raise _NotSendableError(
                "RawPart.raw has no Chat Completions wire form: only custom tool_calls and "
                "function_call can hold it; rebuild the turn without it"
            )
    if texts:
        param["content"] = "".join(texts)
    if tool_calls:
        param["tool_calls"] = tool_calls
    # cast: `ReasoningPart.raw` deliberately exceeds the assistant parameter `TypedDict`.
    return cast("ChatCompletionMessageParam", param)


def _wire_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    """Keep the bound system prompt outside messages.

    Raises:
        _NotSendableError: ToolMessage contains ImagePart, ImageUrlPart, or AudioPart.
            RawPart.raw may also have no wire form.
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

    `BoundAdapter` accepts `BaseModel` because the neutral core imports no SDK.
    This adapter's stream produces every valid value.

    Raises:
        TypeError: raw is not an openai ChatCompletion.
    """
    if not isinstance(raw, ChatCompletion):
        raise TypeError(f"expected an openai ChatCompletion, got {type(raw).__name__}")
    return raw


def _assistant_message_from(message: ChatCompletionMessage) -> AssistantMessage:
    """Preserve replayable provider values inside AssistantMessage.turn.

    A non-empty reasoning_content becomes ReasoningPart first.
    message.content then becomes TextPart when non-empty.
    message.refusal then becomes TextPart when non-empty.
    message.function_call then becomes RawPart when present.
    Each function message.tool_calls entry becomes ToolCall.
    Each custom message.tool_calls entry becomes RawPart.
    The message.tool_calls entry order is preserved.
    openai 2.51.0 defines both variants.
    message.annotations and message.audio reach no TurnPart.
    Read those fields from Response.raw.
    """
    turn: list[TurnPart] = []
    reasoning_content = _reasoning_content_extra(message)
    if reasoning_content is not None:
        turn.append(
            ReasoningPart(raw={"reasoning_content": reasoning_content}, text=reasoning_content)
        )
    if message.content:
        turn.append(TextPart(text=message.content))
    if message.refusal:
        turn.append(TextPart(text=message.refusal))
    if message.function_call is not None:
        turn.append(
            RawPart(
                raw=message.model_dump(mode="python", include={"function_call"}, exclude_none=True)
            )
        )
    for tool_call in message.tool_calls or ():
        if tool_call.type == "function":
            turn.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args_json=tool_call.function.arguments,
                )
            )
        else:
            turn.append(RawPart(raw=tool_call.model_dump(mode="python", exclude_none=True)))
    return AssistantMessage(turn=tuple(turn))


@dataclass(frozen=True, kw_only=True)
class _FinishedTurn:
    """A choice langchaint can read a turn from: its finish reason, its message, and its converted turn."""

    finish_reason: str
    message: ChatCompletionMessage
    assistant_message: AssistantMessage


def _finished_turn_or_unfinished(completion: ChatCompletion) -> _FinishedTurn | UnfinishedTurn:
    """Read the first choice as a finished turn, or report why no turn can be read.

    Missing choices and `finish_reason=None` return `UnfinishedTurn`.
    The result carries any partial turn.
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


def _adapter_result[OutputT](
    finished_turn: _FinishedTurn, output: OutputT
) -> AdapterResult[OutputT]:
    return AdapterResult(
        output=output,
        assistant_message=finished_turn.assistant_message,
        stop_reason=_normalized_stop_reason(finished_turn),
    )


def cache_read_tokens_from_usage_openai(usage: CompletionUsage) -> int:
    """Read the cache-read counter openai reports: prompt_tokens_details.cached_tokens, 0 absent.

    This is the default `OpenAIChatCompletionsAdapter.cache_read_tokens_from_usage`.
    Providers with extra usage fields supply another reader.
    """
    details = usage.prompt_tokens_details
    if details is None:
        return 0
    return details.cached_tokens or 0


def _billing_from_chat_completion(
    completion: ChatCompletion,
    *,
    pricing: OpenAIPricingTable,
    cache_read_tokens_from_usage: Callable[[CompletionUsage], int],
) -> Billing:
    """Price response counters at the reported `service_tier`.

    `prompt_tokens` includes cached and cache-write tokens.
    Source: https://developers.openai.com/api/docs/guides/prompt-caching, read 2026-07-25.
    DeepSeek cache-hit and cache-miss counters also sum to `prompt_tokens`.
    Source: https://api-docs.deepseek.com/guides/kv_cache, read 2026-08-03.
    `cache_read_tokens_from_usage` supports provider-specific cache-read fields.
    Missing `prompt_tokens_details` means zero cache-write tokens.

    A response without usage bills zero counters.

    Raises:
        pydantic.ValidationError: Cache counters exceed `prompt_tokens`.
    """
    service_tier = _priced_tier(completion.service_tier)
    usage = completion.usage
    input_tokens_total = 0 if usage is None else usage.prompt_tokens
    rates = pricing.rates_for(
        service_tier=completion.service_tier,
        input_tokens_total=input_tokens_total,
        regional_processing=False,
    )
    provider_executed_tool_cost_in_usd = (
        nan if any(choice.message.annotations for choice in completion.choices) else 0.0
    )
    if usage is None:
        return rates.price(
            service_tier=service_tier,
            usage_raw=None,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            input_tokens_cache_none=0,
            output_tokens=0,
            output_tokens_reasoning=0,
            provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
        )
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    cache_read_tokens = cache_read_tokens_from_usage(usage)
    cache_write_tokens = (
        prompt_details.cache_write_tokens or 0 if prompt_details is not None else 0
    )
    return rates.price(
        service_tier=service_tier,
        usage_raw=usage,
        input_tokens_cache_read=cache_read_tokens,
        input_tokens_cache_write=cache_write_tokens,
        input_tokens_cache_none=usage.prompt_tokens - cache_read_tokens - cache_write_tokens,
        output_tokens=usage.completion_tokens,
        output_tokens_reasoning=(
            completion_details.reasoning_tokens or 0 if completion_details is not None else 0
        ),
        provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
    )


def _wire_response_format(response_format: type[BaseModel]) -> ResponseFormatJSONSchema:
    """Build the response_format the structured binding sends for the caller's model.

    `strict=False` leaves validation to the adapter after response assembly.
    """
    json_schema: JSONSchema = {
        "name": response_format.__name__,
        "schema": response_format.model_json_schema(),
        "strict": False,
    }
    return {"type": "json_schema", "json_schema": json_schema}


class OpenAIChatCompletionsAdapter(Adapter):
    """Adapter over an AsyncOpenAI, AsyncBedrockOpenAI, or AsyncAzureOpenAI client.

    All three expose `chat.completions.create` and `with_options`.
    `AsyncOpenAI.base_url` also supports OpenAI-compatible providers.
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
        pricing: OpenAIPricingTable,
        provider_name: str,
        supports_prompt_cache_options: bool,
        cache_read_tokens_from_usage: Callable[
            [CompletionUsage], int
        ] = cache_read_tokens_from_usage_openai,
        service_tier: OpenAIServiceTier | None = None,
    ) -> None:
        """Store request and pricing configuration without sending a request.

        The stored client disables SDK retries.
        `provider_name` identifies the provider reached by the client.
        Bedrock and Azure clients require their fixed `provider_name` values.
        `AsyncOpenAI` uses the caller's value because `base_url` selects its provider.
        `supports_prompt_cache_options` identifies gpt-5.6-and-later support documented by openai 2.45.0.
        It sets `Adapter.automatic_cache_breakpoints_default` to the inverse value.
        `cache_read_tokens_from_usage` reads provider-specific cache-read counters.
        The default reads `prompt_tokens_details.cached_tokens`.
        `pricing` supplies rates and modifiers.
        `service_tier` requests a tier, while the reported tier selects pricing.

        Raises:
            ValueError: `provider_name` contradicts a Bedrock or Azure client class.
        """
        super().__init__(
            client=client,
            model=model,
            provider_name=provider_name,
            automatic_cache_breakpoints_default=not supports_prompt_cache_options,
        )
        self.client: AsyncOpenAI = client_without_retries(client)
        self.pricing: OpenAIPricingTable = pricing
        self.supports_prompt_cache_options: bool = supports_prompt_cache_options
        self.cache_read_tokens_from_usage: Callable[[CompletionUsage], int] = (
            cache_read_tokens_from_usage
        )
        self.service_tier: OpenAIServiceTier | None = service_tier

    def _precompute_fields(self, binding: Binding) -> _ChatCompletionsPrecomputedFields:
        """Precompute the typed request fields the binding determines.

        Raises:
            ValueError: `automatic_cache_breakpoints=False` requires `prompt_cache_options` support.
            ValueError: `extra_body` contains a key in `_ADAPTER_POPULATED_WIRE_KEYS` or `web_search_options`.
            ValueError: `provider_executed_tools` is nonempty.
        """
        if binding.provider_executed_tools:
            raise ValueError(
                "OpenAIChatCompletionsAdapter cannot send provider_executed_tools. "
                "Use OpenAIResponsesAdapter for provider-executed tools."
            )
        if binding.extra_body is not None and "web_search_options" in binding.extra_body:
            raise ValueError(
                "OpenAIChatCompletionsAdapter cannot price web_search_options. "
                "Use OpenAIResponsesAdapter for web search."
            )
        reject_extra_body_keys_the_adapter_populates(
            binding.extra_body, populated_keys=_ADAPTER_POPULATED_WIRE_KEYS
        )
        require_prompt_cache_options_support(
            model=self.model,
            automatic_cache_breakpoints=binding.automatic_cache_breakpoints,
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
                omit
                if binding.automatic_cache_breakpoints
                else PromptCacheOptions(mode="explicit")
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

    A tool_calls.function.arguments.done event for index means the state assembled a call there.
    The snapshot is built leniently (openai 2.51.0 construct_type).
    The declared str therefore reads None until the fragment carrying the id arrives.
    """
    tool_calls = state.current_completion_snapshot.choices[0].message.tool_calls or ()
    return tool_calls[index].id


class _ChatCompletionsStream(AdapterStream):
    """One open Chat Completions stream, assembled by the SDK's ChatCompletionStreamState."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncStream[ChatCompletionChunk],
        pricing: OpenAIPricingTable,
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
            openai.APIStatusError: An SSE payload contains an error object.
                The error carries status 200, request headers, and the SDK error body.
                `APIError` subclasses propagate unchanged.
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
        """Translate chunks into StreamItem values.

        SDK events carry answer deltas and completed tool calls.
        Each chunk supplies `reasoning_content` before its SDK events.
        The stream tracks the last non-`None` usage because SDK accumulation may reset it.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.

        Raises:
            openai.APIStatusError: _chunks rewrapped the SDK's mid-stream error raise.
            StreamProtocolError: The SDK rejects a tool-call index or the stream ends without `finish_reason`.
        """
        finish_reason_seen = False
        pending_args: dict[int, str] = {}
        """Argument fragments held while their call's snapshot id is still None, keyed by index."""
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
                elif event.type == "tool_calls.function.arguments.delta" and event.arguments_delta:
                    call_id = _snapshot_tool_call_id(self._state, event.index)
                    held_args = pending_args.pop(event.index, "") + event.arguments_delta
                    if call_id is None:
                        pending_args[event.index] = held_args
                    else:
                        yield ToolCallDelta(
                            id=call_id, name=event.name, partial_args_json=held_args
                        )
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

        The adapter preserves the lenient SDK snapshot without revalidation.
        Revalidation could reject partial output and billing.

        Raises:
            StreamProtocolError: items() was not exhausted first, so there is nothing assembled.
        """
        if not self._chunk_received:
            raise StreamProtocolError("final() requires items() to be exhausted first")
        return self._snapshot_with_tracked_usage()

    @override
    def billing_reported(self) -> Billing | None:
        """Return what the tracked usage bills at the snapshot's tier, or None before one arrives.

        `stream_options` requests usage on the trailing chunk.
        An earlier cutoff returns `None`.
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

        openai 3.0.0 exposes `AsyncStream.response` as `httpx2.Response`.
        The header is readable when the stream opens.
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
        """Price counters using the reported `service_tier`.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
            pydantic.ValidationError: Cache counters exceed total input tokens.
        """
        return _billing_from_chat_completion(
            _as_chat_completion(raw),
            pricing=self._adapter.pricing,
            cache_read_tokens_from_usage=self._adapter.cache_read_tokens_from_usage,
        )

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Combine the response's id and model with request_id.

        id and model are both required str fields on the SDK's ChatCompletion (openai 2.51.0),
        so neither is absent and neither needs converting.
        request_id comes from AdapterStream.request_id().

        Raises:
            TypeError: raw is not an openai ChatCompletion.
        """
        completion = _as_chat_completion(raw)
        return ResponseIdentity(
            model_served=completion.model,
            response_id=completion.id,
            request_id=request_id,
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

        Refusals and truncations still return their text with a matching stop reason.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
        """
        finished_turn = _finished_turn_or_unfinished(_as_chat_completion(raw))
        if isinstance(finished_turn, NoOutput):
            return finished_turn
        return _adapter_result(finished_turn, finished_turn.assistant_message.text)


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

        It replaces the omitted binding field in every request.
        """
        self._adapter = adapter
        self._response_format = response_format
        self._precomputed_fields = replace(
            precomputed_fields, response_format=_wire_response_format(response_format)
        )

    def _parsed_outcome(self, finished_turn: _FinishedTurn) -> ResponseOutcome[ModelT | None]:
        """Validate message.content after the response enters langchaint.

        Therefore, rejected content and billing remain available.
        message.refusal never enters validation.
        A valid instance returns first.
        Otherwise, message.refusal or finish_reason "content_filter" returns Refusal.
        finish_reason "length" returns MaxCompletionTokensExceeded.
        A tool_use stop with non-empty AssistantMessage.tool_calls returns None.
        Remaining invalid content returns SchemaViolation.
        Every remaining response returns EmptyTurn.
        """
        validation_error: ValidationError | None = None
        text = finished_turn.message.content
        if text:
            try:
                output = self._response_format.model_validate_json(text)
                return _adapter_result(finished_turn, output)
            except ValidationError as rejection:
                validation_error = rejection
        assistant_message = finished_turn.assistant_message
        if _normalized_stop_reason(finished_turn) == "tool_use" and assistant_message.tool_calls:
            return _adapter_result(finished_turn, None)
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
        return self._parsed_outcome(finished_turn)
