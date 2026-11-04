"""Implement Chat Completions through the official openai SDK.

OpenAI-compatible providers serve Chat Completions through `AsyncOpenAI.base_url`.
`langchaint.deepseek` configures this adapter for DeepSeek.

Cache parameters match the Responses API values and follow `Binding.automatic_cache_breakpoints`.
"""

import base64
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast, override

import openai
from openai import AsyncOpenAI, AsyncStream, Omit, omit
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAllowedToolChoiceParam,
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
from openai.types.chat.chat_completion import Choice
from openai.types.chat.completion_create_params import PromptCacheOptions
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params import FunctionDefinition
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel

from langchaint.adapter import (
    AdapterStream,
    AllowedToolsChoice,
    Binding,
    BoundAdapter,
    FinishedResponse,
    ReasoningDelta,
    RejectedMessages,
    RequestParams,
    ResponseIdentity,
    SpecificToolChoice,
    StreamItem,
    ToolCallDelta,
    ToolChoice,
    _RejectedMessagesError,
    narrowed_request_params,
    reject_extra_body_keys_the_adapter_populates,
    request_params_json,
)
from langchaint.billing.pricing import ProviderBilling, billing_at_rates
from langchaint.common.exceptions import StreamProtocolError
from langchaint.common.messages import (
    AssistantMessage,
    AssistantPart,
    ContentPart,
    Message,
    ReasoningPart,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.openai.shared import (
    OpenAIPricingTable,
    OpenAIServiceTier,
    _image_data_uri,
    _OpenAIGenerationAdapterBase,
    _priced_tier,
    client_without_retries,
    require_prompt_cache_options_support,
)
from langchaint.tools import ToolSchema

type _WireToolChoice = (
    Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoiceParam
    | ChatCompletionAllowedToolChoiceParam
)
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""

_AUDIO_FORMAT_BY_MEDIA_TYPE: Mapping[str, Literal["wav", "mp3"]] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
}
"""AudioPart.media_type values mapped to input_audio.format."""


@dataclass(frozen=True, kw_only=True)
class _OpenAIChatCompletionsPrecomputedFields:
    """The typed request fields one binding precomputes.

    Fields set to the SDK's omit sentinel leave the provider default in place.
    Passing them as explicit keywords keeps the SDK's overload resolution intact.
    tool_choice and parallel_tool_calls are omitted without tools because the API rejects them otherwise.
    """

    model: str
    messages_prefix: list[ChatCompletionMessageParam]
    """Messages sent before each request's `Sequence[Message]`.

    A bound `system_prompt` becomes one system-role message.
    An absent `system_prompt` leaves this list empty.
    """

    max_completion_tokens: int | Omit
    temperature: float | Omit
    reasoning_effort: str | Omit
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


@dataclass(frozen=True, kw_only=True)
class _OpenAIChatCompletionsRequestParams(RequestParams):
    """One Chat Completions request: the binding's precomputed fields and one input's messages."""

    precomputed: _OpenAIChatCompletionsPrecomputedFields
    messages: list[ChatCompletionMessageParam]
    """What goes on the wire as messages: the binding's messages_prefix followed by the Sequence[Message]."""

    @override
    def as_json(self) -> str:
        """Render the request as a JSON object, dropping every field left to the provider's default."""
        return request_params_json(self, omitted_class=Omit)


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
    """Convert one TextPart to a text content part.

    A cache breakpoint uses `prompt_cache_breakpoint` without a client-side cap.
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


def _text_only_tool_message_error(part: ContentPart) -> _RejectedMessagesError:
    return _RejectedMessagesError(
        f"OpenAIChatCompletionsAdapter cannot send {type(part).__name__} inside "
        "ToolMessage.content: the tool message param's content is text-only"
    )


def _user_message(user_message: UserMessage) -> ChatCompletionUserMessageParam:
    """Convert one UserMessage to a user-role message param.

    A cache breakpoint uses `prompt_cache_breakpoint`.

    Raises:
        _RejectedMessagesError: AudioPart.media_type has no input_audio.format mapping.
    """
    if isinstance(user_message.content, str):
        return {"role": "user", "content": user_message.content}
    wire_parts: list[ChatCompletionContentPartParam] = []
    for part in user_message.content:
        match part.kind:
            case "text":
                wire_parts.append(_text_part_param(part))
            case "image":
                wire_parts.append(
                    _image_part_param(
                        _image_data_uri(part),
                        cache_breakpoint=part.cache_breakpoint,
                    )
                )
            case "image_url":
                wire_parts.append(
                    _image_part_param(
                        part.url,
                        cache_breakpoint=part.cache_breakpoint,
                    )
                )
            case "audio":
                audio_format = _AUDIO_FORMAT_BY_MEDIA_TYPE.get(part.media_type)
                if audio_format is None:
                    raise _RejectedMessagesError(
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
                wire_parts.append(wire_audio)
    return {"role": "user", "content": wire_parts}


def _tool_message(tool_message: ToolMessage) -> ChatCompletionToolMessageParam:
    """Convert one ToolMessage to a tool-role message param.

    The API has no is_error flag, so the error text in content is the only error signal.

    Raises:
        _RejectedMessagesError: content holds ImagePart, ImageUrlPart, or AudioPart.
    """
    if isinstance(tool_message.content, str):
        return {
            "role": "tool",
            "tool_call_id": tool_message.tool_call_id,
            "content": tool_message.content,
        }
    wire_parts: list[ChatCompletionContentPartTextParam] = []
    for part in tool_message.content:
        match part.kind:
            case "text":
                wire_parts.append(_text_part_param(part))
            case "image" | "image_url" | "audio":
                raise _text_only_tool_message_error(part)
    return {
        "role": "tool",
        "tool_call_id": tool_message.tool_call_id,
        "content": wire_parts,
    }


def _assistant_message_param(assistant_message: AssistantMessage) -> ChatCompletionMessageParam:
    """TextPart values concatenate into content.

    ToolCall values become function tool_calls entries in emission order.
    Each ReasoningPart.raw merges into the message fields.
    The text content and function tool_calls match openai 3.44.0's ChatCompletionAssistantMessageParam.
    DeepSeek requires replayed `reasoning_content` during a tool loop.
    Source: https://api-docs.deepseek.com/guides/thinking_mode, read 2026-08-03.

    Raises:
        _RejectedMessagesError: The assistant message contains a RawPart.
    """
    param: dict[str, object] = {"role": "assistant"}
    texts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCallUnionParam] = []
    for part in assistant_message.content_parts:
        match part.kind:
            case "reasoning":
                param.update(part.raw)
            case "text":
                texts.append(part.text)
            case "tool_call":
                tool_calls.append({
                    "id": part.id,
                    "type": "function",
                    "function": {"name": part.name, "arguments": part.args_json},
                })
            case "raw":
                raise _RejectedMessagesError(
                    "RawPart.raw has no Chat Completions wire form; "
                    "rebuild the assistant message without it"
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
        _RejectedMessagesError: ToolMessage contains ImagePart, ImageUrlPart, or AudioPart.
            AssistantMessage contains a RawPart.
    """
    wire: list[ChatCompletionMessageParam] = []
    for message in messages:
        match message.kind:
            case "tool":
                wire.append(_tool_message(message))
            case "user":
                wire.append(_user_message(message))
            case "assistant":
                wire.append(_assistant_message_param(message))
    return wire


def _wire_tool_choice(tool_choice: ToolChoice) -> _WireToolChoice:
    """Convert the neutral tool choice."""
    if isinstance(tool_choice, SpecificToolChoice):
        return {"type": "function", "function": {"name": tool_choice.tool_name}}
    if isinstance(tool_choice, AllowedToolsChoice):
        return {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": tool_choice.mode,
                "tools": [
                    {"type": "function", "function": {"name": tool_name}}
                    for tool_name in tool_choice.tool_names
                ],
            },
        }
    return tool_choice


def _wire_tools(tool_schemas: tuple[ToolSchema, ...]) -> list[ChatCompletionFunctionToolParam]:
    """Convert tool schemas to function tools.

    strict is left unset, which uses the provider's non-strict default.
    The non-strict default accepts the schemas that `ToolManager` generates.
    """
    return [
        {"type": "function", "function": _wire_function_definition(tool_schema)}
        for tool_schema in tool_schemas
    ]


def _wire_function_definition(tool_schema: ToolSchema) -> FunctionDefinition:
    """Convert one tool schema. `description=None` omits the description key."""
    function: FunctionDefinition = {
        "name": tool_schema.name,
        "parameters": dict(tool_schema.args_schema),
    }
    if tool_schema.description is not None:
        function["description"] = tool_schema.description
    return function


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
    """Preserve replayable provider values inside AssistantMessage.content.

    A non-empty reasoning_content becomes ReasoningPart first.
    message.content then becomes TextPart when present, including empty text.
    message.refusal then becomes TextPart when non-empty.
    Each message.tool_calls entry then becomes ToolCall in order.
    The request sends only function tools, so every entry is a function tool call.
    message.annotations and message.audio reach no AssistantPart.
    Read those fields from `Generation.raw`.
    """
    content: list[AssistantPart] = []
    reasoning_content = _reasoning_content_extra(message)
    if reasoning_content is not None:
        content.append(
            ReasoningPart(raw={"reasoning_content": reasoning_content}, text=reasoning_content)
        )
    if message.content is not None:
        content.append(TextPart(text=message.content))
    if message.refusal:
        content.append(TextPart(text=message.refusal))
    content.extend(
        ToolCall(id=x.id, name=x.function.name, args_json=x.function.arguments)
        for x in message.tool_calls or ()
        if x.type == "function"
    )
    return AssistantMessage(content=tuple(content))


def _normalized_stop_reason(choice: Choice, assistant_message: AssistantMessage) -> StopReason:
    """Map the finish reason, with a populated refusal field taking precedence."""
    if choice.message.refusal:
        return "refusal"
    match choice.finish_reason:
        case "stop":
            return "tool_call" if assistant_message.tool_calls else "stop"
        case "tool_calls":
            return "tool_call"
        case "length":
            return "max_completion_tokens"
        case "content_filter":
            return "refusal"
        case _:
            return "other"


def _cannot_continue(choice: Choice, stop_reason: StopReason) -> str | None:
    """Return the error text for a finish reason that maps to the stop reason `other`, or `None` for any other."""
    if stop_reason != "other":
        return None
    return (
        f"openai returned finish_reason {choice.finish_reason!r}, which langchaint cannot continue"
    )


def input_tokens_cache_read_from_usage_openai(usage: CompletionUsage) -> int:
    """Read the cache-read counter openai reports: prompt_tokens_details.cached_tokens, 0 absent.

    This is the default `OpenAIChatCompletionsAdapter.input_tokens_cache_read_from_usage`.
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
    input_tokens_cache_read_from_usage: Callable[[CompletionUsage], int],
) -> ProviderBilling:
    """Price response counters at the reported `service_tier`.

    `prompt_tokens` includes cached and cache-write tokens.
    Source: https://developers.openai.com/api/docs/guides/prompt-caching, read 2026-07-25.
    DeepSeek cache-hit and cache-miss counters also sum to `prompt_tokens`.
    Source: https://api-docs.deepseek.com/guides/kv_cache, read 2026-08-03.
    `input_tokens_cache_read_from_usage` supports provider-specific cache-read fields.
    Missing `prompt_tokens_details` means zero cache-write tokens.

    A response without usage bills zero counters.

    Raises:
        pydantic.ValidationError: Cache counters exceed `prompt_tokens`.
    """
    service_tier = _priced_tier(completion.service_tier)
    usage_raw = completion.usage
    counters = usage_raw or CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    input_tokens_total = counters.prompt_tokens
    rates = pricing.rates_for(service_tier=service_tier, input_tokens_total=input_tokens_total)
    provider_executed_tool_cost_in_usd = (
        float("nan") if any(choice.message.annotations for choice in completion.choices) else 0.0
    )
    prompt_details = counters.prompt_tokens_details
    completion_details = counters.completion_tokens_details
    input_tokens_cache_read = input_tokens_cache_read_from_usage(counters)
    input_tokens_cache_write = (
        prompt_details.cache_write_tokens or 0 if prompt_details is not None else 0
    )
    return ProviderBilling(
        billing=billing_at_rates(
            rates,
            service_tier=service_tier,
            input_tokens_cache_read=input_tokens_cache_read,
            input_tokens_cache_write=input_tokens_cache_write,
            input_tokens_cache_none=input_tokens_total
            - input_tokens_cache_read
            - input_tokens_cache_write,
            output_tokens=counters.completion_tokens,
            output_tokens_reasoning=(
                completion_details.reasoning_tokens or 0 if completion_details is not None else 0
            ),
            provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
        ),
        usage_raw=usage_raw,
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


class OpenAIChatCompletionsAdapter(_OpenAIGenerationAdapterBase):
    """Adapter over an AsyncOpenAI, AsyncBedrockOpenAI, or AsyncAzureOpenAI client.

    All three expose `chat.completions.create` and `with_options`.
    `AsyncOpenAI.base_url` also supports OpenAI-compatible providers.
    The client parameter is annotated AsyncOpenAI because the other two classes subclass it (openai 3.3.1).
    provider_name_by_client_class distinguishes the client classes.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        pricing: OpenAIPricingTable,
        provider_name: str,
        supports_prompt_cache_options: bool,
        input_tokens_cache_read_from_usage: Callable[
            [CompletionUsage], int
        ] = input_tokens_cache_read_from_usage_openai,
        service_tier: OpenAIServiceTier | None = None,
    ) -> None:
        """Store request and pricing configuration without sending a request.

        The stored client disables SDK retries.
        `provider_name` identifies the provider reached by the client.
        Bedrock and Azure clients require their fixed `provider_name` values.
        `AsyncOpenAI` uses the caller's value because `base_url` selects its provider.
        `supports_prompt_cache_options` states whether the provider API accepts `prompt_cache_options`.
        That request field carries `automatic_cache_breakpoints=False`.
        OpenAI 3.42.1 documents `prompt_cache_options` for gpt-5.6 and later models.
        `supports_prompt_cache_options` sets `Adapter.automatic_cache_breakpoints_default` to its inverse.
        A user, tool, or system-prompt part with `cache_breakpoint=True` sends `prompt_cache_breakpoint`.
        `supports_prompt_cache_options` does not change that.
        `input_tokens_cache_read_from_usage` reads provider-specific cache-read counters.
        A response without usage passes a new `CompletionUsage` with zero counters and no details.
        The reader must return 0 for it.
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
        self.input_tokens_cache_read_from_usage: Callable[[CompletionUsage], int] = (
            input_tokens_cache_read_from_usage
        )
        self.service_tier: OpenAIServiceTier | None = service_tier

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return stored request configuration outside `Binding`."""
        return {
            "service_tier": self.service_tier,
            "supports_prompt_cache_options": self.supports_prompt_cache_options,
        }

    def _precompute_fields(self, binding: Binding) -> _OpenAIChatCompletionsPrecomputedFields:
        """Precompute the typed request fields the binding determines.

        Raises:
            ValueError: `automatic_cache_breakpoints=False` requires `prompt_cache_options` support.
            ValueError: `extra_body` contains a key in `_ADAPTER_POPULATED_WIRE_KEYS` or `web_search_options`.
            ValueError: `extra_body` contains `functions`, which declares tools outside `tools`.
            ValueError: `extra_body` sets `n` to a value other than 1 or `None`, because the adapter reads one choice.
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
        if binding.extra_body is not None and "functions" in binding.extra_body:
            raise ValueError(
                "extra_body key 'functions' declares tools outside `tools`. "
                "Bind tools through `tools`, because result types follow the bound tools."
            )
        if binding.extra_body is not None and binding.extra_body.get("n") not in (None, 1):
            raise ValueError(
                "extra_body key 'n' must be 1, None, or absent, "
                "because OpenAIChatCompletionsAdapter reads one choice per response. "
                "Pass the input once per choice."
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
        return _OpenAIChatCompletionsPrecomputedFields(
            model=self.model,
            messages_prefix=messages_prefix,
            max_completion_tokens=(
                binding.max_completion_tokens
                if binding.max_completion_tokens is not None
                else omit
            ),
            temperature=(binding.temperature if binding.temperature is not None else omit),
            reasoning_effort=(
                binding.reasoning_level if binding.reasoning_level is not None else omit
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
    def bind_text(self, binding: Binding) -> BoundAdapter:
        """Bind for plain-text output without I/O.

        Raises:
            ValueError: `binding` contains unsupported values.
        """
        return _BoundOpenAIChatCompletions(
            adapter=self, precomputed_fields=self._precompute_fields(binding)
        )

    @override
    def bind_structured(self, binding: Binding, response_format: type[BaseModel]) -> BoundAdapter:
        """Bind for structured output in response_format without I/O.

        Raises:
            ValueError: `binding` contains unsupported values.
            pydantic.PydanticInvalidForJsonSchema: `response_format` cannot produce a JSON schema.
            pydantic.PydanticUserError: `response_format` is not fully defined.
        """
        return _BoundOpenAIChatCompletions(
            adapter=self,
            precomputed_fields=replace(
                self._precompute_fields(binding),
                response_format=_wire_response_format(response_format),
            ),
        )


def _snapshot_tool_call_id(state: ChatCompletionStreamState, index: int) -> str:
    """Read the id of the assembled tool call at index, which the state's events do not carry.

    A tool_calls.function.arguments.done event for index means the state assembled a call there.
    The snapshot is built leniently (openai 3.44.0 construct_type).
    The declared str therefore reads None until the fragment carrying the id arrives.
    """
    tool_calls = state.current_completion_snapshot.choices[0].message.tool_calls or ()
    return tool_calls[index].id


class _OpenAIChatCompletionsStream(AdapterStream):
    """One open Chat Completions stream, assembled by the SDK's ChatCompletionStreamState."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncStream[ChatCompletionChunk],
        pricing: OpenAIPricingTable,
        input_tokens_cache_read_from_usage: Callable[[CompletionUsage], int],
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._input_tokens_cache_read_from_usage = input_tokens_cache_read_from_usage
        self._state = ChatCompletionStreamState()
        self._chunk_received = False
        self._content_filtered = False
        """Whether an Azure annotation chunk reported `finish_reason="content_filter"`."""

    async def _chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        """Iterate the SDK stream, rewrapping its mid-stream error raise for openai_request_failure.

        Yields:
            Every chunk whose `object` is `chat.completion.chunk`.
            Azure also sends content-filter annotation chunks with an empty `object`, `id`, and `model`.
            Its first chunk carries prompt-filter results, and its asynchronous filter sends choices without `delta`.
            `ChatCompletionStreamState` would take an empty id and model from the first.
            It raises `AttributeError` on the second.
            The SDK's own stream helper skips them too (openai 3.44.0 `_is_valid_chat_completion_chunk_weak`).
            In asynchronous filter mode, a block arrives only as an annotation chunk's `content_filter` finish reason.
            `_content_filtered` records that block (Azure content streaming docs, blocked-by-filters sample).

        Raises:
            openai.APIStatusError: An SSE payload contains an error object.
                The error carries status 200, request headers, and the SDK error body.
            openai.APIError: The SDK raises a specific `APIError` subclass.
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
            if chunk.object == "chat.completion.chunk":
                yield chunk
            elif any(choice.finish_reason == "content_filter" for choice in chunk.choices or ()):
                self._content_filtered = True

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate chunks into StreamItem values.

        SDK events carry answer deltas and completed tool calls.
        Each chunk supplies `reasoning_content` before its SDK events.
        Argument fragments wait for their call id, then precede the next emitted fragment.

        Yields:
            Stream items. SDK events that langchaint does not model are dropped.

        Raises:
            openai.APIStatusError: _chunks rewrapped the SDK's mid-stream error raise.
            StreamProtocolError: The SDK rejects a tool-call index or the stream ends without `finish_reason`.
        """
        finish_reason_seen = False
        pending_args: dict[int, str] = {}
        """Argument fragments held while their call's snapshot id is still None, keyed by index."""
        async for chunk in self._chunks():
            if any(choice.finish_reason for choice in chunk.choices):
                finish_reason_seen = True
            try:
                events = self._state.handle_chunk(chunk)
            except IndexError as defect:
                raise StreamProtocolError(
                    "openai sent a tool-call fragment whose index does not follow its "
                    "predecessors, which the SDK's stream state cannot place"
                ) from defect
            self._chunk_received = True
            reasoning_delta = (
                _reasoning_content_extra(chunk.choices[0].delta) if chunk.choices else None
            )
            if reasoning_delta is not None:
                yield ReasoningDelta(text=reasoning_delta)
            for event in events:
                if event.type == "content.delta":
                    yield event.delta
                elif event.type == "tool_calls.function.arguments.delta" and event.arguments_delta:
                    call_id = _snapshot_tool_call_id(self._state, event.index)
                    partial_args_json = pending_args.pop(event.index, "") + event.arguments_delta
                    if call_id is None:
                        pending_args[event.index] = partial_args_json
                    else:
                        yield ToolCallDelta(
                            id=call_id, name=event.name, partial_args_json=partial_args_json
                        )
                elif event.type == "tool_calls.function.arguments.done":
                    yield ToolCall(
                        id=_snapshot_tool_call_id(self._state, event.index),
                        name=event.name,
                        args_json=event.arguments,
                    )
        filtered = (
            self._content_filtered
            and self._chunk_received
            and bool(self._state.current_completion_snapshot.choices)
        )
        if not finish_reason_seen and not filtered:
            raise StreamProtocolError("stream ended without a finish reason")

    @override
    async def final(self) -> ChatCompletion:
        """Return the response the SDK's state assembled.

        The adapter preserves the lenient SDK snapshot without revalidation.
        Revalidation could reject partial output and billing.
        An Azure content-filter block sets every choice's `finish_reason` to `"content_filter"`.
        `get_final_completion` raises on length and content-filter responses (openai 3.44.0).

        Raises:
            StreamProtocolError: items() was not exhausted first, so there is nothing assembled.
        """
        if not self._chunk_received:
            raise StreamProtocolError("final() requires items() to be exhausted first")
        snapshot = self._state.current_completion_snapshot
        if not self._content_filtered:
            return snapshot
        return snapshot.model_copy(
            update={
                "choices": [
                    choice.model_copy(update={"finish_reason": "content_filter"})
                    for choice in snapshot.choices
                ]
            }
        )

    @override
    def provider_billing(self) -> ProviderBilling | None:
        """Return what the snapshot's usage bills at its tier, or None before usage arrives.

        `stream_options` requests usage on the last chunk.
        An earlier cutoff returns `None`.
        """
        if not self._chunk_received:
            return None
        snapshot = self._state.current_completion_snapshot
        if snapshot.usage is None:
            return None
        return _billing_from_chat_completion(
            snapshot,
            pricing=self._pricing,
            input_tokens_cache_read_from_usage=self._input_tokens_cache_read_from_usage,
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
        await self._sdk_stream.close()


class _BoundOpenAIChatCompletions(BoundAdapter):
    """A Chat Completions adapter bound to a frozen prefix."""

    def __init__(
        self,
        *,
        adapter: OpenAIChatCompletionsAdapter,
        precomputed_fields: _OpenAIChatCompletionsPrecomputedFields,
    ) -> None:
        self._adapter = adapter
        self._precomputed_fields = precomputed_fields

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Price counters using the reported `service_tier`.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
            pydantic.ValidationError: Cache counters exceed total input tokens.
        """
        return _billing_from_chat_completion(
            _as_chat_completion(raw),
            pricing=self._adapter.pricing,
            input_tokens_cache_read_from_usage=self._adapter.input_tokens_cache_read_from_usage,
        )

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Combine the response's id and model with request_id.

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
    def build_request_params(
        self, messages: Sequence[Message]
    ) -> RequestParams | RejectedMessages:
        """Convert messages into the wire messages every request for one input sends."""
        try:
            wire_messages = _wire_messages(messages)
        except _RejectedMessagesError as rejected:
            return RejectedMessages(error_text=str(rejected))
        return _OpenAIChatCompletionsRequestParams(
            precomputed=self._precomputed_fields,
            messages=[*self._precomputed_fields.messages_prefix, *wire_messages],
        )

    @override
    async def open_stream(self, request_params: RequestParams) -> AdapterStream:
        """Open one streaming create call and return the live stream.

        Raises:
            TypeError: request was built by another adapter.
            Exception: The SDK fails to open the stream.
        """
        params = narrowed_request_params(request_params, _OpenAIChatCompletionsRequestParams)
        precomputed = params.precomputed
        sdk_stream = await self._adapter.client.chat.completions.create(
            model=precomputed.model,
            messages=params.messages,
            max_completion_tokens=precomputed.max_completion_tokens,
            temperature=precomputed.temperature,
            # cast: `reasoning_level` deliberately exceeds the OpenAI SDK effort literal.
            reasoning_effort=cast(
                "Literal['high', 'low', 'max', 'medium', 'minimal', 'none', 'xhigh'] | Omit",
                precomputed.reasoning_effort,
            ),
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
        return _OpenAIChatCompletionsStream(
            sdk_stream=sdk_stream,
            pricing=self._adapter.pricing,
            input_tokens_cache_read_from_usage=self._adapter.input_tokens_cache_read_from_usage,
        )

    @override
    def interpret(self, raw: BaseModel) -> FinishedResponse:
        """Read the first choice, which a stream that ended with a finish reason always has.

        Raises:
            TypeError: raw is not an openai ChatCompletion.
        """
        choice = _as_chat_completion(raw).choices[0]
        assistant_message = _assistant_message_from(choice.message)
        stop_reason = _normalized_stop_reason(choice, assistant_message)
        return FinishedResponse(
            assistant_message=assistant_message,
            stop_reason=stop_reason,
            cannot_continue=_cannot_continue(choice, stop_reason),
        )
