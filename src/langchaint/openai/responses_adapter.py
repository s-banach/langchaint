"""Implement the OpenAI Responses API through the official openai SDK.

The following SDK facts were verified against openai 2.45.0.
A structured binding builds `text.format` with `type_to_text_format_param` and validates response text itself.
SDK parsing may reject text before the terminal response exposes its billing.
`type_to_text_format_param` is a private SDK function that may move between SDK versions.
`responses.stream` yields typed events and assembles the response.
Usage and status arrive with `response.completed`, `response.incomplete`, or `response.failed`.
`get_final_response()` raises unless the terminal event is `response.completed`.

`prompt_cache_options` supports gpt-5.6 and later.
`{"mode": "explicit"}` without breakpoints disables caching.
Implicit mode writes through the latest three breakpoints.
Explicit mode writes through the latest four breakpoints.
Older breakpoints remain available for matching.
`ttl` accepts only `"30m"`.
`automatic_cache_breakpoints=False` sends explicit mode and requires `prompt_cache_options` support.
`automatic_cache_breakpoints=True` sends no `prompt_cache_options` and preserves implicit caching.
Every marked part sends `prompt_cache_breakpoint: {"mode": "explicit"}`.
The adapter sends every breakpoint and lets the API enforce its write limits.
Marked parts re-enable caching under `automatic_cache_breakpoints=False`.

The API stores responses by default.
The adapter sends `store=False` because `GenerationInput` contains the complete state.
Every request includes `reasoning.encrypted_content` so reasoning can replay under `store=False`.
`reasoning.summary` accepts `"auto"`, `"concise"`, or `"detailed"`.
The deprecated `generate_summary` accepts the same values, so the adapter sends only `reasoning.summary`.
The adapter omits unset `reasoning.effort` and `reasoning.summary` keys.

Content mappings were verified against openai 2.53.0.
- `ImagePart` becomes a data URL in `image_url`.
- `ImageUrlPart.url` becomes `image_url` unchanged.
- `AudioPart` returns `InvalidRequest` inside `UserMessage` and `ToolMessage`.
- Web search and file search produce distinct output item types.

Request and response mappings:
- A string `system_prompt` becomes `instructions`.
- A parts `system_prompt` becomes the first developer-role input message because only parts support breakpoints.
- `AssistantMessage` replays `TurnPart` values in emission order under `store=False`.
- `ReasoningPart` and `RawPart` replay their stored items unchanged.
- `ToolCall` becomes `function_call`, and adjacent `TextPart` values become one assistant message.
- `ToolMessage` becomes `function_call_output` keyed by `tool_call_id`.
- The API has no `is_error` field, so `ToolMessage.content` carries the error signal.
- `ResponseOutputRefusal` becomes `TextPart` and maps to `stop_reason="refusal"`.
- Anthropic 0.120.0 represents refusals as text with the same `stop_reason`.
- A `function_call` output item maps to `stop_reason="tool_use"`.
- Status `"completed"` maps to `"end_turn"`.
- Status `"incomplete"` maps `"max_output_tokens"` to `"max_tokens"` and `"content_filter"` to `"refusal"`.
- Other outcomes map to `"other"`.
- Status `"failed"` returns `_provider_failure` with billing, even when emitted text validates.
- Streaming yields answer text, `ReasoningDelta`, `ToolCallDelta`, and one `ToolCall` per completed item.
- `final()` supplies usage, cost, and stop reason.
"""

from abc import ABC
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from math import nan
from typing import Any, ClassVar, Literal, cast, override

import openai
from openai import AsyncOpenAI, Omit, omit
from openai.lib._parsing._responses import type_to_text_format_param
from openai.lib.streaming.responses import AsyncResponseStream
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseErrorEvent,
    ResponseFunctionCallOutputItemListParam,
    ResponseFunctionToolCallParam,
    ResponseIncludable,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseReasoningItem,
    ResponseTextConfigParam,
    ToolChoiceAllowedParam,
    ToolChoiceFunctionParam,
    ToolParam,
)
from openai.types.responses import (
    Response as OpenAIResponse,
)
from openai.types.responses.response_create_params import PromptCacheOptions
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseInputItemParam,
)
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, ValidationError

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
    NoOutputOutcome,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
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
    validated_provider_executed_tool_types,
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
    _DEFAULT_TIER,
    _DISPOSITION_BY_ERROR_CODE,
    OPENAI_FAILURE_TYPES,
    PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS,
    OpenAIPricingTable,
    OpenAIResponsesServiceTier,
    OpenAIServiceTier,
    _image_data_uri,
    _priced_tier,
    classify_openai,
    client_without_retries,
    parse_openai,
    request_id_from_openai_error,
    require_prompt_cache_options_support,
)
from langchaint.pricing import (
    ProviderBilling,
    invocation_cost_in_usd,
    require_finite_nonnegative_rate,
)
from langchaint.shared_backoff import Verdict
from langchaint.tools import ToolSchema

type _WireToolChoice = (
    Literal["none", "auto", "required"] | ToolChoiceFunctionParam | ToolChoiceAllowedParam
)
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""

type ReasoningSummary = Literal["auto", "concise", "detailed"]
"""How much readable text to ask the API for, the values reasoning.summary takes."""

_SUPPORTED_PROVIDER_EXECUTED_TOOL_TYPES = frozenset({
    "file_search",
    "web_search",
    "web_search_2025_08_26",
    "web_search_preview",
    "web_search_preview_2025_03_11",
})
_WEB_SEARCH_TOOL_TYPES = _SUPPORTED_PROVIDER_EXECUTED_TOOL_TYPES - {"file_search"}
_UNPRICEABLE_OUTPUT_TYPES = frozenset({
    "code_interpreter_call",
    "computer_call",
    "image_generation_call",
    "shell_call",
})


def _wire_reasoning(
    effort: ReasoningEffort | None, summary: ReasoningSummary | None
) -> Reasoning | Omit:
    """Assemble the reasoning object from the keys that are set, omitting it when neither is.

    `Reasoning` permits explicit `None`, which differs from omitting a key.
    The adapter never sends `context`, `mode`, or deprecated `generate_summary`.
    """
    reasoning: Reasoning = {}
    if effort is not None:
        reasoning["effort"] = effort
    if summary is not None:
        reasoning["summary"] = summary
    return reasoning or omit


@dataclass(frozen=True, kw_only=True)
class _OpenAIPrecomputedFields:
    """The typed request fields one binding precomputes.

    The SDK `omit` sentinel preserves provider defaults.
    Explicit keywords preserve SDK overload resolution.
    tool_choice and parallel_tool_calls are omitted without tools because the API rejects them otherwise.
    `include` always contains `"reasoning.encrypted_content"` for later replay.
    """

    model: str
    instructions: str | None
    input_prefix: list[ResponseInputItemParam]

    max_output_tokens: int | Omit
    temperature: float | Omit
    reasoning: Reasoning | Omit
    tools: list[ToolParam] | Omit
    tool_choice: _WireToolChoice | Omit
    parallel_tool_calls: bool | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    service_tier: OpenAIResponsesServiceTier | Omit
    include: list[ResponseIncludable]
    text: ResponseTextConfigParam | Omit
    """The structured binding's JSON-schema format, omitted by the text binding, which asks for none."""

    extra_body: Mapping[str, object] | None
    charged_provider_tools: bool


_ADAPTER_POPULATED_WIRE_KEYS = frozenset({
    "model",
    "instructions",
    "max_output_tokens",
    "temperature",
    "reasoning",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "prompt_cache_options",
    "service_tier",
    "include",
    "text",
    "store",
    "input",
    "stream",
})


@dataclass(frozen=True, kw_only=True)
class _OpenAIRequestParams(RequestParams):
    """One responses request: the binding's precomputed fields and this call's converted input."""

    precomputed: _OpenAIPrecomputedFields
    input: list[ResponseInputItemParam]
    """What goes on the wire as input: the binding's input_prefix followed by the Sequence[Message]."""

    @override
    def as_json(self) -> str:
        """Render the request as a JSON object, dropping every field left to the provider's default."""
        return request_json(self, omitted_class=Omit)


class _NotSendableError(Exception):
    """A Sequence[Message] this adapter cannot send."""

    def __init__(self, reason: str) -> None:
        """Store the InvalidRequest.reason."""
        super().__init__(reason)
        self.reason = reason


def _user_image_param(image_url: str, *, cache_breakpoint: bool) -> ResponseInputImageParam:
    wire_image: ResponseInputImageParam = {
        "type": "input_image",
        "image_url": image_url,
        "detail": "auto",
    }
    if cache_breakpoint:
        wire_image["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return wire_image


def _tool_image_param(image_url: str, *, cache_breakpoint: bool) -> ResponseInputImageContentParam:
    output_image: ResponseInputImageContentParam = {
        "type": "input_image",
        "image_url": image_url,
        "detail": "auto",
    }
    if cache_breakpoint:
        output_image["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return output_image


def _user_item(user_message: UserMessage) -> EasyInputMessageParam:
    """Convert one UserMessage to a user message item.

    Marked parts send `prompt_cache_breakpoint`.

    Raises:
        _NotSendableError: content holds AudioPart.
    """
    if isinstance(user_message.content, str):
        return {"role": "user", "content": user_message.content}
    parts: ResponseInputMessageContentListParam = []
    for part in user_message.content:
        match part.kind:
            case "text":
                wire_text: ResponseInputTextParam = {"type": "input_text", "text": part.text}
                if part.cache_breakpoint:
                    wire_text["prompt_cache_breakpoint"] = {"mode": "explicit"}
                parts.append(wire_text)
            case "image":
                parts.append(
                    _user_image_param(
                        _image_data_uri(part), cache_breakpoint=part.cache_breakpoint
                    )
                )
            case "image_url":
                parts.append(_user_image_param(part.url, cache_breakpoint=part.cache_breakpoint))
            case "audio":
                raise _NotSendableError(
                    "OpenAIResponsesAdapter cannot send AudioPart inside UserMessage.content: "
                    "ResponseInputContentParam has no audio variant"
                )
    return {"role": "user", "content": parts}


def _function_call_output(
    content: str | tuple[ContentPart, ...],
) -> str | ResponseFunctionCallOutputItemListParam:
    """Convert one ToolMessage's content to the function_call_output output field.

    The output field accepts str or ResponseFunctionCallOutputItemListParam.
    ResponseFunctionCallOutputItemListParam accepts text and image content.
    A bare string passes through.
    A sequence of parts becomes that structured content list.
    The image content param differs from the user-message input_image param.
    This function builds the output image dict and shares only the data URI encoding.
    A part with cache_breakpoint carries prompt_cache_breakpoint on its wire part.
    The latest-N server rule in `_user_item` also applies here.

    Raises:
        _NotSendableError: content holds AudioPart.
    """
    if isinstance(content, str):
        return content
    output_content: ResponseFunctionCallOutputItemListParam = []
    for part in content:
        match part.kind:
            case "text":
                output_text: ResponseInputTextContentParam = {
                    "type": "input_text",
                    "text": part.text,
                }
                if part.cache_breakpoint:
                    output_text["prompt_cache_breakpoint"] = {"mode": "explicit"}
                output_content.append(output_text)
            case "image":
                output_content.append(
                    _tool_image_param(
                        _image_data_uri(part), cache_breakpoint=part.cache_breakpoint
                    )
                )
            case "image_url":
                output_content.append(
                    _tool_image_param(part.url, cache_breakpoint=part.cache_breakpoint)
                )
            case "audio":
                raise _NotSendableError(
                    "OpenAIResponsesAdapter cannot send AudioPart inside ToolMessage.content: "
                    "ResponseFunctionCallOutputItemListParam has no audio variant"
                )
    return output_content


def _replayed_item(raw: Mapping[str, object]) -> ResponseInputItemParam:
    """Copy one stored SDK dump into the input item it came from, unread and unchanged.

    This adapter's SDK dump already matches the wire parameter.
    Another provider's dump passes through unchanged for the API to validate.
    The copy prevents mutable request state from changing the stored payload.
    """
    # cast: a deliberately-opaque value re-enters the typed API whose own serialization produced it.
    return cast("ResponseInputItemParam", dict(raw))


def _assistant_items(assistant_message: AssistantMessage) -> list[ResponseInputItemParam]:
    """Convert one AssistantMessage to its input items in turn order.

    The API requires the original item order for replay under store=False.
    Adjacent `TextPart` values become one assistant message item.
    Each `ToolCall` becomes `function_call` keyed by `call_id`.
    `ReasoningPart.raw` and `RawPart.raw` replay unchanged by their `type` keys.
    The API validates another provider's unknown `type` key.
    """
    items: list[ResponseInputItemParam] = []
    pending_texts: list[str] = []

    def flush_text_run() -> None:
        if pending_texts:
            items.append({"role": "assistant", "content": "".join(pending_texts)})
            pending_texts.clear()

    for part in assistant_message.turn:
        if isinstance(part, TextPart):
            if part.text:
                pending_texts.append(part.text)
        elif isinstance(part, ToolCall):
            flush_text_run()
            function_call_item: ResponseFunctionToolCallParam = {
                "type": "function_call",
                "call_id": part.id,
                "name": part.name,
                "arguments": part.args_json,
            }
            items.append(function_call_item)
        else:
            flush_text_run()
            items.append(_replayed_item(part.raw))
    flush_text_run()
    return items


def _wire_input(messages: Sequence[Message]) -> list[ResponseInputItemParam]:
    """Convert messages to input items.

    The system prompt is separate.

    Raises:
        _NotSendableError: A ContentPart has no Responses wire form.
    """
    wire: list[ResponseInputItemParam] = []
    for message in messages:
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
    """Convert the neutral tool choice."""
    if isinstance(tool_choice, SpecificToolChoice):
        return {"type": "function", "name": tool_choice.tool_name}
    if isinstance(tool_choice, AllowedToolsChoice):
        return {
            "type": "allowed_tools",
            "mode": tool_choice.mode,
            "tools": [
                {"type": "function", "name": tool_name} for tool_name in tool_choice.tool_names
            ],
        }
    return tool_choice


def _wire_tools(
    tool_schemas: tuple[ToolSchema, ...],
    provider_executed_tools: tuple[Mapping[str, object], ...],
) -> list[ToolParam]:
    """Convert every bound tool to one ordered wire list.

    strict is a required key of FunctionToolParam.
    None leaves the provider's non-strict default in place.
    The ToolManager schemas do not satisfy strict mode's restrictions.
    """
    tools: list[ToolParam] = [
        {
            "type": "function",
            "name": tool_schema.name,
            "description": tool_schema.description,
            "parameters": dict(tool_schema.args_schema),
            "strict": None,
        }
        for tool_schema in tool_schemas
    ]
    # cast: the neutral Mapping type exceeds the SDK TypedDict union.
    # The adapter validated each mapping's type discriminator.
    tools.extend(cast("ToolParam", tool) for tool in provider_executed_tools)
    return tools


def _provider_failure(
    response: OpenAIResponse, *, assistant_message: AssistantMessage
) -> ProviderFailedTransiently | ProviderFailedTerminally:
    """Return the failure variant selected by `response.error.code`.

    Both variants carry emitted fragments in `assistant_message`.
    `reason` preserves `response.error.message` verbatim.
    Missing errors and unknown codes produce terminal failures.
    `rate_limit_exceeded` produces a rate-limit transient failure without a server-stated wait.
    """
    error = response.error
    if error is None:
        return ProviderFailedTerminally(
            reason="openai reported status 'failed' and no error object",
            assistant_message=assistant_message,
        )
    if _DISPOSITION_BY_ERROR_CODE.get(error.code) == "transient":
        return ProviderFailedTransiently(
            reason=error.message,
            is_rate_limit=error.code == "rate_limit_exceeded",
            assistant_message=assistant_message,
        )
    return ProviderFailedTerminally(reason=error.message, assistant_message=assistant_message)


def _has_refusal(response: OpenAIResponse) -> bool:
    return any(
        content_part.type == "refusal"
        for item in response.output
        if item.type == "message"
        for content_part in item.content
    )


def _as_response(raw: BaseModel) -> OpenAIResponse:
    """Narrow a raw response to the SDK response this adapter produces.

    `BoundAdapter` accepts `BaseModel` because the neutral core imports no SDK.
    This adapter's stream produces every valid value.

    Raises:
        TypeError: raw is not an openai Response.
    """
    if not isinstance(raw, OpenAIResponse):
        raise TypeError(f"expected an openai Response, got {type(raw).__name__}")
    return raw


def _first_output_text(response: OpenAIResponse) -> str | None:
    """Return the text of the turn's first output_text content part, None when it holds none.

    Structured output validation uses this part.
    SDK parsing validates every `output_text` part and returns the first instance.
    """
    for item in response.output:
        if item.type == "message":
            for content_part in item.content:
                if content_part.type == "output_text":
                    return content_part.text
    return None


def _normalized_stop_reason(response: OpenAIResponse) -> StopReason:
    """Derive the stop reason.

    The API reports no finish reason field.

    An incomplete `content_filter` result maps to `refusal` without retry.
    """
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
        case "incomplete" if (
            response.incomplete_details is not None
            and response.incomplete_details.reason == "content_filter"
        ):
            return "refusal"
        case _:
            return "other"


def _reasoning_text(item: ResponseReasoningItem) -> str | None:
    """Join non-empty reasoning parts with `REASONING_PART_SEPARATOR`.

    Reasoning content takes precedence over its summary in openai 2.48.0.
    Empty content and summary parts produce `None`.
    """
    summary = REASONING_PART_SEPARATOR.join(part.text for part in item.summary if part.text)
    content = REASONING_PART_SEPARATOR.join(part.text for part in item.content or () if part.text)
    return content or summary or None


def _assistant_message_from(response: OpenAIResponse) -> AssistantMessage:
    """Build the langchaint assistant turn from the output items, item order preserved.

    Reasoning items become replayable `ReasoningPart` values with readable text.
    Message content becomes ordered `TextPart` values, including refusals.
    Other items become replayable `RawPart` values.
    """
    turn: list[TurnPart] = []
    for item in response.output:
        if item.type == "reasoning":
            turn.append(
                ReasoningPart(
                    raw=item.model_dump(mode="json", exclude_none=True),
                    text=_reasoning_text(item),
                )
            )
        elif item.type == "function_call":
            turn.append(ToolCall(id=item.call_id, name=item.name, args_json=item.arguments))
        elif item.type == "message":
            for content_part in item.content:
                if content_part.type == "output_text":
                    turn.append(TextPart(text=content_part.text))
                elif content_part.type == "refusal":
                    turn.append(TextPart(text=content_part.refusal))
        else:
            turn.append(RawPart(raw=item.model_dump(mode="json", exclude_none=True)))
    return AssistantMessage(turn=tuple(turn))


def _billing_from_response(
    response: OpenAIResponse,
    pricing: OpenAIPricingTable,
    *,
    regional_processing: bool = False,
) -> ProviderBilling:
    """Price response counters at the reported `service_tier`.

    `input_tokens` includes cached and cache-write tokens.
    Source: https://developers.openai.com/api/docs/guides/prompt-caching, read 2026-07-25.
    `output_tokens_details.reasoning_tokens` is required.
    Only web-search `search` actions incur invocation costs.
    Source: https://developers.openai.com/api/docs/guides/tools-web-search, read 2026-08-09.
    Each `file_search_call` incurs one file-search fee.
    Source: https://developers.openai.com/api/docs/pricing.

    A response without usage bills zero counters.

    Raises:
        pydantic.ValidationError: Cache counters exceed `input_tokens`.
    """
    service_tier = _priced_tier(response.service_tier)
    usage = response.usage
    input_tokens_total = 0 if usage is None else usage.input_tokens
    rates = pricing.rates_for(
        service_tier=response.service_tier,
        input_tokens_total=input_tokens_total,
        regional_processing=regional_processing,
    )
    web_search_invocations = sum(
        1
        for item in response.output
        if item.type == "web_search_call" and item.action.type == "search"
    )
    file_search_invocations = sum(1 for item in response.output if item.type == "file_search_call")
    provider_executed_tool_cost_in_usd = invocation_cost_in_usd(
        web_search_invocations,
        usd_per_invocation=pricing.web_search_usd_per_invocation,
    ) + invocation_cost_in_usd(
        file_search_invocations,
        usd_per_invocation=pricing.file_search_usd_per_invocation,
    )
    if any(item.type in _UNPRICEABLE_OUTPUT_TYPES for item in response.output):
        provider_executed_tool_cost_in_usd = nan
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
    details = usage.input_tokens_details
    return rates.price(
        service_tier=service_tier,
        usage_raw=usage,
        input_tokens_cache_read=details.cached_tokens,
        input_tokens_cache_write=details.cache_write_tokens,
        input_tokens_cache_none=(
            usage.input_tokens - details.cached_tokens - details.cache_write_tokens
        ),
        output_tokens=usage.output_tokens,
        output_tokens_reasoning=usage.output_tokens_details.reasoning_tokens,
        provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
    )


def _adapter_result[OutputT](
    response: OpenAIResponse, output: OutputT, assistant_message: AssistantMessage
) -> AdapterResult[OutputT]:
    """Normalize one completed request around already-extracted output and its turn."""
    return AdapterResult(
        output=output,
        assistant_message=assistant_message,
        stop_reason=_normalized_stop_reason(response),
    )


class OpenAIResponsesAdapter(Adapter):
    """Adapter over an AsyncOpenAI, AsyncBedrockOpenAI, or AsyncAzureOpenAI client.

    All three expose the same responses.stream method and with_options.
    The adapter logic is shared across the first-party API, Bedrock, and Azure.
    The client parameter is annotated AsyncOpenAI because the other two classes subclass it (openai 3.3.1).
    provider_name_by_client_class distinguishes the client classes.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = (
        PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS
    )
    """The shared openai-SDK map, whose docstring states why AsyncOpenAI is absent from it."""

    def __init__(  # noqa: PLR0913 (each request and billing parameter remains explicit)
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        pricing: OpenAIPricingTable,
        provider_name: str,
        regional_processing: bool = False,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIResponsesServiceTier | None = None,
    ) -> None:
        """Store request and pricing configuration without sending a request.

        The stored client disables SDK retries.
        `reasoning_summary` requests readable reasoning text.
        `None` preserves the provider default, and a requested summary may still be absent.
        `provider_name` identifies OpenAI, Bedrock, or Azure.
        Bedrock and Azure clients require their fixed `provider_name` values.
        `AsyncOpenAI` uses the caller's value because `base_url` selects its provider.
        `supports_prompt_cache_options` identifies gpt-5.6-and-later support in openai 2.45.0.
        `supports_prompt_cache_options` sets `Adapter.automatic_cache_breakpoints_default` to its inverse.
        `OpenAI.model` derives cataloged values from `PROMPT_CACHE_OPTIONS_MODELS`.
        It requires the parameter for uncataloged identifiers.
        `OpenAIBedrock.model` always requires it because Bedrock ids have no catalog.
        `pricing` supplies rates and modifiers.
        `regional_processing=False` uses the standard `1.0` token-price multiplier.
        `regional_processing=True` applies the regional token-price multiplier.
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
        self.regional_processing: bool = regional_processing
        self.supports_prompt_cache_options: bool = supports_prompt_cache_options
        self.reasoning_summary: ReasoningSummary | None = reasoning_summary
        self.service_tier: OpenAIResponsesServiceTier | None = service_tier

    def _precompute_fields(self, binding: Binding) -> _OpenAIPrecomputedFields:
        """Precompute the typed request fields the binding determines.

        Raises:
            ValueError: `automatic_cache_breakpoints=False` requires `prompt_cache_options` support.
            ValueError: `extra_body` contains a key in `_ADAPTER_POPULATED_WIRE_KEYS`.
            ValueError: A provider-executed tool type is unsupported or uses another provider.
            ValueError: A configured charged rate is not finite and nonnegative.
        """
        reject_extra_body_keys_the_adapter_populates(
            binding.extra_body, populated_keys=_ADAPTER_POPULATED_WIRE_KEYS
        )
        require_prompt_cache_options_support(
            model=self.model,
            automatic_cache_breakpoints=binding.automatic_cache_breakpoints,
            supports_prompt_cache_options=self.supports_prompt_cache_options,
        )
        provider_executed_tool_types = validated_provider_executed_tool_types(
            binding.provider_executed_tools,
            supported_types=_SUPPORTED_PROVIDER_EXECUTED_TOOL_TYPES,
            adapter_name="OpenAI Responses",
        )
        if provider_executed_tool_types and self.provider_name != "openai":
            raise ValueError(
                "OpenAI Responses provider_executed_tools require provider_name='openai'"
            )
        if provider_executed_tool_types & _WEB_SEARCH_TOOL_TYPES:
            require_finite_nonnegative_rate(
                rate_name="web_search_usd_per_invocation",
                rate=self.pricing.web_search_usd_per_invocation,
            )
        if "file_search" in provider_executed_tool_types:
            require_finite_nonnegative_rate(
                rate_name="file_search_usd_per_invocation",
                rate=self.pricing.file_search_usd_per_invocation,
            )
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
        tools: list[ToolParam] | Omit = omit
        tool_choice: _WireToolChoice | Omit = omit
        parallel_tool_calls: bool | Omit = omit
        if binding.tool_schemas or binding.provider_executed_tools:
            tools = _wire_tools(binding.tool_schemas, binding.provider_executed_tools)
            tool_choice = _wire_tool_choice(binding.tool_choice)
            parallel_tool_calls = binding.parallel_tool_calls
        return _OpenAIPrecomputedFields(
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
            reasoning=_wire_reasoning(
                binding.inference_params.reasoning_effort, self.reasoning_summary
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
            include=["reasoning.encrypted_content"],
            text=omit,
            extra_body=binding.extra_body,
            charged_provider_tools=bool(provider_executed_tool_types),
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output without I/O.

        Raises:
            ValueError: `binding` contains unsupported values.
        """
        return _BoundOpenAIText(adapter=self, precomputed_fields=self._precompute_fields(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format without I/O.

        Raises:
            ValueError: `binding` contains unsupported values.
            pydantic.PydanticInvalidForJsonSchema: `response_format` cannot produce a JSON schema.
            pydantic.PydanticUserError: `response_format` is not fully defined.
        """
        return _BoundOpenAIStructured(
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


class _OpenAIStream(AdapterStream):
    """One open Responses stream, backed by the SDK's stream helper."""

    def __init__(
        self,
        *,
        sdk_stream: AsyncResponseStream[Any],
        pricing: OpenAIPricingTable,
        regional_processing: bool,
        charged_provider_tools: bool,
    ) -> None:
        self._sdk_stream = sdk_stream
        self._pricing = pricing
        self._regional_processing = regional_processing
        self._charged_provider_tools = charged_provider_tools
        self._terminal_response: OpenAIResponse | None = None

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate the SDK stream into StreamItem values.

        The terminal event's response is kept for final(), which must not call the SDK's get_final_response():
        that raises RuntimeError unless the terminal event is response.completed.

        `forming_calls` uses required `output_index` because item ids are optional.
        The SDK emits the added event before its deltas.

        Reasoning arrives on two independent event types and both are forwarded:
        summary deltas requested by reasoning_summary, and text deltas from a reasoning item's content.
        A stream yielding no ReasoningDelta is a model returning no readable reasoning.

        Done events delimit reasoning parts without text.
        `REASONING_PART_SEPARATOR` precedes the next non-empty reasoning delta.
        Empty deltas emit no text or separator.
        The next reasoning delta from either channel consumes a pending separator.

        Yields:
            Stream items for SDK events langchaint models.

        Raises:
            openai.APIStatusError: The stream ends after an error event without a terminal response.
                It carries status 200, request headers, and the event body.
                OpenAI 2.51.0 does not raise this event itself.
            StreamProtocolError: The stream ends without a terminal response or error event.
        """
        error_event: ResponseErrorEvent | None = None
        reasoning_delta_yielded = False
        separator_pending = False
        forming_calls: dict[int, tuple[str, str]] = {}
        """Each forming function_call item's (call_id, name), keyed by output_index."""
        async for sdk_event in self._sdk_stream:
            if sdk_event.type == "response.output_text.delta":
                yield sdk_event.delta
            elif (
                sdk_event.type == "response.output_item.added"
                and sdk_event.item.type == "function_call"
            ):
                forming_calls[sdk_event.output_index] = (
                    sdk_event.item.call_id,
                    sdk_event.item.name,
                )
            elif sdk_event.type == "response.function_call_arguments.delta" and sdk_event.delta:
                call_id, name = forming_calls[sdk_event.output_index]
                yield ToolCallDelta(id=call_id, name=name, partial_args_json=sdk_event.delta)
            elif sdk_event.type in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                if sdk_event.delta:
                    if separator_pending:
                        separator_pending = False
                        yield ReasoningDelta(text=REASONING_PART_SEPARATOR)
                    reasoning_delta_yielded = True
                    yield ReasoningDelta(text=sdk_event.delta)
            elif sdk_event.type in (
                "response.reasoning_summary_text.done",
                "response.reasoning_text.done",
            ):
                separator_pending = reasoning_delta_yielded
            elif (
                sdk_event.type == "response.output_item.done"
                and sdk_event.item.type == "function_call"
            ):
                yield ToolCall(
                    id=sdk_event.item.call_id,
                    name=sdk_event.item.name,
                    args_json=sdk_event.item.arguments,
                )
            elif sdk_event.type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                self._terminal_response = sdk_event.response
            elif sdk_event.type == "error":
                error_event = sdk_event
        if self._terminal_response is None:
            if error_event is not None:
                raise openai.APIStatusError(
                    error_event.message,
                    response=self._sdk_stream._response,  # noqa: SLF001
                    body={
                        "code": error_event.code,
                        "message": error_event.message,
                        "param": error_event.param,
                    },
                )
            raise StreamProtocolError("stream ended without a terminal response")

    @override
    async def final(self) -> OpenAIResponse:
        """Return the terminal event's response, exactly as the SDK built it.

        The adapter preserves the lenient SDK response without revalidation.
        Revalidation could reject unmodeled output and discard partial output or billing.

        Raises:
            StreamProtocolError: items() was not exhausted first, so no terminal response was captured.
        """
        if self._terminal_response is None:
            raise StreamProtocolError("final() requires items() to be exhausted first")
        return self._terminal_response

    @override
    def billing_reported(self) -> ProviderBilling | None:
        """Return terminal billing or NaN for incomplete charged provider tools.

        OpenAI 2.45.0 stream state accumulates output items without counters.
        `ResponseUsage` arrives only with the terminal response.
        """
        if self._terminal_response is not None:
            return _billing_from_response(
                self._terminal_response,
                self._pricing,
                regional_processing=self._regional_processing,
            )
        if not self._charged_provider_tools:
            return None
        rates = self._pricing.rates_for(
            service_tier=_DEFAULT_TIER,
            input_tokens_total=0,
            regional_processing=self._regional_processing,
        )
        return rates.price(
            service_tier=_DEFAULT_TIER,
            usage_raw=None,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            input_tokens_cache_none=0,
            output_tokens=0,
            output_tokens_reasoning=0,
            provider_executed_tool_cost_in_usd=nan,
        )

    @override
    def request_id(self) -> str | None:
        """Read the request-id header off the response the SDK stream is reading.

        openai 3.0.0 exposes `AsyncResponseStream._response` as `httpx2.Response`.
        No public attribute exposes the same headers.
        """
        http_response = self._sdk_stream._response  # noqa: SLF001
        request_id: str | None = http_response.headers.get("x-request-id")
        return request_id

    @override
    async def close(self) -> None:
        await self._sdk_stream.close()


class _BoundOpenAI[OutputT](BoundAdapter[OutputT], ABC):
    """What both openai bindings share: the request path, and what a response says about itself.

    A subclass sets _adapter and _precomputed_fields in its own __init__ and implements interpret.
    """

    _adapter: OpenAIResponsesAdapter
    _precomputed_fields: _OpenAIPrecomputedFields

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Price counters using the reported `service_tier`.

        Raises:
            TypeError: raw is not an openai Response.
            pydantic.ValidationError: Cache counters exceed total input tokens.
        """
        return _billing_from_response(
            _as_response(raw),
            pricing=self._adapter.pricing,
            regional_processing=self._adapter.regional_processing,
        )

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Combine the response's id and model with request_id.

        Raises:
            TypeError: raw is not an openai Response.
        """
        response = _as_response(raw)
        return ResponseIdentity(
            model_served=response.model,
            response_id=response.id,
            request_id=request_id,
        )

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Convert messages into each attempt's input."""
        try:
            wire_input = _wire_input(messages)
        except _NotSendableError as not_sendable:
            return InvalidRequest(reason=not_sendable.reason)
        return _OpenAIRequestParams(
            precomputed=self._precomputed_fields,
            input=[*self._precomputed_fields.input_prefix, *wire_input],
        )

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open one responses.stream and return the live stream.

        Raises:
            TypeError: request was built by another adapter.
            Exception: The SDK fails to open the stream.
        """
        params = narrowed_request(request, _OpenAIRequestParams)
        precomputed = params.precomputed
        manager = self._adapter.client.responses.stream(
            model=precomputed.model,
            instructions=precomputed.instructions,
            max_output_tokens=precomputed.max_output_tokens,
            temperature=precomputed.temperature,
            reasoning=precomputed.reasoning,
            tools=precomputed.tools,
            tool_choice=precomputed.tool_choice,
            parallel_tool_calls=precomputed.parallel_tool_calls,
            prompt_cache_options=precomputed.prompt_cache_options,
            # The Responses service-tier vocabulary is wider than the openai 3.0.0 SDK literal.
            service_tier=cast("OpenAIServiceTier | Omit", precomputed.service_tier),
            include=precomputed.include,
            text=precomputed.text,
            store=False,
            input=params.input,
            extra_body=precomputed.extra_body,
        )
        return _OpenAIStream(
            sdk_stream=await manager.__aenter__(),
            pricing=self._adapter.pricing,
            regional_processing=self._adapter.regional_processing,
            charged_provider_tools=precomputed.charged_provider_tools,
        )


class _BoundOpenAIText(_BoundOpenAI[str]):
    """Text-bound adapter: output is the concatenated text of the turn."""

    def __init__(
        self, *, adapter: OpenAIResponsesAdapter, precomputed_fields: _OpenAIPrecomputedFields
    ) -> None:
        self._adapter = adapter
        self._precomputed_fields = precomputed_fields

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Read the turn's text as this binding's output, or report the run openai says failed.

        A failed status returns `_provider_failure` because emitted items are fragments.
        An incomplete status returns partial text with `max_tokens` or `refusal`.
        `assistant_message.text` includes refusal text that `response.output_text` omits.

        Raises:
            TypeError: raw is not an openai Response.
        """
        response = _as_response(raw)
        assistant_message = _assistant_message_from(response)
        if response.status == "failed":
            return _provider_failure(response, assistant_message=assistant_message)
        return _adapter_result(response, assistant_message.text, assistant_message)


class _BoundOpenAIStructured[ModelT: BaseModel](_BoundOpenAI[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the turn's text."""

    def __init__(
        self,
        *,
        adapter: OpenAIResponsesAdapter,
        precomputed_fields: _OpenAIPrecomputedFields,
        response_format: type[ModelT],
    ) -> None:
        """Precompute the request's text parameter, the JSON-schema format this binding asks for.

        `type_to_text_format_param` matches `responses.parse(text_format=...)`.
        The format replaces the omitted binding field in every request.
        """
        self._adapter = adapter
        self._response_format = response_format
        self._precomputed_fields = replace(
            precomputed_fields, text={"format": type_to_text_format_param(response_format)}
        )

    def _no_instance(
        self,
        response: OpenAIResponse,
        validation_error: ValidationError | None,
        assistant_message: AssistantMessage,
    ) -> NoOutputOutcome:
        """Report why the turn produced no instance and no tool call.

        Failed status takes precedence because emitted items are fragments.
        Refusal and truncation take precedence over validation.
        Completed text with `validation_error` returns `SchemaViolation`.
        Completed output without text returns `EmptyTurn`.
        Other statuses return `UnfinishedTurn`.
        Every variant carries `assistant_message`.
        """
        if response.status == "failed":
            return _provider_failure(response, assistant_message=assistant_message)
        if _normalized_stop_reason(response) == "refusal":
            return Refusal(assistant_message=assistant_message)
        if (
            response.status == "incomplete"
            and response.incomplete_details is not None
            and response.incomplete_details.reason == "max_output_tokens"
        ):
            return MaxCompletionTokensExceeded(assistant_message=assistant_message)
        if response.status == "completed":
            if validation_error is not None:
                return SchemaViolation(
                    validation_error_json=validation_error.json(include_url=False),
                    assistant_message=assistant_message,
                )
            return EmptyTurn(assistant_message=assistant_message)
        return UnfinishedTurn(
            reason=f"openai returned status {response.status!r}",
            assistant_message=assistant_message,
        )

    def _parsed_outcome(
        self, response: OpenAIResponse, assistant_message: AssistantMessage
    ) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, report a tool-call turn as None, or report why neither exists.

        Validation occurs after the attempt records its response and billing.
        Failed status returns `_provider_failure` even when fragment text validates.
        A valid instance takes precedence over tool calls.
        A tool-call turn without an instance returns `None`.
        """
        validation_error: ValidationError | None = None
        text = _first_output_text(response)
        if response.status != "failed" and text is not None:
            try:
                output = self._response_format.model_validate_json(text)
                return _adapter_result(response, output, assistant_message)
            except ValidationError as rejection:
                validation_error = rejection
        if response.status == "completed" and _normalized_stop_reason(response) == "tool_use":
            return _adapter_result(response, None, assistant_message)
        return self._no_instance(response, validation_error, assistant_message)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, or report why the response produced none.

        Raises:
            TypeError: raw is not an openai Response.
        """
        response = _as_response(raw)
        assistant_message = _assistant_message_from(response)
        return self._parsed_outcome(response, assistant_message)
