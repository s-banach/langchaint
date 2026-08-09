"""Adapter for the OpenAI Responses API over the official SDK.

Verified against openai 2.45.0:
- A structured binding sends `text.format` built by `type_to_text_format_param(Model)`, the same
  `text` parameter `responses.parse(text_format=Model)` builds, and validates the response text itself.
  The SDK validates inside `parse`, and inside the stream on `response.output_text.done` while its
  terminal response is still unset, so a rejection raised there reaches langchaint with neither the
  response nor its billing attached.
  `type_to_text_format_param` lives in `openai.lib._parsing._responses`, a private module, so an SDK
  upgrade can move it; `tests/test_provider_facts.py` pins the import.
- `responses.stream(...)` returns a manager whose entered stream yields typed events and assembles the response.
  Usage and status arrive on the terminal `response.completed`, `response.incomplete`,
  or `response.failed` event's response;
  the adapter captures that response itself because the SDK's `get_final_response()`
  raises RuntimeError unless the terminal event is `response.completed`.
- `prompt_cache_options` controls caching per request. Its own and its `mode` field's SDK docstrings
  state the caching rules this adapter is built on: the parameter is supported on gpt-5.6 and later;
  `{"mode": "explicit"}` with no explicit breakpoints disables caching;
  implicit mode writes up to the latest three explicit breakpoints and explicit mode up to the
  latest four, older marks staying readable for matching; and `ttl` takes "30m" as its only value,
  so there is no TTL to configure and this adapter has no counterpart to the anthropic adapter's `cache_ttl`.
  The adapter sends the parameter when the binding sets automatic_prompt_caching False; bound True,
  the provider's implicit caching is left in place and nothing is sent.
  On a model whose supports_prompt_cache_options is False, a binding that declines caching raises at
  bind time, the parameter that would carry it being one the model does not take.
- A part with cache_breakpoint True becomes `prompt_cache_breakpoint: {"mode": "explicit"}` on its wire part,
  under either binding value, and the adapter sends every mark and caps nothing,
  the per-request write limits above being the API's to apply.
  With automatic_prompt_caching False on a model taking `prompt_cache_options`,
  marked parts are what re-enables caching at exactly those boundaries.
- The API stores responses server-side for later retrieval by default;
  the adapter always sends `store=False`: the caller's `GenerationInput` is the whole state,
  and a stored copy would be an unused side effect.
- The adapter sends `include=["reasoning.encrypted_content"]` on every request,
  so reasoning items come back with `encrypted_content` populated and round-trip statelessly under `store=False`.
  The SDK documents `include` as what populates `encrypted_content`;
  a live run on 2026-07-17 saw it populated without the flag, undocumented behavior the adapter does not rely on.
- `reasoning.summary` carries the constructor's `reasoning_summary`.
  The SDK types it as "auto", "concise", or "detailed" and describes it as a summary of the reasoning
  the model performed; `generate_summary` carries the same values and the SDK marks it deprecated in
  favor of `summary`, so it is never sent.
  `reasoning.effort` and `reasoning.summary` are assembled key by key so an unset one is omitted
  rather than sent as an explicit null.

Cache writes bill starting with gpt-5.6, so the OpenAIPricingTable's cache-write rate applies here too.
That is a price rather than a wire fact, so it comes from the page the subpackage docstring cites,
not from the SDK.

`usage.input_tokens` includes `input_tokens_details.cached_tokens` and
`input_tokens_details.cache_write_tokens`, so it is the provider-reported all-inclusive input total
the Usage partition is checked against. That one is verified by docs rather than by introspection:
the SDK documents no relationship among the input counters,
so `_billing_from_response` carries the page that does.

Mapping decisions:
- A str system_prompt travels as the `instructions` parameter, not as an input item;
  a parts system_prompt travels as a developer-role input message first in every request's input,
  the message the SDK documents `instructions` as inserting, because only input message parts
  carry prompt_cache_breakpoint.
- An AssistantMessage re-feeds its TurnPart values in emission order,
  which the API requires for replay under store=False:
  a ReasoningPart is its reasoning item re-sent unchanged,
  a RawPart re-sends its stored item unchanged,
  a ToolCall one `function_call` item,
  and a maximal run of adjacent TextParts one assistant message item;
  ToolMessage becomes a `function_call_output` item keyed by call_id.
  The API has no is_error flag, so the error text in output is the only error signal.
- ImagePart becomes an `input_image` item with a data: URI and `detail="auto"`.
- The API reports no finish reason; stop_reason is derived: a `ResponseOutputRefusal` content block means refusal,
  else any `function_call` output item means tool_use, otherwise status "completed" means end_turn,
  status "incomplete" means max_tokens or refusal by its reason ("max_output_tokens" or
  "content_filter", the only two the SDK types), and anything else is "other".
- A `ResponseOutputRefusal` content part becomes a TextPart, so the refusal the model wrote is the
  turn's text and replays as text. anthropic's ContentBlock union has no refusal variant
  (anthropic 0.120.0), so there a refusal arrives as ordinary text with stop_reason "refusal";
  mapping openai's part to a TextPart gives the two providers one neutral shape and leaves the stop
  reason as the signal on both.
- Status "failed" is the API reporting that the run did not finish (`response.error` names why), so
  whatever it emitted is a fragment rather than the turn. Both bindings report it as the variant
  `_provider_failure` picks off `response.error.code`, carrying that response's billing, and a
  structured binding does so whether or not the fragment happened to validate.
- Streaming yields the SDK's own answer delta strings unwrapped, each reasoning delta in a ReasoningDelta,
  each argument fragment in a ToolCallDelta, and each tool call once, complete, from its
  `response.output_item.done` event.
  Usage, cost, and stop reason arrive only on final()'s AdapterResult.
"""

from abc import ABC
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast, override

import openai
from openai import AsyncOpenAI, Omit, omit
from openai.lib._parsing._responses import type_to_text_format_param
from openai.lib.streaming.responses import AsyncResponseStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
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
    ToolChoiceFunctionParam,
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
    Binding,
    BoundAdapter,
    EmptyTurn,
    ErrorClassification,
    MaxCompletionTokensExceeded,
    NoOutput,
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
    request_id_from_raw,
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
    _DEFAULT_TIER,
    _DISPOSITION_BY_ERROR_CODE,
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

type _WireToolChoice = Literal["none", "auto", "required"] | ToolChoiceFunctionParam
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""

type ReasoningSummary = Literal["auto", "concise", "detailed"]
"""How much readable text to ask the API for, the values reasoning.summary takes."""


def _wire_reasoning(
    effort: ReasoningEffort | None, summary: ReasoningSummary | None
) -> Reasoning | Omit:
    """Assemble the reasoning object from the keys that are set, omitting it when neither is.

    Reasoning is a total=False TypedDict whose effort and summary are both Optional, so passing None
    type-checks and sends an explicit null, a different request from omitting the key.
    The other keys the TypedDict carries (context, mode, and the deprecated generate_summary alias
    of summary) are not mapped, so they are never sent.
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

    Fields set to the SDK's omit sentinel leave the provider default in place; passing them as explicit keywords
    (never **kwargs) keeps the SDK's overload resolution intact.
    instructions is the bound str system prompt; a parts system prompt travels in input_prefix instead.
    tool_choice and parallel_tool_calls are omitted without tools because the API rejects them otherwise.
    include is always ["reasoning.encrypted_content"]:
    the adapter re-feeds the whole Sequence[Message] every turn, so every response's reasoning items
    must carry the payload a later request replays.
    """

    model: str
    instructions: str | None
    input_prefix: list[ResponseInputItemParam]
    """Items sent ahead of the Sequence[Message] every request: a system_prompt bound as parts becomes
    one developer-role input message here (its parts carry prompt_cache_breakpoint marks,
    which the instructions string cannot), and a str or absent system_prompt leaves it empty."""

    max_output_tokens: int | Omit
    temperature: float | Omit
    reasoning: Reasoning | Omit
    tools: list[FunctionToolParam] | Omit
    tool_choice: _WireToolChoice | Omit
    parallel_tool_calls: bool | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    service_tier: OpenAIServiceTier | Omit
    include: list[ResponseIncludable]
    text: ResponseTextConfigParam | Omit
    """The structured binding's JSON-schema format, omitted by the text binding, which asks for none."""

    extra_body: Mapping[str, object] | None


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
"""The wire keys an extra_body must not hold: every keyword open_stream passes,
plus stream, which the SDK's stream method sets and its event parsing depends on."""


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


def _function_call_output(
    content: str | tuple[ContentPart, ...],
) -> str | ResponseFunctionCallOutputItemListParam:
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


def _replayed_item(raw: Mapping[str, object]) -> ResponseInputItemParam:
    """Copy one stored SDK dump into the input item it came from, unread and unchanged.

    The dict is the producing SDK item's model_dump; when this adapter produced it, its shape is the
    wire param's by construction, so the cast holds. A dump another provider produced is not this
    shape; it is passed through unchanged, never dropped or neutralized here (trimming is the app's
    job), and left to the API. Reconstructing it field by field would risk changing the payload the
    API re-reads.
    The shallow copy keeps the wire path from ever aliasing the frozen message's stored payload into
    a mutable request structure.
    """
    # cast: a deliberately-opaque value re-enters the typed API whose own serialization produced it.
    return cast("ResponseInputItemParam", dict(raw))


def _assistant_items(assistant_message: AssistantMessage) -> list[ResponseInputItemParam]:
    """Convert one AssistantMessage to its input items in turn order.

    The API requires the original item order for replay under store=False.
    A maximal run of adjacent TextParts becomes one assistant message item whose content joins their texts
    (turn carries no message-item boundary, so the run is the inverse of the produce rule's per-part split);
    each ToolCall becomes a function_call item keyed by call_id,
    which the paired ToolMessage's function_call_output references.
    ReasoningPart.raw and RawPart.raw go to the wire unchanged, routed by their own
    type key, so encrypted_content replays byte-identical.
    A dump another provider produced goes to the wire the same way and the API rejects its
    unknown type key, so a Sequence[Message] replayed through the wrong provider fails loudly.
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
    """Convert messages to input items; the system prompt is not one."""
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


def _provider_failure(
    response: OpenAIResponse, *, assistant_message: AssistantMessage
) -> ProviderFailedTransiently | ProviderFailedTerminally:
    """Report a failed status as the variant its error code's disposition selects.

    A failed status is the API saying no generation completed, so whatever output items the response
    holds are a fragment rather than the turn. Both variants carry that fragment as their turn.

    reason is response.error.message verbatim, the only description of a condition langchaint does
    not model; a response reporting the failure with no error object at all gets langchaint's own
    sentence, which says exactly that.
    An error code outside the table is terminal: retrying is what spends the budget, so a code nobody
    classified fails the item rather than being resent at full price.
    rate_limit_exceeded sets is_rate_limit, which the retry loop's TransientError carries into the
    admitted() block's exit, where parse maps it to PauseAll and pauses admission the way a 429
    status does. Neither variant carries a server-stated wait, so the pause runs for the drawn wait.
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

    The BoundAdapter methods that read a response take BaseModel, because BoundLLM holds them and
    the neutral core imports no SDK. Every value reaching them came from this adapter's own stream,
    so another type is a defect in langchaint and not a provider behavior.

    Raises:
        TypeError: raw is not an openai Response.
    """
    if not isinstance(raw, OpenAIResponse):
        raise TypeError(f"expected an openai Response, got {type(raw).__name__}")
    return raw


def _identity_from_response(raw: BaseModel) -> ResponseIdentity:
    """Read the response's own id, the model it reports serving the request, and the request id.

    id and model are both required on openai.types.responses.Response and every value of each is a
    str (openai 2.48.0), so neither is absent and neither needs converting.

    Raises:
        TypeError: raw is not an openai Response.
    """
    response = _as_response(raw)
    return ResponseIdentity(
        model_served=response.model,
        response_id=response.id,
        request_id=request_id_from_raw(response),
    )


def _first_output_text(response: OpenAIResponse) -> str | None:
    """Return the text of the turn's first output_text content part, None when it holds none.

    The part a structured turn's instance is validated from. Reading the first part matches what the
    SDK's own parse yields for every response it does not raise on: it validates every output_text
    part and returns the first instance, so a response whose first part is not the instance raises there.
    """
    for item in response.output:
        if item.type == "message":
            for content_part in item.content:
                if content_part.type == "output_text":
                    return content_part.text
    return None


def _normalized_stop_reason(response: OpenAIResponse) -> StopReason:
    """Derive the stop reason; the API reports no finish reason field.

    Status "incomplete" with reason "content_filter" is a refusal: the provider's filter blocked
    the turn, so the structured path fails the item under RefusalError's no-retry policy instead
    of spending the retry budget resending a request the filter blocks every time.
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
    """Join a reasoning item's readable text, None when it holds none.

    The text arrives in parts, several per item: the SDK types summary and content as lists, and the
    stream carries one summary_index or content_index per part, each accumulating its own text
    deltas, so a part is a separately delimited unit and the parts join on a blank line rather than
    concatenating into one run.
    Asking for a summary is what the constructor's reasoning_summary does.

    content wins over summary where both hold text: the SDK types a content element reasoning_text
    and a summary element summary_text (openai 2.48.0), so content is the reasoning a model wrote and
    summary is a rendering of it.
    Which of the two a given model fills is request-time behavior SDK introspection cannot show, so
    the adapter reads both; reading only one would drop returned text into an unreportable None.

    Empty parts are dropped before the join, so an item whose parts are all empty yields None
    rather than the separator alone; text-free stays the single condition text is None.
    """
    summary = REASONING_PART_SEPARATOR.join(part.text for part in item.summary if part.text)
    content = REASONING_PART_SEPARATOR.join(part.text for part in item.content or () if part.text)
    return content or summary or None


def _assistant_message_from(response: OpenAIResponse) -> AssistantMessage:
    """Build the langchaint assistant turn from the output items, item order preserved.

    A reasoning item becomes a ReasoningPart carrying the item's own model_dump for verbatim replay,
    beside the readable text _reasoning_text extracts from it;
    a message item becomes one TextPart per content part it holds, in their order, from an
    output_text part and from a refusal part alike, because the sentences the model wrote to refuse
    are the turn's text and a turn built without them replays as nothing;
    every other item, a built-in tool call among them, becomes a RawPart holding the item's
    own model_dump, so the turn carries what the response was billed for.
    """
    turn: list[TurnPart] = []
    for item in response.output:
        if item.type == "reasoning":
            turn.append(
                ReasoningPart(
                    raw=item.model_dump(mode="python", exclude_none=True),
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
            turn.append(RawPart(raw=item.model_dump(mode="python", exclude_none=True)))
    return AssistantMessage(turn=tuple(turn))


def _billing_from_response(
    response: OpenAIResponse, pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable]
) -> Billing:
    """Price one response's raw counters at the table its priced tier selects.

    The whole response is the argument, not its usage: the tier that selects the rates is on the
    response and the counters are on the usage, and pricing one response's counters at another
    response's tier is the mistake worth making impossible.

    input_tokens is the all-inclusive input total,
    so the uncached counter is the remainder after subtracting cached and cache-write tokens.
    The SDK documents no relationship among the input counters, so the source is the provider's
    prompt-caching page, whose worked example reports 1920 cached tokens inside a 2006-token
    prompt total, read 2026-07-25:
    https://developers.openai.com/api/docs/guides/prompt-caching
    output_tokens_details and its reasoning_tokens counter are both required on the SDK Usage.

    A response with no usage at all bills zero counters, at the priced tier's rates.

    Raises:
        pydantic.ValidationError: the counters leave input_tokens_cache_none negative, a response
            over-reporting its cache counters, so the priced Usage rejects it.
    """
    service_tier = _priced_tier(response.service_tier)
    table = pricing.get(service_tier, _UNPRICED)
    usage = response.usage
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
    details = usage.input_tokens_details
    return table.price(
        service_tier=service_tier,
        usage_raw=usage,
        input_tokens_cache_read=details.cached_tokens,
        input_tokens_cache_write=details.cache_write_tokens,
        input_tokens_cache_none=(
            usage.input_tokens - details.cached_tokens - details.cache_write_tokens
        ),
        output_tokens=usage.output_tokens,
        output_tokens_reasoning=usage.output_tokens_details.reasoning_tokens,
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

    All three expose the same responses.stream method and with_options,
    so the adapter logic is identical across the first-party API, Bedrock, and Azure.
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
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIServiceTier | None = None,
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        The stored client is a with_options(max_retries=0) copy: langchaint's retry loop owns all retrying,
        counts every request as an attempt, and feeds each failure to SharedBackoff through parse,
        so the SDK must never retry beneath it.

        reasoning_summary asks the API for readable text, which reaches ReasoningPart.text where the
        reasoning item carries no reasoning text of its own;
        None sends no summary field and leaves the provider default in place.
        A model may return no summary even when one is requested.
        It is a constructor parameter rather than an InferenceParams field because InferenceParams
        is neutral and anthropic has no reasoning summary of its own.

        provider_name says which provider the client reaches: "openai" for AsyncOpenAI,
        "aws.bedrock" for AsyncBedrockOpenAI, "azure.ai.openai" for AsyncAzureOpenAI.
        openai_model passes "openai" and openai_bedrock_model "aws.bedrock".
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
        openai_model defaults an unstated value from PROMPT_CACHE_OPTIONS_MODELS for a cataloged
        id and requires it for any other; openai_bedrock_model always requires it,
        having no catalog of Bedrock ids to read.
        It is a parameter here rather than a lookup on model because model is a str whose namespace
        this adapter cannot know: it serves the platforms provider_name_by_client_class maps and
        every OpenAI-compatible endpoint a base AsyncOpenAI's base_url reaches.

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
        self.reasoning_summary = reasoning_summary
        self.service_tier: OpenAIServiceTier | None = service_tier

    def _precompute_fields(self, binding: Binding) -> _OpenAIPrecomputedFields:
        """Precompute the typed request fields the binding determines.

        A str system_prompt travels as the instructions parameter,
        which the SDK documents as "a system (or developer) message inserted into the model's context".
        A parts system_prompt travels as that message itself, a developer-role input message
        first in every request's input, because only input message parts carry prompt_cache_breakpoint.

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
                omit if binding.automatic_prompt_caching else PromptCacheOptions(mode="explicit")
            ),
            service_tier=self.service_tier if self.service_tier is not None else omit,
            include=["reasoning.encrypted_content"],
            text=omit,
            extra_body=binding.extra_body,
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output; pure conversion, no I/O.

        Propagates _precompute_fields' ValueError.
        """
        return _BoundOpenAIText(adapter=self, precomputed_fields=self._precompute_fields(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format; pure conversion, no I/O.

        Propagates _precompute_fields' ValueError.
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

    def __init__(self, *, sdk_stream: AsyncResponseStream[Any]) -> None:
        self._sdk_stream = sdk_stream
        self._terminal_response: OpenAIResponse | None = None

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate the SDK stream into StreamItem values.

        The terminal event's response is kept for final(), which must not call the SDK's get_final_response():
        that raises RuntimeError unless the terminal event is response.completed.

        forming_calls is keyed by output_index, the one identifier required on both of its event
        types (the item's own id is optional).
        The SDK's stream state asserts that the added event precedes the item's deltas, so the
        lookup cannot miss.

        Reasoning arrives on two independent event types and both are forwarded:
        summary deltas, which the constructor's reasoning_summary asks for,
        and reasoning text deltas from a model that fills the reasoning item's content.
        A stream yielding no ReasoningDelta is a model returning no readable reasoning.

        Reasoning text arrives in parts, and the API breaks between two parts structurally, never
        as text: each part ends with its own done event instead. That done event puts a
        REASONING_PART_SEPARATOR delta before the next part's first delta, so a caller
        concatenating ReasoningDelta text gets those breaks as characters.
        A part holding no text is not a part: a delta carrying no characters is dropped, and a
        separator falls only between two parts that streamed text, so the reasoning never opens or
        ends on a blank line and never doubles one. That is the rule _reasoning_text applies when it
        drops empty parts before joining.
        A pending separator is scoped to neither one item nor one channel: whichever reasoning delta
        comes next consumes it.

        Yields:
            Stream items; SDK events langchaint does not model (built-in tool activity) stream
            nothing and reach the caller in the turn final()'s response carries.

        Raises:
            openai.APIStatusError: the stream ended without a terminal response after an error
                event; raised on the live response, so it carries the 200 status and the open
                request's headers, with the event's code, message, and param as its body. The SDK
                itself never raises on the event (openai 2.51.0 forwards it accumulated-only), so
                this raise is what turns a mid-stream error into a failure parse can verdict.
            StreamProtocolError: the stream ended with neither a terminal response nor an error
                event.
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

        It is never re-validated into another model: the SDK constructs a response leniently and
        tolerates an output item type or an enum value it does not model, so validating that
        response against the SDK's own strict model can raise pydantic's ValidationError over a
        response whose partial output and billing the caller is owed.

        Raises:
            StreamProtocolError: items() was not exhausted first, so no terminal response was captured.
        """
        if self._terminal_response is None:
            raise StreamProtocolError("final() requires items() to be exhausted first")
        return self._terminal_response

    @override
    def billing_reported(self) -> None:
        """None: openai reports usage only on the terminal response, so an open stream has none.

        The SDK's stream state accumulates the response's output items and no counters
        (openai 2.45.0), and ResponseUsage arrives on the response the completed event carries,
        which is exactly the event a stream that ends early never receives.
        """

    @override
    def request_id(self) -> str | None:
        """Read the request-id header off the response the SDK stream is reading.

        AsyncResponseStream exposes its httpx response only as _response, which it sets in its
        constructor, so this is readable from the moment the stream opens and there is no public
        route to the same headers (openai 2.48.0).
        """
        http_response = self._sdk_stream._response  # noqa: SLF001
        request_id: str | None = http_response.headers.get("x-request-id")
        return request_id

    @override
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundOpenAI[OutputT](BoundAdapter[OutputT], ABC):
    """What both openai bindings share: the request path, and what a response says about itself.

    A subclass sets _adapter and _precomputed_fields in its own __init__ and implements interpret.
    """

    _adapter: OpenAIResponsesAdapter
    _precomputed_fields: _OpenAIPrecomputedFields

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Price the response's counters at the table its priced tier selects.

        Raises:
            TypeError: raw is not an openai Response.
            pydantic.ValidationError: the response over-reports its cache counters, leaving the
                derived uncached-input counter negative.
        """
        return _billing_from_response(_as_response(raw), pricing=self._adapter.pricing)

    @override
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Read the response's own id, the model it reports serving the request, and the request id.

        Raises:
            TypeError: raw is not an openai Response.
        """
        return _identity_from_response(raw)

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams:
        """Convert messages into the input every attempt of this call sends.

        Returns no InvalidRequest: this adapter puts every Sequence[Message] on the wire.
        """
        return _OpenAIRequestParams(
            precomputed=self._precomputed_fields,
            input=[*self._precomputed_fields.input_prefix, *_wire_input(messages)],
        )

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open one responses.stream and return the live stream; connection failures raise here.

        Raises:
            TypeError: request was built by another adapter.
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
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
            service_tier=precomputed.service_tier,
            include=precomputed.include,
            text=precomputed.text,
            store=False,
            input=params.input,
            extra_body=precomputed.extra_body,
        )
        return _OpenAIStream(sdk_stream=await manager.__aenter__())


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

        A failed status means the run did not finish, and response.error names why, so the output
        items hold whatever had been emitted rather than the turn; reporting that as a Response would
        present a fragment as the answer. _provider_failure states which variant the error code picks.
        An incomplete status is deliberately not this case: a turn cut off at the token cap or by a
        content filter is the answer as far as it got, and stop_reason ("max_tokens" or "refusal")
        is how the caller sees that.
        The text is the assistant turn's, not response.output_text: output_text concatenates the
        output_text content parts alone, so a refusal turn would come back with an empty output while
        the same Response's assistant_message carried the sentences the model wrote to refuse.

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

        The format is built by the same type_to_text_format_param call responses.parse makes, so the
        request carries what passing text_format would have sent.
        It replaces the binding's omitted text field, so every request this binding builds carries it
        and the two bindings send the same fields.
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

        validation_error is pydantic's rejection of the turn's text, None when the turn carried no
        text to validate. On a completed turn the two answers are SchemaViolation and EmptyTurn;
        everywhere else the status names the failure and the rejection adds nothing.

        Each variant carries assistant_message, so the turn a rejected 200 did produce reaches the
        caller on the failure.
        No variant carries a stop reason: each GenerationError subclass fixes it, and _normalized_stop_reason, used
        here only to detect the refusal, tests a function_call item ahead of the response status,
        which is right for what a Response reports and wrong for a truncated turn.
        A failed status is tested first, ahead of the refusal and the truncation: the API is reporting
        that the run did not finish, so whatever items it emitted are a fragment, and a refusal part
        among them is no more the turn than a text part is. Testing the refusal first would make one
        response Refusal here and a provider failure on the text binding, which reads the same
        status first.
        A content-filtered response reaches Refusal through the stop reason, so it fails the
        item once instead of being retried at full price for an outcome that will not change.
        The completed status is what SchemaViolation and EmptyTurn are tested on, so a status
        reporting that the run never finished cannot be reported as a turn that finished.
        Every remaining status is a run that stopped short of a turn, which is UnfinishedTurn
        naming the status.
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

    def _parsed_output(
        self, response: OpenAIResponse, assistant_message: AssistantMessage
    ) -> ModelT | None | NoOutputOutcome:
        """Validate the turn's text into the instance, report a tool-call turn as None, or report why neither exists.

        Validating here rather than in the SDK is what puts the response and its text in scope when
        the text is rejected: the variant returned for a rejection is one the retry loop can place
        against the attempt it already recorded, where a raise from inside the SDK is not.

        A failed status is rejected even when the text validates: the run did not finish, and
        response.error names why, so an instance built from the fragment it had emitted would be
        presented as the answer. _no_instance reports it as the failure variant _provider_failure chose.

        None is the tool-call turn and nothing else: the turn is the tool calls, which the assistant
        message carries, so a turn whose text is not the instance yields no instance without anything
        having gone wrong. The instance wins where a completed turn carries both, because a turn that
        produced the instance answered the request whether or not it also called a tool.
        """
        validation_error: ValidationError | None = None
        text = _first_output_text(response)
        if response.status != "failed" and text is not None:
            try:
                return self._response_format.model_validate_json(text)
            except ValidationError as rejection:
                validation_error = rejection
        if response.status == "completed" and _normalized_stop_reason(response) == "tool_use":
            return None
        return self._no_instance(response, validation_error, assistant_message)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        """Validate the turn's text into the instance, or report why the response produced none.

        Raises:
            TypeError: raw is not an openai Response.
        """
        response = _as_response(raw)
        assistant_message = _assistant_message_from(response)
        output = self._parsed_output(response, assistant_message)
        if isinstance(output, NoOutput):
            return output
        return _adapter_result(response, output, assistant_message)
