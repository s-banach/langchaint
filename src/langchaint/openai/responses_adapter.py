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
  The adapter sends the parameter only when the binding sets automatic_prompt_caching False and the
  constructor's supports_prompt_cache_options is True; bound True,
  the provider's implicit caching is left in place and nothing is sent.
  A model that predates the parameter therefore has no way to turn caching off and gets no caching
  parameter at all, which keeps automatic_prompt_caching a binding parameter every model accepts.
- A part with cache_breakpoint True becomes `prompt_cache_breakpoint: {"mode": "explicit"}` on its wire part,
  under either binding value, and the adapter sends every mark and caps nothing,
  the per-request write limits above being the API's to apply.
  With automatic_prompt_caching False on a model taking `prompt_cache_options`,
  marked parts are what re-enables caching at exactly those boundaries.
- The API stores responses server-side for later retrieval by default;
  the adapter always sends `store=False` because conversation state is the caller's conversation argument,
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

Cache writes bill starting with gpt-5.6, so the PricingTable's cache-write rate applies here too.
That is a price rather than a wire fact, so it comes from the page the subpackage docstring cites,
not from the SDK.

`usage.input_tokens` includes `input_tokens_details.cached_tokens` and
`input_tokens_details.cache_write_tokens`, so it is the provider-reported all-inclusive input total
the Usage partition is checked against. That one is verified by docs rather than by introspection:
the SDK documents no relationship among the input counters,
so `_normalized_usage` carries the page that does.

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
  status "incomplete" means max_tokens or refusal by its reason ("max_output_tokens" or
  "content_filter", the only two the SDK types), and anything else is "other".
- A `ResponseOutputRefusal` content part becomes a TextPart, so the refusal the model wrote is the
  turn's text and replays as text. anthropic's ContentBlock union has no refusal member
  (anthropic 0.120.0), so there a refusal arrives as ordinary text with stop_reason "refusal";
  mapping openai's part to a TextPart gives the two providers one neutral shape and leaves the stop
  reason as the signal on both.
- Status "failed" is the API reporting that the run did not finish (`response.error` names why), so
  whatever it emitted is a fragment rather than the turn. Both bindings report it as the member
  `_provider_failure` picks off `response.error.code`, carrying that response's billing, and a
  structured binding does so whether or not the fragment happened to validate.
- Streaming yields the SDK's own delta strings unwrapped and each tool call once, complete,
  from its `response.output_item.done` event; argument fragments are never surfaced.
  Usage, cost, and stop reason arrive only on final()'s AdapterResult.
"""

import base64
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, override

import openai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI, Omit, omit
from openai.lib._parsing._responses import type_to_text_format_param
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
    ResponseFunctionCallOutputItemListParam,
    ResponseInputItemParam,
)
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam

from langchaint.adapter import (
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
from langchaint.inference_params import ReasoningEffort
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
from langchaint.pricing import UNPRICED, PricingTable
from langchaint.tools import ToolSchema
from langchaint.usage import ZERO_USAGE, Usage

_RATE_LIMIT_STATUSES = frozenset({429})

type _WireToolChoice = Literal["none", "auto", "required"] | ToolChoiceFunctionParam
"""The subset of the API's tool_choice union the neutral vocabulary maps onto."""

type ReasoningSummary = Literal["auto", "concise", "detailed"]
"""How much readable text to ask the API for, the values reasoning.summary takes."""

type OpenAIServiceTier = Literal["auto", "default", "flex", "scale", "priority"]
"""What a request may ask for, and what a response reports (openai 2.45.0 types both with this literal).

The API documents the response value as the processing mode actually used and says it may differ
from the value the request set, so the tier is read off each response rather than assumed.
"""

type OpenAIPricedServiceTier = Literal["default", "flex", "scale", "priority"]
"""The pricing mapping's key: the tiers a caller can hold rates for.

"auto" is excluded. It is in the response literal only because request and response share one type,
and it names no processing mode, so it is not a tier anyone has a rate for.
"""

_DEFAULT_TIER: OpenAIPricedServiceTier = "default"


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
    service_tier: OpenAIServiceTier | Omit
    include: list[ResponseIncludable]


def _image_data_uri(image_part: ImagePart) -> str:
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


def _function_call_output(
    content: str | tuple[Part, ...],
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


type _FailureDisposition = Literal["transient", "terminal"]

_DISPOSITION_BY_ERROR_CODE: Mapping[str, _FailureDisposition] = {
    "server_error": "transient",
    "rate_limit_exceeded": "transient",
    "vector_store_timeout": "transient",
    "invalid_prompt": "terminal",
    "bio_policy": "terminal",
    "invalid_image": "terminal",
    "invalid_image_format": "terminal",
    "invalid_base64_image": "terminal",
    "invalid_image_url": "terminal",
    "image_too_large": "terminal",
    "image_too_small": "terminal",
    "image_parse_error": "terminal",
    "image_content_policy_violation": "terminal",
    "invalid_image_mode": "terminal",
    "image_file_too_large": "terminal",
    "unsupported_image_media_type": "terminal",
    "empty_image_file": "terminal",
    "failed_to_download_image": "terminal",
    "image_file_not_found": "terminal",
}
"""Whether a resend may get past the failure each ResponseError.code names (openai 2.45.0).

Every member of the SDK's code literal is a key, which tests/test_provider_facts.py pins, so the
unknown-code path below is reached only by a code newer than the installed SDK.
The three transient codes are read off their names: the SDK documents none of the codes, and these
three name a condition of the moment while every other names a property of the request.
failed_to_download_image is terminal for that reason: a URL the caller got wrong fails identically
on every resend.
"""


def _provider_failure(
    response: OpenAIResponse, *, assistant_message: AssistantMessage
) -> ProviderFailedTransiently | ProviderFailedTerminally:
    """Report a failed status as the member its error code's disposition selects.

    A failed status is the API saying no generation completed, so whatever output items the response
    holds are a fragment rather than the turn. Both members carry that fragment as their turn.

    reason is response.error.message verbatim, the only description of a condition langchaint does
    not model; a response reporting the failure with no error object at all gets langchaint's own
    sentence, which says exactly that.
    An error code outside the table is terminal: retrying is what spends the budget, so a code nobody
    classified fails the item rather than being resent at full price.
    rate_limit_exceeded sets is_rate_limit, which the retry loop's TransientError carries to the
    RateLimiter, pausing admission the way a 429 status does. Neither member carries a server-stated
    wait, so that pause runs for the jittered backoff.
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
    the neutral core imports no SDK. Every value reaching them came from this adapter's own send or
    stream, so another type is a defect in langchaint and not a provider behavior.

    Raises:
        TypeError: raw is not an openai Response.
    """
    if not isinstance(raw, OpenAIResponse):
        raise TypeError(f"expected an openai Response, got {type(raw).__name__}")
    return raw


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

    The text arrives as summary parts, several per item: the SDK types summary as a list, and the
    stream carries one summary_index per part, each accumulating its own text deltas, so a part is a
    separately delimited unit and the parts join on a blank line rather than concatenating into one run.
    Asking for a summary is what the constructor's reasoning_summary does.

    summary wins over content where both hold text, because summary is the list the request asks for.
    content is read at all because the SDK models it as a list of reasoning_text elements and ships
    delta and done events for it; which of the two a given model fills is request-time behavior SDK
    introspection cannot show, so the adapter reads both.
    Reading only summary would drop returned text into an unreportable None.

    Empty parts are dropped before the join, so an item whose parts are all empty yields None
    rather than the separator alone; text-free stays the single condition text is None.
    """
    summary = "\n\n".join(part.text for part in item.summary if part.text)
    content = "\n\n".join(part.text for part in item.content or () if part.text)
    return summary or content or None


def _assistant_message_from(response: OpenAIResponse) -> AssistantMessage:
    """Build the langchaint assistant turn from the output items, item order preserved.

    A reasoning item becomes a ReasoningTrace carrying the item's own model_dump for verbatim replay,
    beside the readable text _reasoning_text extracts from it;
    a message item becomes one TextPart per content part it holds, in their order, from an
    output_text part and from a refusal part alike, because the sentences the model wrote to refuse
    are the turn's text and a turn built without them replays as nothing;
    built-in tool call items are dropped (built-in tools are out of scope).
    """
    turn: list[TurnElement] = []
    for item in response.output:
        if item.type == "reasoning":
            turn.append(
                ReasoningTrace(
                    reasoning=item.model_dump(mode="python", exclude_none=True),
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
    return AssistantMessage(turn=tuple(turn))


def _priced_tier(service_tier: OpenAIServiceTier | None) -> OpenAIPricedServiceTier:
    """Which rates a response asks for: what it reports, or the default tier when it names none.

    "auto" is a request word that names no processing mode, and the API documents the response
    field as the mode actually used, so a response carrying it says nothing about what served it.
    It prices at the default key like a response reporting no tier at all, rather than as NaN:
    the account was most likely on the default tier, and a number that may be wrong by a tier
    multiplier beats destroying the cost of a call that was paid for.
    """
    if service_tier is None or service_tier == "auto":
        return _DEFAULT_TIER
    return service_tier


def _normalized_usage(
    response: OpenAIResponse, pricing: Mapping[OpenAIPricedServiceTier, PricingTable]
) -> Usage:
    """Map one response's raw counters onto langchaint's disjoint partition and price them.

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

    A response with no usage at all prices to ZERO_USAGE, which is zero counters and zero cost.
    """
    usage = response.usage
    if usage is None:
        return ZERO_USAGE
    details = usage.input_tokens_details
    return pricing.get(_priced_tier(response.service_tier), UNPRICED).price(
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

    All three expose the same responses.create/parse/stream methods and with_options,
    so the adapter logic is identical across the first-party API, Bedrock, and Azure.
    The client parameter is annotated AsyncOpenAI because the other two subclass it;
    provider_name_by_client_class is what tells them apart.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = {
        AsyncBedrockOpenAI: "aws.bedrock",
        AsyncAzureOpenAI: "azure.ai.openai",
    }
    """AsyncOpenAI is deliberately absent: it reaches whatever its base_url points at.

    The two classes here each speak one platform's auth and URL scheme, so the class fixes the
    provider. A plain AsyncOpenAI does not: pointing its base_url at another vendor's
    OpenAI-compatible endpoint is how Groq, DeepSeek, and xAI are reached, all of them
    gen_ai.provider.name values.
    Mapping AsyncOpenAI to "openai" would make __init__ raise for every one of them.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        pricing: Mapping[OpenAIPricedServiceTier, PricingTable],
        provider_name: str,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIServiceTier | None = None,
    ) -> None:
        """Store the SDK client, which owns credentials and endpoints.

        The stored client is a with_options(max_retries=0) copy: langchaint's retry loop owns all retrying,
        counts every request as an attempt, and feeds rate-limit errors to the RateLimiter,
        so the SDK must never retry beneath it.

        reasoning_summary asks the API for readable text, which arrives on each
        ReasoningTrace.text and on the traced conversation; None sends no summary field and leaves
        the provider default in place. A model may return no summary even when one is requested.
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
        False leaves the parameter unsent under either binding value, so a model that cannot be
        told to stop caching is sent no caching parameter instead of one it does not document,
        and automatic_prompt_caching stays a binding parameter every model accepts.
        It has no default because a wrong value is silent either way: True on a model without the
        parameter risks a rejected request, False on one with it leaves caching running for a
        caller who declined it, at whatever that model charges for it. openai_model reads the
        value from PROMPT_CACHE_OPTIONS_MODELS; openai_bedrock_model requires it from its own
        caller, having no catalog of Bedrock ids to read. It is a parameter here rather than a
        lookup on model because model is a str whose namespace this adapter cannot know: it
        serves the platforms provider_name_by_client_class maps and every OpenAI-compatible
        endpoint a base AsyncOpenAI's base_url reaches.

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
        if _DEFAULT_TIER not in pricing:
            raise ValueError(
                f"pricing for model {model!r} has no {_DEFAULT_TIER!r} key; "
                f"it prices every response that reports no service tier, so it is required"
            )
        super().__init__(client=client, model=model, provider_name=provider_name)
        self.client = client.with_options(max_retries=0)
        self.pricing = pricing
        self.supports_prompt_cache_options = supports_prompt_cache_options
        self.reasoning_summary = reasoning_summary
        self.service_tier: OpenAIServiceTier | None = service_tier

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
            reasoning=_wire_reasoning(
                binding.inference_params.reasoning_effort, self.reasoning_summary
            ),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            prompt_cache_options=(
                PromptCacheOptions(mode="explicit")
                if self.supports_prompt_cache_options and not binding.automatic_prompt_caching
                else omit
            ),
            service_tier=self.service_tier if self.service_tier is not None else omit,
            include=["reasoning.encrypted_content"],
        )

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output; pure conversion, no I/O."""
        return _BoundOpenAIText(adapter=self, request=self._request(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output validated into response_format; pure conversion, no I/O."""
        return _BoundOpenAIStructured(
            adapter=self,
            request=self._request(binding),
            response_format=response_format,
        )

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Map the SDK exception to rate_limit, transient, invalid_request, or unrecognized.

        A response's status decides, not the SDK exception class: _make_status_error returns a
        specific subclass only for the statuses it lists and the bare APIStatusError for every
        other one (verified against openai 2.45.0), so a class list would silently drop 413, which
        openai maps to no class, and whatever status the provider adds next.
        classification_from_response holds the shared rule; 429 is the only rate-limit status here,
        openai having no counterpart to anthropic's 529.
        503 is not added to match it: the SDK draws no line between it and any other 5xx,
        and transient already retries it.

        APIConnectionError, which APITimeoutError subclasses, carries no response and is transient.
        Anything else the SDK raises is unrecognized, which fails this item without a retry.
        """
        if isinstance(error, openai.APIConnectionError):
            return "transient"
        if not isinstance(error, openai.APIStatusError):
            return "unrecognized"
        return classification_from_response(
            status_code=error.response.status_code,
            headers=error.response.headers,
            rate_limit_statuses=_RATE_LIMIT_STATUSES,
        )

    @override
    def retry_after_seconds(self, error: Exception) -> float | None:
        """Read the server-stated wait from the SDK exception's response headers."""
        if isinstance(error, openai.APIStatusError):
            return retry_after_seconds_from_headers(error.response.headers)
        return None


class _OpenAIStream(AdapterStream):
    """One open Responses stream, backed by the SDK's stream helper."""

    def __init__(self, *, sdk_stream: AsyncResponseStream[Any]) -> None:
        self._sdk_stream = sdk_stream
        self._terminal_response: OpenAIResponse | None = None

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Translate the SDK stream into text chunks and completed tool calls.

        The terminal event's response is kept for final(), which must not call the SDK's get_final_response():
        that raises RuntimeError unless the terminal event is response.completed.

        Yields:
            Stream items; SDK events langchaint does not model (reasoning, built-in tool activity) are dropped.

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
            elif sdk_event.type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                self._terminal_response = sdk_event.response
        if self._terminal_response is None:
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
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        await self._sdk_stream.close()


class _BoundOpenAIText(BoundAdapter[str]):
    """Text-bound adapter: output is the concatenated text of the turn."""

    def __init__(self, *, adapter: OpenAIResponsesAdapter, request: _OpenAIRequest) -> None:
        self._adapter = adapter
        self._request = request

    @override
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Price the response's counters at the tier it reports.

        Raises:
            TypeError: raw is not an openai Response.
        """
        return _normalized_usage(_as_response(raw), pricing=self._adapter.pricing)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Read the turn's text as this binding's output, or report the run openai says failed.

        A failed status means the run did not finish, and response.error names why, so the output
        items hold whatever had been emitted rather than the turn; reporting that as a Response would
        present a fragment as the answer. _provider_failure states which member the error code picks.
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

    @override
    async def send(self, conversation: Sequence[Message]) -> OpenAIResponse:
        """Send one non-streaming request via responses.create."""
        return await self._adapter.client.responses.create(
            model=self._request.model,
            instructions=self._request.instructions,
            max_output_tokens=self._request.max_output_tokens,
            temperature=self._request.temperature,
            reasoning=self._request.reasoning,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            parallel_tool_calls=self._request.parallel_tool_calls,
            prompt_cache_options=self._request.prompt_cache_options,
            service_tier=self._request.service_tier,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream:
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
            service_tier=self._request.service_tier,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
        )
        sdk_stream = await manager.__aenter__()
        return _OpenAIStream(sdk_stream=sdk_stream)


class _BoundOpenAIStructured[ModelT: BaseModel](BoundAdapter[ModelT | None]):
    """Structured-bound adapter: output is the response_format instance validated from the turn's text."""

    def __init__(
        self,
        *,
        adapter: OpenAIResponsesAdapter,
        request: _OpenAIRequest,
        response_format: type[ModelT],
    ) -> None:
        """Precompute the request's text parameter, the JSON-schema format this binding asks for.

        The format is built by the same type_to_text_format_param call responses.parse makes, so the
        request carries what passing text_format would have sent.
        """
        self._adapter = adapter
        self._request = request
        self._response_format = response_format
        self._text: ResponseTextConfigParam = {
            "format": type_to_text_format_param(response_format)
        }

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

        Each member carries assistant_message, so the turn a rejected 200 did produce reaches the
        caller on the failure.
        No member carries a stop reason: each GenerationError subclass fixes it, and _normalized_stop_reason, used
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
        the text is rejected: the member returned for a rejection is one the retry loop can place
        against the attempt it already recorded, where a raise from inside the SDK is not.

        A failed status is rejected even when the text validates: the run did not finish, and
        response.error names why, so an instance built from the fragment it had emitted would be
        presented as the answer. _no_instance reports it as the failure member _provider_failure chose.

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
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Price the response's counters at the tier it reports.

        Raises:
            TypeError: raw is not an openai Response.
        """
        return _normalized_usage(_as_response(raw), pricing=self._adapter.pricing)

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

    @override
    async def send(self, conversation: Sequence[Message]) -> OpenAIResponse:
        """Send one non-streaming request via responses.create.

        The return type omits InvalidRequest: this adapter sends every conversation.
        """
        return await self._adapter.client.responses.create(
            model=self._request.model,
            instructions=self._request.instructions,
            max_output_tokens=self._request.max_output_tokens,
            temperature=self._request.temperature,
            reasoning=self._request.reasoning,
            tools=self._request.tools,
            tool_choice=self._request.tool_choice,
            parallel_tool_calls=self._request.parallel_tool_calls,
            prompt_cache_options=self._request.prompt_cache_options,
            service_tier=self._request.service_tier,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
            text=self._text,
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream:
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
            service_tier=self._request.service_tier,
            include=self._request.include,
            store=False,
            input=[*self._request.input_prefix, *_wire_input(conversation)],
            text=self._text,
        )
        sdk_stream = await manager.__aenter__()
        return _OpenAIStream(sdk_stream=sdk_stream)
