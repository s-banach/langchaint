"""OTel span tracing for langchaint, as a thin wrapper that never fakes an event boundary it traces.

TracedLLM wraps an LLM and mirrors bind / rebind so every binding stays traced;
TracedBoundLLM wraps a BoundLLM and opens one span per generate call (CLIENT for generate_one,
which wraps one outbound call, INTERNAL for generate_many's aggregate span, which makes no call of its own);
TracedStreamHandle wraps a StreamHandle and opens one CLIENT span across the stream's life.
TracedToolManager is a ToolManager whose every dispatch opens one INTERNAL execute_tool span;
it is a subclass, not a wrapper, so it passes to LLM.bind's tool_manager parameter as one object,
and dispatch_many gains per-call spans by inheritance because it runs through self.dispatch.
Every Traced class accepts extra_attributes, a constant mapping set on each span it opens at span start
(an agent name for cross-trace aggregation, a deployment tag);
an attribute set at completion (a mapper's, an outcome's) wins a key collision.
Every Traced class also requires capture_message_content, which decides whether the spans carry
the conversation itself; it has no default, because recording prompts is a privacy choice the library never makes.

The attributes each span kind carries, capture_message_content True included:

generate_one (CLIENT) and the stream span (CLIENT): gen_ai.provider.name, gen_ai.request.model,
gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, gen_ai.usage.reasoning.output_tokens,
gen_ai.usage.cache_read.input_tokens, gen_ai.usage.cache_creation.input_tokens,
gen_ai.response.finish_reasons, langchaint.attempts, langchaint.cost_in_usd,
and under capture gen_ai.system_instructions, gen_ai.tool.definitions, gen_ai.input.messages,
and gen_ai.output.messages.
The stream span adds gen_ai.response.time_to_first_chunk.
generate_many (INTERNAL): langchaint.batch_item_count and the extra_attributes, nothing else;
it has no single result to map and no single conversation to capture.
execute_tool (INTERNAL): gen_ai.tool.name, gen_ai.tool.call.id, error.type on a failure,
and under capture gen_ai.tool.call.arguments and gen_ai.tool.call.result.
Every span kind carries error.type when it ends on an exception.
langchaint.* is the prefix for the attributes the GenAI convention defines no counterpart for,
which is exactly langchaint.attempts, langchaint.cost_in_usd, and langchaint.batch_item_count.
The langchaint.attempt_failed span event carries error_text and elapsed_seconds per failed attempt.

The wrapper owns the span lifecycle (lazy start, exactly-once end, error status on every exception path)
and never fakes an event boundary a span is supposed to measure:
TracedStreamHandle iterates so it can record gen_ai.response.time_to_first_chunk
and close the span on a failing or abandoned stream, rather than delegating the iteration it needs to witness.
generate_many is the exception by design: it is a bulk convenience,
so it delegates to BoundLLM.generate_many under one aggregate span
and leaves per-item detail to the returned rows (to_row).
The mapper owns only attribute names and values, the part that varies by convention;
a mapper cannot change the span name, kind, or status, and a raising mapper is caught and logged, never propagated.

Importing this subpackage requires opentelemetry-api;
the import below raises a ModuleNotFoundError naming the package to install.
The wrapper imports only opentelemetry-api, so a production app installs the api and wires its own SDK.

Two sources at two revisions verify what this module emits, and neither is asserted from memory.
Attribute key names are verified against opentelemetry-semantic-conventions 0.64b0,
which opentelemetry-sdk requires exactly, so the test suite (which needs the sdk anyway)
asserts every gen_ai.* literal below against that revision's constants.
The chat-completion operation value is "chat" (GenAiOperationNameValues.CHAT);
the tool-execution operation value is "execute_tool" (GenAiOperationNameValues.EXECUTE_TOOL),
and the tool span's identity keys are gen_ai.tool.name and gen_ai.tool.call.id.
The pinned revision defines no reasoning message part;
reasoning appears only as the counter gen_ai.usage.reasoning.output_tokens.
Content payload shapes are verified against the JSON schemas in open-telemetry/semantic-conventions-genai at main,
fetched 2026-07-20; GenAI moved out of open-telemetry/semantic-conventions,
so those schemas are not in the tree the pinned package ships and cannot be pinned in pyproject.toml.
A convention change is a deliberate edit to this module.
"""

import importlib.metadata
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, overload, override

from pydantic import BaseModel

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
except ModuleNotFoundError as exc:
    if exc.name is not None and not exc.name.startswith("opentelemetry"):
        raise
    raise ModuleNotFoundError(
        "langchaint's tracing subpackage requires opentelemetry-api; install opentelemetry-api."
    ) from exc

from langchaint.exceptions import GenerationError
from langchaint.inference_params import InferenceParams
from langchaint.llm import (
    LLM,
    UNCHANGED,
    BoundLLM,
    SequenceNotStr,
    Unchanged,
)
from langchaint.messages import (
    Message,
    Part,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnElement,
    UserMessage,
)
from langchaint.provider import Binding, Provider, StreamItem, ToolChoice
from langchaint.rate_limiter import RateLimiter
from langchaint.response import Response
from langchaint.streaming import StreamHandle
from langchaint.tools import (
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchOutcome,
    Tool,
    ToolManager,
    ToolSchema,
)

type SpanAttributes = Mapping[str, str | bool | int | float | Sequence[str]]
"""A span's attributes, keyed by name.

The value union is the subset of OTel's AttributeValue this package emits;
the Sequence[str] arm exists because OTel attribute values include homogeneous string arrays
and the GenAI convention's finish-reason key gen_ai.response.finish_reasons is one.
"""

type AttributeMapper = Callable[[Response[object] | GenerationError], SpanAttributes]
"""Maps one generate result to its span attributes.

The parameter is Response[object] | GenerationError
because the mapper reads the shared Response/GenerationError fields; Response[object] accepts any Response[OutputT]
because Response's OutputT is inferred covariant (frozen dataclass, PEP 695 inference).
No mapper receives the conversation, so gen_ai_attributes cannot put a prompt on a span.
A custom mapper is bounded only by what it reaches on the result, which includes raw, the SDK response object
held by reference; openai 2.45.0's response model declares an instructions field,
which is where a str system_prompt is sent.
Capturing prompt content is the capture_message_content parameter, which the wrapper applies itself
because the wrapper already has the conversation in scope as a method argument.
"""

_PACKAGE_VERSION = importlib.metadata.version("langchaint")
_CHAT_OPERATION = "chat"
"""The GenAI operation value for a chat completion (GenAiOperationNameValues.CHAT)."""

_EXECUTE_TOOL_OPERATION = "execute_tool"
"""The GenAI operation value for a tool execution (GenAiOperationNameValues.EXECUTE_TOOL)."""

_logger = logging.getLogger("langchaint.tracing")


@dataclass(frozen=True, kw_only=True)
class _SpanConfig:
    """The tracing configuration TracedLLM resolves once, carried by every span descended from it.

    These four values are identical at every step of TracedLLM -> TracedBoundLLM -> TracedStreamHandle,
    so they travel as one object rather than as four parallel parameters restated at each constructor.
    TracedToolManager holds its own and does not take one: it is constructed by the application, not by
    a TracedLLM, and it has no attribute_mapper seam.
    """

    tracer: Tracer
    attribute_mapper: AttributeMapper
    extra_attributes: SpanAttributes
    capture_message_content: bool


_CONVENTION_FINISH_REASONS: Mapping[StopReason, str] = {
    "end_turn": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
}
"""The StopReason values with an exact counterpart in the convention's finish-reason vocabulary.

refusal and other are absent deliberately and pass through unmapped:
the convention's content_filter means a provider filter blocked content, not a model declining,
and no value corresponds to other.
The convention's enum is open (the output schema types the field as the enum or a string),
so passing a value through keeps the emitted set honest rather than forcing a wrong member.
"""


def _finish_reason(stop_reason: StopReason) -> str:
    """Map a StopReason onto the convention's finish-reason vocabulary, passing unmapped values through.

    One function for both places a finish reason is emitted, gen_ai.response.finish_reasons
    and the per-message finish_reason inside gen_ai.output.messages,
    so one span cannot carry two spellings of one concept.
    """
    return _CONVENTION_FINISH_REASONS.get(stop_reason, stop_reason)


def gen_ai_attributes(result: Response[object] | GenerationError) -> SpanAttributes:
    """Map a generate result to GenAI-convention span attributes plus langchaint scalars.

    It is the default attribute_mapper, and public so a custom AttributeMapper extends it
    instead of restating its keys, for a value derived from the result that the keys below do not carry:
    {**gen_ai_attributes(result), "app.request_seconds": sum(a.elapsed_seconds for a in result.attempt_records)}.
    An extending key belongs in the application's own namespace, not under langchaint.*, which is reserved
    for the keys listed in the module docstring and can grow.
    A constant needs no mapper; extra_attributes sets one on every span.
    Each call builds and returns a fresh dict, so extending the result mutates nothing shared.
    Reads only the shared Response/GenerationError fields, so it cannot leak a prompt and cannot meaningfully fail.
    A key stays under the langchaint.* prefix only where the GenAI convention defines no counterpart,
    which is langchaint.attempts and langchaint.cost_in_usd here.
    The cache counters are the convention's own: gen_ai.usage.input_tokens includes cached tokens
    ("This value SHOULD include all types of input tokens, including cached tokens"),
    which is exactly Usage.input_tokens_total, and each of gen_ai.usage.cache_read.input_tokens
    and gen_ai.usage.cache_creation.input_tokens is a part of it ("The value SHOULD be included in
    gen_ai.usage.input_tokens").
    No cache_none counter is emitted: it is the total minus the other two, a subtraction any consumer can do,
    and Usage derives it by subtraction on the openai path, so emitting it would re-export a derived value
    in a shape implying it was measured.
    gen_ai.response.finish_reasons is the plural array the convention defines, carrying the mapped
    finish reason rather than the raw StopReason so it agrees with gen_ai.output.messages;
    it is omitted when stop_reason is None (no completed turn).
    The usage and cost attributes are the call's paid totals across every attempt (result.usage is that scope),
    not one request's counts; per-attempt detail stays visible as the langchaint.attempt_failed span events.
    """
    usage = result.usage
    attributes: dict[str, str | bool | int | float | Sequence[str]] = {
        "gen_ai.provider.name": result.provider_name,
        "gen_ai.request.model": result.model,
        "gen_ai.usage.input_tokens": usage.input_tokens_total,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "gen_ai.usage.reasoning.output_tokens": usage.output_tokens_reasoning,
        "gen_ai.usage.cache_read.input_tokens": usage.input_tokens_cache_read,
        "gen_ai.usage.cache_creation.input_tokens": usage.input_tokens_cache_write,
        "langchaint.attempts": result.attempts,
        "langchaint.cost_in_usd": usage.cost_in_usd,
    }
    if result.stop_reason is not None:
        attributes["gen_ai.response.finish_reasons"] = [_finish_reason(result.stop_reason)]
    return attributes


def _content_parts(content: str | tuple[Part, ...]) -> list[dict[str, object]]:
    """Render a MessageContent as the convention's parts array.

    A str becomes a one-element text array rather than staying a bare string,
    so a content-carrying key holds one shape on every call and no consumer sniffs the type before reading it.
    An ImagePart becomes {"type": "blob", "mime_type": ...} with the bytes dropped:
    an image is routinely megabytes and base64 in a span attribute can dwarf the span itself,
    and the schema's GenericPart arm permits it (it requires only type and allows additional properties).
    Image bytes therefore never appear in a trace.
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    parts: list[dict[str, object]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "content": part.text})
        else:
            parts.append({"type": "blob", "mime_type": part.media_type})
    return parts


def _turn_parts(turn: tuple[TurnElement, ...]) -> list[dict[str, object]]:
    """Render an assistant turn as the convention's parts array, in emission order.

    ReasoningTrace elements are skipped: the payload is the producing SDK item's model_dump,
    opaque by construction (an anthropic signature that may be redacted, an openai encrypted_content),
    so shipping it buys a reader nothing, and the pinned revision defines no reasoning message part to put it in.
    A turn holding only reasoning therefore renders as an empty parts array, not as a missing message.
    args_json rides through unparsed, so the arguments reach the backend exactly as the model produced them.
    """
    parts: list[dict[str, object]] = []
    for element in turn:
        if isinstance(element, TextPart):
            parts.append({"type": "text", "content": element.text})
        elif isinstance(element, ToolCall):
            parts.append({
                "type": "tool_call",
                "id": element.id,
                "name": element.name,
                "arguments": element.args_json,
            })
    return parts


def _conversation_messages(conversation: str | Sequence[Message]) -> list[dict[str, object]]:
    """Render a conversation as the convention's message array.

    A bare str conversation is the one-user-message form BoundLLM accepts, and renders as that message.
    A ToolMessage becomes a tool_call_response part inside a tool-role message,
    the shape the schema specifies rather than the package's own.
    """
    if isinstance(conversation, str):
        return [{"role": "user", "parts": [{"type": "text", "content": conversation}]}]
    return [_message(message) for message in conversation]


def _message(message: Message) -> dict[str, object]:
    """Render one Message as the convention's {role, parts} shape."""
    if isinstance(message, UserMessage):
        return {"role": "user", "parts": _content_parts(message.content)}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "parts": [_tool_call_response_part(message)]}
    return {"role": "assistant", "parts": _turn_parts(message.turn)}


def _tool_call_response_part(message: ToolMessage) -> dict[str, object]:
    """Render one ToolMessage as the convention's tool_call_response part.

    One tool result reaches a backend under this one shape from both spans that report it:
    inside gen_ai.input.messages on a generate span, and as gen_ai.tool.call.result on a tool span.
    """
    return {
        "type": "tool_call_response",
        "id": message.tool_call_id,
        "response": _content_parts(message.content),
    }


def _system_instructions(system_prompt: str | tuple[TextPart, ...]) -> list[dict[str, object]]:
    """Render a bound system prompt as the convention's instruction array.

    A bound str is one element and bound TextParts are one element each;
    cache_breakpoint is a wire-level caching mark with no convention counterpart and is not emitted.
    """
    if isinstance(system_prompt, str):
        return [{"type": "text", "content": system_prompt}]
    return [{"type": "text", "content": part.text} for part in system_prompt]


def _tool_definitions(tool_schemas: tuple[ToolSchema, ...]) -> list[dict[str, object]]:
    """Render the bound tool schemas as the convention's tool-definition array.

    description and parameters are populated although the schema marks both NOT RECOMMENDED by default
    on size grounds: a tool list without its argument schemas does not record what the model was offered,
    which is the question the attribute exists to answer.
    This is a deliberate departure from that recommendation.
    """
    return [
        {
            "type": "function",
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.args_schema,
        }
        for schema in tool_schemas
    ]


def _input_content_attributes(
    binding: Binding, conversation: str | Sequence[Message]
) -> dict[str, str | bool | int | float | Sequence[str]]:
    """Build the input-side content attributes for one call, each a JSON string.

    OTel attribute values cannot nest, and the schemas say a span MAY record these as a JSON string
    when structured form is unsupported.
    A key whose source is empty or absent is omitted rather than emitted as [] or null,
    so a bound system_prompt of None omits gen_ai.system_instructions and no bound tools omits
    gen_ai.tool.definitions; a backend consequently cannot tell "no tools bound" from "capture off"
    by the attribute alone.
    """
    attributes: dict[str, str | bool | int | float | Sequence[str]] = {}
    if binding.system_prompt is not None:
        attributes["gen_ai.system_instructions"] = json.dumps(
            _system_instructions(binding.system_prompt)
        )
    if binding.tool_schemas:
        attributes["gen_ai.tool.definitions"] = json.dumps(_tool_definitions(binding.tool_schemas))
    messages = _conversation_messages(conversation)
    if messages:
        attributes["gen_ai.input.messages"] = json.dumps(messages)
    return attributes


def _output_content_attributes(
    response: Response[object],
) -> dict[str, str | bool | int | float | Sequence[str]]:
    """Build gen_ai.output.messages from a successful Response.

    Only Response carries an assistant turn, so a failed call records the input attributes and no output key;
    there is no assistant turn to record.
    """
    return {
        "gen_ai.output.messages": json.dumps([
            {
                "role": "assistant",
                "parts": _turn_parts(response.assistant_message.turn),
                "finish_reason": _finish_reason(response.stop_reason),
            }
        ])
    }


def _record_attempt_failed_events(span: Span, result: Response[object] | GenerationError) -> None:
    """Add one langchaint.attempt_failed event per failed attempt in the result's records.

    Each event carries the attempt's error text and its own elapsed_seconds;
    events are stamped at recording time and need no wall-clock origin (the records carry only monotonic brackets).
    They answer the first question a slow traced call raises: was it the request or the retries.
    """
    for record in result.attempt_records:
        if record.error is not None:
            span.add_event(
                "langchaint.attempt_failed",
                {"error_text": str(record.error), "elapsed_seconds": record.elapsed_seconds},
            )


def _apply_result_attributes(
    span: Span,
    result: Response[object] | GenerationError,
    attribute_mapper: AttributeMapper,
) -> None:
    """Set the mapper's attributes and the langchaint.attempt_failed events on a recording span.

    Called on the success and the GenerationError paths, both of which carry the shared Response/GenerationError fields;
    never on the other-exception path, which has no such fields.
    Skipped entirely when the span is not recording (no TracerProvider configured, a sampler drop, or an ended span),
    the OTel guard for not computing attributes a non-recording span discards;
    the guard matters because a user AttributeMapper can be arbitrarily expensive.
    A mapper exception is caught and logged at warning level and never propagated,
    so a telemetry bug never discards a paid result; the langchaint.attempt_failed events are added first,
    so they survive a raising mapper, and the span keeps whatever attributes were already set.
    The events are caught under their own guard rather than the mapper's,
    so an error whose str() raises leaves the events partial and the mapper's attributes still set.
    """
    if not span.is_recording():
        return
    try:
        _record_attempt_failed_events(span, result)
    except Exception:
        _logger.warning(
            "attempt_failed events raised; leaving span events partial", exc_info=True
        )
    try:
        attributes = attribute_mapper(result)
    except Exception:
        _logger.warning(
            "attribute_mapper raised; leaving span attributes partial", exc_info=True
        )
        return
    span.set_attributes(attributes)


def _apply_content_attributes(span: Span, build: Callable[[], SpanAttributes]) -> None:
    """Set built content attributes on a recording span, catching a failure to build them.

    The content keys are JSON strings, and some of what they serialize is arbitrary application data:
    a JSONSchemaTool args_schema is Mapping[str, object] the application supplies verbatim,
    so a value json.dumps cannot serialize reaches this module and raises.
    The build is caught and logged at warning level and never propagated, the same way a raising
    AttributeMapper is, so a telemetry defect never breaks a paid call or discards its result;
    the span keeps whatever attributes were already set.
    Building inside the is_recording guard is why the conversation is serialized here rather than earlier:
    an application with no configured TracerProvider gets non-recording no-op spans and pays nothing.
    """
    if not span.is_recording():
        return
    try:
        attributes = build()
    except Exception:
        _logger.warning(
            "content capture raised; leaving span content attributes partial", exc_info=True
        )
        return
    span.set_attributes(attributes)


def _apply_extra_attributes(span: Span, extra_attributes: SpanAttributes) -> None:
    """Set the constant extra_attributes on a just-started span, when recording and non-empty.

    Applied at span start, so the attributes are present however the span later ends;
    an attribute set at completion (a mapper's, a dispatch outcome's) is set after these
    and wins a key collision.
    """
    if extra_attributes and span.is_recording():
        span.set_attributes(extra_attributes)


def _set_generation_error_status(span: Span, error: GenerationError) -> None:
    """Set error.type and error status from a terminal GenerationError, whose attributes are set separately.

    error.type is the exception's class name, so the GenerationError leaves (RetriesExhaustedError,
    RefusalError, MaxCompletionTokensExceededError) are groupable by kind rather than only by the
    error_text message string.
    """
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(Status(StatusCode.ERROR, error.error_text))


def _record_other_exception(span: Span, exc: Exception) -> None:
    """Record a non-GenerationError exception, set error.type and error status; no shared-field attributes exist.

    error.type is the exception's class name, the convention's low-cardinality classification of how an operation
    ended, which gives every span kind here a groupable failure signal (AbortBatchError, StreamProtocolError,
    a tool function's own exception) beside the message string record_exception carries.
    """
    span.record_exception(exc)
    span.set_attribute("error.type", type(exc).__name__)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


class TracedLLM:
    """Wraps an LLM so every binding it produces is traced.

    Wrapping is unconditional: an app wraps every LLM at construction and types its signatures as the Traced classes.
    Enabling, disabling, or routing tracing is OTel SDK configuration (a TracerProvider, a sampler, an exporter),
    never an application code change; an app that never configures an SDK gets non-recording no-op spans.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        capture_message_content: bool,
        attribute_mapper: AttributeMapper = gen_ai_attributes,
        extra_attributes: SpanAttributes | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Resolve the tracer once, at construction.

        capture_message_content True puts the bound system prompt, the bound tool definitions, the conversation,
        and the assistant turn on every span this LLM's bindings open.
        It is required and has no default: recording prompts is a privacy choice the library never makes for the user,
        the way automatic_prompt_caching is a billing choice bind never makes.
        The convention takes the same position, that instrumentations SHOULD NOT capture content by default
        but SHOULD provide an opt-in; requiring the keyword is stricter, in the safe direction.
        The value propagates to every binding and every stream handle, and rebind carries it unchanged,
        so a rebound object cannot silently gain or lose capture.
        tracer None resolves trace.get_tracer("langchaint.tracing", <package version>) now, not at import.
        attribute_mapper is passed down unchanged to every binding; it defaults to gen_ai_attributes,
        the OTel GenAI semantic convention at the revision the module docstring pins.
        extra_attributes is a constant mapping set at span start on every span every binding opens
        (an agent name for cross-trace aggregation, a deployment tag); None means no such attributes.
        A key the mapper also emits resolves to the mapper's value, set at completion.
        """
        self._llm = llm
        self._span_config = _SpanConfig(
            tracer=(
                tracer
                if tracer is not None
                else trace.get_tracer("langchaint.tracing", _PACKAGE_VERSION)
            ),
            attribute_mapper=attribute_mapper,
            extra_attributes=extra_attributes if extra_attributes is not None else {},
            capture_message_content=capture_message_content,
        )

    @property
    def provider(self) -> Provider:
        """The wrapped LLM's provider, so an app never reaches for a private field."""
        return self._llm.provider

    @property
    def rate_limiter(self) -> RateLimiter:
        """The wrapped LLM's RateLimiter, so an app sharing a budget never reaches for a private field."""
        return self._llm.rate_limiter

    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: ToolManager | None = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        automatic_prompt_caching: bool,
    ) -> "TracedBoundLLM[ModelT]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: ToolManager | None = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        automatic_prompt_caching: bool,
    ) -> "TracedBoundLLM[str]": ...
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tool_manager: ToolManager | None = None,
        response_format: type[BaseModel] | None = None,
        inference_params: InferenceParams | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        automatic_prompt_caching: bool,
    ) -> "TracedBoundLLM[Any]":
        """Mirror LLM.bind and wrap its BoundLLM in a TracedBoundLLM carrying this tracer and mapper.

        The overloads re-declare LLM.bind's response_format split
        so a bound structured model gives TracedBoundLLM[Model] and an absent one gives TracedBoundLLM[str].
        """
        return TracedBoundLLM(
            bound_llm=self._llm.bind(
                system_prompt=system_prompt,
                tool_manager=tool_manager,
                response_format=response_format,
                inference_params=inference_params,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                automatic_prompt_caching=automatic_prompt_caching,
            ),
            span_config=self._span_config,
        )


class TracedBoundLLM[OutputT]:
    """Wraps a BoundLLM so every generate call opens a span.

    generate_one opens one CLIENT span (one outbound call);
    generate_many opens one INTERNAL span around the delegated batch (no per-item child spans,
    no outbound call of its own).
    The span name is the GenAI convention {operation} {model}, wrapper-owned,
    so a custom mapper changes attributes only, never the name, kind, or status.
    There is no langchaint.elapsed_seconds attribute:
    the span brackets the same interval elapsed_seconds measures (request start to completion,
    RateLimiter slot waits and backoff included), so the span's own duration already carries it.
    """

    def __init__(self, *, bound_llm: BoundLLM[OutputT], span_config: _SpanConfig) -> None:
        """Store the wrapped BoundLLM and the span configuration; compute the span name once.

        span_config is TracedLLM's, unchanged; TracedLLM documents what each of its values means.
        """
        self._bound_llm = bound_llm
        self._span_config = span_config
        self._span_name = f"{_CHAT_OPERATION} {bound_llm.provider.model}"

    def _apply_input_content(self, span: Span, conversation: str | Sequence[Message]) -> None:
        """Set the input-side content attributes on a just-started span, when capture is on and it is recording.

        Set at span start alongside extra_attributes, so they are present however the span ends,
        including on the paths that raise.
        """
        if self._span_config.capture_message_content:
            _apply_content_attributes(
                span, lambda: _input_content_attributes(self._bound_llm.binding, conversation)
            )

    def _apply_output_content(self, span: Span, response: Response[OutputT]) -> None:
        """Set gen_ai.output.messages from a successful Response, when capture is on and the span is recording."""
        if self._span_config.capture_message_content:
            _apply_content_attributes(span, lambda: _output_content_attributes(response))

    @property
    def provider(self) -> Provider:
        """The wrapped BoundLLM's provider."""
        return self._bound_llm.provider

    @property
    def binding(self) -> Binding:
        """The wrapped BoundLLM's frozen binding."""
        return self._bound_llm.binding

    @property
    def response_format(self) -> type[OutputT] | None:
        """The wrapped BoundLLM's response_format."""
        return self._bound_llm.response_format

    @property
    def tool_manager(self) -> ToolManager | None:
        """The wrapped BoundLLM's ToolManager, so the manual tool loop reads it as it read bound.tool_manager."""
        return self._bound_llm.tool_manager

    @property
    def rate_limiter(self) -> RateLimiter:
        """The wrapped BoundLLM's RateLimiter."""
        return self._bound_llm.rate_limiter

    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_manager: ToolManager | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "TracedBoundLLM[NewModelT]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_manager: ToolManager | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "TracedBoundLLM[str]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_manager: ToolManager | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "TracedBoundLLM[OutputT]": ...
    def rebind(
        self,
        *,
        response_format: type[BaseModel] | None | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tool_manager: ToolManager | None | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        inference_params: InferenceParams | Unchanged = UNCHANGED,
        automatic_prompt_caching: bool | Unchanged = UNCHANGED,
    ) -> "TracedBoundLLM[Any]":
        """Mirror BoundLLM.rebind and re-wrap the plain BoundLLM in a TracedBoundLLM.

        rebind is mirrored so a rebound object stays traced;
        a traced object whose rebind returned an untraced one would silently drop tracing,
        the worst failure mode available here.
        The three overloads re-declare BoundLLM.rebind's response_format split.
        """
        return TracedBoundLLM(
            bound_llm=self._bound_llm.rebind(
                response_format=response_format,
                system_prompt=system_prompt,
                tool_manager=tool_manager,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                inference_params=inference_params,
                automatic_prompt_caching=automatic_prompt_caching,
            ),
            span_config=self._span_config,
        )

    async def generate_one(self, conversation: str | Sequence[Message]) -> Response[OutputT]:
        """Open a span around the whole generate_one call, delegate, attribute, and end the span.

        The span brackets the same interval as elapsed_seconds (slot waits and backoff included).
        A GenerationError sets error status and the shared-field attributes and re-raises;
        any other exception sets error status and record_exception and re-raises;
        a success sets OK status and the attributes.
        Under capture_message_content the input attributes are set at span start, so they are present on the
        failing paths too; gen_ai.output.messages is set only on success, GenerationError carrying no
        assistant turn to record.
        The span ends exactly once, in the finally.

        Raises:
            GenerationError: the wrapped generate_one raised a terminal per-item result (retries exhausted,
                a refusal, or a truncation); the span is attributed and closed first.
            AbortBatchError: the wrapped generate_one classified an error as abort;
                the span records the exception and closes first.
        """
        span = self._span_config.tracer.start_span(self._span_name, kind=SpanKind.CLIENT)
        try:
            _apply_extra_attributes(span, self._span_config.extra_attributes)
            self._apply_input_content(span, conversation)
            try:
                response = await self._bound_llm.generate_one(conversation)
            except GenerationError as exc:
                _apply_result_attributes(span, exc, self._span_config.attribute_mapper)
                _set_generation_error_status(span, exc)
                raise
            except Exception as exc:
                _record_other_exception(span, exc)
                raise
            _apply_result_attributes(span, response, self._span_config.attribute_mapper)
            self._apply_output_content(span, response)
            span.set_status(Status(StatusCode.OK))
            return response
        finally:
            span.end()

    async def generate_many(
        self,
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = False,
    ) -> list[Response[OutputT] | GenerationError]:
        """Order-aligned batch under one INTERNAL span; per-item detail lives in the returned rows.

        warm_cache passes through to BoundLLM.generate_many, which documents it;
        the span brackets the warming first item and the rest alike.

        Delegates to BoundLLM.generate_many rather than re-gathering into per-item child spans:
        generate_many is a bulk convenience (dataset passes, evals),
        not the agent loop, which traces individual generate_one / stream_one turns;
        a batch user reads per-item cost and timing from the returned Response / GenerationError rows (to_row),
        for which spans add nothing.
        The one span here brackets the whole batch's wall time;
        kind is INTERNAL because it makes no outbound call of its own, the delegated items do;
        a CLIENT batch span would register as a phantom outbound call in APM dependency graphs.
        The mapper is not invoked (there is no single result to map);
        the span carries the extra_attributes and langchaint.batch_item_count, the number of conversations
        (a langchaint.* attribute because the GenAI convention defines no batch-size key).
        No content attributes are set under any capture_message_content value:
        the span covers a batch with no single conversation and no single assistant turn,
        the same reason the mapper is not invoked here.
        The span stays OK on mixed per-item results, which come back as rows,
        and takes error status only when an AbortBatchError propagates.

        Raises:
            TypeError: conversations is a bare str (the whole-batch guard, in the delegated method).
            AbortBatchError: one item classified an error as abort;
                the delegated method cancels the in-flight siblings and the span records the exception
                before re-raising.
        """
        span = self._span_config.tracer.start_span(self._span_name, kind=SpanKind.INTERNAL)
        try:
            _apply_extra_attributes(span, self._span_config.extra_attributes)
            try:
                results = await self._bound_llm.generate_many(conversations, warm_cache=warm_cache)
            except Exception as exc:
                _record_other_exception(span, exc)
                raise
            if span.is_recording():
                span.set_attribute("langchaint.batch_item_count", len(results))
            span.set_status(Status(StatusCode.OK))
            return results
        finally:
            span.end()

    def stream_one(self, conversation: str | Sequence[Message]) -> "TracedStreamHandle[OutputT]":
        """Wrap the BoundLLM's StreamHandle in a TracedStreamHandle; no I/O and no span yet.

        The span opens lazily at the first item or at final(), matching StreamHandle's own contract that
        nothing starts until the stream is first driven.
        The binding and the conversation are passed down rather than rendered here:
        the handle needs them to build its input attributes when its lazy span starts,
        and rendering them here would serialize the conversation unconditionally, including for the
        non-recording spans an application with no configured TracerProvider gets.
        The cost is that the handle holds the conversation for the stream's whole life.
        """
        return TracedStreamHandle(
            stream_handle=self._bound_llm.stream_one(conversation),
            span_config=self._span_config,
            span_name=self._span_name,
            binding=self._bound_llm.binding,
            conversation=conversation,
        )


class TracedStreamHandle[OutputT]:
    """Wraps a StreamHandle, owning one span across the stream's life.

    Items pass through by reference;
    nothing is rewrapped (the no-rewrap rule bans copying data into same-shape containers;
    observing an iterator is unaffected).
    The span opens lazily at the first __anext__ or at final(),
    records gen_ai.response.time_to_first_chunk at the first item,
    takes error status on a failing or abandoned stream, and ends exactly once.
    Under capture_message_content the input content attributes are set when that lazy span starts,
    and gen_ai.output.messages when final() returns a Response.
    """

    def __init__(
        self,
        *,
        stream_handle: StreamHandle[OutputT],
        span_config: _SpanConfig,
        span_name: str,
        binding: Binding,
        conversation: str | Sequence[Message],
    ) -> None:
        """Store the wrapped handle and the span pieces; the span is not started here.

        span_config is the binding's, unchanged; TracedLLM documents what each of its values means.
        binding and conversation are held only to build the input content attributes when the span starts,
        and are read for nothing else.
        """
        self._stream_handle = stream_handle
        self._span_config = span_config
        self._span_name = span_name
        self._binding = binding
        self._conversation = conversation
        self._span: Span | None = None
        self._span_started_at_monotonic_seconds: float | None = None
        self._span_ended = False
        self._first_item_seen = False

    def _ensure_span(self) -> Span:
        """Start the span on first use, recording its start time for gen_ai.response.time_to_first_chunk.

        The input content attributes are built here rather than in stream_one: stream_one opens no span and
        does no I/O by contract, so rendering there would serialize the conversation even for the
        non-recording spans an unconfigured application gets, which _apply_content_attributes skips.
        """
        if self._span is None:
            self._span = self._span_config.tracer.start_span(self._span_name, kind=SpanKind.CLIENT)
            _apply_extra_attributes(self._span, self._span_config.extra_attributes)
            if self._span_config.capture_message_content:
                _apply_content_attributes(
                    self._span, lambda: _input_content_attributes(self._binding, self._conversation)
                )
            self._span_started_at_monotonic_seconds = time.monotonic()
        return self._span

    def _end_span(self) -> None:
        """End the span if one is open and it has not already ended; ends at most once."""
        if self._span is not None and not self._span_ended:
            self._span.end()
            self._span_ended = True

    def _mark_first_item(self, span: Span) -> None:
        """Record the gen_ai.response.time_to_first_chunk attribute on the first item's arrival, once.

        The value is the monotonic seconds from the span's start (the first __anext__,
        which is when the underlying request begins) to the first item.
        The convention defines this key as measured from request issuance;
        the span starts one step earlier, at the first __anext__, so the value here also covers the
        RateLimiter slot wait and any backoff before the request went out.
        That is the interval a caller waited for its first chunk, and the wider origin is stated
        so a reader comparing this against another instrumentation's value knows which way it leans.
        Set only when a first item passes through this iterator,
        so a stream drained by final() without iteration carries no such attribute.
        """
        if self._first_item_seen:
            return
        self._first_item_seen = True
        if span.is_recording() and self._span_started_at_monotonic_seconds is not None:
            span.set_attribute(
                "gen_ai.response.time_to_first_chunk",
                time.monotonic() - self._span_started_at_monotonic_seconds,
            )

    def __aiter__(self) -> "TracedStreamHandle[OutputT]":
        """Return self; the wrapper is its own iterator."""
        return self

    async def __anext__(self) -> StreamItem:
        """Delegate to the inner handle, opening the span lazily and observing the first item and failures.

        StopAsyncIteration passes through without ending the span,
        so a following final() can still set attributes and end it.
        Any other exception ends the span with error status (a failing stream, or a cancellation, does not leak a span).

        Raises:
            StopAsyncIteration: the inner stream is exhausted; the span is left open for final().
            Exception: the inner stream raised (a transient failure after items, an abort, a protocol violation);
                the span records it, takes error status, and ends before the re-raise.
        """
        span = self._ensure_span()
        try:
            item = await self._stream_handle.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as exc:
            _record_other_exception(span, exc)
            self._end_span()
            raise
        except BaseException:
            self._end_span()
            raise
        self._mark_first_item(span)
        return item

    async def __aenter__(self) -> "TracedStreamHandle[OutputT]":
        """Enter the inner handle's context and return self; opening is still deferred."""
        await self._stream_handle.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the inner handle, then end the span if it is still open.

        A never-started span (constructed and abandoned without iterating or calling final()) ends nothing.
        A span already ended by a mid-iteration failure or by final() is left alone.
        An open span abandoned with an in-flight exception (a consuming loop body that raised) records that exception
        and takes error status before ending.
        """
        try:
            await self._stream_handle.__aexit__(exc_type, exc, traceback)
        finally:
            if self._span is not None and not self._span_ended:
                if isinstance(exc, Exception):
                    _record_other_exception(self._span, exc)
                self._end_span()

    async def final(self) -> Response[OutputT]:
        """Drain the inner stream, attribute the span from the Response, and end the span.

        A GenerationError from the inner final() (a structured refusal or truncation detected
        when the SDK parses the assembled message) sets error status and the shared-field attributes and re-raises;
        any other exception records it and takes error status; a success sets OK status and the attributes.
        The span ends exactly once: if a prior final(), a mid-iteration failure, or __aexit__ already ended it,
        this delegates to the inner final() (which re-raises or returns its cached Response)
        without touching the span again.

        Raises:
            GenerationError: the inner final() raised a terminal per-item result (a refusal or a truncation
                on the structured path, or retries exhausted while draining); the span is attributed and closed first.
            AbortBatchError: draining hit an error classified as abort; the span records it and closes.
            StreamProtocolError: the inner stream violated the event contract; the span records it and closes.
        """
        span = self._ensure_span()
        if self._span_ended:
            return await self._stream_handle.final()
        try:
            response = await self._stream_handle.final()
        except GenerationError as exc:
            _apply_result_attributes(span, exc, self._span_config.attribute_mapper)
            _set_generation_error_status(span, exc)
            self._end_span()
            raise
        except Exception as exc:
            _record_other_exception(span, exc)
            self._end_span()
            raise
        except BaseException:
            self._end_span()
            raise
        _apply_result_attributes(span, response, self._span_config.attribute_mapper)
        if self._span_config.capture_message_content:
            _apply_content_attributes(span, lambda: _output_content_attributes(response))
        span.set_status(Status(StatusCode.OK))
        self._end_span()
        return response


def _dispatch_error_type(outcome: DispatchOutcome) -> str | None:
    """Classify a dispatch outcome for error.type, or None where the call succeeded.

    The three values are the documented error list the convention asks instrumentations to publish:
    tool_error (the tool ran and authored a failure), invalid_tool_args, and unknown_tool.
    The latter two are exactly the outcomes where the tool function never ran,
    so error.type IN ("invalid_tool_args", "unknown_tool") separates a model producing calls the tool layer
    rejects from a defect in application tool code, as a closed set rather than a disjunction.
    A raising tool function is classified by _record_other_exception with its exception class name instead.
    """
    if isinstance(outcome, DispatchHandled):
        return "tool_error" if outcome.tool_message.is_error else None
    if isinstance(outcome, DispatchInvalidToolArgs):
        return "invalid_tool_args"
    return "unknown_tool"


class TracedToolManager(ToolManager):
    """A ToolManager whose every dispatch opens one execute_tool span.

    A subclass rather than a TracedLLM-style wrapper, for two reasons the wrapper shape cannot meet:
    LLM.bind types tool_manager as ToolManager, so only a subclass reaches bind and the app's loop
    as one object; and ToolManager.dispatch_many runs through self.dispatch,
    so overriding dispatch alone gives every concurrent call of a batch its own span
    without restating the settle-and-group semantics documented on dispatch_many.
    The span name is "execute_tool {call.name}", the GenAI convention's {operation} {target} pattern;
    kind is INTERNAL because dispatch runs an in-process function
    (a CLIENT span would register as an outbound call this package cannot see being made).
    dispatch makes its span current while the tool function runs (trace.use_span).
    Spans the function starts (an instrumented HTTP request, a nested agent loop) nest under the execute_tool span.
    dispatch_many stays safe: asyncio.gather runs each dispatch in its own task with a copied context.
    Concurrent dispatch spans are therefore siblings, never nested in one another.
    The identity attributes gen_ai.tool.name and gen_ai.tool.call.id are set at span start;
    at completion the span takes its status and error.type from the outcome:

    | dispatch result                     | status | error.type              |
    | ----------------------------------- | ------ | ----------------------- |
    | DispatchHandled, is_error False     | OK     | absent                  |
    | DispatchHandled, is_error True      | ERROR  | tool_error              |
    | DispatchInvalidToolArgs             | ERROR  | invalid_tool_args       |
    | DispatchUnknownTool                 | ERROR  | unknown_tool            |
    | the tool function raised            | ERROR  | the exception class name|

    invalid_tool_args and unknown_tool are the two values meaning the tool function never ran.
    A tool returning is_error True is designed control flow here, not a malfunction: the model reads the failure
    and corrects, and the same holds for the other two failure arms.
    So a healthy agent doing one argument-validation retry emits ERROR spans as a matter of routine,
    and a dashboard reading span status as a health signal will show that.
    That is accepted rather than worked around: error.type is the field an operator filters on,
    and OTel's position is that status describes the operation's outcome rather than the system's health
    (an HTTP client span for a 404 takes ERROR status though the request itself worked).

    Under capture_message_content the span also carries gen_ai.tool.call.arguments at span start and
    gen_ai.tool.call.result at completion.
    gen_ai.tool.call.result carries the tool_message as a tool_call_response part, the same shape
    gen_ai.input.messages carries it in on a generate span, so one tool result reaches a backend under one
    shape from both spans that report it.
    That shape is an object, which is what the key's note asks for ("It's expected to be an object");
    no JSON schema governs the key, so the choice among objects is this module's.
    Its response field is the parts array every other content key uses, so a str content and a
    Sequence[Part] content reach a backend in one shape.
    gen_ai.tool.call.arguments is the model's own argument JSON, passed through unparsed:
    a malformed or non-object args_json is exactly what the DispatchInvalidToolArgs arm exists to report,
    and rewriting it would hide the value that produced the failure the span records.

    gen_ai.tool.call.result is recorded on every arm, including the two where the tool function never ran.
    The convention defines that key as the result "if any and if execution was successful",
    so this is a deliberate departure: on those arms the value is the package-rendered correction the model
    reads and adapts to, which is the payload a reader debugging a tool loop wants, and error.type on the
    same span already says no tool produced it, so a consumer reading both is not misled.
    gen_ai.tool.call.result is what dispatch returned, which is not necessarily what the model read:
    the application owns the loop, so on any arm it may rewrite, replace, or drop the tool_message it
    received before appending it to the conversation.
    The generate span's gen_ai.input.messages then carries different text for that call, and both spans are
    correct, each reporting its own boundary; the difference is the application's edit made visible.
    The two join on the tool call id, which is gen_ai.tool.call.id here and the tool_call_response part's id there.

    There is no attribute mapper seam: the attributes are the fixed keys above,
    and app constants ride in through extra_attributes.
    """

    def __init__(
        self,
        tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        *,
        capture_message_content: bool,
        tracer: Tracer | None = None,
        extra_attributes: SpanAttributes | None = None,
    ) -> None:
        """Index the tools (ToolManager.__init__) and resolve the span pieces once.

        capture_message_content is required and has its own value here, inheriting nothing from TracedLLM:
        this object is constructed by the application and passed to bind, so there is nothing to inherit from.
        It has no default for the reason TracedLLM's does not.
        tracer None resolves trace.get_tracer("langchaint.tracing", <package version>), as on TracedLLM.
        extra_attributes is a constant mapping set at span start on every dispatch span;
        None means no such attributes.
        A key dispatch also sets (an identity or outcome attribute) resolves to the dispatch-set value.

        Raises:
            ValueError: two tools share a name.
        """
        super().__init__(tools)
        self._capture_message_content = capture_message_content
        self._tracer = (
            tracer
            if tracer is not None
            else trace.get_tracer("langchaint.tracing", _PACKAGE_VERSION)
        )
        self._extra_attributes: SpanAttributes = (
            extra_attributes if extra_attributes is not None else {}
        )

    @override
    async def dispatch(self, call: ToolCall) -> DispatchOutcome:
        """Open one execute_tool span around ToolManager.dispatch and attribute it from the outcome.

        The dispatch semantics are the base method's own; the override adds only the span.
        The span is current while the base dispatch runs, so a span the tool function starts nests under it.
        A function exception (a user-code defect) is recorded on the span, sets error status, and propagates.
        trace.use_span ends the span exactly once on exit, with its exception recording and status setting off.
        This method records and sets status itself, from the table on the class.
        """
        span = self._tracer.start_span(
            f"{_EXECUTE_TOOL_OPERATION} {call.name}", kind=SpanKind.INTERNAL
        )
        with trace.use_span(
            span, end_on_exit=True, record_exception=False, set_status_on_exception=False
        ):
            _apply_extra_attributes(span, self._extra_attributes)
            if span.is_recording():
                span.set_attributes({
                    "gen_ai.tool.name": call.name,
                    "gen_ai.tool.call.id": call.id,
                })
                if self._capture_message_content:
                    span.set_attribute("gen_ai.tool.call.arguments", call.args_json)
            try:
                outcome = await super().dispatch(call)
            except Exception as exc:
                _record_other_exception(span, exc)
                raise
            error_type = _dispatch_error_type(outcome)
            if span.is_recording() and error_type is not None:
                span.set_attribute("error.type", error_type)
            if self._capture_message_content:
                _apply_content_attributes(
                    span,
                    lambda: {
                        "gen_ai.tool.call.result": json.dumps(
                            _tool_call_response_part(outcome.tool_message)
                        )
                    },
                )
            if error_type is None:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, error_type))
            return outcome


__all__ = [
    "AttributeMapper",
    "SpanAttributes",
    "TracedBoundLLM",
    "TracedLLM",
    "TracedStreamHandle",
    "TracedToolManager",
    "gen_ai_attributes",
]
