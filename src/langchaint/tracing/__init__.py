"""Trace langchaint calls with OTel spans.

Importing this subpackage requires `opentelemetry-api`.
Applications configure the OTel SDK.

`TracedBoundLLM` opens one CLIENT span for each generation call.
`generate_many` opens one span per started input.
`TracedStreamHandle` keeps one CLIENT span open for the stream.
`TracedToolManager.dispatch` opens one INTERNAL `execute_tool` span.
`ToolManager.dispatch_many` uses `dispatch` and gets one span per tool call.
`precomputed` opens no span because it executes no tool.
`agent_span` opens one INTERNAL `invoke_agent` span around an application agent loop.

Each traced class requires `capture_message_content` because prompt recording is a privacy choice.
`extra_attributes` sets constant attributes when each span starts.
Completion attributes replace matching `extra_attributes` keys.
Required `gen_ai.operation.name` values also replace matching `extra_attributes` keys.

Chat and stream spans use `gen_ai.operation.name="chat"`.
They report provider, request model, response model, finish reasons, token usage, attempts, and cost.
With capture enabled, they report system instructions, tool definitions, input messages, and output messages.
Stream spans also report `gen_ai.response.time_to_first_chunk`.
Tool spans use `gen_ai.operation.name="execute_tool"` and report tool name and tool call id.
With capture enabled, tool spans report arguments and results.
Agent spans use `gen_ai.operation.name="invoke_agent"` and report agent name and `langchaint.agent_path`.
Agent spans sum the run's usage partition and cost.
Agent exit `extra_attributes` add application results.

`langchaint.*` names only attempts, cost, and agent path because the convention has no matching keys.
Each failed attempt adds `langchaint.attempt_failed` with `error_text` and `elapsed_seconds`.

Each traced operation starts and ends its span exactly once.
`TracedStreamHandle` closes failed or abandoned streams.
A mapper may set only attribute names and values.
A mapper cannot change the span name, kind, or status.
Mapper failures are logged and never propagate.

Attribute names match opentelemetry-semantic-conventions 0.64b0.
`GenAiOperationNameValues.CHAT`, `.EXECUTE_TOOL`, and `.INVOKE_AGENT` define the operation values.
Tool identity uses `gen_ai.tool.name` and `gen_ai.tool.call.id`.
Agent identity uses `gen_ai.agent.name`.
Reasoning usage uses `gen_ai.usage.reasoning.output_tokens`.
Readable reasoning uses `ReasoningPart` in content payloads.
Content payloads follow the convention's JSON schemas.
`gen_ai.tool.call.arguments` may carry a JSON value other than an object.
"""

import importlib.metadata
import json
import logging
import math
import time
from collections.abc import Callable, Coroutine, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Never, NoReturn, overload, override

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

from langchaint.adapter import Adapter, Binding, StreamItem, ToolChoice
from langchaint.exceptions import AbandonedCallError, GenerationError
from langchaint.inference_params import InferenceParams
from langchaint.llm import (
    LLM,
    UNCHANGED,
    BoundLLM,
    Deadline,
    GenerationInput,
    Unchanged,
    WallClockDeadline,
)
from langchaint.messages import (
    AssistantMessage,
    ContentPart,
    Message,
    ReasoningPart,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnPart,
    UserMessage,
)
from langchaint.response import CallResult, GenerateResult, Response, ToolCallTurn
from langchaint.sequence_not_str import SequenceNotStr
from langchaint.shared_backoff import SharedBackoff
from langchaint.streaming import StreamHandle
from langchaint.tools import (
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchOutcome,
    Tool,
    ToolManager,
    ToolSchema,
)
from langchaint.usage import Usage

type SpanAttributeValue = str | bool | int | float | list[str] | tuple[str, ...]
"""One span attribute's value."""

type SpanAttributes = Mapping[str, SpanAttributeValue]
"""A span's attributes, keyed by name."""

type AttributeMapper = Callable[[CallResult[object]], SpanAttributes]
"""Maps one generate result to its span attributes.

The parameter is CallResult[object]
because the mapper reads the fields every CallResult variant shares; the object type argument accepts
any OutputT because the success variants' OutputT is inferred covariant (frozen dataclass, PEP 695 inference).
No mapper receives the GenerationInput, so gen_ai_attributes cannot put a prompt on a span.
A custom mapper is bounded only by what it reaches on the result, which includes raw, the SDK response object
held by reference; openai 2.45.0's response model declares an instructions field,
which is where a str system_prompt is sent.
Capturing prompt content is the capture_message_content parameter, which the wrapper applies itself
because the wrapper already has the GenerationInput in scope as a method argument.
"""

_PACKAGE_VERSION = importlib.metadata.version("langchaint")
_CHAT_OPERATION = "chat"
"""The GenAI operation value for a chat completion (GenAiOperationNameValues.CHAT)."""

_EXECUTE_TOOL_OPERATION = "execute_tool"
"""The GenAI operation value for a tool execution (GenAiOperationNameValues.EXECUTE_TOOL)."""

_logger = logging.getLogger("langchaint.tracing")


@contextmanager
def _guarding_telemetry_failures(what: str) -> Generator[None]:
    """Log whatever the block raises instead of letting it out.

    OTel Exception values are logged because they must not replace a result or active exception.
    Application callables use guards that name the callable.
    Only Exception is caught, so a cancellation still reaches the caller.
    """
    try:
        yield
    except Exception:
        _logger.warning("%s raised; this span's telemetry is incomplete", what, exc_info=True)


def _is_recording(span: Span) -> bool:
    """Read whether the span records, treating an Exception as not recording.

    An Exception returns False so telemetry cannot replace a result or active exception.
    """
    try:
        return span.is_recording()
    except Exception:
        _logger.warning(
            "reading whether the span records raised; treating it as not recording", exc_info=True
        )
        return False


def _start_span(tracer: Tracer, name: str, *, kind: SpanKind) -> Span:
    """Start one span, returning a non-recording span when the tracer itself raises.

    INVALID_SPAN lets the traced call continue without telemetry.
    """
    try:
        return tracer.start_span(name, kind=kind)
    except Exception:
        _logger.warning(
            "starting the %s span raised; this call runs untraced", name, exc_info=True
        )
        return trace.INVALID_SPAN


def _end_span(span: Span) -> None:
    """End one span, without letting the end reach the caller.

    opentelemetry-sdk 1.43.0 Span.end does not guard SpanProcessor.on_end.
    This guard preserves the result or active exception.
    """
    with _guarding_telemetry_failures("ending the span"):
        span.end()


def _set_ok_status(span: Span) -> None:
    """Mark one span successful, without letting the call reach the caller."""
    with _guarding_telemetry_failures("setting the span status"):
        span.set_status(Status(StatusCode.OK))


def _set_span_attribute(span: Span, key: str, value: SpanAttributeValue) -> None:
    """Set one attribute on a recording span, without letting the call reach the caller."""
    with _guarding_telemetry_failures(f"setting {key}"):
        if _is_recording(span):
            span.set_attribute(key, value)


def _set_span_attributes(span: Span, attributes: SpanAttributes) -> None:
    """Set a mapping of attributes on a recording span, without letting the call reach the caller."""
    with _guarding_telemetry_failures("setting the span attributes"):
        if attributes and _is_recording(span):
            span.set_attributes(attributes)


@dataclass(frozen=True, kw_only=True)
class _SpanConfig:
    tracer: Tracer
    attribute_mapper: AttributeMapper
    extra_attributes: SpanAttributes
    capture_message_content: bool


def _resolve_traced_tool_manager(
    tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]] | None,
    *,
    span_config: _SpanConfig,
) -> ToolManager | None:
    """Raise `ValueError` when a `tools` sequence contains duplicate names."""
    if isinstance(tools, ToolManager) or tools is None:
        return tools
    return TracedToolManager(
        tools,
        tracer=span_config.tracer,
        extra_attributes=span_config.extra_attributes,
        capture_message_content=span_config.capture_message_content,
    )


_CONVENTION_FINISH_REASONS: Mapping[StopReason, str] = {
    "end_turn": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
}
"""The StopReason values with an exact counterpart in the convention's finish-reason vocabulary.

refusal, context_window_exceeded, and other are absent deliberately and pass through unmapped:
the convention's content_filter means a provider filter blocked content, not a model declining,
and no value corresponds to a context-window overflow or to other.
The convention's enum is open (the output schema types the field as the enum or a string),
so passing a value through keeps the emitted set honest rather than forcing a wrong member.
"""


_NO_COMPLETED_TURN_FINISH_REASON = "error"
"""What gen_ai.output.messages reports for a turn whose result states no stop reason.

The convention's enum member for a generation that failed, which is what a turn on a failure the
provider never finished is. gen_ai.response.finish_reasons omits itself instead, being an optional
top-level attribute; the per-message field is required, so a turn recorded from a failure needs a value.
"""


def _finish_reason(stop_reason: StopReason) -> str:
    """Map a StopReason onto the convention's finish-reason vocabulary, passing unmapped values through.

    gen_ai.response.finish_reasons and gen_ai.output.messages use this mapping.
    """
    return _CONVENTION_FINISH_REASONS.get(stop_reason, stop_reason)


def gen_ai_attributes(result: CallResult[object]) -> SpanAttributes:
    """Map a generate result to GenAI-convention span attributes plus langchaint scalars.

    This is the default attribute_mapper.
    A custom AttributeMapper can extend its result.
    Extension keys must use the application's namespace because langchaint.* is reserved.
    A constant needs no mapper; extra_attributes sets one on every span.
    Each call builds and returns a fresh dict, so extending the result mutates nothing shared.
    Reads only the fields every CallResult variant shares, so it cannot leak a prompt and cannot meaningfully fail.
    A key stays under the langchaint.* prefix only where the GenAI convention defines no counterpart,
    which is langchaint.attempts and langchaint.cost_in_usd here.
    gen_ai.usage.input_tokens is Usage.input_tokens_total.
    The cache-read and cache-creation attributes are parts of that total.
    No cache_none counter is emitted because it is derived.
    gen_ai.response.finish_reasons contains the mapped stop_reason and is omitted when stop_reason is None.
    gen_ai.response.model is the last attempt's model_served and is omitted when unavailable.
    The usage and cost attributes are the call's paid totals across every attempt (result.usage is that scope),
    not one request's counts; per-attempt detail stays visible as the langchaint.attempt_failed span events.
    """
    usage = result.usage
    records = result.attempt_records
    model_served = records[-1].model_served if records else None
    attributes: dict[str, SpanAttributeValue] = {
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
    if model_served is not None:
        attributes["gen_ai.response.model"] = model_served
    if result.stop_reason is not None:
        attributes["gen_ai.response.finish_reasons"] = [_finish_reason(result.stop_reason)]
    return attributes


@contextmanager
def agent_span(
    tracer: Tracer,
    *,
    agent_name: str,
    agent_path: str,
    usage: Callable[[], Usage],
    extra_attributes: Callable[[], SpanAttributes] | None = None,
) -> Generator[Span]:
    """Open one INTERNAL invoke_agent span around an application's own agent loop.

    langchaint ships no agent loop.
    This context manager records an application's loop with identity, paid usage, and cost.

    The usage keys are the same ones gen_ai_attributes emits for a single call, here summed over the run:
    gen_ai.usage.input_tokens is Usage.input_tokens_total.
    reasoning.output_tokens is the reasoning share.
    langchaint.cost_in_usd is estimated cost.

    usage and extra_attributes run once at exit.
    Identity and usage attributes overwrite extra_attributes.
    A usage key is overwritten only when usage returns.

    An Exception from usage or extra_attributes is logged and does not propagate.

    Guarded helpers start and end the span.
    trace.use_span disables its exception and status handling.
    This module records body exceptions and always ends the span.

    Yields:
        The current invoke_agent span.

    Raises:
        Exception: the wrapped body raised. error.type and error status are recorded first.
        BaseException: the body, usage, or extra_attributes raised a non-Exception BaseException.
    """
    identity: dict[str, str] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": agent_name,
        "langchaint.agent_path": agent_path,
    }
    span = _start_span(tracer, f"invoke_agent {agent_name}", kind=SpanKind.INTERNAL)
    _set_span_attributes(span, identity)
    try:
        try:
            with trace.use_span(
                span, end_on_exit=False, record_exception=False, set_status_on_exception=False
            ):
                try:
                    yield span
                except Exception as exc:
                    _record_other_exception(span, exc)
                    raise
        finally:
            _apply_agent_exit_attributes(
                span, identity=identity, usage=usage, extra_attributes=extra_attributes
            )
    finally:
        _end_span(span)


def _apply_agent_exit_attributes(
    span: Span,
    *,
    identity: Mapping[str, str],
    usage: Callable[[], Usage],
    extra_attributes: Callable[[], SpanAttributes] | None,
) -> None:
    """Set an ending invoke_agent span's extras, then re-set its identity, then set its usage.

    Identity overwrites extras. Usage overwrites extras only when usage returns. Each callable has an independent guard.
    The set_attributes calls are guarded for the same reason.
    A non-recording span skips attribute construction.
    """
    if not _is_recording(span):
        return
    if extra_attributes is not None:
        try:
            attributes = extra_attributes()
        except Exception:
            _logger.warning(
                "agent_span extra_attributes raised; leaving span attributes partial",
                exc_info=True,
            )
        else:
            with _guarding_telemetry_failures("setting the agent_span extra attributes"):
                span.set_attributes(attributes)
    with _guarding_telemetry_failures("setting the agent_span identity attributes"):
        span.set_attributes(identity)
    try:
        spent = usage()
    except Exception:
        _logger.warning(
            "agent_span usage raised; leaving span usage attributes unset", exc_info=True
        )
        return
    with _guarding_telemetry_failures("setting the agent_span usage attributes"):
        span.set_attributes({
            "gen_ai.usage.input_tokens": spent.input_tokens_total,
            "gen_ai.usage.output_tokens": spent.output_tokens,
            "gen_ai.usage.reasoning.output_tokens": spent.output_tokens_reasoning,
            "gen_ai.usage.cache_read.input_tokens": spent.input_tokens_cache_read,
            "gen_ai.usage.cache_creation.input_tokens": spent.input_tokens_cache_write,
            "langchaint.cost_in_usd": spent.cost_in_usd,
        })


def _content_parts(content: str | tuple[ContentPart, ...]) -> list[dict[str, object]]:
    """Render a MessageContent as the convention's parts array.

    A str becomes one text part.
    ImagePart and AudioPart become blob metadata without data.
    ImageUrlPart keeps url and its optional media_type.
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    parts: list[dict[str, object]] = []
    for part in content:
        match part.kind:
            case "text":
                parts.append({"type": "text", "content": part.text})
            case "image":
                parts.append({"type": "blob", "mime_type": part.media_type})
            case "image_url":
                image_url: dict[str, object] = {"type": "image_url", "url": part.url}
                if part.media_type is not None:
                    image_url["mime_type"] = part.media_type
                parts.append(image_url)
            case "audio":
                parts.append({"type": "blob", "mime_type": part.media_type})
    return parts


def _finite_float(number_text: str) -> float:
    """Parse a JSON number, rejecting one that overflows the float range.

    json.dumps writes a non-finite float back as the bare token Infinity or NaN, which is not JSON,
    so a value that cannot round-trip is treated as a parse failure instead.

    Raises:
        ValueError: the text parses to a non-finite float (1e400 overflows to inf).
            _tool_call_arguments catches this type to reach its raw-text fallback.
    """
    value = float(number_text)
    if not math.isfinite(value):
        raise ValueError(f"JSON number is not finite as a float: {number_text}")
    return value


def _reject_non_json_constant(token: str) -> NoReturn:
    """Reject the Infinity, -Infinity, and NaN literals json.loads accepts as an extension.

    They are not JSON, and json.dumps writes them straight back out, so they are a parse failure here.

    Raises:
        ValueError: always; json.loads calls this only for those three literals.
            _tool_call_arguments catches this type to reach its raw-text fallback.
    """
    raise ValueError(f"not a JSON constant: {token}")


def _tool_call_arguments(args_json: str) -> object:
    """Deserialize a tool call's argument JSON, falling back to the raw text when it does not parse.

    The convention requests best-effort deserialization of serialized arguments.
    Any JSON value is returned.
    Unparseable text is returned unchanged so DispatchInvalidToolArgs remains visible.

    The two parse hooks narrow json.loads to what json.dumps can write back as JSON.
    RFC 8259 excludes Infinity and NaN.
    The hooks route them to raw text so nested attributes remain valid JSON.
    Routing these to the raw-text fallback keeps every emitted payload standard JSON.

    Only ValueError is caught, the parse failure this fallback is for.
    RecursionError propagates to the telemetry guard.
    """
    try:
        parsed: object = json.loads(
            args_json, parse_float=_finite_float, parse_constant=_reject_non_json_constant
        )
    except ValueError:
        return args_json
    return parsed


def _turn_parts(turn: tuple[TurnPart, ...]) -> list[dict[str, object]]:
    """Render an assistant turn as the convention's parts array, in emission order.

    ReasoningPart and TextPart emit their text.
    Text-free parts emit nothing.
    ReasoningPart.raw is opaque and is never emitted.
    A RawPart renders as nothing because it has no text.
    A turn holding only text-free parts therefore renders as an empty parts array,
    not as a missing message.
    """
    parts: list[dict[str, object]] = []
    for part in turn:
        if isinstance(part, ReasoningPart):
            if part.text:
                parts.append({"type": "reasoning", "content": part.text})
        elif isinstance(part, TextPart):
            if part.text:
                parts.append({"type": "text", "content": part.text})
        elif isinstance(part, ToolCall):
            parts.append({
                "type": "tool_call",
                "id": part.id,
                "name": part.name,
                "arguments": _tool_call_arguments(part.args_json),
            })
    return parts


def _input_messages(generation_input: GenerationInput) -> list[dict[str, object]]:
    """Render a GenerationInput as the convention's message array.

    A bare str is the one-user-message form BoundLLM accepts, and renders as that message.
    A ToolMessage becomes a tool_call_response part inside a tool-role message,
    the shape the schema specifies rather than langchaint's own.
    """
    if isinstance(generation_input, str):
        return [{"role": "user", "parts": [{"type": "text", "content": generation_input}]}]
    return [_message(message) for message in generation_input]


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

    description and parameters record what the model received despite the schema's size warning.
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
    binding: Binding, generation_input: GenerationInput
) -> dict[str, SpanAttributeValue]:
    """Build the input-side content attributes for one call, each a JSON string.

    OTel attribute values cannot nest, so structured values use the permitted JSON string form.
    A key whose source is empty or absent is omitted rather than emitted as [] or null,
    system_prompt None omits gen_ai.system_instructions.
    No bound tools omits gen_ai.tool.definitions.
    These omissions are indistinguishable from disabled capture.
    """
    attributes: dict[str, SpanAttributeValue] = {}
    if binding.system_prompt is not None:
        attributes["gen_ai.system_instructions"] = json.dumps(
            _system_instructions(binding.system_prompt)
        )
    if binding.tool_schemas:
        attributes["gen_ai.tool.definitions"] = json.dumps(_tool_definitions(binding.tool_schemas))
    input_messages = _input_messages(generation_input)
    if input_messages:
        attributes["gen_ai.input.messages"] = json.dumps(input_messages)
    return attributes


def _output_content_attributes(
    assistant_message: AssistantMessage, stop_reason: StopReason | None
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.output.messages from one assistant turn.

    One function for the success and the failure paths, so one turn renders the same whichever reported it.
    One key contains one message for one turn. A missing stop_reason uses the required "error" enum member.
    """
    return {
        "gen_ai.output.messages": json.dumps([
            {
                "role": "assistant",
                "parts": _turn_parts(assistant_message.turn),
                "finish_reason": (
                    _NO_COMPLETED_TURN_FINISH_REASON
                    if stop_reason is None
                    else _finish_reason(stop_reason)
                ),
            }
        ])
    }


def _apply_output_content(
    span: Span, result: CallResult[object], span_config: _SpanConfig
) -> None:
    """Set gen_ai.output.messages from the result's turn, when capture is on and the span is recording.

    GenerationError carries the last produced turn. The key is omitted when no attempt produced a turn.
    Per-attempt detail stays on the langchaint.attempt_failed events, which carry no content.
    """
    if not span_config.capture_message_content:
        return
    assistant_message = result.assistant_message
    if assistant_message is None:
        return
    stop_reason = result.stop_reason
    _apply_content_attributes(
        span, lambda: _output_content_attributes(assistant_message, stop_reason)
    )


def _record_attempt_failed_events(span: Span, result: CallResult[object]) -> None:
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
    result: CallResult[object],
    attribute_mapper: AttributeMapper,
) -> None:
    """Set the mapper's attributes and the langchaint.attempt_failed events on a recording span.

    Called on the success and the GenerationError paths, both of which carry the shared CallResult fields;
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
    if not _is_recording(span):
        return
    try:
        _record_attempt_failed_events(span, result)
    except Exception:
        _logger.warning("attempt_failed events raised; leaving span events partial", exc_info=True)
    try:
        attributes = attribute_mapper(result)
    except Exception:
        _logger.warning("attribute_mapper raised; leaving span attributes partial", exc_info=True)
        return
    with _guarding_telemetry_failures("setting the mapper's attributes"):
        span.set_attributes(attributes)


def _apply_content_attributes(span: Span, build: Callable[[], SpanAttributes]) -> None:
    """Set built content attributes on a recording span, catching a failure to build them.

    The content keys are JSON strings, and some of what they serialize is arbitrary application data:
    a JSONSchemaTool args_schema is Mapping[str, object] the application supplies verbatim,
    so a value json.dumps cannot serialize reaches this module and raises.
    A build Exception is logged and does not propagate. Existing span attributes remain.
    Building inside the is_recording guard is why the GenerationInput is serialized here rather than earlier:
    an application with no configured TracerProvider gets non-recording no-op spans and pays nothing.
    """
    if not _is_recording(span):
        return
    try:
        attributes = build()
    except Exception:
        _logger.warning(
            "content capture raised; leaving span content attributes partial", exc_info=True
        )
        return
    with _guarding_telemetry_failures("setting the content attributes"):
        span.set_attributes(attributes)


def _apply_extra_attributes(span: Span, extra_attributes: SpanAttributes) -> None:
    """Set the constant extra_attributes on a just-started span, when recording and non-empty.

    These attributes apply first. Module-owned attributes win later key collisions.
    """
    _set_span_attributes(span, extra_attributes)


def _apply_operation_name(span: Span, operation_name: str) -> None:
    """Set gen_ai.operation.name on a just-started span.

    The convention requires this attribute on every span kind this module opens.
    Applied at span start, so it is present however the span ends, and after extra_attributes,
    so an application constant cannot displace a required attribute.
    """
    _set_span_attribute(span, "gen_ai.operation.name", operation_name)


def _set_generation_error_status(span: Span, error: GenerationError) -> None:
    """Set error.type and error status from a terminal GenerationError, whose attributes are set separately.

    error.type is the GenerationError subclass name, which groups errors independently of error_text.
    """
    _set_span_attribute(span, "error.type", type(error).__name__)
    with _guarding_telemetry_failures("setting the error status"):
        span.set_status(Status(StatusCode.ERROR, error.error_text))


def _record_other_exception(span: Span, exc: Exception) -> None:
    """Record the exception as a span event, set error.type from its class, and set error status.

    error.type uses the exception class name for low-cardinality grouping.
    Sets no shared-field attributes: this records the exception itself, not a call's result.
    """
    with _guarding_telemetry_failures("recording the exception"):
        span.record_exception(exc)
    _set_span_attribute(span, "error.type", type(exc).__name__)
    with _guarding_telemetry_failures("setting the error status"):
        span.set_status(Status(StatusCode.ERROR, str(exc)))


def _record_stream_conclusion(span: Span, exc: Exception, span_config: _SpanConfig) -> None:
    """Record the exception that concluded a stream, attributing the span by what it is.

    A GenerationError is that call's result, so the span takes the same result attributes, content,
    and status a success variant would give it; anything else is an exception the span only records.
    The open, item pull, and final() use the same recording rule.
    """
    if isinstance(exc, GenerationError):
        _apply_result_attributes(span, exc, span_config.attribute_mapper)
        _apply_output_content(span, exc, span_config)
        _set_generation_error_status(span, exc)
        return
    _record_other_exception(span, exc)


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

        capture_message_content True puts the bound system prompt, the bound tool definitions, the GenerationInput,
        and the assistant turn on every span this LLM's bindings open.
        It is required and has no default.
        Recording prompts is a privacy choice langchaint never makes.
        The convention says instrumentations SHOULD NOT capture content by default.
        It says they SHOULD provide an opt-in.
        This required keyword supplies the opt-in.
        The value propagates to every binding and every stream handle, and rebind carries it unchanged,
        so a rebound object cannot silently gain or lose capture.
        tracer None resolves trace.get_tracer("langchaint.tracing", <package version>) now, not at import.
        attribute_mapper is passed down unchanged to every binding; it defaults to gen_ai_attributes,
        the OTel GenAI semantic convention at the revision the module docstring pins.
        extra_attributes applies at span start to every span. None supplies no extra attributes.
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
    def adapter(self) -> Adapter:
        """The wrapped LLM's adapter, so an app never reaches for a private field."""
        return self._llm.adapter

    @property
    def shared_backoff(self) -> SharedBackoff:
        """The wrapped `LLM.shared_backoff` for applications sharing a rate-limit quota."""
        return self._llm.shared_backoff

    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "TracedBoundLLM[ModelT, ToolManager]": ...
    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "TracedBoundLLM[ModelT, None]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "TracedBoundLLM[str, ToolManager]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "TracedBoundLLM[str, None]": ...
    def bind(  # noqa: PLR0913 (mirrors LLM.bind, which states every binding choice)
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]] | None = None,
        provider_executed_tools: Sequence[Mapping[str, object]] = (),
        response_format: type[BaseModel] | None = None,
        inference_params: InferenceParams | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_attempts: int = 3,
        automatic_cache_breakpoints: bool | None = None,
    ) -> "TracedBoundLLM[Any, Any]":
        """Mirror LLM.bind and wrap its BoundLLM in a TracedBoundLLM carrying this tracer and mapper.

        A `tools` sequence constructs `TracedToolManager`.
        `tools=ToolManager(...)` binds that `ToolManager` unchanged.
        max_attempts counts requests sent including the first, so 1 means no retrying.

        Raises:
            ValueError: A `tools` sequence contains duplicate names.
                Also raised when the wrapped `LLM.bind` rejects the binding.
        """
        tools = _resolve_traced_tool_manager(tools, span_config=self._span_config)
        return TracedBoundLLM(
            bound_llm=self._llm.bind(
                system_prompt=system_prompt,
                tools=tools,
                provider_executed_tools=provider_executed_tools,
                response_format=response_format,
                inference_params=inference_params,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                extra_body=extra_body,
                max_attempts=max_attempts,
                automatic_cache_breakpoints=automatic_cache_breakpoints,
            ),
            span_config=self._span_config,
        )


class TracedBoundLLM[OutputT, ToolManagerT: ToolManager | None = None]:
    """Wraps a BoundLLM so every generate call opens a span.

    Each outbound call opens one CLIENT span. generate_many opens one per item.
    The span name is the GenAI convention {operation} {model}, wrapper-owned,
    so a custom mapper changes attributes only, never the name, kind, or status.
    There is no langchaint.elapsed_seconds attribute:
    the span brackets the same interval elapsed_seconds measures (request start to completion,
    admission waits and backoff included), so the span's own duration already carries it.
    """

    def __init__(
        self, *, bound_llm: BoundLLM[OutputT, ToolManagerT], span_config: _SpanConfig
    ) -> None:
        """Store the wrapped `BoundLLM` and `_SpanConfig`; compute the span name once.

        `span_config` is the originating `TracedLLM._span_config` object.
        """
        self._bound_llm = bound_llm
        self._span_config = span_config
        self._span_name = f"{_CHAT_OPERATION} {bound_llm.adapter.model}"

    def _apply_input_content(self, span: Span, generation_input: GenerationInput) -> None:
        """Set the input-side content attributes on a just-started span, when capture is on and it is recording.

        Set at span start alongside extra_attributes, so they are present however the span ends,
        including on the paths that raise.
        """
        if self._span_config.capture_message_content:
            _apply_content_attributes(
                span, lambda: _input_content_attributes(self._bound_llm.binding, generation_input)
            )

    @property
    def adapter(self) -> Adapter:
        """The wrapped BoundLLM's adapter."""
        return self._bound_llm.adapter

    @property
    def binding(self) -> Binding:
        """The wrapped BoundLLM's frozen binding."""
        return self._bound_llm.binding

    @property
    def response_format(self) -> type[OutputT] | None:
        """The wrapped BoundLLM's response_format."""
        return self._bound_llm.response_format

    @property
    def tool_manager(self) -> ToolManagerT:
        """The wrapped BoundLLM's ToolManager, or None where none was bound."""
        return self._bound_llm.tool_manager

    @property
    def shared_backoff(self) -> SharedBackoff:
        """The wrapped BoundLLM's SharedBackoff."""
        return self._bound_llm.shared_backoff

    @property
    def max_attempts(self) -> int:
        """max_attempts counts requests sent including the first."""
        return self._bound_llm.max_attempts

    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[NewModelT, ToolManager]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[NewModelT, None]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self: "TracedBoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: type[NewModelT],
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[NewModelT, ToolManagerT]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[str, ToolManager]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[str, None]": ...
    @overload
    def rebind(
        self: "TracedBoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: None,
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[str, ToolManagerT]": ...
    @overload
    def rebind(
        self: "TracedBoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[OutputT, ToolManager]": ...
    @overload
    def rebind(
        self: "TracedBoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[OutputT, None]": ...
    @overload
    def rebind(
        self: "TracedBoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "TracedBoundLLM[OutputT, ToolManagerT]": ...
    def rebind(  # noqa: PLR0913 (mirrors BoundLLM.rebind, which takes every field bind takes)
        self,
        *,
        response_format: type[BaseModel] | None | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tools: (
            ToolManager
            | Sequence[Tool[BaseModel | Mapping[str, object] | None]]
            | None
            | Unchanged
        ) = UNCHANGED,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        inference_params: InferenceParams | Unchanged = UNCHANGED,
        extra_body: Mapping[str, object] | None | Unchanged = UNCHANGED,
        max_attempts: int | Unchanged = UNCHANGED,
        automatic_cache_breakpoints: bool | None | Unchanged = UNCHANGED,
    ) -> "TracedBoundLLM[Any, Any]":
        """Return a traced wrapper around the rebound `BoundLLM`.

        This preserves `_SpanConfig`.
        A `tools` sequence constructs a replacement `TracedToolManager`.
        `tools=ToolManager(...)` binds that replacement unchanged.
        Omitting `tools` preserves the bound `ToolManager`.
        Passing `tools=None` removes the bound `ToolManager`.

        Raises:
            ValueError: A `tools` sequence contains duplicate names.
                Also raised when the wrapped `BoundLLM.rebind` rejects the binding.
        """
        if not isinstance(tools, Unchanged):
            tools = _resolve_traced_tool_manager(tools, span_config=self._span_config)
        return TracedBoundLLM(
            bound_llm=self._bound_llm.rebind(
                response_format=response_format,
                system_prompt=system_prompt,
                tools=tools,
                provider_executed_tools=provider_executed_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                inference_params=inference_params,
                extra_body=extra_body,
                max_attempts=max_attempts,
                automatic_cache_breakpoints=automatic_cache_breakpoints,
            ),
            span_config=self._span_config,
        )

    @overload
    async def generate_one(
        self: "TracedBoundLLM[str, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> Response[str]: ...
    @overload
    async def generate_one(
        self: "TracedBoundLLM[OutputT, ToolManager]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> GenerateResult[OutputT]: ...
    @overload
    async def generate_one(
        self: "TracedBoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> Response[OutputT]: ...
    async def generate_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> GenerateResult[Any]:
        """Open a span around the whole generate_one call, delegate, attribute, and end the span.

        The overloads mirror BoundLLM.generate_one's, so output is typed per binding.

        Raises:
            GenerationError: generate_one produced a terminal per-item error. The span is attributed and closed first.
            asyncio.CancelledError: an outer scope cancelled the call. The span ends without status.
        """
        return await self._generate_one_any_binding(
            generation_input, deadline=WallClockDeadline(timeout_seconds)
        )

    async def _generate_one_any_binding(
        self, generation_input: GenerationInput, *, deadline: Deadline
    ) -> GenerateResult[Any]:
        """Run one call under a chat span of its own.

        This unoverloaded entry point supports generic bindings.
        generate_many uses it for one span per item.
        deadline passes through unchanged.

        Raises:
            GenerationError: the call failed. The span is attributed and closed first.
            asyncio.CancelledError: an outer scope cancelled the call. The span ends without status.
        """
        return await self._under_chat_span(
            generation_input,
            self._bound_llm._generate_one_any_binding(  # noqa: SLF001
                generation_input, deadline=deadline
            ),
        )

    async def _under_chat_span(
        self, generation_input: GenerationInput, call: Coroutine[Any, Any, GenerateResult[Any]]
    ) -> GenerateResult[Any]:
        """Await one call inside a CLIENT chat span, attributing the span from however it ends.

        The span brackets the same interval as elapsed_seconds (permit waits and backoff included).
        capture_message_content records input at span start and records the result's turn at completion.

        Raises:
            GenerationError: whatever the call failed with, the span attributed and closed first.
            Exception: anything else the call raised, recorded on the span before it re-raises.
            asyncio.CancelledError: an outer scope cancelled the call. The span ends without status.
        """
        span = _start_span(self._span_config.tracer, self._span_name, kind=SpanKind.CLIENT)
        try:
            _apply_extra_attributes(span, self._span_config.extra_attributes)
            _apply_operation_name(span, _CHAT_OPERATION)
            self._apply_input_content(span, generation_input)
            try:
                result = await call
            except GenerationError as exc:
                _apply_result_attributes(span, exc, self._span_config.attribute_mapper)
                _apply_output_content(span, exc, self._span_config)
                _set_generation_error_status(span, exc)
                raise
            except Exception as exc:
                _record_other_exception(span, exc)
                raise
            _apply_result_attributes(span, result, self._span_config.attribute_mapper)
            _apply_output_content(span, result, self._span_config)
            _set_ok_status(span)
            return result
        finally:
            _end_span(span)

    @overload
    async def generate_many(
        self: "TracedBoundLLM[str, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_item: float | None = ...,
    ) -> list[Response[str] | GenerationError]: ...
    @overload
    async def generate_many(
        self: "TracedBoundLLM[OutputT, ToolManager]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_item: float | None = ...,
    ) -> list[CallResult[OutputT]]: ...
    @overload
    async def generate_many(
        self: "TracedBoundLLM[OutputT, None]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_item: float | None = ...,
    ) -> list[Response[OutputT] | GenerationError]: ...
    async def generate_many(
        self,
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = False,
        max_working_seconds_per_item: float | None = None,
        # `list` invariance requires `Any` to share one implementation across these overloads.
    ) -> list[Any]:
        """Order-aligned batch, traced as one chat span per started item and nothing else.

        The overloads mirror BoundLLM.generate_many's, so each result's output is typed per binding.

        warm_cache and max_working_seconds_per_item pass through to BoundLLM.generate_many.

        Configure the OTel SDK sampler to limit span volume.

        Raises:
            asyncio.CancelledError: an outer scope cancelled the batch. Each started span ended.
            BaseException: an item raised a non-Exception BaseException.
                Started items are cancelled and awaited before it propagates.
        """
        # `_generate_one_any_binding` accepts each overloaded `response_format`.
        return await self._bound_llm._generate_many_any_binding(  # noqa: SLF001
            generation_inputs,
            warm_cache=warm_cache,
            generate_item=self._generate_one_any_binding,
            max_working_seconds_per_item=max_working_seconds_per_item,
        )

    @overload
    def stream_one(
        self: "TracedBoundLLM[str, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> "TracedStreamHandle[str]": ...
    @overload
    def stream_one(
        self: "TracedBoundLLM[OutputT, ToolManager]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> "TracedStreamHandle[OutputT, ToolCallTurn[OutputT]]": ...
    @overload
    def stream_one(
        self: "TracedBoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> "TracedStreamHandle[OutputT]": ...
    def stream_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> "TracedStreamHandle[Any, Any]":
        """Wrap the BoundLLM's StreamHandle in a TracedStreamHandle; no I/O and no span yet.

        The overloads mirror BoundLLM.stream_one's, so final()'s result is typed per binding.

        The span and request open when the handle is entered.
        The binding and the generation_input are passed down rather than rendered here:
        the handle needs them to build its input attributes when the span starts,
        Rendering here would serialize generation_input for non-recording spans.
        The cost is that the handle holds the generation_input for the stream's whole life.
        """
        return TracedStreamHandle(
            # `_generate_one_any_binding` accepts each overloaded `response_format`.
            stream_handle=self._bound_llm._stream_one_any_binding(  # noqa: SLF001
                generation_input, timeout_seconds=timeout_seconds
            ),
            span_config=self._span_config,
            span_name=self._span_name,
            binding=self._bound_llm.binding,
            generation_input=generation_input,
        )


class TracedStreamHandle[OutputT, ToolTurnT: ToolCallTurn[object] = Never]:
    """Wraps a StreamHandle, owning one span across the stream's life.

    Items pass through by reference;
    nothing is rewrapped (the no-rewrap rule bans copying data into same-shape containers;
    observing an iterator is unaffected).
    The span opens at __aenter__, where the request opens,
    records gen_ai.response.time_to_first_chunk at the first item,
    takes error status when the stream raises or its consuming block leaves with an exception,
    and ends exactly once.
    Under capture_message_content the input content attributes are set when the span starts,
    gen_ai.output.messages records any terminal result that carries a turn.
    """

    def __init__(
        self,
        *,
        stream_handle: StreamHandle[OutputT, ToolTurnT],
        span_config: _SpanConfig,
        span_name: str,
        binding: Binding,
        generation_input: GenerationInput,
    ) -> None:
        """Store the wrapped handle and the span pieces; the span is not started here.

        span_config is the binding's, unchanged; TracedLLM documents what each of its values means.
        binding and generation_input are held only to build the input content attributes when the span starts,
        and are read for nothing else.
        """
        self._stream_handle = stream_handle
        self._span_config = span_config
        self._span_name = span_name
        self._binding = binding
        self._generation_input = generation_input
        self._span: Span | None = None
        self._span_started_at_monotonic_seconds: float | None = None
        self._span_ended = False
        self._first_item_seen = False

    @property
    def abandoned(self) -> AbandonedCallError | None:
        """The wrapped handle's account of this call where a cancellation cut it off, None otherwise.

        StreamHandle.abandoned says when it is set.
        """
        return self._stream_handle.abandoned

    def _start_span_once(self) -> Span:
        """Start this handle's one span, recording its start time for gen_ai.response.time_to_first_chunk.

        Called by __aenter__ alone, which raises on a second entry, so this runs at most once per handle.
        stream_one opens no span and does no I/O. Building here lets _apply_content_attributes skip non-recording spans.
        """
        span = _start_span(self._span_config.tracer, self._span_name, kind=SpanKind.CLIENT)
        self._span = span
        _apply_extra_attributes(span, self._span_config.extra_attributes)
        _apply_operation_name(span, _CHAT_OPERATION)
        if self._span_config.capture_message_content:
            _apply_content_attributes(
                span, lambda: _input_content_attributes(self._binding, self._generation_input)
            )
        self._span_started_at_monotonic_seconds = time.monotonic()
        return span

    def _end_span_once(self) -> None:
        if self._span is not None and not self._span_ended:
            _end_span(self._span)
            self._span_ended = True

    def _mark_first_item(self, span: Span) -> None:
        """Record the gen_ai.response.time_to_first_chunk attribute on the first item's arrival, once.

        The value is the monotonic seconds from the span's start (__aenter__) to the first item.
        The convention defines this key as measured from request issuance;
        This value also includes admission waits and backoff because the span starts before request issuance.
        Set only when a first item passes through this iterator,
        so a stream drained by final() without iteration carries no such attribute.
        """
        if self._first_item_seen:
            return
        self._first_item_seen = True
        if self._span_started_at_monotonic_seconds is not None:
            _set_span_attribute(
                span,
                "gen_ai.response.time_to_first_chunk",
                time.monotonic() - self._span_started_at_monotonic_seconds,
            )

    def __aiter__(self) -> "TracedStreamHandle[OutputT, ToolTurnT]":
        """Return self; the wrapper is its own iterator."""
        return self

    async def __anext__(self) -> StreamItem:
        """Delegate to the inner handle, observing the first item and any failure.

        StopAsyncIteration passes through without ending the span,
        so a following final() can still set attributes and end it.
        Cancellation passes through so __aexit__ can distinguish timeout_seconds from outer cancellation.

        Raises:
            StopAsyncIteration: the inner stream is exhausted; the span is left open for final().
            Exception: the inner stream raised after open_stream() returned.
        """
        if self._span is None or self._span_ended:
            # The inner handle reports calls before entry or after span closure.
            return await self._stream_handle.__anext__()
        span = self._span
        try:
            item = await self._stream_handle.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as exc:
            _record_stream_conclusion(span, exc, self._span_config)
            self._end_span_once()
            raise
        self._mark_first_item(span)
        return item

    async def __aenter__(self) -> "TracedStreamHandle[OutputT, ToolTurnT]":
        """Start the span, then open the inner handle's request.

        The span starts first so a failing open is recorded on it rather than escaping untraced.
        __aexit__ does not run when __aenter__ raises, so the span is ended here on that path.
        A second entry raises before the span is touched, so it cannot mark the first stream's span failed.

        Raises:
            RuntimeError: This handle was already entered.
            Exception: the inner handle failed to open. The span records the exception and ends.
        """
        if self._span is not None:
            raise RuntimeError("stream already entered: call stream_one again for a new one")
        span = self._start_span_once()
        try:
            await self._stream_handle.__aenter__()
        except Exception as exc:
            _record_stream_conclusion(span, exc, self._span_config)
            self._end_span_once()
            raise
        except BaseException:
            self._end_span_once()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the inner handle, then end the span if it is still open.

        A span already ended by a mid-iteration failure or by final() is left alone.
        An active Exception is recorded before the span ends.

        __aexit__ distinguishes timeout_seconds from outer cancellation.
        A timeout becomes GenerationError with terminal result attributes.
        Outer cancellation ends the span without status.
        """
        try:
            await self._stream_handle.__aexit__(exc_type, exc, traceback)
        except GenerationError as conclusion:
            if self._span is not None and not self._span_ended:
                _record_stream_conclusion(self._span, conclusion, self._span_config)
            raise
        finally:
            if self._span is not None and not self._span_ended:
                if isinstance(exc, Exception):
                    _record_other_exception(self._span, exc)
                self._end_span_once()

    @overload
    async def final(self: "TracedStreamHandle[OutputT, Never]") -> Response[OutputT]: ...
    @overload
    async def final(self) -> "Response[OutputT] | ToolTurnT": ...
    async def final(self) -> Response[OutputT] | ToolCallTurn[object]:
        """Drain the inner stream, attribute the span from the result, and end the span.

        The span ends once. After it ends, this delegates to the cached inner final() result.

        Raises:
            GenerationError: inner final() raised a terminal per-item result.
            StreamProtocolError: the provider's event stream ended without a terminal event;
                the span records it and closes.
        """
        if self._span is None or self._span_ended:
            # Calls before entry and calls after closure do not change the span.
            return await self._stream_handle.final()
        span = self._span
        try:
            result = await self._stream_handle.final()
        except Exception as exc:
            _record_stream_conclusion(span, exc, self._span_config)
            self._end_span_once()
            raise
        _apply_result_attributes(span, result, self._span_config.attribute_mapper)
        _apply_output_content(span, result, self._span_config)
        _set_ok_status(span)
        self._end_span_once()
        return result


def _dispatch_error_type(outcome: DispatchOutcome) -> str | None:
    """Classify a dispatch outcome for error.type, or None where the call succeeded.

    The three values are the documented error list the convention asks instrumentations to publish:
    tool_error (the tool ran and authored a failure), invalid_tool_args, and unknown_tool.
    error.type values "invalid_tool_args" and "unknown_tool" mean the tool function never ran.
    A raising tool function is classified by _record_other_exception with its exception class name instead.
    """
    if isinstance(outcome, DispatchHandled):
        return "tool_error" if outcome.tool_message.is_error else None
    if isinstance(outcome, DispatchInvalidToolArgs):
        return "invalid_tool_args"
    return "unknown_tool"


class TracedToolManager(ToolManager):
    """A ToolManager whose every dispatch opens one execute_tool span.

    `TracedToolManager` subclasses `ToolManager` for `LLM.bind(tools=TracedToolManager(...))`.
    `ToolManager.dispatch_many` calls `self.dispatch`.
    Overriding `dispatch` adds spans without duplicating `ToolManager.dispatch_many`.
    The span name is "execute_tool {call.name}". SpanKind.INTERNAL represents in-process dispatch.
    dispatch makes its span current while the tool function runs (trace.use_span).
    Spans the function starts (an instrumented HTTP request, a nested agent loop) nest under the execute_tool span.
    dispatch_many stays safe: asyncio.gather runs each dispatch in its own task with a copied context.
    Concurrent dispatch spans are therefore siblings, never nested in one another.
    The identity attributes gen_ai.operation.name, gen_ai.tool.name, and gen_ai.tool.call.id are set at span start;
    at completion the span takes its status and error.type from the outcome:

    | dispatch result                     | status | error.type              |
    | ----------------------------------- | ------ | ----------------------- |
    | DispatchHandled, is_error False     | OK     | absent                  |
    | DispatchHandled, is_error True      | ERROR  | tool_error              |
    | DispatchInvalidToolArgs             | ERROR  | invalid_tool_args       |
    | DispatchUnknownTool                 | ERROR  | unknown_tool            |
    | the tool function raised            | ERROR  | the exception class name|

    invalid_tool_args and unknown_tool are the two values meaning the tool function never ran.
    A tool returning is_error True lets the model correct the call.
    The other failure variants serve the same control flow.
    So a healthy agent doing one argument-validation retry emits ERROR spans as a matter of routine,
    and a dashboard reading span status as a health signal will show that.
    That is accepted rather than worked around: error.type is the field an operator filters on,
    OTel status describes the operation outcome, so these failures use ERROR status.

    capture_message_content records gen_ai.tool.call.arguments at span start and gen_ai.tool.call.result at completion.
    gen_ai.tool.call.result and gen_ai.input.messages use the same tool_call_response shape.
    That shape is an object, which is what the key's note asks for ("It's expected to be an object");
    The key's JSON schema requires an object without required properties.
    This module uses a response parts array for both str and Sequence[ContentPart].
    gen_ai.tool.call.arguments is the model's own argument JSON, deserialized on a best effort:
    The convention requests best-effort deserialization to an object.
    The nested value is reserialized because span attributes cannot nest.
    On this key the effect is therefore normalization; the nesting matters on the generate span,
    whose gen_ai.input.messages embeds the same arguments in a structure serialized as a whole.
    Unparseable text is preserved as a quoted JSON string.

    gen_ai.tool.call.result is recorded on every variant, including the two where the tool function never ran.
    The convention defines that key as the result "if any and if execution was successful",
    This module also records langchaint's correction for failure variants. error.type states that no tool produced it.
    gen_ai.tool.call.result is what dispatch returned, which is not necessarily what the model read:
    The application may change or drop tool_message before appending it to Sequence[Message].
    gen_ai.input.messages records the resulting value.
    The two join on the tool call id, which is gen_ai.tool.call.id here and the tool_call_response part's id there.

    There is no attribute_mapper parameter: the attributes are the fixed keys above,
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

        `capture_message_content` has no default because content capture affects privacy.
        `tracer=None` resolves the langchaint tracer.
        `extra_attributes=None` sets no constant dispatch attributes.
        Dispatch attributes override colliding `extra_attributes` keys.

        Raises:
            ValueError: two tools share a name.
        """
        super().__init__(tools)
        self._tracer = (
            tracer
            if tracer is not None
            else trace.get_tracer("langchaint.tracing", _PACKAGE_VERSION)
        )
        self._extra_attributes: SpanAttributes = (
            extra_attributes if extra_attributes is not None else {}
        )
        self._capture_message_content = capture_message_content

    @override
    async def dispatch(self, call: ToolCall) -> DispatchOutcome:
        """Open one execute_tool span around ToolManager.dispatch and attribute it from the outcome.

        The dispatch semantics are the base method's own; the override adds only the span.
        The span is current while the base dispatch runs, so a span the tool function starts nests under it.
        A function exception (a user-code defect) is recorded on the span, sets error status, and propagates.
        trace.use_span makes the span current with its exception recording and status setting off;
        the finally ends it exactly once, through _end_span.
        This method records and sets status itself, from the table on the class.
        """
        span = _start_span(
            self._tracer,
            f"{_EXECUTE_TOOL_OPERATION} {call.name}",
            kind=SpanKind.INTERNAL,
        )
        with trace.use_span(
            span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            try:
                _apply_extra_attributes(span, self._extra_attributes)
                with _guarding_telemetry_failures("setting the tool identity attributes"):
                    if _is_recording(span):
                        span.set_attributes({
                            "gen_ai.operation.name": _EXECUTE_TOOL_OPERATION,
                            "gen_ai.tool.name": call.name,
                            "gen_ai.tool.call.id": call.id,
                        })
                if self._capture_message_content:
                    _apply_content_attributes(
                        span,
                        lambda: {
                            "gen_ai.tool.call.arguments": json.dumps(
                                _tool_call_arguments(call.args_json)
                            )
                        },
                    )
                try:
                    outcome = await super().dispatch(call)
                except Exception as exc:
                    _record_other_exception(span, exc)
                    raise
                error_type = _dispatch_error_type(outcome)
                if error_type is not None:
                    _set_span_attribute(span, "error.type", error_type)
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
                    _set_ok_status(span)
                else:
                    with _guarding_telemetry_failures("setting the error status"):
                        span.set_status(Status(StatusCode.ERROR, error_type))
                return outcome
            finally:
                _end_span(span)


__all__ = [
    "AttributeMapper",
    "SpanAttributes",
    "TracedBoundLLM",
    "TracedLLM",
    "TracedStreamHandle",
    "TracedToolManager",
    "agent_span",
    "gen_ai_attributes",
]
