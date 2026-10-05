"""Trace langchaint generations and tool dispatches with OTel spans.

Importing this subpackage requires `opentelemetry-api`.
Applications configure the OTel SDK.

Pass one `OtelObserver` to a backend constructor, as in `OpenAI(observer=OtelObserver(...))`.
Every `LLM`, binding, and stream that backend creates is then traced, and the application's code is otherwise unchanged.
A `ToolManager` that `bind` builds from a tool sequence traces its dispatches.
A `ToolManager` the application builds traces its dispatches when built with `observer=`.

Each input opens one CLIENT span.
`generate_many` opens one span per started input.
`generate_many_records` opens one span per input that requires generation.
Restored records open no span.
A stream opens one CLIENT span when its handle is entered.
Each tool dispatch opens one INTERNAL `execute_tool` span.
`ToolManager.dispatch_many` uses `dispatch` and gets one span per tool call.
`precomputed` opens no span because it executes no tool.
A generation span is current during its retry loop, and a dispatch span is current while the tool function runs.
Spans started there, such as HTTP client spans, nest under them.
A stream span is never current, because the application's code runs between stream items.

`OtelObserver` requires `capture_message_content` because prompt recording is a privacy choice.
`capture_message_content=True` records every content part unchanged, including image and audio bytes.
`content_filter` decides per part what each content attribute records.
`gen_ai.tool.definitions` is not filtered because a tool schema is not a message part.
A filter that raises or returns a part of another kind is logged, and every attribute in the same build is omitted.
Organisation-wide redaction of recorded text belongs in an OpenTelemetry Collector processor.
`extra_attributes` sets constant attributes when each span starts.
Request and outcome attributes replace matching `extra_attributes` keys.
Required `gen_ai.operation.name` values also replace matching `extra_attributes` keys.

Chat and stream spans use `gen_ai.operation.name="chat"`.
They report provider, request model, response model, finish reasons, token usage, request count, and cost.
They report the standard attributes for each set request field.
They report `gen_ai.output.type` for every input.
With capture enabled, they report system instructions, tool definitions, input messages, and output messages.
Stream spans also report `gen_ai.request.stream=True`.
A stream span reports `gen_ai.response.time_to_first_chunk` once an item arrives, including for a stream left early.
A stream span whose block exits before the stream stores a `GenerationOutcome` reports its usage and cost.
Its status stays unset, because handling the input did not fail.
Tool spans use `gen_ai.operation.name="execute_tool"` and report tool name and tool call id.
With capture enabled, tool spans report arguments and results.

`langchaint.*` names the request count and cost because the convention has no matching keys.
Each failed request adds `langchaint.request_failed` with `error_text` and `elapsed_seconds`.

Each span starts and ends exactly once, including for failed, cancelled, and abandoned streams.
A mapper may set only attribute names and values.
A mapper cannot change the span name, kind, or status.
Telemetry failures, including mapper and content filter failures, are logged and never propagate.

Readable reasoning uses `ReasoningPart` in content payloads.
Content payloads follow the convention's JSON schemas.
`gen_ai.tool.call.arguments` may carry a JSON value other than an object.

"""

import base64
import importlib.metadata
import json
import logging
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, overload

from pydantic import TypeAdapter, ValidationError

try:
    from opentelemetry import context, trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer, TracerProvider
except ModuleNotFoundError as exc:
    if exc.name is not None and not exc.name.startswith("opentelemetry"):
        raise
    raise ModuleNotFoundError(
        "langchaint's tracing subpackage requires opentelemetry-api; install langchaint[tracing]."
    ) from exc

from langchaint.adapter import Binding
from langchaint.common.messages import (
    AssistantPart,
    ContentPart,
    JsonValue,
    Message,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
)
from langchaint.common.observed_operation import ObservedOperation
from langchaint.generation.errors import GenerationError
from langchaint.generation.observer import GenerationStart
from langchaint.generation.request_history import AbandonedStreamRecord
from langchaint.generation.response import GenerationOutcome
from langchaint.tools import DispatchOutcome, ToolSchema

type SpanAttributeValue = str | bool | int | float | list[str] | tuple[str, ...]
"""One span attribute's value."""

type SpanAttributes = Mapping[str, SpanAttributeValue]
"""A span's attributes, keyed by name."""

type AttributeMapper = Callable[
    [GenerationOutcome[object] | AbandonedStreamRecord], SpanAttributes
]
"""Maps one `GenerationOutcome`, or a stream's `abandoned` record, to its span attributes.

The mapper reads the fields shared by each `GenerationOutcome` variant and `AbandonedStreamRecord`.
No mapper receives the input messages, so `gen_ai_attributes` cannot put a prompt on a span.
A custom mapper can reach `raw` on a `Generation`, which holds the final SDK response by reference.
It can reach `request_provider_data` on a `Generation` or `GenerationError`.
`AbandonedStreamRecord` has no `raw`, because no response completed.
`capture_message_content` controls prompt capture because `OtelObserver` receives the messages when an input starts.
"""

type ContentFilter = Callable[
    [str, ContentPart | AssistantPart], ContentPart | AssistantPart | None
]
"""Decides what one content attribute records for one part.

The first argument is the content attribute name, such as `"gen_ai.input.messages"`.
The second argument is the part langchaint would record.
The return value is the part to record, or `None` to omit the part.
A returned part must have the same `kind` as the part passed in.
`gen_ai.tool.call.arguments` holds one `ToolCall`, so `None` omits that attribute.
A user or assistant message whose every part is omitted records an empty `parts` array.
A tool message whose every part is omitted records one tool_call_response part with an empty `response`.
A bare `str` prompt or content is passed as one `TextPart`.
Message-level fields such as `role`, `tool_call_id`, `is_error`, and `finish_reason` never reach the filter.
"""


def _record_every_part(
    _attribute_name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart:
    """Return `part` unchanged, as the filter that `content_filter=None` selects."""
    return part


@overload
def _filtered_part(
    content_filter: ContentFilter, attribute_name: str, part: ToolCall
) -> ToolCall | None: ...
@overload
def _filtered_part(
    content_filter: ContentFilter, attribute_name: str, part: ContentPart
) -> ContentPart | None: ...
@overload
def _filtered_part(
    content_filter: ContentFilter, attribute_name: str, part: AssistantPart
) -> AssistantPart | None: ...
def _filtered_part(
    content_filter: ContentFilter, attribute_name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart | None:
    """Run the filter on one part.

    The overloads state what the `isinstance` check guarantees: a kept part has the class of `part`.

    Raises:
        TypeError: The filter returned a part of another kind, which cannot take the place of `part`.
    """
    filtered = content_filter(attribute_name, part)
    if filtered is None or isinstance(filtered, type(part)):
        return filtered
    raise TypeError(
        f"the content filter for {attribute_name} returned {type(filtered).__name__} "
        f"for {type(part).__name__}"
    )


_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_PACKAGE_VERSION = importlib.metadata.version("langchaint")
_CHAT_OPERATION = "chat"
"""The GenAI operation value for a chat completion."""

_EXECUTE_TOOL_OPERATION = "execute_tool"
"""The GenAI operation value for a tool execution."""

_logger = logging.getLogger("langchaint.tracing")


@contextmanager
def _guarding_telemetry_failures(what: str) -> Generator[None]:
    """Log whatever the block raises instead of letting it out.

    OTel Exception values are logged because they must not replace a return value or active exception.
    Application callables use guards that name the callable.
    Only Exception is caught, so a cancellation still reaches the caller.
    """
    try:
        yield
    except Exception:
        _logger.warning("%s raised; this span's telemetry is incomplete", what, exc_info=True)


def _set_ok_status(span: Span) -> None:
    """Mark one span OK, without letting an OTel exception reach the caller."""
    with _guarding_telemetry_failures("setting the span status"):
        span.set_status(Status(StatusCode.OK))


def _set_span_attributes(span: Span, attributes: SpanAttributes) -> None:
    """Set a mapping of attributes, without letting an OTel exception reach the caller.

    A non-recording span ignores them, as the OTel API specifies.
    """
    with _guarding_telemetry_failures("setting the span attributes"):
        span.set_attributes(attributes)


def _set_built_attributes(span: Span, what: str, build: Callable[[], SpanAttributes]) -> None:
    """Set the attributes `build` returns on a recording span, logging an Exception from either step.

    `build` runs only for a recording span, so an application without a TracerProvider pays nothing for it.
    `what` names the step in the log message.
    Existing span attributes remain when `build` raises.
    """
    with _guarding_telemetry_failures(what):
        if span.is_recording():
            span.set_attributes(build())


@dataclass(frozen=True, kw_only=True)
class _SpanConfig:
    tracer: Tracer
    attribute_mapper: AttributeMapper
    extra_attributes: SpanAttributes
    capture_message_content: bool
    content_filter: ContentFilter


_CONVENTION_FINISH_REASONS: Mapping[StopReason, str] = {
    "max_completion_tokens": "length",
}
"""The StopReason values whose counterpart in the convention's finish-reason vocabulary has another name.

stop and tool_call are the convention's own values.
refusal, context_window_exceeded, and other are absent deliberately and pass through unmapped:
the convention's content_filter means a provider filter blocked content.
No convention value corresponds to a context-window overflow or to other.
The convention's enum is open because the output schema permits the enum or a string.
Unmapped values pass through unchanged.
"""


_NO_STOP_REASON_FINISH_REASON = "error"
"""What gen_ai.output.messages reports for an assistant message whose outcome states no stop reason.

The convention uses `error` when handling an input fails.
`gen_ai.response.finish_reasons` is optional and is omitted.
The per-message field is required, so an assistant message recorded from a failure needs a value.
"""


def _finish_reason(stop_reason: StopReason) -> str:
    """Map a StopReason onto the convention's finish-reason vocabulary, passing unmapped values through.

    gen_ai.response.finish_reasons and gen_ai.output.messages use this mapping.
    """
    return _CONVENTION_FINISH_REASONS.get(stop_reason, stop_reason)


def gen_ai_attributes[OutputT](
    outcome: GenerationOutcome[OutputT] | AbandonedStreamRecord,
) -> SpanAttributes:
    """Map a `GenerationOutcome` to GenAI-convention span attributes plus langchaint scalars.

    `outcome` supplies the response identity, usage, stop reason, and request records.
    This is the default attribute_mapper.
    A custom AttributeMapper can extend the returned dict.
    Extension keys must use the application's namespace because langchaint.* is reserved.
    `extra_attributes` sets a constant on every span.
    Each call builds and returns a fresh dict, so extending it mutates nothing shared.
    This function reads only the fields shared by every `GenerationOutcome` variant and `AbandonedStreamRecord`.
    The langchaint.* prefix is used only when the GenAI convention has no corresponding key.
    This applies to langchaint.request_count and langchaint.cost_in_usd.
    gen_ai.usage.input_tokens is Usage.input_tokens_total.
    The cache-read and cache-write attributes are parts of that total.
    No cache_none counter is emitted because it is derived.
    gen_ai.response.finish_reasons contains the mapped stop_reason and is omitted when stop_reason is None.
    gen_ai.response.model is the last request's model_served and is omitted when unavailable.
    gen_ai.response.time_to_first_chunk is the last request's first_item_after_seconds, settled or cut off.
    That value runs from sending the request to the first stream item, so only a stream has it.
    The usage and cost attributes are the input's paid totals across every request.
    `outcome.usage` has that scope.
    `langchaint.request_failed` span events retain per-request detail.
    """
    usage = outcome.usage
    records = outcome.request_records
    final_record = records[-1] if records else None
    final_settled_record = (
        final_record if final_record is not None and final_record.kind == "settled" else None
    )
    attributes: dict[str, SpanAttributeValue] = {
        "gen_ai.provider.name": outcome.provider_name,
        "gen_ai.request.model": outcome.model,
        "gen_ai.usage.input_tokens": usage.input_tokens_total,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "gen_ai.usage.reasoning.output_tokens": usage.output_tokens_reasoning,
        "gen_ai.usage.cache_read.input_tokens": usage.input_tokens_cache_read,
        "gen_ai.usage.cache_write.input_tokens": usage.input_tokens_cache_write,
        "langchaint.request_count": outcome.request_count,
        "langchaint.cost_in_usd": usage.cost_in_usd,
    }
    if final_settled_record is not None and final_settled_record.model_served is not None:
        attributes["gen_ai.response.model"] = final_settled_record.model_served
    if final_record is not None and final_record.first_item_after_seconds is not None:
        attributes["gen_ai.response.time_to_first_chunk"] = final_record.first_item_after_seconds
    if outcome.stop_reason is not None:
        attributes["gen_ai.response.finish_reasons"] = [_finish_reason(outcome.stop_reason)]
    return attributes


def _blob_part(modality: str, media_type: str, data: bytes) -> dict[str, object]:
    """Render inline bytes as the convention's BlobPart with standard base64 content."""
    return {
        "type": "blob",
        "modality": modality,
        "mime_type": media_type,
        "content": base64.b64encode(data).decode("ascii"),
    }


def _content_part(part: ContentPart) -> dict[str, object]:
    """Render one ContentPart as the convention's part object.

    ImagePart and AudioPart become BlobPart objects carrying their bytes.
    ImageUrlPart becomes an image uri part with its optional media_type as mime_type.
    `cache_breakpoint` is omitted because the convention has no corresponding field.
    """
    match part.kind:
        case "text":
            return {"type": "text", "content": part.text}
        case "image":
            return _blob_part("image", part.media_type, part.data)
        case "image_url":
            mime_type = {} if part.media_type is None else {"mime_type": part.media_type}
            return {"type": "uri", "modality": "image", "uri": part.url, **mime_type}
        case "audio":
            return _blob_part("audio", part.media_type, part.data)


def _recorded_content(
    content: str | tuple[ContentPart, ...], content_filter: ContentFilter, attribute_name: str
) -> list[dict[str, object]]:
    """Filter content parts and render the kept ones as the convention's parts array.

    A str is filtered as one `TextPart`.
    A bound system prompt is a str or a tuple of `TextPart`, so it renders here too.
    gen_ai.system_instructions items share the text part shape.
    """
    content_parts: tuple[ContentPart, ...] = (
        (TextPart(text=content),) if isinstance(content, str) else content
    )
    kept = (_filtered_part(content_filter, attribute_name, part) for part in content_parts)
    return [_content_part(part) for part in kept if part is not None]


def _tool_call_arguments(args_json: str) -> JsonValue:
    """Deserialize a tool call's argument JSON, falling back to the raw text when it is not standard JSON.

    The convention requests best-effort deserialization of serialized arguments.
    Any JSON value is returned.
    Text that does not validate as a `JsonValue` is returned unchanged, so DispatchInvalidToolArgs remains visible.
    That includes Infinity, NaN, and a number such as 1e400 that overflows to a non-finite float.
    RFC 8259 excludes those values, so the fallback keeps every emitted payload standard JSON.
    """
    try:
        return _JSON_VALUE_ADAPTER.validate_json(args_json)
    except ValidationError:
        return args_json


def _assistant_part(part: AssistantPart) -> dict[str, object] | None:
    """Render one AssistantPart as the convention's part object, or None when it records nothing.

    ReasoningPart and TextPart emit their text, and render as None when it is empty.
    ReasoningPart.raw is opaque and is never emitted.
    A RawPart renders as None because it has no text.
    """
    match part.kind:
        case "reasoning":
            return {"type": "reasoning", "content": part.text} if part.text else None
        case "text":
            return {"type": "text", "content": part.text} if part.text else None
        case "tool_call":
            return {
                "type": "tool_call",
                "id": part.id,
                "name": part.name,
                "arguments": _tool_call_arguments(part.args_json),
            }
        case "raw":
            return None


def _recorded_assistant_parts(
    parts: tuple[AssistantPart, ...], content_filter: ContentFilter, attribute_name: str
) -> list[dict[str, object]]:
    """Filter an assistant message's parts and render the kept ones as the convention's parts array, in order.

    If every part is omitted or renders as None, the message renders as an empty parts array, not as a missing message.
    """
    kept = (_filtered_part(content_filter, attribute_name, part) for part in parts)
    rendered = (_assistant_part(part) for part in kept if part is not None)
    return [part for part in rendered if part is not None]


def _input_message(message: Message, content_filter: ContentFilter) -> dict[str, object]:
    """Render one input Message as the convention's {role, parts} shape, with every part filtered.

    A `ToolMessage` becomes a tool_call_response part inside a tool-role message.
    """
    attribute_name = "gen_ai.input.messages"
    match message.kind:
        case "user":
            otel_parts = _recorded_content(message.content, content_filter, attribute_name)
            return {"role": "user", "parts": otel_parts}
        case "tool":
            otel_parts = [_tool_call_response_part(message, content_filter, attribute_name)]
            return {"role": "tool", "parts": otel_parts}
        case "assistant":
            otel_parts = _recorded_assistant_parts(message.parts, content_filter, attribute_name)
            return {"role": "assistant", "parts": otel_parts}


def _tool_call_response_part(
    message: ToolMessage, content_filter: ContentFilter, attribute_name: str
) -> dict[str, object]:
    """Render one ToolMessage as the convention's tool_call_response part, with its content filtered.

    One tool result reaches a backend under this one shape from both spans that report it:
    inside gen_ai.input.messages on a generate span, and as gen_ai.tool.call.result on a tool span.
    `tool_call_id` and `is_error` never reach the filter.
    """
    return {
        "type": "tool_call_response",
        "id": message.tool_call_id,
        "is_error": message.is_error,
        "response": _recorded_content(message.content, content_filter, attribute_name),
    }


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
    binding: Binding, messages: Sequence[Message], *, content_filter: ContentFilter
) -> dict[str, SpanAttributeValue]:
    """Build the input-side content attributes for one input, each a JSON string.

    OTel attribute values cannot nest, so structured values use the permitted JSON string form.
    A key whose source is empty or absent is omitted.
    `system_prompt=None` omits gen_ai.system_instructions, and so does a prompt whose every part filters away.
    No bound tools omits gen_ai.tool.definitions.
    These omissions are indistinguishable from disabled capture.
    gen_ai.tool.definitions does not pass through `content_filter` because a tool schema is not a part.
    """
    attributes: dict[str, SpanAttributeValue] = {}
    if binding.system_prompt is not None:
        system_instructions = _recorded_content(
            binding.system_prompt, content_filter, "gen_ai.system_instructions"
        )
        if system_instructions:
            attributes["gen_ai.system_instructions"] = json.dumps(system_instructions)
    if binding.tool_schemas:
        attributes["gen_ai.tool.definitions"] = json.dumps(_tool_definitions(binding.tool_schemas))
    input_messages = [_input_message(message, content_filter) for message in messages]
    if input_messages:
        attributes["gen_ai.input.messages"] = json.dumps(input_messages)
    return attributes


def _request_attributes(start: GenerationStart) -> dict[str, SpanAttributeValue]:
    """Map the operation and stored request configuration onto standard OTel attributes."""
    binding = start.binding
    attributes: dict[str, SpanAttributeValue] = {
        "gen_ai.operation.name": _CHAT_OPERATION,
        "gen_ai.provider.name": start.provider_name,
        "gen_ai.request.model": start.model,
        "gen_ai.output.type": "text" if start.response_format is None else "json",
    }
    if binding.max_completion_tokens is not None:
        attributes["gen_ai.request.max_tokens"] = binding.max_completion_tokens
    if binding.reasoning_level is not None:
        attributes["gen_ai.request.reasoning.level"] = binding.reasoning_level
    if binding.temperature is not None:
        attributes["gen_ai.request.temperature"] = binding.temperature
    if start.stream:
        attributes["gen_ai.request.stream"] = True
    return attributes


def _output_content_attributes(
    parts: tuple[AssistantPart, ...], stop_reason: StopReason | None, content_filter: ContentFilter
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.output.messages from one assistant message's parts.

    One function for the generation and the failure paths, so an assistant message renders the same from either.
    One key contains one message.
    """
    return {
        "gen_ai.output.messages": json.dumps([
            {
                "role": "assistant",
                "parts": _recorded_assistant_parts(
                    parts, content_filter, "gen_ai.output.messages"
                ),
                "finish_reason": (
                    _NO_STOP_REASON_FINISH_REASON
                    if stop_reason is None
                    else _finish_reason(stop_reason)
                ),
            }
        ])
    }


def _apply_output_content[OutputT](
    span: Span,
    outcome: GenerationOutcome[OutputT] | AbandonedStreamRecord,
    span_config: _SpanConfig,
) -> None:
    """Set gen_ai.output.messages from the outcome's assistant message, when capture is on and the span is recording.

    GenerationError carries the last assistant message. The key is omitted when no request produced one.
    Per-request detail stays on the langchaint.request_failed events, which carry no content.
    """
    assistant_message = outcome.assistant_message
    if assistant_message is None:
        return
    stop_reason = outcome.stop_reason
    _apply_content_attributes(
        span,
        span_config,
        lambda content_filter: _output_content_attributes(
            assistant_message.parts, stop_reason, content_filter
        ),
    )


def _record_request_failed_events[OutputT](
    span: Span, outcome: GenerationOutcome[OutputT] | AbandonedStreamRecord
) -> None:
    """Add one langchaint.request_failed event per failed request in the outcome's request records.

    Each event carries the request's error text and its own `elapsed_seconds`.
    Events are stamped at recording time because the records carry only monotonic brackets.
    They answer the first question a slow traced input raises: was it one request or the retries.
    """
    for record in outcome.request_records:
        if record.kind == "settled" and record.error is not None:
            span.add_event(
                "langchaint.request_failed",
                {"error_text": str(record.error), "elapsed_seconds": record.elapsed_seconds},
            )


def _apply_outcome_attributes[OutputT](
    span: Span,
    outcome: GenerationOutcome[OutputT] | AbandonedStreamRecord,
    attribute_mapper: AttributeMapper,
) -> None:
    """Set the langchaint.request_failed events and the mapper's attributes on a recording span.

    A non-recording span skips both, because an `AttributeMapper` may be expensive.
    The events are added before the mapper runs.
    Events and mapper attributes use separate guards, so a mapper that raises keeps the events.
    An error whose str() raises can leave the events partial.
    """
    with _guarding_telemetry_failures("adding the langchaint.request_failed events"):
        if span.is_recording():
            _record_request_failed_events(span, outcome)
    _set_built_attributes(span, "attribute_mapper", lambda: attribute_mapper(outcome))


def _apply_content_attributes(
    span: Span, span_config: _SpanConfig, build: Callable[[ContentFilter], SpanAttributes]
) -> None:
    """Set the content attributes `build` returns, when capture is on and the span is recording.

    `build` receives the observer's content filter.
    The content keys are JSON strings, and some of what they serialize is arbitrary application data:
    An application supplies `JSONSchemaTool.args_schema` values verbatim.
    `json.dumps` can reject one of those values.
    A `ContentFilter` can raise or return a part of another kind.
    A build Exception is logged and does not propagate.
    """
    if span_config.capture_message_content:
        _set_built_attributes(span, "content capture", lambda: build(span_config.content_filter))


def _set_error_status(span: Span, error_type: str, description: str) -> None:
    """Set error.type and error status, without letting an OTel exception reach the caller.

    An empty description sets a status without a description.
    """
    _set_span_attributes(span, {"error.type": error_type})
    with _guarding_telemetry_failures("setting the error status"):
        span.set_status(Status(StatusCode.ERROR, description or None))


def _record_tool_exception(span: Span, exc: Exception) -> None:
    """Record the tool function's exception as a span event, set error.type from its class, and set error status.

    error.type uses the exception class name for low-cardinality grouping.
    """
    with _guarding_telemetry_failures("recording the exception"):
        span.record_exception(exc)
    _set_error_status(span, type(exc).__name__, str(exc))


def _tool_call_arguments_attribute(
    tool_call: ToolCall, content_filter: ContentFilter
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.tool.call.arguments from `tool_call`, empty when the filter omits it."""
    filtered = _filtered_part(content_filter, "gen_ai.tool.call.arguments", tool_call)
    if filtered is None:
        return {}
    return {"gen_ai.tool.call.arguments": json.dumps(_tool_call_arguments(filtered.args_json))}


def _dispatch_error_type(outcome: DispatchOutcome) -> str | None:
    """Classify a dispatch outcome for error.type, or None where the call succeeded.

    error.type values "invalid_tool_args" and "unknown_tool" mean the tool function never ran.
    A raising tool function is classified by _record_tool_exception with its exception class name instead.
    """
    match outcome.kind:
        case "handled":
            return "tool_error" if outcome.tool_message.is_error else None
        case "invalid_tool_args" | "unknown_tool":
            return outcome.kind


class _SpanOperation:
    """The span of one observed operation."""

    def __init__(self, span: Span, span_config: _SpanConfig) -> None:
        """Hold the started span and the observer's configuration."""
        self._span = span
        self._span_config = span_config

    @contextmanager
    def current(self) -> Generator[None]:
        """Make the span current for the block."""
        token = context.attach(trace.set_span_in_context(self._span))
        try:
            yield
        finally:
            context.detach(token)

    def end(self) -> None:
        """End the span."""
        self._span.end()


class _GenerationSpan(_SpanOperation):
    """The CLIENT chat span of one input."""

    def conclude(self, outcome: GenerationOutcome[object] | AbandonedStreamRecord) -> None:
        """Set the span's outcome attributes, output content, and status from the input's outcome.

        A `Generation` sets OK status, and a `GenerationError` sets error status and error.type.
        An `AbandonedStreamRecord` leaves the status unset, because the application's code ended the stream.
        """
        _apply_outcome_attributes(self._span, outcome, self._span_config.attribute_mapper)
        _apply_output_content(self._span, outcome, self._span_config)
        if isinstance(outcome, GenerationError):
            _set_error_status(self._span, outcome.kind, outcome.error_text)
        elif outcome.kind != "abandoned_stream":
            _set_ok_status(self._span)


class _DispatchSpan(_SpanOperation):
    """The INTERNAL execute_tool span of one tool dispatch."""

    def conclude(self, outcome: DispatchOutcome | Exception) -> None:
        """Attribute the span from the dispatch outcome, or record the tool function's exception.

        The outcome selects the span status and error.type, as `OtelObserver.dispatch_started` lists.
        With capture on, `gen_ai.tool.call.result` records the outcome's `ToolMessage`.
        """
        if isinstance(outcome, Exception):
            _record_tool_exception(self._span, outcome)
            return
        _apply_content_attributes(
            self._span,
            self._span_config,
            lambda content_filter: {
                "gen_ai.tool.call.result": json.dumps(
                    _tool_call_response_part(
                        outcome.tool_message, content_filter, "gen_ai.tool.call.result"
                    )
                )
            },
        )
        error_type = _dispatch_error_type(outcome)
        if error_type is None:
            _set_ok_status(self._span)
        else:
            _set_error_status(self._span, error_type, error_type)


class OtelObserver:
    """Trace generations and tool dispatches with OTel spans.

    Pass one instance to a backend constructor, `LLM(observer=...)`, or `ToolManager(observer=...)`.
    The OTel SDK configures whether tracing records and where it sends spans.
    An application without an SDK configuration gets non-recording no-op spans.
    This class owns each span's name, kind, and status.
    A custom mapper changes only attributes.
    There is no langchaint.elapsed_seconds attribute:
    The span duration covers request admission, backoff, and completion.
    """

    @overload
    def __init__(
        self,
        *,
        capture_message_content: Literal[True],
        content_filter: ContentFilter | None = None,
        attribute_mapper: AttributeMapper = gen_ai_attributes,
        extra_attributes: SpanAttributes | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        capture_message_content: bool,
        attribute_mapper: AttributeMapper = gen_ai_attributes,
        extra_attributes: SpanAttributes | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None: ...
    def __init__(
        self,
        *,
        capture_message_content: bool,
        content_filter: ContentFilter | None = None,
        attribute_mapper: AttributeMapper = gen_ai_attributes,
        extra_attributes: SpanAttributes | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        """Resolve the tracer once, at construction.

        `capture_message_content` has no default because content capture affects privacy.
        `capture_message_content=True` records bound prompts, tool definitions, inputs, and assistant messages.
        It also records tool arguments and tool results.
        `content_filter` decides per part what those attributes record.
        The overloads accept `content_filter` only with `capture_message_content=True`.
        `content_filter=None` records every part unchanged, including image and audio bytes.
        A filter that keeps everything except inline bytes:

            def drop_binary(name: str, part: ContentPart | AssistantPart) -> ContentPart | AssistantPart | None:
                return None if part.kind in ("image", "audio") else part

        `attribute_mapper` sets the outcome attributes of each chat span.
        It defaults to `gen_ai_attributes`.
        `extra_attributes` applies at the start of every span.
        `extra_attributes=None` supplies no extra attributes.
        Request and dispatch identity attributes replace matching `extra_attributes` keys at span start.
        A key the mapper also emits resolves to the mapper's value, set with the outcome attributes.
        `tracer_provider` decides where spans go, and `tracer_provider=None` uses the global tracer provider.
        Either way, the tracer is named `langchaint.tracing` with the package version.
        That name is every span's instrumentation scope, which identifies langchaint as the span's source.
        """
        self._span_config = _SpanConfig(
            tracer=trace.get_tracer(
                "langchaint.tracing", _PACKAGE_VERSION, tracer_provider=tracer_provider
            ),
            attribute_mapper=attribute_mapper,
            extra_attributes=extra_attributes if extra_attributes is not None else {},
            capture_message_content=capture_message_content,
            content_filter=content_filter if content_filter is not None else _record_every_part,
        )

    def generation_started(
        self, start: GenerationStart
    ) -> ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord]:
        """Open the CLIENT chat span and set its start attributes.

        The span is named "chat {start.model}".
        With capture on, the input content attributes are set at span start and remain on failed spans.
        They are built only for a recording span, so an application without a TracerProvider serializes nothing.
        """
        span_config = self._span_config
        span = span_config.tracer.start_span(
            f"{_CHAT_OPERATION} {start.model}", kind=SpanKind.CLIENT
        )
        _set_span_attributes(span, span_config.extra_attributes)
        _set_span_attributes(span, _request_attributes(start))
        _apply_content_attributes(
            span,
            span_config,
            lambda content_filter: _input_content_attributes(
                start.binding, start.messages, content_filter=content_filter
            ),
        )
        return _GenerationSpan(span, span_config)

    def dispatch_started(
        self, tool_call: ToolCall
    ) -> ObservedOperation[DispatchOutcome | Exception]:
        """Open the INTERNAL execute_tool span and set its identity attributes.

        The span name is "execute_tool {tool_call.name}".
        The identity attributes gen_ai.operation.name, gen_ai.tool.name, and gen_ai.tool.call.id are set at span start.
        With capture on, gen_ai.tool.call.arguments is set at span start.
        `content_filter` sees the `ToolCall` under gen_ai.tool.call.arguments.
        It sees each `ToolMessage` content part under gen_ai.tool.call.result.
        gen_ai.tool.call.arguments uses best-effort JSON deserialization.
        Unparseable text is preserved as a quoted JSON string.
        The outcome selects the span status and error.type:

        | dispatch outcome                    | status | error.type              |
        | ----------------------------------- | ------ | ----------------------- |
        | DispatchHandled, is_error False     | OK     | absent                  |
        | DispatchHandled, is_error True      | ERROR  | tool_error              |
        | DispatchInvalidToolArgs             | ERROR  | invalid_tool_args       |
        | DispatchUnknownTool                 | ERROR  | unknown_tool            |
        | the tool function raised            | ERROR  | the exception class name|

        invalid_tool_args and unknown_tool mean that the tool function never ran.
        """
        span_config = self._span_config
        span = span_config.tracer.start_span(
            f"{_EXECUTE_TOOL_OPERATION} {tool_call.name}", kind=SpanKind.INTERNAL
        )
        _set_span_attributes(span, span_config.extra_attributes)
        _set_span_attributes(
            span,
            {
                "gen_ai.operation.name": _EXECUTE_TOOL_OPERATION,
                "gen_ai.tool.name": tool_call.name,
                "gen_ai.tool.call.id": tool_call.id,
            },
        )
        _apply_content_attributes(
            span,
            span_config,
            lambda content_filter: _tool_call_arguments_attribute(tool_call, content_filter),
        )
        return _DispatchSpan(span, span_config)


__all__ = [
    "AttributeMapper",
    "ContentFilter",
    "OtelObserver",
    "SpanAttributes",
    "gen_ai_attributes",
]
