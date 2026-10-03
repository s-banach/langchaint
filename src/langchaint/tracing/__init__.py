"""Trace langchaint calls with OTel spans.

Importing this subpackage requires `opentelemetry-api`.
Applications configure the OTel SDK.

Pass one `OtelObserver` to a backend constructor, as in `OpenAI(observer=OtelObserver(...))`.
Every `LLM`, binding, and stream that backend creates is then traced, and the application's code is otherwise unchanged.
A `ToolManager` that `bind` builds from a tool sequence traces its dispatches.
A `ToolManager` the application builds traces its dispatches when built with `observer=`.

Each generation call opens one CLIENT span.
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
Request and completion attributes replace matching `extra_attributes` keys.
Required `gen_ai.operation.name` values also replace matching `extra_attributes` keys.

Chat and stream spans use `gen_ai.operation.name="chat"`.
They report provider, request model, response model, finish reasons, token usage, attempts, and cost.
They report the standard attributes for each set request field.
They report `gen_ai.output.type` for every call.
With capture enabled, they report system instructions, tool definitions, input messages, and output messages.
Stream spans also report `gen_ai.request.stream=True`.
A stream span reports `gen_ai.response.time_to_first_chunk` once an item arrives, including for a stream left early.
A stream span whose block exits before the conclusion reports its usage and cost.
Its status stays unset, because the call did not fail.
Tool spans use `gen_ai.operation.name="execute_tool"` and report tool name and tool call id.
With capture enabled, tool spans report arguments and results.

`langchaint.*` names attempts and cost because the convention has no matching keys.
Each failed attempt adds `langchaint.attempt_failed` with `error_text` and `elapsed_seconds`.

Each span starts and ends exactly once, including for failed, cancelled, and abandoned streams.
A mapper may set only attribute names and values.
A mapper cannot change the span name, kind, or status.
Telemetry failures, including mapper and content filter failures, are logged and never propagate.

Attribute names except `gen_ai.request.reasoning.level` match opentelemetry-semantic-conventions 0.64b0.
`gen_ai.request.reasoning.level` comes from the OpenTelemetry semantic-conventions-genai repository.
`GenAiOperationNameValues.CHAT` and `.EXECUTE_TOOL` define the operation values.
Tool identity uses `gen_ai.tool.name` and `gen_ai.tool.call.id`.
Reasoning usage uses `gen_ai.usage.reasoning.output_tokens`.
Readable reasoning uses `ReasoningPart` in content payloads.
Content payloads follow the convention's JSON schemas.
`gen_ai.tool.call.arguments` may carry a JSON value other than an object.

"""

import base64
import importlib.metadata
import json
import logging
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal, NoReturn, overload

try:
    from opentelemetry import context, trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
except ModuleNotFoundError as exc:
    if exc.name is not None and not exc.name.startswith("opentelemetry"):
        raise
    raise ModuleNotFoundError(
        "langchaint's tracing subpackage requires opentelemetry-api; install langchaint[tracing]."
    ) from exc

from langchaint.adapter import Binding
from langchaint.common.messages import (
    AssistantMessage,
    ContentPart,
    Message,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnPart,
    UserMessage,
)
from langchaint.common.observed_operation import ObservedOperation
from langchaint.generation.call import AbandonedCallRecord
from langchaint.generation.errors import GenerationError
from langchaint.generation.observer import GenerationStart
from langchaint.generation.response import CallResult, GenerateResult
from langchaint.tools import DispatchOutcome, ToolSchema

type SpanAttributeValue = str | bool | int | float | list[str] | tuple[str, ...]
"""One span attribute's value."""

type SpanAttributes = Mapping[str, SpanAttributeValue]
"""A span's attributes, keyed by name."""

type AttributeMapper = Callable[[CallResult[object] | AbandonedCallRecord], SpanAttributes]
"""Maps one generate result, or a stream's `abandoned` record, to its span attributes.

The mapper reads the fields shared by each `CallResult` variant and `AbandonedCallRecord`.
No mapper receives the call's input messages, so `gen_ai_attributes` cannot put a prompt on a span.
A custom mapper can reach `CallResult.raw`, which holds the SDK response by reference.
`AbandonedCallRecord` has no `raw`, because no response completed.
`capture_message_content` controls prompt capture because `OtelObserver` receives the input messages at call start.
"""

type ContentFilter = Callable[[str, ContentPart | TurnPart], ContentPart | TurnPart | None]
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
    _attribute_name: str, part: ContentPart | TurnPart
) -> ContentPart | TurnPart:
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
    content_filter: ContentFilter, attribute_name: str, part: TurnPart
) -> TurnPart | None: ...
def _filtered_part(
    content_filter: ContentFilter, attribute_name: str, part: ContentPart | TurnPart
) -> ContentPart | TurnPart | None:
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


def _filtered_content(
    content: str | tuple[ContentPart, ...], content_filter: ContentFilter, attribute_name: str
) -> tuple[ContentPart, ...]:
    """Pass each `ContentPart` through the filter, with a str as one `TextPart`.

    A bound system prompt is a str or a tuple of `TextPart`, so it passes through this function too.
    """
    content_parts: tuple[ContentPart, ...] = (
        (TextPart(text=content),) if isinstance(content, str) else content
    )
    kept = (_filtered_part(content_filter, attribute_name, part) for part in content_parts)
    return tuple(part for part in kept if part is not None)


def _filtered_turn(
    turn: tuple[TurnPart, ...], content_filter: ContentFilter, attribute_name: str
) -> tuple[TurnPart, ...]:
    """Pass each `TurnPart` through the filter."""
    kept = (_filtered_part(content_filter, attribute_name, part) for part in turn)
    return tuple(part for part in kept if part is not None)


def _filtered_tool_message(
    message: ToolMessage, content_filter: ContentFilter, attribute_name: str
) -> ToolMessage:
    """Copy the message with its content filtered; `tool_call_id` and `is_error` never reach the filter."""
    return ToolMessage(
        tool_call_id=message.tool_call_id,
        content=_filtered_content(message.content, content_filter, attribute_name),
        is_error=message.is_error,
    )


def _filtered_message(message: Message, content_filter: ContentFilter) -> Message:
    """Copy one input message with its parts filtered under gen_ai.input.messages."""
    attribute_name = "gen_ai.input.messages"
    if message.kind == "user":
        return UserMessage(
            content=_filtered_content(message.content, content_filter, attribute_name)
        )
    if message.kind == "tool":
        return _filtered_tool_message(message, content_filter, attribute_name)
    return AssistantMessage(turn=_filtered_turn(message.turn, content_filter, attribute_name))


def _filtered_messages(
    messages: Sequence[Message], content_filter: ContentFilter
) -> tuple[Message, ...]:
    """Copy the input messages with every part filtered."""
    return tuple(_filtered_message(message, content_filter) for message in messages)


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
    content_filter: ContentFilter


_CONVENTION_FINISH_REASONS: Mapping[StopReason, str] = {
    "end_turn": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
}
"""The StopReason values with an exact counterpart in the convention's finish-reason vocabulary.

refusal, context_window_exceeded, and other are absent deliberately and pass through unmapped:
the convention's content_filter means a provider filter blocked content.
No convention value corresponds to a context-window overflow or to other.
The convention's enum is open because the output schema permits the enum or a string.
Unmapped values pass through unchanged.
"""


_NO_COMPLETED_TURN_FINISH_REASON = "error"
"""What gen_ai.output.messages reports for a turn whose result states no stop reason.

The convention's `error` enum member identifies a failed generation.
`gen_ai.response.finish_reasons` is optional and is omitted.
The per-message field is required, so a turn recorded from a failure needs a value.
"""


def _finish_reason(stop_reason: StopReason) -> str:
    """Map a StopReason onto the convention's finish-reason vocabulary, passing unmapped values through.

    gen_ai.response.finish_reasons and gen_ai.output.messages use this mapping.
    """
    return _CONVENTION_FINISH_REASONS.get(stop_reason, stop_reason)


def gen_ai_attributes[OutputT](
    result: CallResult[OutputT] | AbandonedCallRecord,
) -> SpanAttributes:
    """Map a generate result to GenAI-convention span attributes plus langchaint scalars.

    `result` supplies the response identity, usage, stop reason, and attempt records.
    This is the default attribute_mapper.
    A custom AttributeMapper can extend its result.
    Extension keys must use the application's namespace because langchaint.* is reserved.
    `extra_attributes` sets a constant on every span.
    Each call builds and returns a fresh dict, so extending the result mutates nothing shared.
    This function reads only the fields shared by every `CallResult` variant and `AbandonedCallRecord`.
    The langchaint.* prefix is used only when the GenAI convention has no corresponding key.
    This applies to langchaint.attempts and langchaint.cost_in_usd.
    gen_ai.usage.input_tokens is Usage.input_tokens_total.
    The cache-read and cache-creation attributes are parts of that total.
    No cache_none counter is emitted because it is derived.
    gen_ai.response.finish_reasons contains the mapped stop_reason and is omitted when stop_reason is None.
    gen_ai.response.model is the last attempt's model_served and is omitted when unavailable.
    gen_ai.response.time_to_first_chunk is the last attempt's seconds_to_first_item, settled or cut off.
    That value runs from sending the request to the first stream item, so only a stream has it.
    The usage and cost attributes are the call's paid totals across every attempt.
    `result.usage` has that scope.
    `langchaint.attempt_failed` span events retain per-attempt detail.
    """
    usage = result.usage
    records = result.attempt_records
    final_record = records[-1] if records else None
    final_settled_record = (
        final_record if final_record is not None and final_record.kind == "settled" else None
    )
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
    if final_settled_record is not None and final_settled_record.model_served is not None:
        attributes["gen_ai.response.model"] = final_settled_record.model_served
    if final_record is not None and final_record.seconds_to_first_item is not None:
        attributes["gen_ai.response.time_to_first_chunk"] = final_record.seconds_to_first_item
    if result.stop_reason is not None:
        attributes["gen_ai.response.finish_reasons"] = [_finish_reason(result.stop_reason)]
    return attributes


def _blob_part(modality: str, media_type: str, data: bytes) -> dict[str, object]:
    """Render inline bytes as the convention's BlobPart with standard base64 content."""
    return {
        "type": "blob",
        "modality": modality,
        "mime_type": media_type,
        "content": base64.b64encode(data).decode("ascii"),
    }


def _content_parts(content: str | tuple[ContentPart, ...]) -> list[dict[str, object]]:
    """Render a MessageContent as the convention's parts array.

    A str becomes one text part.
    ImagePart and AudioPart become BlobPart objects carrying their bytes.
    ImageUrlPart becomes an image uri part with its optional media_type as mime_type.
    gen_ai.system_instructions uses the same rendering because its items share the text part shape.
    `cache_breakpoint` is omitted because the convention has no corresponding field.
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    parts: list[dict[str, object]] = []
    for part in content:
        match part.kind:
            case "text":
                parts.append({"type": "text", "content": part.text})
            case "image":
                parts.append(_blob_part("image", part.media_type, part.data))
            case "image_url":
                image_uri: dict[str, object] = {
                    "type": "uri",
                    "modality": "image",
                    "uri": part.url,
                }
                if part.media_type is not None:
                    image_uri["mime_type"] = part.media_type
                parts.append(image_uri)
            case "audio":
                parts.append(_blob_part("audio", part.media_type, part.data))
    return parts


def _finite_float(number_text: str) -> float:
    """Parse a JSON number, rejecting one that overflows the float range.

    `json.dumps` writes a non-finite float as the bare token Infinity or NaN.
    Those tokens are not JSON, so this function rejects them.

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
        ValueError: `json.loads` passes one of those three literals.
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
    A turn holding only text-free parts therefore renders as an empty parts array, not as a missing message.
    """
    parts: list[dict[str, object]] = []
    for part in turn:
        match part.kind:
            case "reasoning_part":
                if part.text:
                    parts.append({"type": "reasoning", "content": part.text})
            case "text":
                if part.text:
                    parts.append({"type": "text", "content": part.text})
            case "tool_call":
                parts.append({
                    "type": "tool_call",
                    "id": part.id,
                    "name": part.name,
                    "arguments": _tool_call_arguments(part.args_json),
                })
            case "raw_part":
                pass
    return parts


def _message(message: Message) -> dict[str, object]:
    """Render one Message as the convention's {role, parts} shape.

    A `ToolMessage` becomes a tool_call_response part inside a tool-role message.
    """
    if message.kind == "user":
        return {"role": "user", "parts": _content_parts(message.content)}
    if message.kind == "tool":
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
        "is_error": message.is_error,
        "response": _content_parts(message.content),
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
    """Build the input-side content attributes for one call, each a JSON string.

    OTel attribute values cannot nest, so structured values use the permitted JSON string form.
    A key whose source is empty or absent is omitted.
    `system_prompt=None` omits gen_ai.system_instructions, and so does a prompt whose every part filters away.
    No bound tools omits gen_ai.tool.definitions.
    These omissions are indistinguishable from disabled capture.
    gen_ai.tool.definitions does not pass through `content_filter` because a tool schema is not a part.
    """
    attributes: dict[str, SpanAttributeValue] = {}
    if binding.system_prompt is not None:
        system_instructions = _content_parts(
            _filtered_content(binding.system_prompt, content_filter, "gen_ai.system_instructions")
        )
        if system_instructions:
            attributes["gen_ai.system_instructions"] = json.dumps(system_instructions)
    if binding.tool_schemas:
        attributes["gen_ai.tool.definitions"] = json.dumps(_tool_definitions(binding.tool_schemas))
    input_messages = [
        _message(message) for message in _filtered_messages(messages, content_filter)
    ]
    if input_messages:
        attributes["gen_ai.input.messages"] = json.dumps(input_messages)
    return attributes


def _request_attributes(start: GenerationStart) -> dict[str, SpanAttributeValue]:
    """Map stored request configuration onto standard OTel attributes."""
    binding = start.binding
    attributes: dict[str, SpanAttributeValue] = {
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
    turn: tuple[TurnPart, ...], stop_reason: StopReason | None
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.output.messages from one assistant turn.

    One function for the success and the failure paths, so one turn renders the same whichever reported it.
    One key contains one message for one turn. A missing stop_reason uses the required "error" enum member.
    """
    return {
        "gen_ai.output.messages": json.dumps([
            {
                "role": "assistant",
                "parts": _turn_parts(turn),
                "finish_reason": (
                    _NO_COMPLETED_TURN_FINISH_REASON
                    if stop_reason is None
                    else _finish_reason(stop_reason)
                ),
            }
        ])
    }


def _apply_output_content[OutputT](
    span: Span, result: CallResult[OutputT] | AbandonedCallRecord, span_config: _SpanConfig
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
        span,
        lambda: _output_content_attributes(
            _filtered_turn(
                assistant_message.turn, span_config.content_filter, "gen_ai.output.messages"
            ),
            stop_reason,
        ),
    )


def _record_attempt_failed_events[OutputT](
    span: Span, result: CallResult[OutputT] | AbandonedCallRecord
) -> None:
    """Add one langchaint.attempt_failed event per failed attempt in the result's records.

    Each event carries the attempt's error text and its own `elapsed_seconds`.
    Events are stamped at recording time because the records carry only monotonic brackets.
    They answer the first question a slow traced call raises: was it the request or the retries.
    """
    for record in result.attempt_records:
        if record.kind == "settled" and record.error is not None:
            span.add_event(
                "langchaint.attempt_failed",
                {"error_text": str(record.error), "elapsed_seconds": record.elapsed_seconds},
            )


def _apply_result_attributes[OutputT](
    span: Span,
    result: CallResult[OutputT] | AbandonedCallRecord,
    attribute_mapper: AttributeMapper,
) -> None:
    """Set the mapper's attributes and the langchaint.attempt_failed events on a recording span.

    Success, `GenerationError`, and `AbandonedCallRecord` values carry the shared `CallResult` fields.
    Other exceptions do not carry those fields.
    A non-recording span skips the mapper because an `AttributeMapper` may be expensive.
    A mapper exception is caught and logged at warning level.
    `langchaint.attempt_failed` events are added before the mapper runs.
    Existing span attributes remain when the mapper raises.
    Events and mapper attributes use separate guards.
    An error whose str() raises can leave the events partial.
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
    An application supplies `JSONSchemaTool.args_schema` values verbatim.
    `json.dumps` can reject one of those values.
    A `ContentFilter` can raise or return a part of another kind.
    A build Exception is logged and does not propagate. Existing span attributes remain.
    Building inside the is_recording guard is why the input messages are serialized here rather than earlier:
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


def _set_generation_error_status(span: Span, error: GenerationError) -> None:
    """Set error.type and error status from a terminal GenerationError."""
    _set_span_attribute(span, "error.type", error.kind)
    with _guarding_telemetry_failures("setting the error status"):
        status = (
            Status(StatusCode.ERROR, error.error_text)
            if error.error_text
            else Status(StatusCode.ERROR)
        )
        span.set_status(status)


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


def _tool_call_arguments_attribute(
    call: ToolCall, content_filter: ContentFilter
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.tool.call.arguments from the call, empty when the filter omits the call."""
    filtered = _filtered_part(content_filter, "gen_ai.tool.call.arguments", call)
    if filtered is None:
        return {}
    return {"gen_ai.tool.call.arguments": json.dumps(_tool_call_arguments(filtered.args_json))}


def _dispatch_error_type(outcome: DispatchOutcome) -> str | None:
    """Classify a dispatch outcome for error.type, or None where the call succeeded.

    error.type values "invalid_tool_args" and "unknown_tool" mean the tool function never ran.
    A raising tool function is classified by _record_other_exception with its exception class name instead.
    """
    match outcome.kind:
        case "handled":
            return "tool_error" if outcome.tool_message.is_error else None
        case "invalid_tool_args":
            return "invalid_tool_args"
        case "unknown_tool":
            return "unknown_tool"


@contextmanager
def _span_made_current(span: Span) -> Generator[None]:
    """Make `span` the current span for the block."""
    token = context.attach(trace.set_span_in_context(span))
    try:
        yield
    finally:
        context.detach(token)


class _SpanOperation:
    """The span of one observed operation."""

    def __init__(self, span: Span, span_config: _SpanConfig) -> None:
        """Hold the started span and the observer's configuration."""
        self._span = span
        self._span_config = span_config

    def current(self) -> AbstractContextManager[None]:
        """Make the span current for the block."""
        return _span_made_current(self._span)

    def end(self) -> None:
        """End the span."""
        self._span.end()


class _GenerationSpan(_SpanOperation):
    """The CLIENT chat span of one generation call."""

    def conclude(self, outcome: GenerateResult[object] | AbandonedCallRecord | Exception) -> None:
        """Attribute the span from the call's result, `abandoned` record, `GenerationError`, or other exception.

        Each value except another exception carries the call result attributes and the output content.
        An `AbandonedCallRecord` leaves the status unset, because the application's code ended the stream.
        Another exception is recorded as a span event.
        """
        if isinstance(outcome, GenerationError):
            self._record_call_result(outcome)
            _set_generation_error_status(self._span, outcome)
        elif isinstance(outcome, Exception):
            _record_other_exception(self._span, outcome)
        elif isinstance(outcome, AbandonedCallRecord):
            self._record_call_result(outcome)
        else:
            self._record_call_result(outcome)
            _set_ok_status(self._span)

    def _record_call_result(self, result: CallResult[object] | AbandonedCallRecord) -> None:
        _apply_result_attributes(self._span, result, self._span_config.attribute_mapper)
        _apply_output_content(self._span, result, self._span_config)


class _DispatchSpan(_SpanOperation):
    """The INTERNAL execute_tool span of one tool dispatch."""

    def conclude(self, outcome: DispatchOutcome | Exception) -> None:
        """Attribute the span from the dispatch outcome, or record the tool function's exception.

        The outcome selects the span status and error.type, as `OtelObserver.dispatch_started` lists.
        With capture on, `gen_ai.tool.call.result` records the outcome's `ToolMessage`.
        """
        if isinstance(outcome, Exception):
            _record_other_exception(self._span, outcome)
            return
        error_type = _dispatch_error_type(outcome)
        if error_type is not None:
            _set_span_attribute(self._span, "error.type", error_type)
        if self._span_config.capture_message_content:
            content_filter = self._span_config.content_filter
            _apply_content_attributes(
                self._span,
                lambda: {
                    "gen_ai.tool.call.result": json.dumps(
                        _tool_call_response_part(
                            _filtered_tool_message(
                                outcome.tool_message, content_filter, "gen_ai.tool.call.result"
                            )
                        )
                    )
                },
            )
        if error_type is None:
            _set_ok_status(self._span)
        else:
            with _guarding_telemetry_failures("setting the error status"):
                self._span.set_status(Status(StatusCode.ERROR, error_type))


class OtelObserver:
    """Trace generation calls and tool dispatches with OTel spans.

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
        tracer: Tracer | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        capture_message_content: bool,
        attribute_mapper: AttributeMapper = gen_ai_attributes,
        extra_attributes: SpanAttributes | None = None,
        tracer: Tracer | None = None,
    ) -> None: ...
    def __init__(
        self,
        *,
        capture_message_content: bool,
        content_filter: ContentFilter | None = None,
        attribute_mapper: AttributeMapper = gen_ai_attributes,
        extra_attributes: SpanAttributes | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Resolve the tracer once, at construction.

        `capture_message_content` has no default because content capture affects privacy.
        `capture_message_content=True` records bound prompts, tool definitions, inputs, and assistant turns.
        It also records tool arguments and tool results.
        `content_filter` decides per part what those attributes record.
        The overloads accept `content_filter` only with `capture_message_content=True`.
        `content_filter=None` records every part unchanged, including image and audio bytes.
        A filter that keeps everything except inline bytes:

            def drop_binary(name: str, part: ContentPart | TurnPart) -> ContentPart | TurnPart | None:
                return None if part.kind in ("image", "audio") else part

        `attribute_mapper` sets the completion attributes of each chat span.
        It defaults to `gen_ai_attributes`.
        `extra_attributes` applies at the start of every span.
        `extra_attributes=None` supplies no extra attributes.
        Request and dispatch identity attributes replace matching `extra_attributes` keys at span start.
        A key the mapper also emits resolves to the mapper's value, set at completion.
        `tracer=None` resolves `trace.get_tracer("langchaint.tracing", <package version>)` during construction.
        """
        self._span_config = _SpanConfig(
            tracer=(
                tracer
                if tracer is not None
                else trace.get_tracer("langchaint.tracing", _PACKAGE_VERSION)
            ),
            attribute_mapper=attribute_mapper,
            extra_attributes=extra_attributes if extra_attributes is not None else {},
            capture_message_content=capture_message_content,
            content_filter=content_filter if content_filter is not None else _record_every_part,
        )

    def generation_started(
        self, start: GenerationStart
    ) -> ObservedOperation[GenerateResult[object] | AbandonedCallRecord]:
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
        _set_span_attribute(span, "gen_ai.operation.name", _CHAT_OPERATION)
        _set_span_attributes(span, _request_attributes(start))
        if span_config.capture_message_content:
            _apply_content_attributes(
                span,
                lambda: _input_content_attributes(
                    start.binding, start.messages, content_filter=span_config.content_filter
                ),
            )
        return _GenerationSpan(span, span_config)

    def dispatch_started(self, call: ToolCall) -> ObservedOperation[DispatchOutcome]:
        """Open the INTERNAL execute_tool span and set its identity attributes.

        The span name is "execute_tool {call.name}".
        The identity attributes gen_ai.operation.name, gen_ai.tool.name, and gen_ai.tool.call.id are set at span start.
        With capture on, gen_ai.tool.call.arguments is set at span start.
        `content_filter` sees the `ToolCall` under gen_ai.tool.call.arguments.
        It sees each result part under gen_ai.tool.call.result.
        gen_ai.tool.call.arguments uses best-effort JSON deserialization.
        Unparseable text is preserved as a quoted JSON string.
        The outcome selects the span status and error.type:

        | dispatch result                     | status | error.type              |
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
            f"{_EXECUTE_TOOL_OPERATION} {call.name}", kind=SpanKind.INTERNAL
        )
        _set_span_attributes(span, span_config.extra_attributes)
        _set_span_attributes(
            span,
            {
                "gen_ai.operation.name": _EXECUTE_TOOL_OPERATION,
                "gen_ai.tool.name": call.name,
                "gen_ai.tool.call.id": call.id,
            },
        )
        if span_config.capture_message_content:
            _apply_content_attributes(
                span, lambda: _tool_call_arguments_attribute(call, span_config.content_filter)
            )
        return _DispatchSpan(span, span_config)


__all__ = [
    "AttributeMapper",
    "ContentFilter",
    "OtelObserver",
    "SpanAttributes",
    "gen_ai_attributes",
]
