"""Trace langchaint generations, tool dispatches, and embedding requests with OTel spans.

Importing this subpackage requires `opentelemetry-api`.
Applications configure the OTel SDK.

Pass one `OtelObserver` to a backend constructor, as in `OpenAI(observer=OtelObserver(...))`.
A `ToolManager` built from a tool sequence by a provider factory or `bind` traces its dispatches.
A `ToolManager` the application builds traces its dispatches when built with `observer=`.
Every `EmbeddingModel` that backend creates traces its requests.

Each input opens one CLIENT span.
`generate_many` opens one span per started input.
`generate_many_records` opens one span per input that requires generation.
Restored records open no span.
A stream opens one CLIENT span when its handle is entered.
Each tool dispatch opens one INTERNAL `execute_tool` span.
`ToolManager.dispatch_many` uses `dispatch` and gets one span per tool call.
`precomputed` opens no span because it executes no tool.
Each embedding request opens one CLIENT `embeddings` span, and each retry is a new request.
A generation span is current during its retry loop, and a dispatch span is current while the tool function runs.
An embeddings span is current while its request is awaited.
Spans started there, such as HTTP client spans, nest under them.
A stream span is never current, because the application's code runs between stream items.

Each span starts with `extra_attributes`, followed by request or dispatch identity attributes.
Request and dispatch identity attributes replace matching `extra_attributes` keys.
langchaint collects content and outcome attributes for recording when the span ends.
Collected attributes replace matching start attributes.
If a `ContentFilter` raises or returns a part of another kind, langchaint logs the error.
For each chat span, langchaint builds the input attributes together.
The input attributes are `gen_ai.system_instructions`, `gen_ai.tool.definitions`, and `gen_ai.input.messages`.
If filtering system instructions or input messages fails, langchaint does not add these three input attributes.
If filtering output messages fails, langchaint does not add `gen_ai.output.messages`.
If filtering tool arguments fails on an `execute_tool` span, langchaint does not add `gen_ai.tool.call.arguments`.
If filtering tool results fails on an `execute_tool` span, langchaint does not add `gen_ai.tool.call.result`.
A filtering failure leaves any matching values supplied through `extra_attributes` unchanged.
Each span kind's attribute transform, such as `chat_attribute_transform`, edits the attributes when the span ends.
Each transform receives the outcome and every attribute, including the start attributes.
The span records the returned attributes, and a returned key replaces the attribute of the same name.
A start attribute missing from the result stays because the OTel `Span` API cannot remove an attribute.
If a transform raises, langchaint logs the exception.
The span keeps its start attributes and any existing `error.type`.
Extension keys must use the application's namespace because langchaint.* is reserved.
Organisation-wide redaction of recorded text belongs in an OpenTelemetry Collector processor.

Chat and stream spans use `gen_ai.operation.name="chat"`.
They report provider, request model, response model, finish reasons, token usage, request count, and cost.
They report the standard attributes for each set request field.
They report `gen_ai.output.type` for every input.
With capture enabled, they report system instructions, tool definitions, input messages, and output messages.
Stream spans also report `gen_ai.request.stream=True`.
A stream span reports `gen_ai.response.time_to_first_chunk` once an item arrives, including for a stream left early.
A stream span whose block exits before the stream stores a `GenerationOutcome` reports its usage and cost.
Its status stays unset, because handling the input did not fail.
Tool spans use `gen_ai.operation.name="execute_tool"` and `gen_ai.tool.type="function"`.
They report tool name and tool call id.
With capture enabled, tool spans report arguments and results.
Embeddings spans use `gen_ai.operation.name="embeddings"`.
They report provider, request model, encoding formats, dimension count, response model, input tokens, and cost.
The convention defines no content attribute for embedding inputs, so embeddings spans record no content.

`langchaint.*` names the request count and cost because the convention has no matching keys.
A chat span adds one `langchaint.request_failed` event per failed request, with `error_text` and `elapsed_seconds`.

Each span starts and ends exactly once, including for failed, cancelled, and abandoned streams.
Telemetry failures, including transform and content filter failures, are logged and never propagate.

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
from typing import overload

from pydantic import TypeAdapter, ValidationError

try:
    from opentelemetry import context, trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, TracerProvider
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
    ToolCall,
    ToolMessage,
    content_parts,
)
from langchaint.common.observed_operation import ObservedOperation
from langchaint.embedding_requests import EmbeddingRequestRecord, EmbeddingRequestStart
from langchaint.generation.errors import GenerationError
from langchaint.generation.observer import GenerationStart
from langchaint.generation.request_history import AbandonedStreamRecord
from langchaint.generation.response import GenerationOutcome
from langchaint.tools import DispatchOutcome, ToolSchema

type SpanAttributeValue = str | bool | int | float | list[str] | tuple[str, ...]
"""One span attribute's value."""

type SpanAttributes = Mapping[str, SpanAttributeValue]
"""A span's attributes, keyed by name."""

type ChatAttributeTransform = Callable[
    [GenerationOutcome[object] | AbandonedStreamRecord | None, SpanAttributes], SpanAttributes
]
"""Edits a chat span's attributes once, when the span ends.

The first argument is `None` for an input cancelled before its outcome.
"""

type ExecuteToolAttributeTransform = Callable[
    [DispatchOutcome | Exception | None, SpanAttributes], SpanAttributes
]
"""Edits an execute_tool span's attributes once, when the span ends.

The first argument is `None` for a dispatch cancelled before its outcome.
The attributes include `gen_ai.tool.name`, so a transform can edit one tool's spans.
When the tool function raises, the span's exception event and status description record its message.
The transform cannot change either.
"""

type EmbeddingsAttributeTransform = Callable[
    [EmbeddingRequestRecord | None, SpanAttributes], SpanAttributes
]
"""Edits an embeddings span's attributes once, when the span ends.

The first argument is `None` for a request cancelled before it settled.
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
When the filter omits every content part, a tool message records one `tool_call_response` part.
Its `response` array is empty.
A bare `str` prompt or content is passed as one `TextPart`.
Message-level fields such as `role`, `tool_call_id`, `is_error`, and `finish_reason` never reach the filter.
"""


def _record_every_part(
    _attribute_name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart:
    return part


def _keep_every_attribute(_outcome: object, attributes: SpanAttributes) -> SpanAttributes:
    return attributes


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
        f"the content filter for {attribute_name} returned {type(filtered).__name__} for {type(part).__name__}"
    )


_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_PACKAGE_VERSION = importlib.metadata.version("langchaint")
_CHAT_OPERATION = "chat"
"""The GenAI operation value for a chat completion."""

_EXECUTE_TOOL_OPERATION = "execute_tool"
"""The GenAI operation value for a tool execution."""

_EMBEDDINGS_OPERATION = "embeddings"
"""The GenAI operation value for an embeddings request."""

_EMBEDDINGS_ENCODING_FORMATS = ("float",)

_logger = logging.getLogger("langchaint.tracing")


_NO_STOP_REASON_FINISH_REASON = "error"
"""What gen_ai.output.messages reports for an assistant message whose outcome states no stop reason.

The convention uses `error` when handling an input fails.
`gen_ai.response.finish_reasons` is optional and is omitted.
The per-message field is required, so an assistant message recorded from a failure needs a value.
"""


def _finish_reason(stop_reason: StopReason) -> str:
    """Translate the token limit to the convention's `length` finish reason.

    Other values pass through because the convention permits arbitrary strings.
    The convention's `content_filter` means a provider blocked content and does not cover every refusal.
    """
    return "length" if stop_reason == "max_completion_tokens" else stop_reason


def _outcome_attributes(
    outcome: GenerationOutcome[object] | AbandonedStreamRecord,
) -> dict[str, SpanAttributeValue]:
    """Build outcome attributes from billing across every request and the final request's response details.

    gen_ai.usage.input_tokens is Usage.input_tokens_total.
    The cache-read and cache-write attributes are parts of that total.
    No cache_none counter is emitted because it is derived.
    First-chunk timing covers the final request even when the stream ends early.
    """
    usage = outcome.usage
    records = outcome.request_history.records
    final_record = records[-1] if records else None
    attributes: dict[str, SpanAttributeValue] = {
        "gen_ai.usage.input_tokens": usage.input_tokens_total,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "gen_ai.usage.reasoning.output_tokens": usage.output_tokens_reasoning,
        "gen_ai.usage.cache_read.input_tokens": usage.input_tokens_cache_read,
        "gen_ai.usage.cache_write.input_tokens": usage.input_tokens_cache_write,
        "langchaint.request_count": len(records),
        "langchaint.cost_in_usd": usage.cost_in_usd,
    }
    if (
        final_record is not None
        and final_record.kind == "settled"
        and final_record.model_served is not None
    ):
        attributes["gen_ai.response.model"] = final_record.model_served
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
    parts: tuple[ContentPart, ...], content_filter: ContentFilter, attribute_name: str
) -> list[dict[str, object]]:
    """Filter content parts and render the kept ones as the convention's parts array.

    A bound system prompt renders here too, as `TextPart` values.
    gen_ai.system_instructions items share the text part shape.
    """
    kept = (_filtered_part(content_filter, attribute_name, part) for part in parts)
    return [_content_part(part) for part in kept if part is not None]


def _tool_call_arguments(args_json: str) -> JsonValue:
    """Deserialize tool-call argument JSON, preserving text that cannot produce a standard JSON value.

    The convention requests best-effort deserialization of serialized arguments.
    Malformed JSON and non-finite floats use the original text, including numbers such as 1e400 that overflow a float.
    The fallback preserves invalid tool arguments without emitting non-finite JSON numbers.
    """
    try:
        return _JSON_VALUE_ADAPTER.validate_json(args_json)
    except ValidationError:
        return args_json


def _assistant_part(part: AssistantPart) -> dict[str, object] | None:
    """Render one AssistantPart as the convention's part object, or None when it records nothing.

    TextPart emits its text, including empty text.
    ReasoningPart renders as None when its text is empty.
    ReasoningPart.raw is opaque and is never emitted.
    A RawPart renders as None because it has no text.
    """
    match part.kind:
        case "reasoning":
            return {"type": "reasoning", "content": part.text} if part.text else None
        case "text":
            return {"type": "text", "content": part.text}
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
    """Filter assistant content and render its kept parts as the convention's parts array, in order.

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
            otel_parts = _recorded_content(message.content_parts, content_filter, attribute_name)
            return {"role": "user", "parts": otel_parts}
        case "tool":
            otel_parts = [_tool_call_response_part(message, content_filter, attribute_name)]
            return {"role": "tool", "parts": otel_parts}
        case "assistant":
            otel_parts = _recorded_assistant_parts(
                message.content_parts, content_filter, attribute_name
            )
            return {"role": "assistant", "parts": otel_parts}


def _tool_call_response_part(
    message: ToolMessage, content_filter: ContentFilter, attribute_name: str
) -> dict[str, object]:
    """Render one ToolMessage as the convention's tool_call_response part, with its content filtered.

    Tool results use the same `tool_call_response` structure in chat and tool spans.
    The attributes are `gen_ai.input.messages` on chat spans and `gen_ai.tool.call.result` on tool spans.
    `tool_call_id` and `is_error` never reach the filter.
    """
    return {
        "type": "tool_call_response",
        "id": message.tool_call_id,
        "is_error": message.is_error,
        "response": _recorded_content(message.content_parts, content_filter, attribute_name),
    }


def _tool_definitions(tool_schemas: tuple[ToolSchema, ...]) -> list[dict[str, object]]:
    """Render the bound tool schemas as the convention's tool-definition array.

    description and parameters record what the model received despite the schema's size warning.
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
    `gen_ai.input.messages` is omitted when `messages` is empty.
    `system_prompt=None` omits gen_ai.system_instructions, and so does a prompt whose every part filters away.
    `gen_ai.tool.definitions` is omitted when no tools are bound.
    These omissions are indistinguishable from disabled capture.
    gen_ai.tool.definitions does not pass through `content_filter` because a tool schema is not a part.
    """
    attributes: dict[str, SpanAttributeValue] = {}
    if binding.system_prompt is not None:
        system_instructions = _recorded_content(
            content_parts(binding.system_prompt), content_filter, "gen_ai.system_instructions"
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
        "gen_ai.output.type": "text" if start.response_format is str else "json",
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
    parts: tuple[AssistantPart, ...],
    stop_reason: StopReason | None,
    content_filter: ContentFilter,
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.output.messages with one assistant message."""
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


def _record_request_failed_events(
    span: Span, outcome: GenerationOutcome[object] | AbandonedStreamRecord
) -> None:
    """Add one langchaint.request_failed event per failed request in the outcome's request records.

    Each event carries the request's error text and its own `elapsed_seconds`.
    Request records contain relative elapsed times, so event timestamps use the recording time.
    """
    for record in outcome.request_history.records:
        if record.kind == "settled" and record.error is not None:
            span.add_event(
                "langchaint.request_failed",
                {"error_text": record.error.error_text, "elapsed_seconds": record.elapsed_seconds},
            )


def _tool_call_arguments_attribute(
    tool_call: ToolCall, content_filter: ContentFilter
) -> dict[str, SpanAttributeValue]:
    """Build gen_ai.tool.call.arguments from `tool_call`, empty when the filter omits it."""
    filtered = _filtered_part(content_filter, "gen_ai.tool.call.arguments", tool_call)
    if filtered is None:
        return {}
    return {"gen_ai.tool.call.arguments": json.dumps(_tool_call_arguments(filtered.args_json))}


class _SpanOperation[OutcomeT]:
    """The span of one observed operation.

    The span holds only its start attributes until it ends.
    The operation collects every other attribute and records them, edited by `transform`, when the span ends.
    `content_filter` is None when content capture is off.
    """

    def __init__(
        self,
        span: Span,
        content_filter: ContentFilter | None,
        start_attributes: SpanAttributes,
        transform: Callable[[OutcomeT | None, SpanAttributes], SpanAttributes],
    ) -> None:
        self._span = span
        self._content_filter = content_filter
        self._transform = transform
        self._attributes: dict[str, SpanAttributeValue] = dict(start_attributes)
        self._outcome: OutcomeT | None = None
        self._status: Status | None = None

    @contextmanager
    def current(self) -> Generator[None]:
        """Make the span current for the block."""
        token = context.attach(trace.set_span_in_context(self._span))
        try:
            yield
        finally:
            context.detach(token)

    def add_content_attribute_group(
        self, build: Callable[[ContentFilter], SpanAttributes]
    ) -> None:
        """Add the content attribute group `build` returns, replacing matching keys, when capture is on.

        `build` receives the observer's content filter.
        `build` runs only for a recording span, so an application without a TracerProvider serializes nothing.
        The content keys are JSON strings, and some of what they serialize is arbitrary application data:
        An application supplies `JSONSchemaTool.args_schema` values verbatim.
        `json.dumps` can reject one of those values.
        A `ContentFilter` can raise or return a part of another kind.
        A `build` Exception is logged, and the attribute group is omitted.
        """
        content_filter = self._content_filter
        if content_filter is None:
            return
        try:
            if self._span.is_recording():
                self._attributes.update(build(content_filter))
        except Exception:
            _logger.warning(
                "content capture raised; this span's telemetry is incomplete", exc_info=True
            )

    def _conclude_with_error(self, error_type: str, description: str) -> None:
        """Add error.type and keep error status for the end of the span.

        An empty description sets a status without a description.
        """
        self._attributes["error.type"] = error_type
        self._status = Status(StatusCode.ERROR, description or None)

    def _transformed_attributes(self) -> SpanAttributes:
        """Return the transform's result, or only error.type when the transform raises.

        The start attributes are on the span already, so the span keeps them either way.
        """
        try:
            return self._transform(self._outcome, self._attributes)
        except Exception:
            _logger.warning(
                "the attribute transform raised; this span keeps its start attributes and any existing error.type",
                exc_info=True,
            )
            error_type = self._attributes.get("error.type")
            return {} if error_type is None else {"error.type": error_type}

    def end(self) -> None:
        """Set the status, record the transformed attributes on a recording span, and end the span."""
        try:
            if self._status is not None:
                self._span.set_status(self._status)
            if self._span.is_recording():
                self._span.set_attributes(self._transformed_attributes())
        finally:
            self._span.end()


class _GenerationSpan(_SpanOperation[GenerationOutcome[object] | AbandonedStreamRecord]):
    """The CLIENT chat span of one input."""

    def conclude(self, outcome: GenerationOutcome[object] | AbandonedStreamRecord) -> None:
        """Add the outcome's events and attribute groups, and keep its status for the end of the span.

        A `Generation` sets OK status, and a `GenerationError` sets error status and error.type.
        An `AbandonedStreamRecord` leaves the status unset, because the application's code ended the stream.
        With capture on, gen_ai.output.messages records the outcome's assistant message.
        GenerationError carries the last assistant message, and the key is omitted when no request produced one.
        Per-request detail stays on the langchaint.request_failed events, which carry no content.
        """
        self._outcome = outcome
        _record_request_failed_events(self._span, outcome)
        self._attributes.update(_outcome_attributes(outcome))
        assistant_message = outcome.assistant_message
        if assistant_message is not None:
            stop_reason = outcome.stop_reason
            self.add_content_attribute_group(
                lambda content_filter: _output_content_attributes(
                    assistant_message.content_parts, stop_reason, content_filter
                )
            )
        if isinstance(outcome, GenerationError):
            self._conclude_with_error(outcome.kind, outcome.error_text)
        elif outcome.kind != "abandoned_stream":
            self._status = Status(StatusCode.OK)


class _DispatchSpan(_SpanOperation[DispatchOutcome | Exception]):
    """The INTERNAL execute_tool span of one tool dispatch."""

    def conclude(self, outcome: DispatchOutcome | Exception) -> None:
        """Add the dispatch outcome's attribute groups, or record the tool function's exception.

        The outcome selects the span status and error.type, as `OtelObserver.dispatch_started` lists.
        With capture on, `gen_ai.tool.call.result` records the outcome's `ToolMessage`.
        A raising tool function adds an exception event.
        Its error.type is the exception class name, for low-cardinality grouping.
        """
        self._outcome = outcome
        if isinstance(outcome, Exception):
            self._span.record_exception(outcome)
            self._conclude_with_error(type(outcome).__name__, str(outcome))
            return
        self.add_content_attribute_group(
            lambda content_filter: {
                "gen_ai.tool.call.result": json.dumps(
                    _tool_call_response_part(
                        outcome.tool_message, content_filter, "gen_ai.tool.call.result"
                    )
                )
            }
        )
        match outcome.kind:
            case "handled" if not outcome.tool_message.is_error:
                self._status = Status(StatusCode.OK)
            case "handled":
                self._conclude_with_error("tool_error", "tool_error")
            case "invalid_tool_args" | "unknown_tool":
                self._conclude_with_error(outcome.kind, outcome.kind)


def _embedding_request_attributes(
    request_record: EmbeddingRequestRecord,
) -> dict[str, SpanAttributeValue]:
    """Map one embedding request record to GenAI-convention span attributes plus langchaint.cost_in_usd.

    gen_ai.usage.input_tokens and langchaint.cost_in_usd are omitted when the provider reported no token count.
    gen_ai.response.model is omitted when the provider reported no model.
    """
    attributes: dict[str, SpanAttributeValue] = {}
    usage = request_record.usage
    if usage is not None:
        attributes["gen_ai.usage.input_tokens"] = usage.input_tokens_total
        attributes["langchaint.cost_in_usd"] = usage.cost_in_usd
    if request_record.model_served is not None:
        attributes["gen_ai.response.model"] = request_record.model_served
    return attributes


class _EmbeddingsSpan(_SpanOperation[EmbeddingRequestRecord]):
    """The CLIENT embeddings span of one embedding request."""

    def conclude(self, outcome: EmbeddingRequestRecord) -> None:
        """Add the request record's attribute group, and keep its status for the end of the span.

        A request that returned usable vectors sets OK status.
        A failed request sets error status and error.type from `error.kind`.
        Its status description is `error.error_text`.
        """
        self._outcome = outcome
        self._attributes.update(_embedding_request_attributes(outcome))
        if outcome.error is None:
            self._status = Status(StatusCode.OK)
        else:
            self._conclude_with_error(outcome.error.kind, outcome.error.error_text)


class OtelObserver:
    """Trace generations, tool dispatches, and embedding requests with OTel spans.

    Pass one instance to a backend constructor or `ToolManager(observer=...)`.
    The OTel SDK configures whether tracing records and where it sends spans.
    An application without an SDK configuration gets non-recording no-op spans.
    This class owns each span's name, kind, status, and events.
    A transform changes only attributes.
    There is no langchaint.elapsed_seconds attribute, because each span's duration measures its operation.
    A chat span's duration includes request admission and backoff between requests.
    """

    def __init__(
        self,
        *,
        capture_message_content: bool | ContentFilter,
        chat_attribute_transform: ChatAttributeTransform | None = None,
        execute_tool_attribute_transform: ExecuteToolAttributeTransform | None = None,
        embeddings_attribute_transform: EmbeddingsAttributeTransform | None = None,
        extra_attributes: SpanAttributes | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        """Resolve the tracer once, at construction.

        `capture_message_content` has no default because content capture affects privacy.
        `capture_message_content=False` records no message content.
        `capture_message_content=True` records bound prompts, tool definitions, inputs, and assistant messages.
        It also records tool arguments and tool results.
        It records every part unchanged, including image and audio bytes.
        A `ContentFilter` passed as `capture_message_content` selects the same attributes.
        The filter controls recorded prompt and message parts, tool arguments, and tool results.
        The filter never sees `gen_ai.tool.definitions`, because a tool schema is not a message part.
        A filter that keeps everything except inline bytes:

            def drop_binary(name: str, part: ContentPart | AssistantPart) -> ContentPart | AssistantPart | None:
                return None if part.kind in ("image", "audio") else part

        `chat_attribute_transform` edits the attributes of each chat span when the span ends.
        `execute_tool_attribute_transform` edits the attributes of each execute_tool span when the span ends.
        `embeddings_attribute_transform` edits the attributes of each embeddings span when the span ends.
        A `None` transform records the attributes unchanged.
        `extra_attributes` applies at the start of every span.
        `extra_attributes=None` supplies no extra attributes.
        Request and dispatch identity attributes replace matching `extra_attributes` keys at span start.
        Content and outcome attributes replace matching `extra_attributes` values when the span ends.
        `tracer_provider` decides where spans go, and `tracer_provider=None` uses the global tracer provider.
        Either way, the tracer is named `langchaint.tracing` with the package version.
        That name is every span's instrumentation scope, which identifies langchaint as the span's source.
        """
        self._tracer = trace.get_tracer(
            "langchaint.tracing", _PACKAGE_VERSION, tracer_provider=tracer_provider
        )
        self._chat_attribute_transform = chat_attribute_transform or _keep_every_attribute
        self._execute_tool_attribute_transform = (
            execute_tool_attribute_transform or _keep_every_attribute
        )
        self._embeddings_attribute_transform = (
            embeddings_attribute_transform or _keep_every_attribute
        )
        self._extra_attributes = extra_attributes or {}
        self._content_filter: ContentFilter | None = (
            None
            if capture_message_content is False
            else _record_every_part
            if capture_message_content is True
            else capture_message_content
        )

    def generation_started(
        self, start: GenerationStart
    ) -> ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord]:
        """Open the CLIENT chat span and set its start attributes.

        The span is named "chat {start.model}".
        The span starts with `extra_attributes` and the request attributes, so a sampler sees them.
        With capture on, the input content attributes are built when the span starts.
        Later changes to `start.messages` therefore do not affect them.
        The span records them when it ends, including when handling the input fails.
        They are built only for a recording span, so an application without a TracerProvider serializes nothing.
        """
        start_attributes = {**self._extra_attributes, **_request_attributes(start)}
        span = self._tracer.start_span(
            f"{_CHAT_OPERATION} {start.model}", kind=SpanKind.CLIENT, attributes=start_attributes
        )
        generation_span = _GenerationSpan(
            span, self._content_filter, start_attributes, self._chat_attribute_transform
        )
        generation_span.add_content_attribute_group(
            lambda content_filter: _input_content_attributes(
                start.binding, start.messages, content_filter=content_filter
            )
        )
        return generation_span

    def dispatch_started(
        self, tool_call: ToolCall
    ) -> ObservedOperation[DispatchOutcome | Exception]:
        """Open the INTERNAL execute_tool span and set its identity attributes.

        The span name is "execute_tool {tool_call.name}".
        The span starts with `extra_attributes` and the identity attributes, so a sampler sees them.
        The identity attributes are gen_ai.operation.name, gen_ai.tool.name, gen_ai.tool.type, and gen_ai.tool.call.id.
        gen_ai.tool.type is "function", the convention's type for a tool the client application executes.
        With capture on, gen_ai.tool.call.arguments is built when the span starts, and the span records it when it ends.
        A `ContentFilter` sees the `ToolCall` under gen_ai.tool.call.arguments.
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
        start_attributes = {
            **self._extra_attributes,
            "gen_ai.operation.name": _EXECUTE_TOOL_OPERATION,
            "gen_ai.tool.name": tool_call.name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": tool_call.id,
        }
        span = self._tracer.start_span(
            f"{_EXECUTE_TOOL_OPERATION} {tool_call.name}",
            kind=SpanKind.INTERNAL,
            attributes=start_attributes,
        )
        dispatch_span = _DispatchSpan(
            span,
            self._content_filter,
            start_attributes,
            self._execute_tool_attribute_transform,
        )
        dispatch_span.add_content_attribute_group(
            lambda content_filter: _tool_call_arguments_attribute(tool_call, content_filter)
        )
        return dispatch_span

    def embedding_request_started(
        self, start: EmbeddingRequestStart
    ) -> ObservedOperation[EmbeddingRequestRecord]:
        """Open the CLIENT embeddings span of one embedding request and set its request attributes.

        The span is named "embeddings {start.model}".
        The span starts with `extra_attributes` and the request attributes, so a sampler sees them.
        The request attributes are gen_ai.operation.name, gen_ai.provider.name, and gen_ai.request.model.
        They also include gen_ai.request.encoding_formats and gen_ai.embeddings.dimension.count.
        The request record selects the span status and error.type.
        error.type is an `EmbeddingRequestErrorKind`, so it means the same for every provider.
        """
        start_attributes = {
            **self._extra_attributes,
            "gen_ai.operation.name": _EMBEDDINGS_OPERATION,
            "gen_ai.provider.name": start.provider_name,
            "gen_ai.request.model": start.model,
            "gen_ai.request.encoding_formats": _EMBEDDINGS_ENCODING_FORMATS,
            "gen_ai.embeddings.dimension.count": start.dimension,
        }
        span = self._tracer.start_span(
            f"{_EMBEDDINGS_OPERATION} {start.model}",
            kind=SpanKind.CLIENT,
            attributes=start_attributes,
        )
        return _EmbeddingsSpan(
            span,
            self._content_filter,
            start_attributes,
            self._embeddings_attribute_transform,
        )


__all__ = [
    "ChatAttributeTransform",
    "ContentFilter",
    "EmbeddingsAttributeTransform",
    "ExecuteToolAttributeTransform",
    "OtelObserver",
    "SpanAttributes",
]
