"""Test tracing with fake adapters and an in-memory exporter.

_SchemaValidatingSpanProcessor validates payload attributes against tests/semconv_genai.
The tests inspect recorded span names, kinds, statuses, attributes, events, and parents.
"""

import asyncio
import functools
import inspect
import json
import logging
import pathlib
import re
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import assert_type, override

import jsonschema
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.attributes import error_attributes as error_semconv
from opentelemetry.trace import NonRecordingSpan, SpanKind, StatusCode
from pydantic import BaseModel

import langchaint.tracing
from langchaint import (
    LLM,
    AssistantMessage,
    AudioPart,
    CallResult,
    CallResultRecord,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchOutcome,
    DispatchUnknownTool,
    GenerationError,
    ImagePart,
    ImageUrlPart,
    JSONSchemaTool,
    PydanticTool,
    ReasoningPart,
    Response,
    ResponseRecord,
    SettledAttemptRecord,
    StreamItem,
    TextPart,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    TransientError,
    UserMessage,
    to_tables,
)
from langchaint.adapter import (
    AdapterResult,
    InvalidRequest,
    Refusal,
    UnfinishedTurn,
)
from langchaint.tracing import (
    AttributeMapper,
    SpanAttributes,
    TracedBoundLLM,
    TracedLLM,
    TracedStreamHandle,
    TracedToolManager,
    agent_span,
    gen_ai_attributes,
)
from langchaint.usage import Usage
from scripts import refresh_semconv_genai
from tests.test_bound_llm import (
    _MAX_COMPLETION_TOKENS_EXCEEDED,
    _REFUSAL,
    _REJECTED_TURN,
    _USAGE,
    _billed,
    _FakeAdapter,
    _FakeStream,
    _fast_shared_backoff,
    _HangsAfterFirstItemStream,
    _ScriptedResponse,
)

_SEMCONV_GENAI_DIR = pathlib.Path(__file__).parent / "semconv_genai"

_PAYLOAD_SCHEMA_FILES: Mapping[str, str] = refresh_semconv_genai.ATTRIBUTE_SCHEMA_FILES
"""Map each structured payload attribute to its vendored schema."""

_VALIDATED_PAYLOAD_ATTRIBUTES: set[str] = set()
"""Track payload attributes checked by _validate_payload_attributes."""

_UNVALIDATED_PAYLOAD_ATTRIBUTES = frozenset({"gen_ai.tool.call.arguments"})
"""Skip schema validation for malformed tool-call argument text.

The schema accepts objects, while DispatchInvalidToolArgs preserves non-object text.
_validate_payload_attributes still validates this attribute when it contains an object.
"""


@functools.cache
def _payload_schema(file: str) -> Mapping[str, object]:
    """Load and cache one vendored schema.

    Raises:
        OSError: the vendored file could not be read.
        json.JSONDecodeError: the file does not hold JSON.
        AssertionError: the file contains a non-object JSON value.
    """
    schema = json.loads((_SEMCONV_GENAI_DIR / file).read_text())
    assert isinstance(schema, dict), f"{file} does not hold a JSON object"
    return schema


def _validate_payload_attributes(span: ReadableSpan) -> None:
    """Validate one span's structured payload attributes.

    Each payload is a JSON string.
    _UNVALIDATED_PAYLOAD_ATTRIBUTES skips non-object tool arguments.
    Exact-equality assertions test fields that the schemas leave optional.

    Raises:
        AssertionError: A payload does not conform, is not a JSON string, or does not parse.
    """
    for key, value in (span.attributes or {}).items():
        file = _PAYLOAD_SCHEMA_FILES.get(key)
        if file is None:
            continue
        assert isinstance(value, str), f"{span.name}: {key} is not a JSON string"
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(f"{span.name}: {key} is not JSON: {error}") from error
        if key in _UNVALIDATED_PAYLOAD_ATTRIBUTES and not isinstance(payload, dict):
            continue
        _VALIDATED_PAYLOAD_ATTRIBUTES.add(key)
        try:
            jsonschema.Draft202012Validator(_payload_schema(file)).validate(payload)
        except jsonschema.ValidationError as error:
            raise AssertionError(
                f"{span.name}: {key} violates {file}. "
                f"Path {list(error.absolute_path)}: {error.message}"
            ) from error


class _SchemaValidatingSpanProcessor(SimpleSpanProcessor):
    """Validate payload attributes before exporting each span."""

    @override
    def on_end(self, span: ReadableSpan) -> None:
        """Validate and export the ending span.

        Raises:
            AssertionError: the span carries a payload that does not conform to its schema.
        """
        _validate_payload_attributes(span)
        super().on_end(span)


def _in_memory_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """Build a schema-validating in-memory tracer."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(_SchemaValidatingSpanProcessor(exporter))
    return tracer_provider.get_tracer("test"), exporter


def _attribute(span: ReadableSpan, key: str) -> object:
    """Read one attribute off a finished span, None where the span carries no such key."""
    return (span.attributes or {}).get(key)


class _RaisingSpanProcessor(SpanProcessor):
    """A SpanProcessor whose on_end raises.

    Span.end calls on_end with no guard of its own, so this raise reaches whatever ended the span.
    on_end raises observably before SimpleSpanProcessor catches exporter failures.
    """

    @override
    def on_end(self, span: ReadableSpan) -> None:
        """Raise instead of exporting.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("span processor boom")


def _raising_processor_tracer() -> trace.Tracer:
    """Build a recording tracer whose every span end raises out of its processor."""
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(_RaisingSpanProcessor())
    return tracer_provider.get_tracer("test")


def test_a_span_processor_that_raises_on_end_does_not_destroy_a_result() -> None:
    """A broken SpanProcessor costs the telemetry, never the result the call returned.

    Each span-owning entry point suppresses SpanProcessor failures.
    """

    async def scenario() -> None:
        """Drive each span-owning entry point under a tracer whose span end raises."""
        tracer = _raising_processor_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)), tracer=tracer, capture_message_content=False
        )
        bound = traced.bind()

        assert (await bound.generate_one("hi")).output == "hi"
        (row,) = await bound.generate_many(["hi"])
        assert isinstance(row, Response)
        assert row.output == "hi"

        async with bound.stream_one("hi") as stream:
            items = [item async for item in stream if isinstance(item, str)]
            assert (await stream.final()).output == "".join(items)

        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        outcome = await tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json='{"text": "hi"}')
        )
        assert isinstance(outcome, DispatchHandled)

        with agent_span(
            tracer, agent_name="specialist", agent_path="root/specialist", usage=lambda: _USAGE
        ):
            pass

    asyncio.run(scenario())


class _IsRecordingRaisesSpan(NonRecordingSpan):
    """A span whose is_recording raises, standing in for a third-party Span implementation."""

    def __init__(self) -> None:
        super().__init__(trace.INVALID_SPAN_CONTEXT)

    @override
    def is_recording(self) -> bool:
        """Raise instead of answering.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("is_recording boom")


class _IsRecordingRaisesTracer(trace.NoOpTracer):
    """A tracer handing out spans whose is_recording raises."""

    @override
    def start_span(self, name: str, *_args: object, **_kwargs: object) -> trace.Span:
        """Return the span whose is_recording raises, ignoring every span argument."""
        return _IsRecordingRaisesSpan()


def test_a_span_whose_is_recording_raises_does_not_displace_the_call_s_error() -> None:
    """A broken span cannot replace GenerationError."""

    async def scenario() -> None:
        """Drive one failing generate_one under a tracer whose spans raise from is_recording."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="misconfigured")])
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=_IsRecordingRaisesTracer(),
            capture_message_content=True,
        )
        with pytest.raises(GenerationError):
            await traced.bind().generate_one("hi")

    asyncio.run(scenario())


class _MidFailStream(_FakeStream):
    """A stream that yields one item, then raises so the failure lands mid-iteration."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield one chunk, then raise a plain exception the classifier maps to transient.

        Yields:
            One text chunk before the raise.

        Raises:
            ValueError: always, after the first yield.
        """
        yield "a"
        raise ValueError("mid-stream boom")


def test_generate_one_success_produces_one_fully_attributed_span() -> None:
    """A success emits one CLIENT span named "chat {model}", OK status, and every gen_ai attribute."""

    async def scenario() -> None:
        """Drive one generate_one to success and inspect the single finished span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)), tracer=tracer, capture_message_content=False
        )
        response = await traced.bind().generate_one("hi")
        assert response.output == "hi"
        (span,) = exporter.get_finished_spans()
        assert span.name == "chat fake-model"
        assert span.kind == SpanKind.CLIENT
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        assert dict(span.attributes) == {
            "gen_ai.operation.name": "chat",
            "gen_ai.output.type": "text",
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.response.model": "fake-model-served",
            "gen_ai.response.finish_reasons": ("stop",),
            "gen_ai.usage.input_tokens": _USAGE.input_tokens_total,
            "gen_ai.usage.output_tokens": _USAGE.output_tokens,
            "gen_ai.usage.reasoning.output_tokens": _USAGE.output_tokens_reasoning,
            "gen_ai.usage.cache_read.input_tokens": _USAGE.input_tokens_cache_read,
            "gen_ai.usage.cache_creation.input_tokens": _USAGE.input_tokens_cache_write,
            "langchaint.attempts": 1,
            "langchaint.cost_in_usd": 0.0,
        }

    asyncio.run(scenario())


def test_generate_one_refusal_span_has_error_status_and_real_tokens() -> None:
    """A GenerationError yields an error span carrying the 200's real token counts and cost."""

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports Refusal, then inspect the error span."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_REFUSAL)])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(GenerationError):
            await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["gen_ai.response.finish_reasons"] == ("refusal",)
        assert span.attributes["langchaint.cost_in_usd"] == 0.25
        assert span.attributes["gen_ai.usage.output_tokens"] == _USAGE.output_tokens

    asyncio.run(scenario())


def test_generate_one_truncation_span_has_error_status_and_real_tokens() -> None:
    """A GenerationError yields an error span with the 200's tokens and max_tokens finish."""

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports MaxCompletionTokensExceeded, then inspect the error span."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_MAX_COMPLETION_TOKENS_EXCEEDED)])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(GenerationError):
            await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["gen_ai.response.finish_reasons"] == ("length",)
        assert span.attributes["langchaint.cost_in_usd"] == 0.25

    asyncio.run(scenario())


def test_generate_one_retries_exhausted_span_has_error_status_and_zero_tokens() -> None:
    """A retries-exhausted failure over transport errors bills zero, so the usage attributes are zero."""

    async def scenario() -> None:
        """Exhaust the budget on transport failures and inspect the error span."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("e1"), TransientError("e2")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(GenerationError):
            await traced.bind(max_attempts=2).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["gen_ai.usage.input_tokens"] == 0
        assert span.attributes["gen_ai.usage.output_tokens"] == 0
        assert span.attributes["langchaint.cost_in_usd"] == 0.0
        # A retries-exhausted failure has no completed turn, so no finish reason is set.
        assert "gen_ai.response.finish_reasons" not in span.attributes
        # No attempt received a response, so no provider ever named the model that served one.
        assert "gen_ai.response.model" not in span.attributes

    asyncio.run(scenario())


def test_generate_one_rejection_span_names_its_own_class_in_error_type() -> None:
    """A rejected request takes error status under its own error.type, not the base class name.

    error.type distinguishes provider rejection from unknown errors.
    """

    async def scenario() -> None:
        """Drive one generate_one whose build_request reports InvalidRequest, then read its span."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="misconfigured")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(GenerationError):
            await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "misconfigured"
        assert span.attributes is not None
        assert span.attributes["error.type"] == "InvalidRequestErrorRecord"
        # Nothing was sent, so the usage attributes are the zeros of a call that never billed.
        assert span.attributes["langchaint.cost_in_usd"] == 0.0
        assert "gen_ai.response.finish_reasons" not in span.attributes

    asyncio.run(scenario())


def test_generate_one_cancellation_ends_the_span_with_its_status_unset() -> None:
    """A cancelled traced generate_one ends its span and sets no status: nothing decided the call."""

    async def scenario() -> None:
        """Time out a traced call whose open hangs, then read the span."""
        adapter = _FakeAdapter(hang_from_open=1)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_cancelled_traced_stream_reads_its_abandoned_through_the_wrapper() -> None:
    """The traced handle surfaces the wrapped handle's abandoned, the only account of a cut-off stream.

    TracedLLM preserves the source LLM's async context manager.
    """

    async def scenario() -> None:
        """Time out an entry whose open never returns, then read the traced handle and the span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        handle = traced.bind().stream_one("hi")

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns. The wait_for below cancels this."""
            async with handle:
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        assert handle.abandoned is not None
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


async def _drain_by_iterating(handle: TracedStreamHandle[str]) -> None:
    """Consume the stream item by item, never calling final()."""
    async for _ in handle:
        pass


async def _drain_by_final(handle: TracedStreamHandle[str]) -> None:
    """Ask final() for the Response, never iterating."""
    await handle.final()


@pytest.mark.parametrize("drain", [_drain_by_iterating, _drain_by_final])
def test_a_traced_streams_expired_deadline_takes_error_status(
    drain: Callable[[TracedStreamHandle[str]], Awaitable[None]],
) -> None:
    """Stream deadlines record GenerationError for each drain method."""

    async def scenario() -> None:
        """Drain a stream that stalls after its first item, under a deadline it outlasts."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(
                _FakeAdapter(stream=_HangsAfterFirstItemStream()),
                shared_backoff=_fast_shared_backoff(),
            ),
            tracer=tracer,
            capture_message_content=False,
        )
        handle = traced.bind().stream_one("hi", timeout_seconds=0.05)

        with pytest.raises(GenerationError):
            async with handle:
                await drain(handle)
        assert handle.abandoned is None
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes[error_semconv.ERROR_TYPE] == "TimedOutErrorRecord"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_cancelled_traced_batch_ends_every_started_items_span() -> None:
    """A cancellation reaching a traced batch ends each started item's span, with no status set.

    Batch cancellation ends each started item span.
    """

    async def scenario() -> None:
        """Time out a traced batch whose opens hang, then read the spans."""
        adapter = _FakeAdapter(hang_from_open=1)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await traced.bind().generate_many(["a", "b"])
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(span.status.status_code == StatusCode.UNSET for span in spans)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_retry_surfaces_as_an_attempt_failed_span_event() -> None:
    """A recovered transient failure becomes one langchaint.attempt_failed event on the success span."""

    async def scenario() -> None:
        """Recover one generate_one from a transient failure, then read the span event."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("boom")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        response = await traced.bind().generate_one("hi")
        assert response.attempts == 2
        (span,) = exporter.get_finished_spans()
        (event,) = span.events
        assert event.name == "langchaint.attempt_failed"
        assert event.attributes is not None
        assert event.attributes["error_text"] == "boom"

    asyncio.run(scenario())


def test_generate_many_emits_one_chat_span_per_item_and_none_for_the_batch() -> None:
    """generate_many emits one chat span per item."""

    async def scenario() -> None:
        """Serialize a three-item batch whose first item is Refusal, then inspect the spans."""
        adapter = _FakeAdapter(
            echo=True,
            scripted_attempts=[_billed(_REFUSAL)],
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=shared_backoff),
            tracer=tracer,
            capture_message_content=False,
        )
        results = await traced.bind().generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
            [UserMessage(content="c")],
        ])
        first, *rest = results
        assert isinstance(first, GenerationError)
        assert all(isinstance(result, Response) for result in rest)
        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        assert all(span.kind == SpanKind.CLIENT for span in spans)
        assert all(_attribute(span, "gen_ai.operation.name") == "chat" for span in spans)
        assert all(_attribute(span, "langchaint.cost_in_usd") is not None for span in spans)
        # max_concurrent_requests=1 serializes the batch, so the refused item is the first span to end.
        refused, *succeeded = spans
        assert refused.status.status_code == StatusCode.ERROR
        assert _attribute(refused, "error.type") == "RefusalErrorRecord"
        assert all(span.status.status_code == StatusCode.OK for span in succeeded)

    asyncio.run(scenario())


def test_generate_many_records_traces_generated_items_and_skips_reused_items(
    tmp_path: pathlib.Path,
) -> None:
    """generate_many_records opens chat spans only for items that send requests."""

    async def scenario() -> None:
        """Persist two samples, reorder them around one new sample, and inspect the span count."""
        adapter = _FakeAdapter(echo=True)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter),
            tracer=tracer,
            capture_message_content=False,
        )
        bound = traced.bind()
        resume_path = tmp_path / "records.json"
        first = await bound.generate_many_records(
            ["a", "b"],
            resume_path=resume_path,
            sample_ids=["sample-a", "sample-b"],
        )
        assert all(isinstance(record, ResponseRecord) for record in first)
        assert len(exporter.get_finished_spans()) == 2

        resumed = await bound.generate_many_records(
            ["b", "c", "a"],
            resume_path=resume_path,
            sample_ids=["sample-b", "sample-c", "sample-a"],
        )
        assert all(isinstance(record, ResponseRecord) for record in resumed)
        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        assert all(span.kind == SpanKind.CLIENT for span in spans)
        assert adapter.bound_adapters[0].open_count == 3

    asyncio.run(scenario())


def test_stream_exhausted_then_final_emits_one_span_with_time_to_first_chunk() -> None:
    """A stream iterated to exhaustion then final() ends exactly one span carrying time_to_first_chunk."""

    async def scenario() -> None:
        """Iterate the stream fully, call final(), and inspect the single finished span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        async with traced.bind().stream_one("hi") as stream:
            texts = [item async for item in stream if isinstance(item, str)]
            response = await stream.final()
        assert "".join(texts) == "ok"
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        time_to_first_chunk = span.attributes["gen_ai.response.time_to_first_chunk"]
        assert isinstance(time_to_first_chunk, float)
        assert time_to_first_chunk >= 0.0
        assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)

    asyncio.run(scenario())


def test_stream_final_is_idempotent_and_ends_the_span_once() -> None:
    """A second final() returns the same Response and does not end a second span."""

    async def scenario() -> None:
        """Call final() twice on one drained stream and count the spans."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        async with traced.bind().stream_one("hi") as stream:
            first = await stream.final()
            second = await stream.final()
        assert first is second
        assert len(exporter.get_finished_spans()) == 1

    asyncio.run(scenario())


def test_stream_abandoned_in_context_ends_its_span() -> None:
    """A stream partially iterated then abandoned inside async with ends its span in __aexit__."""

    async def scenario() -> None:
        """Break out after one item and confirm one span ended without error status."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        async with traced.bind().stream_one("hi") as stream:
            async for _item in stream:
                break
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET

    asyncio.run(scenario())


def test_stream_entered_but_never_iterated_emits_a_span() -> None:
    """Entering opens a request, so an entered handle emits a span even with no item pulled.

    The request is billed whether or not the caller reads it. A silent span would hide it.
    """

    async def scenario() -> None:
        """Enter and leave the context without driving the stream."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        async with traced.bind().stream_one("hi"):
            pass
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET
        assert span.attributes is not None
        assert "gen_ai.response.time_to_first_chunk" not in span.attributes

    asyncio.run(scenario())


def test_traced_stream_iterated_after_the_block_touches_no_ended_span(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Iterating after the block raises from the inner handle without writing to the closed span.

    Recording RuntimeError on the ended span would make the OTel SDK log a warning.
    The test rejects that warning for an ordinary caller mistake.
    """

    async def scenario() -> None:
        """Drain a stream, leave the block, then pull one more item."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        handle = traced.bind().stream_one("hi")
        async with handle:
            _ = [item async for item in handle]
            await handle.final()
        with (
            caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"),
            pytest.raises(RuntimeError, match="finished"),
        ):
            await anext(handle)
        assert "ended span" not in caplog.text
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK

    asyncio.run(scenario())


def test_traced_stream_second_entry_leaves_the_first_span_intact() -> None:
    """Re-entering a traced handle raises without marking the completed stream's span failed."""

    async def scenario() -> None:
        """Drain a stream, leave the block, then enter the same handle again."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        handle = traced.bind().stream_one("hi")
        async with handle:
            _ = [item async for item in handle]
            await handle.final()
        with pytest.raises(RuntimeError, match="already entered"):
            async with handle:
                pass
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK

    asyncio.run(scenario())


def test_stream_never_entered_emits_no_span() -> None:
    """stream_one does no I/O, so a handle abandoned without entering emits no span."""

    async def scenario() -> None:
        """Build a handle and drop it."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        _handle = traced.bind().stream_one("hi")
        assert exporter.get_finished_spans() == ()

    asyncio.run(scenario())


def test_stream_failing_mid_iteration_ends_its_span_like_any_other_generation_error() -> None:
    """A stream failure records GenerationError and attempt_failed."""

    async def _drain(traced: TracedLLM) -> None:
        """Iterate the mid-failing stream to its raise inside an async with block."""
        async with traced.bind().stream_one("hi") as stream:
            async for _item in stream:
                pass

    async def scenario() -> None:
        """Iterate a mid-failing stream and confirm the error span."""
        adapter = _FakeAdapter(stream=_MidFailStream(), classify_result="transient")
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(GenerationError):
            await _drain(traced)
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "RetryUnavailableErrorRecord"
        assert [event.name for event in span.events] == ["langchaint.attempt_failed"]

    asyncio.run(scenario())


def test_stream_open_exhausting_retries_ends_its_span_with_the_calls_attributes() -> None:
    """Retries exhausted while opening ends the span the way a failure read from final() does.

    __aenter__ GenerationError records error.type and call attributes.
    """

    async def scenario() -> None:
        """Enter a stream whose every open fails, and inspect the span it left."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("connection reset")] * 4)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(GenerationError):
            async with traced.bind(max_attempts=2).stream_one("hi"):
                pass
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "RetriesExhaustedErrorRecord"
        assert span.attributes["langchaint.attempts"] == 2
        assert [event.name for event in span.events] == ["langchaint.attempt_failed"] * 2

    asyncio.run(scenario())


def test_stream_final_refusal_ends_the_span_with_error_status() -> None:
    """A structured refusal detected in the stream's final() ends the span with error status and tokens."""

    async def scenario() -> None:
        """Drain a stream whose final() reports Refusal and inspect the error span."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_REFUSAL))
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=False,
        )
        async with traced.bind().stream_one("hi") as stream:
            with pytest.raises(GenerationError):
                await stream.final()
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["langchaint.cost_in_usd"] == 0.25

    asyncio.run(scenario())


class _Answer(BaseModel):
    """A response_format model for the bind and covariance type checks."""

    value: int


@pytest.mark.parametrize(
    ("stream", "response_format", "output_type"),
    [(False, None, "text"), (True, None, "text"), (False, _Answer, "json")],
)
def test_request_attributes_cover_generate_stream_and_structured_output(
    *,
    stream: bool,
    response_format: type[_Answer] | None,
    output_type: str,
) -> None:
    """Request attributes describe text, structured, and streaming calls."""

    async def scenario() -> None:
        """Run the selected call and inspect its request attributes."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)),
            tracer=tracer,
            capture_message_content=False,
        )
        bound = traced.bind(
            response_format=response_format,
            max_completion_tokens=123,
            reasoning_level="high",
            temperature=0.25,
        )
        if response_format is not None:
            with pytest.raises(GenerationError):
                await bound.generate_one("hi")
        elif stream:
            async with bound.stream_one("hi") as handle:
                await handle.final()
        else:
            await bound.generate_one("hi")

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        expected: dict[str, object] = {
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.request.max_tokens": 123,
            "gen_ai.request.reasoning.level": "high",
            "gen_ai.request.temperature": 0.25,
            "gen_ai.output.type": output_type,
        }
        if stream:
            expected["gen_ai.request.stream"] = True
        assert {key: span.attributes[key] for key in expected} == expected
        assert ("gen_ai.request.stream" in span.attributes) is stream

    asyncio.run(scenario())


def test_bind_stays_traced_and_shares_the_mapper() -> None:
    """The replacement object stays traced and uses the same custom mapper."""

    async def scenario() -> None:
        """Generate and stream on a replacement object under a key-recording mapper."""
        keys_seen: list[frozenset[str]] = []

        def _mapper(_result: CallResult[object]) -> SpanAttributes:
            """Record its own key set and emit exactly one attribute."""
            keys_seen.append(frozenset({"custom.mapped"}))
            return {"custom.mapped": True}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        replacement_bound = traced.bind(system_prompt="s").bind(system_prompt="s2")
        assert_type(replacement_bound, TracedBoundLLM[str])
        await replacement_bound.generate_one("hi")
        async with replacement_bound.stream_one("hi") as stream:
            async for _item in stream:
                pass
            await stream.final()
        generate_span, stream_span = exporter.get_finished_spans()
        # gen_ai.operation.name is the wrapper's required attribute, outside the mapper's control.
        assert generate_span.attributes == {
            "gen_ai.operation.name": "chat",
            "gen_ai.output.type": "text",
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "custom.mapped": True,
        }
        assert stream_span.attributes is not None
        # The stream span also carries the wrapper-owned time_to_first_chunk plus the mapped one.
        assert stream_span.attributes["custom.mapped"] is True
        assert keys_seen == [frozenset({"custom.mapped"}), frozenset({"custom.mapped"})]

    asyncio.run(scenario())


def test_traced_initial_and_replacement_bind_forward_binding_options() -> None:
    """Forward binding options through initial, omitted, replacement, and clearing values."""
    extra_body = {"safety_identifier": "user-7"}
    provider_tool = {"type": "web_search"}
    traced = TracedLLM(
        LLM(_FakeAdapter(automatic_cache_breakpoints_default=False)),
        capture_message_content=False,
    )
    assert traced.bind().max_attempts == 3
    bound = traced.bind(
        extra_body=extra_body,
        provider_executed_tools=[provider_tool],
        automatic_cache_breakpoints=True,
        max_completion_tokens=100,
        reasoning_level="HIGH",
        temperature=0.2,
        max_attempts=2,
    )
    assert bound.binding.extra_body is extra_body
    assert bound.binding.provider_executed_tools == (provider_tool,)
    assert bound.binding.provider_executed_tools[0] is provider_tool
    assert bound.binding.automatic_cache_breakpoints is True
    assert bound.binding.max_completion_tokens == 100
    assert bound.binding.reasoning_level == "HIGH"
    assert bound.binding.temperature == 0.2
    assert bound.max_attempts == 2

    kept = bound.bind()
    assert kept.binding.extra_body is extra_body
    assert kept.binding.provider_executed_tools[0] is provider_tool
    assert kept.binding.automatic_cache_breakpoints is True
    assert kept.binding.max_completion_tokens == 100
    assert kept.binding.reasoning_level == "HIGH"
    assert kept.binding.temperature == 0.2
    assert kept.max_attempts == 2

    replacement = {"safety_identifier": "user-8"}
    replacement_tool = {"type": "file_search"}
    replaced = bound.bind(
        extra_body=replacement,
        provider_executed_tools=[replacement_tool],
        automatic_cache_breakpoints=None,
        max_completion_tokens=None,
        reasoning_level="LOW",
        temperature=0.8,
        max_attempts=4,
    )
    assert replaced.binding.extra_body is replacement
    assert replaced.binding.provider_executed_tools == (replacement_tool,)
    assert replaced.binding.provider_executed_tools[0] is replacement_tool
    assert replaced.binding.automatic_cache_breakpoints is False
    assert replaced.binding.max_completion_tokens is None
    assert replaced.binding.reasoning_level == "LOW"
    assert replaced.binding.temperature == 0.8
    assert replaced.max_attempts == 4

    cleared = bound.bind(extra_body=None, provider_executed_tools=())
    assert cleared.binding.extra_body is None
    assert cleared.binding.provider_executed_tools == ()


def test_custom_attribute_mapper_replaces_default_result_attributes() -> None:
    """A custom attribute_mapper replaces default result attributes."""

    async def scenario() -> None:
        """Generate under a two-key mapper and inspect the wrapper and mapper attributes."""

        def _mapper(result: CallResult[object]) -> SpanAttributes:
            """Emit two fixed attributes drawn from the result."""
            return {"custom.model": result.model, "custom.attempts": result.attempts}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes == {
            "gen_ai.operation.name": "chat",
            "gen_ai.output.type": "text",
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "custom.model": "fake-model",
            "custom.attempts": 1,
        }

    asyncio.run(scenario())


def test_mapper_not_invoked_on_a_non_recording_span() -> None:
    """A custom attribute_mapper never fires when the tracer's spans are non-recording."""

    async def scenario() -> None:
        """Generate under a TracerProvider-less tracer and assert the mapper never ran."""
        calls: list[int] = []

        def _mapper(_result: CallResult[object]) -> SpanAttributes:
            """Count each invocation."""
            calls.append(1)
            return {}

        # No global SDK provider is configured, so get_tracer yields non-recording spans.
        tracer = trace.get_tracer("no-sdk")
        traced = TracedLLM(
            LLM(_FakeAdapter()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        response = await traced.bind().generate_one("hi")
        assert response.output == "ok"
        assert calls == []

    asyncio.run(scenario())


def test_raising_mapper_is_caught_and_the_result_survives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising mapper is logged, generate_one still returns its Response, and the span still ends."""

    async def scenario() -> None:
        """Generate under a mapper that raises and confirm the result and span survive."""

        def _mapper(_result: CallResult[object]) -> SpanAttributes:
            """Raise to simulate a buggy user mapper.

            Raises:
                RuntimeError: always.
            """
            raise RuntimeError("mapper bug")

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            response = await traced.bind().generate_one("hi")
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK
        assert any("mapper" in record.message for record in caplog.records)

    asyncio.run(scenario())


def test_generate_many_invokes_the_mapper_once_per_item() -> None:
    """Map each item's span from that item's result once."""

    async def scenario() -> None:
        """Run a two-item batch under a counting mapper and read what each item's span carries."""
        mapped_outputs: list[object] = []

        def _mapper(result: CallResult[object]) -> SpanAttributes:
            """Record the result mapped and emit it as an attribute."""
            output = result.output if isinstance(result, Response) else None
            mapped_outputs.append(output)
            return {"custom.mapped_output": str(output)}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(
                _FakeAdapter(echo=True),
                shared_backoff=_fast_shared_backoff(max_concurrent_requests=1),
            ),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        results = await traced.bind().generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "a"
        assert isinstance(second, Response)
        assert second.output == "b"
        assert mapped_outputs == ["a", "b"]
        spans = exporter.get_finished_spans()
        assert [_attribute(span, "custom.mapped_output") for span in spans] == ["a", "b"]

    asyncio.run(scenario())


def test_raising_mapper_in_final_still_returns_the_response() -> None:
    """A raising mapper in the stream's final() still returns the assembled Response."""

    async def scenario() -> None:
        """Drain a stream under a raising mapper and read final()."""

        def _mapper(_result: CallResult[object]) -> SpanAttributes:
            """Raise on every call.

            Raises:
                RuntimeError: always.
            """
            raise RuntimeError("mapper bug")

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        async with traced.bind().stream_one("hi") as stream:
            async for _item in stream:
                pass
            response = await stream.final()
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK

    asyncio.run(scenario())


def _bind_overload_pin() -> None:
    """Pin that the bind overloads mirror LLM.bind: a model gives TracedBoundLLM[Model], absent gives [str].

    pyrefly type-checks this module, so a break in the overload split surfaces as a type error here.
    Not a test: assert_type is a runtime no-op, so pytest could only ever report it as passing.
    """
    traced = TracedLLM(LLM(_FakeAdapter()), capture_message_content=False)
    structured = traced.bind(response_format=_Answer)
    assert_type(structured, TracedBoundLLM[_Answer])
    text = traced.bind()
    assert_type(text, TracedBoundLLM[str])
    text_with_tools = traced.bind(tools=[_echo_tool()])
    assert_type(text_with_tools, TracedBoundLLM[str, ToolManager])
    structured_with_tools = traced.bind(
        response_format=_Answer,
        tools=[_echo_tool()],
    )
    assert_type(structured_with_tools, TracedBoundLLM[_Answer, ToolManager])
    tool_manager = ToolManager([])
    assert_type(
        traced.bind(tools=tool_manager),
        TracedBoundLLM[str, ToolManager],
    )
    assert_type(
        structured.bind(tools=[_echo_tool()]),
        TracedBoundLLM[_Answer, ToolManager],
    )
    assert_type(structured.bind(tools=tool_manager), TracedBoundLLM[_Answer, ToolManager])
    assert_type(structured_with_tools.bind(tools=None), TracedBoundLLM[_Answer])


async def _generate_many_records_overload_pin() -> None:
    """Pin `TracedBoundLLM.generate_many_records` output types for text and structured bindings."""
    traced = TracedLLM(LLM(_FakeAdapter()), capture_message_content=False)
    text_records = await traced.bind().generate_many_records(
        ["hi"], resume_path=pathlib.Path("records.json")
    )
    structured_records = await traced.bind(response_format=_Answer).generate_many_records(
        ["hi"], resume_path=pathlib.Path("records.json")
    )
    assert_type(text_records, list[CallResultRecord[str]])
    assert_type(structured_records, list[CallResultRecord[_Answer]])


def _covariance_pin(mapper: AttributeMapper, response: Response[_Answer]) -> SpanAttributes:
    """Pin the mapper covariance: a Response[_Answer] must satisfy the Response[object] parameter.

    pyrefly checks Response OutputT covariance at the call below.
    """
    return mapper(response)


def test_traced_passthroughs_reach_the_wrapped_objects() -> None:
    """The adapter and shared_backoff pass through TracedLLM. BoundLLM fields pass through TracedBoundLLM."""
    adapter = _FakeAdapter()
    shared_backoff = _fast_shared_backoff()
    traced = TracedLLM(LLM(adapter, shared_backoff=shared_backoff), capture_message_content=False)
    assert traced.adapter is adapter
    assert traced.shared_backoff is shared_backoff
    bound = traced.bind(response_format=_Answer)
    assert bound.adapter is adapter
    assert bound.shared_backoff is shared_backoff
    assert bound.response_format is _Answer
    assert bound.tool_manager is None
    assert bound.binding.system_prompt is None
    assert (
        bound.config_fingerprint()
        == LLM(adapter).bind(response_format=_Answer).config_fingerprint()
    )


def test_extra_attributes_ride_on_generate_spans_and_mapper_wins_collisions() -> None:
    """extra_attributes land at span start on generate spans. A mapper key of the same name wins."""

    async def scenario() -> None:
        """Generate under extra_attributes plus a colliding mapper key and inspect the span."""

        def _mapper(_result: CallResult[object]) -> SpanAttributes:
            """Emit one attribute colliding with an extra_attributes key."""
            return {"shared.key": "mapped"}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)),
            attribute_mapper=_mapper,
            extra_attributes={"gen_ai.agent.name": "agent_a", "shared.key": "extra"},
            tracer=tracer,
            capture_message_content=False,
        )
        await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.agent.name"] == "agent_a"
        assert span.attributes["shared.key"] == "mapped"

    asyncio.run(scenario())


def test_extra_attributes_survive_bind_and_reach_stream_and_batch_item_spans() -> None:
    """extra_attributes pass through bind and land on the stream span and each batch item's span."""

    async def scenario() -> None:
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(
                _FakeAdapter(echo=True),
                shared_backoff=_fast_shared_backoff(max_concurrent_requests=1),
            ),
            extra_attributes={"gen_ai.agent.name": "agent_a"},
            tracer=tracer,
            capture_message_content=False,
        )
        replacement_bound = traced.bind(system_prompt="s").bind(system_prompt="s2")
        async with replacement_bound.stream_one("hi") as stream:
            await stream.final()
        await replacement_bound.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        spans = exporter.get_finished_spans()
        # One stream span plus one per batch item.
        assert len(spans) == 3
        assert all(
            span.attributes is not None and span.attributes["gen_ai.agent.name"] == "agent_a"
            for span in spans
        )

    asyncio.run(scenario())


def test_gen_ai_attributes_is_public_and_composable() -> None:
    """A custom mapper can extend gen_ai_attributes with result data."""

    async def scenario() -> None:
        """Generate under a composed mapper and check a standard key and the derived key."""

        def _mapper(result: CallResult[object]) -> SpanAttributes:
            """Extend the built-in attributes with the call's total request time."""
            return {
                **gen_ai_attributes(result),
                "app.request_seconds": sum(
                    attempt.elapsed_seconds
                    for attempt in result.attempt_records
                    if isinstance(attempt, SettledAttemptRecord)
                ),
            }

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        response = await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["app.request_seconds"] == sum(
            a.elapsed_seconds for a in response.attempt_records
        )
        assert span.attributes["gen_ai.request.model"] == "fake-model"
        assert span.attributes["langchaint.attempts"] == 1

    asyncio.run(scenario())


class _EchoToolArgs(BaseModel):
    """Arguments of the echo tool the TracedToolManager tests dispatch."""

    text: str


async def _echo_tool_function(args: _EchoToolArgs) -> str:
    """Return the validated text unchanged."""
    return args.text


async def _unserializable_schema_tool_function(_args: Mapping[str, object]) -> str:
    """Stand in for the tool function. The capture tests never dispatch a call to it."""
    return ""


def _unserializable_schema_tool() -> JSONSchemaTool:
    """Build a tool whose args_schema json.dumps cannot serialize.

    args_schema may contain application values without a JSON form.
    """
    return JSONSchemaTool(
        name="broken",
        description="a tool whose schema holds a set",
        args_schema={"type": "object", "properties": {"x": {"default": {1, 2}}}},
        function=_unserializable_schema_tool_function,
    )


async def _raising_tool_function(_args: _EchoToolArgs) -> str:
    """Raise to simulate a tool-function defect.

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError("tool bug")


def _echo_tool() -> PydanticTool[_EchoToolArgs]:
    return PydanticTool(
        name="echo",
        description="Echo the text back",
        args_model=_EchoToolArgs,
        function=_echo_tool_function,
    )


def _raising_tool() -> PydanticTool[_EchoToolArgs]:
    """Build a tool whose function always raises, a user-code defect."""
    return PydanticTool(
        name="boom",
        description="Always raises",
        args_model=_EchoToolArgs,
        function=_raising_tool_function,
    )


async def _erring_tool_function(args: _EchoToolArgs) -> ToolOutputExplicit[None]:
    """Return a function-authored failure: a handled outcome whose ToolMessage carries is_error True."""
    return ToolOutputExplicit(content=f"cannot process {args.text}", is_error=True)


def _erring_tool() -> PydanticTool[_EchoToolArgs]:
    """Build a tool whose function returns a model-visible failure instead of raising."""
    return PydanticTool(
        name="erring",
        description="Always returns a model-visible failure",
        args_model=_EchoToolArgs,
        function=_erring_tool_function,
    )


@pytest.mark.parametrize(
    ("build_tool", "tool_call", "expected_outcome_type", "expected_error_type"),
    [
        (
            _echo_tool,
            ToolCall(id="call1", name="echo", args_json='{"text": "hi"}'),
            DispatchHandled,
            None,
        ),
        (
            _erring_tool,
            ToolCall(id="call1", name="erring", args_json='{"text": "x"}'),
            DispatchHandled,
            "tool_error",
        ),
        (
            _echo_tool,
            ToolCall(id="call1", name="echo", args_json='{"wrong": 1}'),
            DispatchInvalidToolArgs,
            "invalid_tool_args",
        ),
        (
            _echo_tool,
            ToolCall(id="call1", name="missing", args_json="{}"),
            DispatchUnknownTool,
            "unknown_tool",
        ),
    ],
    ids=["handled", "function_authored_failure", "invalid_tool_args", "unknown_tool"],
)
def test_traced_tool_manager_dispatch_emits_one_span_classified_by_its_outcome(
    build_tool: Callable[[], PydanticTool[_EchoToolArgs]],
    tool_call: ToolCall,
    expected_outcome_type: type[DispatchOutcome],
    expected_error_type: str | None,
) -> None:
    """Each dispatch emits one execute_tool span classified by its outcome."""
    expected_attributes: dict[str, object] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool_call.name,
        "gen_ai.tool.call.id": "call1",
    }
    if expected_error_type is not None:
        expected_attributes["error.type"] = expected_error_type

    async def scenario() -> None:
        """Dispatch the call and inspect the single finished span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [build_tool()], tracer=tracer, capture_message_content=False
        )
        outcome = await tool_manager.dispatch(tool_call)
        assert isinstance(outcome, expected_outcome_type)
        (span,) = exporter.get_finished_spans()
        assert span.name == f"execute_tool {tool_call.name}"
        assert span.kind == SpanKind.INTERNAL
        assert span.status.status_code == (
            StatusCode.OK if expected_error_type is None else StatusCode.ERROR
        )
        assert span.attributes is not None
        assert dict(span.attributes) == expected_attributes

    asyncio.run(scenario())


def test_traced_tool_manager_function_exception_marks_the_span_error_and_propagates() -> None:
    """A tool-function defect records the exception, sets error status, and propagates."""

    async def scenario() -> None:
        """Dispatch a call whose function raises and inspect the error span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_raising_tool()], tracer=tracer, capture_message_content=False
        )
        with pytest.raises(RuntimeError, match="tool bug"):
            await tool_manager.dispatch(
                ToolCall(id="call1", name="boom", args_json='{"text": "x"}')
            )
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]
        assert span.attributes is not None
        # A raising function is classified by its exception class, the one open-ended error.type value.
        assert span.attributes["error.type"] == "RuntimeError"

    asyncio.run(scenario())


def test_traced_tool_manager_dispatch_many_spans_every_call() -> None:
    """dispatch_many inherits per-call spans: two calls yield two execute_tool spans, outcomes ordered."""

    async def scenario() -> None:
        """Dispatch two calls concurrently and read both spans."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        outcomes = await tool_manager.dispatch_many([
            ToolCall(id="call1", name="echo", args_json='{"text": "a"}'),
            ToolCall(id="call2", name="missing", args_json="{}"),
        ])
        assert isinstance(outcomes[0], DispatchHandled)
        assert isinstance(outcomes[1], DispatchUnknownTool)
        spans = exporter.get_finished_spans()
        assert sorted(span.name for span in spans) == ["execute_tool echo", "execute_tool missing"]
        call_ids = {
            span.attributes["gen_ai.tool.call.id"] for span in spans if span.attributes is not None
        }
        assert call_ids == {"call1", "call2"}

    asyncio.run(scenario())


def test_traced_tool_manager_span_is_current_inside_the_tool_function() -> None:
    """The dispatch span is current while the function runs: a span the function starts nests under it."""

    async def scenario() -> None:
        """Dispatch a tool whose function opens its own span and assert the parentage."""
        tracer, exporter = _in_memory_tracer()

        async def nesting_tool_function(args: _EchoToolArgs) -> str:
            """Open one inner span on the same tracer and return the text."""
            with tracer.start_as_current_span("inner"):
                return args.text

        tool = PydanticTool(
            name="nesting",
            description="Opens an inner span",
            args_model=_EchoToolArgs,
            function=nesting_tool_function,
        )
        tool_manager = TracedToolManager([tool], tracer=tracer, capture_message_content=False)
        await tool_manager.dispatch(
            ToolCall(id="call1", name="nesting", args_json='{"text": "x"}')
        )
        inner_span, dispatch_span = exporter.get_finished_spans()
        assert inner_span.name == "inner"
        assert dispatch_span.name == "execute_tool nesting"
        assert dispatch_span.parent is None
        assert dispatch_span.context is not None
        assert inner_span.parent is not None
        assert inner_span.parent.span_id == dispatch_span.context.span_id

    asyncio.run(scenario())


def test_traced_initial_and_replacement_bind_preserve_tool_manager() -> None:
    """`TracedLLM.bind` and `TracedBoundLLM.bind` preserve `tools=ToolManager(...)`."""
    traced = TracedLLM(LLM(_FakeAdapter()), capture_message_content=False)
    bound_tool_manager = TracedToolManager([_echo_tool()], capture_message_content=False)
    bound = traced.bind(tools=bound_tool_manager)
    assert bound.tool_manager is bound_tool_manager
    replacement_tool_manager = ToolManager([_echo_tool()])
    replacement_bound = bound.bind(tools=replacement_tool_manager)
    assert replacement_bound.tool_manager is replacement_tool_manager


def test_traced_bind_sequences_construct_traced_tool_managers() -> None:
    """`TracedLLM.bind` and `TracedBoundLLM.bind` construct `TracedToolManager` from sequences."""

    async def scenario() -> None:
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter()),
            tracer=tracer,
            extra_attributes={"gen_ai.agent.name": "agent_a"},
            capture_message_content=True,
        )
        bound = traced.bind(tools=[_echo_tool()])
        replacement_bound = bound.bind(tools=[_echo_tool()])
        assert isinstance(bound.tool_manager, TracedToolManager)
        assert isinstance(replacement_bound.tool_manager, TracedToolManager)
        assert replacement_bound.tool_manager is not bound.tool_manager
        await bound.tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json='{"text": "a"}')
        )
        await replacement_bound.tool_manager.dispatch(
            ToolCall(id="call2", name="echo", args_json='{"text": "b"}')
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        for span in spans:
            assert span.attributes is not None
            assert span.attributes["gen_ai.agent.name"] == "agent_a"
            assert "gen_ai.tool.call.arguments" in span.attributes
            assert "gen_ai.tool.call.result" in span.attributes

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("colliding_key", "expected_value"),
    [("gen_ai.tool.name", "echo"), ("gen_ai.operation.name", "execute_tool")],
    ids=["tool_name", "operation_name"],
)
def test_extra_attributes_ride_on_a_dispatch_span_without_displacing_its_identity_keys(
    colliding_key: str, expected_value: str
) -> None:
    """A non-colliding extra lands on the span, and a dispatch-set key of the same name wins.

    Dispatch spans apply extras independently of generate spans.
    """

    async def scenario() -> None:
        """Dispatch under extra_attributes claiming the key, and inspect the span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()],
            tracer=tracer,
            extra_attributes={"gen_ai.agent.name": "agent_a", colliding_key: "spoofed"},
            capture_message_content=False,
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "a"}'))
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.agent.name"] == "agent_a"
        assert span.attributes[colliding_key] == expected_value

    asyncio.run(scenario())


def test_generate_many_passes_warm_cache_through() -> None:
    """warm_cache reaches BoundLLM.generate_many: the warming item never overlaps a sibling."""

    async def scenario() -> None:
        """Run a three-item batch on a slow fake with a wide concurrency bound and read the recorded peak."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.01)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=8)),
            tracer=tracer,
            capture_message_content=False,
        )
        results = await traced.bind().generate_many(
            [[UserMessage(content=str(index))] for index in range(3)], warm_cache=True
        )
        assert all(isinstance(result, Response) for result in results)
        assert adapter.bound_adapters[0].peak_in_flight == 2
        # The warming item is traced like every other item, so three items are three spans.
        assert len(exporter.get_finished_spans()) == 3

    asyncio.run(scenario())


def test_each_convention_defined_span_kind_carries_the_required_operation_name() -> None:
    """Each traced span carries its required gen_ai.operation.name."""

    async def scenario() -> None:
        """Open each span kind and inspect completion order."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=False)
        bound = traced.bind()
        await bound.generate_one("hi")
        await bound.generate_many([[UserMessage(content="hi")]])
        async with bound.stream_one("hi") as stream:
            await stream.final()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "hi"}'))
        spans = exporter.get_finished_spans()
        # The one-item batch opens that item's chat span and nothing around it.
        assert len(spans) == 4
        assert [span.name for span in spans] == [
            "chat fake-model",
            "chat fake-model",
            "chat fake-model",
            "execute_tool echo",
        ]
        assert [(span.attributes or {}).get("gen_ai.operation.name") for span in spans] == [
            "chat",
            "chat",
            "chat",
            "execute_tool",
        ]

    asyncio.run(scenario())


def test_extra_attributes_cannot_displace_the_operation_name() -> None:
    """A required attribute set at span start wins over an application constant of the same key."""

    async def scenario() -> None:
        """Generate under extra_attributes claiming the operation name key."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter()),
            tracer=tracer,
            capture_message_content=False,
            extra_attributes={"gen_ai.operation.name": "not-the-operation"},
        )
        await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.operation.name"] == "chat"

    asyncio.run(scenario())


def test_agent_span_carries_the_run_identity_and_summed_usage() -> None:
    """agent_span emits one INTERNAL invoke_agent span carrying the run's identity, usage, and extras."""
    tracer, exporter = _in_memory_tracer()
    spent = Usage(
        input_tokens_cache_read=2,
        input_tokens_cache_write=3,
        input_tokens_cache_none=5,
        output_tokens=7,
        output_tokens_reasoning=4,
        input_tokens_cache_read_cost_in_usd=0.0,
        input_tokens_cache_write_cost_in_usd=0.0,
        input_tokens_cache_none_cost_in_usd=0.0,
        output_tokens_cost_in_usd=0.5,
        provider_executed_tool_cost_in_usd=0.0,
    )
    with agent_span(
        tracer,
        agent_name="research_climate",
        agent_path="root/research_climate",
        usage=lambda: spent,
        extra_attributes=lambda: {"langchaint.agent.turns": 3},
    ) as span:
        assert span.is_recording()
    (finished,) = exporter.get_finished_spans()
    assert finished.name == "invoke_agent research_climate"
    assert finished.kind == SpanKind.INTERNAL
    # agent_span leaves status UNSET on success.
    # agent_span sets status ERROR when the wrapped body raises.
    # start_as_current_span uses the same UNSET default.
    assert finished.status.status_code == StatusCode.UNSET
    assert finished.attributes is not None
    assert dict(finished.attributes) == {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "research_climate",
        "langchaint.agent_path": "root/research_climate",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 7,
        "gen_ai.usage.reasoning.output_tokens": 4,
        "gen_ai.usage.cache_read.input_tokens": 2,
        "gen_ai.usage.cache_creation.input_tokens": 3,
        "langchaint.cost_in_usd": 0.5,
        "langchaint.agent.turns": 3,
    }


def test_agent_span_reads_usage_at_exit_and_records_the_spend_on_an_exception() -> None:
    """usage() is read on the way out, so a run that raises still closes the span with its final spend."""
    tracer, exporter = _in_memory_tracer()
    spent = Usage(
        input_tokens_cache_read=0,
        input_tokens_cache_write=0,
        input_tokens_cache_none=0,
        output_tokens=0,
        output_tokens_reasoning=0,
        input_tokens_cache_read_cost_in_usd=0.0,
        input_tokens_cache_write_cost_in_usd=0.0,
        input_tokens_cache_none_cost_in_usd=0.0,
        output_tokens_cost_in_usd=0.0,
        provider_executed_tool_cost_in_usd=0.0,
    )
    with (  # noqa: PT012
        pytest.raises(RuntimeError, match="loop gave up"),
        agent_span(
            tracer,
            agent_name="specialist",
            agent_path="root/specialist",
            usage=lambda: spent,
        ),
    ):
        spent = Usage(
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            input_tokens_cache_none=6,
            output_tokens=2,
            output_tokens_reasoning=0,
            input_tokens_cache_read_cost_in_usd=0.0,
            input_tokens_cache_write_cost_in_usd=0.0,
            input_tokens_cache_none_cost_in_usd=0.0,
            output_tokens_cost_in_usd=0.02,
            provider_executed_tool_cost_in_usd=0.0,
        )
        raise RuntimeError("loop gave up")
    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.attributes is not None
    assert finished.attributes["error.type"] == "RuntimeError"
    assert finished.attributes["gen_ai.usage.input_tokens"] == 6
    assert finished.attributes["gen_ai.usage.output_tokens"] == 2
    assert finished.attributes["langchaint.cost_in_usd"] == pytest.approx(0.02)


def test_agent_span_extra_attributes_cannot_displace_identity_or_usage_keys() -> None:
    """Agent identity and Usage attributes override colliding extras."""
    tracer, exporter = _in_memory_tracer()
    spent = Usage(
        input_tokens_cache_read=0,
        input_tokens_cache_write=0,
        input_tokens_cache_none=1,
        output_tokens=1,
        output_tokens_reasoning=0,
        input_tokens_cache_read_cost_in_usd=0.0,
        input_tokens_cache_write_cost_in_usd=0.0,
        input_tokens_cache_none_cost_in_usd=0.0,
        output_tokens_cost_in_usd=0.01,
        provider_executed_tool_cost_in_usd=0.0,
    )
    with agent_span(
        tracer,
        agent_name="specialist",
        agent_path="root/specialist",
        usage=lambda: spent,
        extra_attributes=lambda: {
            "gen_ai.operation.name": "chat",
            "gen_ai.agent.name": "impostor",
            "gen_ai.usage.input_tokens": 999,
            "langchaint.agent.turns": 2,
        },
    ):
        pass
    (finished,) = exporter.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert finished.attributes["gen_ai.agent.name"] == "specialist"
    assert finished.attributes["gen_ai.usage.input_tokens"] == 1
    assert finished.attributes["langchaint.agent.turns"] == 2


def test_agent_span_logs_a_raising_usage_callable_instead_of_propagating(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exit attribute failures log without replacing the body error."""
    tracer, exporter = _in_memory_tracer()

    def raising_usage() -> Usage:
        raise ValueError("usage read failed")

    def raising_extras() -> dict[str, int]:
        raise ValueError("extras read failed")

    with (
        pytest.raises(RuntimeError, match="loop gave up"),
        caplog.at_level(logging.WARNING, logger="langchaint.tracing"),
        agent_span(
            tracer,
            agent_name="specialist",
            agent_path="root/specialist",
            usage=raising_usage,
            extra_attributes=raising_extras,
        ),
    ):
        raise RuntimeError("loop gave up")
    assert "agent_span usage raised" in caplog.text
    assert "agent_span extra_attributes raised" in caplog.text
    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.attributes is not None
    assert "gen_ai.usage.input_tokens" not in finished.attributes
    assert finished.attributes["gen_ai.agent.name"] == "specialist"


def test_agent_span_ends_even_when_the_exit_attribute_pass_raises_a_base_exception() -> None:
    """A BaseException from usage still ends the agent span."""
    tracer, exporter = _in_memory_tracer()

    def raising_usage() -> Usage:
        raise KeyboardInterrupt

    with (
        pytest.raises(KeyboardInterrupt),
        agent_span(
            tracer, agent_name="specialist", agent_path="root/specialist", usage=raising_usage
        ),
    ):
        pass
    (finished,) = exporter.get_finished_spans()
    assert finished.name == "invoke_agent specialist"
    # The identity was set at span start, before usage() ran, so the interrupt does not discard it.
    assert finished.attributes is not None
    assert finished.attributes["gen_ai.agent.name"] == "specialist"


def _emitted_convention_keys() -> set[str]:
    """Collect quoted gen_ai.* literals from the tracing module."""
    source = pathlib.Path(inspect.getfile(langchaint.tracing)).read_text()
    return set(re.findall(r'"(gen_ai\.[a-z_.]+)"', source))


def test_vendored_schemas_and_the_payload_attributes_account_for_each_other() -> None:
    """Vendored schemas match mapped and emitted payload attributes.

    Each schema filename is derived from its payload attribute.
    """
    vendored = {path.name for path in _SEMCONV_GENAI_DIR.glob("gen-ai-*.json")}
    assert vendored, "no vendored schemas found, so this assertion would pass vacuously"
    assert vendored == set(_PAYLOAD_SCHEMA_FILES.values())
    assert set(_PAYLOAD_SCHEMA_FILES) <= _emitted_convention_keys()
    assert set(_PAYLOAD_SCHEMA_FILES) >= _UNVALIDATED_PAYLOAD_ATTRIBUTES
    for key, file in _PAYLOAD_SCHEMA_FILES.items():
        assert file == key.replace(".", "-").replace("_", "-") + ".json"


def test_refresh_creates_a_populated_source_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """git clone populates SOURCE_CHECKOUT before the cleanliness check."""
    source_checkout = tmp_path / "semantic-conventions-genai-source"
    resolved_sha = "a" * 40
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, working_directory: pathlib.Path = refresh_semconv_genai.ROOT
    ) -> str:
        assert working_directory in (refresh_semconv_genai.ROOT, source_checkout)
        commands.append(command)
        if command[:2] == ["git", "clone"]:
            assert "--no-checkout" not in command
            source_checkout.joinpath(".git").mkdir(parents=True)
            return ""
        if command == ["git", "remote", "get-url", "origin"]:
            return refresh_semconv_genai.SOURCE_URL
        if command == ["git", "status", "--porcelain", "--untracked-files=all"]:
            return ""
        if command == ["git", "rev-parse", "HEAD"]:
            return resolved_sha
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(refresh_semconv_genai, "SOURCE_CHECKOUT", source_checkout)
    monkeypatch.setattr(refresh_semconv_genai, "_run", fake_run)
    refresh_semconv_genai._prepare_source_checkout(resolved_sha)
    assert commands[0][:2] == ["git", "clone"]


def test_refresh_uses_the_workflow_prepared_source_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """PREPARED_SOURCE_REF prevents a second resolution of main."""
    source_checkout = tmp_path / "semantic-conventions-genai-source"
    source_checkout.joinpath(".git").mkdir(parents=True)
    resolved_sha = "a" * 40

    def fake_run(
        command: list[str], *, working_directory: pathlib.Path = refresh_semconv_genai.ROOT
    ) -> str:
        assert command == [
            "git",
            "rev-parse",
            "--verify",
            refresh_semconv_genai.PREPARED_SOURCE_REF,
        ]
        assert working_directory == source_checkout
        return resolved_sha

    monkeypatch.setattr(refresh_semconv_genai, "SOURCE_CHECKOUT", source_checkout)
    monkeypatch.setattr(refresh_semconv_genai, "_run", fake_run)
    assert refresh_semconv_genai._resolved_source_sha() == resolved_sha


def test_refresh_resolves_main_without_a_prepared_source_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A missing PREPARED_SOURCE_REF makes a local refresh resolve main."""
    source_checkout = tmp_path / "semantic-conventions-genai-source"
    source_checkout.joinpath(".git").mkdir(parents=True)
    resolved_sha = "a" * 40

    def missing_prepared_ref(
        command: list[str], *, working_directory: pathlib.Path = refresh_semconv_genai.ROOT
    ) -> str:
        assert command == [
            "git",
            "rev-parse",
            "--verify",
            refresh_semconv_genai.PREPARED_SOURCE_REF,
        ]
        assert working_directory == source_checkout
        raise subprocess.CalledProcessError(1, command)

    def resolved_main_sha() -> str:
        return resolved_sha

    monkeypatch.setattr(refresh_semconv_genai, "SOURCE_CHECKOUT", source_checkout)
    monkeypatch.setattr(refresh_semconv_genai, "_run", missing_prepared_ref)
    monkeypatch.setattr(refresh_semconv_genai, "_resolved_main_sha", resolved_main_sha)
    assert refresh_semconv_genai._resolved_source_sha() == resolved_sha


def test_the_exempted_attribute_still_disagrees_with_its_schema() -> None:
    """The exempted payload still violates its schema."""
    assert frozenset({"gen_ai.tool.call.arguments"}) == _UNVALIDATED_PAYLOAD_ATTRIBUTES

    async def scenario() -> None:
        """Dispatch malformed arguments and validate the emitted payload."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=True
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json="not json at all"))
        emitted = _captured(exporter, "gen_ai.tool.call.arguments")
        schema = _payload_schema(_PAYLOAD_SCHEMA_FILES["gen_ai.tool.call.arguments"])
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(emitted)

    asyncio.run(scenario())


def test_every_payload_attribute_reaches_validation() -> None:
    """Each payload attribute reaches schema validation."""

    async def scenario() -> None:
        """Generate and dispatch values containing each payload attribute."""
        tracer, _exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            system_prompt="be brief",
            tools=ToolManager([_echo_tool()]),
        )
        await bound.generate_one([UserMessage(content="look it up")])
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=True
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "x"}'))

    _VALIDATED_PAYLOAD_ATTRIBUTES.clear()
    asyncio.run(scenario())
    assert set(_PAYLOAD_SCHEMA_FILES) == _VALIDATED_PAYLOAD_ATTRIBUTES


def test_a_payload_that_violates_its_schema_fails_the_span() -> None:
    """Span completion propagates payload schema violations."""
    tracer, _exporter = _in_memory_tracer()
    with (
        pytest.raises(AssertionError, match=re.escape("gen_ai.output.messages violates")),
        tracer.start_as_current_span("chat") as span,
    ):
        span.set_attribute("gen_ai.output.messages", json.dumps({"role": "assistant"}))


_REASONING_ONLY_OUTCOME = AdapterResult(
    output="",
    assistant_message=AssistantMessage(turn=(ReasoningPart(raw={"signature": "opaque"}),)),
    stop_reason="end_turn",
)
_EMPTY_TEXT_TURN_OUTCOME = AdapterResult(
    output="",
    assistant_message=AssistantMessage(
        turn=(
            ReasoningPart(raw={"signature": "opaque"}, text=""),
            TextPart(text=""),
        )
    ),
    stop_reason="end_turn",
)
_REASONING_WITH_TEXT_OUTCOME = AdapterResult(
    output="answer",
    assistant_message=AssistantMessage(
        turn=(
            ReasoningPart(raw={"signature": "opaque"}, text="thought it over"),
            TextPart(text="answer"),
        )
    ),
    stop_reason="end_turn",
)


def _captured(exporter: InMemorySpanExporter, key: str) -> object:
    """Read one span's JSON content attribute back as Python data."""
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    value = span.attributes[key]
    assert isinstance(value, str)
    parsed: object = json.loads(value)
    return parsed


def test_capture_off_leaves_every_content_key_off_the_span() -> None:
    """capture_message_content=False omits content attributes with populated sources."""

    async def scenario() -> None:
        """Generate with populated content sources and capture disabled."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeAdapter(echo=True)), tracer=tracer, capture_message_content=False
        )
        bound = traced.bind(
            system_prompt="be brief",
            tools=ToolManager([_echo_tool()]),
        )
        await bound.generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert not {
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "gen_ai.input.messages",
            "gen_ai.output.messages",
        } & set(span.attributes)

    asyncio.run(scenario())


def test_capture_on_records_all_four_content_attributes_in_convention_shape() -> None:
    """capture_message_content True records the system prompt, tools, GenerationInput, and assistant turn."""

    async def scenario() -> None:
        """Generate over a Sequence[Message] carrying every message role and inspect the shapes."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            system_prompt="be brief",
            tools=ToolManager([_echo_tool()]),
        )
        await bound.generate_one([
            UserMessage(content="look it up"),
            AssistantMessage(turn=(ToolCall(id="call1", name="echo", args_json='{"text": "x"}'),)),
            ToolMessage(tool_call_id="call1", content="x"),
        ])
        assert _captured(exporter, "gen_ai.system_instructions") == [
            {"type": "text", "content": "be brief"}
        ]
        assert _captured(exporter, "gen_ai.tool.definitions") == [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo the text back",
                "parameters": _EchoToolArgs.model_json_schema(),
            }
        ]
        assert _captured(exporter, "gen_ai.input.messages") == [
            {"role": "user", "parts": [{"type": "text", "content": "look it up"}]},
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call1",
                        "name": "echo",
                        "arguments": {"text": "x"},
                    }
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call1",
                        "is_error": False,
                        "response": [{"type": "text", "content": "x"}],
                    }
                ],
            },
        ]
        assert _captured(exporter, "gen_ai.output.messages") == [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "ok"}],
                "finish_reason": "stop",
            }
        ]

    asyncio.run(scenario())


def test_a_str_generation_input_is_captured_as_one_user_message() -> None:
    """The bare-str GenerationInput form renders as the one user message it means."""

    async def scenario() -> None:
        """Generate from a str GenerationInput and read the input messages back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        await traced.bind().generate_one("hi")
        assert _captured(exporter, "gen_ai.input.messages") == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]

    asyncio.run(scenario())


def test_image_part_image_url_part_and_audio_part_capture_metadata_without_data() -> None:
    """ImagePart and AudioPart omit data. ImageUrlPart records URL metadata."""

    async def scenario() -> None:
        """Generate over Sequence[Message] containing ImagePart, ImageUrlPart, and AudioPart."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        await traced.bind().generate_one([
            UserMessage(
                content=(
                    TextPart(text="what is this"),
                    ImagePart(data=b"\x89PNGsecret", media_type="image/png"),
                    ImageUrlPart(
                        url="https://example.com/image.png",
                        media_type="image/png",
                    ),
                    ImageUrlPart(url="https://example.com/unknown"),
                    AudioPart(data=b"WAVsecret", media_type="audio/wav"),
                )
            )
        ])
        assert _captured(exporter, "gen_ai.input.messages") == [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "what is this"},
                    {"type": "blob", "mime_type": "image/png"},
                    {
                        "type": "image_url",
                        "url": "https://example.com/image.png",
                        "mime_type": "image/png",
                    },
                    {"type": "image_url", "url": "https://example.com/unknown"},
                    {"type": "blob", "mime_type": "audio/wav"},
                ],
            }
        ]
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "PNGsecret" not in str(span.attributes["gen_ai.input.messages"])
        assert "WAVsecret" not in str(span.attributes["gen_ai.input.messages"])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "outcome",
    [_REASONING_ONLY_OUTCOME, _EMPTY_TEXT_TURN_OUTCOME],
    ids=["text_free_reasoning_alone", "empty_text_on_every_part"],
)
def test_a_turn_carrying_no_readable_text_emits_its_message_with_an_empty_parts_array(
    outcome: AdapterResult[str],
) -> None:
    """A turn without readable text records empty output parts."""

    async def scenario() -> None:
        """Generate the turn and read the output messages back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(
                _FakeAdapter(scripted_attempts=[_ScriptedResponse(outcome=outcome, usage=_USAGE)])
            ),
            tracer=tracer,
            capture_message_content=True,
        )
        await traced.bind().generate_one("hi")
        assert _captured(exporter, "gen_ai.output.messages") == [
            {"role": "assistant", "parts": [], "finish_reason": "stop"}
        ]
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "opaque" not in str(span.attributes["gen_ai.output.messages"])

    asyncio.run(scenario())


def test_reasoning_text_becomes_a_reasoning_part_without_its_payload() -> None:
    """ReasoningPart.text emits as the convention's reasoning part.

    ReasoningPart.raw stays off the span when ReasoningPart.text is present.
    The span receives the readable copy without the signature.
    """

    async def scenario() -> None:
        """Generate a texted ReasoningPart and TextPart, then read both messages."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(
                _FakeAdapter(
                    scripted_attempts=[
                        _ScriptedResponse(outcome=_REASONING_WITH_TEXT_OUTCOME, usage=_USAGE)
                    ]
                )
            ),
            tracer=tracer,
            capture_message_content=True,
        )
        await traced.bind().generate_one("hi")
        assert _captured(exporter, "gen_ai.output.messages") == [
            {
                "role": "assistant",
                "parts": [
                    {"type": "reasoning", "content": "thought it over"},
                    {"type": "text", "content": "answer"},
                ],
                "finish_reason": "stop",
            }
        ]
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "opaque" not in str(span.attributes["gen_ai.output.messages"])

    asyncio.run(scenario())


def test_an_absent_system_prompt_omits_its_key_while_capture_stays_on() -> None:
    """No bound system prompt omits gen_ai.system_instructions. The GenerationInput is still captured.

    The captured input messages are what separates this from the capture-off case.
    """

    async def scenario() -> None:
        """Generate under a binding with no system prompt and no tools."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.system_instructions" not in span.attributes
        assert "gen_ai.tool.definitions" not in span.attributes
        assert "gen_ai.input.messages" in span.attributes

    asyncio.run(scenario())


def test_system_prompt_parts_become_one_instruction_element_each() -> None:
    """A parts-form system prompt emits one text element per part."""

    async def scenario() -> None:
        """Bind a two-part system prompt and read the instructions back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            system_prompt=[TextPart(text="be brief"), TextPart(text="cite sources")],
        )
        await bound.generate_one("hi")
        assert _captured(exporter, "gen_ai.system_instructions") == [
            {"type": "text", "content": "be brief"},
            {"type": "text", "content": "cite sources"},
        ]

    asyncio.run(scenario())


def test_the_error_path_captures_input_and_the_turn_the_failure_carried() -> None:
    """A failed call keeps the input attributes set at span start and emits the turn it recorded."""

    async def scenario() -> None:
        """Drive a refusal under capture and inspect the error span."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_REFUSAL)])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=True,
        )
        with pytest.raises(GenerationError):
            await traced.bind(system_prompt="be brief").generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.input.messages" in span.attributes
        assert "gen_ai.system_instructions" in span.attributes
        assert span.attributes["error.type"] == "RefusalErrorRecord"
        (message,) = json.loads(str(span.attributes["gen_ai.output.messages"]))
        assert message["role"] == "assistant"
        assert message["finish_reason"] == "refusal"

    asyncio.run(scenario())


def test_a_failure_that_produced_no_turn_emits_no_output_messages() -> None:
    """The output key is omitted when no attempt reached a 200, there being no turn to record."""

    async def scenario() -> None:
        """Exhaust the retry budget on transport failures and inspect the error span."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("e1"), TransientError("e2")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=True,
        )
        with pytest.raises(GenerationError):
            await traced.bind(max_attempts=2).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.input.messages" in span.attributes
        assert "gen_ai.output.messages" not in span.attributes

    asyncio.run(scenario())


def test_generate_many_captures_each_items_own_generation_input_under_capture() -> None:
    """Each item's span carries that item's generation_input, not the batch's, which has no single one."""

    async def scenario() -> None:
        """Run a two-item batch under capture and read the content off each item's span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(
                _FakeAdapter(echo=True),
                shared_backoff=_fast_shared_backoff(max_concurrent_requests=1),
            ),
            tracer=tracer,
            capture_message_content=True,
        )
        await traced.bind().generate_many(["a", "b"])
        first, second = exporter.get_finished_spans()
        assert "a" in str(_attribute(first, "gen_ai.input.messages"))
        assert "b" not in str(_attribute(first, "gen_ai.input.messages"))
        assert "b" in str(_attribute(second, "gen_ai.input.messages"))
        assert "a" in str(_attribute(first, "gen_ai.output.messages"))

    asyncio.run(scenario())


def test_content_that_cannot_be_serialized_is_logged_and_never_reaches_the_caller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSONSchemaTool args_schema json.dumps cannot serialize leaves the call and its result intact.

    The tracing wrapper catches serialization errors.
    The three input keys build as one dict, so the failure drops all three rather than a subset.
    """

    async def scenario() -> None:
        """Generate under capture with the unserializable tool bound, then read the span and the log."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            tools=ToolManager([_unserializable_schema_tool()]),
        )
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            response = await bound.generate_one("hi")
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert not {
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "gen_ai.input.messages",
        } & set(span.attributes)
        assert "gen_ai.output.messages" in span.attributes
        assert "gen_ai.usage.output_tokens" in span.attributes
        assert "content capture raised" in caplog.text

    asyncio.run(scenario())


def test_unserializable_content_leaves_the_stream_and_its_span_intact(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unserializable input content preserves the stream and span."""

    async def scenario() -> None:
        """Stream to completion with the unserializable tool bound, then read the span and the log."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            tools=ToolManager([_unserializable_schema_tool()]),
        )
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            async with bound.stream_one("hi") as stream:
                _ = [item async for item in stream]
                response = await stream.final()
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.output.messages" in span.attributes
        assert "gen_ai.response.time_to_first_chunk" in span.attributes
        assert "content capture raised" in caplog.text

    asyncio.run(scenario())


def test_the_stream_span_captures_input_at_start_and_output_at_final() -> None:
    """A traced stream records the input attributes when its span starts and the turn at final()."""

    async def scenario() -> None:
        """Drive a stream to completion under capture and read both sides back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(system_prompt="be brief")
        async with bound.stream_one("hi") as stream:
            _ = [item async for item in stream]
            await stream.final()
        assert _captured(exporter, "gen_ai.input.messages") == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]
        assert _captured(exporter, "gen_ai.system_instructions") == [
            {"type": "text", "content": "be brief"}
        ]
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.output.messages" in span.attributes

    asyncio.run(scenario())


def test_capture_survives_bind_and_reaches_the_replacement_binding() -> None:
    """A replacement object retains capture_message_content."""

    async def scenario() -> None:
        """Replace a captured binding and confirm the new one still captures."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        replacement_bound = traced.bind(system_prompt="s").bind(system_prompt="s2")
        await replacement_bound.generate_one("hi")
        assert _captured(exporter, "gen_ai.system_instructions") == [
            {"type": "text", "content": "s2"}
        ]

    asyncio.run(scenario())


def test_tool_span_captures_arguments_and_result_under_capture() -> None:
    """A dispatch span records the arguments as an object and the tool_message as a tool_call_response part."""

    async def scenario() -> None:
        """Dispatch one valid call under capture and read both content keys."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=True
        )
        await tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json='{"text":"hi",   "n":1}')
        )
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        # The expected value checks the attribute string after normalization.
        # Its spacing differs from args_json.
        # Decoding the attribute would hide a missing normalization step.
        assert span.attributes["gen_ai.tool.call.arguments"] == '{"text": "hi", "n": 1}'
        assert _captured(exporter, "gen_ai.tool.call.result") == {
            "type": "tool_call_response",
            "id": "call1",
            "is_error": False,
            "response": [{"type": "text", "content": "hi"}],
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "args_json",
    ['{"n": 1e400}', '{"n": Infinity}', '{"n": -Infinity}', '{"n": NaN}'],
)
def test_tool_span_arguments_fall_back_when_the_parse_cannot_re_serialize_as_json(
    args_json: str,
) -> None:
    """Non-finite argument values fall back to args_json."""

    async def scenario() -> None:
        """Dispatch a call carrying the number and read the attribute back."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=True
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json=args_json))
        assert _captured(exporter, "gen_ai.tool.call.arguments") == args_json

    asyncio.run(scenario())


def test_tool_span_arguments_fall_back_to_the_raw_text_when_the_json_does_not_parse() -> None:
    """Malformed args_json is preserved as a JSON string."""

    async def scenario() -> None:
        """Dispatch a call whose argument text is not JSON and read the attribute back."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=True
        )
        outcome = await tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json="not json at all")
        )
        assert isinstance(outcome, DispatchInvalidToolArgs)
        assert _captured(exporter, "gen_ai.tool.call.arguments") == "not json at all"

    asyncio.run(scenario())


def test_input_tool_calls_nest_parsed_arguments_and_keep_unparseable_text() -> None:
    """Input tool calls preserve parsed objects and malformed text."""

    async def scenario() -> None:
        """Generate over a turn holding one parseable and one unparseable tool call."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        bound = traced.bind()
        await bound.generate_one([
            AssistantMessage(
                turn=(
                    ToolCall(id="call1", name="echo", args_json='{"text": "x"}'),
                    ToolCall(id="call2", name="echo", args_json="{oops"),
                )
            )
        ])
        assert _captured(exporter, "gen_ai.input.messages") == [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call1",
                        "name": "echo",
                        "arguments": {"text": "x"},
                    },
                    {"type": "tool_call", "id": "call2", "name": "echo", "arguments": "{oops"},
                ],
            }
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [("[1, 2]", [1, 2]), ('"bare"', "bare"), ("7", 7), ("null", None)],
)
def test_a_non_object_argument_value_still_nests_as_the_value_it_parses_to(
    args_json: str, expected: object
) -> None:
    """Input tool calls preserve parsed non-object values."""

    async def scenario() -> None:
        """Generate over a turn whose tool call carries a non-object argument value."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        await traced.bind().generate_one([
            AssistantMessage(turn=(ToolCall(id="call1", name="echo", args_json=args_json),))
        ])
        assert _captured(exporter, "gen_ai.input.messages") == [
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool_call", "id": "call1", "name": "echo", "arguments": expected}
                ],
            }
        ]

    asyncio.run(scenario())


def test_an_ordinary_float_argument_survives_the_parse() -> None:
    """A finite float remains numeric in traced tool arguments."""

    async def scenario() -> None:
        """Generate over a turn whose tool call carries a finite float and read the arguments back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeAdapter()), tracer=tracer, capture_message_content=True)
        await traced.bind().generate_one([
            AssistantMessage(
                turn=(ToolCall(id="call1", name="echo", args_json='{"n": 1.5, "big": 1e300}'),)
            )
        ])
        assert _captured(exporter, "gen_ai.input.messages") == [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call1",
                        "name": "echo",
                        "arguments": {"n": 1.5, "big": 1e300},
                    }
                ],
            }
        ]

    asyncio.run(scenario())


def test_tool_span_capture_off_omits_both_content_keys() -> None:
    """capture_message_content False leaves the arguments and result off the dispatch span."""

    async def scenario() -> None:
        """Dispatch one valid call with capture off and confirm neither key is present."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "hi"}'))
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.tool.call.arguments" not in span.attributes
        assert "gen_ai.tool.call.result" not in span.attributes

    asyncio.run(scenario())


def test_tool_span_captures_the_result_on_both_variants_where_no_tool_ran() -> None:
    """Tool failure spans record the model-facing result and error.type."""

    async def scenario() -> None:
        """Dispatch an off-list name and an invalid-argument call under capture, checking each span."""
        for call, expected_error_type in (
            (ToolCall(id="call1", name="missing", args_json="{}"), "unknown_tool"),
            (ToolCall(id="call2", name="echo", args_json='{"wrong": 1}'), "invalid_tool_args"),
        ):
            tracer, exporter = _in_memory_tracer()
            expected = await ToolManager([_echo_tool()]).dispatch(call)
            assert isinstance(expected.tool_message.content, str)
            tool_manager = TracedToolManager(
                [_echo_tool()], tracer=tracer, capture_message_content=True
            )
            await tool_manager.dispatch(call)
            (span,) = exporter.get_finished_spans()
            assert span.attributes is not None
            assert span.attributes["error.type"] == expected_error_type
            assert _captured(exporter, "gen_ai.tool.call.result") == {
                "type": "tool_call_response",
                "id": call.id,
                "is_error": expected.tool_message.is_error,
                "response": [{"type": "text", "content": expected.tool_message.content}],
            }

    asyncio.run(scenario())


_CONTENT_SENTINEL = "sentinel-string-no-ungated-channel-may-carry"
"""The generated text the content-rule test traces through every reporting channel."""


@pytest.mark.parametrize("capture_message_content", [False, True])
def test_a_failures_turn_reaches_a_span_only_through_the_gated_output_key(
    *, capture_message_content: bool
) -> None:
    """A failed turn reaches spans only through gated gen_ai.output.messages."""
    turn = AssistantMessage(turn=(TextPart(text=_CONTENT_SENTINEL),))

    async def scenario() -> None:
        """Fail two attempts and inspect content channels."""
        adapter = _FakeAdapter(
            scripted_attempts=[
                TransientError("the first attempt failed"),
                _billed(Refusal(assistant_message=turn)),
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=capture_message_content,
        )
        with pytest.raises(GenerationError) as raised:
            await traced.bind(max_attempts=3).generate_one("hi")

        error = raised.value
        assert error.attempts == 2
        assert _CONTENT_SENTINEL not in error.error_text
        assert _CONTENT_SENTINEL not in str(error)
        assert error.assistant_message == turn
        assert _CONTENT_SENTINEL in str(to_tables(error).attempts[-1]["assistant_message_json"])

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        carrying = {
            key for key, value in span.attributes.items() if _CONTENT_SENTINEL in str(value)
        }
        assert carrying == ({"gen_ai.output.messages"} if capture_message_content else set())
        for event in span.events:
            assert event.attributes is not None
            assert not any(_CONTENT_SENTINEL in str(value) for value in event.attributes.values())
        assert _CONTENT_SENTINEL not in str(span.status.description)

    asyncio.run(scenario())


def test_a_turn_whose_result_states_no_stop_reason_reports_the_error_finish_reason() -> None:
    """A failed turn without stop_reason records finish_reason="error"."""

    async def scenario() -> None:
        """Capture an unfinished turn's finish_reason."""
        adapter = _FakeAdapter(
            scripted_attempts=[
                _billed(UnfinishedTurn(assistant_message=_REJECTED_TURN, reason="in_progress"))
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(adapter, shared_backoff=_fast_shared_backoff()),
            tracer=tracer,
            capture_message_content=True,
        )
        with pytest.raises(GenerationError) as raised:
            await traced.bind().generate_one("hi")
        assert raised.value.stop_reason is None
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        (message,) = json.loads(str(span.attributes["gen_ai.output.messages"]))
        assert message["finish_reason"] == "error"

    asyncio.run(scenario())
