"""Test tracing with fake adapters and an in-memory exporter.

ValidatingSpanExporter validates attribute keys and payload attributes against tests/semconv_genai.
The tests inspect recorded span names, kinds, statuses, attributes, events, and parents.
Tests of the pure content renderers call them directly.
"""

import json
import logging
import pathlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import ClassVar, Literal, NamedTuple, override

import jsonschema
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import BaseModel, JsonValue, TypeAdapter

from langchaint import (
    LLM,
    AbandonedStreamRecord,
    AssistantMessage,
    AssistantPart,
    AudioPart,
    BoundLLM,
    ContentPart,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchOutcome,
    DispatchUnknownTool,
    Generation,
    GenerationError,
    GenerationOutcome,
    GenerationWithoutToolCalls,
    ImagePart,
    ImageUrlPart,
    JSONSchemaTool,
    PydanticTool,
    ReasoningPart,
    StreamHandle,
    StreamItem,
    TextPart,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolReturnExplicit,
    UserMessage,
    to_tables,
)
from langchaint.adapter import (
    AdapterStream,
    Refusal,
    RejectedMessages,
    RequestParams,
    TransientError,
    UsableResponse,
)
from langchaint.common.messages import StopReason
from langchaint.span_parsing import generation_input_from_otel, parse_otel
from langchaint.tracing import (
    AttributeMapper,
    ContentFilter,
    OtelObserver,
    SpanAttributes,
    _input_content_attributes,
    _output_content_attributes,
    _record_every_part,
    _tool_call_arguments,
    gen_ai_attributes,
)
from scripts import refresh_semconv_genai
from tests.fake_adapter import (
    MAX_COMPLETION_TOKENS_EXCEEDED,
    REFUSAL,
    REJECTED_ASSISTANT_MESSAGE,
    USAGE,
    FakeAdapter,
    FakeBoundAdapter,
    FakeStream,
    HangsAfterFirstItemStream,
    ScriptedResponse,
    billed,
    fast_shared_backoff,
)
from tests.helpers import (
    LANGCHAINT_KEYS,
    SEMCONV_GENAI_DIR,
    ValidatingSpanExporter,
    payload_schema,
    run_with_timeout,
    time_out_when,
    transient,
)

_APPLICATION_KEYS = frozenset({
    "custom.agent",
    "custom.mapped_output",
    "custom.model",
    "shared.key",
})
"""The keys this module's mappers and `extra_attributes` set as an application would."""


def _in_memory_tracer_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Build a tracer provider whose in-memory exporter validates the spans a test reads."""
    exporter = ValidatingSpanExporter(application_keys=_APPLICATION_KEYS)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider, exporter


def _traced(
    adapter: FakeAdapter,
    *,
    capture_message_content: bool = False,
    attribute_mapper: AttributeMapper = gen_ai_attributes,
    extra_attributes: SpanAttributes | None = None,
    max_concurrent_requests: int = 8,
) -> tuple[LLM, InMemorySpanExporter]:
    """Build an `LLM` over `adapter` whose `OtelObserver` spans reach a schema-validating in-memory exporter.

    `max_concurrent_requests=1` serializes a batch, so its item spans end in input order.
    """
    tracer_provider, exporter = _in_memory_tracer_provider()
    llm = LLM(
        adapter,
        shared_backoff=fast_shared_backoff(max_concurrent_requests=max_concurrent_requests),
        observer=OtelObserver(
            capture_message_content=capture_message_content,
            attribute_mapper=attribute_mapper,
            extra_attributes=extra_attributes,
            tracer_provider=tracer_provider,
        ),
    )
    return llm, exporter


type _GenerationPath = Literal["generate", "stream"]


async def _generate_through[ToolManagerT: ToolManager | None](
    path: _GenerationPath, bound_llm: BoundLLM[str, ToolManagerT]
) -> Generation[str]:
    """Run one input through `generate_one`, or through `stream_one` drained before its `final()`.

    Raises:
        GenerationError: the input ends in a terminal failure.
    """
    if path == "generate":
        return await bound_llm.generate_one("hi")
    async with bound_llm.stream_one("hi") as handle:
        _ = [item async for item in handle]
        return await handle.final()


def _attribute(span: ReadableSpan, key: str) -> object:
    """Read one attribute off a finished span, None where the span carries no such key."""
    return (span.attributes or {}).get(key)


def _json_attribute(span: ReadableSpan, key: str) -> object:
    """Read one JSON content attribute off a finished span as Python data."""
    value = _attribute(span, key)
    assert isinstance(value, str)
    parsed: object = json.loads(value)
    return parsed


def _captured(exporter: InMemorySpanExporter, key: str) -> object:
    """Read one JSON content attribute off the only finished span as Python data."""
    (span,) = exporter.get_finished_spans()
    return _json_attribute(span, key)


class _MidFailStream(FakeStream):
    """A stream that yields one item, then raises so the failure lands mid-iteration."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield one chunk, then raise a plain exception the adapter places as transient.

        Yields:
            One text chunk before the raise.

        Raises:
            ValueError: always, after the first yield.
        """
        yield "a"
        raise ValueError("mid-stream boom")


def test_generate_one_generation_produces_one_fully_attributed_span() -> None:
    """A generation emits one CLIENT span named "chat {model}", OK status, and every gen_ai attribute.

    Capture is off, so the bound system prompt and tools leave no content attribute on the span.
    """

    async def scenario() -> None:
        """Drive one generate_one to a generation and inspect the single finished span."""
        llm, exporter = _traced(FakeAdapter(echo=True))
        bound = llm.bind(system_prompt="be brief", tools=ToolManager([_echo_tool()]))
        result = await bound.generate_one("hi")
        assert result.kind == "without_tool_calls"
        assert result.output == "hi"
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
            "gen_ai.usage.input_tokens": USAGE.input_tokens_total,
            "gen_ai.usage.output_tokens": USAGE.output_tokens,
            "gen_ai.usage.reasoning.output_tokens": USAGE.output_tokens_reasoning,
            "gen_ai.usage.cache_read.input_tokens": USAGE.input_tokens_cache_read,
            "gen_ai.usage.cache_write.input_tokens": USAGE.input_tokens_cache_write,
            "langchaint.request_count": 1,
            "langchaint.cost_in_usd": 0.0,
        }

    run_with_timeout(scenario())


class _CurrentSpanRecordingBoundAdapter(FakeBoundAdapter):
    """A fake bound adapter that records the current span at each open_stream call."""

    def __init__(self, adapter: FakeAdapter) -> None:
        """Start with no recorded span."""
        super().__init__(adapter)
        self.spans_current_at_open: list[trace.Span] = []

    @override
    async def open_stream(self, request_params: RequestParams) -> AdapterStream:
        """Record the current span, then open as FakeBoundAdapter does.

        Raises:
            Exception: whatever FakeBoundAdapter.open_stream raises.
        """
        self.spans_current_at_open.append(trace.get_current_span())
        return await super().open_stream(request_params)


class _CurrentSpanRecordingAdapter(FakeAdapter):
    """A fake adapter whose bound adapters record the current span at each open."""

    _bound_adapter_class: ClassVar[type[FakeBoundAdapter]] = _CurrentSpanRecordingBoundAdapter


@pytest.mark.parametrize("path", ["generate", "stream"])
def test_the_chat_span_is_current_during_generate_one_and_never_during_a_stream(
    path: _GenerationPath,
) -> None:
    """generate_one makes its chat span current, so spans that adapter and SDK code starts nest under it.

    A stream leaves the caller's span current, because the caller's code runs between its items.
    """

    async def scenario() -> None:
        """Run the input and compare the span current at open with the finished chat span."""
        adapter = _CurrentSpanRecordingAdapter()
        llm, exporter = _traced(adapter)
        await _generate_through(path, llm.bind())
        (bound_adapter,) = adapter.bound_adapters
        assert isinstance(bound_adapter, _CurrentSpanRecordingBoundAdapter)
        (span_current_at_open,) = bound_adapter.spans_current_at_open
        (chat_span,) = exporter.get_finished_spans()
        assert chat_span.context is not None
        if path == "generate":
            assert span_current_at_open.get_span_context() == chat_span.context
        else:
            assert span_current_at_open is trace.INVALID_SPAN

    run_with_timeout(scenario())


class _TerminalCase(NamedTuple):
    """One input that ends in GenerationError, and what its span must record."""

    adapter: Callable[[], FakeAdapter]
    error_type: str
    status_description: str | None
    finish_reasons: tuple[str, ...] | None
    """The finish reasons of the 200 that ended the input, or None where no request reached a 200."""
    request_count: int
    request_failed_events: int


@pytest.mark.parametrize(
    "case",
    [
        _TerminalCase(
            adapter=lambda: FakeAdapter(stream=FakeStream(outcome=REFUSAL)),
            error_type="refusal_error",
            status_description=None,
            finish_reasons=("refusal",),
            request_count=1,
            request_failed_events=0,
        ),
        _TerminalCase(
            adapter=lambda: FakeAdapter(stream=FakeStream(outcome=MAX_COMPLETION_TOKENS_EXCEEDED)),
            error_type="max_completion_tokens_exceeded_error",
            status_description=None,
            finish_reasons=("length",),
            request_count=1,
            request_failed_events=0,
        ),
        _TerminalCase(
            adapter=lambda: FakeAdapter(
                scripted_requests=[TransientError("connection reset")] * 2
            ),
            error_type="retries_exhausted_error",
            status_description="request 1: connection reset\nrequest 2: connection reset",
            finish_reasons=None,
            request_count=2,
            request_failed_events=2,
        ),
        _TerminalCase(
            adapter=lambda: FakeAdapter(
                rejected_messages=[RejectedMessages(error_text="misconfigured")]
            ),
            error_type="rejected_error",
            status_description="misconfigured",
            finish_reasons=None,
            request_count=0,
            request_failed_events=0,
        ),
    ],
    ids=["refusal", "max_completion_tokens_exceeded", "retries_exhausted", "rejected"],
)
@pytest.mark.parametrize("path", ["generate", "stream"])
def test_a_generation_error_ends_the_span_with_error_status_and_the_inputs_attributes(
    path: _GenerationPath, case: _TerminalCase
) -> None:
    """A GenerationError ends the span with error status, error.type, and the input's billing.

    A 200 that ended the input contributes its real tokens, cost, finish reason, and assistant message.
    An input that reached no 200 records zero usage and no finish reason, response model, or output.
    The input attributes set at span start stay on the span.
    """

    async def scenario() -> None:
        """Drive the input under capture and inspect the error span."""
        llm, exporter = _traced(case.adapter(), capture_message_content=True)
        with pytest.raises(GenerationError):
            await _generate_through(path, llm.bind(system_prompt="be brief", max_requests=2))
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == case.status_description
        assert span.attributes is not None
        assert span.attributes["error.type"] == case.error_type
        assert span.attributes["langchaint.request_count"] == case.request_count
        assert [event.name for event in span.events] == [
            "langchaint.request_failed"
        ] * case.request_failed_events
        assert _json_attribute(span, "gen_ai.input.messages") == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]
        assert _json_attribute(span, "gen_ai.system_instructions") == [
            {"type": "text", "content": "be brief"}
        ]
        reached_a_200 = case.finish_reasons is not None
        assert span.attributes["langchaint.cost_in_usd"] == (0.25 if reached_a_200 else 0.0)
        assert span.attributes["gen_ai.usage.input_tokens"] == (
            USAGE.input_tokens_total if reached_a_200 else 0
        )
        assert span.attributes["gen_ai.usage.output_tokens"] == (
            USAGE.output_tokens if reached_a_200 else 0
        )
        assert _attribute(span, "gen_ai.response.finish_reasons") == case.finish_reasons
        assert ("gen_ai.response.model" in span.attributes) is reached_a_200
        if case.finish_reasons is None:
            assert "gen_ai.output.messages" not in span.attributes
        else:
            output_messages = _json_attribute(span, "gen_ai.output.messages")
            assert isinstance(output_messages, list)
            assert [message["finish_reason"] for message in output_messages] == list(
                case.finish_reasons
            )

    run_with_timeout(scenario())


async def _drain_by_iterating(handle: StreamHandle[str]) -> None:
    """Consume the stream item by item, never calling final()."""
    async for _ in handle:
        pass


async def _drain_by_final(handle: StreamHandle[str]) -> None:
    """Ask final() for the Generation, never iterating."""
    await handle.final()


@pytest.mark.parametrize("drain", [_drain_by_iterating, _drain_by_final])
def test_a_traced_streams_expired_deadline_takes_error_status(
    drain: Callable[[StreamHandle[str]], Awaitable[None]],
) -> None:
    """Stream deadlines record GenerationError for each drain method."""

    async def scenario() -> None:
        """Drain a stream that stalls after its first item, under a deadline it outlasts."""
        llm, exporter = _traced(FakeAdapter(stream=HangsAfterFirstItemStream()))
        handle = llm.bind().stream_one("hi", timeout_seconds=0.02)

        with pytest.raises(GenerationError):
            async with handle:
                await drain(handle)
        assert handle.abandoned is None
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description is None
        assert span.attributes is not None
        assert span.attributes["error.type"] == "timed_out_error"
        assert "gen_ai.response.time_to_first_chunk" in span.attributes

    run_with_timeout(scenario())


def test_a_cancelled_traced_batch_ends_every_started_items_span() -> None:
    """A cancellation reaching a traced batch ends each started item's span, with no status set."""

    async def scenario() -> None:
        """Time out a traced batch whose opens hang, then read the spans."""
        adapter = FakeAdapter(hang_from_open=1)
        llm, exporter = _traced(adapter)
        with pytest.raises(TimeoutError):
            await time_out_when(
                llm.bind().generate_many(["a", "b"]),
                lambda: adapter.bound_adapters[0].open_count == 2,
            )
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(span.status.status_code == StatusCode.UNSET for span in spans)

    run_with_timeout(scenario())


def test_retry_surfaces_as_a_request_failed_span_event() -> None:
    """A recovered transient failure becomes one langchaint.request_failed event on the generation's span."""

    async def scenario() -> None:
        """Recover one generate_one from a transient failure, then read the span event."""
        llm, exporter = _traced(FakeAdapter(scripted_requests=[TransientError("boom")]))
        generation = await llm.bind().generate_one("hi")
        assert generation.request_count == 2
        (span,) = exporter.get_finished_spans()
        (event,) = span.events
        assert event.name == "langchaint.request_failed"
        assert event.attributes is not None
        assert event.attributes["error_text"] == "boom"

    run_with_timeout(scenario())


def test_generate_many_emits_one_chat_span_per_item_and_none_for_the_batch() -> None:
    """generate_many emits one chat span per item."""

    async def scenario() -> None:
        """Serialize a three-item batch whose first item is Refusal, then inspect the spans."""
        adapter = FakeAdapter(echo=True, scripted_requests=[billed(REFUSAL)])
        llm, exporter = _traced(adapter, max_concurrent_requests=1)
        results = await llm.bind().generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
            [UserMessage(content="c")],
        ])
        first, *rest = results
        assert isinstance(first, GenerationError)
        assert all(result.kind == "without_tool_calls" for result in rest)
        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        assert all(span.kind == SpanKind.CLIENT for span in spans)
        assert all(_attribute(span, "gen_ai.operation.name") == "chat" for span in spans)
        assert all(_attribute(span, "langchaint.cost_in_usd") is not None for span in spans)
        # max_concurrent_requests=1 serializes the batch, so the refused item is the first span to end.
        refused, *succeeded = spans
        assert refused.status.status_code == StatusCode.ERROR
        assert refused.status.description is None
        assert _attribute(refused, "error.type") == "refusal_error"
        assert all(span.status.status_code == StatusCode.OK for span in succeeded)

    run_with_timeout(scenario())


def test_generate_many_maps_and_captures_each_item_from_its_own_outcome() -> None:
    """Each item's span carries that item's mapped outcome and generation_input, never the batch's."""

    async def scenario() -> None:
        """Run a serialized two-item batch under a recording mapper and capture."""
        mapped_outputs: list[object] = []

        def _mapper(outcome: GenerationOutcome[object] | AbandonedStreamRecord) -> SpanAttributes:
            """Record the mapped outcome and emit it as an attribute."""
            output = outcome.output if outcome.kind == "without_tool_calls" else None
            mapped_outputs.append(output)
            return {"custom.mapped_output": str(output)}

        llm, exporter = _traced(
            FakeAdapter(echo=True),
            capture_message_content=True,
            attribute_mapper=_mapper,
            max_concurrent_requests=1,
        )
        results = await llm.bind().generate_many(["a", "b"])
        assert [result.output for result in results if result.kind == "without_tool_calls"] == [
            "a",
            "b",
        ]
        assert mapped_outputs == ["a", "b"]
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        for span, content in zip(spans, ["a", "b"], strict=True):
            assert _attribute(span, "custom.mapped_output") == content
            assert _json_attribute(span, "gen_ai.input.messages") == [
                {"role": "user", "parts": [{"type": "text", "content": content}]}
            ]
            assert _json_attribute(span, "gen_ai.output.messages") == [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": content}],
                    "finish_reason": "stop",
                }
            ]

    run_with_timeout(scenario())


def test_generate_many_records_traces_generated_items_and_skips_reused_items(
    tmp_path: pathlib.Path,
) -> None:
    """generate_many_records opens chat spans only for items that send requests."""

    async def scenario() -> None:
        """Persist two inputs, reorder them around one new input, and inspect the span count."""
        adapter = FakeAdapter(echo=True)
        llm, exporter = _traced(adapter)
        bound = llm.bind()
        resume_path = tmp_path / "records.json"
        first = await bound.generate_many_records(
            ["a", "b"],
            resume_path=resume_path,
            input_ids=["input-a", "input-b"],
        )
        assert all(record.kind == "without_tool_calls" for record in first)
        assert len(exporter.get_finished_spans()) == 2

        resumed = await bound.generate_many_records(
            ["b", "c", "a"],
            resume_path=resume_path,
            input_ids=["input-b", "input-c", "input-a"],
        )
        assert all(record.kind == "without_tool_calls" for record in resumed)
        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        assert all(span.kind == SpanKind.CLIENT for span in spans)
        assert adapter.bound_adapters[0].open_count == 3

    run_with_timeout(scenario())


def test_stream_exhausted_then_final_emits_one_span_with_time_to_first_chunk() -> None:
    """A drained stream ends exactly one span carrying time_to_first_chunk.

    A second final() returns the same Generation and ends no second span.
    """

    async def scenario() -> None:
        """Iterate the stream fully, call final() twice, and inspect the single finished span."""
        llm, exporter = _traced(FakeAdapter())
        async with llm.bind().stream_one("hi") as stream:
            texts = [item async for item in stream if isinstance(item, str)]
            generation = await stream.final()
            assert await stream.final() is generation
        assert "".join(texts) == "ok"
        assert generation.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.name == "chat fake-model"
        assert span.kind == SpanKind.CLIENT
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        assert span.attributes["gen_ai.operation.name"] == "chat"
        time_to_first_chunk = span.attributes["gen_ai.response.time_to_first_chunk"]
        assert isinstance(time_to_first_chunk, float)
        assert time_to_first_chunk >= 0.0
        assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)

    run_with_timeout(scenario())


@pytest.mark.parametrize("block_exit", ["break", "another_inputs_generation_error"])
def test_a_stream_block_left_after_its_first_item_reports_the_stream_input_with_status_unset(
    block_exit: Literal["break", "another_inputs_generation_error"],
) -> None:
    """Leaving the block early is no failure of the stream's input, which still reports its first chunk.

    A GenerationError from another input that escapes the block belongs to the application.
    The stream span reports its own input, never the other input's status or error.type.
    """

    async def scenario() -> None:
        """Pull one item, leave the block as `block_exit` names, then read the stream span."""
        llm, exporter = _traced(FakeAdapter())
        other_llm = LLM(
            FakeAdapter(rejected_messages=[RejectedMessages(error_text="misconfigured")]),
            shared_backoff=fast_shared_backoff(),
        )

        async def leave_after_the_first_item() -> None:
            """Pull one item, then leave the block.

            Raises:
                GenerationError: `block_exit` runs the other input, whose request params are invalid.
            """
            async with llm.bind().stream_one("hi") as stream:
                _ = await anext(stream)
                if block_exit == "another_inputs_generation_error":
                    await other_llm.bind().generate_one("hi")

        if block_exit == "another_inputs_generation_error":
            with pytest.raises(GenerationError):
                await leave_after_the_first_item()
        else:
            await leave_after_the_first_item()
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET
        assert span.attributes is not None
        assert "error.type" not in span.attributes
        assert span.attributes["langchaint.request_count"] == 1
        assert "gen_ai.response.time_to_first_chunk" in span.attributes

    run_with_timeout(scenario())


def test_stream_entered_but_never_iterated_emits_a_span() -> None:
    """Entering opens a request, so an entered handle emits a span even with no item pulled.

    The request is billed whether or not the caller reads it. A silent span would hide it.
    """

    async def scenario() -> None:
        """Enter and leave the context without driving the stream."""
        llm, exporter = _traced(FakeAdapter())
        async with llm.bind().stream_one("hi"):
            pass
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET
        assert span.attributes is not None
        assert "gen_ai.response.time_to_first_chunk" not in span.attributes

    run_with_timeout(scenario())


def test_stream_never_entered_emits_no_span() -> None:
    """stream_one does no I/O, so a handle abandoned without entering emits no span."""

    async def scenario() -> None:
        """Build a handle and drop it."""
        llm, exporter = _traced(FakeAdapter())
        _handle = llm.bind().stream_one("hi")
        assert exporter.get_finished_spans() == ()

    run_with_timeout(scenario())


def test_stream_failing_mid_iteration_ends_its_span_like_any_other_generation_error() -> None:
    """A stream failure records GenerationError and langchaint.request_failed."""

    async def _drain(llm: LLM) -> None:
        """Iterate the mid-failing stream to its raise inside an async with block."""
        async with llm.bind().stream_one("hi") as stream:
            async for _item in stream:
                pass

    async def scenario() -> None:
        """Iterate a mid-failing stream and confirm the error span."""
        llm, exporter = _traced(
            FakeAdapter(stream=_MidFailStream(), request_failure=transient(pauses_quota=False))
        )
        with pytest.raises(GenerationError):
            await _drain(llm)
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "retry_unavailable_error"
        assert [event.name for event in span.events] == ["langchaint.request_failed"]

    run_with_timeout(scenario())


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
    """Request attributes describe text, structured, and streaming inputs."""

    async def scenario() -> None:
        """Run the selected input and inspect its request attributes."""
        llm, exporter = _traced(FakeAdapter(echo=True))
        bound = llm.bind(
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

    run_with_timeout(scenario())


@pytest.mark.parametrize("path", ["generate", "stream"])
def test_span_parsing_reads_every_convention_attribute_a_chat_span_writes(
    path: _GenerationPath,
) -> None:
    """`parse_otel` reads each key of a captured chat span into a field, except the langchaint keys.

    The usage counters read back unchanged.
    """

    async def scenario() -> None:
        """Run one fully configured input and parse its span's exported attributes."""
        llm, exporter = _traced(FakeAdapter(echo=True), capture_message_content=True)
        bound = llm.bind(
            system_prompt="be brief",
            tools=[_echo_tool()],
            max_completion_tokens=123,
            reasoning_level="high",
            temperature=0.25,
        )
        _ = await _generate_through(path, bound)
        (span,) = exporter.get_finished_spans()
        exported = TypeAdapter(dict[str, JsonValue]).validate_json(
            json.dumps(dict(span.attributes or {}))
        )
        parsed = parse_otel(exported)
        assert parsed.unused_attributes.keys() == LANGCHAINT_KEYS
        assert parsed.usage_input_tokens == USAGE.input_tokens_total
        assert parsed.usage_output_tokens == USAGE.output_tokens
        assert parsed.usage_reasoning_output_tokens == USAGE.output_tokens_reasoning
        assert parsed.usage_cache_read_input_tokens == USAGE.input_tokens_cache_read
        assert parsed.usage_cache_write_input_tokens == USAGE.input_tokens_cache_write

    run_with_timeout(scenario())


def test_mapper_not_invoked_on_a_non_recording_span() -> None:
    """A custom attribute_mapper never fires when the tracer's spans are non-recording."""

    async def scenario() -> None:
        """Generate under a no-op tracer and assert the mapper never ran."""
        calls: list[int] = []

        def _mapper(_outcome: GenerationOutcome[object] | AbandonedStreamRecord) -> SpanAttributes:
            """Count each invocation."""
            calls.append(1)
            return {}

        tracer_provider = trace.NoOpTracerProvider()
        llm = LLM(
            FakeAdapter(),
            observer=OtelObserver(
                attribute_mapper=_mapper,
                tracer_provider=tracer_provider,
                capture_message_content=False,
            ),
        )
        generation = await llm.bind().generate_one("hi")
        assert generation.output == "ok"
        assert calls == []

    run_with_timeout(scenario())


def _raising_mapper(_outcome: GenerationOutcome[object] | AbandonedStreamRecord) -> SpanAttributes:
    """Raise to simulate a buggy user mapper.

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError("mapper bug")


@pytest.mark.parametrize("path", ["generate", "stream"])
def test_raising_mapper_is_caught_and_the_generation_survives(
    path: _GenerationPath, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising mapper is logged, the input still returns its Generation, and the span still ends."""

    async def scenario() -> None:
        """Run the input under a mapper that raises and confirm the generation and span survive."""
        llm, exporter = _traced(FakeAdapter(), attribute_mapper=_raising_mapper)
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            response = await _generate_through(path, llm.bind())
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK
        assert any("mapper" in record.message for record in caplog.records)

    run_with_timeout(scenario())


def _covariance_pin(
    mapper: AttributeMapper, generation: GenerationWithoutToolCalls[_Answer]
) -> SpanAttributes:
    """Pin that a GenerationWithoutToolCalls[_Answer] satisfies the mapper's GenerationOutcome[object] parameter.

    pyrefly checks the OutputT covariance at the call below.
    """
    return mapper(generation)


def test_a_custom_mapper_and_extra_attributes_reach_every_chat_span_across_bind() -> None:
    """A custom mapper replaces the default outcome attributes on generate, stream, and batch item spans.

    A replacement binding keeps the observer.
    A mapper key of the same name as an extra wins when the outcome attributes are set.
    The observer-owned gen_ai.operation.name wins at span start.
    """

    async def scenario() -> None:
        """Generate, stream, and batch on a replacement binding, then read every span."""
        mapped_models: list[str] = []

        def _mapper(outcome: GenerationOutcome[object] | AbandonedStreamRecord) -> SpanAttributes:
            """Record the outcome and emit one attribute from it and one colliding with an extra."""
            mapped_models.append(outcome.model)
            return {"custom.model": outcome.model, "shared.key": "mapped"}

        llm, exporter = _traced(
            FakeAdapter(echo=True),
            attribute_mapper=_mapper,
            extra_attributes={
                "custom.agent": "agent_a",
                "shared.key": "extra",
                "gen_ai.operation.name": "not-the-operation",
            },
        )
        replacement_bound = llm.bind(system_prompt="s").bind(system_prompt="s2")
        for path in ("generate", "stream"):
            await _generate_through(path, replacement_bound)
        await replacement_bound.generate_many(["a", "b"])
        spans = exporter.get_finished_spans()
        # One generate span, one stream span, and one span per batch item.
        assert len(spans) == 4
        assert mapped_models == ["fake-model"] * 4
        # The observer sets the request attributes at span start, outside the mapper's control.
        assert spans[0].attributes == {
            "gen_ai.operation.name": "chat",
            "gen_ai.output.type": "text",
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "custom.agent": "agent_a",
            "custom.model": "fake-model",
            "shared.key": "mapped",
        }
        for span in spans:
            assert _attribute(span, "custom.agent") == "agent_a"
            assert _attribute(span, "custom.model") == "fake-model"
            assert _attribute(span, "shared.key") == "mapped"
            assert _attribute(span, "gen_ai.operation.name") == "chat"

    run_with_timeout(scenario())


class _EchoToolArgs(BaseModel):
    """Arguments of the echo tool the dispatch span tests dispatch."""

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


async def _erring_tool_function(args: _EchoToolArgs) -> ToolReturnExplicit[None]:
    """Return a function-authored failure: a handled outcome whose ToolMessage carries is_error True."""
    return ToolReturnExplicit(content=f"cannot process {args.text}", is_error=True)


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
def test_tool_manager_dispatch_emits_one_span_classified_by_its_outcome(
    build_tool: Callable[[], PydanticTool[_EchoToolArgs]],
    tool_call: ToolCall,
    expected_outcome_type: type[DispatchOutcome],
    expected_error_type: str | None,
) -> None:
    """Each dispatch emits one execute_tool span classified by its outcome.

    Capture is off, so the exact attribute set shows that no arguments or result are recorded.
    """
    expected_attributes: dict[str, object] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool_call.name,
        "gen_ai.tool.call.id": "call1",
    }
    if expected_error_type is not None:
        expected_attributes["error.type"] = expected_error_type

    async def scenario() -> None:
        """Dispatch the call and inspect the single finished span."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        tool_manager = ToolManager(
            [build_tool()],
            observer=OtelObserver(tracer_provider=tracer_provider, capture_message_content=False),
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

    run_with_timeout(scenario())


def test_tool_manager_function_exception_marks_the_span_error_and_propagates() -> None:
    """A tool-function defect records the exception, sets error status, and propagates."""

    async def scenario() -> None:
        """Dispatch a call whose function raises and inspect the error span."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        tool_manager = ToolManager(
            [_raising_tool()],
            observer=OtelObserver(tracer_provider=tracer_provider, capture_message_content=False),
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

    run_with_timeout(scenario())


def test_tool_manager_dispatch_many_spans_every_call() -> None:
    """dispatch_many observes each call through dispatch: two calls yield two execute_tool spans, outcomes ordered."""

    async def scenario() -> None:
        """Dispatch two calls concurrently and read both spans."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        tool_manager = ToolManager(
            [_echo_tool()],
            observer=OtelObserver(tracer_provider=tracer_provider, capture_message_content=False),
        )
        outcomes = await tool_manager.dispatch_many([
            ToolCall(id="call1", name="echo", args_json='{"text": "a"}'),
            ToolCall(id="call2", name="missing", args_json="{}"),
        ])
        assert outcomes[0].kind == "handled"
        assert outcomes[1].kind == "unknown_tool"
        spans = exporter.get_finished_spans()
        assert sorted(span.name for span in spans) == ["execute_tool echo", "execute_tool missing"]
        call_ids = {
            span.attributes["gen_ai.tool.call.id"] for span in spans if span.attributes is not None
        }
        assert call_ids == {"call1", "call2"}

    run_with_timeout(scenario())


def test_tool_manager_span_is_current_inside_the_tool_function() -> None:
    """The dispatch span is current while the function runs: a span the function starts nests under it."""

    async def scenario() -> None:
        """Dispatch a tool whose function opens its own span and assert the parentage."""
        tracer_provider, exporter = _in_memory_tracer_provider()

        async def nesting_tool_function(args: _EchoToolArgs) -> str:
            """Open one inner span on the same tracer provider and return the text."""
            with tracer_provider.get_tracer("test").start_as_current_span("inner"):
                return args.text

        tool = PydanticTool(
            name="nesting",
            description="Opens an inner span",
            args_model=_EchoToolArgs,
            function=nesting_tool_function,
        )
        tool_manager = ToolManager(
            [tool],
            observer=OtelObserver(tracer_provider=tracer_provider, capture_message_content=False),
        )
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

    run_with_timeout(scenario())


def test_bind_gives_the_observer_to_tool_managers_it_builds_from_sequences() -> None:
    """`LLM.bind` and `BoundLLM.bind` pass the observer to a `ToolManager` built from a tool sequence.

    A `ToolManager` the application built keeps its own observer, here none.
    """

    async def scenario() -> None:
        """Dispatch through two managers bind built and one the application built, then count spans."""
        llm, exporter = _traced(
            FakeAdapter(),
            capture_message_content=True,
            extra_attributes={"gen_ai.agent.name": "agent_a"},
        )
        bound = llm.bind(tools=[_echo_tool()])
        replacement_bound = bound.bind(tools=[_echo_tool()])
        unobserved_bound = bound.bind(tools=ToolManager([_echo_tool()]))
        await bound.tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json='{"text": "a"}')
        )
        await replacement_bound.tool_manager.dispatch(
            ToolCall(id="call2", name="echo", args_json='{"text": "b"}')
        )
        await unobserved_bound.tool_manager.dispatch(
            ToolCall(id="call3", name="echo", args_json='{"text": "c"}')
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        for span in spans:
            assert span.attributes is not None
            assert span.attributes["gen_ai.agent.name"] == "agent_a"
            assert "gen_ai.tool.call.arguments" in span.attributes
            assert "gen_ai.tool.call.result" in span.attributes

    run_with_timeout(scenario())


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
        tracer_provider, exporter = _in_memory_tracer_provider()
        tool_manager = ToolManager(
            [_echo_tool()],
            observer=OtelObserver(
                tracer_provider=tracer_provider,
                extra_attributes={"gen_ai.agent.name": "agent_a", colliding_key: "spoofed"},
                capture_message_content=False,
            ),
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "a"}'))
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.agent.name"] == "agent_a"
        assert span.attributes[colliding_key] == expected_value

    run_with_timeout(scenario())


def test_vendored_payload_schemas_match_the_schema_mapping() -> None:
    """Compare vendored schema filenames with the configured mapping."""
    vendored = {path.name for path in SEMCONV_GENAI_DIR.glob("gen-ai-*.json")}
    assert vendored, "no vendored schemas found, so this assertion would pass vacuously"
    assert vendored == set(refresh_semconv_genai.ATTRIBUTE_SCHEMA_FILES.values())


def test_refresh_accepts_a_structured_attribute_name_array(tmp_path: pathlib.Path) -> None:
    """Accept generated structured attribute names as a JSON array of strings."""
    generated_path = tmp_path / refresh_semconv_genai.RUNTIME_STRUCTURED_ATTRIBUTES_FILE
    _ = generated_path.write_text('["gen_ai.input.messages"]')
    refresh_semconv_genai._validate_json_file(generated_path, expected_shape="string_array")


@pytest.mark.parametrize("content", ['{"gen_ai.input.messages": true}', "[1]"])
def test_refresh_rejects_a_malformed_structured_attribute_name_array(
    content: str, tmp_path: pathlib.Path
) -> None:
    """Reject generated structured attribute names outside a JSON array of strings."""
    generated_path = tmp_path / refresh_semconv_genai.RUNTIME_STRUCTURED_ATTRIBUTES_FILE
    _ = generated_path.write_text(content)
    with pytest.raises(TypeError, match="must contain a JSON array of strings"):
        refresh_semconv_genai._validate_json_file(generated_path, expected_shape="string_array")


def _stage_refresh_directories(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Return staged, destination, and runtime directories with matching committed data."""
    staged = tmp_path / "staged"
    destination = tmp_path / "destination"
    runtime = tmp_path / "runtime"
    for directory in (staged, destination, runtime):
        directory.mkdir()
    _ = (staged / refresh_semconv_genai.CHAT_SPAN_ATTRIBUTES_FILE).write_text("{}")
    _ = (destination / refresh_semconv_genai.CHAT_SPAN_ATTRIBUTES_FILE).write_text("{}")
    _ = (staged / refresh_semconv_genai.RUNTIME_STRUCTURED_ATTRIBUTES_FILE).write_text("[]")
    _ = (runtime / refresh_semconv_genai.RUNTIME_STRUCTURED_ATTRIBUTES_FILE).write_text("[]")
    _ = (staged / "SOURCE.md").write_text("Resolved commit SHA: `new`.\n")
    _ = (destination / "SOURCE.md").write_text("Resolved commit SHA: `old`.\n")
    monkeypatch.setattr(refresh_semconv_genai, "DESTINATION", destination)
    monkeypatch.setattr(refresh_semconv_genai, "RUNTIME_DESTINATION", runtime)
    monkeypatch.setattr(refresh_semconv_genai, "SOURCE_DOC", destination / "SOURCE.md")
    return staged, destination, runtime


def test_refresh_leaves_source_doc_unchanged_when_data_matches(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the recorded SHA when every staged data file matches its committed copy."""
    staged, destination, _ = _stage_refresh_directories(tmp_path, monkeypatch)
    refresh_semconv_genai._replace_committed_files(staged)
    assert (destination / "SOURCE.md").read_text() == "Resolved commit SHA: `old`.\n"


def test_refresh_rewrites_source_doc_when_runtime_data_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write the runtime data file and SOURCE.md when only the runtime data differs."""
    staged, destination, runtime = _stage_refresh_directories(tmp_path, monkeypatch)
    runtime_file = refresh_semconv_genai.RUNTIME_STRUCTURED_ATTRIBUTES_FILE
    _ = (staged / runtime_file).write_text('["gen_ai.input.messages"]')
    refresh_semconv_genai._replace_committed_files(staged)
    assert (runtime / runtime_file).read_text() == '["gen_ai.input.messages"]'
    assert (destination / "SOURCE.md").read_text() == "Resolved commit SHA: `new`.\n"


def test_refresh_removes_an_obsolete_file_and_rewrites_source_doc(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat a committed obsolete file as a data change."""
    staged, destination, _ = _stage_refresh_directories(tmp_path, monkeypatch)
    obsolete_path = destination / "provider-name-values.json"
    _ = obsolete_path.write_text("{}")
    refresh_semconv_genai._replace_committed_files(staged)
    assert not obsolete_path.exists()
    assert (destination / "SOURCE.md").read_text() == "Resolved commit SHA: `new`.\n"


def test_capture_on_records_all_four_content_attributes_in_convention_shape() -> None:
    """capture_message_content True records the system prompt, tools, GenerationInput, and assistant message.

    Capture carries over to a replacement binding.
    """

    async def scenario() -> None:
        """Generate over a Sequence[Message] carrying every message role and inspect the shapes."""
        llm, exporter = _traced(FakeAdapter(), capture_message_content=True)
        bound = llm.bind(system_prompt="replaced").bind(
            system_prompt="be brief",
            tools=ToolManager([_echo_tool()]),
        )
        await bound.generate_one([
            UserMessage(content="look it up"),
            AssistantMessage(
                parts=(ToolCall(id="call1", name="echo", args_json='{"text": "x"}'),)
            ),
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

    run_with_timeout(scenario())


_MULTIMODAL_USER_MESSAGE = UserMessage(
    content=(
        TextPart(text="what is this"),
        ImagePart(data=b"\x89PNGsecret", media_type="image/png"),
        ImageUrlPart(url="https://example.com/image.png", media_type="image/png"),
        ImageUrlPart(url="https://example.com/unknown"),
        AudioPart(data=b"WAVsecret", media_type="audio/wav"),
    )
)
"""One user message holding every ContentPart variant."""


def _drop_binary(
    _name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart | None:
    """Keep every part except inline bytes, the filter the OtelObserver docstring shows."""
    return None if part.kind in ("image", "audio") else part


def _captured_user_parts(exporter: InMemorySpanExporter) -> list[object]:
    """Read the parts of the one recorded input message."""
    captured_messages = _captured(exporter, "gen_ai.input.messages")
    assert isinstance(captured_messages, list)
    (captured_message,) = captured_messages
    assert isinstance(captured_message, dict)
    captured_parts: object = captured_message["parts"]
    assert isinstance(captured_parts, list)
    return captured_parts


def _parts_of_type(parts: list[object], part_type: str) -> list[object]:
    """Select the recorded parts whose type field is part_type."""
    return [part for part in parts if isinstance(part, dict) and part.get("type") == part_type]


def test_true_records_image_and_audio_bytes_as_blob_parts_that_round_trip() -> None:
    """capture_message_content=True records ImagePart and AudioPart as convention BlobPart objects.

    ImageUrlPart is a convention UriPart.
    span_parsing converts the recorded message back to the original message.
    """

    async def scenario() -> None:
        """Generate over the multimodal message and read the recorded parts back."""
        llm, exporter = _traced(FakeAdapter(), capture_message_content=True)
        await llm.bind().generate_one([_MULTIMODAL_USER_MESSAGE])
        captured_parts = _captured_user_parts(exporter)
        schema_definitions = payload_schema("gen-ai-input-messages.json")["$defs"]
        blob_parts = _parts_of_type(captured_parts, "blob")
        assert len(blob_parts) == 2
        blob_part_validator = jsonschema.Draft202012Validator({
            "$defs": schema_definitions,
            "$ref": "#/$defs/BlobPart",
        })
        for blob_part in blob_parts:
            blob_part_validator.validate(blob_part)
        uri_parts = _parts_of_type(captured_parts, "uri")
        assert len(uri_parts) == 2
        image_uri_part_validator = jsonschema.Draft202012Validator({
            "$defs": schema_definitions,
            "allOf": [
                {"$ref": "#/$defs/UriPart"},
                {"properties": {"modality": {"const": "image"}}},
            ],
        })
        for uri_part in uri_parts:
            image_uri_part_validator.validate(uri_part)
        (span,) = exporter.get_finished_spans()
        input_messages_json = _attribute(span, "gen_ai.input.messages")
        assert isinstance(input_messages_json, str)
        parsed = parse_otel({
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": input_messages_json,
        })
        assert generation_input_from_otel(parsed) == (_MULTIMODAL_USER_MESSAGE,)

    run_with_timeout(scenario())


def test_a_filter_returning_none_drops_image_and_audio_parts() -> None:
    """_drop_binary leaves the text and uri parts and records no blob part."""

    async def scenario() -> None:
        """Generate over the multimodal message under _drop_binary and read the recorded parts back."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        content_filter: ContentFilter = _drop_binary
        llm = LLM(
            FakeAdapter(),
            observer=OtelObserver(
                tracer_provider=tracer_provider,
                capture_message_content=True,
                content_filter=content_filter,
            ),
        )
        await llm.bind().generate_one([_MULTIMODAL_USER_MESSAGE])
        captured_parts = _captured_user_parts(exporter)
        assert [part.get("type") for part in captured_parts if isinstance(part, dict)] == [
            "text",
            "uri",
            "uri",
        ]

    run_with_timeout(scenario())


_MARKER = "SECRET-MARKER"
"""The text a scrubbing filter must remove from every content attribute."""


def _scrub_marker(
    _name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart | None:
    """Replace _MARKER in every text-carrying part and keep the rest unchanged."""
    match part.kind:
        case "text":
            return TextPart(text=part.text.replace(_MARKER, "[redacted]"))
        case "reasoning":
            if part.text is None:
                return part
            return part.model_copy(update={"text": part.text.replace(_MARKER, "[redacted]")})
        case "tool_call":
            return part.model_copy(
                update={"args_json": part.args_json.replace(_MARKER, "[redacted]")}
            )
        case _:
            return part


def test_a_scrubbing_filter_reaches_every_content_attribute(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_MARKER placed in every part position is absent from every recorded content attribute."""

    async def scenario() -> None:
        """Generate and dispatch under _scrub_marker, then scan every content attribute on both spans."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        scripted_assistant_message = AssistantMessage(
            parts=(
                ReasoningPart(raw={"signature": "opaque"}, text=f"thinking {_MARKER}"),
                TextPart(text=f"answer {_MARKER}"),
            )
        )
        observer = OtelObserver(
            tracer_provider=tracer_provider,
            capture_message_content=True,
            content_filter=_scrub_marker,
        )
        llm = LLM(
            FakeAdapter(
                scripted_requests=[
                    ScriptedResponse(
                        outcome=UsableResponse(
                            output="answer",
                            assistant_message=scripted_assistant_message,
                            stop_reason="stop",
                        ),
                        usage=USAGE,
                    )
                ]
            ),
            observer=observer,
        )
        bound = llm.bind(system_prompt=f"rules {_MARKER}", tools=ToolManager([_echo_tool()]))
        tool_call = ToolCall(id="call1", name="echo", args_json=f'{{"text": "{_MARKER}"}}')
        await bound.generate_one([
            UserMessage(content=f"question {_MARKER}"),
            AssistantMessage(parts=(tool_call,)),
            ToolMessage(tool_call_id="call1", content=f"echoed {_MARKER}"),
        ])
        tool_manager = ToolManager([_echo_tool()], observer=observer)
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            await tool_manager.dispatch(tool_call)
        assert "content capture raised" not in caplog.text
        recorded_content_keys: set[str] = set()
        for span in exporter.get_finished_spans():
            assert span.attributes is not None
            for key, value in span.attributes.items():
                if key in refresh_semconv_genai.ATTRIBUTE_SCHEMA_FILES:
                    recorded_content_keys.add(key)
                    assert _MARKER not in str(value), f"{span.name}: {key} leaks {_MARKER}"
        assert recorded_content_keys == set(refresh_semconv_genai.ATTRIBUTE_SCHEMA_FILES)

    run_with_timeout(scenario())


def _drop_system_instructions(
    name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart | None:
    """Omit every part of the system prompt and keep every other part."""
    return None if name == "gen_ai.system_instructions" else part


@pytest.mark.parametrize(
    ("system_prompt", "content_filter", "expected_instructions"),
    [
        (None, _record_every_part, None),
        (
            [TextPart(text="be brief"), TextPart(text="cite sources")],
            _record_every_part,
            [{"type": "text", "content": "be brief"}, {"type": "text", "content": "cite sources"}],
        ),
        ("rules", _drop_system_instructions, None),
    ],
    ids=["absent", "one_element_per_part", "every_part_filtered_away"],
)
def test_system_instructions_render_one_element_per_part_and_omit_an_empty_array(
    system_prompt: str | list[TextPart] | None,
    content_filter: ContentFilter,
    expected_instructions: object,
) -> None:
    """A system prompt with no parts left to record omits its key, while the input messages stay."""
    binding = LLM(FakeAdapter()).bind(system_prompt=system_prompt).binding
    attributes = _input_content_attributes(
        binding, [UserMessage(content="hi")], content_filter=content_filter
    )
    instructions = attributes.get("gen_ai.system_instructions")
    assert (None if instructions is None else json.loads(str(instructions))) == (
        expected_instructions
    )
    assert "gen_ai.tool.definitions" not in attributes
    assert "gen_ai.input.messages" in attributes


def _raise_on_input_messages(
    name: str, part: ContentPart | AssistantPart
) -> ContentPart | AssistantPart:
    """Raise on every input-message part and keep every other part.

    Raises:
        RuntimeError: `name` is gen_ai.input.messages.
    """
    if name == "gen_ai.input.messages":
        raise RuntimeError("filter defect")
    return part


@pytest.mark.parametrize(
    ("content_filter", "build_tool", "logged_error_text"),
    [
        (_raise_on_input_messages, _echo_tool, "filter defect"),
        (_record_every_part, _unserializable_schema_tool, "is not JSON serializable"),
    ],
    ids=["raising_filter", "unserializable_tool_schema"],
)
@pytest.mark.parametrize("path", ["generate", "stream"])
def test_a_failure_building_input_content_omits_the_three_input_attributes(
    path: _GenerationPath,
    content_filter: ContentFilter,
    build_tool: Callable[[], PydanticTool[_EchoToolArgs] | JSONSchemaTool],
    logged_error_text: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The three input keys build as one dict, so one failure drops all three and the input proceeds.

    The failure is logged with its exception, and the output content, usage, and stream timing still reach the span.
    """

    async def scenario() -> None:
        """Run an input whose content cannot be built, then read the span and the log."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        llm = LLM(
            FakeAdapter(),
            observer=OtelObserver(
                tracer_provider=tracer_provider,
                capture_message_content=True,
                content_filter=content_filter,
            ),
        )
        bound = llm.bind(system_prompt="rules", tools=ToolManager([build_tool()]))
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            result = await _generate_through(path, bound)
        assert result.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert not {
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "gen_ai.input.messages",
        } & set(span.attributes)
        assert _json_attribute(span, "gen_ai.output.messages") == [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "ok"}],
                "finish_reason": "stop",
            }
        ]
        assert "gen_ai.usage.output_tokens" in span.attributes
        assert ("gen_ai.response.time_to_first_chunk" in span.attributes) is (path == "stream")
        assert "content capture raised" in caplog.text
        assert logged_error_text in caplog.text

    run_with_timeout(scenario())


def test_a_filter_returning_a_part_of_another_kind_omits_the_attribute(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A TextPart returned for the ToolCall cannot take its place, so the arguments key is omitted."""

    def text_for_tool_call(
        _name: str, part: ContentPart | AssistantPart
    ) -> ContentPart | AssistantPart:
        return TextPart(text="not a tool call") if part.kind == "tool_call" else part

    async def scenario() -> None:
        """Dispatch under the mismatching filter, then read the span and the log."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        tool_manager = ToolManager(
            [_echo_tool()],
            observer=OtelObserver(
                tracer_provider=tracer_provider,
                capture_message_content=True,
                content_filter=text_for_tool_call,
            ),
        )
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            outcome = await tool_manager.dispatch(
                ToolCall(id="call1", name="echo", args_json='{"text": "hi"}')
            )
        assert isinstance(outcome, DispatchHandled)
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.tool.call.arguments" not in span.attributes
        assert "gen_ai.tool.call.result" in span.attributes
        assert "content capture raised" in caplog.text
        assert "returned TextPart for ToolCall" in caplog.text

    run_with_timeout(scenario())


@pytest.mark.parametrize(
    ("parts", "stop_reason", "expected_parts", "expected_finish_reason"),
    [
        ((ReasoningPart(raw={"signature": "opaque"}),), "stop", [], "stop"),
        (
            (ReasoningPart(raw={"signature": "opaque"}, text=""), TextPart(text="")),
            "stop",
            [],
            "stop",
        ),
        (
            (
                ReasoningPart(raw={"signature": "opaque"}, text="thought it over"),
                TextPart(text="answer"),
            ),
            "stop",
            [
                {"type": "reasoning", "content": "thought it over"},
                {"type": "text", "content": "answer"},
            ],
            "stop",
        ),
        (
            REJECTED_ASSISTANT_MESSAGE.parts,
            None,
            [{"type": "text", "content": "what the rejected 200 carried"}],
            "error",
        ),
    ],
    ids=[
        "text_free_reasoning_alone",
        "empty_text_on_every_part",
        "reasoning_text",
        "no_stop_reason",
    ],
)
def test_output_messages_render_readable_text_and_never_the_reasoning_payload(
    parts: tuple[AssistantPart, ...],
    stop_reason: StopReason | None,
    expected_parts: list[object],
    expected_finish_reason: str,
) -> None:
    """An assistant message renders its readable text in order, ReasoningPart.raw never, and "error" for no stop reason.

    An assistant message without readable text still renders its message, with an empty parts array.
    """
    attributes = _output_content_attributes(parts, stop_reason, _record_every_part)
    output_messages = str(attributes["gen_ai.output.messages"])
    assert json.loads(output_messages) == [
        {"role": "assistant", "parts": expected_parts, "finish_reason": expected_finish_reason}
    ]
    assert "opaque" not in output_messages


@pytest.mark.parametrize(
    ("tool_call", "expected_arguments"),
    [
        (
            ToolCall(id="call1", name="echo", args_json='{"text":"hi",   "n":1}'),
            {"text": "hi", "n": 1},
        ),
        (ToolCall(id="call1", name="missing", args_json="{}"), {}),
        (ToolCall(id="call1", name="echo", args_json="not json at all"), "not json at all"),
    ],
    ids=["handled", "unknown_tool", "invalid_tool_args"],
)
def test_tool_span_captures_arguments_and_result_under_capture(
    tool_call: ToolCall, expected_arguments: object
) -> None:
    """A dispatch span records the parsed arguments and the tool_message as a tool_call_response part.

    Argument text that does not parse is recorded unchanged as a JSON string.
    The tool result is recorded on the variants where no tool ran, too.
    """

    async def scenario() -> None:
        """Dispatch one call under capture and read both content keys."""
        tracer_provider, exporter = _in_memory_tracer_provider()
        tool_manager = ToolManager(
            [_echo_tool()],
            observer=OtelObserver(tracer_provider=tracer_provider, capture_message_content=True),
        )
        outcome = await tool_manager.dispatch(tool_call)
        assert isinstance(outcome.tool_message.content, str)
        assert _captured(exporter, "gen_ai.tool.call.arguments") == expected_arguments
        assert _captured(exporter, "gen_ai.tool.call.result") == {
            "type": "tool_call_response",
            "id": "call1",
            "is_error": outcome.tool_message.is_error,
            "response": [{"type": "text", "content": outcome.tool_message.content}],
        }

    run_with_timeout(scenario())


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [
        ('{"text": "x"}', {"text": "x"}),
        ('{"n": 1.5, "big": 1e300}', {"n": 1.5, "big": 1e300}),
        ("[1, 2]", [1, 2]),
        ('"bare"', "bare"),
        ("7", 7),
        ("null", None),
        ("not json at all", "not json at all"),
        ("{oops", "{oops"),
        ('{"n": 1e400}', '{"n": 1e400}'),
        ('{"n": Infinity}', '{"n": Infinity}'),
        ('{"n": -Infinity}', '{"n": -Infinity}'),
        ('{"n": NaN}', '{"n": NaN}'),
    ],
)
def test_tool_call_arguments_parse_any_json_value_and_keep_other_text_unchanged(
    args_json: str, expected: object
) -> None:
    """Parse argument text to the JSON value it holds, and keep text that is not standard JSON as the text.

    A non-finite number is not standard JSON, so it keeps the text rather than parse to a float.
    """
    assert _tool_call_arguments(args_json) == expected


def test_input_tool_calls_nest_parsed_arguments_and_keep_unparseable_text() -> None:
    """Input tool calls record parsed objects and malformed text in a payload the schema accepts."""

    async def scenario() -> None:
        """Generate over an assistant message holding one parseable and one unparseable tool call."""
        llm, exporter = _traced(FakeAdapter(), capture_message_content=True)
        await llm.bind().generate_one([
            AssistantMessage(
                parts=(
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

    run_with_timeout(scenario())


_CONTENT_SENTINEL = "sentinel-string-no-ungated-channel-may-carry"
"""The generated text the content-rule test traces through every reporting channel."""


@pytest.mark.parametrize("capture_message_content", [False, True])
def test_a_failures_assistant_message_reaches_a_span_only_through_the_gated_output_key(
    *, capture_message_content: bool
) -> None:
    """A failed input's assistant message reaches spans only through gated gen_ai.output.messages."""
    assistant_message = AssistantMessage(parts=(TextPart(text=_CONTENT_SENTINEL),))

    async def scenario() -> None:
        """Fail two requests and inspect content channels."""
        adapter = FakeAdapter(
            scripted_requests=[
                TransientError("the first request failed"),
                billed(Refusal(assistant_message=assistant_message)),
            ]
        )
        llm, exporter = _traced(adapter, capture_message_content=capture_message_content)
        with pytest.raises(GenerationError) as raised:
            await llm.bind(max_requests=3).generate_one("hi")

        error = raised.value
        assert error.request_count == 2
        assert _CONTENT_SENTINEL not in error.error_text
        assert _CONTENT_SENTINEL not in str(error)
        assert error.assistant_message == assistant_message
        assert _CONTENT_SENTINEL in str(to_tables(error).requests[-1]["assistant_message_json"])

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

    run_with_timeout(scenario())
