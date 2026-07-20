"""Tracing wrappers driven by fake providers and an in-memory exporter.

The fake providers are the ones test_bound_llm builds; here they feed the tracing wrapper instead of the plain BoundLLM.
TracedToolManager is driven directly with constructed ToolCalls; its spans land in the same exporter.
A locally built TracerProvider with a SimpleSpanProcessor and an InMemorySpanExporter captures every span,
so the assertions read span names, kinds, statuses, attributes, events,
and parentage without any collector or network access.
"""

import asyncio
import inspect
import json
import logging
import pathlib
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import assert_type, override

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai_semconv
from opentelemetry.semconv.attributes import error_attributes as error_semconv
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import BaseModel

import langchaint.tracing
from langchaint import (
    LLM,
    AbortBatchError,
    AssistantMessage,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchUnknownTool,
    GenerationError,
    ImagePart,
    JSONSchemaTool,
    MaxCompletionTokensExceededError,
    Message,
    PydanticTool,
    ReasoningTrace,
    RefusalError,
    Response,
    RetriesExhaustedError,
    StreamItem,
    TextPart,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    TransientError,
    UserMessage,
    to_row,
)
from langchaint.provider import Binding, BoundProvider, ProviderResult
from langchaint.tracing import (
    AttributeMapper,
    SpanAttributes,
    TracedBoundLLM,
    TracedLLM,
    TracedStreamHandle,
    TracedToolManager,
    gen_ai_attributes,
)
from tests.test_bound_llm import (
    _FAKE_RAW_USAGE,
    _USAGE,
    _USAGE_BILLED,
    _FakeBoundProvider,
    _FakeProvider,
    _FakeRawResponse,
    _FakeStream,
    _fast_rate_limiter,
    _RefusingStream,
)


def _in_memory_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """Build a fresh recording tracer whose spans land in an InMemorySpanExporter."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider.get_tracer("test"), exporter


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
            LLM(_FakeProvider(echo=True)), tracer=tracer, capture_message_content=False
        )
        response = await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        assert response.output == "hi"
        (span,) = exporter.get_finished_spans()
        assert span.name == "chat fake-model"
        assert span.kind == SpanKind.CLIENT
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        assert dict(span.attributes) == {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
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
    """A refusal leaf yields an error span carrying the rejected 200's real token counts and cost."""

    async def scenario() -> None:
        """Drive one generate_one whose send refuses, then inspect the error span."""
        provider = _FakeProvider(
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
                )
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(RefusalError):
            await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["gen_ai.response.finish_reasons"] == ("refusal",)
        assert span.attributes["langchaint.cost_in_usd"] == 0.25
        assert span.attributes["gen_ai.usage.output_tokens"] == _USAGE.output_tokens

    asyncio.run(scenario())


def test_generate_one_truncation_span_has_error_status_and_real_tokens() -> None:
    """A truncation leaf yields an error span with the rejected 200's tokens and max_tokens finish."""

    async def scenario() -> None:
        """Drive one generate_one whose send truncates, then inspect the error span."""
        provider = _FakeProvider(
            failures=[
                MaxCompletionTokensExceededError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="max_tokens"
                )
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(MaxCompletionTokensExceededError):
            await traced.bind(automatic_prompt_caching=True).generate_one("hi")
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
        provider = _FakeProvider(failures=[TransientError("e1"), TransientError("e2")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter(max_attempts=2)),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(RetriesExhaustedError):
            await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["gen_ai.usage.input_tokens"] == 0
        assert span.attributes["gen_ai.usage.output_tokens"] == 0
        assert span.attributes["langchaint.cost_in_usd"] == 0.0
        # A retries-exhausted failure has no completed turn, so no finish reason is set.
        assert "gen_ai.response.finish_reasons" not in span.attributes

    asyncio.run(scenario())


def test_generate_one_abort_records_the_exception_and_ends_the_span() -> None:
    """An AbortBatchError sets error status, records the exception, and still ends the span."""

    async def scenario() -> None:
        """Drive one generate_one whose send aborts, then inspect the error span."""
        provider = _FakeProvider(failures=[AbortBatchError("misconfigured")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(AbortBatchError):
            await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        # An abort carries no shared-field attributes; it is recorded as an exception event
        # plus the error.type classification every failing span kind takes.
        assert dict(span.attributes or {}) == {
            "gen_ai.operation.name": "chat",
            "error.type": "AbortBatchError",
        }
        assert [event.name for event in span.events] == ["exception"]

    asyncio.run(scenario())


def test_retry_surfaces_as_an_attempt_failed_span_event() -> None:
    """A recovered transient failure becomes one langchaint.attempt_failed event on the success span."""

    async def scenario() -> None:
        """Recover one generate_one from a transient failure, then read the span event."""
        provider = _FakeProvider(failures=[TransientError("boom")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=False,
        )
        response = await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        assert response.attempts == 2
        (span,) = exporter.get_finished_spans()
        (event,) = span.events
        assert event.name == "langchaint.attempt_failed"
        assert event.attributes is not None
        assert event.attributes["error_text"] == "boom"

    asyncio.run(scenario())


def test_generate_many_emits_one_internal_batch_span() -> None:
    """A mixed batch emits exactly one OK INTERNAL span carrying the item count, no per-item spans."""

    async def scenario() -> None:
        """Serialize a three-item batch whose first item refuses, then inspect the batch span."""
        provider = _FakeProvider(
            echo=True,
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
                )
            ],
        )
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=rate_limiter), tracer=tracer, capture_message_content=False
        )
        results = await traced.bind(automatic_prompt_caching=True).generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
            [UserMessage(content="c")],
        ])
        first, *rest = results
        assert isinstance(first, RefusalError)
        assert all(isinstance(result, Response) for result in rest)
        # generate_many delegates to BoundLLM.generate_many under one aggregate span; no per-item child spans,
        # so a mixed batch emits exactly one span, not one per item.
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK
        # The batch span makes no outbound call of its own, so its kind is INTERNAL, not CLIENT.
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes is not None
        assert span.attributes["langchaint.batch_item_count"] == 3
        # Per-item detail lives in the returned rows, not on the span; the mapper is not invoked here.
        assert "langchaint.cost_in_usd" not in span.attributes

    asyncio.run(scenario())


def test_generate_many_matches_bound_llm_row_shapes() -> None:
    """The same mixed batch through BoundLLM and TracedBoundLLM yields identical row shapes.

    TracedBoundLLM.generate_many delegates to BoundLLM.generate_many, so wrapping must not alter the rows:
    any divergence in row keys or the success/failure pattern would show here.
    """

    async def scenario() -> None:
        """Run one scripted mixed batch through each generate_many and compare the rows."""
        conversations = [
            [UserMessage(content="a")],
            [UserMessage(content="b")],
            [UserMessage(content="c")],
        ]

        def _provider() -> _FakeProvider:
            """Build a fresh provider whose first serialized item exhausts under a one-attempt budget."""
            return _FakeProvider(echo=True, failures=[TransientError("x")])

        plain = (
            await LLM(
                _provider(), rate_limiter=_fast_rate_limiter(max_attempts=1, max_in_flight=1)
            )
            .bind(automatic_prompt_caching=True)
            .generate_many(conversations)
        )
        tracer, _exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_provider(), rate_limiter=_fast_rate_limiter(max_attempts=1, max_in_flight=1)),
            tracer=tracer,
            capture_message_content=False,
        )
        wrapped = await traced.bind(automatic_prompt_caching=True).generate_many(conversations)
        plain_rows = [to_row(result) for result in plain]
        wrapped_rows = [to_row(result) for result in wrapped]
        assert [sorted(row) for row in plain_rows] == [sorted(row) for row in wrapped_rows]
        assert [row["output"] for row in plain_rows] == [row["output"] for row in wrapped_rows]
        assert [row["error_text"] is None for row in plain_rows] == [
            row["error_text"] is None for row in wrapped_rows
        ]

    asyncio.run(scenario())


def test_generate_many_abort_marks_the_batch_span_error() -> None:
    """An AbortBatchError propagating from the delegated batch marks the one batch span error."""

    async def scenario() -> None:
        """Serialize a two-item batch whose first item aborts, then inspect the batch span."""
        provider = _FakeProvider(echo=True, failures=[AbortBatchError("misconfigured")])
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=rate_limiter), tracer=tracer, capture_message_content=False
        )
        with pytest.raises(AbortBatchError):
            await traced.bind(automatic_prompt_caching=True).generate_many([
                [UserMessage(content="a")],
                [UserMessage(content="b")],
            ])
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]

    asyncio.run(scenario())


def test_stream_exhausted_then_final_emits_one_span_with_time_to_first_chunk() -> None:
    """A stream iterated to exhaustion then final() ends exactly one span carrying time_to_first_chunk."""

    async def scenario() -> None:
        """Iterate the stream fully, call final(), and inspect the single finished span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=False)
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi") as stream:
            texts = [item async for item in stream if isinstance(item, str)]
            response = await stream.final()
        assert "".join(texts) == "ab"
        assert response.output == "ab"
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
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=False)
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi") as stream:
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
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=False)
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi") as stream:
            async for _item in stream:
                break
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.UNSET

    asyncio.run(scenario())


def test_stream_never_iterated_emits_no_span() -> None:
    """A handle constructed and abandoned without iterating or calling final() emits no span."""

    async def scenario() -> None:
        """Enter and leave the context without driving the stream."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=False)
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi"):
            pass
        assert exporter.get_finished_spans() == ()

    asyncio.run(scenario())


def test_stream_failing_mid_iteration_ends_its_span_with_error_status() -> None:
    """A stream that raises after its first item ends the span with error status."""

    async def _drain(traced: TracedLLM) -> None:
        """Iterate the mid-failing stream to its raise inside an async with block."""
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi") as stream:
            async for _item in stream:
                pass

    async def scenario() -> None:
        """Iterate a mid-failing stream and confirm the error span."""
        provider = _FakeProvider(stream=_MidFailStream(), classify_result="transient")
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=False,
        )
        with pytest.raises(TransientError):
            await _drain(traced)
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]

    asyncio.run(scenario())


def test_stream_final_refusal_ends_the_span_with_error_status() -> None:
    """A structured refusal detected in the stream's final() ends the span with error status and tokens."""

    async def scenario() -> None:
        """Drain a stream whose final() refuses and inspect the error span."""
        provider = _FakeProvider(stream=_RefusingStream())
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=False,
        )
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi") as stream:
            with pytest.raises(RefusalError):
                await stream.final()
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["langchaint.cost_in_usd"] == 0.25

    asyncio.run(scenario())


class _Answer(BaseModel):
    """A response_format model for the rebind and covariance type checks."""

    value: int


def test_rebind_stays_traced_and_shares_the_mapper() -> None:
    """The rebound object stays traced: its generate and stream spans use the same custom mapper."""

    async def scenario() -> None:
        """Rebind, then generate and stream on the rebound object under a key-recording mapper."""
        keys_seen: list[frozenset[str]] = []

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Record its own key set and emit exactly one attribute."""
            keys_seen.append(frozenset({"custom.mapped"}))
            return {"custom.mapped": True}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True)),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        rebound = traced.bind(system_prompt="s", automatic_prompt_caching=True).rebind(
            system_prompt="s2"
        )
        assert_type(rebound, TracedBoundLLM[str])
        await rebound.generate_one("hi")
        async with rebound.stream_one("hi") as stream:
            async for _item in stream:
                pass
            await stream.final()
        generate_span, stream_span = exporter.get_finished_spans()
        # gen_ai.operation.name is the wrapper's required attribute, outside the mapper's control.
        assert generate_span.attributes == {
            "gen_ai.operation.name": "chat",
            "custom.mapped": True,
        }
        assert stream_span.attributes is not None
        # The stream span also carries the wrapper-owned time_to_first_chunk plus the mapped one.
        assert stream_span.attributes["custom.mapped"] is True
        assert keys_seen == [frozenset({"custom.mapped"}), frozenset({"custom.mapped"})]

    asyncio.run(scenario())


def test_custom_attribute_mapper_emits_exactly_its_keys() -> None:
    """A custom attribute_mapper displaces every mapped gen_ai key, keeping only the required one.

    The mapper owns the result-derived attributes, so none of gen_ai_attributes' keys survive it.
    gen_ai.operation.name is the exception by design: the wrapper sets it at span start, before the
    mapper runs, because the convention marks it required on the span kinds it defines.
    """

    async def scenario() -> None:
        """Generate under a two-key mapper and assert the span carries those two plus the required one."""

        def _mapper(result: Response[object] | GenerationError) -> SpanAttributes:
            """Emit two fixed attributes drawn from the result."""
            return {"custom.model": result.model, "custom.attempts": result.attempts}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes == {
            "gen_ai.operation.name": "chat",
            "custom.model": "fake-model",
            "custom.attempts": 1,
        }

    asyncio.run(scenario())


def test_mapper_not_invoked_on_a_non_recording_span() -> None:
    """A custom attribute_mapper never fires when the tracer's spans are non-recording."""

    async def scenario() -> None:
        """Generate under a TracerProvider-less tracer and assert the mapper never ran."""
        calls: list[int] = []

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Count each invocation."""
            calls.append(1)
            return {}

        # No global SDK provider is configured, so get_tracer yields non-recording spans.
        tracer = trace.get_tracer("no-sdk")
        traced = TracedLLM(
            LLM(_FakeProvider()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        response = await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        assert response.output == "ok"
        assert calls == []

    asyncio.run(scenario())


def test_raising_mapper_is_caught_and_the_result_survives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising mapper is logged, generate_one still returns its Response, and the span still ends."""

    async def scenario() -> None:
        """Generate under a mapper that raises and confirm the result and span survive."""

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Raise to simulate a buggy user mapper.

            Raises:
                RuntimeError: always.
            """
            raise RuntimeError("mapper bug")

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            response = await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        assert response.output == "ok"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK
        assert any("mapper" in record.message for record in caplog.records)

    asyncio.run(scenario())


def test_generate_many_does_not_invoke_the_mapper() -> None:
    """The batch span carries no mapped attributes: generate_many delegates and never calls the mapper.

    A batch has no single result to map; per-item detail is in the returned rows.
    This uses a counting mapper
    so a regression that mapped the batch span (or each item) would show as a non-zero count.
    """

    async def scenario() -> None:
        """Run a two-item batch under a counting mapper and confirm it never fired."""
        calls: list[int] = []

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Count each invocation and emit one attribute."""
            calls.append(1)
            return {"custom.mapped": True}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True), rate_limiter=_fast_rate_limiter(max_in_flight=1)),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        results = await traced.bind(automatic_prompt_caching=True).generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "a"
        assert isinstance(second, Response)
        assert second.output == "b"
        assert calls == []
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "custom.mapped" not in span.attributes

    asyncio.run(scenario())


def test_raising_mapper_in_final_still_returns_the_response() -> None:
    """A raising mapper in the stream's final() still returns the assembled Response."""

    async def scenario() -> None:
        """Drain a stream under a raising mapper and read final()."""

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Raise on every call.

            Raises:
                RuntimeError: always.
            """
            raise RuntimeError("mapper bug")

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider()),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        async with traced.bind(automatic_prompt_caching=True).stream_one("hi") as stream:
            async for _item in stream:
                pass
            response = await stream.final()
        assert response.output == "ab"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK

    asyncio.run(scenario())


def test_bind_output_types_are_mirrored() -> None:
    """The overloads mirror LLM.bind: a model gives TracedBoundLLM[Model], absent gives [str]."""
    traced = TracedLLM(LLM(_FakeProvider()), capture_message_content=False)
    structured = traced.bind(response_format=_Answer, automatic_prompt_caching=True)
    assert_type(structured, TracedBoundLLM[_Answer])
    text = traced.bind(automatic_prompt_caching=True)
    assert_type(text, TracedBoundLLM[str])


def _covariance_pin(mapper: AttributeMapper, response: Response[_Answer]) -> SpanAttributes:
    """Pin the mapper covariance: a Response[_Answer] must satisfy the Response[object] parameter.

    pyrefly type-checks this module,
    so a break in Response's inferred OutputT covariance surfaces as a type error on the call below
    rather than as a runtime surprise.
    """
    return mapper(response)


def test_traced_passthroughs_reach_the_wrapped_objects() -> None:
    """The provider and rate_limiter pass through TracedLLM; the BoundLLM fields through TracedBoundLLM."""
    provider = _FakeProvider()
    rate_limiter = _fast_rate_limiter()
    traced = TracedLLM(LLM(provider, rate_limiter=rate_limiter), capture_message_content=False)
    assert traced.provider is provider
    assert traced.rate_limiter is rate_limiter
    bound = traced.bind(response_format=_Answer, automatic_prompt_caching=True)
    assert bound.provider is provider
    assert bound.rate_limiter is rate_limiter
    assert bound.response_format is _Answer
    assert bound.tool_manager is None
    assert bound.binding.system_prompt is None


def test_wrapping_a_stream_creates_a_traced_stream() -> None:
    """The stream_one call returns a TracedStreamHandle, the wrapper that owns the stream span."""
    traced = TracedLLM(LLM(_FakeProvider()), capture_message_content=False)
    handle = traced.bind(automatic_prompt_caching=True).stream_one("hi")
    assert isinstance(handle, TracedStreamHandle)


def test_extra_attributes_ride_on_generate_spans_and_mapper_wins_collisions() -> None:
    """extra_attributes land at span start on generate spans; a mapper key of the same name wins."""

    async def scenario() -> None:
        """Generate under extra_attributes plus a colliding mapper key and inspect the span."""

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Emit one attribute colliding with an extra_attributes key."""
            return {"shared.key": "mapped"}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True)),
            attribute_mapper=_mapper,
            extra_attributes={"gen_ai.agent.name": "agent_a", "shared.key": "extra"},
            tracer=tracer,
            capture_message_content=False,
        )
        await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.agent.name"] == "agent_a"
        assert span.attributes["shared.key"] == "mapped"

    asyncio.run(scenario())


def test_extra_attributes_survive_rebind_and_reach_stream_and_batch_spans() -> None:
    """extra_attributes pass through rebind and land on the stream span and the batch span."""

    async def scenario() -> None:
        """Rebind, then stream and batch under one extra_attributes mapping; both spans carry it."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True), rate_limiter=_fast_rate_limiter(max_in_flight=1)),
            extra_attributes={"gen_ai.agent.name": "agent_a"},
            tracer=tracer,
            capture_message_content=False,
        )
        rebound = traced.bind(system_prompt="s", automatic_prompt_caching=True).rebind(
            system_prompt="s2"
        )
        async with rebound.stream_one("hi") as stream:
            await stream.final()
        await rebound.generate_many([[UserMessage(content="a")], [UserMessage(content="b")]])
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(
            span.attributes is not None and span.attributes["gen_ai.agent.name"] == "agent_a"
            for span in spans
        )

    asyncio.run(scenario())


def test_gen_ai_attributes_is_public_and_composable() -> None:
    """A custom mapper extending gen_ai_attributes lands every standard key plus the added one.

    The extension is a value derived from the result, the case a mapper exists for:
    a constant would ride in through extra_attributes instead.
    Deriving it also proves the mapper receives the real result, not a stand-in carrying only the mapped keys.
    """

    async def scenario() -> None:
        """Generate under a composed mapper and check a standard key and the derived key."""

        def _mapper(result: Response[object] | GenerationError) -> SpanAttributes:
            """Extend the built-in attributes with the call's total request time."""
            return {
                **gen_ai_attributes(result),
                "app.request_seconds": sum(a.elapsed_seconds for a in result.attempt_records),
            }

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True)),
            attribute_mapper=_mapper,
            tracer=tracer,
            capture_message_content=False,
        )
        response = await traced.bind(automatic_prompt_caching=True).generate_one("hi")
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
    """Stand in for the tool function; the capture tests never dispatch a call to it."""
    return ""


def _unserializable_schema_tool() -> JSONSchemaTool:
    """Build a tool whose args_schema json.dumps cannot serialize.

    args_schema is Mapping[str, object] the application supplies verbatim, so it can hold a value with no
    JSON form; the set below is the smallest one.
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
    """Build the echo tool."""
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


def test_traced_tool_manager_handled_dispatch_emits_one_execute_tool_span() -> None:
    """A successful dispatch emits one OK INTERNAL execute_tool span with the identity keys and no error.type."""

    async def scenario() -> None:
        """Dispatch one valid call and inspect the single finished span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        outcome = await tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json='{"text": "hi"}')
        )
        assert isinstance(outcome, DispatchHandled)
        assert outcome.tool_message.content == "hi"
        (span,) = exporter.get_finished_spans()
        assert span.name == "execute_tool echo"
        assert span.kind == SpanKind.INTERNAL
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        assert dict(span.attributes) == {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "echo",
            "gen_ai.tool.call.id": "call1",
        }

    asyncio.run(scenario())


def test_traced_tool_manager_function_authored_failure_is_tool_error() -> None:
    """A function-authored failure is error.type tool_error, the one value read off the ToolMessage bool.

    This is the input separating tool_error from the no-error case within one DispatchHandled arm,
    so an implementation classifying by outcome arm alone instead of the ToolMessage bool fails here.
    """

    async def scenario() -> None:
        """Dispatch a call whose function returns ToolOutputExplicit(is_error=True) and inspect the span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_erring_tool()], tracer=tracer, capture_message_content=False
        )
        outcome = await tool_manager.dispatch(
            ToolCall(id="call1", name="erring", args_json='{"text": "x"}')
        )
        assert isinstance(outcome, DispatchHandled)
        assert outcome.tool_message.is_error is True
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "tool_error"

    asyncio.run(scenario())


def test_traced_tool_manager_invalid_args_span_is_error_with_invalid_tool_args() -> None:
    """An invalid-arguments dispatch is an ERROR span classified invalid_tool_args: the tool never ran."""

    async def scenario() -> None:
        """Dispatch a call whose arguments fail validation and inspect the span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        outcome = await tool_manager.dispatch(
            ToolCall(id="call1", name="echo", args_json='{"wrong": 1}')
        )
        assert isinstance(outcome, DispatchInvalidToolArgs)
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "invalid_tool_args"

    asyncio.run(scenario())


def test_traced_tool_manager_unknown_tool_span_is_error_with_unknown_tool() -> None:
    """An off-list name is an ERROR span named for the called name, classified unknown_tool."""

    async def scenario() -> None:
        """Dispatch a call naming a tool the manager does not hold and inspect the span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        outcome = await tool_manager.dispatch(ToolCall(id="call1", name="missing", args_json="{}"))
        assert isinstance(outcome, DispatchUnknownTool)
        (span,) = exporter.get_finished_spans()
        assert span.name == "execute_tool missing"
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "unknown_tool"

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


def test_traced_tool_manager_is_a_tool_manager_bindable_as_one() -> None:
    """TracedToolManager subclasses ToolManager, so bind's tool_manager parameter accepts it unchanged."""
    tool_manager = TracedToolManager([_echo_tool()], capture_message_content=False)
    assert isinstance(tool_manager, ToolManager)
    bound = TracedLLM(LLM(_FakeProvider()), capture_message_content=False).bind(
        tool_manager=tool_manager, automatic_prompt_caching=True
    )
    assert bound.tool_manager is tool_manager
    (schema,) = tool_manager.schemas()
    assert schema.name == "echo"


def test_traced_tool_manager_extra_attributes_ride_on_dispatch_spans() -> None:
    """extra_attributes land on every dispatch span; a dispatch-set identity key of the same name wins."""

    async def scenario() -> None:
        """Dispatch under extra_attributes including a colliding identity key and inspect the span."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()],
            tracer=tracer,
            extra_attributes={"gen_ai.agent.name": "agent_a", "gen_ai.tool.name": "spoofed"},
            capture_message_content=False,
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "a"}'))
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.agent.name"] == "agent_a"
        assert span.attributes["gen_ai.tool.name"] == "echo"

    asyncio.run(scenario())


def test_generate_many_passes_warm_cache_through() -> None:
    """warm_cache reaches BoundLLM.generate_many: the warming item never overlaps a sibling."""

    async def scenario() -> None:
        """Run a three-item batch on a slow fake with a wide slot and read the recorded peak."""
        provider = _FakeProvider(echo=True, send_seconds=0.01)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter(max_in_flight=8)),
            tracer=tracer,
            capture_message_content=False,
        )
        results = await traced.bind(automatic_prompt_caching=True).generate_many(
            [[UserMessage(content=str(index))] for index in range(3)], warm_cache=True
        )
        assert all(isinstance(result, Response) for result in results)
        assert provider.bound_providers[0].peak_in_flight == 2
        (span,) = exporter.get_finished_spans()
        assert span.kind == SpanKind.INTERNAL

    asyncio.run(scenario())


def test_each_convention_defined_span_kind_carries_the_required_operation_name() -> None:
    """gen_ai.operation.name is set on every span kind but generate_many's aggregate.

    One test over every kind rather than one assertion added to each kind's own test:
    a required attribute missing from one span-opening site is the failure worth catching,
    and that is visible only by covering the sites together.
    The values are read in end order rather than keyed by span name, because three of the four
    spans share the name "chat fake-model"; keying by name would collapse them and let a wrong
    value on generate_one or generate_many pass.
    """

    async def scenario() -> None:
        """Open one span of each kind and read the operation name each carries, in end order."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=False)
        bound = traced.bind(automatic_prompt_caching=True)
        await bound.generate_one("hi")
        await bound.generate_many([[UserMessage(content="hi")]])
        async with bound.stream_one("hi") as stream:
            await stream.final()
        tool_manager = TracedToolManager(
            [_echo_tool()], tracer=tracer, capture_message_content=False
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "hi"}'))
        spans = exporter.get_finished_spans()
        # generate_many delegates to the plain BoundLLM, so its batch span has no traced children.
        assert len(spans) == 4
        assert [span.name for span in spans] == [
            "chat fake-model",
            "chat fake-model",
            "chat fake-model",
            "execute_tool echo",
        ]
        assert [(span.attributes or {}).get("gen_ai.operation.name") for span in spans] == [
            "chat",
            None,
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
            LLM(_FakeProvider()),
            tracer=tracer,
            capture_message_content=False,
            extra_attributes={"gen_ai.operation.name": "not-the-operation"},
        )
        await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.operation.name"] == "chat"

    asyncio.run(scenario())


def test_extra_attributes_cannot_displace_the_operation_name_on_a_dispatch_span() -> None:
    """The same precedence holds on the dispatch span, which sets the key inline.

    A dispatch span does not route through the helper the generate spans use, so the ordering is
    written twice and a test of one site does not cover the other.
    """

    async def scenario() -> None:
        """Dispatch under extra_attributes claiming the operation name key."""
        tracer, exporter = _in_memory_tracer()
        tool_manager = TracedToolManager(
            [_echo_tool()],
            tracer=tracer,
            capture_message_content=False,
            extra_attributes={"gen_ai.operation.name": "not-the-operation"},
        )
        await tool_manager.dispatch(ToolCall(id="call1", name="echo", args_json='{"text": "hi"}'))
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"

    asyncio.run(scenario())


def _emitted_convention_keys() -> set[str]:
    """Collect every quoted gen_ai.* literal in the tracing module's source.

    Reads the source rather than a hand-kept list, so a key added to the module without a matching
    constant is caught by the next run instead of by a reader noticing.
    Bare gen_ai.* mentions in the module docstring carry no quote and are not collected.
    """
    source = pathlib.Path(inspect.getfile(langchaint.tracing)).read_text()
    return set(re.findall(r'"(gen_ai\.[a-z_.]+)"', source))


def test_emitted_convention_keys_are_defined_at_the_pinned_revision() -> None:
    """Every gen_ai.* key the module emits, and error.type, resolves in the installed semantic conventions.

    opentelemetry-api does not depend on semantic-conventions, which is why the module writes bare string
    literals; importing semconv at runtime would break its api-only import tenet.
    opentelemetry-sdk pins opentelemetry-semantic-conventions exactly and the tests already require the sdk,
    so the constants are present here and absent from the runtime dependency set.
    Because that pin is exact, this also fires on an sdk bump, which is when a renamed or withdrawn key
    needs to be heard about.
    It cannot check that a key is semantically right, nor check payload shapes.
    """
    defined = {
        value
        for name, value in vars(gen_ai_semconv).items()
        if name.startswith("GEN_AI_") and isinstance(value, str)
    }
    emitted = _emitted_convention_keys()
    assert emitted, "the source scan found no keys, so this assertion would pass vacuously"
    assert emitted <= defined
    assert error_semconv.ERROR_TYPE == "error.type"


class _ReasoningOnlyBoundProvider(_FakeBoundProvider):
    """A bound provider whose success carries a turn of reasoning and nothing else."""

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[str]:
        """Return a result whose assistant turn holds one ReasoningTrace and no text."""
        return ProviderResult(
            output="",
            assistant_message=AssistantMessage(
                turn=(ReasoningTrace(reasoning={"signature": "opaque"}),)
            ),
            usage=_USAGE,
            usage_raw=_FAKE_RAW_USAGE,
            stop_reason="end_turn",
            raw=_FakeRawResponse(),
        )


class _ReasoningOnlyProvider(_FakeProvider):
    """A provider handing out bound providers whose turns hold only reasoning."""

    @override
    def bind_text(self, binding: Binding) -> BoundProvider[str]:
        """Hand out the reasoning-only bound provider."""
        return _ReasoningOnlyBoundProvider()


def _captured(exporter: InMemorySpanExporter, key: str) -> object:
    """Read one span's JSON content attribute back as Python data."""
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    value = span.attributes[key]
    assert isinstance(value, str)
    return json.loads(value)


def test_capture_off_leaves_every_content_key_off_the_span() -> None:
    """capture_message_content False emits none of the four content keys, even with all four sources present.

    Kept separate from the absent-source tests below: those omit a key because its source is empty,
    and would pass vacuously against a capture-off implementation.
    """

    async def scenario() -> None:
        """Generate under a binding carrying a system prompt and a tool, with capture off."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True)), tracer=tracer, capture_message_content=False
        )
        bound = traced.bind(
            system_prompt="be brief",
            tool_manager=ToolManager([_echo_tool()]),
            automatic_prompt_caching=True,
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
    """capture_message_content True records the system prompt, tools, conversation, and assistant turn."""

    async def scenario() -> None:
        """Generate over a conversation carrying every message role and inspect the shapes."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            system_prompt="be brief",
            tool_manager=ToolManager([_echo_tool()]),
            automatic_prompt_caching=True,
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


def test_a_str_conversation_is_captured_as_one_user_message() -> None:
    """The bare-str conversation form renders as the one user message it means."""

    async def scenario() -> None:
        """Generate from a str conversation and read the input messages back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        assert _captured(exporter, "gen_ai.input.messages") == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]

    asyncio.run(scenario())


def test_image_parts_are_captured_without_their_bytes() -> None:
    """An ImagePart records its media type and never its base64 payload."""

    async def scenario() -> None:
        """Generate over a conversation holding an image and read the input messages back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        await traced.bind(automatic_prompt_caching=True).generate_one([
            UserMessage(
                content=(
                    TextPart(text="what is this"),
                    ImagePart(data=b"\x89PNGsecret", media_type="image/png"),
                )
            )
        ])
        assert _captured(exporter, "gen_ai.input.messages") == [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "what is this"},
                    {"type": "blob", "mime_type": "image/png"},
                ],
            }
        ]
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "PNGsecret" not in str(span.attributes["gen_ai.input.messages"])

    asyncio.run(scenario())


def test_reasoning_is_excluded_leaving_an_empty_parts_array() -> None:
    """A turn of reasoning alone still emits its message, with parts emptied by the exclusion.

    The empty array is the deliberate exception to the omit-an-absent-source rule:
    there was an assistant turn, and excluding reasoning is what emptied it.
    """

    async def scenario() -> None:
        """Generate a reasoning-only turn and read the output messages back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_ReasoningOnlyProvider()), tracer=tracer, capture_message_content=True
        )
        await traced.bind(automatic_prompt_caching=True).generate_one("hi")
        assert _captured(exporter, "gen_ai.output.messages") == [
            {"role": "assistant", "parts": [], "finish_reason": "stop"}
        ]
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "opaque" not in str(span.attributes["gen_ai.output.messages"])

    asyncio.run(scenario())


def test_an_absent_system_prompt_omits_its_key_while_capture_stays_on() -> None:
    """No bound system prompt omits gen_ai.system_instructions; the conversation is still captured.

    The captured input messages are what separates this from the capture-off case.
    """

    async def scenario() -> None:
        """Generate under a binding with no system prompt and no tools."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        await traced.bind(automatic_prompt_caching=True).generate_one("hi")
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
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            system_prompt=[TextPart(text="be brief"), TextPart(text="cite sources")],
            automatic_prompt_caching=True,
        )
        await bound.generate_one("hi")
        assert _captured(exporter, "gen_ai.system_instructions") == [
            {"type": "text", "content": "be brief"},
            {"type": "text", "content": "cite sources"},
        ]

    asyncio.run(scenario())


def test_the_error_path_captures_input_and_no_output() -> None:
    """A failed call keeps the input attributes set at span start and emits no output messages."""

    async def scenario() -> None:
        """Drive a refusal under capture and inspect the error span."""
        provider = _FakeProvider(
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
                )
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter()),
            tracer=tracer,
            capture_message_content=True,
        )
        with pytest.raises(RefusalError):
            await traced.bind(
                system_prompt="be brief", automatic_prompt_caching=True
            ).generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.input.messages" in span.attributes
        assert "gen_ai.system_instructions" in span.attributes
        assert "gen_ai.output.messages" not in span.attributes
        assert span.attributes["error.type"] == "RefusalError"

    asyncio.run(scenario())


def test_generate_many_captures_no_content_even_under_capture() -> None:
    """The batch span has no single conversation and no single turn, so it records neither."""

    async def scenario() -> None:
        """Run a two-item batch under capture and inspect the aggregate span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_FakeProvider(echo=True), rate_limiter=_fast_rate_limiter(max_in_flight=1)),
            tracer=tracer,
            capture_message_content=True,
        )
        await traced.bind(automatic_prompt_caching=True).generate_many(["a", "b"])
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert not {
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "gen_ai.input.messages",
            "gen_ai.output.messages",
        } & set(span.attributes)
        assert span.attributes["langchaint.batch_item_count"] == 2

    asyncio.run(scenario())


def test_content_that_cannot_be_serialized_is_logged_and_never_reaches_the_caller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSONSchemaTool args_schema json.dumps cannot serialize leaves the call and its result intact.

    Serializing it raises inside the tracing wrapper, which catches it the same way it catches a raising
    AttributeMapper.
    The three input keys build as one dict, so the failure drops all three rather than a subset.
    """

    async def scenario() -> None:
        """Generate under capture with the unserializable tool bound, then read the span and the log."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            tool_manager=ToolManager([_unserializable_schema_tool()]),
            automatic_prompt_caching=True,
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


def test_unserializable_content_leaves_a_bare_iterator_stream_and_its_span_intact(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stream path holds the same guarantee generate_one does, at _ensure_span.

    _ensure_span is the one site on this path a reachable value can fail at.
    An unserializable args_schema reaches json.dumps only through _input_content_attributes;
    every field final()'s output build serializes is typed str.
    Driven as a bare async iterator rather than with async with, so nothing outside the handle would end
    the span: an unguarded raise out of _ensure_span both kills the stream and orphans the span,
    since __anext__ and final() both call it outside their own try.
    """

    async def scenario() -> None:
        """Stream to completion with the unserializable tool bound, then read the span and the log."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(
            tool_manager=ToolManager([_unserializable_schema_tool()]),
            automatic_prompt_caching=True,
        )
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            stream = bound.stream_one("hi")
            _ = [item async for item in stream]
            response = await stream.final()
        assert response.output == "ab"
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.output.messages" in span.attributes
        assert "gen_ai.response.time_to_first_chunk" in span.attributes
        assert "content capture raised" in caplog.text

    asyncio.run(scenario())


def test_the_stream_span_captures_input_at_start_and_output_at_final() -> None:
    """A traced stream records the input attributes when its lazy span starts and the turn at final()."""

    async def scenario() -> None:
        """Drive a stream to completion under capture and read both sides back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(system_prompt="be brief", automatic_prompt_caching=True)
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


def test_capture_survives_rebind_and_reaches_the_rebound_binding() -> None:
    """Rebind carries capture_message_content through, so a rebound object cannot silently lose it."""

    async def scenario() -> None:
        """Rebind a captured binding and confirm the new one still captures."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        rebound = traced.bind(system_prompt="s", automatic_prompt_caching=True).rebind(
            system_prompt="s2"
        )
        await rebound.generate_one("hi")
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
        # The attribute's own string, not its decoded value: the effect on this key is normalization,
        # and the spacing here differs from args_json, so a decoded comparison would also pass
        # against an implementation that set the attribute to args_json untouched.
        assert span.attributes["gen_ai.tool.call.arguments"] == '{"text": "hi", "n": 1}'
        assert _captured(exporter, "gen_ai.tool.call.result") == {
            "type": "tool_call_response",
            "id": "call1",
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
    """Input whose parse does not round-trip as JSON falls back to the raw text.

    Python's json accepts and emits Infinity and NaN, which JSON has no syntax for, and reaches them
    from ordinary input too (1e400 overflows to inf). Emitting the parsed value would put a bare
    Infinity or NaN token on the span, which a strict consumer rejects, so these route to the
    raw-text arm and go out as a JSON string instead.
    """

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
    """Malformed argument text is preserved on the span, as a JSON string rather than as raw bytes.

    The deserialization is best effort, so the span still shows what the model emitted rather than
    dropping the key or raising; the outcome is the DispatchInvalidToolArgs arm reporting the same input.
    The attribute goes out through the same re-serialization as a parsed value, so the text arrives
    quoted and escaped; a consumer decodes the attribute and gets the original back.
    """

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


def test_conversation_tool_calls_nest_parsed_arguments_and_keep_unparseable_text() -> None:
    """A parsed argument object nests inside gen_ai.input.messages; unparseable text stays a string there.

    This is the site where the nesting matters: the parts array is serialized as a whole, so a parsed
    object arrives as nested JSON rather than as a string a consumer decodes a second time.
    One turn carrying both kinds pins them against each other, so an implementation that parsed every
    call, dropping or raising on the second, fails here.
    """

    async def scenario() -> None:
        """Generate over a turn holding one parseable and one unparseable tool call."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        bound = traced.bind(automatic_prompt_caching=True)
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
    [('[1, 2]', [1, 2]), ('"bare"', "bare"), ("7", 7), ("null", None)],
)
def test_a_non_object_argument_value_still_nests_as_the_value_it_parses_to(
    args_json: str, expected: object
) -> None:
    """Best effort covers any JSON value, not only an object, so a non-object nests rather than falling back.

    The convention expects an object here and every other argument test feeds one, so an implementation
    that parsed and then kept the text unless the result was a dict would pass all of them.
    A model producing a non-object is what the DispatchInvalidToolArgs arm reports; the span records
    the value it produced.
    """

    async def scenario() -> None:
        """Generate over a turn whose tool call carries a non-object argument value."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        await traced.bind(automatic_prompt_caching=True).generate_one([
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
    """A finite float nests as the number it is, so the number hooks reject only what cannot round-trip.

    The other argument tests carry strings and integers, which reach json.loads' parse_int, never its
    parse_float; without this case a parse_float that rejected every input would pass them all,
    since the non-finite cases expect the fallback anyway.
    """

    async def scenario() -> None:
        """Generate over a turn whose tool call carries a finite float and read the arguments back."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer, capture_message_content=True)
        await traced.bind(automatic_prompt_caching=True).generate_one([
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


def test_tool_span_captures_the_result_on_both_arms_where_no_tool_ran() -> None:
    """gen_ai.tool.call.result is recorded on both failure arms, beside the error.type saying why.

    The convention defines the key as the result "if any and if execution was successful", so recording it
    here is the deliberate departure the TracedToolManager docstring states.
    Pinned in both directions on each arm: the package-rendered correction is what the model reads and
    adapts to, and error.type on the same span is what tells a consumer no tool produced it.
    """

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
                "response": [{"type": "text", "content": expected.tool_message.content}],
            }

    asyncio.run(scenario())
