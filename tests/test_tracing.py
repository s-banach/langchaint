"""TracedLLM / TracedBoundLLM / TracedStreamHandle driven by fake providers and an in-memory exporter.

The fake providers are the ones test_bound_llm builds; here they feed the tracing wrapper instead of the plain BoundLLM.
A locally built TracerProvider with a SimpleSpanProcessor and an InMemorySpanExporter captures every span,
so the assertions read span names, kinds, statuses, attributes, events,
and parentage without any collector or network access.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import assert_type, override

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import BaseModel

from langchaint import (
    LLM,
    AbortBatchError,
    ExceededMaxCompletionTokensError,
    GenerationError,
    RefusalError,
    Response,
    RetriesExhaustedError,
    StreamItem,
    TransientError,
    UserMessage,
    to_row,
)
from langchaint.tracing import (
    AttributeMapper,
    SpanAttributes,
    TracedBoundLLM,
    TracedLLM,
    TracedStreamHandle,
)
from tests.test_bound_llm import (
    _USAGE,
    _FakeProvider,
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
        traced = TracedLLM(LLM(_FakeProvider(echo=True)), tracer=tracer)
        response = await traced.bind().generate_one("hi")
        assert response.output == "hi"
        (span,) = exporter.get_finished_spans()
        assert span.name == "chat fake-model"
        assert span.kind == SpanKind.CLIENT
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        assert dict(span.attributes) == {
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.response.finish_reasons": ("end_turn",),
            "gen_ai.usage.input_tokens": _USAGE.input_tokens_total,
            "gen_ai.usage.output_tokens": _USAGE.output_tokens,
            "langchaint.attempts": 1,
            "langchaint.cost_in_usd": 0.0,
            "langchaint.input_tokens_cache_read": _USAGE.input_tokens_cache_read,
            "langchaint.input_tokens_cache_write": _USAGE.input_tokens_cache_write,
            "langchaint.input_tokens_cache_none": _USAGE.input_tokens_cache_none,
        }

    asyncio.run(scenario())


def test_generate_one_refusal_span_has_error_status_and_real_tokens() -> None:
    """A refusal leaf yields an error span carrying the rejected 200's real token counts and cost."""

    async def scenario() -> None:
        """Drive one generate_one whose send refuses, then inspect the error span."""
        provider = _FakeProvider(
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE, cost_in_usd=0.25, stop_reason="refusal"
                )
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(provider, rate_limiter=_fast_rate_limiter()), tracer=tracer)
        with pytest.raises(RefusalError):
            await traced.bind().generate_one("hi")
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
                ExceededMaxCompletionTokensError.for_rejected_200(
                    usage=_USAGE, cost_in_usd=0.25, stop_reason="max_tokens"
                )
            ]
        )
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(provider, rate_limiter=_fast_rate_limiter()), tracer=tracer)
        with pytest.raises(ExceededMaxCompletionTokensError):
            await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["gen_ai.response.finish_reasons"] == ("max_tokens",)
        assert span.attributes["langchaint.cost_in_usd"] == 0.25

    asyncio.run(scenario())


def test_generate_one_retries_exhausted_span_has_error_status_and_zero_tokens() -> None:
    """A retries-exhausted failure over transport errors bills zero, so the usage attributes are zero."""

    async def scenario() -> None:
        """Exhaust the budget on transport failures and inspect the error span."""
        provider = _FakeProvider(failures=[TransientError("e1"), TransientError("e2")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(provider, rate_limiter=_fast_rate_limiter(max_attempts=2)), tracer=tracer
        )
        with pytest.raises(RetriesExhaustedError):
            await traced.bind().generate_one("hi")
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
        traced = TracedLLM(LLM(provider, rate_limiter=_fast_rate_limiter()), tracer=tracer)
        with pytest.raises(AbortBatchError):
            await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        # An abort carries no shared-field attributes; it is recorded as an exception event instead.
        assert span.attributes == {}
        assert [event.name for event in span.events] == ["exception"]

    asyncio.run(scenario())


def test_retry_surfaces_as_an_attempt_failed_span_event() -> None:
    """A recovered transient failure becomes one langchaint.attempt_failed event on the success span."""

    async def scenario() -> None:
        """Recover one generate_one from a transient failure, then read the span event."""
        provider = _FakeProvider(failures=[TransientError("boom")])
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(provider, rate_limiter=_fast_rate_limiter()), tracer=tracer)
        response = await traced.bind().generate_one("hi")
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
                    usage=_USAGE, cost_in_usd=0.25, stop_reason="refusal"
                )
            ],
        )
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(provider, rate_limiter=rate_limiter), tracer=tracer)
        results = await traced.bind().generate_many([
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
            .bind()
            .generate_many(conversations)
        )
        tracer, _exporter = _in_memory_tracer()
        traced = TracedLLM(
            LLM(_provider(), rate_limiter=_fast_rate_limiter(max_attempts=1, max_in_flight=1)),
            tracer=tracer,
        )
        wrapped = await traced.bind().generate_many(conversations)
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
        traced = TracedLLM(LLM(provider, rate_limiter=rate_limiter), tracer=tracer)
        with pytest.raises(AbortBatchError):
            await traced.bind().generate_many([
                [UserMessage(content="a")],
                [UserMessage(content="b")],
            ])
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]

    asyncio.run(scenario())


def test_stream_exhausted_then_final_emits_one_span_with_time_to_first_token() -> None:
    """A stream iterated to exhaustion then final() ends exactly one span carrying the TTFT attribute."""

    async def scenario() -> None:
        """Iterate the stream fully, call final(), and inspect the single finished span."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer)
        async with traced.bind().stream_one("hi") as stream:
            texts = [item async for item in stream if isinstance(item, str)]
            response = await stream.final()
        assert "".join(texts) == "ab"
        assert response.output == "ab"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK
        assert span.attributes is not None
        time_to_first_token_seconds = span.attributes["langchaint.time_to_first_token_seconds"]
        assert isinstance(time_to_first_token_seconds, float)
        assert time_to_first_token_seconds >= 0.0
        assert span.attributes["gen_ai.response.finish_reasons"] == ("end_turn",)

    asyncio.run(scenario())


def test_stream_final_is_idempotent_and_ends_the_span_once() -> None:
    """A second final() returns the same Response and does not end a second span."""

    async def scenario() -> None:
        """Call final() twice on one drained stream and count the spans."""
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer)
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
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer)
        async with traced.bind().stream_one("hi") as stream:
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
        traced = TracedLLM(LLM(_FakeProvider()), tracer=tracer)
        async with traced.bind().stream_one("hi"):
            pass
        assert exporter.get_finished_spans() == ()

    asyncio.run(scenario())


def test_stream_failing_mid_iteration_ends_its_span_with_error_status() -> None:
    """A stream that raises after its first item ends the span with error status."""

    async def _drain(traced: TracedLLM) -> None:
        """Iterate the mid-failing stream to its raise inside an async with block."""
        async with traced.bind().stream_one("hi") as stream:
            async for _item in stream:
                pass

    async def scenario() -> None:
        """Iterate a mid-failing stream and confirm the error span."""
        provider = _FakeProvider(stream=_MidFailStream(), classify_result="transient")
        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(provider, rate_limiter=_fast_rate_limiter()), tracer=tracer)
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
        traced = TracedLLM(LLM(provider, rate_limiter=_fast_rate_limiter()), tracer=tracer)
        async with traced.bind().stream_one("hi") as stream:
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
        traced = TracedLLM(LLM(_FakeProvider(echo=True)), attribute_format=_mapper, tracer=tracer)
        rebound = traced.bind(system_prompt="s").rebind(system_prompt="s2")
        assert_type(rebound, TracedBoundLLM[str])
        await rebound.generate_one("hi")
        async with rebound.stream_one("hi") as stream:
            async for _item in stream:
                pass
            await stream.final()
        generate_span, stream_span = exporter.get_finished_spans()
        assert generate_span.attributes == {"custom.mapped": True}
        assert stream_span.attributes is not None
        # The stream span also carries the wrapper-owned TTFT attribute plus the mapped one.
        assert stream_span.attributes["custom.mapped"] is True
        assert keys_seen == [frozenset({"custom.mapped"}), frozenset({"custom.mapped"})]

    asyncio.run(scenario())


def test_callable_attribute_format_emits_exactly_its_keys() -> None:
    """A callable attribute_format sets exactly the keys it returns, no gen_ai keys."""

    async def scenario() -> None:
        """Generate under a two-key mapper and assert the span carries only those two keys."""

        def _mapper(result: Response[object] | GenerationError) -> SpanAttributes:
            """Emit two fixed attributes drawn from the result."""
            return {"custom.model": result.model, "custom.attempts": result.attempts}

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), attribute_format=_mapper, tracer=tracer)
        await traced.bind().generate_one("hi")
        (span,) = exporter.get_finished_spans()
        assert span.attributes == {"custom.model": "fake-model", "custom.attempts": 1}

    asyncio.run(scenario())


def test_mapper_not_invoked_on_a_non_recording_span() -> None:
    """A callable attribute_format never fires when the tracer's spans are non-recording."""

    async def scenario() -> None:
        """Generate under a TracerProvider-less tracer and assert the mapper never ran."""
        calls: list[int] = []

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Count each invocation."""
            calls.append(1)
            return {}

        # No global SDK provider is configured, so get_tracer yields non-recording spans.
        tracer = trace.get_tracer("no-sdk")
        traced = TracedLLM(LLM(_FakeProvider()), attribute_format=_mapper, tracer=tracer)
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

        def _mapper(_result: Response[object] | GenerationError) -> SpanAttributes:
            """Raise to simulate a buggy user mapper.

            Raises:
                RuntimeError: always.
            """
            raise RuntimeError("mapper bug")

        tracer, exporter = _in_memory_tracer()
        traced = TracedLLM(LLM(_FakeProvider()), attribute_format=_mapper, tracer=tracer)
        with caplog.at_level(logging.WARNING, logger="langchaint.tracing"):
            response = await traced.bind().generate_one("hi")
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
            attribute_format=_mapper,
            tracer=tracer,
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
        traced = TracedLLM(LLM(_FakeProvider()), attribute_format=_mapper, tracer=tracer)
        async with traced.bind().stream_one("hi") as stream:
            async for _item in stream:
                pass
            response = await stream.final()
        assert response.output == "ab"
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.OK

    asyncio.run(scenario())


def test_unknown_attribute_format_string_raises_key_error() -> None:
    """An untyped caller passing an unknown format string gets a KeyError naming the known formats."""
    llm = LLM(_FakeProvider())
    with pytest.raises(KeyError, match="gen_ai"):
        # pyrefly: ignore[bad-argument-type]  # the Literal rejects this statically; runtime guard here
        TracedLLM(llm, attribute_format="openinference")


def test_bind_output_types_are_mirrored() -> None:
    """The overloads mirror LLM.bind: a model gives TracedBoundLLM[Model], absent gives [str]."""
    traced = TracedLLM(LLM(_FakeProvider()))
    structured = traced.bind(response_format=_Answer)
    assert_type(structured, TracedBoundLLM[_Answer])
    text = traced.bind()
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
    traced = TracedLLM(LLM(provider, rate_limiter=rate_limiter))
    assert traced.provider is provider
    assert traced.rate_limiter is rate_limiter
    bound = traced.bind(response_format=_Answer)
    assert bound.provider is provider
    assert bound.rate_limiter is rate_limiter
    assert bound.response_format is _Answer
    assert bound.tool_manager is None
    assert bound.binding.system_prompt is None


def test_wrapping_a_stream_creates_a_traced_stream() -> None:
    """The stream_one call returns a TracedStreamHandle, the wrapper that owns the stream span."""
    traced = TracedLLM(LLM(_FakeProvider()))
    handle = traced.bind().stream_one("hi")
    assert isinstance(handle, TracedStreamHandle)
