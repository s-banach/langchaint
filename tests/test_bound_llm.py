"""BoundLLM and StreamHandle driven by fake adapters.

A fake BoundAdapter scripts send to fail a fixed number of times before succeeding,
and a fake AdapterStream emits a fixed item sequence.
Together they pin the retry loop, rebind rebuild, batch ordering, and the stream contract without any network access.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import assert_type, override

import pytest
from pydantic import BaseModel

from langchaint import (
    LLM,
    AbortBatchError,
    AssistantMessage,
    BoundLLM,
    GenerationError,
    MaxCompletionTokensExceededError,
    Message,
    PricingTable,
    RateLimiter,
    RefusalError,
    Response,
    RetriesExhaustedError,
    StreamItem,
    StreamProtocolError,
    TextPart,
    ToolCall,
    TransientError,
    Usage,
    UserMessage,
)
from langchaint import rate_limiter as rate_limiter_module
from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    BoundAdapter,
    ErrorClass,
)
from langchaint.llm import UNCHANGED

_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)
_USAGE = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=1,
    output_tokens=1,
    output_tokens_reasoning=0,
    cost_in_usd=0.0,
)
_USAGE_BILLED = _USAGE.model_copy(update={"cost_in_usd": 0.25})
"""The billing a rejected 200 (a refusal or truncation) carries."""
_USAGE_STREAM = _USAGE.model_copy(update={"cost_in_usd": 0.001})
"""The stream final()'s assembled usage, distinct so a stream cost is visible."""


def _fast_rate_limiter(*, max_attempts: int = 3, max_in_flight: int = 8) -> RateLimiter:
    """Build a fresh near-zero-backoff rate limiter; one instance serves one event loop."""
    return RateLimiter(
        max_attempts=max_attempts,
        backoff_base_seconds=0.001,
        max_in_flight=max_in_flight,
    )


def _batch_outputs(results: list[Response[str] | GenerationError]) -> list[str]:
    """Assert every batch result is a success and return the outputs in order.

    Narrowing each result to Response is what makes result.output well-typed,
    and it doubles as the assertion that no item exhausted its retries in a batch the test expects to succeed whole.
    """
    outputs: list[str] = []
    for result in results:
        assert isinstance(result, Response)
        outputs.append(result.output)
    return outputs


class _FakeRawResponse(BaseModel):
    """Stands in for the SDK response model a real adapter holds in raw."""

    id: str = "fake-response"


class _FakeRawUsage(BaseModel):
    """Stands in for the SDK usage object a real adapter holds in usage_raw."""


_FAKE_RAW_USAGE = _FakeRawUsage()


def _success_result(content: str) -> AdapterResult[str]:
    """Build a successful text AdapterResult carrying the given content."""
    return AdapterResult(
        output=content,
        assistant_message=AssistantMessage(turn=(TextPart(text=content),)),
        usage=_USAGE,
        usage_raw=_FAKE_RAW_USAGE,
        stop_reason="end_turn",
        raw=_FakeRawResponse(),
    )


_FAKE_TOOL_CALL = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')


class _FakeStream(AdapterStream[str]):
    """A fixed item sequence and a fixed assembled result."""

    def __init__(self) -> None:
        """Start unclosed; close records that it ran."""
        self.closed = False

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield two text chunks, then one completed tool call.

        Yields:
            The scripted stream items in order.
        """
        yield "a"
        yield "b"
        yield _FAKE_TOOL_CALL

    @override
    async def final(self) -> AdapterResult[str]:
        """Return the assembled result the SDK would produce."""
        return AdapterResult(
            output="ab",
            assistant_message=AssistantMessage(turn=(TextPart(text="ab"),)),
            usage=_USAGE_STREAM,
            usage_raw=_FAKE_RAW_USAGE,
            stop_reason="end_turn",
            raw=_FakeRawResponse(id="fake-final"),
        )

    @override
    async def close(self) -> None:
        """Record that the connection was closed."""
        self.closed = True


class _RefusingStream(_FakeStream):
    """A stream that yields items normally but whose final() detects a structured refusal.

    Mirrors an adapter that parses the assembled message in AdapterStream.final() and finds a refusal,
    raising the bare leaf carrying only the rejected 200's billing.
    """

    @override
    async def final(self) -> AdapterResult[str]:
        """Raise the adapter-side refusal leaf instead of assembling a result.

        Raises:
            RefusalError: always, carrying this attempt's billing.
        """
        raise RefusalError.for_rejected_200(
            usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
        )


class _ProtocolErrorStream(_FakeStream):
    """A stream whose items() violates the stream contract immediately."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Raise StreamProtocolError before yielding anything.

        Yields:
            Nothing; the raise precedes the first yield.

        Raises:
            StreamProtocolError: always, before the first yield.
        """
        raise StreamProtocolError("stream ended without a stop event")
        yield "unreachable"


class _FailsBeforeFirstItemStream(_FakeStream):
    """A stream whose first items() call fails transiently before yielding, then behaves normally.

    One instance is reused across reopens, so the counter records which items() call is running.
    """

    def __init__(self) -> None:
        """Start with no items() call made."""
        super().__init__()
        self.items_calls = 0

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Fail before the first yield on the first call, else yield the base sequence.

        Yields:
            Nothing on the first call; the base class's items on every later call.

        Raises:
            TransientError: on the first call, before the first yield.
        """
        self.items_calls += 1
        if self.items_calls == 1:
            raise TransientError("dropped before the first item")
        async for item in super().items():
            yield item


class _HangingStream(_FakeStream):
    """A stream whose items() opens then suspends forever, to be cancelled mid-iteration."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Suspend on an event that never fires, before yielding anything.

        Yields:
            Nothing; the wait never returns.
        """
        await asyncio.Event().wait()
        yield "unreachable"


class _FakeBoundAdapter(BoundAdapter[str]):
    """A bound adapter whose send follows a scripted failure sequence."""

    def __init__(
        self,
        *,
        failures: Sequence[Exception] = (),
        open_failures: Sequence[Exception] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        send_seconds: float = 0.0,
        hang_from_open: int | None = None,
    ) -> None:
        """Store the failure scripts, echo mode, and the stream open_stream returns.

        failures scripts send; open_failures scripts open_stream, exercising the pre-first-item stream retry path.
        send_seconds > 0 makes each send suspend that long,
        so a batch overlaps and peak_in_flight records the concurrency it reached.
        hang_from_open is the 1-based open_stream call from which every open suspends forever,
        so a cancellation lands on the open itself rather than on a later item pull.
        """
        self._failures = list(failures)
        self._open_failures = list(open_failures)
        self._echo = echo
        self._send_seconds = send_seconds
        self._hang_from_open = hang_from_open
        self.stream = stream if stream is not None else _FakeStream()
        self.send_count = 0
        self.open_count = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    @override
    async def send(self, conversation: Sequence[Message]) -> AdapterResult[str]:
        """Raise the next scripted failure, else return a success result."""
        self.send_count += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._send_seconds:
                await asyncio.sleep(self._send_seconds)
            if self._failures:
                raise self._failures.pop(0)
            first = conversation[0]
            content = (
                first.content
                if self._echo and isinstance(first, UserMessage) and isinstance(first.content, str)
                else "ok"
            )
            return _success_result(content)
        finally:
            self.in_flight -= 1

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream[str]:
        """Count the attempt, suspend or raise the next scripted open failure, else return the stored fake stream."""
        self.open_count += 1
        if self._hang_from_open is not None and self.open_count >= self._hang_from_open:
            await asyncio.Event().wait()
        if self._open_failures:
            raise self._open_failures.pop(0)
        return self.stream


class _FakeStructuredBoundAdapter[ModelT: BaseModel](BoundAdapter[ModelT]):
    """A structured bound adapter for response_format rebind tests; it never generates.

    Those tests check binding identity and the switched content type, not structured output,
    so send and open_stream stay unreachable.
    """

    @override
    async def send(self, conversation: Sequence[Message]) -> AdapterResult[ModelT]:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream[ModelT]:
        """Unreachable: response_format rebind tests do not stream."""
        raise NotImplementedError


class _FakeAdapter(Adapter):
    """An adapter whose bind_text hands out fake bound adapters."""

    def __init__(
        self,
        *,
        failures: Sequence[Exception] = (),
        open_failures: Sequence[Exception] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        classify_result: ErrorClass = "abort",
        send_seconds: float = 0.0,
        hang_from_open: int | None = None,
    ) -> None:
        """Store how each freshly bound adapter behaves and the classify verdict."""
        # This adapter reaches no SDK, so it passes client=None, which matches no entry in the
        # base's empty provider_name_by_client_class, leaving the stated "fake" to stand.
        super().__init__(client=None, model="fake-model", pricing=_PRICING, provider_name="fake")
        self._failures = failures
        self._open_failures = open_failures
        self._echo = echo
        self._stream = stream
        self._classify_result = classify_result
        self._send_seconds = send_seconds
        self._hang_from_open = hang_from_open
        self.bound_adapters: list[_FakeBoundAdapter] = []
        self.structured_bind_count = 0

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Build a fresh fake bound adapter and record it."""
        bound = _FakeBoundAdapter(
            failures=self._failures,
            open_failures=self._open_failures,
            echo=self._echo,
            stream=self._stream,
            send_seconds=self._send_seconds,
            hang_from_open=self._hang_from_open,
        )
        self.bound_adapters.append(bound)
        return bound

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT]:
        """Build a structured bound adapter and count the call."""
        self.structured_bind_count += 1
        bound: BoundAdapter[ModelT] = _FakeStructuredBoundAdapter()
        return bound

    @override
    def classify(self, error: Exception) -> ErrorClass:
        """Return the fixed verdict for unrecognized exceptions."""
        return self._classify_result


def test_retry_recovers_after_a_transient_failure() -> None:
    """One transient failure then success yields a two-attempt success Response."""

    async def scenario() -> None:
        """Drive one generate_one through a single transient failure."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            system_prompt="s", automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 2
        assert response.model == "fake-model"
        assert response.provider_name == "fake"
        assert adapter.bound_adapters[0].send_count == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "boom"
        assert succeeded.error is None
        assert (
            failed.started_at_monotonic_seconds
            <= failed.ended_at_monotonic_seconds
            <= succeeded.started_at_monotonic_seconds
            <= succeeded.ended_at_monotonic_seconds
        )
        records_span = succeeded.ended_at_monotonic_seconds - failed.started_at_monotonic_seconds
        assert response.elapsed_seconds >= records_span

    asyncio.run(scenario())


def test_retry_exhaustion_raises_ordered_failure() -> None:
    """Exhausting the budget raises RetriesExhaustedError carrying the ordered errors."""

    async def scenario() -> None:
        """Drive one generate_one to exhaustion under a two-attempt budget."""
        adapter = _FakeAdapter(failures=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_attempts=2)).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(RetriesExhaustedError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        failure = exhausted.value
        assert [str(error) for error in failure.errors_from_attempts] == ["e1", "e2"]
        assert [str(record.error) for record in failure.attempt_records] == ["e1", "e2"]
        assert failure.error_text == "attempt 1: e1; attempt 2: e2"
        assert failure.attempts == 2
        assert failure.model == adapter.model
        assert failure.provider_name == adapter.provider_name

    asyncio.run(scenario())


def test_attempt_record_bracket_excludes_the_backoff_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failed record's own span stays small; the backoff shows up as the gap between records.

    The full-jitter draw is pinned to its ceiling so the backoff gap is deterministic here.
    """
    monkeypatch.setattr(rate_limiter_module.random, "uniform", lambda _low, high: high)

    async def scenario() -> None:
        """Recover from one failure under a visible 0.05s backoff."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        rate_limiter = RateLimiter(max_attempts=2, backoff_base_seconds=0.05)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        failed, succeeded = response.attempt_records
        assert failed.elapsed_seconds < 0.05
        backoff_gap = succeeded.started_at_monotonic_seconds - failed.ended_at_monotonic_seconds
        assert backoff_gap >= 0.05
        assert response.elapsed_seconds >= 0.05

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_abort_batch_error_raises_immediately_without_retry() -> None:
    """An AbortBatchError from send is raised on the first attempt and never retried.

    classify returns "transient" here, so retrying is what the classifier asks for:
    only the retry loop honoring AbortBatchError ahead of classification stops the second attempt.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send raises AbortBatchError under a transient classify verdict."""
        adapter = _FakeAdapter(failures=[AbortBatchError("nope")], classify_result="transient")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1

    asyncio.run(scenario())


def test_refusal_leaf_from_send_raises_enriched_without_retry() -> None:
    """A RefusalError from send is enriched with the attempt record and never retried."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises the adapter-side refusal leaf."""
        adapter = _FakeAdapter(
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
                )
            ]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(RefusalError) as refused:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = refused.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.usage.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_truncation_leaf_from_send_raises_enriched_without_retry() -> None:
    """A MaxCompletionTokensExceededError from send is enriched and never retried."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises the adapter-side truncation leaf."""
        adapter = _FakeAdapter(
            failures=[
                MaxCompletionTokensExceededError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="max_tokens"
                )
            ]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(MaxCompletionTokensExceededError) as truncated:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = truncated.value
        assert failure.attempts == 1
        assert failure.stop_reason == "max_tokens"
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_unrecognized_error_classified_transient_is_retried() -> None:
    """A plain exception classified transient is wrapped and retried to success."""

    async def scenario() -> None:
        """Drive one generate_one over two classify-transient failures."""
        adapter = _FakeAdapter(
            failures=[ValueError("x1"), ValueError("x2")], classify_result="transient"
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 3

    asyncio.run(scenario())


def test_unrecognized_error_classified_abort_raises() -> None:
    """A plain exception classified abort raises AbortBatchError on the first attempt."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-abort exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="abort")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1

    asyncio.run(scenario())


def test_rebind_unchanged_keeps_binding_equal_and_still_rebuilds() -> None:
    """An all-unchanged rebind keeps the binding equal but still builds a new bound adapter."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(system_prompt="s", automatic_prompt_caching=True)
    same = bound_llm.rebind()
    assert same.binding == bound_llm.binding
    assert same._bound_adapter is not bound_llm._bound_adapter
    assert len(adapter.bound_adapters) == 2


def test_rebind_changed_field_creates_new_binding_and_bound_adapter() -> None:
    """Changing a field produces a new Binding and a freshly bound adapter."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(system_prompt="s", automatic_prompt_caching=True)
    changed = bound_llm.rebind(system_prompt="s2")
    assert changed.binding != bound_llm.binding
    assert changed.binding.system_prompt == "s2"
    assert changed._bound_adapter is not bound_llm._bound_adapter
    assert len(adapter.bound_adapters) == 2


class _Answer(BaseModel):
    """A response_format model for the rebind content-type tests."""

    value: int


def test_rebind_to_a_response_format_rebinds_even_when_the_binding_is_unchanged() -> None:
    """response_format is not part of Binding, so a rebind that only changes it must still re-bind."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(system_prompt="s", automatic_prompt_caching=True)
    structured = bound_llm.rebind(response_format=_Answer)
    assert_type(structured, BoundLLM[_Answer])
    assert adapter.structured_bind_count == 1
    assert structured.binding == bound_llm.binding
    assert structured._bound_adapter is not bound_llm._bound_adapter


def test_rebind_leaving_response_format_out_keeps_the_content_type_and_rebuilds() -> None:
    """Omitting response_format keeps BoundLLM[str] and rebuilds through bind_text."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(system_prompt="s", automatic_prompt_caching=True)
    same = bound_llm.rebind()
    assert_type(same, BoundLLM[str])
    assert adapter.structured_bind_count == 0
    assert same._bound_adapter is not bound_llm._bound_adapter


def test_rebind_response_format_none_switches_structured_back_to_text() -> None:
    """From a structured binding, response_format=None returns BoundLLM[str] via bind_text."""
    adapter = _FakeAdapter()
    structured = LLM(adapter).bind(
        system_prompt="s", response_format=_Answer, automatic_prompt_caching=True
    )
    assert len(adapter.bound_adapters) == 0
    text = structured.rebind(response_format=None)
    assert_type(text, BoundLLM[str])
    assert len(adapter.bound_adapters) == 1
    assert text._bound_adapter is not structured._bound_adapter


def test_rebind_leaving_structured_response_format_out_rebuilds_through_bind_structured() -> None:
    """A prefix change with response_format left out rebuilds from the stored model type."""
    adapter = _FakeAdapter()
    structured = LLM(adapter).bind(
        system_prompt="s", response_format=_Answer, automatic_prompt_caching=True
    )
    assert adapter.structured_bind_count == 1
    rebound = structured.rebind(system_prompt="s2")
    assert_type(rebound, BoundLLM[_Answer])
    assert adapter.structured_bind_count == 2
    assert rebound._bound_adapter is not structured._bound_adapter


def test_response_format_is_a_public_field_bind_and_rebind_carry_it() -> None:
    """response_format is public inspectable state that bind sets and rebind carries and switches."""
    adapter = _FakeAdapter()
    assert LLM(adapter).bind(automatic_prompt_caching=True).response_format is None
    structured = LLM(adapter).bind(response_format=_Answer, automatic_prompt_caching=True)
    assert structured.response_format is _Answer
    assert structured.rebind(system_prompt="s2").response_format is _Answer
    assert structured.rebind(response_format=None).response_format is None


def test_unchanged_sentinel_reprs_as_its_name() -> None:
    """The sentinel renders as UNCHANGED so the rebind signature reads cleanly in help()."""
    assert repr(UNCHANGED) == "UNCHANGED"


def test_automatic_prompt_caching_participates_in_binding_equality() -> None:
    """The caching flag is part of Binding equality, so flipping it rebinds."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
    flipped = bound_llm.rebind(automatic_prompt_caching=False)
    assert flipped.binding != bound_llm.binding
    assert flipped._bound_adapter is not bound_llm._bound_adapter
    unchanged = bound_llm.rebind(automatic_prompt_caching=True)
    assert unchanged.binding == bound_llm.binding
    assert unchanged._bound_adapter is not bound_llm._bound_adapter


def test_generate_many_aligns_results_with_inputs() -> None:
    """Result i belongs to conversations[i], preserving input order."""

    async def scenario() -> None:
        """Run a two-item batch whose fake echoes each conversation's first turn."""
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        assert _batch_outputs(results) == ["a", "b"]

    asyncio.run(scenario())


def test_generate_many_returns_exhaustion_as_a_failure_row() -> None:
    """An item that exhausts its retries comes back as the RetriesExhaustedError, not a raise."""

    async def scenario() -> None:
        """Run a two-item batch whose every send fails transiently under a two-attempt budget."""
        adapter = _FakeAdapter(failures=[TransientError("x")] * 4)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_attempts=2)).bind(
            automatic_prompt_caching=True
        )
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        assert len(results) == 2
        for result in results:
            assert isinstance(result, RetriesExhaustedError)
            assert result.attempts == 2

    asyncio.run(scenario())


def test_generate_many_aligns_a_failure_among_successes() -> None:
    """A mixed batch keeps each result in its input slot: the failure where it failed, successes elsewhere."""

    async def scenario() -> None:
        """Serialize a three-item batch (max_in_flight=1) whose first send fails under a one-attempt budget.

        One slot runs the items in submission order,
        so the single scripted failure lands on the first item and the other two succeed,
        which is exactly the mixed-outcome alignment under test.
        """
        adapter = _FakeAdapter(echo=True, failures=[TransientError("x")])
        rate_limiter = _fast_rate_limiter(max_attempts=1, max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
            [UserMessage(content="c")],
        ])
        first, second, third = results
        assert isinstance(first, RetriesExhaustedError)
        assert isinstance(second, Response)
        assert second.output == "b"
        assert isinstance(third, Response)
        assert third.output == "c"

    asyncio.run(scenario())


def test_generate_many_returns_a_refusal_as_a_failure_row() -> None:
    """An item whose send refuses comes back as the RefusalError in its slot, siblings succeed."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send refuses."""
        adapter = _FakeAdapter(
            echo=True,
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
                )
            ],
        )
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, RefusalError)
        assert first.stop_reason == "refusal"
        assert first.usage.cost_in_usd == 0.25
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(scenario())


def test_generate_many_aborts_the_whole_batch_and_cancels_siblings() -> None:
    """An AbortBatchError in one item raises out of the batch instead of becoming a row."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send aborts."""
        adapter = _FakeAdapter(echo=True, failures=[AbortBatchError("misconfigured")])
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_many([[UserMessage(content="a")], [UserMessage(content="b")]])

    asyncio.run(scenario())


def test_generate_many_warm_cache_runs_the_first_item_alone_then_the_rest_together() -> None:
    """warm_cache completes conversations[0] before any sibling starts; the rest run at normal concurrency."""

    async def scenario() -> None:
        """Run an identical three-item batch on two fresh slow fakes and compare the recorded peaks.

        Warmed, the first send overlaps nothing and the remaining two overlap each other, so the peak is 2;
        the unwarmed control reaches 3, proving warm_cache alone changed the ordering.
        A fresh adapter per run keeps the two peaks independent readings.
        """
        conversations = [[UserMessage(content=str(index))] for index in range(3)]
        warmed_adapter = _FakeAdapter(echo=True, send_seconds=0.01)
        warmed_bound_llm = LLM(
            warmed_adapter, rate_limiter=_fast_rate_limiter(max_in_flight=8)
        ).bind(automatic_prompt_caching=True)
        warmed = await warmed_bound_llm.generate_many(conversations, warm_cache=True)
        assert _batch_outputs(warmed) == ["0", "1", "2"]
        assert warmed_adapter.bound_adapters[0].peak_in_flight == 2
        control_adapter = _FakeAdapter(echo=True, send_seconds=0.01)
        control_bound_llm = LLM(
            control_adapter, rate_limiter=_fast_rate_limiter(max_in_flight=8)
        ).bind(automatic_prompt_caching=True)
        control = await control_bound_llm.generate_many(conversations)
        assert _batch_outputs(control) == ["0", "1", "2"]
        assert control_adapter.bound_adapters[0].peak_in_flight == 3

    asyncio.run(scenario())


def test_generate_many_warm_cache_first_failure_still_admits_the_rest() -> None:
    """A first item ending in a GenerationError stays in its slot and the siblings still run."""

    async def scenario() -> None:
        """Fail the deterministic first send under a one-attempt budget; the other two succeed."""
        adapter = _FakeAdapter(echo=True, failures=[TransientError("x")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_attempts=1)).bind(
            automatic_prompt_caching=True
        )
        results = await bound_llm.generate_many(
            [[UserMessage(content="a")], [UserMessage(content="b")], [UserMessage(content="c")]],
            warm_cache=True,
        )
        first, second, third = results
        assert isinstance(first, RetriesExhaustedError)
        assert isinstance(second, Response)
        assert second.output == "b"
        assert isinstance(third, Response)
        assert third.output == "c"

    asyncio.run(scenario())


def test_generate_many_warm_cache_abort_on_the_first_item_starts_no_sibling() -> None:
    """An AbortBatchError from the warming item raises before any sibling sends."""

    async def scenario() -> None:
        """Abort the deterministic first send and count the sends that happened."""
        adapter = _FakeAdapter(echo=True, failures=[AbortBatchError("misconfigured")])
        bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_many(
                [[UserMessage(content="a")], [UserMessage(content="b")]], warm_cache=True
            )
        assert adapter.bound_adapters[0].send_count == 1

    asyncio.run(scenario())


def test_generate_many_warm_cache_empty_batch_returns_empty() -> None:
    """An empty batch under warm_cache returns [] instead of indexing a first item."""

    async def scenario() -> None:
        """Run the empty batch."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        assert await bound_llm.generate_many([], warm_cache=True) == []

    asyncio.run(scenario())


def test_bare_str_is_shorthand_for_one_user_message() -> None:
    """A bare str reaches the adapter as a conversation of one UserMessage."""

    async def scenario() -> None:
        """Drive each generate method with a bare str against the echo fake.

        The echo fake returns the first turn's content only when that turn is a UserMessage with str content,
        so an echoed value proves the coercion built a real UserMessage.
        """
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
        response = await bound_llm.generate_one("hi")
        assert response.output == "hi"
        results = await bound_llm.generate_many(["a", [UserMessage(content="b")]])
        assert _batch_outputs(results) == ["a", "b"]

    asyncio.run(scenario())


def test_generate_many_rejects_a_bare_str_batch() -> None:
    """A bare str as the whole batch raises instead of running per-character requests."""

    async def scenario() -> None:
        """Pass a bare str where generate_many expects the batch.

        The suppressed pyrefly error is SequenceNotStr statically rejecting the bare str;
        the suppression doubles as a canary, since pyrefly reports it as unused
        if typeshed drift ever makes str satisfy SequenceNotStr and the static rejection lapses.
        """
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
        with pytest.raises(TypeError, match="bare str"):
            # pyrefly: ignore[bad-argument-type]
            await bound_llm.generate_many("hi")
        assert adapter.bound_adapters[0].send_count == 0

    asyncio.run(scenario())


def test_stream_one_accepts_a_bare_str() -> None:
    """stream_one coerces a bare str to a conversation of one UserMessage."""

    async def scenario() -> None:
        """Build a handle from a bare str and check the stored conversation."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one("hi") as handle:
            assert handle._conversation == (UserMessage(content="hi"),)
            response = await handle.final()
        assert response.output == "ab"

    asyncio.run(scenario())


def test_stream_cancelled_mid_iteration_releases_the_slot() -> None:
    """A cancelled item pull returns its slot without waiting for the block to exit."""

    async def scenario() -> None:
        """Cancel a suspended item pull inside the block, then prove the slot is free."""
        adapter = _FakeAdapter(stream=_HangingStream())
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            consumer = asyncio.create_task(anext(handle))
            await asyncio.sleep(0.01)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            # Still inside the block, so only the cancellation can have freed the one in-flight slot.
            admission = await asyncio.wait_for(rate_limiter.acquire(), timeout=1.0)
            rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_during_the_open_releases_the_slot() -> None:
    """A cancellation while the open is in flight returns its slot.

    __aexit__ never runs when __aenter__ raises, so only __aenter__ itself can free the admission here.
    """

    async def scenario() -> None:
        """Time out an entry whose open_stream never returns, then prove the slot is free."""
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(_FakeAdapter(hang_from_open=1), rate_limiter=rate_limiter).bind(
            automatic_prompt_caching=True
        )

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns; the wait_for below cancels this."""
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        admission = await asyncio.wait_for(rate_limiter.acquire(), timeout=1.0)
        rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_during_a_reopen_releases_the_slot() -> None:
    """A cancellation while the pre-first-item retry is reopening returns its slot.

    The reopen runs inside __anext__'s transient-failure handler, which no sibling except clause covers,
    so the release here is __aexit__'s: the block is still open, unlike a cancellation inside __aenter__.
    """

    async def scenario() -> None:
        """Time out an iteration whose second open never returns, then prove the slot is free."""
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream(), hang_from_open=2)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)

        async def drain() -> None:
            """Enter and iterate; the first items() fails, so the retry reopens into the hang."""
            async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)
        admission = await asyncio.wait_for(rate_limiter.acquire(), timeout=1.0)
        rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_sends_one_request_when_final_follows_the_block() -> None:
    """final() after the block raises instead of opening a second billed request."""

    async def scenario() -> None:
        """Drain a stream inside the block, then call final() after it."""
        adapter = _FakeAdapter()
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])
        async with handle:
            async for _item in handle:
                pass
        with pytest.raises(RuntimeError, match="finished"):
            await handle.final()
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_stream_unentered_handle_refuses_to_open() -> None:
    """Iterating or draining a handle that was never entered raises rather than opening a request."""

    async def scenario() -> None:
        """Use a handle straight from stream_one, without async with."""
        adapter = _FakeAdapter()
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])
        with pytest.raises(RuntimeError, match="async with"):
            await anext(handle)
        with pytest.raises(RuntimeError, match="async with"):
            await handle.final()
        assert adapter.bound_adapters[0].open_count == 0

    asyncio.run(scenario())


def test_stream_handle_refuses_a_second_entry() -> None:
    """Re-entering a spent handle raises rather than opening a second request."""

    async def scenario() -> None:
        """Enter, leave, then enter the same handle again."""
        adapter = _FakeAdapter()
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])
        async with handle:
            pass
        with pytest.raises(RuntimeError, match="already entered"):
            async with handle:
                pass
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_stream_passes_items_through_and_assembles_final() -> None:
    """Iterating yields the text chunks and final() assembles the Response fields."""

    async def scenario() -> None:
        """Iterate the stream fully, then read final()."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            texts = [item async for item in handle if isinstance(item, str)]
            response = await handle.final()
        assert "".join(texts) == "ab"
        assert response.output == "ab"
        assert response.stop_reason == "end_turn"
        assert response.model == "fake-model"
        assert response.provider_name == "fake"
        assert response.attempts == 1
        (record,) = response.attempt_records
        assert record.error is None
        assert record.started_at_monotonic_seconds <= record.ended_at_monotonic_seconds

    asyncio.run(scenario())


def test_stream_final_refusal_raises_enriched_without_retry() -> None:
    """A structured refusal detected in the stream's final() surfaces as the enriched RefusalError.

    The stream already yielded items to the caller, so the leaf is not retried;
    final() records the one rejected 200 and re-raises the leaf enriched with the handle's attempt records and billing.
    """

    async def scenario() -> None:
        """Drain a stream whose final() refuses, then read the enriched leaf."""
        adapter = _FakeAdapter(stream=_RefusingStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RefusalError) as refused:
                await handle.final()
        failure = refused.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.usage.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_stream_retry_populates_attempt_records() -> None:
    """A pre-first-item connection failure lands as an errored record before the success record."""

    async def scenario() -> None:
        """Open a stream whose first open_stream call fails, then drain it."""
        adapter = _FakeAdapter(open_failures=[TransientError("conn refused")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        assert response.output == "ab"
        assert response.attempts == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "conn refused"
        assert succeeded.error is None
        assert (
            failed.started_at_monotonic_seconds
            <= failed.ended_at_monotonic_seconds
            <= succeeded.started_at_monotonic_seconds
            <= succeeded.ended_at_monotonic_seconds
        )

    asyncio.run(scenario())


def test_stream_open_raising_a_generation_leaf_propagates_it_from_the_entry() -> None:
    """A GenerationError leaf raised by open_stream reaches the caller from the async with, unretried."""

    async def scenario() -> None:
        """Enter a handle whose open_stream raises a refusal leaf."""
        leaf = RefusalError.for_rejected_200(
            usage=_USAGE_BILLED, usage_raw=_FAKE_RAW_USAGE, stop_reason="refusal"
        )
        adapter = _FakeAdapter(open_failures=[leaf])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(RefusalError):
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_stream_item_failure_before_the_first_item_reopens_and_retries() -> None:
    """A transient failure from items() before any item reopens the stream and records both attempts."""

    async def scenario() -> None:
        """Drain a stream whose first items() call fails before yielding."""
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            collected_items = [item async for item in handle]
            response = await handle.final()
        assert collected_items == ["a", "b", _FAKE_TOOL_CALL]
        assert adapter.bound_adapters[0].open_count == 2
        assert response.attempts == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "dropped before the first item"
        assert succeeded.error is None

    asyncio.run(scenario())


def test_stream_open_exhaustion_raises_retries_exhausted() -> None:
    """Opens that keep failing past the budget raise RetriesExhaustedError with the table fields set."""

    async def scenario() -> None:
        """Open a stream under a two-attempt budget whose every open_stream fails transiently."""
        adapter = _FakeAdapter(open_failures=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_attempts=2)).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(RetriesExhaustedError) as exhausted:
            async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
                await handle.final()
        failure = exhausted.value
        assert failure.attempts == 2
        assert [str(error) for error in failure.errors_from_attempts] == ["e1", "e2"]
        assert failure.model == "fake-model"
        assert failure.provider_name == "fake"
        assert failure.elapsed_seconds >= 0.0

    asyncio.run(scenario())


def test_stream_record_and_elapsed_end_at_exhaustion_not_at_final() -> None:
    """Idle time between draining the stream and calling final() lands in neither measurement."""

    async def scenario() -> None:
        """Drain the stream, idle, then call final()."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                pass
            drained_at = time.monotonic()
            await asyncio.sleep(0.02)
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.ended_at_monotonic_seconds <= drained_at
        assert response.elapsed_seconds == record.elapsed_seconds

    asyncio.run(scenario())


def test_stream_final_is_idempotent() -> None:
    """A second final() returns the same cached Response object."""

    async def scenario() -> None:
        """Call final() twice on one drained stream."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            first: Response[str] = await handle.final()
            second: Response[str] = await handle.final()
        assert first is second

    asyncio.run(scenario())


def test_stream_yields_items_in_order_with_complete_tool_call() -> None:
    """Text chunks arrive as bare strings and the tool call arrives once, complete."""

    async def scenario() -> None:
        """Collect every item the stream yields."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            collected_items = [item async for item in handle]
        assert collected_items == ["a", "b", _FAKE_TOOL_CALL]

    asyncio.run(scenario())


def test_stream_closes_on_context_exit() -> None:
    """Leaving the async with block closes the underlying adapter stream."""

    async def scenario() -> None:
        """Open the stream, consume one item, then leave the context."""
        stream = _FakeStream()
        bound_llm = LLM(_FakeAdapter(stream=stream)).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                break
        assert stream.closed is True

    asyncio.run(scenario())


def test_server_stated_retry_after_overrides_exponential_backoff() -> None:
    """A tiny retry_after_seconds beats a backoff base that would stall the test."""

    async def scenario() -> None:
        """Recover from one rate-limit failure whose server-stated wait is near zero."""
        adapter = _FakeAdapter(
            failures=[TransientError("rate limited", retry_after_seconds=0.001)]
        )
        rate_limiter = RateLimiter(max_attempts=2, backoff_base_seconds=30.0)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_max_in_flight_bounds_batch_concurrency() -> None:
    """A five-item batch under max_in_flight=2 never overlaps more than two sends."""

    async def scenario() -> None:
        """Run the batch on a slow fake and read the recorded peak."""
        adapter = _FakeAdapter(echo=True, send_seconds=0.01)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_in_flight=2)).bind(
            automatic_prompt_caching=True
        )
        conversations = [[UserMessage(content=str(index))] for index in range(5)]
        results = await bound_llm.generate_many(conversations)
        assert _batch_outputs(results) == ["0", "1", "2", "3", "4"]
        assert adapter.bound_adapters[0].peak_in_flight == 2

    asyncio.run(scenario())


def test_backoff_sleep_does_not_hold_the_in_flight_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With max_in_flight=1, a task backing off lets another request run.

    The failure carries no retry_after_seconds, so nothing pauses admission;
    only a held slot could delay the second request.
    What pins the release is the second request's own duration: it is admitted while the first backs off,
    so it finishes in far less than the backoff it would otherwise queue behind.
    first_task.done() cannot pin this alone: a slot held across the sleep passes to the waiting second request
    the moment the first retries, so the first is unfinished under either placement.
    The full-jitter draw is pinned to its ceiling so the backoff outlasts the second request deterministically.
    """
    monkeypatch.setattr(rate_limiter_module.random, "uniform", lambda _low, high: high)

    async def scenario() -> None:
        """Interleave a retrying item with a clean one under one slot."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        backoff_base_seconds = 0.2
        rate_limiter = RateLimiter(
            max_attempts=2, backoff_base_seconds=backoff_base_seconds, max_in_flight=1
        )
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        first_task = asyncio.create_task(bound_llm.generate_one([UserMessage(content="a")]))
        await asyncio.sleep(0.01)
        started_at = time.monotonic()
        second = await bound_llm.generate_one([UserMessage(content="b")])
        second_elapsed_seconds = time.monotonic() - started_at
        assert second.output == "ok"
        assert second_elapsed_seconds < backoff_base_seconds / 2
        assert not first_task.done()
        first = await first_task
        assert first.output == "ok"
        assert first.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_protocol_error_releases_the_slot() -> None:
    """A StreamProtocolError from items() returns the slot and closes the stream."""

    async def scenario() -> None:
        """Drive final() into the protocol error, then acquire the slot inside the still-open block."""
        stream = _ProtocolErrorStream()
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(_FakeAdapter(stream=stream), rate_limiter=rate_limiter).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(StreamProtocolError):
                await handle.final()
            assert stream.closed is True
            async with rate_limiter.slot():
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_releases_its_slot_when_exhausted() -> None:
    """Exhausting a stream returns its RateLimiter slot before the handle's block exits.

    The acquire sits inside the still-open block, so only the release on exhaustion can satisfy it;
    acquiring after the block would be satisfied by the release on block exit instead.
    """

    async def scenario() -> None:
        """Drain one stream under max_in_flight=1, then acquire the slot inside the still-open block."""
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(_FakeAdapter(), rate_limiter=rate_limiter).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                pass
            async with rate_limiter.slot():
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_rate_limited_attempt_then_success_ends_the_recovery() -> None:
    """The retry loop registers the rate-limit failure and the success, so admission fully reopens."""

    async def scenario() -> None:
        """Recover one generate_one from a rate-limit error, then confirm admission fully reopened."""
        adapter = _FakeAdapter(
            failures=[
                TransientError("rate limited", retry_after_seconds=0.001, is_rate_limit=True)
            ]
        )
        rate_limiter = _fast_rate_limiter(max_attempts=2)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 2
        # Admission fully reopened only if two concurrent acquires are both admitted; probe-only recovery admits one,
        # so a skipped register_success fails here.
        acquires = [asyncio.create_task(rate_limiter.acquire()) for _ in range(2)]
        await asyncio.sleep(0.01)
        assert sum(task.done() for task in acquires) == 2
        for admission in await asyncio.gather(*acquires):
            rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_open_rate_limit_registers_and_recovery_ends_at_open() -> None:
    """A rate-limited open_stream pauses admission; the successful open ends the recovery.

    Recovery must be over before any item is pulled: the open is a completed request that already cleared the quota,
    so a stream slow to first token must not keep holding the probe.
    """

    async def scenario() -> None:
        """Retry a stream open through a rate-limit error, then confirm admission reopened at open."""
        adapter = _FakeAdapter(
            open_failures=[
                TransientError("rate limited", retry_after_seconds=0.001, is_rate_limit=True)
            ]
        )
        rate_limiter = _fast_rate_limiter()
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            # Entering opened the stream and no item has been pulled yet,
            # so the open alone must have ended recovery.
            assert handle._yielded_any is False
            # Admission fully reopened only if two concurrent acquires are both admitted;
            # probe-only recovery admits one, so a register_success not fired at open fails here.
            acquires = [asyncio.create_task(rate_limiter.acquire()) for _ in range(2)]
            await asyncio.sleep(0.01)
            assert sum(task.done() for task in acquires) == 2
            for admission in await asyncio.gather(*acquires):
                rate_limiter.release(admission)
            response = await handle.final()
        assert response.output == "ab"
        assert response.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_bind_coerces_system_prompt_parts_to_a_tuple() -> None:
    """A list of system parts freezes to a tuple on the binding; a str passes through."""
    parts = [TextPart(text="stable", cache_breakpoint=True), TextPart(text="context")]
    bound_llm = LLM(_FakeAdapter()).bind(system_prompt=parts, automatic_prompt_caching=True)
    assert bound_llm.binding.system_prompt == tuple(parts)
    assert isinstance(bound_llm.binding.system_prompt, tuple)


def test_bind_rejects_an_empty_system_prompt_parts_sequence() -> None:
    """Empty parts are a configuration error; None is the way to bind no system prompt."""
    with pytest.raises(ValueError, match="empty"):
        LLM(_FakeAdapter()).bind(system_prompt=[], automatic_prompt_caching=True)
