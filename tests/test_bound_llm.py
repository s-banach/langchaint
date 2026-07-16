"""BoundLLM and StreamHandle driven by fake providers.

A fake BoundProvider scripts send to fail a fixed number of times before succeeding,
and a fake ProviderStream emits a fixed item sequence.
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
    ExceededMaxCompletionTokensError,
    GenerationError,
    Message,
    PricingTable,
    RateLimiter,
    RefusalError,
    Response,
    RetriesExhaustedError,
    StreamItem,
    StreamProtocolError,
    ToolCall,
    TransientError,
    Usage,
    UserMessage,
)
from langchaint import rate_limiter as rate_limiter_module
from langchaint.llm import UNCHANGED
from langchaint.provider import (
    Binding,
    BoundProvider,
    ErrorClass,
    Provider,
    ProviderResult,
    ProviderStream,
)

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
)


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


def _success_result(content: str) -> ProviderResult[str]:
    """Build a successful text ProviderResult carrying the given content."""
    return ProviderResult(
        output=content,
        assistant_message=AssistantMessage(content=content),
        usage=_USAGE,
        cost_in_usd=0.0,
        stop_reason="end_turn",
        raw=_FakeRawResponse(),
    )


_FAKE_TOOL_CALL = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')


class _FakeStream(ProviderStream[str]):
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
    async def final(self) -> ProviderResult[str]:
        """Return the assembled result the SDK would produce."""
        return ProviderResult(
            output="ab",
            assistant_message=AssistantMessage(content="ab"),
            usage=_USAGE,
            cost_in_usd=0.001,
            stop_reason="end_turn",
            raw=_FakeRawResponse(id="fake-final"),
        )

    @override
    async def close(self) -> None:
        """Record that the connection was closed."""
        self.closed = True


class _RefusingStream(_FakeStream):
    """A stream that yields items normally but whose final() detects a structured refusal.

    Mirrors an adapter that parses the assembled message in ProviderStream.final() and finds a refusal,
    raising the bare leaf carrying only the rejected 200's billing.
    """

    @override
    async def final(self) -> ProviderResult[str]:
        """Raise the adapter-side refusal leaf instead of assembling a result.

        Raises:
            RefusalError: always, carrying this attempt's billing.
        """
        raise RefusalError.for_rejected_200(
            usage=_USAGE, cost_in_usd=0.25, stop_reason="refusal"
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


class _FakeBoundProvider(BoundProvider[str]):
    """A bound provider whose send follows a scripted failure sequence."""

    def __init__(
        self,
        *,
        failures: Sequence[Exception] = (),
        open_failures: Sequence[Exception] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        send_seconds: float = 0.0,
    ) -> None:
        """Store the failure scripts, echo mode, and the stream open_stream returns.

        failures scripts send; open_failures scripts open_stream, exercising the pre-first-item stream retry path.
        send_seconds > 0 makes each send suspend that long,
        so a batch overlaps and peak_in_flight records the concurrency it reached.
        """
        self._failures = list(failures)
        self._open_failures = list(open_failures)
        self._echo = echo
        self._send_seconds = send_seconds
        self.stream = stream if stream is not None else _FakeStream()
        self.send_count = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[str]:
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
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[str]:
        """Raise the next scripted open failure, else return the stored fake stream."""
        if self._open_failures:
            raise self._open_failures.pop(0)
        return self.stream


class _FakeStructuredBoundProvider[ModelT: BaseModel](BoundProvider[ModelT]):
    """A structured bound provider for response_format rebind tests; it never generates.

    Those tests check binding identity and the switched content type, not structured output,
    so send and open_stream stay unreachable.
    """

    @override
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[ModelT]:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[ModelT]:
        """Unreachable: response_format rebind tests do not stream."""
        raise NotImplementedError


class _FakeProvider(Provider):
    """A provider whose bind_text hands out fake bound providers."""

    name = "fake"

    def __init__(
        self,
        *,
        failures: Sequence[Exception] = (),
        open_failures: Sequence[Exception] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        classify_result: ErrorClass = "abort",
        send_seconds: float = 0.0,
    ) -> None:
        """Store how each freshly bound provider behaves and the classify verdict."""
        super().__init__(model="fake-model", pricing=_PRICING)
        self._failures = failures
        self._open_failures = open_failures
        self._echo = echo
        self._stream = stream
        self._classify_result = classify_result
        self._send_seconds = send_seconds
        self.bound_providers: list[_FakeBoundProvider] = []
        self.structured_bind_count = 0

    @override
    def bind_text(self, binding: Binding) -> BoundProvider[str]:
        """Build a fresh fake bound provider and record it."""
        bound = _FakeBoundProvider(
            failures=self._failures,
            open_failures=self._open_failures,
            echo=self._echo,
            stream=self._stream,
            send_seconds=self._send_seconds,
        )
        self.bound_providers.append(bound)
        return bound

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundProvider[ModelT]:
        """Build a structured bound provider and count the call."""
        self.structured_bind_count += 1
        bound: BoundProvider[ModelT] = _FakeStructuredBoundProvider()
        return bound

    @override
    def classify(self, error: Exception) -> ErrorClass:
        """Return the fixed verdict for unrecognized exceptions."""
        return self._classify_result


def test_retry_recovers_after_a_transient_failure() -> None:
    """One transient failure then success yields a two-attempt success Response."""

    async def scenario() -> None:
        """Drive one generate_one through a single transient failure."""
        provider = _FakeProvider(failures=[TransientError("boom")])
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind(system_prompt="s")
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 2
        assert response.model == "fake-model"
        assert response.provider_name == "fake"
        assert provider.bound_providers[0].send_count == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "boom"
        assert succeeded.error is None
        assert (
            failed.started_at_monotonic_seconds
            <= failed.ended_at_monotonic_seconds
            <= succeeded.started_at_monotonic_seconds
            <= succeeded.ended_at_monotonic_seconds
        )
        records_span = (
            succeeded.ended_at_monotonic_seconds - failed.started_at_monotonic_seconds
        )
        assert response.elapsed_seconds >= records_span

    asyncio.run(scenario())


def test_retry_exhaustion_raises_ordered_failure() -> None:
    """Exhausting the budget raises RetriesExhaustedError carrying the ordered errors."""

    async def scenario() -> None:
        """Drive one generate_one to exhaustion under a two-attempt budget."""
        provider = _FakeProvider(failures=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter(max_attempts=2)).bind()
        with pytest.raises(RetriesExhaustedError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        failure = exhausted.value
        assert [str(error) for error in failure.errors_from_attempts] == ["e1", "e2"]
        assert [str(record.error) for record in failure.attempt_records] == ["e1", "e2"]
        assert failure.error_text == "attempt 1: e1; attempt 2: e2"
        assert failure.attempts == 2
        assert failure.model == provider.model
        assert failure.provider_name == provider.name

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
        provider = _FakeProvider(failures=[TransientError("boom")])
        rate_limiter = RateLimiter(max_attempts=2, backoff_base_seconds=0.05)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        failed, succeeded = response.attempt_records
        assert failed.elapsed_seconds < 0.05
        backoff_gap = (
            succeeded.started_at_monotonic_seconds - failed.ended_at_monotonic_seconds
        )
        assert backoff_gap >= 0.05
        assert response.elapsed_seconds >= 0.05

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_abort_batch_error_raises_immediately_without_retry() -> None:
    """An AbortBatchError from send is raised on the first attempt and never retried."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises AbortBatchError."""
        provider = _FakeProvider(failures=[AbortBatchError("nope")])
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert provider.bound_providers[0].send_count == 1

    asyncio.run(scenario())


def test_refusal_leaf_from_send_raises_enriched_without_retry() -> None:
    """A RefusalError from send is enriched with the attempt record and never retried."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises the adapter-side refusal leaf."""
        provider = _FakeProvider(
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE, cost_in_usd=0.25, stop_reason="refusal"
                )
            ]
        )
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
        with pytest.raises(RefusalError) as refused:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert provider.bound_providers[0].send_count == 1
        failure = refused.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_truncation_leaf_from_send_raises_enriched_without_retry() -> None:
    """An ExceededMaxCompletionTokensError from send is enriched and never retried."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises the adapter-side truncation leaf."""
        provider = _FakeProvider(
            failures=[
                ExceededMaxCompletionTokensError.for_rejected_200(
                    usage=_USAGE, cost_in_usd=0.25, stop_reason="max_tokens"
                )
            ]
        )
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
        with pytest.raises(ExceededMaxCompletionTokensError) as truncated:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert provider.bound_providers[0].send_count == 1
        failure = truncated.value
        assert failure.attempts == 1
        assert failure.stop_reason == "max_tokens"
        assert failure.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_unrecognized_error_classified_transient_is_retried() -> None:
    """A plain exception classified transient is wrapped and retried to success."""

    async def scenario() -> None:
        """Drive one generate_one over two classify-transient failures."""
        provider = _FakeProvider(
            failures=[ValueError("x1"), ValueError("x2")], classify_result="transient"
        )
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 3

    asyncio.run(scenario())


def test_unrecognized_error_classified_abort_raises() -> None:
    """A plain exception classified abort raises AbortBatchError on the first attempt."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-abort exception."""
        provider = _FakeProvider(failures=[ValueError("boom")], classify_result="abort")
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert provider.bound_providers[0].send_count == 1

    asyncio.run(scenario())


def test_rebind_unchanged_keeps_binding_equal_and_still_rebuilds() -> None:
    """An all-unchanged rebind keeps the binding equal but still builds a new bound provider."""
    provider = _FakeProvider()
    bound_llm = LLM(provider).bind(system_prompt="s")
    same = bound_llm.rebind()
    assert same.binding == bound_llm.binding
    assert same._bound_provider is not bound_llm._bound_provider
    assert len(provider.bound_providers) == 2


def test_rebind_changed_field_creates_new_binding_and_bound_provider() -> None:
    """Changing a field produces a new Binding and a freshly bound provider."""
    provider = _FakeProvider()
    bound_llm = LLM(provider).bind(system_prompt="s")
    changed = bound_llm.rebind(system_prompt="s2")
    assert changed.binding != bound_llm.binding
    assert changed.binding.system_prompt == "s2"
    assert changed._bound_provider is not bound_llm._bound_provider
    assert len(provider.bound_providers) == 2


class _Answer(BaseModel):
    """A response_format model for the rebind content-type tests."""

    value: int


def test_rebind_to_a_response_format_rebinds_even_when_the_binding_is_unchanged() -> None:
    """response_format is not part of Binding, so a rebind that only changes it must still re-bind."""
    provider = _FakeProvider()
    bound_llm = LLM(provider).bind(system_prompt="s")
    structured = bound_llm.rebind(response_format=_Answer)
    assert_type(structured, BoundLLM[_Answer])
    assert provider.structured_bind_count == 1
    assert structured.binding == bound_llm.binding
    assert structured._bound_provider is not bound_llm._bound_provider


def test_rebind_leaving_response_format_out_keeps_the_content_type_and_rebuilds() -> None:
    """Omitting response_format keeps BoundLLM[str] and rebuilds through bind_text."""
    provider = _FakeProvider()
    bound_llm = LLM(provider).bind(system_prompt="s")
    same = bound_llm.rebind()
    assert_type(same, BoundLLM[str])
    assert provider.structured_bind_count == 0
    assert same._bound_provider is not bound_llm._bound_provider


def test_rebind_response_format_none_switches_structured_back_to_text() -> None:
    """From a structured binding, response_format=None returns BoundLLM[str] via bind_text."""
    provider = _FakeProvider()
    structured = LLM(provider).bind(system_prompt="s", response_format=_Answer)
    assert len(provider.bound_providers) == 0
    text = structured.rebind(response_format=None)
    assert_type(text, BoundLLM[str])
    assert len(provider.bound_providers) == 1
    assert text._bound_provider is not structured._bound_provider


def test_rebind_leaving_structured_response_format_out_rebuilds_through_bind_structured() -> None:
    """A prefix change with response_format left out rebuilds from the stored model type."""
    provider = _FakeProvider()
    structured = LLM(provider).bind(system_prompt="s", response_format=_Answer)
    assert provider.structured_bind_count == 1
    rebound = structured.rebind(system_prompt="s2")
    assert_type(rebound, BoundLLM[_Answer])
    assert provider.structured_bind_count == 2
    assert rebound._bound_provider is not structured._bound_provider


def test_response_format_is_a_public_field_bind_and_rebind_carry_it() -> None:
    """response_format is public inspectable state that bind sets and rebind carries and switches."""
    provider = _FakeProvider()
    assert LLM(provider).bind().response_format is None
    structured = LLM(provider).bind(response_format=_Answer)
    assert structured.response_format is _Answer
    assert structured.rebind(system_prompt="s2").response_format is _Answer
    assert structured.rebind(response_format=None).response_format is None


def test_unchanged_sentinel_reprs_as_its_name() -> None:
    """The sentinel renders as UNCHANGED so the rebind signature reads cleanly in help()."""
    assert repr(UNCHANGED) == "UNCHANGED"


def test_automatic_prompt_caching_participates_in_binding_equality() -> None:
    """The caching flag is part of Binding equality, so flipping it rebinds."""
    provider = _FakeProvider()
    bound_llm = LLM(provider).bind(automatic_prompt_caching=True)
    flipped = bound_llm.rebind(automatic_prompt_caching=False)
    assert flipped.binding != bound_llm.binding
    assert flipped._bound_provider is not bound_llm._bound_provider
    unchanged = bound_llm.rebind(automatic_prompt_caching=True)
    assert unchanged.binding == bound_llm.binding
    assert unchanged._bound_provider is not bound_llm._bound_provider


def test_generate_many_aligns_results_with_inputs() -> None:
    """Result i belongs to conversations[i], preserving input order."""

    async def scenario() -> None:
        """Run a two-item batch whose fake echoes each conversation's first turn."""
        provider = _FakeProvider(echo=True)
        bound_llm = LLM(provider).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="a")], [UserMessage(content="b")]]
        )
        assert _batch_outputs(results) == ["a", "b"]

    asyncio.run(scenario())


def test_generate_many_returns_exhaustion_as_a_failure_row() -> None:
    """An item that exhausts its retries comes back as the RetriesExhaustedError, not a raise."""

    async def scenario() -> None:
        """Run a two-item batch whose every send fails transiently under a two-attempt budget."""
        provider = _FakeProvider(failures=[TransientError("x")] * 4)
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter(max_attempts=2)).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="a")], [UserMessage(content="b")]]
        )
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
        provider = _FakeProvider(echo=True, failures=[TransientError("x")])
        rate_limiter = _fast_rate_limiter(max_attempts=1, max_in_flight=1)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="a")], [UserMessage(content="b")], [UserMessage(content="c")]]
        )
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
        provider = _FakeProvider(
            echo=True,
            failures=[
                RefusalError.for_rejected_200(
                    usage=_USAGE, cost_in_usd=0.25, stop_reason="refusal"
                )
            ],
        )
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="a")], [UserMessage(content="b")]]
        )
        first, second = results
        assert isinstance(first, RefusalError)
        assert first.stop_reason == "refusal"
        assert first.cost_in_usd == 0.25
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(scenario())


def test_generate_many_aborts_the_whole_batch_and_cancels_siblings() -> None:
    """An AbortBatchError in one item raises out of the batch instead of becoming a row."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send aborts."""
        provider = _FakeProvider(echo=True, failures=[AbortBatchError("misconfigured")])
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        with pytest.raises(AbortBatchError):
            await bound_llm.generate_many(
                [[UserMessage(content="a")], [UserMessage(content="b")]]
            )

    asyncio.run(scenario())


def test_bare_str_is_shorthand_for_one_user_message() -> None:
    """A bare str reaches the provider as a conversation of one UserMessage."""

    async def scenario() -> None:
        """Drive each generate method with a bare str against the echo fake.

        The echo fake returns the first turn's content only when that turn is a UserMessage with str content,
        so an echoed value proves the coercion built a real UserMessage.
        """
        provider = _FakeProvider(echo=True)
        bound_llm = LLM(provider).bind()
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
        provider = _FakeProvider(echo=True)
        bound_llm = LLM(provider).bind()
        with pytest.raises(TypeError, match="bare str"):
            # pyrefly: ignore[bad-argument-type]
            await bound_llm.generate_many("hi")
        assert provider.bound_providers[0].send_count == 0

    asyncio.run(scenario())


def test_stream_one_accepts_a_bare_str() -> None:
    """stream_one coerces a bare str to a conversation of one UserMessage."""

    async def scenario() -> None:
        """Build a handle from a bare str and check the stored conversation."""
        bound_llm = LLM(_FakeProvider()).bind()
        async with bound_llm.stream_one("hi") as handle:
            assert handle._conversation == (UserMessage(content="hi"),)
            response = await handle.final()
        assert response.output == "ab"

    asyncio.run(scenario())


def test_stream_cancelled_mid_iteration_releases_the_slot() -> None:
    """A StreamHandle iterated without async with and cancelled mid-item returns its slot."""

    async def scenario() -> None:
        """Open a hanging stream, cancel the suspended item pull, then prove the slot is free."""
        provider = _FakeProvider(stream=_HangingStream())
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])
        consumer = asyncio.create_task(anext(handle))
        await asyncio.sleep(0.01)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        # The one in-flight slot must be free again; a slot leaked by the cancellation would block this.
        admission = await asyncio.wait_for(rate_limiter.acquire(), timeout=1.0)
        rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_passes_items_through_and_assembles_final() -> None:
    """Iterating yields the text chunks and final() assembles the Response fields."""

    async def scenario() -> None:
        """Iterate the stream fully, then read final()."""
        bound_llm = LLM(_FakeProvider()).bind()
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
        provider = _FakeProvider(stream=_RefusingStream())
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RefusalError) as refused:
                await handle.final()
        failure = refused.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_stream_retry_populates_attempt_records() -> None:
    """A pre-first-item connection failure lands as an errored record before the success record."""

    async def scenario() -> None:
        """Open a stream whose first open_stream call fails, then drain it."""
        provider = _FakeProvider(open_failures=[TransientError("conn refused")])
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter()).bind()
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


def test_stream_open_exhaustion_raises_retries_exhausted() -> None:
    """Opens that keep failing past the budget raise RetriesExhaustedError with the table fields set."""

    async def scenario() -> None:
        """Open a stream under a two-attempt budget whose every open_stream fails transiently."""
        provider = _FakeProvider(open_failures=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter(max_attempts=2)).bind()
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
        bound_llm = LLM(_FakeProvider()).bind()
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
        bound_llm = LLM(_FakeProvider()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            first: Response[str] = await handle.final()
            second: Response[str] = await handle.final()
        assert first is second

    asyncio.run(scenario())


def test_stream_yields_items_in_order_with_complete_tool_call() -> None:
    """Text chunks arrive as bare strings and the tool call arrives once, complete."""

    async def scenario() -> None:
        """Collect every item the stream yields."""
        bound_llm = LLM(_FakeProvider()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            collected_items = [item async for item in handle]
        assert collected_items == ["a", "b", _FAKE_TOOL_CALL]

    asyncio.run(scenario())


def test_stream_closes_on_context_exit() -> None:
    """Leaving the async with block closes the underlying provider stream."""

    async def scenario() -> None:
        """Open the stream, consume one item, then leave the context."""
        stream = _FakeStream()
        bound_llm = LLM(_FakeProvider(stream=stream)).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                break
        assert stream.closed is True

    asyncio.run(scenario())


def test_server_stated_retry_after_overrides_exponential_backoff() -> None:
    """A tiny retry_after_seconds beats a backoff base that would stall the test."""

    async def scenario() -> None:
        """Recover from one rate-limit failure whose server-stated wait is near zero."""
        provider = _FakeProvider(
            failures=[TransientError("rate limited", retry_after_seconds=0.001)]
        )
        rate_limiter = RateLimiter(max_attempts=2, backoff_base_seconds=30.0)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_max_in_flight_bounds_batch_concurrency() -> None:
    """A five-item batch under max_in_flight=2 never overlaps more than two sends."""

    async def scenario() -> None:
        """Run the batch on a slow fake and read the recorded peak."""
        provider = _FakeProvider(echo=True, send_seconds=0.01)
        bound_llm = LLM(provider, rate_limiter=_fast_rate_limiter(max_in_flight=2)).bind()
        conversations = [[UserMessage(content=str(index))] for index in range(5)]
        results = await bound_llm.generate_many(conversations)
        assert _batch_outputs(results) == ["0", "1", "2", "3", "4"]
        assert provider.bound_providers[0].peak_in_flight == 2

    asyncio.run(scenario())


def test_backoff_sleep_does_not_hold_the_in_flight_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With max_in_flight=1, a task backing off lets another request run.

    The failure carries no retry_after_seconds, so nothing pauses admission;
    only a held slot could delay the second request.
    The second request must complete while the first is still backing off, which the first_task.done() check pins.
    The full-jitter draw is pinned to its ceiling so the backoff outlasts the second request deterministically.
    """
    monkeypatch.setattr(rate_limiter_module.random, "uniform", lambda _low, high: high)

    async def scenario() -> None:
        """Interleave a retrying item with a clean one under one slot."""
        provider = _FakeProvider(failures=[TransientError("boom")])
        rate_limiter = RateLimiter(
            max_attempts=2, backoff_base_seconds=0.2, max_in_flight=1
        )
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        first_task = asyncio.create_task(bound_llm.generate_one([UserMessage(content="a")]))
        await asyncio.sleep(0.01)
        second = await bound_llm.generate_one([UserMessage(content="b")])
        assert second.output == "ok"
        assert not first_task.done()
        first = await first_task
        assert first.output == "ok"
        assert first.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_protocol_error_releases_the_slot() -> None:
    """A StreamProtocolError from items() returns the slot and closes the stream."""

    async def scenario() -> None:
        """Drive final() into the protocol error without a context manager."""
        stream = _ProtocolErrorStream()
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(_FakeProvider(stream=stream), rate_limiter=rate_limiter).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])
        with pytest.raises(StreamProtocolError):
            await handle.final()
        assert stream.closed is True
        async with rate_limiter.slot():
            pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_releases_its_slot_when_exhausted() -> None:
    """After a stream drains, its RateLimiter slot is available again."""

    async def scenario() -> None:
        """Drain one stream under max_in_flight=1, then acquire the slot."""
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(_FakeProvider(), rate_limiter=rate_limiter).bind()
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
        provider = _FakeProvider(
            failures=[
                TransientError("rate limited", retry_after_seconds=0.001, is_rate_limit=True)
            ]
        )
        rate_limiter = _fast_rate_limiter(max_attempts=2)
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
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
        provider = _FakeProvider(
            open_failures=[
                TransientError("rate limited", retry_after_seconds=0.001, is_rate_limit=True)
            ]
        )
        rate_limiter = _fast_rate_limiter()
        bound_llm = LLM(provider, rate_limiter=rate_limiter).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            await handle._ensure_stream_open()
            # No item has been pulled yet; the open alone must have ended recovery.
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
