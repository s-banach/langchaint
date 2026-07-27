"""BoundLLM and StreamHandle driven by fake adapters.

A fake BoundAdapter scripts send to fail a fixed number of times before succeeding,
and a fake AdapterStream emits a fixed item sequence.
Together they pin the retry loop, rebind rebuild, batch ordering, and the stream contract without any network access.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, assert_type, override

import pytest
from pydantic import BaseModel

from langchaint import (
    LLM,
    ZERO_USAGE,
    AbandonedCall,
    AssistantMessage,
    BoundLLM,
    ContextWindowExceededError,
    EmptyTurnError,
    GenerationError,
    HasTools,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    Message,
    NoTools,
    ProviderFailedTerminallyError,
    RateLimiter,
    RefusalError,
    Response,
    RetriesExhaustedError,
    SchemaViolationError,
    StreamItem,
    StreamProtocolError,
    TextPart,
    ToolCall,
    ToolManager,
    TransientError,
    UnfinishedTurnError,
    UnrecognizedError,
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
    ContextWindowExceeded,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    Refusal,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.llm import UNCHANGED
from langchaint.streaming import StreamHandle
from tests.helpers import uniform_returns_ceiling

_USAGE = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=1,
    output_tokens=1,
    output_tokens_reasoning=0,
    input_tokens_cache_read_cost_in_usd=0.0,
    input_tokens_cache_write_cost_in_usd=0.0,
    input_tokens_cache_none_cost_in_usd=0.0,
    output_tokens_cost_in_usd=0.0,
)
_USAGE_BILLED = _USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.25})
"""The billing a 200 that produced no output (a refusal or truncation) carries."""
_USAGE_STREAM = _USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.001})
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
    """Stands in for the SDK response model a real adapter holds in raw.

    Its id is what the fake bound adapter looks the scripted receipt up under, standing in for the
    fields a real adapter reads the turn and the counters off.
    """

    id: str


def _raw_id(raw: BaseModel) -> str:
    """Narrow a raw response to the fake one and return its id.

    Raises:
        TypeError: raw is not a _FakeRawResponse, which the real adapters raise for the same reason.
    """
    if not isinstance(raw, _FakeRawResponse):
        raise TypeError(f"expected a _FakeRawResponse, got {type(raw).__name__}")
    return raw.id


@dataclass(frozen=True, kw_only=True)
class _Receipt:
    """One response the fake hands back: what interpret reads off it, and what it billed."""

    outcome: ResponseOutcome[str]
    usage: Usage


def _billed(outcome: ResponseOutcome[str]) -> _Receipt:
    """Script one 200 the provider billed, whatever interpret goes on to make of it."""
    return _Receipt(outcome=outcome, usage=_USAGE_BILLED)


_REJECTED_TURN = AssistantMessage(turn=(TextPart(text="what the rejected 200 carried"),))
"""The turn a 200 that produced no output still carried, which every such member takes."""


def _success_result(content: str) -> AdapterResult[str]:
    """Build a successful text AdapterResult carrying the given content."""
    return AdapterResult(
        output=content,
        assistant_message=AssistantMessage(turn=(TextPart(text=content),)),
        stop_reason="end_turn",
    )


_FAKE_TOOL_CALL = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')


class _FakeStream(AdapterStream):
    """A fixed item sequence and a fixed assembled response.

    final() hands back the response object, exactly as a real adapter's stream does; receipt() is
    what the fake bound adapter registers for it, standing in for interpret and usage_from_raw
    reading that response.
    """

    def __init__(self) -> None:
        """Start unclosed; close records that it ran."""
        self.closed = False
        self.raw = _FakeRawResponse(id="fake-final")

    def receipt(self) -> _Receipt:
        """Return the assembled result the SDK would produce, and what the stream billed."""
        return _Receipt(
            outcome=AdapterResult(
                output="ab",
                assistant_message=AssistantMessage(turn=(TextPart(text="ab"),)),
                stop_reason="end_turn",
            ),
            usage=_USAGE_STREAM,
        )

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        yield "a"
        yield "b"
        yield _FAKE_TOOL_CALL

    @override
    async def final(self) -> BaseModel:
        """Return the response the stream's events assembled into."""
        return self.raw

    @override
    async def close(self) -> None:
        self.closed = True


class _RefusingStream(_FakeStream):
    """A stream that yields items normally but whose assembled response holds a refusal.

    Mirrors an adapter that reads the assembled message and finds a refusal, reporting the Refusal
    member carrying the turn the model wrote to refuse.
    """

    @override
    def receipt(self) -> _Receipt:
        """Report the refusal instead of a result, carrying this attempt's billing."""
        return _billed(Refusal(assistant_message=_REJECTED_TURN))


class _UnfinishedTurnStream(_FakeStream):
    """A stream that yields items normally but whose assembled message is not a finished turn.

    Mirrors an adapter that reads a stop reason it cannot call finished, reporting UnfinishedTurn
    with the reason naming the provider's own word.
    """

    @override
    def receipt(self) -> _Receipt:
        """Report the unfinished turn instead of a result, carrying this attempt's billing."""
        return _billed(
            UnfinishedTurn(
                reason="anthropic returned stop_reason 'pause_turn'",
                assistant_message=_REJECTED_TURN,
            )
        )


_VALIDATION_ERROR_JSON = (
    '[{"type":"value_error","loc":["celsius"],'
    '"msg":"Value error, SENTINEL is not a temperature","input":"SENTINEL"}]'
)
"""A pydantic rejection whose msg embeds the rejected value, as a caller's field_validator writes it."""


class _SchemaViolationStream(_FakeStream):
    """A stream that yields items normally but whose assembled text the response_format rejects.

    Mirrors an adapter that validates the assembled message and gets a rejection, reporting
    SchemaViolation with what pydantic rejected.
    """

    @override
    def receipt(self) -> _Receipt:
        """Report the rejection instead of a result, carrying this attempt's billing."""
        return _billed(
            SchemaViolation(
                validation_error_json=_VALIDATION_ERROR_JSON, assistant_message=_REJECTED_TURN
            )
        )


_PROVIDER_FAILURE_REASON = "The server had an error while processing your request."
"""A provider's own description of a failure, as openai puts it in a failed response's error."""

_PROVIDER_FAILED_TRANSIENTLY = ProviderFailedTransiently(
    reason=_PROVIDER_FAILURE_REASON, is_rate_limit=False, assistant_message=_REJECTED_TURN
)
"""One 200 whose body reports a failure a resend may get past, with no rate limit named."""


class _ProviderFailedTransientlyStream(_FakeStream):
    """A stream whose assembled response reports a provider failure a resend may get past.

    Mirrors an adapter reading the assembled response in AdapterStream.final() and finding a body
    that reports the run failed, reporting ProviderFailedTransiently with the 200's billing.
    """

    @override
    def receipt(self) -> _Receipt:
        """Report the failure instead of a result, carrying this attempt's billing."""
        return _billed(_PROVIDER_FAILED_TRANSIENTLY)


class _ProviderFailedTerminallyStream(_FakeStream):
    """A stream whose assembled response reports a provider failure a resend would hit again."""

    @override
    def receipt(self) -> _Receipt:
        """Report the failure instead of a result, carrying this attempt's billing."""
        return _billed(
            ProviderFailedTerminally(
                reason=_PROVIDER_FAILURE_REASON, assistant_message=_REJECTED_TURN
            )
        )


class _FinalRaisesStream(_FakeStream):
    """A stream that yields items normally but whose final() raises instead of returning a response.

    Mirrors an adapter whose assembly step failed, so this call reached no response at all.
    Each call raises a fresh error, so a replayed one is identifiable by identity.
    """

    def __init__(self) -> None:
        """Start with no final() call counted."""
        super().__init__()
        self.final_calls = 0

    @override
    async def final(self) -> BaseModel:
        """Count the call and raise.

        Raises:
            RuntimeError: always.
        """
        self.final_calls += 1
        raise RuntimeError("assembly failed")


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


class _UnnamedItemErrorStream(_FakeStream):
    """A stream whose items() raises an exception the adapter cannot name, before the first item."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Raise a plain exception before yielding anything.

        Yields:
            Nothing; the raise precedes the first yield.

        Raises:
            ValueError: always, before the first yield.
        """
        raise ValueError("boom")
        yield "unreachable"


class _FailsAfterFirstItemStream(_FakeStream):
    """A stream that yields one item and then fails, past the point where reopening is allowed."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield one chunk, then raise.

        Yields:
            One text chunk, before the raise.

        Raises:
            ValueError: after the first yield.
        """
        yield "a"
        raise ValueError("dropped mid-stream")


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


class _FailingCloseStream(_FakeStream):
    """A stream whose close() raises, standing in for a provider teardown that fails."""

    @override
    async def close(self) -> None:
        """Record the attempt, then raise.

        Raises:
            OSError: always.
        """
        self.closed = True
        raise OSError("connection reset while closing")


type _ScriptedSend = Exception | _Receipt | InvalidRequest
"""One scripted send: an exception the fake raises, a response it hands back, or a refusal to send.

The three exist because the adapter contract splits that way: an attempt with no response to read is
an exception for Adapter.classify, a response is what send returns and interpret then reads, and a
conversation the adapter will not put on the wire is InvalidRequest.
"""

type _ScriptedOpen = Exception | InvalidRequest
"""One scripted open_stream outcome; open_stream returns no response-shaped outcome."""


class _FakeBoundAdapter(BoundAdapter[str]):
    """A bound adapter whose send follows a scripted failure sequence."""

    def __init__(
        self,
        *,
        failures: Sequence[_ScriptedSend] = (),
        open_failures: Sequence[_ScriptedOpen] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        send_seconds: float = 0.0,
        hang_from_open: int | None = None,
        hang_from_send: int | None = None,
    ) -> None:
        """Store the failure scripts, echo mode, and the stream open_stream returns.

        failures scripts send; open_failures scripts open_stream, exercising the pre-first-item stream retry path.
        send_seconds > 0 makes each send suspend that long,
        so a batch overlaps and peak_in_flight records the concurrency it reached.
        hang_from_open is the 1-based open_stream call from which every open suspends forever,
        so a cancellation lands on the open itself rather than on a later item pull.
        hang_from_send is the same for send, so a deadline fires while that attempt is in flight.
        sent_raws collects the response objects send handed back, in order, so a test can assert the
        caller got one of them and not a copy.
        """
        self._failures = list(failures)
        self._open_failures = list(open_failures)
        self._echo = echo
        self._send_seconds = send_seconds
        self._hang_from_open = hang_from_open
        self._hang_from_send = hang_from_send
        self.stream = stream if stream is not None else _FakeStream()
        self._receipt_by_raw_id: dict[str, _Receipt] = {}
        self.sent_raws: list[_FakeRawResponse] = []
        self.send_count = 0
        self.open_count = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    @override
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Return what the response under this raw was scripted to have billed."""
        return self._receipt_by_raw_id[_raw_id(raw)].usage

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Return what the response under this raw was scripted to produce."""
        return self._receipt_by_raw_id[_raw_id(raw)].outcome

    @override
    async def send(self, conversation: Sequence[Message]) -> BaseModel | InvalidRequest:
        """Raise or return the next scripted send, else hand back a success response."""
        self.send_count += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._hang_from_send is not None and self.send_count >= self._hang_from_send:
                await asyncio.Event().wait()
            if self._send_seconds:
                await asyncio.sleep(self._send_seconds)
            raw = _FakeRawResponse(id=f"fake-response-{self.send_count}")
            if self._failures:
                scripted = self._failures.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                if isinstance(scripted, InvalidRequest):
                    return scripted
                self._receipt_by_raw_id[raw.id] = scripted
                self.sent_raws.append(raw)
                return raw
            first = conversation[0]
            content = (
                first.content
                if self._echo and isinstance(first, UserMessage) and isinstance(first.content, str)
                else "ok"
            )
            self._receipt_by_raw_id[raw.id] = _Receipt(
                outcome=_success_result(content), usage=_USAGE
            )
            self.sent_raws.append(raw)
            return raw
        finally:
            self.in_flight -= 1

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream | InvalidRequest:
        """Count the attempt, suspend or take the next scripted open outcome, else return the stored fake stream."""
        self.open_count += 1
        if self._hang_from_open is not None and self.open_count >= self._hang_from_open:
            await asyncio.Event().wait()
        if self._open_failures:
            scripted = self._open_failures.pop(0)
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        self._receipt_by_raw_id[self.stream.raw.id] = self.stream.receipt()
        return self.stream


class _FakeStructuredBoundAdapter[ModelT: BaseModel](BoundAdapter[ModelT]):
    """A structured bound adapter for response_format rebind tests; it never generates.

    Those tests check binding identity and the switched content type, not structured output,
    so send and open_stream stay unreachable.
    """

    @override
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT]:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def send(self, conversation: Sequence[Message]) -> BaseModel | InvalidRequest:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream | InvalidRequest:
        """Unreachable: response_format rebind tests do not stream."""
        raise NotImplementedError


class _FakeAdapter(Adapter):
    """An adapter whose bind_text hands out fake bound adapters."""

    _bound_adapter_class: ClassVar[type[_FakeBoundAdapter]] = _FakeBoundAdapter
    """The class bind_text hands out; a subclass names its own to vary what interpret does."""

    def __init__(
        self,
        *,
        failures: Sequence[_ScriptedSend] = (),
        open_failures: Sequence[_ScriptedOpen] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        classify_result: ErrorClassification = "unrecognized",
        send_seconds: float = 0.0,
        hang_from_open: int | None = None,
        hang_from_send: int | None = None,
    ) -> None:
        """Store how each freshly bound adapter behaves and the classify verdict."""
        # This adapter reaches no SDK, so it passes client=None, which matches no entry in the
        # base's empty provider_name_by_client_class, leaving the stated "fake" to stand.
        super().__init__(client=None, model="fake-model", provider_name="fake")
        self._failures = failures
        self._open_failures = open_failures
        self._echo = echo
        self._stream = stream
        self._classify_result = classify_result
        self._send_seconds = send_seconds
        self._hang_from_open = hang_from_open
        self._hang_from_send = hang_from_send
        self.bound_adapters: list[_FakeBoundAdapter] = []
        self.structured_bind_count = 0

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        bound = self._bound_adapter_class(
            failures=self._failures,
            open_failures=self._open_failures,
            echo=self._echo,
            stream=self._stream,
            send_seconds=self._send_seconds,
            hang_from_open=self._hang_from_open,
            hang_from_send=self._hang_from_send,
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
    def classify(self, error: Exception) -> ErrorClassification:
        """Return the fixed verdict for every exception classify sees."""
        return self._classify_result


class _InterpretRaisesBoundAdapter(_FakeBoundAdapter):
    """A bound adapter whose interpret raises over a response send already handed back."""

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Raise instead of reading the response.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("interpretation failed")


class _InterpretRaisesAdapter(_FakeAdapter):
    """An adapter whose bound adapters price a response and then raise reading it."""

    _bound_adapter_class = _InterpretRaisesBoundAdapter


def test_a_raise_from_interpret_leaves_the_response_and_its_billing_on_the_record() -> None:
    """The attempt keeps the response it received and what that response billed, with no turn.

    The 200 arrived and was paid for before interpret read it, so an exception from that read must
    not take the attempt off the call: the record is the only account of what the item spent.
    """

    async def scenario() -> None:
        """Drive one generate_one whose interpret raises over the response send returned."""
        bound_llm = LLM(_InterpretRaisesAdapter(), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnrecognizedError) as unrecognized:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = unrecognized.value.attempt_records
        assert isinstance(record.raw, _FakeRawResponse)
        assert record.usage == _USAGE
        assert record.assistant_message is None
        assert record.error is None
        assert isinstance(unrecognized.value.__cause__, RuntimeError)

    asyncio.run(scenario())


def test_stream_final_records_the_response_before_interpreting_it() -> None:
    """A raise from interpret leaves the assembled response and its billing on the attempt record.

    The stream's one request is paid for by the time final() has a response, so an exception from
    reading it must not erase what it billed.
    """

    async def scenario() -> None:
        """Call final() on a stream whose interpret raises, then freeze the ledger it left."""
        stream = _FakeStream()
        bound_llm = LLM(
            _InterpretRaisesAdapter(stream=stream), rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RuntimeError, match="interpretation failed"):
                await handle.final()
            (record,) = handle._ledger.freeze().attempt_records
        assert record.raw is stream.raw
        assert record.usage == _USAGE_STREAM
        assert record.assistant_message is None
        assert record.error is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_retry_recovers_after_a_transient_failure() -> None:
    """One transient failure then success yields a two-attempt success Response.

    The succeeding record carries the answer turn, and the failed one carries no turn at all.
    The Response carries the object the succeeding send returned (identity, not equality): an equal
    copy would silently introduce the per-request deep copy the no-rewrap rule bans.
    """

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
        assert failed.assistant_message is None
        assert succeeded.error is None
        assert succeeded.assistant_message == _success_result("ok").assistant_message
        (succeeding_raw,) = adapter.bound_adapters[0].sent_raws
        assert response.raw is succeeding_raw
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
    monkeypatch.setattr(rate_limiter_module.random, "uniform", uniform_returns_ceiling)

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


def test_pre_send_rejection_raises_immediately_without_retry() -> None:
    """A InvalidRequest outcome fails the item on the first attempt and is never retried.

    classify returns "transient" here, and the arm never reaches it: a returned outcome is not an
    exception, so no classify verdict can turn this into a second attempt.
    The InvalidRequestError the loop builds carries the reason and the row-shape fields.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send reports InvalidRequest under a transient classify verdict."""
        adapter = _FakeAdapter(
            failures=[InvalidRequest(reason="nope")], classify_result="transient"
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        assert rejected.value.reason == "nope"
        assert rejected.value.error_text == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.provider_name == adapter.provider_name
        assert rejected.value.attempt_records == ()
        assert rejected.value.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_pre_send_rejection_registers_no_success_with_the_rate_limiter() -> None:
    """An unsendable conversation is not a completed request, so it must not end the recovery.

    A registered success would clear a rate-limit pause that no provider response lifted.
    """

    async def scenario() -> None:
        """Put the limiter into recovery, then report InvalidRequest while holding its probe slot."""
        rate_limiter = _fast_rate_limiter()
        failing_admission = await rate_limiter.acquire()
        rate_limiter.register_transient_error(
            failing_admission,
            (TransientError("429", retry_after_seconds=0.0, is_rate_limit=True),),
        )
        rate_limiter.release(failing_admission)
        assert rate_limiter._recovering
        adapter = _FakeAdapter(failures=[InvalidRequest(reason="nope")])
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        with pytest.raises(InvalidRequestError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert rate_limiter._recovering

    asyncio.run(scenario())


def test_rejection_after_transient_attempts_carries_their_records() -> None:
    """An InvalidRequestError after transient attempts carries their records.

    The prior attempts' usage rides the error, so a billed 200 retried before the rejection stays
    accounted for. The two routes to the error differ in what the failing attempt leaves: a rejection
    classify names went out, so it gets a record, and a InvalidRequest outcome never did, so it does not.
    """

    async def scenario() -> None:
        """Settle one billed transient attempt, then take each route to the error in its own call."""
        classified_adapter = _FakeAdapter(
            failures=[
                _billed(_PROVIDER_FAILED_TRANSIENTLY),
                ValueError("bad request"),
            ],
            classify_result="invalid_request",
        )
        bound_llm = LLM(classified_adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as classified:
            await bound_llm.generate_one([UserMessage(content="hi")])
        billed_record, rejected_record = classified.value.attempt_records
        assert billed_record.usage.cost_in_usd == 0.25
        assert rejected_record.error is None
        assert rejected_record.usage == ZERO_USAGE
        assert rejected_record.raw is None
        assert isinstance(classified.value.__cause__, ValueError)
        invalid_request_adapter = _FakeAdapter(
            failures=[
                _billed(_PROVIDER_FAILED_TRANSIENTLY),
                InvalidRequest(reason="reported by the adapter"),
            ],
        )
        invalid_request_bound_llm = LLM(
            invalid_request_adapter, rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)
        with pytest.raises(InvalidRequestError) as invalid_request:
            await invalid_request_bound_llm.generate_one([UserMessage(content="hi")])
        (invalid_request_billed_record,) = invalid_request.value.attempt_records
        assert invalid_request_billed_record.usage.cost_in_usd == 0.25
        assert invalid_request.value.reason == "reported by the adapter"

    asyncio.run(scenario())


def test_refusal_outcome_from_send_raises_row_shaped_without_retry() -> None:
    """A Refusal outcome becomes a RefusalError carrying the attempt record, never retried.

    The record carries the turn the refusal arrived on and the response it was read from.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send reports the Refusal member."""
        adapter = _FakeAdapter(failures=[_billed(Refusal(assistant_message=_REJECTED_TURN))])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(RefusalError) as refusal:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = refusal.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.usage.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.usage.cost_in_usd == 0.25
        assert record.assistant_message == _REJECTED_TURN
        assert record.raw is not None

    asyncio.run(scenario())


def test_max_completion_tokens_exceeded_outcome_from_send_raises_row_shaped_without_retry() -> (
    None
):
    """A MaxCompletionTokensExceeded outcome becomes a row-shaped MaxCompletionTokensExceededError, never retried."""

    async def scenario() -> None:
        """Drive one generate_one whose send reports the MaxCompletionTokensExceeded arm."""
        adapter = _FakeAdapter(
            failures=[_billed(MaxCompletionTokensExceeded(assistant_message=_REJECTED_TURN))]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(MaxCompletionTokensExceededError) as max_completion_tokens_exceeded:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = max_completion_tokens_exceeded.value
        assert failure.attempts == 1
        assert failure.stop_reason == "max_tokens"
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_empty_turn_outcome_from_send_raises_row_shaped_without_retry() -> None:
    """An EmptyTurn outcome becomes an EmptyTurnError, never retried: the model finished and said nothing."""

    async def scenario() -> None:
        """Drive one generate_one whose send reports EmptyTurn."""
        adapter = _FakeAdapter(failures=[_billed(EmptyTurn(assistant_message=_REJECTED_TURN))])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(EmptyTurnError) as empty_turn:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = empty_turn.value
        assert failure.attempts == 1
        assert failure.stop_reason == "end_turn"
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_schema_violation_outcome_from_send_raises_row_shaped_without_retry() -> None:
    """A SchemaViolation outcome fails the item, and pydantic's rejection travels on the error.

    Never retried: the turn completed, so nothing about the attempt was transient.
    error_text carries none of the rejection, whose msg embeds the value a caller's own validator
    rejected; the tracing layer writes error_text into every span whatever capture_message_content
    the caller chose.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send reports SchemaViolation."""
        adapter = _FakeAdapter(
            failures=[
                _billed(
                    SchemaViolation(
                        validation_error_json=_VALIDATION_ERROR_JSON,
                        assistant_message=_REJECTED_TURN,
                    )
                )
            ]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(SchemaViolationError) as schema_violation:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = schema_violation.value
        assert failure.attempts == 1
        assert failure.stop_reason == "end_turn"
        assert failure.validation_error_json == _VALIDATION_ERROR_JSON
        assert "SENTINEL" not in failure.error_text
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_context_window_exceeded_outcome_from_send_raises_row_shaped_without_retry() -> None:
    """A ContextWindowExceeded outcome fails the item: a request too long to serve stays too long."""

    async def scenario() -> None:
        """Drive one generate_one whose send reports ContextWindowExceeded."""
        adapter = _FakeAdapter(
            failures=[_billed(ContextWindowExceeded(assistant_message=_REJECTED_TURN))]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ContextWindowExceededError) as context_window_exceeded:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = context_window_exceeded.value
        assert failure.attempts == 1
        assert failure.stop_reason == "context_window_exceeded"
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_unfinished_turn_outcome_from_send_raises_carrying_the_adapter_s_reason() -> None:
    """An UnfinishedTurn outcome fails the item, and the adapter's reason reaches error_text.

    The reason is the only description of a 200 langchaint does not model, so the error must carry
    it rather than a constant of its own.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send reports UnfinishedTurn."""
        adapter = _FakeAdapter(
            failures=[
                _billed(
                    UnfinishedTurn(
                        reason="anthropic returned stop_reason 'pause_turn'",
                        assistant_message=_REJECTED_TURN,
                    )
                )
            ]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnfinishedTurnError) as unfinished_turn:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = unfinished_turn.value
        assert "pause_turn" in failure.error_text
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_provider_failed_transiently_from_send_is_retried_and_keeps_its_billing() -> None:
    """The outcome is retried, that 200's billing lands on its record, and the reason on its error."""

    async def scenario() -> None:
        """Drive one generate_one whose first send reports the failure and whose second succeeds."""
        adapter = _FakeAdapter(failures=[_billed(_PROVIDER_FAILED_TRANSIENTLY)])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 2
        assert response.attempts == 2
        rejected, succeeded = response.attempt_records
        assert isinstance(rejected.error, TransientError)
        assert str(rejected.error) == _PROVIDER_FAILURE_REASON
        assert rejected.usage.cost_in_usd == 0.25
        assert succeeded.error is None

    asyncio.run(scenario())


def test_provider_failed_transiently_ends_the_rate_limiter_recovery() -> None:
    """A failed 200 is a completed request, so it ends recovery like any other 200.

    The provider served the request, which is what the recovery probe asks; that its body reports a
    failure is the item's problem, not the account's quota's.
    """

    async def scenario() -> None:
        """Put the limiter into recovery, then report the failure while holding its probe slot."""
        rate_limiter = _fast_rate_limiter()
        failing_admission = await rate_limiter.acquire()
        rate_limiter.register_transient_error(
            failing_admission,
            (TransientError("429", retry_after_seconds=0.0, is_rate_limit=True),),
        )
        rate_limiter.release(failing_admission)
        assert rate_limiter._recovering
        adapter = _FakeAdapter(failures=[_billed(_PROVIDER_FAILED_TRANSIENTLY)])
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        await bound_llm.generate_one([UserMessage(content="hi")])
        assert not rate_limiter._recovering

    asyncio.run(scenario())


def test_provider_failed_transiently_carrying_the_rate_limit_flag_pauses_admission() -> None:
    """A failure the provider named a rate limit puts the shared limiter into recovery.

    The flag is the only thing distinguishing this 200 from any other failed one, and it has to
    reach the RateLimiter for a rate limit reported inside a 200 to pace the account at all.
    """

    async def scenario() -> None:
        """Spend the whole budget on rate-limited failures, so recovery is still in force at the end."""
        rate_limiter = _fast_rate_limiter(max_attempts=1)
        adapter = _FakeAdapter(
            failures=[
                _billed(
                    ProviderFailedTransiently(
                        reason="Rate limit reached for gpt-5.6",
                        is_rate_limit=True,
                        assistant_message=_REJECTED_TURN,
                    )
                )
            ]
        )
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        with pytest.raises(RetriesExhaustedError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = exhausted.value.attempt_records
        assert isinstance(record.error, TransientError)
        assert record.error.is_rate_limit
        assert rate_limiter._recovering

    asyncio.run(scenario())


def test_provider_failed_terminally_from_send_raises_row_shaped_without_retry() -> None:
    """A terminal provider failure fails the item once, with the provider's own text as the reason.

    Never retried: what the body names is a property of the request, so the retry budget would buy
    the same body at full price each time.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send reports the terminal failure."""
        adapter = _FakeAdapter(
            failures=[
                _billed(
                    ProviderFailedTerminally(
                        reason=_PROVIDER_FAILURE_REASON, assistant_message=_REJECTED_TURN
                    )
                )
            ]
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ProviderFailedTerminallyError) as provider_failure:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = provider_failure.value
        assert failure.attempts == 1
        assert failure.stop_reason is None
        assert _PROVIDER_FAILURE_REASON in failure.error_text
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


def test_exception_classified_invalid_request_fails_the_item_without_retry() -> None:
    """A plain exception classified invalid_request raises InvalidRequestError on the first attempt.

    InvalidRequestError is a GenerationError, so in a batch it becomes the item's failure row
    rather than touching the siblings; the classified exception stays reachable as __cause__.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-invalid_request exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="invalid_request")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        assert rejected.value.reason == "the provider rejected the request: boom"
        assert isinstance(rejected.value.__cause__, ValueError)

    asyncio.run(scenario())


def test_exception_classified_unrecognized_fails_the_item_without_retry() -> None:
    """A plain exception classified unrecognized raises UnrecognizedError on the first attempt.

    UnrecognizedError is a GenerationError, so in a batch it becomes the item's failure row
    and the siblings run on.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-unrecognized exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="unrecognized")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnrecognizedError) as unrecognized:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = unrecognized.value
        assert isinstance(failure, GenerationError)
        assert isinstance(failure.error, ValueError)
        assert isinstance(failure.__cause__, ValueError)
        assert failure.error_text == "unrecognized provider error: boom"
        assert failure.stop_reason is None
        assert failure.attempt_records == ()
        assert failure.model == adapter.model
        assert failure.provider_name == adapter.provider_name

    asyncio.run(scenario())


def test_unrecognized_error_becomes_the_items_failure_row_and_siblings_continue() -> None:
    """A classify-unrecognized item comes back as its UnrecognizedError row; the sibling succeeds."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send is unrecognized."""
        adapter = _FakeAdapter(
            echo=True, failures=[ValueError("boom")], classify_result="unrecognized"
        )
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, UnrecognizedError)
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(scenario())


def test_generate_one_success_and_failure_append_no_abandoned_call() -> None:
    """A call that returns or raises appends nothing: its usage travels on the value."""

    async def scenario() -> None:
        """Drive one clean finish and one retry exhaustion over the same log."""
        abandoned_call_log: list[AbandonedCall] = []
        adapter = _FakeAdapter()
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one(
            [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
        )
        assert response.output == "ok"
        failing_adapter = _FakeAdapter(failures=[TransientError("e1"), TransientError("e2")])
        failing_bound_llm = LLM(
            failing_adapter, rate_limiter=_fast_rate_limiter(max_attempts=2)
        ).bind(automatic_prompt_caching=True)
        with pytest.raises(RetriesExhaustedError):
            await failing_bound_llm.generate_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            )
        assert abandoned_call_log == []

    asyncio.run(scenario())


def test_generate_one_cancellation_appends_the_settled_attempts() -> None:
    """A caller's deadline cancelling the call leaves an AbandonedCall carrying the settled usage.

    The settled attempt's record lives inside the cancelled retry frame, so without the append the
    billed 200's known paid usage would vanish with the frame; the CancelledError still propagates,
    which is the caller's scope converting it to this TimeoutError.
    """

    async def scenario() -> None:
        """Settle one billed transient attempt, then hang the retry into the caller's deadline."""
        adapter = _FakeAdapter(
            failures=[_billed(_PROVIDER_FAILED_TRANSIENTLY)],
            hang_from_send=2,
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        abandoned_call_log: list[AbandonedCall] = []
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await bound_llm.generate_one(
                    [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
                )
        assert adapter.bound_adapters[0].send_count == 2
        (abandoned_call,) = abandoned_call_log
        assert abandoned_call.usage_settled.cost_in_usd == 0.25
        assert len(abandoned_call.attempt_records) == 1
        assert abandoned_call.model == adapter.model
        assert abandoned_call.provider_name == adapter.provider_name
        assert abandoned_call.elapsed_seconds > 0.0

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_one_cancellation_with_nothing_settled_fabricates_no_usage() -> None:
    """A call cancelled before any attempt settled appends a record with no attempt records."""

    async def scenario() -> None:
        """Cancel the call's task while the first send hangs."""
        adapter = _FakeAdapter(hang_from_send=1)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        abandoned_call_log: list[AbandonedCall] = []
        call = asyncio.create_task(
            bound_llm.generate_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            )
        )
        await asyncio.sleep(0.01)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        (abandoned_call,) = abandoned_call_log
        assert abandoned_call.attempt_records == ()
        assert abandoned_call.usage_settled == ZERO_USAGE

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_one_rejection_appends_no_abandoned_call() -> None:
    """An InvalidRequestError propagates with nothing appended: it abandons no request."""

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-invalid_request exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="invalid_request")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        abandoned_call_log: list[AbandonedCall] = []
        with pytest.raises(InvalidRequestError):
            await bound_llm.generate_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            )
        assert abandoned_call_log == []

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


def test_bind_and_rebind_type_output_by_whether_a_tool_manager_is_bound() -> None:
    """Pin every binding's static output type, the thing a caller writes `if output is None` against.

    output is optional on one binding only: structured plus a ToolManager, whose turn may be the tool
    calls. A text binding never types it optional, a tool-call turn's text being "" and not None.
    Every transition is exact, so dropping the ToolManager drops the None with it.
    """
    llm = LLM(_FakeAdapter())
    tool_manager = ToolManager([])

    text = llm.bind(automatic_prompt_caching=True)
    assert_type(text, BoundLLM[str, NoTools])
    text_with_tools = llm.bind(tool_manager=tool_manager, automatic_prompt_caching=True)
    assert_type(text_with_tools, BoundLLM[str, HasTools])
    structured = llm.bind(response_format=_Answer, automatic_prompt_caching=True)
    assert_type(structured, BoundLLM[_Answer, NoTools])
    structured_with_tools = llm.bind(
        response_format=_Answer, tool_manager=tool_manager, automatic_prompt_caching=True
    )
    assert_type(structured_with_tools, BoundLLM[_Answer, HasTools])

    # BoundLLM[X] is BoundLLM[X, NoTools]: the PEP 696 default keeps the common annotation short.
    assert_type(structured, BoundLLM[_Answer])

    assert_type(structured.rebind(tool_manager=tool_manager), BoundLLM[_Answer, HasTools])
    assert_type(structured_with_tools.rebind(tool_manager=None), BoundLLM[_Answer, NoTools])
    assert_type(text_with_tools.rebind(response_format=_Answer), BoundLLM[_Answer, HasTools])
    assert_type(structured_with_tools.rebind(response_format=None), BoundLLM[str, HasTools])
    assert_type(structured_with_tools.rebind(system_prompt="s"), BoundLLM[_Answer, HasTools])


async def _pin_request_method_return_types(llm: LLM, tool_manager: ToolManager) -> None:
    """Pin the return types the ToolsT overloads produce, which is what the parameter is for.

    Never called: pyrefly checks this body, and the assertions are about types alone. Running it
    would need a structured fake that sends, which _FakeStructuredBoundAdapter deliberately is not.
    """
    structured_with_tools = llm.bind(
        response_format=_Answer, tool_manager=tool_manager, automatic_prompt_caching=True
    )
    assert_type(await structured_with_tools.generate_one("hi"), Response[_Answer | None])
    assert_type(structured_with_tools.stream_one("hi"), StreamHandle[_Answer | None])
    assert_type(
        await structured_with_tools.generate_many(["hi"]),
        list[Response[_Answer | None] | GenerationError],
    )
    structured = llm.bind(response_format=_Answer, automatic_prompt_caching=True)
    assert_type(await structured.generate_one("hi"), Response[_Answer])
    text_with_tools = llm.bind(tool_manager=tool_manager, automatic_prompt_caching=True)
    assert_type(await text_with_tools.generate_one("hi"), Response[str])
    assert_type(text_with_tools.stream_one("hi"), StreamHandle[str])


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
    """An item whose send reports Refusal comes back as the RefusalError in its slot, siblings succeed."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send reports Refusal."""
        adapter = _FakeAdapter(
            echo=True,
            failures=[_billed(Refusal(assistant_message=_REJECTED_TURN))],
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


def test_invalid_request_becomes_the_items_failure_row_and_siblings_continue() -> None:
    """A rejected item comes back as its InvalidRequestError row; the sibling still succeeds.

    Nothing a single item does reaches a sibling, so the batch returns one outcome per conversation.
    """

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send reports InvalidRequest."""
        adapter = _FakeAdapter(echo=True, failures=[InvalidRequest(reason="misconfigured")])
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, InvalidRequestError)
        assert first.reason == "misconfigured"
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


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


def test_generate_many_warm_cache_empty_batch_returns_empty() -> None:
    """An empty batch under warm_cache returns [] instead of indexing a first item."""

    async def scenario() -> None:
        """Run the empty batch."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        assert await bound_llm.generate_many([], warm_cache=True) == []

    asyncio.run(scenario())


def test_generate_many_cancellation_appends_one_abandoned_call_per_item() -> None:
    """A deadline cancelling the batch leaves every item's settled usage in the log.

    The returned list dies with the cancelled frame, so without the appends the whole batch's known
    paid usage vanishes at once, one item's worth per item.
    """

    async def scenario() -> None:
        """Settle the two scripted billed attempts, then hang the retries into the deadline."""
        adapter = _FakeAdapter(
            failures=[
                _billed(_PROVIDER_FAILED_TRANSIENTLY),
                _billed(_PROVIDER_FAILED_TRANSIENTLY),
            ],
            hang_from_send=3,
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        abandoned_call_log: list[AbandonedCall] = []
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await bound_llm.generate_many(
                    [[UserMessage(content="a")], [UserMessage(content="b")]],
                    abandoned_call_log=abandoned_call_log,
                )
        assert len(abandoned_call_log) == 2
        # The two scripted failures settle before either retry hangs, whichever item sends first.
        settled = sum(call.usage_settled.cost_in_usd for call in abandoned_call_log)
        assert settled == 2 * _USAGE_BILLED.cost_in_usd

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


class _ClassifyRaisesAdapter(_FakeAdapter):
    """A _FakeAdapter whose classify raises, so one item ends in something that is not a GenerationError."""

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Raise instead of classifying.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("classify defect")


def test_generate_many_appends_the_abandoned_calls_of_items_a_sibling_defect_cancels() -> None:
    """An item raising past the GenerationError arms leaves its siblings' settled usage in the log.

    gather propagates that raise while the siblings are still running, so unless they are awaited
    after being cancelled, their appends land after the raise has reached the caller and the log the
    caller reads is missing their settled billing.
    """

    async def scenario() -> None:
        """Settle one billed attempt on the first item, then raise past the arms on the second."""
        adapter = _ClassifyRaisesAdapter(
            failures=[
                _billed(_PROVIDER_FAILED_TRANSIENTLY),
                _billed(Refusal(assistant_message=_REJECTED_TURN)),
                ValueError("defect"),
            ],
            send_seconds=0.02,
            hang_from_send=4,
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        abandoned_call_log: list[AbandonedCall] = []
        with pytest.raises(RuntimeError, match="classify defect"):
            await bound_llm.generate_many(
                [
                    [UserMessage(content="a")],
                    [UserMessage(content="b")],
                    [UserMessage(content="c")],
                ],
                abandoned_call_log=abandoned_call_log,
            )
        # The first item settles its billed attempt and hangs on the retry, the second settles its
        # refusal row, and the third raises; each of the two records one billed attempt.
        assert len(abandoned_call_log) == 2
        settled = sum(call.usage_settled.cost_in_usd for call in abandoned_call_log)
        assert settled == 2 * _USAGE_BILLED.cost_in_usd

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_appends_the_abandoned_calls_of_the_items_that_already_settled() -> None:
    """A cancelled batch accounts for the items that finished, not only the ones still running.

    A settled item's row dies with the discarded result list, and in a batch cut off late those
    rows are most of what the batch spent, so leaving them out understates it by the largest term.
    Both branches are driven: without warm_cache every item settles in a _gather task,
    with it the warming item settles in generate_many's own frame.
    """

    async def scenario() -> None:
        """Let the first send succeed, hang the second, and cut the batch off on a deadline."""
        for warm_cache in (False, True):
            adapter = _FakeAdapter(hang_from_send=2)
            bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
                automatic_prompt_caching=True
            )
            abandoned_call_log: list[AbandonedCall] = []
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await bound_llm.generate_many(
                        [[UserMessage(content="a")], [UserMessage(content="b")]],
                        warm_cache=warm_cache,
                        abandoned_call_log=abandoned_call_log,
                    )
            assert len(abandoned_call_log) == 2
            # The hanging item settled no attempt; the item that succeeded settled its one.
            assert sorted(len(call.attempt_records) for call in abandoned_call_log) == [0, 1]
            assert sum(call.usage_settled.output_tokens for call in abandoned_call_log) == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


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

        The suppressed pyrefly error is SequenceNotStr statically rejecting the bare str, which
        leaves generate_many's overloads with nothing to match; the suppression doubles as a canary,
        since pyrefly reports it as unused if typeshed drift ever makes str satisfy SequenceNotStr
        and the static rejection lapses.
        """
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
        with pytest.raises(TypeError, match="bare str"):
            # pyrefly: ignore[no-matching-overload]
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

    The reopen runs inside _next_item's transient-failure handler, which no sibling except clause covers.
    _open_stream_with_retries releases in its own BaseException handler, so the block's exit is not what
    returns the slot.
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


def test_stream_cancelled_inside_the_block_appends_one_abandoned_call() -> None:
    """A cancellation unwinding the block leaves an AbandonedCall; the stream's own spend is unobservable.

    The record's attempt_records are empty because the streaming request is the in-flight attempt:
    only pre-first-item open failures settle records on a stream.
    """

    async def scenario() -> None:
        """Time out a consumer suspended on a hanging stream, then read the log."""
        abandoned_call_log: list[AbandonedCall] = []
        adapter = _FakeAdapter(stream=_HangingStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )

        async def drain() -> None:
            """Enter and iterate into the hang; the wait_for below cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ) as handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)
        (abandoned_call,) = abandoned_call_log
        assert abandoned_call.attempt_records == ()
        assert abandoned_call.usage_settled == ZERO_USAGE
        assert abandoned_call.model == adapter.model

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_close_that_raises_still_returns_the_in_flight_slot() -> None:
    """A failed teardown does not cost the shared budget a slot, and does not reach the caller.

    An admission the close skips is gone for the process's life, so the limiter's capacity shrinks
    by one on every such stream.
    """

    async def scenario() -> None:
        """Leave the block early over a stream whose close raises, then check both slots are free."""
        rate_limiter = _fast_rate_limiter(max_in_flight=2)
        bound_llm = LLM(
            _FakeAdapter(stream=_FailingCloseStream()), rate_limiter=rate_limiter
        ).bind(automatic_prompt_caching=True)
        # Leaving after one item keeps the admission held into __aexit__, so the close is the only
        # path that can return it; exhausting the iterator releases it before the close is reached.
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                break
        admissions = [await rate_limiter.acquire() for _ in range(2)]
        for admission in admissions:
            rate_limiter.release(admission)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_an_abandoned_call_log_that_raises_does_not_replace_the_cancellation() -> None:
    """A log whose append raises leaves the CancelledError to propagate as the task's outcome.

    A task ending on any other exception is not cancelled, so asyncio.timeout would report the
    log's error in place of TimeoutError and a TaskGroup's shutdown would see the substitution.
    """

    class _RaisingLog:
        """An abandoned_call_log whose append fails, standing in for a defective application log."""

        def append(self, _abandoned_call: AbandonedCall, /) -> None:
            """Raise instead of recording.

            Raises:
                RuntimeError: always.
            """
            raise RuntimeError("the application's log is broken")

    async def scenario() -> None:
        """Time out a consumer suspended on a hanging stream whose abandonment cannot be recorded."""
        bound_llm = LLM(
            _FakeAdapter(stream=_HangingStream()), rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)

        async def drain() -> None:
            """Enter and iterate into the hang; the wait_for below cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=_RaisingLog()
            ) as handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_during_the_open_appends_one_abandoned_call() -> None:
    """A cancellation landing in __aenter__ still records the abandonment.

    __aexit__ never runs when __aenter__ raises, so only __aenter__ itself can append here.
    """

    async def scenario() -> None:
        """Time out an entry whose open never returns, then read the log."""
        abandoned_call_log: list[AbandonedCall] = []
        bound_llm = LLM(_FakeAdapter(hang_from_open=1), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns; the wait_for below cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ):
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        assert len(abandoned_call_log) == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_completed_or_left_early_appends_no_abandoned_call() -> None:
    """final() completing, or a consumer leaving the block voluntarily, appends nothing.

    Only a cancellation gets a record: an assembled Response reached the caller, and a voluntary
    early exit is the caller walking away in live code.
    """

    async def scenario() -> None:
        """Consume one stream to final(), leave a second before its first item."""
        abandoned_call_log: list[AbandonedCall] = []
        bound_llm = LLM(_FakeAdapter(), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one(
            [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
        ) as handle:
            await handle.final()
        second_adapter = _FakeAdapter(stream=_HangingStream())
        second_bound_llm = LLM(second_adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with second_bound_llm.stream_one(
            [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
        ):
            pass
        assert abandoned_call_log == []

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_after_final_raised_appends_no_abandoned_call() -> None:
    """A cancellation after final() raised its GenerationError appends nothing.

    The RefusalError carried its usage to the caller, so an AbandonedCall here would
    double-count that spend and mislabel a concluded call as an in-flight abandonment.
    """

    async def scenario() -> None:
        """Absorb a refusal from final() inside the block, then hang into the caller's deadline."""
        abandoned_call_log: list[AbandonedCall] = []
        adapter = _FakeAdapter(stream=_RefusingStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )

        async def consume() -> None:
            """Let final() report Refusal, then sleep; the wait_for below cancels this inside the block."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ) as handle:
                with pytest.raises(RefusalError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert abandoned_call_log == []

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_after_a_protocol_error_appends_the_abandoned_call() -> None:
    """A StreamProtocolError accounts for nothing, so a cancellation after one still logs the call.

    The record is the only account of the stream that was opened: the error carries no model,
    no attempt records, and no usage.
    """

    async def scenario() -> None:
        """Absorb the protocol error inside the block, then hang into the caller's deadline."""
        abandoned_call_log: list[AbandonedCall] = []
        bound_llm = LLM(
            _FakeAdapter(stream=_ProtocolErrorStream()), rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)

        async def consume() -> None:
            """Let final() hit the protocol error, then sleep; the wait_for below cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ) as handle:
                with pytest.raises(StreamProtocolError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        (abandoned_call,) = abandoned_call_log
        assert abandoned_call.model == "fake-model"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


_MID_STREAM_RETRY_AFTER_SECONDS = 30.0
"""The server-stated wait _StatesRetryAfterAdapter reports, under the limiter's 60-second cap so it passes through."""


class _StatesRetryAfterAdapter(_FakeAdapter):
    """A _FakeAdapter reporting a fixed server-stated wait on every failure it is asked about."""

    @override
    def retry_after_seconds(self, error: Exception) -> float | None:
        """Report the fixed wait, whatever the failure."""
        return _MID_STREAM_RETRY_AFTER_SECONDS


def test_a_mid_stream_rate_limit_pauses_the_account() -> None:
    """A rate limit that lands after the first item still pauses admission and reaches the caller.

    This stream is past reopening, so nothing here retries; the pause protects every other caller
    sharing the limiter, and dropping it because this one stream is finished would leave them all
    sending into the limit. The raised TransientError carries the same verdict and the same
    server-stated wait, so an application reading it sees the rate limit rather than an
    unclassified failure.
    """

    async def scenario() -> None:
        """Let the iteration fail after one item, then read the limiter's state and the error."""
        rate_limiter = _fast_rate_limiter()
        bound_llm = LLM(
            _StatesRetryAfterAdapter(
                stream=_FailsAfterFirstItemStream(), classify_result="rate_limit"
            ),
            rate_limiter=rate_limiter,
        ).bind(automatic_prompt_caching=True)
        before_monotonic_seconds = time.monotonic()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(TransientError) as raised:
                async for _item in handle:
                    pass
        assert raised.value.is_rate_limit
        assert raised.value.retry_after_seconds == _MID_STREAM_RETRY_AFTER_SECONDS
        assert rate_limiter._recovering
        # A server-stated wait is followed un-jittered, so the pause is that wait from the failure.
        assert (
            rate_limiter._paused_until
            >= before_monotonic_seconds + _MID_STREAM_RETRY_AFTER_SECONDS
        )

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


class _RaisesItsOwnTransientErrorStream(_FakeStream):
    """A stream that yields one item and then raises a TransientError it classified itself.

    Mirrors an adapter that reads the provider's rate-limit verdict and its retry-after header and
    states both, rather than leaving the handle to ask the adapter about a bare exception.
    """

    def __init__(self) -> None:
        """Hold the error to be raised, so a test can compare it by identity."""
        super().__init__()
        self.error = TransientError(
            "rate limited mid-stream",
            is_rate_limit=True,
            retry_after_seconds=_MID_STREAM_RETRY_AFTER_SECONDS,
        )

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield one chunk, then raise the held TransientError.

        Yields:
            One text chunk, before the raise.

        Raises:
            TransientError: after the first yield, stating its own verdict and wait.
        """
        yield "a"
        raise self.error


def test_an_adapter_stated_mid_stream_transient_error_reaches_the_caller_unwrapped() -> None:
    """An adapter that states the verdict itself keeps it, and does not become its own cause.

    Wrapping it again would replace the adapter's own message, and raising the wrapper `from` the
    same object would make the error its own __cause__.
    """

    async def scenario() -> None:
        """Let the iteration fail after one item and read the error the caller catches."""
        stream = _RaisesItsOwnTransientErrorStream()
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(TransientError) as raised:
                async for _item in handle:
                    pass
        assert raised.value is stream.error
        assert raised.value.__cause__ is not raised.value
        assert raised.value.is_rate_limit
        assert raised.value.retry_after_seconds == _MID_STREAM_RETRY_AFTER_SECONDS

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


class _CloseRaisesBaseExceptionStream(_FakeStream):
    """A stream whose close() raises a BaseException, which _close_adapter_stream does not catch."""

    @override
    async def close(self) -> None:
        """Mark the stream closed, then raise past every Exception handler.

        Raises:
            KeyboardInterrupt: always, after recording that close ran.
        """
        self.closed = True
        raise KeyboardInterrupt("interrupted during teardown")


def test_a_close_raising_a_base_exception_still_appends_the_abandoned_call() -> None:
    """The record survives a teardown that raises past every Exception handler.

    __aexit__ closes before it appends, so that the record reports a returned slot and a closed
    connection. Without the append in a finally, the one exception the close does not swallow
    would take the cancelled stream's only account with it.
    """

    async def scenario() -> None:
        """Cancel the block, then let the close raise on the way out."""
        abandoned_call_log: list[AbandonedCall] = []
        bound_llm = LLM(
            _FakeAdapter(stream=_CloseRaisesBaseExceptionStream(), classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)

        async def consume() -> None:
            """Hang inside the block; the wait_for cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ):
                await asyncio.sleep(60)

        with pytest.raises(KeyboardInterrupt):
            await asyncio.wait_for(consume(), timeout=0.05)
        (abandoned_call,) = abandoned_call_log
        assert abandoned_call.model == "fake-model"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_after_a_mid_stream_failure_appends_the_abandoned_call() -> None:
    """A failure after the first item is no conclusion, so a cancellation after one logs the call.

    The TransientError names no model and carries no billing, so the record is the only account
    of the stream that was opened. Its one settled attempt is the mid-stream failure, which bills
    nothing, so usage_settled reads zero while the stream was paid for the item it delivered.
    """

    async def scenario() -> None:
        """Absorb the mid-stream failure inside the block, then hang into the caller's deadline."""
        abandoned_call_log: list[AbandonedCall] = []
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream(), classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)

        async def consume() -> None:
            """Let the iteration fail after one item, then sleep; the wait_for cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ) as handle:
                with pytest.raises(TransientError, match="after items were yielded"):
                    async for _item in handle:
                        pass
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        (abandoned_call,) = abandoned_call_log
        assert abandoned_call.model == "fake-model"
        (record,) = abandoned_call.attempt_records
        assert isinstance(record.error, TransientError)
        assert abandoned_call.usage_settled == ZERO_USAGE

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_after_a_drain_failure_appends_no_abandoned_call() -> None:
    """A GenerationError raised while draining carries the call, so a cancellation logs nothing.

    The error already handed the caller a CallRecord, so an AbandonedCall here would report the
    same call twice.
    """

    async def scenario() -> None:
        """Absorb an unrecognized item failure inside the block, then hang into the deadline."""
        abandoned_call_log: list[AbandonedCall] = []
        bound_llm = LLM(
            _FakeAdapter(stream=_UnnamedItemErrorStream(), classify_result="unrecognized"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)

        async def consume() -> None:
            """Let final() hit the unrecognized failure, then sleep; the wait_for cancels this."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ) as handle:
                with pytest.raises(UnrecognizedError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert abandoned_call_log == []

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


def test_stream_unentered_handle_raises_instead_of_opening() -> None:
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


def test_stream_handle_raises_on_a_second_entry() -> None:
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


def test_stream_final_refusal_raises_row_shaped_without_retry() -> None:
    """A structured refusal detected in the stream's final() surfaces as a row-shaped RefusalError.

    The stream already yielded items to the caller, so the error is not retried;
    final() records the one 200 that produced no output and raises the RefusalError carrying that record.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports Refusal, then read the raised RefusalError."""
        adapter = _FakeAdapter(stream=_RefusingStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RefusalError) as refusal:
                await handle.final()
        failure = refusal.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.usage.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_stream_final_unfinished_turn_raises_carrying_the_adapter_s_reason() -> None:
    """An UnfinishedTurn from the stream's final() fails the call, carrying the adapter's reason.

    The stream loop builds this error separately from the generate loop, and it is the one member
    with a field of its own, so the reason must survive the trip through both.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports UnfinishedTurn, then read the raised error."""
        adapter = _FakeAdapter(stream=_UnfinishedTurnStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(UnfinishedTurnError) as unfinished_turn:
                await handle.final()
        failure = unfinished_turn.value
        assert failure.attempts == 1
        assert "pause_turn" in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = failure.attempt_records
        assert record.error is None

    asyncio.run(scenario())


def test_stream_final_schema_violation_raises_carrying_the_rejection() -> None:
    """A SchemaViolation from the stream's final() fails the call, carrying pydantic's rejection.

    The stream loop builds this error separately from the generate loop, so the rejection must
    survive the trip through both, and stay out of error_text on both.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports SchemaViolation, then read the raised error."""
        adapter = _FakeAdapter(stream=_SchemaViolationStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(SchemaViolationError) as schema_violation:
                await handle.final()
        failure = schema_violation.value
        assert failure.attempts == 1
        assert failure.validation_error_json == _VALIDATION_ERROR_JSON
        assert "SENTINEL" not in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = failure.attempt_records
        assert record.error is None

    asyncio.run(scenario())


def test_stream_final_provider_failed_transiently_raises_transient_carrying_its_billing() -> None:
    """The outcome propagates as a TransientError carrying the provider's reason and the billing.

    The stream already yielded items to the caller, so it is not reopened; that 200's usage
    reaches the caller on the error itself rather than on an attempt record.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports the failure, then read the raised error."""
        adapter = _FakeAdapter(stream=_ProviderFailedTransientlyStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(TransientError) as provider_failure:
                await handle.final()
        assert str(provider_failure.value) == _PROVIDER_FAILURE_REASON
        assert provider_failure.value.usage.cost_in_usd == 0.25
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_stream_final_provider_failed_terminally_raises_carrying_the_providers_reason() -> None:
    """The outcome fails the call once, and the provider's own text is the error's message."""

    async def scenario() -> None:
        """Drain a stream whose final() reports the terminal failure, then read the raised error."""
        adapter = _FakeAdapter(stream=_ProviderFailedTerminallyStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(ProviderFailedTerminallyError) as provider_failure:
                await handle.final()
        failure = provider_failure.value
        assert failure.attempts == 1
        assert _PROVIDER_FAILURE_REASON in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = failure.attempt_records
        assert record.error is None

    asyncio.run(scenario())


def test_stream_cancelled_after_absorbing_a_provider_failure_appends_no_abandoned_call() -> None:
    """A cancellation after final() raised its TransientError appends nothing.

    The error carried that 200's usage to the caller, so an AbandonedCall here would
    double-count that spend and mislabel a concluded call as an in-flight abandonment.
    """

    async def scenario() -> None:
        """Absorb the failure from final() inside the block, then hang into the caller's deadline."""
        abandoned_call_log: list[AbandonedCall] = []
        adapter = _FakeAdapter(stream=_ProviderFailedTransientlyStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )

        async def consume() -> None:
            """Let final() report the failure, then sleep; the wait_for below cancels in the block."""
            async with bound_llm.stream_one(
                [UserMessage(content="hi")], abandoned_call_log=abandoned_call_log
            ) as handle:
                with pytest.raises(TransientError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert abandoned_call_log == []

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_retry_populates_attempt_records() -> None:
    """A pre-first-item connection failure lands as an errored record before the success record."""

    async def scenario() -> None:
        """Open a stream whose first open_stream call fails, then drain it."""
        adapter = _FakeAdapter(open_failures=[TransientError("connection reset")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        assert response.output == "ab"
        assert response.attempts == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "connection reset"
        assert succeeded.error is None
        assert (
            failed.started_at_monotonic_seconds
            <= failed.ended_at_monotonic_seconds
            <= succeeded.started_at_monotonic_seconds
            <= succeeded.ended_at_monotonic_seconds
        )

    asyncio.run(scenario())


def test_stream_open_classified_invalid_request_carries_the_prior_attempts_records() -> None:
    """A rejected open raises InvalidRequestError from the entry with the prior transient record."""

    async def scenario() -> None:
        """Enter a handle whose first open fails transiently and whose second is rejected."""
        adapter = _FakeAdapter(
            open_failures=[TransientError("connection reset"), ValueError("boom")],
            classify_result="invalid_request",
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert isinstance(rejected.value.__cause__, ValueError)
        assert rejected.value.reason == "the provider rejected the request: boom"
        transient_record, rejected_record = rejected.value.attempt_records
        assert str(transient_record.error) == "connection reset"
        assert rejected_record.error is None
        assert rejected_record.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_stream_open_reported_invalid_request_reaches_the_caller_row_shaped() -> None:
    """An adapter that reports the conversation as InvalidRequest fails the item, row-shaped.

    The handle builds the InvalidRequestError, so a caller reading model or attempt_records off it succeeds.
    """

    async def scenario() -> None:
        """Enter a handle whose open reports InvalidRequest."""
        adapter = _FakeAdapter(open_failures=[InvalidRequest(reason="nope")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert rejected.value.reason == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.attempt_records == ()

    asyncio.run(scenario())


def test_stream_open_classified_unrecognized_raises_the_items_failure() -> None:
    """An open failure classified unrecognized raises UnrecognizedError from the entry, unretried."""

    async def scenario() -> None:
        """Enter a handle whose open raises a classify-unrecognized exception."""
        adapter = _FakeAdapter(open_failures=[ValueError("boom")], classify_result="unrecognized")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnrecognizedError) as unrecognized:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(unrecognized.value.error, ValueError)
        assert unrecognized.value.error_text == "unrecognized provider error: boom"

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
    idle_seconds = 0.02

    async def scenario() -> None:
        """Drain the stream, idle, then call final()."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                pass
            drained_at = time.monotonic()
            await asyncio.sleep(idle_seconds)
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.ended_at_monotonic_seconds <= drained_at
        assert response.elapsed_seconds < idle_seconds

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


@pytest.mark.parametrize(
    ("stream", "expected_error"),
    [
        (_RefusingStream(), RefusalError),
        (_ProviderFailedTransientlyStream(), TransientError),
        (_ProtocolErrorStream(), StreamProtocolError),
    ],
    ids=["refusal", "provider_failed_transiently", "protocol_error"],
)
def test_stream_final_replays_every_error_that_concluded_the_call(
    stream: _FakeStream, expected_error: type[Exception]
) -> None:
    """A second final() raises the first call's error again and records no second attempt.

    Covers both routes to a conclusion, the outcome final() reads and the error draining raises,
    because either one re-run would ask the adapter stream again and bill the same request twice
    in the attempt records.
    """

    async def scenario() -> None:
        """Call final() twice on a stream whose call cannot end in a Response."""
        bound_llm = LLM(_FakeAdapter(stream=stream), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(expected_error) as first:
                await handle.final()
            records_after_first = handle._ledger.attempt_records
            with pytest.raises(expected_error) as second:
                await handle.final()
            assert handle._ledger.attempt_records == records_after_first
        assert second.value is first.value

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_final_replays_a_raise_from_the_adapter_stream() -> None:
    """A raise from the adapter stream's final() concludes the call, so a second call replays it.

    Without the store, a second call asks the adapter stream again,
    re-running an assembly whose one request is already paid for.
    """

    async def scenario() -> None:
        """Call final() twice on a stream whose own final() raises."""
        stream = _FinalRaisesStream()
        bound_llm = LLM(_FakeAdapter(stream=stream), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RuntimeError, match="assembly failed") as first:
                await handle.final()
            with pytest.raises(RuntimeError, match="assembly failed") as second:
                await handle.final()
        assert second.value is first.value
        assert stream.final_calls == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


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
    monkeypatch.setattr(rate_limiter_module.random, "uniform", uniform_returns_ceiling)

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
