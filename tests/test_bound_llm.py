"""BoundLLM and StreamHandle driven by fake adapters.

A fake BoundAdapter scripts send to fail a fixed number of times before succeeding,
and a fake AdapterStream emits a fixed item sequence.
Together they pin the retry loop, rebind rebuild, batch ordering, and the stream contract without any network access.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, assert_type, override

import pytest
from pydantic import BaseModel

from langchaint import (
    LLM,
    ZERO_USAGE,
    AssistantMessage,
    Billing,
    BoundLLM,
    ContextWindowExceededError,
    EmptyTurnError,
    EscapedExceptionError,
    GenerationError,
    HasTools,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    Message,
    NoTools,
    ProviderDeclaredFinalError,
    ProviderFailedTerminallyError,
    RateLimiter,
    RefusalError,
    Response,
    RetriesExhaustedError,
    RetryUnavailableError,
    SchemaViolationError,
    StopReason,
    StreamItem,
    StreamProtocolError,
    TextPart,
    TimedOutError,
    ToolCall,
    ToolManager,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
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
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.call import ResponseIdentity
from langchaint.llm import UNCHANGED, Unchanged
from langchaint.streaming import StreamHandle
from tests.helpers import stated_billing, uniform_returns_ceiling

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

    Its id is what the fake bound adapter looks the scripted response up under, standing in for the
    fields a real adapter reads the turn and the counters off.
    request_id stands in for the header both SDKs attach to a response they parsed from an HTTP
    body, so a response a stream assembled leaves it None as a real one does.
    """

    id: str
    request_id: str | None = None


def _as_fake_raw(raw: BaseModel) -> _FakeRawResponse:
    """Narrow a raw response to the fake one.

    Raises:
        TypeError: raw is not a _FakeRawResponse, which the real adapters raise for the same reason.
    """
    if not isinstance(raw, _FakeRawResponse):
        raise TypeError(f"expected a _FakeRawResponse, got {type(raw).__name__}")
    return raw


@dataclass(frozen=True, kw_only=True)
class _ScriptedResponse:
    """One response the fake hands back: what interpret reads off it, and what it billed."""

    outcome: ResponseOutcome[str]
    usage: Usage


def _billed(outcome: ResponseOutcome[str]) -> _ScriptedResponse:
    """Script one 200 the provider billed, whatever interpret goes on to make of it."""
    return _ScriptedResponse(outcome=outcome, usage=_USAGE_BILLED)


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

    final() hands back the response object, exactly as a real adapter's stream does;
    scripted_response() is what the fake bound adapter registers for it, standing in for interpret
    and billing_from_raw reading that response.
    """

    def __init__(self) -> None:
        """Start unclosed; close records that it ran."""
        self.closed = False
        self.raw = _FakeRawResponse(id="fake-final")
        self._usage_reported: Usage | None = None
        """What billing_reported wraps; None stands for an adapter with no such channel."""

    @override
    def billing_reported(self) -> Billing | None:
        """Wrap whatever the test set, defaulting to the None an openai stream returns."""
        return None if self._usage_reported is None else stated_billing(self._usage_reported)

    @override
    def request_id(self) -> str | None:
        """Report a fixed header, standing in for the response headers a real SDK stream reads."""
        return "req-fake-stream"

    def scripted_response(self) -> _ScriptedResponse:
        """Return the assembled result the SDK would produce, and what the stream billed."""
        return _ScriptedResponse(
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
    def scripted_response(self) -> _ScriptedResponse:
        """Report the refusal instead of a result, carrying this attempt's billing."""
        return _billed(Refusal(assistant_message=_REJECTED_TURN))


class _UnfinishedTurnStream(_FakeStream):
    """A stream that yields items normally but whose assembled message is not a finished turn.

    Mirrors an adapter that reads a stop reason it cannot call finished, reporting UnfinishedTurn
    with the reason naming the provider's own word.
    """

    @override
    def scripted_response(self) -> _ScriptedResponse:
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
    def scripted_response(self) -> _ScriptedResponse:
        """Report the rejection instead of a result, carrying this attempt's billing."""
        return _billed(
            SchemaViolation(
                validation_error_json=_VALIDATION_ERROR_JSON, assistant_message=_REJECTED_TURN
            )
        )


class _MaxCompletionTokensExceededStream(_FakeStream):
    """A stream whose assembled response reached the token cap before its JSON closed."""

    @override
    def scripted_response(self) -> _ScriptedResponse:
        """Report the truncation instead of a result, carrying this attempt's billing."""
        return _billed(MaxCompletionTokensExceeded(assistant_message=_REJECTED_TURN))


class _EmptyTurnStream(_FakeStream):
    """A stream whose assembled turn produced no instance and no tool call."""

    @override
    def scripted_response(self) -> _ScriptedResponse:
        """Report the empty turn instead of a result, carrying this attempt's billing."""
        return _billed(EmptyTurn(assistant_message=_REJECTED_TURN))


class _ContextWindowExceededStream(_FakeStream):
    """A stream whose assembled response reports the conversation overflowed the context window."""

    @override
    def scripted_response(self) -> _ScriptedResponse:
        """Report the overflow instead of a result, carrying this attempt's billing."""
        return _billed(ContextWindowExceeded(assistant_message=_REJECTED_TURN))


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
    def scripted_response(self) -> _ScriptedResponse:
        """Report the failure instead of a result, carrying this attempt's billing."""
        return _billed(_PROVIDER_FAILED_TRANSIENTLY)


class _YieldsNothingThenFailsTransientlyStream(_ProviderFailedTransientlyStream):
    """A stream whose run failed before emitting anything, so it yields no item at all."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Emit no item at all, so the failure is visible only in the assembled response.

        Yields:
            Nothing: the unreachable yield below is what makes this an async generator.
        """
        return
        yield  # pragma: no cover


class _ProviderFailedTerminallyStream(_FakeStream):
    """A stream whose assembled response reports a provider failure a resend would hit again."""

    @override
    def scripted_response(self) -> _ScriptedResponse:
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


class _SlowAfterFirstItemStream(_FakeStream):
    """A stream that waits a measurable interval before every item after its first."""

    gap_seconds = 0.02
    """Long enough to separate a stamp taken on the first item from one taken on any later item."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield the base sequence, sleeping before each item but the first.

        Yields:
            The base class's items, spread over gap_seconds apiece.
        """
        first = True
        async for item in super().items():
            if not first:
                await asyncio.sleep(self.gap_seconds)
            first = False
            yield item


class _FailsBeforeFirstItemStream(_FakeStream):
    """A stream whose first items() call fails transiently before yielding, then behaves normally.

    One instance is reused across reopens, so the counter records which items() call is running.
    """

    def __init__(self) -> None:
        """Start with no items() call made."""
        super().__init__()
        self.items_calls = 0

    def _first_call_error(self) -> Exception:
        """Return the error the first items() call raises."""
        return TransientError("dropped before the first item")

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Fail before the first yield on the first call, else yield the base sequence.

        Yields:
            Nothing on the first call; the base class's items on every later call.

        Raises:
            Exception: _first_call_error's, on the first call, before the first yield.
        """
        self.items_calls += 1
        if self.items_calls == 1:
            raise self._first_call_error()
        async for item in super().items():
            yield item


class _FailsWithARequestIdBeforeFirstItemStream(_FailsBeforeFirstItemStream):
    """A stream whose first items() call fails on an error naming its own request.

    That error is not a TransientError, so reopening it takes an adapter whose classify calls it
    transient.
    """

    @override
    def _first_call_error(self) -> Exception:
        """Name an error carrying the request id of the attempt it ends."""
        return _RequestIdError("dropped before the first item", "req-from-items-error")


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


class _HangsAfterFirstItemStream(_FakeStream):
    """A stream that delivers one item then suspends forever, to be cancelled mid-iteration."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield one item, then wait on an event that never fires.

        Yields:
            One item, and nothing after it.
        """
        yield "a"
        await asyncio.Event().wait()


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


type _ScriptedSend = Exception | _ScriptedResponse
"""One scripted send: an exception the fake raises, or a response it hands back.

The two exist because the adapter contract splits that way: an attempt with no response to read is
an exception for Adapter.classify, and a response is what send returns and interpret then reads.
A conversation the fake will not put on the wire is its invalid_request, which build_request reports
before any send.
"""


@dataclass(frozen=True, kw_only=True)
class _FakeRequest(RequestParams):
    """What the fake would put on the wire, which is the conversation and nothing else."""

    conversation: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        """Render the conversation as a JSON array of each message's dump."""
        return json.dumps([message.model_dump(mode="json") for message in self.conversation])


def _as_fake_request(request: RequestParams) -> _FakeRequest:
    """Narrow a request to the fake one.

    Raises:
        TypeError: request is not a _FakeRequest, which the real adapters raise for the same reason.
    """
    if not isinstance(request, _FakeRequest):
        raise TypeError(f"expected a _FakeRequest, got {type(request).__name__}")
    return request


class _FakeBoundAdapter(BoundAdapter[str]):
    """A bound adapter whose send follows a scripted failure sequence."""

    def __init__(
        self,
        *,
        failures: Sequence[_ScriptedSend] = (),
        open_failures: Sequence[Exception] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        send_seconds: float = 0.0,
        hang_from_open: int | None = None,
        hang_from_send: int | None = None,
    ) -> None:
        """Store the failure scripts, echo mode, and the stream open_stream returns.

        failures scripts send; open_failures scripts open_stream, exercising the pre-first-item stream retry path.
        invalid_requests scripts build_request, one entry per call, and an entry is reported
        instead of a request, so neither send nor open_stream is reached for that call.
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
        self._invalid_requests = list(invalid_requests)
        self._echo = echo
        self._send_seconds = send_seconds
        self._hang_from_open = hang_from_open
        self._hang_from_send = hang_from_send
        self.stream = stream if stream is not None else _FakeStream()
        self._scripted_by_raw_id: dict[str, _ScriptedResponse] = {}
        self.sent_raws: list[_FakeRawResponse] = []
        self.build_count = 0
        self.send_count = 0
        self.open_count = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Return what the response under this raw was scripted to have billed."""
        return stated_billing(self._scripted_by_raw_id[_as_fake_raw(raw).id].usage)

    @override
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Name the fake model, take the response id from the raw's own, and the request id as it came."""
        fake_raw = _as_fake_raw(raw)
        return ResponseIdentity(
            model_served="fake-model-served",
            response_id=fake_raw.id,
            request_id=fake_raw.request_id,
        )

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Return what the response under this raw was scripted to produce."""
        return self._scripted_by_raw_id[_as_fake_raw(raw).id].outcome

    @override
    def build_request(self, conversation: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Report the scripted refusal, else carry the conversation into the request."""
        if self._invalid_requests:
            return self._invalid_requests.pop(0)
        self.build_count += 1
        return _FakeRequest(conversation=tuple(conversation))

    @override
    async def send(self, request: RequestParams) -> BaseModel:
        """Raise or return the next scripted send, else hand back a success response.

        Raises:
            TypeError: request is not a _FakeRequest.
        """
        conversation = _as_fake_request(request).conversation
        self.send_count += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._hang_from_send is not None and self.send_count >= self._hang_from_send:
                await asyncio.Event().wait()
            if self._send_seconds:
                await asyncio.sleep(self._send_seconds)
            raw = _FakeRawResponse(
                id=f"fake-response-{self.send_count}",
                request_id=f"req-fake-response-{self.send_count}",
            )
            if self._failures:
                scripted = self._failures.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                self._scripted_by_raw_id[raw.id] = scripted
                self.sent_raws.append(raw)
                return raw
            first = conversation[0]
            content = (
                first.content
                if self._echo and isinstance(first, UserMessage) and isinstance(first.content, str)
                else "ok"
            )
            self._scripted_by_raw_id[raw.id] = _ScriptedResponse(
                outcome=_success_result(content), usage=_USAGE
            )
            self.sent_raws.append(raw)
            return raw
        finally:
            self.in_flight -= 1

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Count the attempt, suspend or raise the next scripted open failure, else return the stored fake stream.

        Raises:
            TypeError: request is not a _FakeRequest.
            Exception: the next scripted open failure.
        """
        _ = _as_fake_request(request)
        self.open_count += 1
        if self._hang_from_open is not None and self.open_count >= self._hang_from_open:
            await asyncio.Event().wait()
        if self._open_failures:
            raise self._open_failures.pop(0)
        self._scripted_by_raw_id[self.stream.raw.id] = self.stream.scripted_response()
        return self.stream


class _FakeStructuredBoundAdapter[ModelT: BaseModel](BoundAdapter[ModelT]):
    """A structured bound adapter for response_format rebind tests; it never generates.

    Those tests check binding identity and the switched content type, not structured output,
    so send and open_stream stay unreachable.
    """

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT]:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    def build_request(self, conversation: Sequence[Message]) -> RequestParams:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def send(self, request: RequestParams) -> BaseModel:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Unreachable: response_format rebind tests do not stream."""
        raise NotImplementedError


class _RequestIdError(RuntimeError):
    """An error carrying the request-id header its response had, as both SDKs' APIStatusError does."""

    def __init__(self, message: str, request_id: str) -> None:
        super().__init__(message)
        self.request_id = request_id


class _TransientRequestIdError(TransientError):
    """The same id, on the class an adapter raises to retry an attempt without going through classify."""

    def __init__(self, message: str, request_id: str) -> None:
        super().__init__(message)
        self.request_id = request_id


class _FakeAdapter(Adapter):
    """An adapter whose bind_text hands out fake bound adapters."""

    _bound_adapter_class: ClassVar[type[_FakeBoundAdapter]] = _FakeBoundAdapter
    """The class bind_text hands out; a subclass names its own to vary what interpret does."""

    def __init__(
        self,
        *,
        failures: Sequence[_ScriptedSend] = (),
        open_failures: Sequence[Exception] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        classify_result: ErrorClassification = "unknown_exception",
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
        self._invalid_requests = invalid_requests
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
            invalid_requests=self._invalid_requests,
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

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        """Read the request id off the errors that carry one, as each SDK's adapter does."""
        if isinstance(error, (_RequestIdError, _TransientRequestIdError)):
            return error.request_id
        return None


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
        with pytest.raises(UnknownExceptionError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = unplaceable.value.attempt_records
        assert isinstance(record.raw, _FakeRawResponse)
        assert record.usage == _USAGE
        assert record.assistant_message is None
        assert record.error is None
        assert isinstance(unplaceable.value.__cause__, RuntimeError)

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

    The succeeding record carries the answer turn and what the response said about itself, and the
    failed one carries neither: its request never came back.
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
        assert (failed.model_served, failed.response_id, failed.request_id) == (None, None, None)
        assert succeeded.error is None
        assert succeeded.assistant_message == _success_result("ok").assistant_message
        assert succeeded.model_served == "fake-model-served"
        assert succeeded.response_id == _as_fake_raw(response.raw).id
        assert succeeded.request_id == f"req-{_as_fake_raw(response.raw).id}"
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


def test_a_call_builds_one_request_and_sends_it_once_per_attempt() -> None:
    """Two transient failures then a success is one build and three sends, on both request paths.

    Building per attempt would let a retry put different bytes on the wire than the attempt it
    retries, so a failure's archived request would not be what the later attempts sent.
    The third case is the stream's other opening site, the reopen after a failure before the first
    item, which reaches open_stream without passing the entry that built the request.
    """

    async def streamed_counts(adapter: _FakeAdapter) -> tuple[int, int]:
        """Drain one stream over adapter and return its bound adapter's build and open counts."""
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            await handle.final()
        bound = adapter.bound_adapters[0]
        return bound.build_count, bound.open_count

    async def scenario() -> None:
        """Drive one generate_one and two streams, each through a failure that reopens or retries."""
        sent_adapter = _FakeAdapter(
            failures=[TransientError("boom"), TransientError("boom again")]
        )
        sent_llm = LLM(sent_adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        await sent_llm.generate_one([UserMessage(content="hi")])
        sent_bound = sent_adapter.bound_adapters[0]
        assert (sent_bound.build_count, sent_bound.send_count) == (1, 3)

        retried = _FakeAdapter(
            open_failures=[TransientError("boom"), TransientError("boom again")]
        )
        assert await streamed_counts(retried) == (1, 3)
        assert await streamed_counts(_FakeAdapter(stream=_FailsBeforeFirstItemStream())) == (1, 2)

    asyncio.run(scenario())


def test_a_failed_attempt_records_the_request_id_off_its_error() -> None:
    """An attempt that received no response still carries the request id its error named.

    The three attempts name three different requests: the second failure names none, so an id on its
    row would be the first attempt's outliving the attempt that read it, and the third attempt's own
    id comes from the response rather than from either error.
    """

    async def scenario() -> None:
        """Drive one generate_one through two transient failures, the first naming its request."""
        adapter = _FakeAdapter(
            failures=[_RequestIdError("boom", "req-from-error"), TransientError("boom again")],
            classify_result="transient",
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        named, unnamed, succeeded = response.attempt_records
        assert named.request_id == "req-from-error"
        assert named.response_id is None
        assert unnamed.request_id is None
        assert succeeded.request_id == f"req-{_as_fake_raw(response.raw).id}"

    asyncio.run(scenario())


def test_an_adapter_raised_transient_error_still_names_its_request() -> None:
    """A TransientError the adapter raised itself never reaches classify, and is read for its id anyway.

    Both retry loops read it: the sent loop in its own except clause, and the stream loop ahead of
    the check that sends a transient error straight back for a retry.
    """

    async def scenario() -> None:
        """Fail one send and one stream open with a transient error naming its request."""
        sent_adapter = _FakeAdapter(
            failures=[_TransientRequestIdError("boom", "req-from-sent-transient")]
        )
        sent_llm = LLM(sent_adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        sent = await sent_llm.generate_one([UserMessage(content="hi")])
        assert sent.attempt_records[0].request_id == "req-from-sent-transient"

        stream_adapter = _FakeAdapter(
            open_failures=[_TransientRequestIdError("boom", "req-from-open-transient")]
        )
        bound_llm = LLM(stream_adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            streamed = await handle.final()
        assert streamed.attempt_records[0].request_id == "req-from-open-transient"

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


def test_a_conversation_build_request_refuses_fails_the_item_with_nothing_sent() -> None:
    """An InvalidRequest from build_request fails the item before any request goes out.

    classify returns "transient" here and is never reached: a returned outcome is not an exception,
    so no classify verdict can turn this into an attempt.
    The InvalidRequestError the loop builds carries the reason and the row-shape fields, and no
    request, there being none to carry.
    """

    async def scenario() -> None:
        """Drive one generate_one whose build_request refuses under a transient classify verdict."""
        adapter = _FakeAdapter(
            invalid_requests=[InvalidRequest(reason="nope")], classify_result="transient"
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 0
        assert rejected.value.request is None
        assert rejected.value.reason == "nope"
        assert rejected.value.error_text == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.provider_name == adapter.provider_name
        assert rejected.value.attempt_records == ()
        assert rejected.value.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_a_rejected_request_registers_no_success_with_the_rate_limiter() -> None:
    """A rejection is not a completed request, so it must not end the recovery.

    A registered success would clear a rate-limit pause that no provider response lifted.
    """

    async def scenario() -> None:
        """Put the limiter into recovery, then take a rejection while holding its probe slot."""
        rate_limiter = _fast_rate_limiter()
        failing_admission = await rate_limiter.acquire()
        rate_limiter.register_transient_error(
            failing_admission,
            (TransientError("429", retry_after_seconds=0.0, is_rate_limit=True),),
        )
        rate_limiter.release(failing_admission)
        assert rate_limiter._recovering
        adapter = _FakeAdapter(failures=[ValueError("nope")], classify_result="invalid_request")
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        with pytest.raises(InvalidRequestError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert rate_limiter._recovering

    asyncio.run(scenario())


def test_rejection_after_transient_attempts_carries_their_records() -> None:
    """An InvalidRequestError the provider's rejection raised carries the earlier attempts' records.

    The prior attempts' usage rides the error, so a billed 200 retried before the rejection stays
    accounted for. The rejected attempt went out, so it gets a record of its own, billing nothing.
    The error carries the request every one of those attempts sent.
    """

    async def scenario() -> None:
        """Settle one billed transient attempt, then have classify call the next one a rejection."""
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
        assert classified.value.request == _FakeRequest(conversation=(UserMessage(content="hi"),))

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


@pytest.mark.parametrize(
    ("outcome", "expected_error", "expected_stop_reason"),
    [
        (
            MaxCompletionTokensExceeded(assistant_message=_REJECTED_TURN),
            MaxCompletionTokensExceededError,
            "max_tokens",
        ),
        (EmptyTurn(assistant_message=_REJECTED_TURN), EmptyTurnError, "end_turn"),
        (
            ContextWindowExceeded(assistant_message=_REJECTED_TURN),
            ContextWindowExceededError,
            "context_window_exceeded",
        ),
    ],
    ids=["max_completion_tokens_exceeded", "empty_turn", "context_window_exceeded"],
)
def test_a_no_output_outcome_from_send_raises_row_shaped_without_retry(
    outcome: ResponseOutcome[str],
    expected_error: type[GenerationError],
    expected_stop_reason: StopReason,
) -> None:
    """Each outcome carrying no output fails the item on the first send, with its own error and reason.

    None of the three is retried: the token cap, the turn that finished saying nothing, and the
    request too long to serve all repeat on a resend.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send reports the outcome."""
        adapter = _FakeAdapter(failures=[_billed(outcome)])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(expected_error) as caught:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = caught.value
        assert failure.attempts == 1
        assert failure.stop_reason == expected_stop_reason
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_schema_violation_outcome_from_send_raises_row_shaped_without_retry() -> None:
    """A SchemaViolation outcome fails the item, and pydantic's rejection travels on the error.

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


def test_a_plain_exception_classified_transient_is_retried() -> None:
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


def test_exception_classified_unknown_exception_fails_the_item_without_retry() -> None:
    """A plain exception classified unknown_exception raises UnknownExceptionError on the first attempt.

    UnknownExceptionError is a GenerationError, so in a batch it becomes the item's failure row
    and the siblings run on. Nothing arrived, so the attempt it ends has no record.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-unknown_exception exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="unknown_exception")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnknownExceptionError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = unplaceable.value
        assert isinstance(failure, GenerationError)
        assert isinstance(failure.error, ValueError)
        assert isinstance(failure.__cause__, ValueError)
        assert failure.error_text == "langchaint could not place this exception: boom"
        assert failure.stop_reason is None
        assert failure.attempt_records == ()
        assert failure.model == adapter.model
        assert failure.provider_name == adapter.provider_name

    asyncio.run(scenario())


def test_exception_classified_declared_final_fails_the_item_with_a_record() -> None:
    """A plain exception classified declared_final raises ProviderDeclaredFinalError, unretried.

    The request reached the provider, which answered, so the attempt has a record and it carries
    ZERO_USAGE: nothing in that answer reported billing.
    """

    async def scenario() -> None:
        """Drive one generate_one whose send raises a classify-declared_final exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="declared_final")
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ProviderDeclaredFinalError) as declared_final:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].send_count == 1
        failure = declared_final.value
        assert isinstance(failure.error, ValueError)
        assert isinstance(failure.__cause__, ValueError)
        assert failure.error_text == "the provider declared this error final: boom"
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.raw is None
        assert record.usage == ZERO_USAGE
        assert failure.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_an_unplaceable_exception_becomes_the_items_failure_row_and_siblings_continue() -> None:
    """A classify-unknown_exception item comes back as its UnknownExceptionError row; the sibling succeeds."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_in_flight=1) whose first send is unplaceable."""
        adapter = _FakeAdapter(
            echo=True, failures=[ValueError("boom")], classify_result="unknown_exception"
        )
        rate_limiter = _fast_rate_limiter(max_in_flight=1)
        bound_llm = LLM(adapter, rate_limiter=rate_limiter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, UnknownExceptionError)
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(scenario())


def test_a_cancelled_batch_propagates_and_leaves_no_result_behind() -> None:
    """A cancellation from outside generate_many takes the whole batch, settled items included.

    The returned list is the batch's own frame's, so the cancellation destroys it before any caller
    reads it.
    """

    async def scenario() -> None:
        """Settle one item, then cancel the batch while the other's send hangs."""
        adapter = _FakeAdapter(hang_from_send=2)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_in_flight=1)).bind(
            automatic_prompt_caching=True
        )
        call = asyncio.create_task(
            bound_llm.generate_many([[UserMessage(content="a")], [UserMessage(content="b")]])
        )
        await asyncio.sleep(0.02)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert adapter.bound_adapters[0].send_count == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


@pytest.mark.parametrize(
    ("new_system_prompt", "expected_system_prompt", "binding_is_equal"),
    [(UNCHANGED, "s", True), ("s2", "s2", False)],
    ids=["all_unchanged", "changed_field"],
)
def test_rebind_builds_a_new_bound_adapter_whether_or_not_the_binding_changed(
    new_system_prompt: str | Unchanged, expected_system_prompt: str, *, binding_is_equal: bool
) -> None:
    """A rebind always binds again; the Binding is what tracks whether a field actually changed."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(system_prompt="s", automatic_prompt_caching=True)
    rebound = bound_llm.rebind(system_prompt=new_system_prompt)
    assert (rebound.binding == bound_llm.binding) is binding_is_equal
    assert rebound.binding.system_prompt == expected_system_prompt
    assert rebound._bound_adapter is not bound_llm._bound_adapter
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
        """Serialize a two-item batch (max_in_flight=1) whose first build_request refuses."""
        adapter = _FakeAdapter(
            echo=True, invalid_requests=[InvalidRequest(reason="misconfigured")]
        )
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


class _ClassifyRaisesAdapter(_FakeAdapter):
    """A _FakeAdapter whose classify raises, standing in for a defect anywhere in langchaint.

    classify runs inside the retry loop, past every handler that turns a request outcome into a
    GenerationError, so what it raises reaches the outermost guard the way a defect in langchaint's
    own machinery would.
    """

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Raise instead of classifying.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("classify defect")


def test_a_defect_becomes_one_items_row_and_leaves_the_batch_complete() -> None:
    """An Exception from langchaint's own machinery fails its item and no sibling.

    Propagating it instead would discard the whole returned list, so a sibling that had already
    settled would lose both its output and the account of what it spent.
    """

    async def scenario() -> None:
        """Raise past the retry loop on the one item whose send fails, and let the other succeed."""
        adapter = _ClassifyRaisesAdapter(failures=[ValueError("defect")], echo=True)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        failures = [result for result in results if isinstance(result, EscapedExceptionError)]
        responses = [result for result in results if isinstance(result, Response)]
        assert len(failures) == 1
        assert len(responses) == 1
        failure = failures[0]
        assert isinstance(failure.error, RuntimeError)
        assert failure.__cause__ is failure.error
        assert "classify defect" in failure.error_text
        assert failure.request is None
        # The attempt was in flight when the defect escaped, so no attempt is settled.
        assert failure.call.attempt_records == ()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_one_raises_a_defect_as_a_generation_error() -> None:
    """generate_one's caller sees the defect as an EscapedExceptionError, not the bare exception.

    Its callers match on GenerationError to write a failure row, so a defect arriving as anything
    else escapes past their handling entirely.
    """

    async def scenario() -> None:
        """Fail the one send and let classify raise past the retry loop."""
        adapter = _ClassifyRaisesAdapter(failures=[ValueError("defect")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(EscapedExceptionError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        assert str(raised.value) == "an exception escaped langchaint: classify defect"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


class _ClassifyRaisesOverStagedResponseAdapter(_ClassifyRaisesAdapter):
    """A _ClassifyRaisesAdapter whose bound adapters price a response and then raise reading it."""

    _bound_adapter_class = _InterpretRaisesBoundAdapter


def test_a_defect_over_a_staged_response_keeps_the_attempt_and_its_billing() -> None:
    """A defect escaping after a 200 arrived leaves that attempt and what it billed on the record.

    The response was paid for before the defect happened, so dropping the attempt would report the
    call as having spent nothing when it spent one response's worth.
    """

    async def scenario() -> None:
        """Stage the response, raise from interpret, then raise again from classify placing it."""
        adapter = _ClassifyRaisesOverStagedResponseAdapter()
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(EscapedExceptionError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        (record,) = raised.value.attempt_records
        assert record.usage == _USAGE
        assert raised.value.usage == _USAGE
        assert raised.value.attempts == 1

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


def test_a_stream_cancelled_inside_the_block_sets_its_abandoned() -> None:
    """A cancellation unwinding the block sets abandoned to what the stream could state.

    The record's attempt_records are empty because the streaming request is the in-flight attempt:
    only pre-first-item open failures settle records on a stream.
    This stream reports no running usage, which is what an openai stream does, so usage folds to
    ZERO_USAGE and billing_in_flight is None.
    """

    async def scenario() -> None:
        """Time out a consumer suspended on a hanging stream, then read the handle."""
        adapter = _FakeAdapter(stream=_HangingStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def drain() -> None:
            """Enter and iterate into the hang; the wait_for below cancels this."""
            async with handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)
        abandoned = handle.abandoned
        assert abandoned is not None
        assert abandoned.attempt_records == ()
        assert abandoned.usage == ZERO_USAGE
        assert abandoned.billing_in_flight is None
        assert abandoned.model == adapter.model

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_cancelled_stream_reports_what_it_billed_before_the_cancellation() -> None:
    """A stream that can state its running spend puts it on abandoned.usage.

    No attempt settled, so the whole reported amount comes from billing_in_flight, and a caller
    reads one paid total without adding two fields.
    The read happens before the close, so a stream the close drops still reports.
    """

    async def scenario() -> None:
        """Time out a consumer on a hanging stream that reports a running spend."""
        stream = _HangingStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(_FakeAdapter(stream=stream), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def drain() -> None:
            """Enter and iterate into the hang; the wait_for below cancels this."""
            async with handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)
        abandoned = handle.abandoned
        assert abandoned is not None
        assert abandoned.attempt_records == ()
        assert abandoned.usage == _USAGE_STREAM

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


def test_a_stream_cancelled_during_the_open_sets_its_abandoned() -> None:
    """A cancellation landing in __aenter__ still records the abandonment.

    __aexit__ never runs when __aenter__ raises, so only __aenter__ itself can set it there.
    """

    async def scenario() -> None:
        """Time out an entry whose open never returns, then read the handle."""
        bound_llm = LLM(_FakeAdapter(hang_from_open=1), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns; the wait_for below cancels this."""
            async with handle:
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        assert handle.abandoned is not None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_completed_or_left_early_sets_no_abandoned() -> None:
    """final() completing, or a consumer leaving the block voluntarily, leaves abandoned None.

    Only a cancellation gets a record: an assembled Response reached the caller, and a voluntary
    early exit is the caller walking away in live code.
    """

    async def scenario() -> None:
        """Consume one stream to final(), leave a second before its first item."""
        bound_llm = LLM(_FakeAdapter(), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        completed = bound_llm.stream_one([UserMessage(content="hi")])
        async with completed:
            await completed.final()
        second_adapter = _FakeAdapter(stream=_HangingStream())
        second_bound_llm = LLM(second_adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        left_early = second_bound_llm.stream_one([UserMessage(content="hi")])
        async with left_early:
            pass
        assert completed.abandoned is None
        assert left_early.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_final_raised_sets_no_abandoned() -> None:
    """A cancellation after final() raised its GenerationError leaves abandoned None.

    The RefusalError carried its usage to the caller, so an AbandonedCallError here would
    double-count that spend and mislabel a concluded call as an in-flight abandonment.
    """

    async def scenario() -> None:
        """Absorb a refusal from final() inside the block, then hang into the caller's deadline."""
        adapter = _FakeAdapter(stream=_RefusingStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() report Refusal, then sleep; the wait_for below cancels this inside the block."""
            async with handle:
                with pytest.raises(RefusalError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_a_protocol_error_sets_its_abandoned() -> None:
    """A StreamProtocolError accounts for nothing, so a cancellation after one still records the call.

    The record is the only account of the stream that was opened: the error carries no model,
    no attempt records, and no usage.
    """

    async def scenario() -> None:
        """Absorb the protocol error inside the block, then hang into the caller's deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_ProtocolErrorStream()), rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() hit the protocol error, then sleep; the wait_for below cancels this."""
            async with handle:
                with pytest.raises(StreamProtocolError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        abandoned = handle.abandoned
        assert abandoned is not None
        assert abandoned.model == "fake-model"

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
    sending into the limit. The RetryUnavailableError's __cause__ carries the same verdict and the
    same server-stated wait, so an application reading it sees the rate limit rather than an
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
            with pytest.raises(RetryUnavailableError) as raised:
                async for _item in handle:
                    pass
        cause = raised.value.__cause__
        assert isinstance(cause, TransientError)
        assert cause.is_rate_limit
        assert cause.retry_after_seconds == _MID_STREAM_RETRY_AFTER_SECONDS
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


def test_an_adapter_stated_mid_stream_transient_error_becomes_the_cause_unwrapped() -> None:
    """An adapter that states the verdict itself has that very object as __cause__.

    Wrapping it again would replace the adapter's own message with the handle's.
    """

    async def scenario() -> None:
        """Let the iteration fail after one item and read the error the caller catches."""
        stream = _RaisesItsOwnTransientErrorStream()
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RetryUnavailableError) as raised:
                async for _item in handle:
                    pass
        assert raised.value.__cause__ is stream.error
        assert stream.error.is_rate_limit
        assert stream.error.retry_after_seconds == _MID_STREAM_RETRY_AFTER_SECONDS

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


def test_a_close_raising_a_base_exception_still_sets_the_abandoned() -> None:
    """The record survives a teardown that raises past every Exception handler.

    __aexit__ closes before it sets abandoned, so that the record reports a returned slot and a
    closed connection. Without the set in a finally, the one exception the close does not swallow
    would take the cancelled stream's only account with it.
    """

    async def scenario() -> None:
        """Cancel the block, then let the close raise on the way out."""
        bound_llm = LLM(
            _FakeAdapter(stream=_CloseRaisesBaseExceptionStream(), classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Hang inside the block; the wait_for cancels this."""
            async with handle:
                await asyncio.sleep(60)

        with pytest.raises(KeyboardInterrupt):
            await asyncio.wait_for(consume(), timeout=0.05)
        abandoned = handle.abandoned
        assert abandoned is not None
        assert abandoned.model == "fake-model"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


@pytest.mark.parametrize(
    ("usage_reported", "expected_usage"),
    [(_USAGE_STREAM, _USAGE_STREAM), (None, ZERO_USAGE)],
    ids=["running_report_present", "no_running_report"],
)
def test_a_stream_that_broke_after_items_records_what_the_provider_reported(
    usage_reported: Usage | None, expected_usage: Usage
) -> None:
    """The dropped attempt's record carries the running counters, which is zero only if none came.

    The stream was paid for the items it delivered, and its terminal event never arrived, so the
    running report is the only account of that spend. RetryUnavailableError concludes the call, so
    this record is where the amount has to be.
    """

    async def scenario() -> None:
        """Let the iteration fail after one item, then read the error's usage."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = usage_reported
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RetryUnavailableError) as raised:
                async for _item in handle:
                    pass
        (record,) = raised.value.attempt_records
        assert record.usage == expected_usage
        assert raised.value.usage == expected_usage

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_a_mid_stream_failure_sets_no_abandoned() -> None:
    """A failure after the first item concludes the call, so a cancellation after one sets nothing.

    RetryUnavailableError hands the caller this call's CallRecord, so an AbandonedCallError would
    report the same call twice and mislabel a concluded call as an in-flight abandonment.
    """

    async def scenario() -> None:
        """Absorb the mid-stream failure inside the block, then hang into the caller's deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream(), classify_result="transient"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let the iteration fail after one item, then sleep; the wait_for cancels this."""
            async with handle:
                with pytest.raises(RetryUnavailableError) as raised:
                    async for _item in handle:
                        pass
                (record,) = raised.value.attempt_records
                assert isinstance(record.error, TransientError)
                assert "after items were yielded" in str(record.error)
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_a_drain_failure_sets_no_abandoned() -> None:
    """A GenerationError raised while draining carries the call, so a cancellation sets nothing.

    The error already handed the caller a CallRecord, so an AbandonedCallError here would report the
    same call twice.
    """

    async def scenario() -> None:
        """Absorb an unplaceable item failure inside the block, then hang into the deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_UnnamedItemErrorStream(), classify_result="unknown_exception"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() hit the unplaceable failure, then sleep; the wait_for cancels this."""
            async with handle:
                with pytest.raises(UnknownExceptionError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

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


@pytest.mark.parametrize(
    ("stream", "expected_error"),
    [
        (_MaxCompletionTokensExceededStream(), MaxCompletionTokensExceededError),
        (_EmptyTurnStream(), EmptyTurnError),
        (_ContextWindowExceededStream(), ContextWindowExceededError),
    ],
    ids=["max_completion_tokens_exceeded", "empty_turn", "context_window_exceeded"],
)
def test_stream_final_reports_each_no_output_outcome_as_its_own_error(
    stream: _FakeStream, expected_error: type[GenerationError]
) -> None:
    """Each outcome the stream's final() reports fails the call with its own error, without retrying.

    The stream loop builds these errors separately from the generate loop, so an outcome mapped
    correctly on the generate path can still be mapped wrongly here. None is retried: the token cap,
    the empty turn, and the overflowed context window all repeat on a resend.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports the outcome, then read the raised error."""
        adapter = _FakeAdapter(stream=stream)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(expected_error) as caught:
                await handle.final()
        failure = caught.value
        assert failure.attempts == 1
        assert failure.usage.cost_in_usd == 0.25
        (record,) = failure.attempt_records
        assert record.error is None

    asyncio.run(scenario())


def test_stream_final_provider_failed_transiently_fails_the_item_with_retry_unavailable() -> None:
    """The outcome fails the item with RetryUnavailableError carrying the call, without reopening.

    The outcome is read from the assembled response, so that stream is over and none is opened in
    its place. That 200's billing and the fragment it carried are on its attempt record, beside the
    TransientError the failure was classified as, which is also __cause__.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports the failure, then read the raised error."""
        adapter = _FakeAdapter(stream=_ProviderFailedTransientlyStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RetryUnavailableError) as retry_unavailable:
                await handle.final()
        failure = retry_unavailable.value
        assert failure.attempts == 1
        assert _PROVIDER_FAILURE_REASON in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = failure.attempt_records
        assert str(record.error) == _PROVIDER_FAILURE_REASON
        assert record.assistant_message == _REJECTED_TURN
        assert record.raw is not None
        assert failure.__cause__ is record.error
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_stream_that_yielded_nothing_still_fails_transiently_without_reopening() -> None:
    """A stream that emitted no item and then reports the failure spends one attempt, not the budget.

    The retry budget is untouched and the stream is not reopened: the outcome is read from the
    assembled response, which arrives when the stream is over, whether or not it yielded anything.
    """

    async def scenario() -> None:
        """Drain a stream that yields nothing, then read what final() raises."""
        adapter = _FakeAdapter(stream=_YieldsNothingThenFailsTransientlyStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_attempts=3)).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            assert [item async for item in handle] == []
            with pytest.raises(RetryUnavailableError) as retry_unavailable:
                await handle.final()
        assert retry_unavailable.value.attempts == 1
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


def test_a_stream_cancelled_after_absorbing_a_provider_failure_sets_no_abandoned() -> None:
    """A cancellation after final() raised its RetryUnavailableError sets nothing.

    That error carried the call, whose record holds the 200's billing, so an AbandonedCallError here
    would double-count that spend and mislabel a concluded call as an in-flight abandonment.
    """

    async def scenario() -> None:
        """Absorb the failure from final() inside the block, then hang into the caller's deadline."""
        adapter = _FakeAdapter(stream=_ProviderFailedTransientlyStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() report the failure, then sleep; the wait_for below cancels in the block."""
            async with handle:
                with pytest.raises(RetryUnavailableError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_that_drops_mid_turn_records_the_id_its_open_response_carried() -> None:
    """A dropped connection raises an error naming no request, and the open stream still has the header.

    That failed attempt is the one someone takes to provider support, so reading only the error
    would leave it with nothing.
    """

    async def scenario() -> None:
        """Drain a stream whose first items() call fails before yielding."""
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            _ = [item async for item in handle]
            response = await handle.final()
        failed, _succeeded = response.attempt_records
        assert failed.response_id is None
        assert failed.request_id == "req-fake-stream"

    asyncio.run(scenario())


def test_the_request_id_an_error_names_outranks_the_streams_own() -> None:
    """The error describes the failure, so its id wins where both channels have one."""

    async def scenario() -> None:
        """Drain a stream whose first items() call fails with an error naming its request."""
        adapter = _FakeAdapter(
            stream=_FailsWithARequestIdBeforeFirstItemStream(), classify_result="transient"
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            _ = [item async for item in handle]
            response = await handle.final()
        failed, _succeeded = response.attempt_records
        assert failed.request_id == "req-from-items-error"

    asyncio.run(scenario())


def test_a_request_id_on_the_assembled_response_outranks_the_streams_own() -> None:
    """The stream fills the request id in where the response has none, and never replaces one.

    The shipped adapters never hit this: their SDKs leave the assembled response without the header.
    An adapter whose stream cannot reach the response headers is what the rule protects, because
    overwriting would trade the id it does have for a null.
    """

    async def scenario() -> None:
        """Drain a stream whose assembled response names its own request."""
        stream = _FakeStream()
        stream.raw = _FakeRawResponse(id="fake-final", request_id="req-from-assembled")
        bound_llm = LLM(_FakeAdapter(stream=stream), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.request_id == "req-from-assembled"

    asyncio.run(scenario())


def test_stream_retry_populates_attempt_records() -> None:
    """A pre-first-item connection failure lands as an errored record before the success record.

    The stream path stages the assembled response's identity too, so the succeeding record carries
    what a sent one does, and the failed open carries the request id its error named.
    The succeeding record's request id is the stream's own: the assembled response carries no
    request-id header, which is what leaves identity_from_raw nothing to report.
    """

    async def scenario() -> None:
        """Open a stream whose first open_stream call fails, then drain it."""
        adapter = _FakeAdapter(
            open_failures=[_RequestIdError("connection reset", "req-from-open-failure")],
            classify_result="transient",
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        assert response.output == "ab"
        assert response.attempts == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "connection reset"
        assert (failed.model_served, failed.response_id) == (None, None)
        assert failed.request_id == "req-from-open-failure"
        assert succeeded.error is None
        assert succeeded.model_served == "fake-model-served"
        assert succeeded.response_id == "fake-final"
        assert succeeded.request_id == "req-fake-stream"
        assert (
            failed.started_at_monotonic_seconds
            <= failed.ended_at_monotonic_seconds
            <= succeeded.started_at_monotonic_seconds
            <= succeeded.ended_at_monotonic_seconds
        )

    asyncio.run(scenario())


def test_a_stream_stamps_its_first_item_and_not_a_later_one() -> None:
    """The stamp lands after the attempt's start and before the second item arrives.

    The stream spreads its items over a gap far wider than the work between them, so a stamp taken
    on any later item would land past the bound asserted here.
    """

    async def scenario() -> None:
        """Drain a stream that waits between its items and read the record it froze."""
        gap_seconds = _SlowAfterFirstItemStream.gap_seconds
        adapter = _FakeAdapter(stream=_SlowAfterFirstItemStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            _ = [item async for item in handle]
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.first_item_at_monotonic_seconds is not None
        seconds_to_first_item = (
            record.first_item_at_monotonic_seconds - record.started_at_monotonic_seconds
        )
        assert 0.0 <= seconds_to_first_item < gap_seconds
        assert record.elapsed_seconds > gap_seconds

    asyncio.run(scenario())


def test_a_non_stream_attempt_leaves_its_first_item_stamp_unset() -> None:
    """generate_one yields no items, so the column it feeds stays null for every non-stream call."""

    async def scenario() -> None:
        """Run one generate_one and read the record it froze."""
        bound_llm = LLM(_FakeAdapter(), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = response.attempt_records
        assert record.first_item_at_monotonic_seconds is None

    asyncio.run(scenario())


def test_a_reopened_stream_stamps_its_own_first_item() -> None:
    """An attempt that reached no item stays null; the reopen that did stamps within its own bracket."""

    async def scenario() -> None:
        """Drain a stream whose first items() call fails before yielding."""
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream())
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            _ = [item async for item in handle]
            response = await handle.final()
        failed, succeeded = response.attempt_records
        assert failed.first_item_at_monotonic_seconds is None
        assert succeeded.first_item_at_monotonic_seconds is not None
        assert succeeded.started_at_monotonic_seconds <= succeeded.first_item_at_monotonic_seconds

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


def test_a_stream_whose_build_request_refuses_fails_the_item_with_nothing_opened() -> None:
    """A conversation build_request refuses fails the item before any stream is opened.

    The handle builds the InvalidRequestError, so a caller reading model or attempt_records off it
    succeeds; it carries no request, there being none to carry.
    """

    async def scenario() -> None:
        """Enter a handle whose build_request refuses."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="nope")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 0
        assert rejected.value.request is None
        assert rejected.value.reason == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.attempt_records == ()

    asyncio.run(scenario())


def test_stream_open_classified_unknown_exception_raises_the_items_failure() -> None:
    """An open failure classified unknown_exception raises UnknownExceptionError from the entry, unretried."""

    async def scenario() -> None:
        """Enter a handle whose open raises a classify-unknown_exception exception."""
        adapter = _FakeAdapter(
            open_failures=[ValueError("boom")], classify_result="unknown_exception"
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnknownExceptionError) as unplaceable:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(unplaceable.value.error, ValueError)
        assert unplaceable.value.error_text == "langchaint could not place this exception: boom"
        assert unplaceable.value.attempt_records == ()

    asyncio.run(scenario())


def test_stream_open_classified_declared_final_raises_the_items_failure() -> None:
    """An open failure the provider declared final raises ProviderDeclaredFinalError, unretried.

    The open reached the provider, which answered, so that attempt has a record.
    """

    async def scenario() -> None:
        """Enter a handle whose open raises a classify-declared_final exception."""
        adapter = _FakeAdapter(
            open_failures=[ValueError("boom")], classify_result="declared_final"
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ProviderDeclaredFinalError) as declared_final:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(declared_final.value.error, ValueError)
        assert declared_final.value.error_text == "the provider declared this error final: boom"
        (record,) = declared_final.value.attempt_records
        assert record.error is None
        assert record.usage == ZERO_USAGE

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


@pytest.mark.parametrize(
    ("classify_result", "expected_error"),
    [
        ("invalid_request", InvalidRequestError),
        ("declared_final", ProviderDeclaredFinalError),
        ("unknown_exception", UnknownExceptionError),
    ],
    ids=["invalid_request", "declared_final", "unknown_exception"],
)
def test_a_terminal_mid_stream_error_records_what_the_stream_reported(
    classify_result: ErrorClassification, expected_error: type[GenerationError]
) -> None:
    """An open stream accounts for its attempt whatever verdict the adapter reaches on the error.

    unknown_exception is the row that has to work with no verdict at all: anthropic reports a
    mid-stream failure as an error event on the 200 that carried the turn, so the adapter reads a
    status the retry policy cannot place. The stream is open in every row, so what the provider
    reported for that attempt is readable and belongs on its record.
    """

    async def scenario() -> None:
        """Let the iteration hit the failure after one item, then read the record."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result=classify_result),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(expected_error) as caught:
                async for _item in handle:
                    pass
        (record,) = caught.value.attempt_records
        assert record.usage == _USAGE_STREAM
        assert caught.value.usage == _USAGE_STREAM

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_an_unplaceable_mid_stream_error_records_the_attempt_with_nothing_reported() -> None:
    """A stream open when the failure hit gets a record whether or not it reported any billing.

    The record says the attempt reached the provider, and billing None on it says the provider
    stated nothing, which is a different fact from the attempt never having happened. Reading the
    stream's billing instead would collapse the two and drop this record.
    """

    async def scenario() -> None:
        """Let the iteration hit the failure after one item on a stream reporting no counters."""
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream(), classify_result="unknown_exception"),
            rate_limiter=_fast_rate_limiter(),
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(UnknownExceptionError) as caught:
                async for _item in handle:
                    pass
        (record,) = caught.value.attempt_records
        assert record.billing is None
        assert record.usage == ZERO_USAGE

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_that_broke_before_its_first_item_records_what_the_provider_reported() -> None:
    """A provider reporting counters before its first item has already billed for that attempt.

    anthropic prices its running snapshot from the turn's first event, which carries no item, so
    an attempt that reopens after failing there is not free. The whole call's usage folds both
    records, so the reopened success does not erase what the failed attempt cost.
    """

    async def scenario() -> None:
        """Drain a stream whose first items() call fails before yielding, then read both records."""
        stream = _FailsBeforeFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        adapter = _FakeAdapter(stream=stream)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                pass
            response = await handle.final()
        failed, succeeded = response.attempt_records
        assert failed.usage == _USAGE_STREAM
        assert succeeded.usage == _USAGE_STREAM
        assert response.usage == Usage.sum_of((_USAGE_STREAM, _USAGE_STREAM))

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
        (_ProviderFailedTransientlyStream(), RetryUnavailableError),
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


@pytest.mark.parametrize("settled_attempts", [1, 0])
def test_a_deadline_expiring_mid_request_counts_the_request_it_cut_off(
    settled_attempts: int,
) -> None:
    """timeout_seconds expiring in flight ends the call as TimedOutError carrying what it spent.

    Nothing closes the cut-off attempt into a record: the deadline's cancellation is a
    BaseException, so the retry loop's except clauses never see it, and freeze() closes only an
    attempt whose response had arrived. AbandonedCallError.attempts adds that request back, so it
    counts one more than there are records, and attempts == 0 is left meaning nothing went out.
    """

    async def scenario() -> None:
        """Fail settled_attempts requests transiently, then hang the next past a deadline."""
        adapter = _FakeAdapter(
            failures=[TransientError("settled attempt")] * settled_attempts,
            # 1-based, so the request that hangs is the one after the settled ones.
            hang_from_send=settled_attempts + 1,
        )
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(TimedOutError) as raised:
            await bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=0.05)
        timed_out = raised.value
        assert timed_out.attempts == settled_attempts + 1
        assert len(timed_out.attempt_records) == settled_attempts
        assert timed_out.in_flight_attempt_started_at_monotonic_seconds is not None
        # No response arrived on any attempt, so nothing the provider reported is in flight.
        assert timed_out.billing_in_flight is None
        assert timed_out.usage == ZERO_USAGE
        assert str(timed_out) == "the call timed out before it produced a result"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_deadline_expiring_before_a_slot_reports_no_attempts() -> None:
    """A call that never got admitted reports attempts == 0, so a caller can tell it never sent.

    max_in_flight is 1 and the holder never returns its slot, so the second call spends its whole
    deadline inside the RateLimiter and sends nothing.
    """

    async def scenario() -> None:
        """Hold the only slot, then run a second call under a deadline it cannot outlast."""
        adapter = _FakeAdapter(hang_from_send=1)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter(max_in_flight=1)).bind(
            automatic_prompt_caching=True
        )
        holder = asyncio.create_task(bound_llm.generate_one([UserMessage(content="held")]))
        await asyncio.sleep(0.02)
        with pytest.raises(TimedOutError) as raised:
            await bound_llm.generate_one([UserMessage(content="queued")], timeout_seconds=0.05)
        timed_out = raised.value
        assert timed_out.attempts == 0
        assert timed_out.in_flight_attempt_started_at_monotonic_seconds is None
        assert timed_out.usage == ZERO_USAGE
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_an_outer_cancellation_inside_the_deadline_stays_a_cancellation() -> None:
    """The scope langchaint owns claims only its own expiry, never a caller's cancellation.

    The deadline here is far from expiring when the outer cancellation lands.
    """

    async def scenario() -> None:
        """Cancel a call from outside while its own generous deadline is still running."""
        bound_llm = LLM(_FakeAdapter(hang_from_send=1), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        call = asyncio.create_task(
            bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=30.0)
        )
        await asyncio.sleep(0.02)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert call.cancelled()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_batch_times_out_one_item_and_returns_its_siblings() -> None:
    """Each item gets its own deadline, so one item running out does not cut a sibling."""

    async def scenario() -> None:
        """Hang the second item's send while the first answers."""
        adapter = _FakeAdapter(hang_from_send=2, echo=True)
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        results = await bound_llm.generate_many(
            [[UserMessage(content="fast")], [UserMessage(content="slow")]],
            timeout_seconds=0.05,
        )
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "fast"
        assert isinstance(second, TimedOutError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_deadline_raises_and_leaves_abandoned_unset() -> None:
    """One cut-off call gets one account: the raise, not the handle field as well.

    Setting both would put the same call in the archive twice.
    """

    async def scenario() -> None:
        """Enter a stream whose open never returns, under a deadline."""
        bound_llm = LLM(_FakeAdapter(hang_from_open=1), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        with pytest.raises(TimedOutError):
            async with handle:
                pass
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_deadline_expiring_mid_items_raises() -> None:
    """The deadline covers the whole block, not the open alone."""

    async def scenario() -> None:
        """Enter a stream that stalls after its first item, under a deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_HangsAfterFirstItemStream()), rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        seen: list[StreamItem] = []

        async def drain() -> None:
            """Consume the stream, recording each item as it arrives.

            The deadline expires mid-iteration, so each item has to reach seen on its own;
            a comprehension would build a list the expiry discards.
            """
            async with handle:
                async for item in handle:
                    seen.append(item)  # noqa: PERF401

        with pytest.raises(TimedOutError):
            await drain()
        assert seen, "the items delivered before the deadline stay delivered"
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_deadline_stops_at_the_calls_conclusion() -> None:
    """The deadline bounds the call, so a caller's work after final() is its own time.

    A timer still armed past the conclusion would raise a TimedOutError for a call that produced a
    Response, putting the same call in the archive twice and charging it the caller's work.
    """

    async def scenario() -> None:
        """Take the Response, then outlive the deadline inside the block."""
        bound_llm = LLM(_FakeAdapter(), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        async with handle:
            response = await handle.final()
            await asyncio.sleep(0.1)
        assert response.output == "ab"
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_deadline_stops_at_a_conclusion_an_item_pull_raised() -> None:
    """A call concluded by a failing item pull closes the deadline, not only one final() concluded.

    An item pull that ends the call stores its own conclusion and never reaches final(), so a
    deadline left armed there would raise a second account of a call that already has one.
    """

    async def scenario() -> None:
        """Take the UnknownExceptionError mid-iteration, then outlive the deadline in the block."""
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream()), rate_limiter=_fast_rate_limiter()
        ).bind(automatic_prompt_caching=True)
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        async with handle:
            with pytest.raises(UnknownExceptionError):
                async for _ in handle:
                    pass
            await asyncio.sleep(0.1)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_failed_stream_entry_leaves_no_armed_deadline() -> None:
    """__aenter__ raising closes the deadline it opened, so no timer outlives the failed entry.

    __aexit__ does not run when __aenter__ raises, so a scope left open would keep a timer that
    cancels this task at an arbitrary later point, far outside the block.
    """

    async def scenario() -> None:
        """Fail an entry under a short deadline, then outlive that deadline uncancelled."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="no")])
        bound_llm = LLM(adapter, rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.02)
        with pytest.raises(InvalidRequestError):
            async with handle:
                pass
        # Outlast the deadline the failed entry opened; a leaked timer cancels this sleep.
        await asyncio.sleep(0.1)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_no_deadline_lets_a_callers_own_scope_expire() -> None:
    """The default claims nothing, so a caller's own scope converts its cancellation as it would."""

    async def scenario() -> None:
        """Run a call with no deadline and let an outer scope cut it."""
        bound_llm = LLM(_FakeAdapter(hang_from_send=1), rate_limiter=_fast_rate_limiter()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await bound_llm.generate_one([UserMessage(content="hi")])

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
