"""BoundLLM and StreamHandle driven by fake adapters.

A fake BoundAdapter scripts each attempt to fail or to produce a scripted response,
and a fake AdapterStream emits a fixed item sequence.
Together they pin the retry loop, rebind rebuild, batch ordering, and the stream contract without any network access.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
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
    CallResult,
    ContextWindowExceededError,
    DoNotRetry,
    EmptyTurnError,
    EscapedExceptionError,
    GenerationError,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    Message,
    ParserContractError,
    PauseAllDoNotRetry,
    ProviderDeclaredFinalError,
    ProviderFailedTerminallyError,
    RefusalError,
    Response,
    RetriesExhaustedError,
    RetryUnavailableError,
    SchemaViolationError,
    SharedBackoff,
    StopReason,
    StreamItem,
    StreamProtocolError,
    TextPart,
    TimedOutError,
    ToolCall,
    ToolCallTurn,
    ToolManager,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
    Usage,
    UserMessage,
    Verdict,
)
from langchaint import shared_backoff as shared_backoff_module
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
    verdict_from_transient_error,
)
from langchaint.call import ResponseIdentity
from langchaint.llm import UNCHANGED, Unchanged
from langchaint.shared_backoff import _NEVER
from langchaint.streaming import StreamHandle
from tests.helpers import random_returns_zero, stated_billing

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
    provider_executed_tool_cost_in_usd=0.0,
)
_USAGE_BILLED = _USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.25})
"""The billing a 200 that produced no output (a refusal or truncation) carries."""
_USAGE_STREAM = _USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.001})
"""The stream final()'s assembled usage, distinct so a stream cost is visible."""


def _parse_fake(failure: Exception) -> Verdict:
    """Map a TransientError with the shared rule, as a provider parse maps its own SDK errors.

    Every other exception the fakes raise stands for a transport failure outside failure_types,
    so this parse never sees one; DoNotRetry is the fallthrough a real parse would end on.
    """
    if isinstance(failure, TransientError):
        return verdict_from_transient_error(failure)
    return DoNotRetry()


def _parse_pause_all_do_not_retry(_failure: Exception) -> Verdict:
    """Verdict every failure the way a 429 the provider marked x-should-retry: false parses.

    retry_after is None, the provider having named no wait, so the pause takes a drawn one.
    """
    return PauseAllDoNotRetry(retry_after=None)


def _parse_raises(_failure: Exception) -> Verdict:
    """Violate the parse contract on every failure, standing in for a buggy provider parse."""
    raise RuntimeError("parse defect")


def _fast_shared_backoff(
    *,
    max_concurrent_requests: int | None = 8,
    parse: Callable[[Exception], Verdict] = _parse_fake,
    longest_wait_seconds: float = 0.002,
    max_request_starts_per_second: float = 10_000.0,
) -> SharedBackoff:
    """Build a fresh near-zero-wait `SharedBackoff`.

    One instance serves one event loop.
    """
    return SharedBackoff(
        parse=parse,
        failure_types=(TransientError,),
        max_concurrent_requests=max_concurrent_requests,
        minimum_wait_ceiling_seconds=0.001,
        longest_wait_seconds=longest_wait_seconds,
        max_request_starts_per_second=max_request_starts_per_second,
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
"""The turn a 200 that produced no output still carried, which every such variant takes."""


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
    variant carrying the turn the model wrote to refuse.
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
    """A stream whose assembled response reports the request overflowed the context window."""

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
    """A stream that yields one item and then fails."""

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
    """A stream whose items() fails transiently before yielding."""

    def _item_error(self) -> Exception:
        """Return the error items() raises."""
        return TransientError("dropped before the first item")

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Fail before the first yield.

        Yields:
            Nothing.

        Raises:
            Exception: _item_error's result.
        """
        raise self._item_error()
        yield "unreachable"


class _FailsWithARequestIdBeforeFirstItemStream(_FailsBeforeFirstItemStream):
    """A stream whose items() failure names its request."""

    @override
    def _item_error(self) -> Exception:
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


type _ScriptedAttempt = Exception | _ScriptedResponse
"""One scripted attempt: an exception open_stream raises, or the response the attempt assembles.

The two exist because the adapter contract splits that way: an attempt with no response to read is
an exception for Adapter.classify, and a response is what the attempt's stream assembles and
interpret then reads.
A Sequence[Message] the fake will not put on the wire is its invalid_request, which build_request reports
before any stream is opened.
"""


class _ScriptedAttemptStream(_FakeStream):
    """One attempt's stream over a scripted response, standing in for a real per-request stream.

    raw is fresh per attempt and carries no request_id, as a response an SDK assembled from stream
    events does; request_id() derives the header from the raw's id, so a record's request id on a
    concluded attempt comes only from the fallback patch off this stream.
    A success yields its content as one chunk and then the fixed tool call, so the drained text and
    final()'s output agree; a scripted no-output response yields nothing.
    """

    def __init__(self, *, raw: _FakeRawResponse, content: str | None) -> None:
        """Hold the fresh raw final() returns and the content items() yields, None yielding none."""
        super().__init__()
        self.raw = raw
        self._content = content

    @override
    def request_id(self) -> str | None:
        """Name this attempt's request, derived from the raw so each attempt's header is its own."""
        return f"req-{self.raw.id}"

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        if self._content is not None:
            yield self._content
            yield _FAKE_TOOL_CALL


@dataclass(frozen=True, kw_only=True)
class _FakeRequest(RequestParams):
    """What the fake would put on the wire, which is the messages and nothing else."""

    messages: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        """Render the messages as a JSON array of each message's dump."""
        return json.dumps([message.model_dump(mode="json") for message in self.messages])


def _as_fake_request(request: RequestParams) -> _FakeRequest:
    """Narrow a request to the fake one.

    Raises:
        TypeError: request is not a _FakeRequest, which the real adapters raise for the same reason.
    """
    if not isinstance(request, _FakeRequest):
        raise TypeError(f"expected a _FakeRequest, got {type(request).__name__}")
    return request


class _FakeBoundAdapter(BoundAdapter[str]):
    """A bound adapter whose open_stream follows a scripted attempt sequence."""

    def __init__(
        self,
        *,
        failures: Sequence[_ScriptedAttempt] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        open_seconds: float = 0.0,
        hang_from_open: int | None = None,
    ) -> None:
        """Store the attempt script, echo mode, and the stream open_stream returns.

        failures scripts each attempt in order: an Exception raises from open_stream, and a
        _ScriptedResponse is what that attempt's stream assembles.
        invalid_requests scripts build_request, one entry per call, and an entry is reported
        instead of a request, so open_stream is not reached for that call.
        stream, when given, is what every unscripted open returns, so a test controls that
        stream's behavior; left None, each unscripted open returns a fresh success stream.
        open_seconds > 0 makes each open suspend that long,
        so a batch overlaps and peak_in_flight records the concurrency it reached.
        hang_from_open is the 1-based open_stream call from which every open suspends forever,
        so a cancellation or a deadline lands while that attempt is in flight.
        final_raws collects the response objects the scripted streams assemble, in order, so a
        test can assert the caller got one of them and not a copy.
        """
        self._failures = list(failures)
        self._invalid_requests = list(invalid_requests)
        self._echo = echo
        self._explicit_stream = stream
        self._open_seconds = open_seconds
        self._hang_from_open = hang_from_open
        self._scripted_by_raw_id: dict[str, _ScriptedResponse] = {}
        self.final_raws: list[_FakeRawResponse] = []
        self.build_count = 0
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
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Report the scripted refusal, else carry messages into the request."""
        if self._invalid_requests:
            return self._invalid_requests.pop(0)
        self.build_count += 1
        return _FakeRequest(messages=tuple(messages))

    def _attempt_stream(
        self, scripted: _ScriptedResponse, *, content: str | None
    ) -> _ScriptedAttemptStream:
        """Register the scripted response under a fresh raw and wrap it in this attempt's stream."""
        raw = _FakeRawResponse(id=f"fake-response-{self.open_count}")
        self._scripted_by_raw_id[raw.id] = scripted
        self.final_raws.append(raw)
        return _ScriptedAttemptStream(raw=raw, content=content)

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Count the attempt, suspend, then raise or return the next scripted attempt's stream.

        Raises:
            TypeError: request is not a _FakeRequest.
            Exception: the next scripted failure.
        """
        messages = _as_fake_request(request).messages
        self.open_count += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._hang_from_open is not None and self.open_count >= self._hang_from_open:
                await asyncio.Event().wait()
            if self._open_seconds:
                await asyncio.sleep(self._open_seconds)
            if self._failures:
                scripted = self._failures.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                return self._attempt_stream(scripted, content=None)
            if self._explicit_stream is not None:
                stream = self._explicit_stream
                self._scripted_by_raw_id[stream.raw.id] = stream.scripted_response()
                return stream
            first = messages[0]
            content = (
                first.content
                if self._echo and isinstance(first, UserMessage) and isinstance(first.content, str)
                else "ok"
            )
            return self._attempt_stream(
                _ScriptedResponse(outcome=_success_result(content), usage=_USAGE), content=content
            )
        finally:
            self.in_flight -= 1


class _FakeStructuredBoundAdapter[ModelT: BaseModel](BoundAdapter[ModelT]):
    """A structured bound adapter for response_format rebind tests; it never generates.

    Those tests check binding identity and the switched content type, not structured output,
    so open_stream stays unreachable.
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
    def build_request(self, messages: Sequence[Message]) -> RequestParams:
        """Unreachable: response_format rebind tests do not generate."""
        raise NotImplementedError

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Unreachable: response_format rebind tests do not generate."""
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
        failures: Sequence[_ScriptedAttempt] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        classify_result: ErrorClassification = "unknown_exception",
        open_seconds: float = 0.0,
        hang_from_open: int | None = None,
    ) -> None:
        """Store how each freshly bound adapter behaves and the classify verdict."""
        # This adapter reaches no SDK, so it passes client=None, which matches no entry in the
        # base's empty provider_name_by_client_class, leaving the stated "fake" to stand.
        super().__init__(client=None, model="fake-model", provider_name="fake")
        self._failures = failures
        self._invalid_requests = invalid_requests
        self._echo = echo
        self._stream = stream
        self._classify_result = classify_result
        self._open_seconds = open_seconds
        self._hang_from_open = hang_from_open
        self.bound_adapters: list[_FakeBoundAdapter] = []
        self.structured_bind_count = 0

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        bound = self._bound_adapter_class(
            failures=self._failures,
            invalid_requests=self._invalid_requests,
            echo=self._echo,
            stream=self._stream,
            open_seconds=self._open_seconds,
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

    failure_types: ClassVar[tuple[type[Exception], ...]] = (TransientError,)

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Delegate to the module-level rule _fast_shared_backoff also parses with."""
        return _parse_fake(failure)

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
    """A bound adapter whose interpret raises over a response its stream already assembled."""

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


def test_llm_rejects_invalid_max_attempts() -> None:
    """Reject `max_attempts` values below one and boolean values."""
    for max_attempts in (True, False, 0, -1):
        with pytest.raises(ValueError, match="max_attempts"):
            _ = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
                automatic_prompt_caching=True,
                max_attempts=max_attempts,
            )


def test_rebind_rejects_invalid_max_attempts() -> None:
    """Reject replacement `max_attempts` values below one and boolean values."""
    bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
        automatic_prompt_caching=True
    )
    for max_attempts in (True, False, 0, -1):
        with pytest.raises(ValueError, match="max_attempts"):
            _ = bound_llm.rebind(max_attempts=max_attempts)


def test_a_raise_from_interpret_leaves_the_response_and_its_billing_on_the_record() -> None:
    """The attempt keeps the response it received and what that response billed, with no turn.

    The 200 arrived and was paid for before interpret read it, so an exception from that read must
    not take the attempt off the call: the record is the only account of what the item spent.
    """

    async def scenario() -> None:
        """Drive one generate_one whose interpret raises over the response its stream assembled."""
        bound_llm = LLM(_InterpretRaisesAdapter(), shared_backoff=_fast_shared_backoff()).bind(
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
            _InterpretRaisesAdapter(stream=stream), shared_backoff=_fast_shared_backoff()
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
    The Response carries the object the succeeding attempt assembled (identity, not equality): an equal
    copy would silently introduce the per-request deep copy the no-rewrap rule bans.
    """

    async def scenario() -> None:
        """Drive one generate_one through a single transient failure."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            system_prompt="s", automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 2
        assert response.model == "fake-model"
        assert response.provider_name == "fake"
        assert adapter.bound_adapters[0].open_count == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "boom"
        assert failed.assistant_message is None
        assert (failed.model_served, failed.response_id, failed.request_id) == (None, None, None)
        assert succeeded.error is None
        assert succeeded.assistant_message == _success_result("ok").assistant_message
        assert succeeded.model_served == "fake-model-served"
        assert succeeded.response_id == _as_fake_raw(response.raw).id
        assert succeeded.request_id == f"req-{_as_fake_raw(response.raw).id}"
        (succeeding_raw,) = adapter.bound_adapters[0].final_raws
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
    """Two transient failures produce one build and three opened requests.

    build_request() runs once per call.
    Every retry sends the same RequestParams.
    """

    async def streamed_counts(adapter: _FakeAdapter) -> tuple[int, int]:
        """Drain one stream over adapter and return its bound adapter's build and open counts."""
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            await handle.final()
        bound = adapter.bound_adapters[0]
        return bound.build_count, bound.open_count

    async def scenario() -> None:
        """Drive generate_one and stream_one through two transient failures."""
        generate_adapter = _FakeAdapter(
            failures=[TransientError("boom"), TransientError("boom again")]
        )
        generate_llm = LLM(generate_adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        await generate_llm.generate_one([UserMessage(content="hi")])
        generate_bound = generate_adapter.bound_adapters[0]
        assert (generate_bound.build_count, generate_bound.open_count) == (1, 3)

        retried = _FakeAdapter(failures=[TransientError("boom"), TransientError("boom again")])
        assert await streamed_counts(retried) == (1, 3)

    asyncio.run(scenario())


def test_a_failed_attempt_records_the_request_id_off_its_error() -> None:
    """An attempt that received no response still carries the request id its error named.

    The three attempts name three different requests: the second failure names none, so an id on its
    record would be the first attempt's outliving the attempt that read it, and the third attempt's own
    id comes from the response rather than from either error.
    """

    async def scenario() -> None:
        """Drive one generate_one through two transient failures, the first naming its request."""
        adapter = _FakeAdapter(
            failures=[_RequestIdError("boom", "req-from-error"), TransientError("boom again")],
            classify_result="transient",
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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

    Both retry loops read it: the generate loop in its own except clause, and the stream loop ahead
    of the check that sends a transient error straight back for a retry.
    """

    async def scenario() -> None:
        """Fail one generate attempt and one stream open with a transient error naming its request."""
        generate_adapter = _FakeAdapter(
            failures=[_TransientRequestIdError("boom", "req-from-generate-transient")]
        )
        generate_llm = LLM(generate_adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        generated = await generate_llm.generate_one([UserMessage(content="hi")])
        assert generated.attempt_records[0].request_id == "req-from-generate-transient"

        stream_adapter = _FakeAdapter(
            failures=[_TransientRequestIdError("boom", "req-from-open-transient")]
        )
        bound_llm = LLM(stream_adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True,
            max_attempts=2,
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

    The random draw is pinned so each drawn wait is its ceiling, making the backoff gap deterministic.
    """
    monkeypatch.setattr(shared_backoff_module.random, "random", random_returns_zero)

    async def scenario() -> None:
        """Recover from one failure under a visible 0.05s backoff."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        shared_backoff = SharedBackoff(
            parse=_parse_fake,
            failure_types=(TransientError,),
            max_concurrent_requests=8,
            minimum_wait_ceiling_seconds=0.05,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True,
            max_attempts=2,
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        failed, succeeded = response.attempt_records
        assert failed.elapsed_seconds < 0.05
        backoff_gap = succeeded.started_at_monotonic_seconds - failed.ended_at_monotonic_seconds
        assert backoff_gap >= 0.05
        assert response.elapsed_seconds >= 0.05

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_build_request_refusing_messages_fails_the_item_with_nothing_sent() -> None:
    """An InvalidRequest from build_request fails the item before any request goes out.

    classify returns "transient" here and is never reached: a returned outcome is not an exception,
    so no classify verdict can turn this into an attempt.
    The InvalidRequestError the loop builds carries the reason and the per-item failure fields, and no
    request, there being none to carry.
    """

    async def scenario() -> None:
        """Drive one generate_one whose build_request refuses under a transient classify verdict."""
        adapter = _FakeAdapter(
            invalid_requests=[InvalidRequest(reason="nope")], classify_result="transient"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 0
        assert rejected.value.request is None
        assert rejected.value.reason == "nope"
        assert rejected.value.error_text == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.provider_name == adapter.provider_name
        assert rejected.value.attempt_records == ()
        assert rejected.value.usage == ZERO_USAGE

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
        bound_llm = LLM(classified_adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        assert classified.value.request == _FakeRequest(messages=(UserMessage(content="hi"),))

    asyncio.run(scenario())


def test_refusal_outcome_raises_without_retry() -> None:
    """A Refusal outcome becomes a RefusalError carrying the attempt record, never retried.

    The record carries the turn the refusal arrived on and the response it was read from.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports the Refusal variant."""
        adapter = _FakeAdapter(failures=[_billed(Refusal(assistant_message=_REJECTED_TURN))])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(RefusalError) as refusal:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
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
def test_a_no_output_outcome_raises_without_retry(
    outcome: ResponseOutcome[str],
    expected_error: type[GenerationError],
    expected_stop_reason: StopReason,
) -> None:
    """Each outcome carrying no output fails the item on the first attempt, with its own error and reason.

    None of the three is retried: the token cap, the turn that finished saying nothing, and the
    request too long to serve all repeat on a resend.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports the outcome."""
        adapter = _FakeAdapter(failures=[_billed(outcome)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(expected_error) as caught:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = caught.value
        assert failure.attempts == 1
        assert failure.stop_reason == expected_stop_reason
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_schema_violation_outcome_raises_without_retry() -> None:
    """A SchemaViolation outcome fails the item, and pydantic's rejection travels on the error.

    error_text carries none of the rejection, whose msg embeds the value a caller's own validator
    rejected; the tracing layer writes error_text into every span whatever capture_message_content
    the caller chose.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports SchemaViolation."""
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(SchemaViolationError) as schema_violation:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = schema_violation.value
        assert failure.attempts == 1
        assert failure.stop_reason == "end_turn"
        assert failure.validation_error_json == _VALIDATION_ERROR_JSON
        assert "SENTINEL" not in failure.error_text
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_unfinished_turn_outcome_raises_carrying_the_adapter_s_reason() -> None:
    """An UnfinishedTurn outcome fails the item, and the adapter's reason reaches error_text.

    The reason is the only description of a 200 langchaint does not model, so the error must carry
    it rather than a constant of its own.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports UnfinishedTurn."""
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnfinishedTurnError) as unfinished_turn:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = unfinished_turn.value
        assert "pause_turn" in failure.error_text
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_provider_failed_transiently_is_retried_and_keeps_its_billing() -> None:
    """The outcome is retried, that 200's billing lands on its record, and the reason on its error."""

    async def scenario() -> None:
        """Drive one generate_one whose first attempt reports the failure and whose second succeeds."""
        adapter = _FakeAdapter(failures=[_billed(_PROVIDER_FAILED_TRANSIENTLY)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 2
        assert response.attempts == 2
        rejected, succeeded = response.attempt_records
        assert isinstance(rejected.error, TransientError)
        assert str(rejected.error) == _PROVIDER_FAILURE_REASON
        assert rejected.usage.cost_in_usd == 0.25
        assert succeeded.error is None

    asyncio.run(scenario())


def test_provider_failed_transiently_carrying_the_rate_limit_flag_pauses_admission() -> None:
    """A failure the provider named a rate limit pauses every request sharing the rate-limit quota.

    `is_rate_limit` is the only distinction from another failed response.
    The exception must leave `admitted()` to pause the rate-limit quota.
    """

    async def scenario() -> None:
        """Spend the whole budget on rate-limited failures, so the pause is still running at the end."""
        shared_backoff = _fast_shared_backoff()
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
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True,
            max_attempts=1,
        )
        with pytest.raises(RetriesExhaustedError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = exhausted.value.attempt_records
        assert isinstance(record.error, TransientError)
        assert record.error.is_rate_limit
        # Only a PauseAll record moves _pause_until off the sentinel, so this is the flag arriving.
        assert shared_backoff._pause_until != _NEVER

    asyncio.run(scenario())


def test_provider_failed_terminally_raises_without_retry() -> None:
    """A terminal provider failure fails the item once, with the provider's own text as the reason.

    Never retried: what the body names is a property of the request, so the retry budget would buy
    the same body at full price each time.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports the terminal failure."""
        adapter = _FakeAdapter(
            failures=[
                _billed(
                    ProviderFailedTerminally(
                        reason=_PROVIDER_FAILURE_REASON, assistant_message=_REJECTED_TURN
                    )
                )
            ]
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ProviderFailedTerminallyError) as provider_failure:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 3

    asyncio.run(scenario())


def test_exception_classified_invalid_request_fails_the_item_without_retry() -> None:
    """A plain exception classified invalid_request raises InvalidRequestError on the first attempt.

    InvalidRequestError is a GenerationError, so in a batch it becomes the item's own failure
    rather than touching the siblings; the classified exception stays reachable as __cause__.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt raises a classify-invalid_request exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="invalid_request")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(InvalidRequestError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        assert rejected.value.reason == "the provider rejected the request: boom"
        assert isinstance(rejected.value.__cause__, ValueError)

    asyncio.run(scenario())


def test_exception_classified_unknown_exception_fails_the_item_without_retry() -> None:
    """A plain exception classified unknown_exception raises UnknownExceptionError on the first attempt.

    UnknownExceptionError is a GenerationError, so in a batch it becomes the item's own failure
    and the siblings run on. Nothing arrived, so the attempt it ends has no record.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt raises a classify-unknown_exception exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="unknown_exception")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnknownExceptionError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
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
        """Drive one generate_one whose attempt raises a classify-declared_final exception."""
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="declared_final")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ProviderDeclaredFinalError) as declared_final:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = declared_final.value
        assert isinstance(failure.error, ValueError)
        assert isinstance(failure.__cause__, ValueError)
        assert failure.error_text == "a final error from the provider: boom"
        (record,) = failure.attempt_records
        assert record.error is None
        assert record.raw is None
        assert record.usage == ZERO_USAGE
        assert failure.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_a_pause_all_do_not_retry_verdict_stops_the_item_and_pauses_the_rate_limit_quota() -> None:
    """The terminal half ends the retry loop, and the pausing half still holds every other request.

    Both halves matter: a verdict that only stopped would leave the siblings sending into the limit,
    and one that only paused would spend the whole retry budget against the provider's own "false".
    classify says invalid_request and the error is ProviderDeclaredFinalError anyway, which is the
    verdict naming the failure: the real classify calls a 429 invalid_request off its status.
    """

    async def scenario() -> None:
        """Drive one generate_one whose only failure parses to PauseAllDoNotRetry."""
        shared_backoff = _fast_shared_backoff(parse=_parse_pause_all_do_not_retry)
        adapter = _FakeAdapter(
            failures=[TransientError("throttled")], classify_result="invalid_request"
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        with pytest.raises(ProviderDeclaredFinalError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        # Only a pausing record moves _pause_until off the sentinel, so this is the pause arriving.
        assert shared_backoff._pause_until != _NEVER

    asyncio.run(scenario())


def test_a_mid_drain_failure_is_retried_and_records_what_the_stream_reported() -> None:
    """An attempt cut off mid-drain is a retried attempt billing the stream's in-flight report.

    The assembled response that would state the attempt's billing never arrived, so the record's
    usage and request id are what the stream had reported when the failure cut it off.
    """

    async def scenario() -> None:
        """Exhaust a one-attempt budget on a stream that fails after its first item."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        adapter = _FakeAdapter(stream=stream, classify_result="transient")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True,
            max_attempts=1,
        )
        with pytest.raises(RetriesExhaustedError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = exhausted.value.attempt_records
        assert record.usage == _USAGE_STREAM
        assert record.request_id == "req-fake-stream"
        assert stream.closed

    asyncio.run(scenario())


def test_a_deadline_expiring_mid_drain_reports_the_streams_in_flight_billing() -> None:
    """A TimedOutError cut out of a mid-drain hang carries what the stream had reported.

    The cancellation lands inside the drain, so no record settles; the loop notes the stream's
    reported billing on the ledger before closing it, and the deadline account reports it.
    """

    async def scenario() -> None:
        """Hang a stream after its first item and let a short deadline expire."""
        stream = _HangsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(TimedOutError) as raised:
            await bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=0.05)
        timed_out = raised.value
        assert timed_out.attempt_records == ()
        assert timed_out.billing_in_flight is not None
        assert timed_out.billing_in_flight.usage == _USAGE_STREAM
        assert timed_out.usage == _USAGE_STREAM
        assert stream.closed

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_settled_attempts_billing_is_counted_once_after_a_later_deadline_cut() -> None:
    """Billing noted in flight is cleared when its attempt settles, so a record is not counted twice.

    The first attempt's mid-drain failure notes the stream's billing, then its record carries the
    same billing; the deadline then cuts the second attempt's open, and the account reports the
    settled record's usage once, with nothing in flight.
    """

    async def scenario() -> None:
        """Fail one attempt mid-drain with billing reported, then hang the second attempt's open."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        adapter = _FakeAdapter(stream=stream, classify_result="transient", hang_from_open=2)
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(TimedOutError) as raised:
            await bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=0.05)
        timed_out = raised.value
        (record,) = timed_out.attempt_records
        assert record.usage == _USAGE_STREAM
        assert timed_out.billing_in_flight is None
        assert timed_out.usage == _USAGE_STREAM

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_generate_request_id_on_the_assembled_response_outranks_the_streams_own() -> None:
    """The generate loop fills the request id in where the response has none, never replacing one.

    The shipped adapters never hit this: their SDKs leave the assembled response without the header.
    An adapter whose stream cannot reach the response headers is what the rule protects, because
    overwriting would trade the id it does have for a null.
    """

    async def scenario() -> None:
        """Generate over a stream whose assembled response names its own request."""
        stream = _FakeStream()
        stream.raw = _FakeRawResponse(id="fake-final", request_id="req-from-assembled")
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = response.attempt_records
        assert record.request_id == "req-from-assembled"

    asyncio.run(scenario())


def test_a_stream_protocol_error_is_retried_to_exhaustion() -> None:
    """A StreamProtocolError is retried without classify and ends as RetriesExhaustedError.

    The class is langchaint's own, so classify cannot place it; no item from the private drain
    reached any caller, so a resend is safe, and each record carries the violation's text.
    """

    async def scenario() -> None:
        """Exhaust a two-attempt budget on a stream that violates the protocol every time."""
        adapter = _FakeAdapter(stream=_ProtocolErrorStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True,
            max_attempts=2,
        )
        with pytest.raises(RetriesExhaustedError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 2
        first, second = exhausted.value.attempt_records
        assert "stream ended without a stop event" in str(first.error)
        assert "stream ended without a stop event" in str(second.error)

    asyncio.run(scenario())


def test_a_close_that_raises_does_not_displace_a_generate_success() -> None:
    """An exception from the post-drain close is logged, and the drained success stands."""

    async def scenario() -> None:
        """Generate over a stream whose close raises after a successful drain."""
        stream = _FailingCloseStream()
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ab"
        assert stream.closed

    asyncio.run(scenario())


def test_a_mid_drain_exception_nobody_can_place_still_records_the_attempt() -> None:
    """An unplaceable failure after the stream opened records that the attempt reached the provider.

    The same classification on an open failure records nothing; here the stream was open, so the
    record exists and bills ZERO_USAGE, the stream having reported nothing before the failure.
    """

    async def scenario() -> None:
        """Fail the drain with an exception classify calls unknown_exception."""
        stream = _UnnamedItemErrorStream()
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(UnknownExceptionError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = unplaceable.value.attempt_records
        assert record.usage == ZERO_USAGE
        assert record.request_id == "req-fake-stream"
        assert stream.closed

    asyncio.run(scenario())


def test_an_unplaceable_exception_fails_only_its_item() -> None:
    """A classify-unknown_exception item comes back as its UnknownExceptionError at its index.

    The sibling succeeds.
    """

    async def scenario() -> None:
        """Serialize a two-item batch (max_concurrent_requests=1) whose first attempt is unplaceable."""
        adapter = _FakeAdapter(
            echo=True, failures=[ValueError("boom")], classify_result="unknown_exception"
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
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
        """Settle one item, then cancel the batch while the other's open hangs."""
        adapter = _FakeAdapter(hang_from_open=2)
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=1)
        ).bind(automatic_prompt_caching=True)
        call = asyncio.create_task(
            bound_llm.generate_many([[UserMessage(content="a")], [UserMessage(content="b")]])
        )
        await asyncio.sleep(0.02)
        _ = call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert adapter.bound_adapters[0].open_count == 2

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
    """Pin every binding's static BoundLLM type, the pair the request-method overloads key on.

    One binding returns a union: structured plus a ToolManager generates GenerateResult, whose
    ToolCallTurn variant is the tool-call turn. Every other binding generates Response alone.
    Every transition is exact, so dropping the ToolManager drops the ToolCallTurn variant with it.
    It also pins tool_manager's own type, which is what a tool loop dispatches through.
    """
    llm = LLM(_FakeAdapter())
    tool_manager = ToolManager([])

    text = llm.bind(automatic_prompt_caching=True)
    assert_type(text, BoundLLM[str, None])
    text_with_tools = llm.bind(tool_manager=tool_manager, automatic_prompt_caching=True)
    assert_type(text_with_tools, BoundLLM[str, ToolManager])
    structured = llm.bind(response_format=_Answer, automatic_prompt_caching=True)
    assert_type(structured, BoundLLM[_Answer, None])
    structured_with_tools = llm.bind(
        response_format=_Answer, tool_manager=tool_manager, automatic_prompt_caching=True
    )
    assert_type(structured_with_tools, BoundLLM[_Answer, ToolManager])
    provider_tool = ({"type": "web_search"},)
    text_with_provider_tool = llm.bind(
        provider_executed_tools=provider_tool,
        automatic_prompt_caching=True,
    )
    assert_type(text_with_provider_tool, BoundLLM[str, None])
    assert_type(
        structured.rebind(provider_executed_tools=provider_tool),
        BoundLLM[_Answer, None],
    )

    # BoundLLM[X] is BoundLLM[X, None]: the PEP 696 default keeps the common annotation short.
    assert_type(structured, BoundLLM[_Answer])

    assert_type(text_with_tools.tool_manager, ToolManager)
    assert_type(text.tool_manager, None)

    assert_type(structured.rebind(tool_manager=tool_manager), BoundLLM[_Answer, ToolManager])
    assert_type(structured_with_tools.rebind(tool_manager=None), BoundLLM[_Answer, None])
    assert_type(text_with_tools.rebind(response_format=_Answer), BoundLLM[_Answer, ToolManager])
    assert_type(structured_with_tools.rebind(response_format=None), BoundLLM[str, ToolManager])
    assert_type(structured_with_tools.rebind(system_prompt="s"), BoundLLM[_Answer, ToolManager])


async def _pin_request_method_return_types(llm: LLM, tool_manager: ToolManager) -> None:
    """Pin the return types the ToolManagerT overloads produce, which is what the parameter is for.

    Never called: pyrefly checks this body, and the assertions are about types alone.
    """
    structured_with_tools = llm.bind(
        response_format=_Answer, tool_manager=tool_manager, automatic_prompt_caching=True
    )
    assert_type(
        await structured_with_tools.generate_one("hi"),
        Response[_Answer] | ToolCallTurn[_Answer],
    )
    assert_type(
        structured_with_tools.stream_one("hi"), StreamHandle[_Answer, ToolCallTurn[_Answer]]
    )
    assert_type(
        await structured_with_tools.generate_many(["hi"]),
        list[CallResult[_Answer]],
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


def test_splits_tool_call_turns_only_on_the_structured_tool_bound_binding() -> None:
    """The split reads the binding: response_format and tool_manager must both be present."""
    llm = LLM(_FakeAdapter())
    tool_manager = ToolManager([])
    assert not llm.bind(automatic_prompt_caching=True)._splits_tool_call_turns
    assert not llm.bind(
        tool_manager=tool_manager, automatic_prompt_caching=True
    )._splits_tool_call_turns
    assert not llm.bind(
        response_format=_Answer, automatic_prompt_caching=True
    )._splits_tool_call_turns
    assert llm.bind(
        response_format=_Answer, tool_manager=tool_manager, automatic_prompt_caching=True
    )._splits_tool_call_turns


class _ScriptedStructuredBoundAdapter(BoundAdapter[_Answer | None]):
    """A structured bound adapter handing every request one scripted outcome.

    The ToolCallTurn split tests generate through it: _FakeStructuredBoundAdapter deliberately never
    generates, and _FakeBoundAdapter is bound to str.
    """

    def __init__(self, outcome: ResponseOutcome[_Answer | None]) -> None:
        """Store the one outcome interpret returns for every response."""
        self._outcome = outcome
        self.stream = _FakeStream()

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Bill every response the fixed _USAGE."""
        return stated_billing(_USAGE)

    @override
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Report a fixed identity; no split test reads it."""
        return ResponseIdentity(
            model_served="fake-model-served",
            response_id="structured-response",
            request_id=None,
        )

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[_Answer | None]:
        """Return the scripted outcome, whichever raw the request path produced."""
        return self._outcome

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams:
        """Carry the messages into the request."""
        return _FakeRequest(messages=tuple(messages))

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Hand back the stored fake stream; interpret ignores the raw it assembles."""
        return self.stream


_STRUCTURED_TOOL_CALL_TURN: AdapterResult[_Answer | None] = AdapterResult(
    output=None,
    assistant_message=AssistantMessage(turn=(_FAKE_TOOL_CALL,)),
    stop_reason="tool_use",
)
"""A structured turn of tool calls alone: nothing parsed, one call to dispatch."""


def _structured_tool_bound_llm(
    outcome: ResponseOutcome[_Answer | None],
) -> BoundLLM[_Answer, ToolManager]:
    """Bind structured plus tools, then swap the scripted fake in for bind's never-generating one.

    Swapping only _bound_adapter keeps the binding LLM.bind built, so _splits_tool_call_turns reads
    the real response_format and tool_manager and the retry loops run unchanged over the scripted outcome.
    """
    bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
        response_format=_Answer, tool_manager=ToolManager([]), automatic_prompt_caching=True
    )
    bound_llm._bound_adapter = _ScriptedStructuredBoundAdapter(outcome)
    return bound_llm


def test_structured_tool_bound_generate_one_returns_the_tool_call_turn_variant() -> None:
    """A structured tool-bound turn that called tools reaches the caller as ToolCallTurn.

    Its output is the unparsed None and its tool_calls are the turn's, what a tool loop dispatches;
    a retry loop that dropped _splits_tool_call_turns would hand back a Response instead.
    """

    async def scenario() -> None:
        result = await _structured_tool_bound_llm(_STRUCTURED_TOOL_CALL_TURN).generate_one("hi")
        assert isinstance(result, ToolCallTurn)
        assert result.kind == "tool_call_turn"
        assert result.output is None
        assert result.tool_calls == (_FAKE_TOOL_CALL,)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_structured_tool_bound_generate_one_returns_the_response_variant_on_a_final_turn() -> None:
    """A turn without tool calls reaches the caller as Response, the parsed instance on output."""

    async def scenario() -> None:
        answer = _Answer(value=7)
        outcome: AdapterResult[_Answer | None] = AdapterResult(
            output=answer,
            assistant_message=AssistantMessage(turn=(TextPart(text=answer.model_dump_json()),)),
            stop_reason="end_turn",
        )
        result = await _structured_tool_bound_llm(outcome).generate_one("hi")
        assert isinstance(result, Response)
        assert result.output is answer

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_structured_tool_bound_stream_final_returns_the_tool_call_turn_variant() -> None:
    """The stream path splits the same way: final() on a tool-call turn is the ToolCallTurn variant."""

    async def scenario() -> None:
        bound_llm = _structured_tool_bound_llm(_STRUCTURED_TOOL_CALL_TURN)
        async with bound_llm.stream_one("hi") as handle:
            result = await handle.final()
        assert isinstance(result, ToolCallTurn)
        assert result.tool_calls == (_FAKE_TOOL_CALL,)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


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


def test_bind_and_rebind_carry_extra_body_by_reference() -> None:
    """LLM.bind puts extra_body on the Binding unchanged; rebind keeps, replaces, or clears it."""
    adapter = _FakeAdapter()
    extra_body = {"safety_identifier": "user-7"}
    bound_llm = LLM(adapter).bind(extra_body=extra_body, automatic_prompt_caching=True)
    assert bound_llm.binding.extra_body is extra_body
    assert bound_llm.rebind(system_prompt="s").binding.extra_body is extra_body
    replacement = {"safety_identifier": "user-8"}
    assert bound_llm.rebind(extra_body=replacement).binding.extra_body is replacement
    assert bound_llm.rebind(extra_body=None).binding.extra_body is None


def test_rebind_keeps_replaces_and_removes_provider_executed_tools() -> None:
    """Rebind preserves omitted tools and removes them with an empty tuple."""
    first_tool = {"type": "web_search"}
    second_tool = {"type": "file_search", "vector_store_ids": ["vs_1"]}
    bound = LLM(_FakeAdapter()).bind(
        provider_executed_tools=(first_tool,),
        max_attempts=5,
        automatic_prompt_caching=True,
    )
    assert bound.binding.provider_executed_tools == (first_tool,)
    assert bound.rebind(system_prompt="s").binding.provider_executed_tools == (first_tool,)
    replaced = bound.rebind(provider_executed_tools=(second_tool,))
    assert replaced.binding.provider_executed_tools == (second_tool,)
    assert replaced.max_attempts == 5
    assert bound.rebind(provider_executed_tools=()).binding.provider_executed_tools == ()


def test_generate_many_aligns_results_with_inputs() -> None:
    """Result i belongs to generation_inputs[i], preserving input order."""

    async def scenario() -> None:
        """Run a two-item batch whose fake echoes each item's first turn."""
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        assert _batch_outputs(results) == ["a", "b"]

    asyncio.run(scenario())


def test_generate_many_aligns_a_failure_among_successes() -> None:
    """A mixed batch keeps each result at its input index: the failure where it failed, successes elsewhere."""

    async def scenario() -> None:
        """Serialize a three-item batch at one concurrent request, failing the first under a one-attempt budget.

        One permit runs the items in submission order,
        so the single scripted failure lands on the first item and the other two succeed,
        which is exactly the mixed-outcome alignment under test.
        """
        adapter = _FakeAdapter(echo=True, failures=[TransientError("x")])
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True,
            max_attempts=1,
        )
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


def test_generate_many_returns_a_refusal_at_its_index() -> None:
    """An item whose attempt reports Refusal comes back as the RefusalError at its index, siblings succeed."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_concurrent_requests=1) whose first attempt reports Refusal."""
        adapter = _FakeAdapter(
            echo=True,
            failures=[_billed(Refusal(assistant_message=_REJECTED_TURN))],
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
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


def test_invalid_request_fails_only_its_item() -> None:
    """A rejected item comes back as its InvalidRequestError at its index; the sibling still succeeds.

    Nothing a single item does reaches a sibling, so the batch returns one outcome per generation_input.
    """

    async def scenario() -> None:
        """Serialize a two-item batch (max_concurrent_requests=1) whose first build_request refuses."""
        adapter = _FakeAdapter(
            echo=True, invalid_requests=[InvalidRequest(reason="misconfigured")]
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
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
    """warm_cache completes generation_inputs[0] before any sibling starts; the rest run at normal concurrency."""

    async def scenario() -> None:
        """Run an identical three-item batch on two fresh slow fakes and compare the recorded peaks.

        Warmed, the first open overlaps nothing and the remaining two overlap each other, so the peak is 2;
        the unwarmed control reaches 3, proving warm_cache alone changed the ordering.
        A fresh adapter per run keeps the two peaks independent readings.
        """
        generation_inputs = [[UserMessage(content=str(index))] for index in range(3)]
        warmed_adapter = _FakeAdapter(echo=True, open_seconds=0.01)
        warmed_bound_llm = LLM(
            warmed_adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=8)
        ).bind(automatic_prompt_caching=True)
        warmed = await warmed_bound_llm.generate_many(generation_inputs, warm_cache=True)
        assert _batch_outputs(warmed) == ["0", "1", "2"]
        assert warmed_adapter.bound_adapters[0].peak_in_flight == 2
        control_adapter = _FakeAdapter(echo=True, open_seconds=0.01)
        control_bound_llm = LLM(
            control_adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=8)
        ).bind(automatic_prompt_caching=True)
        control = await control_bound_llm.generate_many(generation_inputs)
        assert _batch_outputs(control) == ["0", "1", "2"]
        assert control_adapter.bound_adapters[0].peak_in_flight == 3

    asyncio.run(scenario())


def test_generate_many_warm_cache_first_failure_still_admits_the_rest() -> None:
    """A first item ending in a GenerationError stays at its index and the siblings still run."""

    async def scenario() -> None:
        """Fail the deterministic first attempt under a one-attempt budget; the other two succeed."""
        adapter = _FakeAdapter(echo=True, failures=[TransientError("x")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True,
            max_attempts=1,
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


def test_a_defect_becomes_one_items_failure_and_leaves_the_batch_complete() -> None:
    """An Exception from langchaint's own machinery fails its item and no sibling.

    Propagating it instead would discard the whole returned list, so a sibling that had already
    settled would lose both its output and the account of what it spent.
    """

    async def scenario() -> None:
        """Raise past the retry loop on the one item whose attempt fails, and let the other succeed."""
        adapter = _ClassifyRaisesAdapter(failures=[ValueError("defect")], echo=True)
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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

    Its callers match on GenerationError to handle the failure, so a defect arriving as anything
    else escapes past their handling entirely.
    """

    async def scenario() -> None:
        """Fail the one attempt and let classify raise past the retry loop."""
        adapter = _ClassifyRaisesAdapter(failures=[ValueError("defect")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(EscapedExceptionError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        assert str(raised.value) == "an exception escaped langchaint: classify defect"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_parse_contract_violation_surfaces_as_langchaints_defect_not_a_provider_outcome() -> (
    None
):
    """generate_one raises EscapedExceptionError whose error is the ParserContractError.

    A parse contract violation must not take the transport-failure path,
    where classify would report it as an UnknownExceptionError, a provider outcome.
    """

    async def scenario() -> None:
        """Fail the one attempt with a TransientError whose parse raises."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff(parse=_parse_raises)).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(EscapedExceptionError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        assert isinstance(raised.value.error, ParserContractError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_parse_contract_violation_on_a_stream_open_reaches_the_caller() -> None:
    """Entering stream_one raises the ParserContractError itself; no stream-path frame wraps it."""

    async def scenario() -> None:
        """Fail the one open with a TransientError whose parse raises."""
        adapter = _FakeAdapter(failures=[TransientError("boom")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff(parse=_parse_raises)).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ParserContractError):
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass

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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
    """A bare str reaches the adapter as a Sequence[Message] of one UserMessage."""

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
        with pytest.raises(TypeError, match="generation_inputs is a bare str"):
            # pyrefly: ignore[no-matching-overload]
            await bound_llm.generate_many("hi")
        assert adapter.bound_adapters[0].open_count == 0

    asyncio.run(scenario())


def test_stream_one_accepts_a_bare_str() -> None:
    """stream_one coerces a bare str to a Sequence[Message] of one UserMessage."""

    async def scenario() -> None:
        """Build a handle from a bare str and check the stored messages."""
        bound_llm = LLM(_FakeAdapter()).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one("hi") as handle:
            assert handle._messages == (UserMessage(content="hi"),)
            response = await handle.final()
        assert response.output == "ok"

    asyncio.run(scenario())


def test_stream_cancelled_mid_iteration_releases_the_permit() -> None:
    """A cancelled item pull returns its permit without waiting for the block to exit."""

    async def scenario() -> None:
        """Cancel a suspended item pull inside the block, then prove the permit is free."""
        adapter = _FakeAdapter(stream=_HangingStream())
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            consumer = asyncio.create_task(anext(handle))
            await asyncio.sleep(0.01)
            _ = consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            # Still inside the block, so only the cancellation can have freed the one permit.
            async with shared_backoff.admitted(budget=1.0):
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_cancelled_during_the_open_returns_the_permit() -> None:
    """A cancellation while the open is in flight returns its permit.

    __aexit__ never runs when __aenter__ raises, so only __aenter__ itself can exit the admission here.
    """

    async def scenario() -> None:
        """Time out an entry whose open_stream never returns, then prove the permit is free."""
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(_FakeAdapter(hang_from_open=1), shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True
        )

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns; the wait_for below cancels this."""
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        async with shared_backoff.admitted(budget=1.0):
            pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_inside_the_block_sets_its_abandoned() -> None:
    """A cancellation unwinding the block sets abandoned to what the stream could state.

    The record's attempt_records are empty because the streaming request is the in-flight attempt:
    only completed attempts settle records on a stream.
    This stream reports no running usage, which is what an openai stream does, so usage folds to
    ZERO_USAGE and billing_in_flight is None.
    """

    async def scenario() -> None:
        """Time out a consumer suspended on a hanging stream, then read the handle."""
        adapter = _FakeAdapter(stream=_HangingStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
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


def test_a_close_that_raises_still_returns_the_in_flight_permit() -> None:
    """A failed teardown does not cost the shared budget a permit, and does not reach the caller.

    Skipping a permit release permanently reduces `max_concurrent_requests`.
    """

    async def scenario() -> None:
        """Leave the block early over a stream whose close raises, then check both permits are free."""
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=2)
        bound_llm = LLM(
            _FakeAdapter(stream=_FailingCloseStream()), shared_backoff=shared_backoff
        ).bind(automatic_prompt_caching=True)
        # Leaving after one item keeps the admission held into __aexit__, so the close is the only
        # path that can return it; exhausting the iterator exits it before the close is reached.
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                break
        async with shared_backoff.admitted(budget=1.0), shared_backoff.admitted(budget=1.0):
            pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_during_the_open_sets_its_abandoned() -> None:
    """A cancellation landing in __aenter__ still records the abandonment.

    __aexit__ never runs when __aenter__ raises, so only __aenter__ itself can set it there.
    """

    async def scenario() -> None:
        """Time out an entry whose open never returns, then read the handle."""
        bound_llm = LLM(
            _FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()
        ).bind(automatic_prompt_caching=True)
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
        bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        completed = bound_llm.stream_one([UserMessage(content="hi")])
        async with completed:
            await completed.final()
        second_adapter = _FakeAdapter(stream=_HangingStream())
        second_bound_llm = LLM(second_adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
            _FakeAdapter(stream=_ProtocolErrorStream()), shared_backoff=_fast_shared_backoff()
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
"""The failing stream's wait, below the default `longest_wait_seconds`."""


def test_a_mid_stream_rate_limit_pauses_the_rate_limit_quota() -> None:
    """An iterator rate limit pauses admission and reaches the caller."""

    async def scenario() -> None:
        """Read the pause after one stream item fails."""
        shared_backoff = SharedBackoff(
            parse=_parse_fake, failure_types=(TransientError,), max_concurrent_requests=8
        )
        bound_llm = LLM(
            _FakeAdapter(stream=_RaisesItsOwnTransientErrorStream()),
            shared_backoff=shared_backoff,
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
        # A server-stated wait is followed un-jittered, so the pause is that wait from the failure.
        assert (
            shared_backoff._pause_until
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
            shared_backoff=_fast_shared_backoff(),
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

    __aexit__ closes before it sets abandoned, so that the record reports a returned permit and a
    closed connection. Without the set in a finally, the one exception the close does not swallow
    would take the cancelled stream's only account with it.
    """

    async def scenario() -> None:
        """Cancel the block, then let the close raise on the way out."""
        bound_llm = LLM(
            _FakeAdapter(stream=_CloseRaisesBaseExceptionStream(), classify_result="transient"),
            shared_backoff=_fast_shared_backoff(),
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
            shared_backoff=_fast_shared_backoff(),
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
            shared_backoff=_fast_shared_backoff(),
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
                assert "open stream failed during iteration" in str(record.error)
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
            shared_backoff=_fast_shared_backoff(),
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        assert "".join(texts) == "ok"
        assert response.output == "ok"
        assert response.stop_reason == "end_turn"
        assert response.model == "fake-model"
        assert response.provider_name == "fake"
        assert response.attempts == 1
        (record,) = response.attempt_records
        assert record.error is None
        assert record.started_at_monotonic_seconds <= record.ended_at_monotonic_seconds

    asyncio.run(scenario())


def test_stream_final_refusal_raises_without_retry() -> None:
    """A structured refusal detected in the stream's final() surfaces as a RefusalError.

    final() records the one 200 that produced no output and raises the RefusalError carrying that record.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports Refusal, then read the raised RefusalError."""
        adapter = _FakeAdapter(stream=_RefusingStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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

    The stream loop builds this error separately from the generate loop, and it is the one variant
    with a field of its own, so the reason must survive the trip through both.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports UnfinishedTurn, then read the raised error."""
        adapter = _FakeAdapter(stream=_UnfinishedTurnStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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


def test_stream_final_provider_failed_terminally_raises_carrying_the_providers_reason() -> None:
    """The outcome fails the call once, and the provider's own text is the error's message."""

    async def scenario() -> None:
        """Drain a stream whose final() reports the terminal failure, then read the raised error."""
        adapter = _FakeAdapter(stream=_ProviderFailedTerminallyStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        """Read a failure raised before the first item."""
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RetryUnavailableError) as raised:
                await handle.final()
        (record,) = raised.value.attempt_records
        assert record.response_id is None
        assert record.request_id == "req-fake-stream"

    asyncio.run(scenario())


def test_the_request_id_an_error_names_outranks_the_streams_own() -> None:
    """The error describes the failure, so its id wins where both channels have one."""

    async def scenario() -> None:
        """Read an item failure naming its request."""
        adapter = _FakeAdapter(
            stream=_FailsWithARequestIdBeforeFirstItemStream(), classify_result="transient"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RetryUnavailableError) as raised:
                await anext(handle)
        (record,) = raised.value.attempt_records
        assert record.request_id == "req-from-items-error"

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
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.request_id == "req-from-assembled"

    asyncio.run(scenario())


def test_stream_retry_populates_attempt_records() -> None:
    """An open_stream() failure precedes the successful attempt record.

    The stream path stages the assembled response's identity too, so the succeeding record carries
    what a sent one does, and the failed open carries the request id its error named.
    The succeeding record's request id is the stream's own: the assembled response carries no
    request-id header, which is what leaves identity_from_raw nothing to report.
    """

    async def scenario() -> None:
        """Open a stream whose first open_stream call fails, then drain it."""
        adapter = _FakeAdapter(
            failures=[_RequestIdError("connection reset", "req-from-open-failure")],
            classify_result="transient",
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        assert response.output == "ok"
        assert response.attempts == 2
        failed, succeeded = response.attempt_records
        assert str(failed.error) == "connection reset"
        assert (failed.model_served, failed.response_id) == (None, None)
        assert failed.request_id == "req-from-open-failure"
        assert succeeded.error is None
        assert succeeded.model_served == "fake-model-served"
        assert succeeded.response_id == "fake-response-2"
        assert succeeded.request_id == "req-fake-response-2"
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = response.attempt_records
        assert record.first_item_at_monotonic_seconds is None

    asyncio.run(scenario())


def test_stream_open_classified_invalid_request_carries_the_prior_attempts_records() -> None:
    """A rejected open raises InvalidRequestError from the entry with the prior transient record."""

    async def scenario() -> None:
        """Enter a handle whose first open fails transiently and whose second is rejected."""
        adapter = _FakeAdapter(
            failures=[TransientError("connection reset"), ValueError("boom")],
            classify_result="invalid_request",
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
    """build_request refusing the messages fails the item before any stream is opened.

    The handle builds the InvalidRequestError, so a caller reading model or attempt_records off it
    succeeds; it carries no request, there being none to carry.
    """

    async def scenario() -> None:
        """Enter a handle whose build_request refuses."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="nope")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="unknown_exception")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        adapter = _FakeAdapter(failures=[ValueError("boom")], classify_result="declared_final")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        with pytest.raises(ProviderDeclaredFinalError) as declared_final:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(declared_final.value.error, ValueError)
        assert declared_final.value.error_text == "a final error from the provider: boom"
        (record,) = declared_final.value.attempt_records
        assert record.error is None
        assert record.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_terminal_pause_stops_stream_open_and_pauses_rate_limit_quota() -> None:
    """A stream open the provider gave up on raises the item's failure instead of reopening.

    The open loop retries every verdict _terminal_error_or_none does not name terminal, so treating
    this one as a retry would spend the whole budget reopening a request the provider ended.
    classify says invalid_request and the error is ProviderDeclaredFinalError anyway, the same
    naming the non-streaming loop does.
    """

    async def scenario() -> None:
        """Enter a handle whose open failure parses to PauseAllDoNotRetry."""
        shared_backoff = _fast_shared_backoff(parse=_parse_pause_all_do_not_retry)
        adapter = _FakeAdapter(
            failures=[TransientError("throttled")], classify_result="invalid_request"
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        with pytest.raises(ProviderDeclaredFinalError):
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        # Only a pausing record moves _pause_until off the sentinel, so this is the pause arriving.
        assert shared_backoff._pause_until != _NEVER

    asyncio.run(scenario())


def test_a_pause_all_do_not_retry_verdict_ends_an_open_stream_with_the_items_failure() -> None:
    """A mid-stream failure the provider gave up on names the item's failure, not a retriable one.

    test_stream_item_failure_after_open_is_not_retried is the same pull under a RetryThisOne
    verdict, which raises RetryUnavailableError instead.
    classify says invalid_request and the error is ProviderDeclaredFinalError anyway, the same
    naming the non-streaming loop does.
    """

    async def scenario() -> None:
        """Read a first-item failure that parses to PauseAllDoNotRetry."""
        adapter = _FakeAdapter(
            stream=_FailsBeforeFirstItemStream(), classify_result="invalid_request"
        )
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(parse=_parse_pause_all_do_not_retry)
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(ProviderDeclaredFinalError) as declared_final:
                _ = await anext(handle)
        assert declared_final.value.error_text == (
            "a final error from the provider: dropped before the first item"
        )

    asyncio.run(scenario())


def test_stream_item_failure_after_open_is_not_retried() -> None:
    """A transient iterator failure ends the open stream."""

    async def scenario() -> None:
        """Read a transient failure before the first item."""
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RetryUnavailableError) as raised:
                await anext(handle)
        assert adapter.bound_adapters[0].open_count == 1
        assert raised.value.attempts == 1
        (record,) = raised.value.attempt_records
        assert str(record.error) == "dropped before the first item"
        assert record.first_item_at_monotonic_seconds is None

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

    unknown_exception is the classification that has to work with no verdict at all: anthropic reports a
    mid-stream failure as an error event on the 200 that carried the turn, so the adapter reads a
    status the retry policy cannot place. The stream is open under every classification, so what the
    provider reported for that attempt is readable and belongs on its record.
    """

    async def scenario() -> None:
        """Let the iteration hit the failure after one item, then read the record."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result=classify_result),
            shared_backoff=_fast_shared_backoff(),
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
            shared_backoff=_fast_shared_backoff(),
        ).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(UnknownExceptionError) as caught:
                async for _item in handle:
                    pass
        (record,) = caught.value.attempt_records
        assert record.billing is None
        assert record.usage == ZERO_USAGE

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_open_exhaustion_raises_retries_exhausted() -> None:
    """Opens that keep failing past the budget raise RetriesExhaustedError with the table fields set."""

    async def scenario() -> None:
        """Open a stream under a two-attempt budget whose every open_stream fails transiently."""
        adapter = _FakeAdapter(failures=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True,
            max_attempts=2,
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
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind(
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
        assert collected_items == ["ok", _FAKE_TOOL_CALL]

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


def test_a_retry_this_one_retry_after_floors_the_private_wait() -> None:
    """A server-stated wait floors private backoff without pausing shared admission.

    `minimum_wait_ceiling_seconds` is 0.001 seconds here.
    Only `retry_after_seconds` can produce the measured delay.
    """

    async def scenario() -> None:
        """Recover from one failure whose server-stated wait exceeds the tiny private ceiling."""
        retry_after_seconds = 0.05
        adapter = _FakeAdapter(
            failures=[TransientError("slow down", retry_after_seconds=retry_after_seconds)]
        )
        shared_backoff = _fast_shared_backoff(longest_wait_seconds=1.0)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True,
            max_attempts=2,
        )
        started_at = time.monotonic()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        elapsed_seconds = time.monotonic() - started_at
        assert response.output == "ok"
        assert response.attempts == 2
        assert elapsed_seconds >= retry_after_seconds
        # Only a PauseAll record moves _pause_until off the sentinel; a RetryThisOne must not.
        assert shared_backoff._pause_until == _NEVER

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_max_concurrent_requests_bounds_batch_concurrency() -> None:
    """A five-item batch under max_concurrent_requests=2 never overlaps more than two requests."""

    async def scenario() -> None:
        """Run the batch on a slow fake and read the recorded peak."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.01)
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=2)
        ).bind(automatic_prompt_caching=True)
        generation_inputs = [[UserMessage(content=str(index))] for index in range(5)]
        results = await bound_llm.generate_many(generation_inputs)
        assert _batch_outputs(results) == ["0", "1", "2", "3", "4"]
        assert adapter.bound_adapters[0].peak_in_flight == 2

    asyncio.run(scenario())


def test_backoff_sleep_does_not_hold_the_permit() -> None:
    """With max_concurrent_requests=1, a task backing off lets another request run.

    The failure is not a rate limit, so nothing pauses admission.
    Only a held permit could delay the second request.
    What pins the release is the second request's own duration.
    It is admitted while the first backs off, so it finishes in far less than that backoff.
    first_task.done() cannot pin this alone.
    A permit held across the sleep passes to the second request the moment the first retries.
    So the first is unfinished under either placement.
    The failure's retry_after_seconds floors the private wait.
    The floor makes the backoff outlast the second request deterministically.
    """

    async def scenario() -> None:
        """Interleave a retrying item with a clean one under one permit."""
        retry_after_seconds = 0.2
        adapter = _FakeAdapter(
            failures=[TransientError("boom", retry_after_seconds=retry_after_seconds)]
        )
        shared_backoff = _fast_shared_backoff(
            max_concurrent_requests=1,
            longest_wait_seconds=1.0,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True,
            max_attempts=2,
        )
        first_task = asyncio.create_task(bound_llm.generate_one([UserMessage(content="a")]))
        await asyncio.sleep(0.01)
        started_at = time.monotonic()
        second = await bound_llm.generate_one([UserMessage(content="b")])
        second_elapsed_seconds = time.monotonic() - started_at
        assert second.output == "ok"
        assert second_elapsed_seconds < retry_after_seconds / 2
        assert not first_task.done()
        first = await first_task
        assert first.output == "ok"
        assert first.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_protocol_error_releases_the_permit() -> None:
    """A StreamProtocolError from items() returns the permit and closes the stream."""

    async def scenario() -> None:
        """Drive final() into the protocol error, then re-admit inside the still-open block."""
        stream = _ProtocolErrorStream()
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(StreamProtocolError):
                await handle.final()
            assert stream.closed is True
            async with shared_backoff.admitted(budget=1.0):
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_releases_its_permit_when_exhausted() -> None:
    """Exhausting a stream returns its permit before the handle's block exits.

    The re-admission sits inside the still-open block, so only the release on exhaustion can
    satisfy it; after the block it would be satisfied by the release on block exit instead.
    """

    async def scenario() -> None:
        """Drain one stream under max_concurrent_requests=1, then re-admit inside the still-open block."""
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(_FakeAdapter(), shared_backoff=shared_backoff).bind(
            automatic_prompt_caching=True
        )
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                pass
            async with shared_backoff.admitted(budget=1.0):
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_rate_limited_stream_open_pauses_the_rate_limit_quota_and_the_retry_succeeds() -> None:
    """A rate-limited open pauses the rate-limit quota before its retry succeeds."""

    async def scenario() -> None:
        """Retry a stream open through a rate-limit failure, then finish the stream."""
        adapter = _FakeAdapter(
            failures=[
                TransientError("rate limited", retry_after_seconds=0.001, is_rate_limit=True)
            ]
        )
        shared_backoff = _fast_shared_backoff()
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        # Only a PauseAll record moves _pause_until off the sentinel, so this is the open's failure arriving.
        assert shared_backoff._pause_until != _NEVER
        assert response.output == "ok"
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
        _ = LLM(_FakeAdapter()).bind(system_prompt=[], automatic_prompt_caching=True)


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
            hang_from_open=settled_attempts + 1,
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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


def test_a_deadline_expiring_before_admission_reports_no_attempts() -> None:
    """A call that never got admitted reports attempts == 0, so a caller can tell it never sent.

    max_concurrent_requests is 1 and the holder never returns its permit, so the second call spends its whole
    deadline waiting for admission and sends nothing.
    """

    async def scenario() -> None:
        """Hold the only permit, then run a second call under a deadline it cannot outlast."""
        adapter = _FakeAdapter(hang_from_open=1)
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=1)
        ).bind(automatic_prompt_caching=True)
        holder = asyncio.create_task(bound_llm.generate_one([UserMessage(content="held")]))
        await asyncio.sleep(0.02)
        with pytest.raises(TimedOutError) as raised:
            await bound_llm.generate_one([UserMessage(content="queued")], timeout_seconds=0.05)
        timed_out = raised.value
        assert timed_out.attempts == 0
        assert timed_out.in_flight_attempt_started_at_monotonic_seconds is None
        assert timed_out.usage == ZERO_USAGE
        _ = holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_an_outer_cancellation_inside_the_deadline_stays_a_cancellation() -> None:
    """The scope langchaint owns claims only its own expiry, never a caller's cancellation.

    The deadline here is far from expiring when the outer cancellation lands.
    """

    async def scenario() -> None:
        """Cancel a call from outside while its own generous deadline is still running."""
        bound_llm = LLM(
            _FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()
        ).bind(automatic_prompt_caching=True)
        call = asyncio.create_task(
            bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=30.0)
        )
        await asyncio.sleep(0.02)
        _ = call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert call.cancelled()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_batch_times_out_one_item_and_returns_its_siblings() -> None:
    """Each item gets its own deadline, so one item running out does not cut a sibling."""

    async def scenario() -> None:
        """Hang the second item's open while the first answers."""
        adapter = _FakeAdapter(hang_from_open=2, echo=True)
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        results = await bound_llm.generate_many(
            [[UserMessage(content="fast")], [UserMessage(content="slow")]],
            max_working_seconds_per_item=0.05,
        )
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "fast"
        assert isinstance(second, TimedOutError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_batch_item_spends_no_budget_waiting_for_a_permit() -> None:
    """One permit and four items: the last waits past its whole budget to start, and still succeeds.

    Each open takes 0.15 seconds and only one item holds a permit at a time, so the fourth item
    starts 0.45 seconds after the batch does, against a budget of 0.30. A deadline running from
    when its task was created would have expired before it sent anything.
    """

    async def scenario() -> None:
        """Queue four items behind one permit, each item working well inside its budget."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.15)
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many(
            [[UserMessage(content=str(index))] for index in range(4)],
            max_working_seconds_per_item=0.30,
        )
        assert _batch_outputs(results) == ["0", "1", "2", "3"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_batch_item_spends_its_budget_once_it_is_admitted() -> None:
    """The budget still expires on an item that is admitted and gets nothing back.

    The second item waits 0.15 seconds for the one permit, which costs it nothing, then hangs.
    It is the hang that expires the budget, so the clock does run once the item is admitted.
    """

    async def scenario() -> None:
        """Let the first item answer, then hang the second one after it is admitted."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.15, hang_from_open=2)
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many(
            [[UserMessage(content="answered")], [UserMessage(content="hangs")]],
            max_working_seconds_per_item=0.30,
        )
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "answered"
        assert isinstance(second, TimedOutError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_batch_item_spends_no_budget_waiting_in_the_admission_queue() -> None:
    """Admission pacing spends none of an item's working-time budget.

    `max_concurrent_requests=None` lets every item reach admission immediately.
    `max_request_starts_per_second=10.0` starts one request every 0.1 seconds.
    The fourth request starts after its 0.1-second working-time budget.
    """

    async def scenario() -> None:
        """Pace four items through the admission queue on a budget shorter than the queue."""
        adapter = _FakeAdapter(echo=True)
        shared_backoff = _fast_shared_backoff(
            max_concurrent_requests=None,
            max_request_starts_per_second=10.0,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many(
            [[UserMessage(content=str(index))] for index in range(4)],
            max_working_seconds_per_item=0.1,
        )
        assert _batch_outputs(results) == ["0", "1", "2", "3"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_batch_item_spends_no_budget_waiting_out_a_shared_pause() -> None:
    """A shared pause spends none of an item's working-time budget.

    A rate-limit `TransientError` pauses admission for 0.3 seconds.
    `PauseAll` adds no private sleep.
    Each request spends 0.05 seconds from the 0.15-second budget.
    `longest_wait_seconds=0.5` keeps the stated pause unchanged.
    """

    async def scenario() -> None:
        """Fail one attempt into a pause longer than the budget, then let the retry answer."""
        adapter = _FakeAdapter(
            echo=True,
            open_seconds=0.05,
            failures=[TransientError("slow down", retry_after_seconds=0.3, is_rate_limit=True)],
        )
        shared_backoff = _fast_shared_backoff(
            max_concurrent_requests=None,
            longest_wait_seconds=0.5,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(automatic_prompt_caching=True)
        results = await bound_llm.generate_many(
            [[UserMessage(content="paused")]], max_working_seconds_per_item=0.15
        )
        assert _batch_outputs(results) == ["paused"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_batch_item_banks_what_is_left_of_its_budget_across_a_retry() -> None:
    """Two attempts draw on one budget, so a retrying item does not get the budget twice.

    Each attempt's open burns 0.12 seconds of a 0.20 second budget. Re-arming with the full
    budget on the second admission would let both attempts finish; banking the remainder leaves
    0.08 seconds, so the second attempt runs out and the item ends as a TimedOutError.
    """

    async def scenario() -> None:
        """Fail the first attempt transiently, then time out inside the retry."""
        adapter = _FakeAdapter(
            echo=True,
            open_seconds=0.12,
            failures=[TransientError("try again")],
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        results = await bound_llm.generate_many(
            [[UserMessage(content="retries")]], max_working_seconds_per_item=0.20
        )
        assert isinstance(results[0], TimedOutError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_stream_deadline_raises_and_leaves_abandoned_unset() -> None:
    """One cut-off call gets one account: the raise, not the handle field as well.

    Setting both would put the same call in the archive twice.
    """

    async def scenario() -> None:
        """Enter a stream whose open never returns, under a deadline."""
        bound_llm = LLM(
            _FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()
        ).bind(automatic_prompt_caching=True)
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
            _FakeAdapter(stream=_HangsAfterFirstItemStream()),
            shared_backoff=_fast_shared_backoff(),
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
        bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
            automatic_prompt_caching=True
        )
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        async with handle:
            response = await handle.final()
            await asyncio.sleep(0.1)
        assert response.output == "ok"
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
            _FakeAdapter(stream=_FailsAfterFirstItemStream()),
            shared_backoff=_fast_shared_backoff(),
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
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
        bound_llm = LLM(
            _FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()
        ).bind(automatic_prompt_caching=True)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await bound_llm.generate_one([UserMessage(content="hi")])

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
