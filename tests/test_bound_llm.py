"""Test BoundLLM and StreamHandle with fake adapters."""

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, assert_type, override

import pytest
from pydantic import BaseModel, TypeAdapter

from langchaint import (
    LLM,
    ZERO_USAGE,
    AllowedToolsChoice,
    AssistantMessage,
    BoundLLM,
    CallResult,
    CallResultRecord,
    CaptureTool,
    CutOffAttemptRecord,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DoNotRetry,
    EscapedExceptionErrorRecord,
    GenerationError,
    GenerationErrorRecord,
    InvalidRequestErrorRecord,
    Message,
    ParserContractError,
    PauseAllDoNotRetry,
    ProviderDeclaredFinalErrorRecord,
    PydanticTool,
    Response,
    ResponseRecord,
    RetriesExhaustedErrorRecord,
    SchemaViolationErrorRecord,
    SettledAttemptRecord,
    SharedBackoff,
    StopReason,
    StreamItem,
    StreamProtocolError,
    TextPart,
    TimedOutErrorRecord,
    ToolCall,
    ToolCallTurn,
    ToolCallTurnRecord,
    ToolManager,
    ToolSchema,
    TransientError,
    TransientErrorRecord,
    UnknownExceptionErrorRecord,
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
    ProviderBilling,
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
from tests.helpers import random_returns_zero, stated_provider_billing


def _settled_attempt_records(
    records: tuple[SettledAttemptRecord | CutOffAttemptRecord, ...],
) -> tuple[SettledAttemptRecord, ...]:
    settled = tuple(record for record in records if isinstance(record, SettledAttemptRecord))
    assert len(settled) == len(records)
    return settled


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
    """Map TransientError with verdict_from_transient_error."""
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
    """Assert success and return each batch output in order."""
    outputs: list[str] = []
    for result in results:
        assert isinstance(result, Response)
        outputs.append(result.output)
    return outputs


def _record_outputs(results: Sequence[CallResultRecord[str]]) -> list[str]:
    """Assert successful records and return each output in order."""
    outputs: list[str] = []
    for result in results:
        assert isinstance(result, ResponseRecord)
        outputs.append(result.output)
    return outputs


def _resume_json_object(resume_path: Path) -> dict[str, object]:
    """Parse a resume file as one JSON object."""
    return TypeAdapter(dict[str, object]).validate_json(resume_path.read_bytes())


class _FakeRawResponse(BaseModel):
    """Identify one fake raw response and its optional request ID."""

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
_REFUSAL = Refusal(assistant_message=_REJECTED_TURN)
_UNFINISHED_TURN = UnfinishedTurn(
    reason="anthropic returned stop_reason 'pause_turn'",
    assistant_message=_REJECTED_TURN,
)


def _success_result(content: str) -> AdapterResult[str]:
    """Build a successful text AdapterResult carrying the given content."""
    return AdapterResult(
        output=content,
        assistant_message=AssistantMessage(turn=(TextPart(text=content),)),
        stop_reason="end_turn",
    )


_FAKE_TOOL_CALL = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')


class _FakeStream(AdapterStream):
    """Provide fixed stream items and an assembled response."""

    def __init__(self, *, outcome: ResponseOutcome[str] | None = None) -> None:
        self.closed = False
        self.raw = _FakeRawResponse(id="fake-final")
        self._usage_reported: Usage | None = None
        """What billing_reported wraps; None stands for an adapter with no such channel."""
        self._outcome = outcome

    @override
    def billing_reported(self) -> ProviderBilling | None:
        """Wrap whatever the test set, defaulting to the None an openai stream returns."""
        return (
            None if self._usage_reported is None else stated_provider_billing(self._usage_reported)
        )

    @override
    def request_id(self) -> str | None:
        """Report a fixed header, standing in for the response headers a real SDK stream reads."""
        return "req-fake-stream"

    def scripted_response(self) -> _ScriptedResponse:
        """Return the assembled result the SDK would produce, and what the stream billed."""
        if self._outcome is not None:
            return _billed(self._outcome)
        return _ScriptedResponse(outcome=_success_result("ab"), usage=_USAGE_STREAM)

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


_VALIDATION_ERROR_JSON = (
    '[{"type":"value_error","loc":["celsius"],'
    '"msg":"Value error, SENTINEL is not a temperature","input":"SENTINEL"}]'
)
"""A pydantic rejection whose msg embeds the rejected value, as a caller's field_validator writes it."""
_SCHEMA_VIOLATION = SchemaViolation(
    validation_error_json=_VALIDATION_ERROR_JSON,
    assistant_message=_REJECTED_TURN,
)
_MAX_COMPLETION_TOKENS_EXCEEDED = MaxCompletionTokensExceeded(assistant_message=_REJECTED_TURN)
_EMPTY_TURN = EmptyTurn(assistant_message=_REJECTED_TURN)
_CONTEXT_WINDOW_EXCEEDED = ContextWindowExceeded(assistant_message=_REJECTED_TURN)


_PROVIDER_FAILURE_REASON = "The server had an error while processing your request."
"""A provider's own description of a failure, as openai puts it in a failed response's error."""

_PROVIDER_FAILED_TRANSIENTLY = ProviderFailedTransiently(
    reason=_PROVIDER_FAILURE_REASON, is_rate_limit=False, assistant_message=_REJECTED_TURN
)
"""One 200 whose body reports a failure a resend may get past, with no rate limit named."""
_PROVIDER_FAILED_TERMINALLY = ProviderFailedTerminally(
    reason=_PROVIDER_FAILURE_REASON,
    assistant_message=_REJECTED_TURN,
)


class _FinalRaisesStream(_FakeStream):
    """Yield items before final raises a fresh error."""

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
            Nothing. The raise precedes the first yield.

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
            Nothing. The raise precedes the first yield.

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
            Nothing. The wait never returns.
        """
        await asyncio.Event().wait()
        yield "unreachable"


class _HangsAfterFirstItemStream(_FakeStream):
    """Yield one item before suspending until cancellation."""

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
"""One open_stream exception or assembled response."""


class _ScriptedAttemptStream(_FakeStream):
    """Stream one scripted response.

    Each attempt has a fresh raw response and request ID.
    A success yields its content and _FAKE_TOOL_CALL.
    """

    def __init__(self, *, raw: _FakeRawResponse, content: str | None) -> None:
        """Store final output and optional streamed content."""
        super().__init__()
        self.raw = raw
        self._content = content

    @override
    def request_id(self) -> str | None:
        """Derive the request ID from raw.id."""
        return f"req-{self.raw.id}"

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        if self._content is not None:
            yield self._content
            yield _FAKE_TOOL_CALL


@dataclass(frozen=True, kw_only=True)
class _FakeRequest(RequestParams):
    """Store messages for a fake request."""

    messages: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        """Serialize messages as JSON."""
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
        scripted_attempts: Sequence[_ScriptedAttempt] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        open_seconds: float = 0.0,
        hang_from_open: int | None = None,
        open_barrier: asyncio.Barrier | None = None,
        open_barrier_from_call: int = 1,
    ) -> None:
        """Configure fake request and stream behavior.

        scripted_attempts and invalid_requests provide ordered outcomes.
        stream overrides unscripted streams.
        open_seconds and open_barrier control concurrency.
        hang_from_open suspends matching calls.
        final_raws records assembled response objects.
        """
        self._scripted_attempts = list(scripted_attempts)
        self._invalid_requests = list(invalid_requests)
        self._echo = echo
        self._explicit_stream = stream
        self._open_seconds = open_seconds
        self._hang_from_open = hang_from_open
        self._open_barrier = open_barrier
        self._open_barrier_from_call = open_barrier_from_call
        self._scripted_by_raw_id: dict[str, _ScriptedResponse] = {}
        self.final_raws: list[_FakeRawResponse] = []
        self.build_count = 0
        self.open_count = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Return what the response under this raw was scripted to have billed."""
        return stated_provider_billing(self._scripted_by_raw_id[_as_fake_raw(raw).id].usage)

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Name the fake model, take the response id from the raw's own, and the request id as it came."""
        fake_raw = _as_fake_raw(raw)
        return ResponseIdentity(
            model_served="fake-model-served",
            response_id=fake_raw.id,
            request_id=request_id,
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
        self, scripted_response: _ScriptedResponse, *, content: str | None
    ) -> _ScriptedAttemptStream:
        """Register the scripted response under a fresh raw and wrap it in this attempt's stream."""
        raw = _FakeRawResponse(id=f"fake-response-{self.open_count}")
        self._scripted_by_raw_id[raw.id] = scripted_response
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
        open_call = self.open_count
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._open_barrier is not None and open_call >= self._open_barrier_from_call:
                await self._open_barrier.wait()
            if self._hang_from_open is not None and self.open_count >= self._hang_from_open:
                await asyncio.Event().wait()
            if self._open_seconds:
                await asyncio.sleep(self._open_seconds)
            if self._scripted_attempts:
                scripted_attempt = self._scripted_attempts.pop(0)
                if isinstance(scripted_attempt, Exception):
                    raise scripted_attempt
                return self._attempt_stream(scripted_attempt, content=None)
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
    """A structured bound adapter for response_format replacement tests. It never generates.

    The replacement tests check binding identity and the switched content type.
    open_stream stays unreachable.
    """

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Unreachable: response_format replacement tests do not generate."""
        raise NotImplementedError

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Unreachable: response_format replacement tests do not generate."""
        raise NotImplementedError

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT]:
        """Unreachable: response_format replacement tests do not generate."""
        raise NotImplementedError

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams:
        """Unreachable: response_format replacement tests do not generate."""
        raise NotImplementedError

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Unreachable: response_format replacement tests do not generate."""
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
        scripted_attempts: Sequence[_ScriptedAttempt] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: _FakeStream | None = None,
        classify_result: ErrorClassification = "unknown_exception",
        open_seconds: float = 0.0,
        hang_from_open: int | None = None,
        open_barrier: asyncio.Barrier | None = None,
        open_barrier_from_call: int = 1,
        automatic_cache_breakpoints_default: bool = False,
    ) -> None:
        """Store how each freshly bound adapter behaves and the classify verdict."""
        # This adapter reaches no SDK, so it passes client=None.
        # The empty provider_name_by_client_class preserves the stated "fake" provider_name.
        super().__init__(
            client=None,
            model="fake-model",
            provider_name="fake",
            automatic_cache_breakpoints_default=automatic_cache_breakpoints_default,
        )
        self._scripted_attempts = scripted_attempts
        self._invalid_requests = invalid_requests
        self._echo = echo
        self._stream = stream
        self._classify_result = classify_result
        self._open_seconds = open_seconds
        self._hang_from_open = hang_from_open
        self._open_barrier = open_barrier
        self._open_barrier_from_call = open_barrier_from_call
        self.bound_adapters: list[_FakeBoundAdapter] = []
        self.structured_bind_count = 0

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return the fake adapter's stored request configuration."""
        return {"automatic_cache_breakpoints_default": self.automatic_cache_breakpoints_default}

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        bound = self._bound_adapter_class(
            scripted_attempts=self._scripted_attempts,
            invalid_requests=self._invalid_requests,
            echo=self._echo,
            stream=self._stream,
            open_seconds=self._open_seconds,
            hang_from_open=self._hang_from_open,
            open_barrier=self._open_barrier,
            open_barrier_from_call=self._open_barrier_from_call,
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
                max_attempts=max_attempts,
            )


def test_bind_rejects_invalid_max_attempts() -> None:
    """Reject replacement `max_attempts` values below one and boolean values."""
    bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind()
    for max_attempts in (True, False, 0, -1):
        with pytest.raises(ValueError, match="max_attempts"):
            _ = bound_llm.bind(max_attempts=max_attempts)


def test_a_raise_from_interpret_leaves_the_response_and_its_billing_on_the_record() -> None:
    """An interpretation failure preserves raw response and Billing."""

    async def scenario() -> None:
        """Drive one generate_one whose interpret raises over the response its stream assembled."""
        bound_llm = LLM(_InterpretRaisesAdapter(), shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = _settled_attempt_records(unplaceable.value.attempt_records)
        (provider_attempt,) = unplaceable.value.provider_attempts
        assert isinstance(provider_attempt.raw, _FakeRawResponse)
        assert record.usage == _USAGE
        assert record.assistant_message is None
        assert record.error is None
        assert isinstance(unplaceable.value.__cause__, RuntimeError)

    asyncio.run(scenario())


def test_stream_final_records_the_response_before_interpreting_it() -> None:
    """Interpret failure preserves the assembled response and Billing."""

    async def scenario() -> None:
        """Call final() on a stream whose interpret raises, then freeze the ledger it left."""
        stream = _FakeStream()
        bound_llm = LLM(
            _InterpretRaisesAdapter(stream=stream), shared_backoff=_fast_shared_backoff()
        ).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(RuntimeError, match="interpretation failed"):
                await handle.final()
            (record,) = _settled_attempt_records(handle._ledger.freeze().attempt_records)
            (provider_attempt,) = handle._ledger.provider_attempts
        assert provider_attempt.raw is stream.raw
        assert record.usage == _USAGE_STREAM
        assert record.assistant_message is None
        assert record.error is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_retry_recovers_after_a_transient_failure() -> None:
    """A transient failure followed by success returns both attempt records."""

    async def scenario() -> None:
        """Drive one generate_one through a single transient failure."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("boom")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(system_prompt="s")
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
        assert failed.started_after_seconds + failed.elapsed_seconds <= (
            succeeded.started_after_seconds
        )
        records_span = (
            succeeded.started_after_seconds
            + succeeded.elapsed_seconds
            - failed.started_after_seconds
        )
        assert response.elapsed_seconds >= records_span

    asyncio.run(scenario())


def test_a_call_builds_one_request_and_sends_it_once_per_attempt() -> None:
    """Two transient failures produce one build and three opened requests.

    build_request() runs once per call.
    Every retry sends the same RequestParams.
    """

    async def streamed_counts(adapter: _FakeAdapter) -> tuple[int, int]:
        """Drain one stream over adapter and return its bound adapter's build and open counts."""
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            await handle.final()
        bound = adapter.bound_adapters[0]
        return bound.build_count, bound.open_count

    async def scenario() -> None:
        """Drive generate_one and stream_one through two transient failures."""
        generate_adapter = _FakeAdapter(
            scripted_attempts=[TransientError("boom"), TransientError("boom again")]
        )
        generate_llm = LLM(generate_adapter, shared_backoff=_fast_shared_backoff()).bind()
        await generate_llm.generate_one([UserMessage(content="hi")])
        generate_bound = generate_adapter.bound_adapters[0]
        assert (generate_bound.build_count, generate_bound.open_count) == (1, 3)

        retried = _FakeAdapter(
            scripted_attempts=[TransientError("boom"), TransientError("boom again")]
        )
        assert await streamed_counts(retried) == (1, 3)

    asyncio.run(scenario())


def test_a_failed_attempt_records_the_request_id_off_its_error() -> None:
    """Each attempt records only its own request ID."""

    async def scenario() -> None:
        """Drive one generate_one through two transient failures, the first naming its request."""
        adapter = _FakeAdapter(
            scripted_attempts=[
                _RequestIdError("boom", "req-from-error"),
                TransientError("boom again"),
            ],
            classify_result="transient",
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        named, unnamed, succeeded = response.attempt_records
        assert named.request_id == "req-from-error"
        assert named.response_id is None
        assert unnamed.request_id is None
        assert succeeded.request_id == f"req-{_as_fake_raw(response.raw).id}"

    asyncio.run(scenario())


def test_an_adapter_raised_transient_error_still_names_its_request() -> None:
    """Adapter-raised TransientError bypasses classify and preserves request ID."""

    async def scenario() -> None:
        """Fail one generate attempt and one stream open with a transient error naming its request."""
        generate_adapter = _FakeAdapter(
            scripted_attempts=[_TransientRequestIdError("boom", "req-from-generate-transient")]
        )
        generate_llm = LLM(generate_adapter, shared_backoff=_fast_shared_backoff()).bind()
        generated = await generate_llm.generate_one([UserMessage(content="hi")])
        assert generated.attempt_records[0].request_id == "req-from-generate-transient"

        stream_adapter = _FakeAdapter(
            scripted_attempts=[_TransientRequestIdError("boom", "req-from-open-transient")]
        )
        bound_llm = LLM(stream_adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            streamed = await handle.final()
        assert streamed.attempt_records[0].request_id == "req-from-open-transient"

    asyncio.run(scenario())


def test_retry_exhaustion_raises_ordered_failure() -> None:
    """Exhausting the budget raises GenerationError carrying the ordered errors."""

    async def scenario() -> None:
        """Drive one generate_one to exhaustion under a two-attempt budget."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            max_attempts=2,
        )
        with pytest.raises(GenerationError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        failure = exhausted.value
        assert isinstance(failure.record, RetriesExhaustedErrorRecord)
        assert [str(error) for error in failure.record.errors_from_attempts] == ["e1", "e2"]
        assert [
            str(record.error) for record in _settled_attempt_records(failure.attempt_records)
        ] == ["e1", "e2"]
        assert failure.error_text == "attempt 1: e1; attempt 2: e2"
        assert failure.attempts == 2
        assert failure.model == adapter.model
        assert failure.provider_name == adapter.provider_name

    asyncio.run(scenario())


def test_attempt_record_bracket_excludes_the_backoff_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failed record's own span stays small. The backoff shows up as the gap between records.

    The random draw is pinned so each drawn wait is its ceiling, making the backoff gap deterministic.
    """
    monkeypatch.setattr(shared_backoff_module.random, "random", random_returns_zero)

    async def scenario() -> None:
        """Recover from one failure under a visible 0.05s backoff."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("boom")])
        shared_backoff = SharedBackoff(
            parse=_parse_fake,
            failure_types=(TransientError,),
            max_concurrent_requests=8,
            minimum_wait_ceiling_seconds=0.05,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            max_attempts=2,
        )
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        failed, succeeded = response.attempt_records
        assert failed.elapsed_seconds < 0.05
        backoff_gap = succeeded.started_after_seconds - (
            failed.started_after_seconds + failed.elapsed_seconds
        )
        assert backoff_gap >= 0.05
        assert response.elapsed_seconds >= 0.05

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_build_request_refusing_messages_fails_the_item_with_nothing_sent() -> None:
    """InvalidRequest fails before request admission and classification."""

    async def scenario() -> None:
        """Drive one generate_one whose build_request refuses under a transient classify verdict."""
        adapter = _FakeAdapter(
            invalid_requests=[InvalidRequest(reason="nope")], classify_result="transient"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 0
        assert rejected.value.request is None
        assert isinstance(rejected.value.record, InvalidRequestErrorRecord)
        assert rejected.value.record.reason == "nope"
        assert rejected.value.error_text == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.provider_name == adapter.provider_name
        assert rejected.value.attempt_records == ()
        assert rejected.value.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_rejection_after_transient_attempts_carries_their_records() -> None:
    """GenerationError preserves earlier attempt records and Usage."""

    async def scenario() -> None:
        """Settle one billed transient attempt, then have classify call the next one a rejection."""
        classified_adapter = _FakeAdapter(
            scripted_attempts=[
                _billed(_PROVIDER_FAILED_TRANSIENTLY),
                ValueError("bad request"),
            ],
            classify_result="invalid_request",
        )
        bound_llm = LLM(classified_adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as classified:
            await bound_llm.generate_one([UserMessage(content="hi")])
        billed_record, rejected_record = _settled_attempt_records(classified.value.attempt_records)
        assert billed_record.usage.cost_in_usd == 0.25
        assert rejected_record.error is None
        assert rejected_record.usage == ZERO_USAGE
        assert classified.value.provider_attempts[-1].raw is None
        assert isinstance(classified.value.__cause__, ValueError)
        assert classified.value.request == _FakeRequest(messages=(UserMessage(content="hi"),))

    asyncio.run(scenario())


def test_refusal_outcome_raises_without_retry() -> None:
    """A Refusal outcome becomes a GenerationError carrying the attempt record, never retried.

    The record carries the turn the refusal arrived on and the response it was read from.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports the Refusal variant."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_REFUSAL)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as refusal:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = refusal.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.usage.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None
        assert record.usage.cost_in_usd == 0.25
        assert record.assistant_message == _REJECTED_TURN
        assert failure.provider_attempts[0].raw is not None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("outcome", "expected_error", "expected_stop_reason"),
    [
        (
            _MAX_COMPLETION_TOKENS_EXCEEDED,
            GenerationError,
            "max_tokens",
        ),
        (_EMPTY_TURN, GenerationError, "end_turn"),
        (
            _CONTEXT_WINDOW_EXCEEDED,
            GenerationError,
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
    """Each terminal no-output outcome fails without retrying."""

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports the outcome."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(outcome)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(expected_error) as caught:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = caught.value
        assert failure.attempts == 1
        assert failure.stop_reason == expected_stop_reason
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_schema_violation_outcome_raises_without_retry() -> None:
    """SchemaViolation preserves ValidationError outside error_text."""

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports SchemaViolation."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_SCHEMA_VIOLATION)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as schema_violation:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = schema_violation.value
        assert failure.attempts == 1
        assert failure.stop_reason == "end_turn"
        assert isinstance(failure.record, SchemaViolationErrorRecord)
        assert failure.record.validation_error_json == _VALIDATION_ERROR_JSON
        assert "SENTINEL" not in failure.error_text
        assert failure.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_unfinished_turn_outcome_raises_carrying_the_adapter_s_reason() -> None:
    """An UnfinishedTurn outcome fails the item, and the adapter's reason reaches error_text.

    The error preserves the provider's reason.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports UnfinishedTurn."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_UNFINISHED_TURN)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as unfinished_turn:
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
        adapter = _FakeAdapter(scripted_attempts=[_billed(_PROVIDER_FAILED_TRANSIENTLY)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 2
        assert response.attempts == 2
        rejected, succeeded = response.attempt_records
        assert isinstance(rejected.error, TransientErrorRecord)
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
            scripted_attempts=[
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
            max_attempts=1,
        )
        with pytest.raises(GenerationError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = _settled_attempt_records(exhausted.value.attempt_records)
        assert isinstance(record.error, TransientErrorRecord)
        assert record.error.is_rate_limit
        # Only a PauseAll record moves _pause_until off the sentinel, so this is the flag arriving.
        assert shared_backoff._pause_until != _NEVER

    asyncio.run(scenario())


def test_provider_failed_terminally_raises_without_retry() -> None:
    """ProviderFailedTerminally fails once with the provider reason."""

    async def scenario() -> None:
        """Drive one generate_one whose attempt reports the terminal failure."""
        adapter = _FakeAdapter(scripted_attempts=[_billed(_PROVIDER_FAILED_TERMINALLY)])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as provider_failure:
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
            scripted_attempts=[ValueError("x1"), ValueError("x2")], classify_result="transient"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ok"
        assert response.attempts == 3

    asyncio.run(scenario())


def test_exception_classified_invalid_request_fails_the_item_without_retry() -> None:
    """A plain exception classified invalid_request raises GenerationError on the first attempt.

    GenerationError fails one batch item and preserves __cause__.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt raises a classify-invalid_request exception."""
        adapter = _FakeAdapter(
            scripted_attempts=[ValueError("boom")], classify_result="invalid_request"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as rejected:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(rejected.value.record, InvalidRequestErrorRecord)
        assert rejected.value.record.reason == "the provider rejected the request: boom"
        assert isinstance(rejected.value.__cause__, ValueError)

    asyncio.run(scenario())


def test_exception_classified_unknown_exception_fails_the_item_without_retry() -> None:
    """A plain exception classified unknown_exception raises GenerationError on the first attempt.

    GenerationError fails one batch item without an AttemptRecord.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt raises a classify-unknown_exception exception."""
        adapter = _FakeAdapter(
            scripted_attempts=[ValueError("boom")], classify_result="unknown_exception"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = unplaceable.value
        assert isinstance(failure, GenerationError)
        assert isinstance(failure.record, UnknownExceptionErrorRecord)
        assert isinstance(failure.__cause__, ValueError)
        assert failure.error_text == "langchaint could not place this exception: boom"
        assert failure.stop_reason is None
        assert failure.attempt_records == ()
        assert failure.model == adapter.model
        assert failure.provider_name == adapter.provider_name

    asyncio.run(scenario())


def test_exception_classified_declared_final_fails_the_item_with_a_record() -> None:
    """A plain exception classified declared_final raises GenerationError, unretried.

    A provider response creates an AttemptRecord with ZERO_USAGE when it reports no Billing.
    """

    async def scenario() -> None:
        """Drive one generate_one whose attempt raises a classify-declared_final exception."""
        adapter = _FakeAdapter(
            scripted_attempts=[ValueError("boom")], classify_result="declared_final"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as declared_final:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        failure = declared_final.value
        assert isinstance(failure.record, ProviderDeclaredFinalErrorRecord)
        assert isinstance(failure.__cause__, ValueError)
        assert failure.error_text == "a final error from the provider: boom"
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None
        assert failure.provider_attempts[0].raw is None
        assert record.usage == ZERO_USAGE
        assert failure.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_a_pause_all_do_not_retry_verdict_stops_the_item_and_pauses_the_rate_limit_quota() -> None:
    """PauseAllDoNotRetry ends the item and pauses SharedBackoff."""

    async def scenario() -> None:
        """Drive one generate_one whose only failure parses to PauseAllDoNotRetry."""
        shared_backoff = _fast_shared_backoff(parse=_parse_pause_all_do_not_retry)
        adapter = _FakeAdapter(
            scripted_attempts=[TransientError("throttled")], classify_result="invalid_request"
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        with pytest.raises(GenerationError):
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 1
        # Only a pausing record moves _pause_until off the sentinel, so this is the pause arriving.
        assert shared_backoff._pause_until != _NEVER

    asyncio.run(scenario())


def test_a_mid_drain_failure_is_retried_and_records_what_the_stream_reported() -> None:
    """A retried mid-drain failure records stream Billing and request ID."""

    async def scenario() -> None:
        """Exhaust a one-attempt budget on a stream that fails after its first item."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        adapter = _FakeAdapter(stream=stream, classify_result="transient")
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            max_attempts=1,
        )
        with pytest.raises(GenerationError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = _settled_attempt_records(exhausted.value.attempt_records)
        assert record.usage == _USAGE_STREAM
        assert record.request_id == "req-fake-stream"
        assert stream.closed

    asyncio.run(scenario())


def test_a_deadline_expiring_mid_drain_reports_the_streams_in_flight_billing() -> None:
    """GenerationError preserves Billing reported before a mid-drain hang."""

    async def scenario() -> None:
        """Hang a stream after its first item and let a short deadline expire."""
        stream = _HangsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=0.05)
        timed_out = raised.value
        (cut_off,) = timed_out.attempt_records
        assert isinstance(cut_off, CutOffAttemptRecord)
        assert cut_off.billing is not None
        assert cut_off.billing.usage == _USAGE_STREAM
        assert timed_out.usage == _USAGE_STREAM
        assert stream.closed

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_settled_attempts_billing_is_counted_once_after_a_later_deadline_cut() -> None:
    """Settled Billing is removed from billing_in_flight."""

    async def scenario() -> None:
        """Fail one attempt mid-drain with billing reported, then hang the second attempt's open."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        adapter = _FakeAdapter(stream=stream, classify_result="transient", hang_from_open=2)
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=0.05)
        timed_out = raised.value
        record, cut_off = timed_out.attempt_records
        assert record.usage == _USAGE_STREAM
        assert isinstance(cut_off, CutOffAttemptRecord)
        assert cut_off.billing is None
        assert timed_out.usage == _USAGE_STREAM

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_reads_the_adapter_stream_request_id() -> None:
    """The generate loop records AdapterStream.request_id()."""

    async def scenario() -> None:
        """Generate over a stream whose assembled response names its own request."""
        stream = _FakeStream()
        stream.raw = _FakeRawResponse(id="fake-final", request_id="req-from-assembled")
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = response.attempt_records
        assert record.request_id == "req-fake-stream"

    asyncio.run(scenario())


def test_a_stream_protocol_error_is_retried_to_exhaustion() -> None:
    """StreamProtocolError retries without classify until exhaustion."""

    async def scenario() -> None:
        """Exhaust a two-attempt budget on a stream that violates the protocol every time."""
        adapter = _FakeAdapter(stream=_ProtocolErrorStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            max_attempts=2,
        )
        with pytest.raises(GenerationError) as exhausted:
            await bound_llm.generate_one([UserMessage(content="hi")])
        assert adapter.bound_adapters[0].open_count == 2
        first, second = _settled_attempt_records(exhausted.value.attempt_records)
        assert "stream ended without a stop event" in str(first.error)
        assert "stream ended without a stop event" in str(second.error)

    asyncio.run(scenario())


def test_a_close_that_raises_does_not_displace_a_generate_success() -> None:
    """An exception from the post-drain close is logged, and the drained success stands."""

    async def scenario() -> None:
        """Generate over a stream whose close raises after a successful drain."""
        stream = _FailingCloseStream()
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        assert response.output == "ab"
        assert stream.closed

    asyncio.run(scenario())


def test_a_mid_drain_exception_nobody_can_place_still_records_the_attempt() -> None:
    """An unplaceable open-stream failure records ZERO_USAGE."""

    async def scenario() -> None:
        """Fail the drain with an exception classify calls unknown_exception."""
        stream = _UnnamedItemErrorStream()
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as unplaceable:
            await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = _settled_attempt_records(unplaceable.value.attempt_records)
        assert record.usage == ZERO_USAGE
        assert record.request_id == "req-fake-stream"
        assert stream.closed

    asyncio.run(scenario())


def test_an_unplaceable_exception_fails_only_its_item() -> None:
    """A classify-unknown_exception item comes back as its GenerationError at its index.

    The sibling succeeds.
    """

    async def scenario() -> None:
        """Serialize a two-item batch (max_concurrent_requests=1) whose first attempt is unplaceable."""
        adapter = _FakeAdapter(
            echo=True, scripted_attempts=[ValueError("boom")], classify_result="unknown_exception"
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, GenerationError)
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(scenario())


def test_a_cancelled_batch_propagates_and_leaves_no_result_behind() -> None:
    """A cancellation from outside generate_many takes the whole batch, settled items included.

    Cancellation prevents the batch from returning its result list.
    """

    async def scenario() -> None:
        """Settle one item, then cancel the batch while the other's open hangs."""
        adapter = _FakeAdapter(hang_from_open=2)
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=1)
        ).bind()
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
def test_bind_builds_a_new_bound_adapter_whether_or_not_the_binding_changed(
    new_system_prompt: str | Unchanged, expected_system_prompt: str, *, binding_is_equal: bool
) -> None:
    """A bind always binds again. The Binding is what tracks whether a field actually changed."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(system_prompt="s")
    replacement_bound = bound_llm.bind(system_prompt=new_system_prompt)
    assert (replacement_bound.binding == bound_llm.binding) is binding_is_equal
    assert replacement_bound.binding.system_prompt == expected_system_prompt
    assert replacement_bound._bound_adapter is not bound_llm._bound_adapter
    assert len(adapter.bound_adapters) == 2


class _Answer(BaseModel):
    """A response_format model for the replacement content-type tests."""

    value: int


class _OtherAnswer(BaseModel):
    """A distinct response format for configuration fingerprint tests."""

    value: str


class _NonRoundTrippableAnswer(BaseModel):
    """A response format used to test JSON round-trip validation."""

    value: float


class _UnschematizableAnswer(BaseModel):
    """A response format whose callable field has no JSON Schema."""

    callback: Callable[[], None]


class _MutableSchemaAnswer(BaseModel):
    """A response format whose field annotation changes in one test."""

    value: int


class _OtherFakeAdapter(_FakeAdapter):
    """Give the same fake behavior a distinct adapter class identity."""


def _capture_tool(name: str = "capture") -> CaptureTool[_Answer]:
    return CaptureTool(
        name=name,
        description="Capture one value.",
        args_model=_Answer,
    )


class _SchemaOnceTool:
    """Expose one schema call so a repeated conversion fails visibly."""

    def __init__(self) -> None:
        self._tool = _capture_tool("schema_once")
        self.schema_calls = 0

    @property
    def name(self) -> str:
        """Return the wrapped tool name."""
        return self._tool.name

    def schema(self) -> ToolSchema:
        """Return the schema once and reject another conversion."""
        self.schema_calls += 1
        if self.schema_calls > 1:
            raise AssertionError("Tool.schema() was called more than once")
        return self._tool.schema()

    async def dispatch(self, call: ToolCall) -> DispatchHandled[_Answer] | DispatchInvalidToolArgs:
        """Dispatch through the wrapped tool."""
        return await self._tool.dispatch(call)


def test_allowed_tools_choice_requires_a_name() -> None:
    """AllowedToolsChoice rejects empty tool_names."""
    with pytest.raises(ValueError, match="must not be empty"):
        _ = AllowedToolsChoice(mode="auto", tool_names=())


def test_allowed_tools_choice_rejects_an_unbound_name_before_adapter_binding() -> None:
    """Binding validates AllowedToolsChoice against its application tool schemas."""
    adapter = _FakeAdapter()
    with pytest.raises(ValueError, match="missing"):
        _ = LLM(adapter).bind(
            tools=[_capture_tool("present")],
            tool_choice=AllowedToolsChoice(mode="auto", tool_names=("missing",)),
        )
    assert adapter.bound_adapters == []


def test_bind_reuses_tool_schemas_when_tools_are_unchanged() -> None:
    """Changing only tool_choice preserves the converted tool definitions by identity."""
    tool = _SchemaOnceTool()
    bound = LLM(_FakeAdapter()).bind(tools=[tool])
    replacement_bound = bound.bind(
        tool_choice=AllowedToolsChoice(mode="required", tool_names=(tool.name,))
    )
    assert tool.schema_calls == 1
    assert replacement_bound.tool_manager is bound.tool_manager
    assert replacement_bound.binding.tool_schemas is bound.binding.tool_schemas


def test_bind_response_format_selects_and_rebuilds_the_adapter_route() -> None:
    """Omission preserves the output type, while a value selects structured or text binding."""
    adapter = _FakeAdapter()
    text = LLM(adapter).bind(system_prompt="s")
    same = text.bind()
    assert_type(same, BoundLLM[str])
    structured = text.bind(response_format=_Answer)
    assert_type(structured, BoundLLM[_Answer])
    assert adapter.structured_bind_count == 1
    assert structured.binding == text.binding
    assert structured._bound_adapter is not text._bound_adapter
    replacement_bound = structured.bind(system_prompt="s2")
    assert_type(replacement_bound, BoundLLM[_Answer])
    assert adapter.structured_bind_count == 2
    assert replacement_bound._bound_adapter is not structured._bound_adapter
    text_again = structured.bind(response_format=None)
    assert_type(text_again, BoundLLM[str])
    assert len(adapter.bound_adapters) == 3
    assert text_again._bound_adapter is not structured._bound_adapter


def test_tools_construct_or_preserve_the_bound_tool_manager() -> None:
    """Tool sequences construct a manager, including empty sequences. Supplied managers pass through."""
    tool = _capture_tool()
    bound = LLM(_FakeAdapter()).bind(tools=[tool])
    assert isinstance(bound.tool_manager, ToolManager)
    assert bound.tool_manager.schemas() == (tool.schema(),)
    empty = LLM(_FakeAdapter()).bind(tools=[])
    assert isinstance(empty.tool_manager, ToolManager)
    assert empty.tool_manager.schemas() == ()
    tool_manager = ToolManager([tool])
    assert LLM(_FakeAdapter()).bind(tools=tool_manager).tool_manager is tool_manager


def test_duplicate_tool_names_fail_before_initial_and_replacement_bind_reach_adapter() -> None:
    """Duplicate tool names fail before either adapter-binding route."""
    adapter = _FakeAdapter()
    duplicate = _capture_tool()
    with pytest.raises(ValueError, match="duplicate tool name"):
        _ = LLM(adapter).bind(tools=[duplicate, duplicate])
    assert adapter.bound_adapters == []
    bound = LLM(adapter).bind()
    with pytest.raises(ValueError, match="duplicate tool name"):
        _ = bound.bind(tools=[duplicate, duplicate])
    assert len(adapter.bound_adapters) == 1


def test_bind_tools_replace_the_manager_and_none_removes_it() -> None:
    """`BoundLLM.bind(tools=[...])` replaces `ToolManager`. `tools=None` removes it."""
    original = _capture_tool("original")
    replacement = _capture_tool("replacement")
    bound = LLM(_FakeAdapter()).bind(tools=[original])
    replacement_bound = bound.bind(tools=[replacement])
    assert replacement_bound.tool_manager is not bound.tool_manager
    assert replacement_bound.tool_manager.schemas() == (replacement.schema(),)
    assert replacement_bound.bind(tools=None).tool_manager is None
    tool_manager = ToolManager([replacement])
    assert bound.bind(tools=tool_manager).tool_manager is tool_manager


def _pin_initial_and_replacement_bind_types(llm: LLM, tool_manager: ToolManager) -> None:
    text = llm.bind()
    assert_type(text, BoundLLM[str, None])
    text_with_constructed_tools = llm.bind(tools=[_capture_tool()])
    assert_type(text_with_constructed_tools, BoundLLM[str, ToolManager])
    text_with_tools = llm.bind(tools=tool_manager)
    assert_type(text_with_tools, BoundLLM[str, ToolManager])
    structured = llm.bind(response_format=_Answer)
    assert_type(structured, BoundLLM[_Answer, None])
    structured_with_tools = llm.bind(response_format=_Answer, tools=tool_manager)
    assert_type(structured_with_tools, BoundLLM[_Answer, ToolManager])
    structured_with_constructed_tools = llm.bind(
        response_format=_Answer,
        tools=[_capture_tool()],
    )
    assert_type(structured_with_constructed_tools, BoundLLM[_Answer, ToolManager])
    provider_tool = ({"type": "web_search"},)
    text_with_provider_tool = llm.bind(
        provider_executed_tools=provider_tool,
    )
    assert_type(text_with_provider_tool, BoundLLM[str, None])
    assert_type(
        structured.bind(provider_executed_tools=provider_tool),
        BoundLLM[_Answer, None],
    )

    # BoundLLM[X] is BoundLLM[X, None]: the PEP 696 default keeps the common annotation short.
    assert_type(structured, BoundLLM[_Answer])

    assert_type(text_with_tools.tool_manager, ToolManager)
    assert_type(text.tool_manager, None)

    replacement_with_tool_manager = structured.bind(tools=tool_manager)
    assert_type(replacement_with_tool_manager, BoundLLM[_Answer, ToolManager])
    assert_type(structured.bind(tools=[_capture_tool()]), BoundLLM[_Answer, ToolManager])
    assert_type(structured_with_tools.bind(tools=None), BoundLLM[_Answer, None])
    assert_type(text_with_tools.bind(response_format=_Answer), BoundLLM[_Answer, ToolManager])
    assert_type(structured_with_tools.bind(response_format=None), BoundLLM[str, ToolManager])
    assert_type(structured_with_tools.bind(system_prompt="s"), BoundLLM[_Answer, ToolManager])


async def _pin_request_method_return_types(llm: LLM, tool_manager: ToolManager) -> None:
    """Pin the return types the ToolManagerT overloads produce, which is what the parameter is for.

    Never called: pyrefly checks this body, and the assertions are about types alone.
    """
    structured_with_tools = llm.bind(response_format=_Answer, tools=tool_manager)
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
    assert_type(
        await structured_with_tools.generate_many_records(
            ["hi"], resume_path=Path("records.json")
        ),
        list[CallResultRecord[_Answer]],
    )
    structured = llm.bind(response_format=_Answer)
    assert_type(await structured.generate_one("hi"), Response[_Answer])
    assert_type(
        await structured.generate_many_records(["hi"], resume_path=Path("records.json")),
        list[ResponseRecord[_Answer] | GenerationErrorRecord],
    )
    text_with_tools = llm.bind(tools=tool_manager)
    assert_type(await text_with_tools.generate_one("hi"), Response[str])
    assert_type(
        await text_with_tools.generate_many_records(["hi"], resume_path=Path("records.json")),
        list[ResponseRecord[str] | GenerationErrorRecord],
    )
    assert_type(text_with_tools.stream_one("hi"), StreamHandle[str])


async def _pin_generic_request_method_return_types[
    OutputT: str | BaseModel,
    ToolManagerT: ToolManager | None,
](bound_llm: BoundLLM[OutputT, ToolManagerT]) -> None:
    assert_type(
        await bound_llm.generate_many(["hi"]),
        list[Response[OutputT] | GenerationError] | list[CallResult[OutputT]],
    )
    assert_type(
        await bound_llm.generate_many_records(["hi"], resume_path=Path("records.json")),
        list[ResponseRecord[OutputT] | GenerationErrorRecord] | list[CallResultRecord[OutputT]],
    )


def test_response_format_is_public_state_that_replacement_bind_carries() -> None:
    """response_format is inspectable state that replacement bindings carry and switch."""
    adapter = _FakeAdapter()
    assert LLM(adapter).bind().response_format is None
    structured = LLM(adapter).bind(response_format=_Answer)
    assert structured.response_format is _Answer
    assert structured.bind(system_prompt="s2").response_format is _Answer
    assert structured.bind(response_format=None).response_format is None


def test_splits_tool_call_turns_only_on_the_structured_tool_bound_binding() -> None:
    """The split reads the binding: response_format and tool_manager must both be present."""
    llm = LLM(_FakeAdapter())
    tool_manager = ToolManager([])
    assert not llm.bind()._splits_tool_call_turns
    assert not llm.bind(tools=tool_manager)._splits_tool_call_turns
    assert not llm.bind(response_format=_Answer)._splits_tool_call_turns
    assert llm.bind(response_format=_Answer, tools=tool_manager)._splits_tool_call_turns


class _ScriptedStructuredBoundAdapter[OutputT](BoundAdapter[OutputT]):
    """A structured bound adapter handing every request one scripted outcome.

    Structured generation tests use this adapter.
    """

    def __init__(self, outcome: ResponseOutcome[OutputT]) -> None:
        """Store the outcome and start `open_count` at zero."""
        self._outcome = outcome
        self.stream = _FakeStream()
        self.open_count = 0

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Bill every response the fixed _USAGE."""
        return stated_provider_billing(_USAGE)

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Report a fixed identity."""
        return ResponseIdentity(
            model_served="fake-model-served",
            response_id="structured-response",
            request_id=request_id,
        )

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[OutputT]:
        """Return the scripted outcome, whichever raw the request path produced."""
        return self._outcome

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams:
        """Carry the messages into the request."""
        return _FakeRequest(messages=tuple(messages))

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Hand back the stored fake stream. interpret ignores the raw it assembles."""
        self.open_count += 1
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

    Replacing _bound_adapter preserves response_format, tool_manager, and retry behavior.
    """
    bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
        response_format=_Answer, tools=ToolManager([])
    )
    bound_llm._bound_adapter = _ScriptedStructuredBoundAdapter(outcome)
    return bound_llm


def test_structured_tool_bound_generate_one_returns_the_tool_call_turn_variant() -> None:
    """A structured tool turn returns ToolCallTurn with output=None."""

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


def test_generate_many_records_restores_the_structured_output_type(tmp_path: Path) -> None:
    """A resumed response record reconstructs the bound `response_format` type."""

    async def scenario() -> None:
        """Generate one structured record, then restore it without another request."""
        answer = _Answer(value=7)
        outcome: AdapterResult[_Answer | None] = AdapterResult(
            output=answer,
            assistant_message=AssistantMessage(turn=(TextPart(text=answer.model_dump_json()),)),
            stop_reason="end_turn",
        )
        bound_llm = _structured_tool_bound_llm(outcome)
        structured_adapter = bound_llm._bound_adapter
        assert isinstance(structured_adapter, _ScriptedStructuredBoundAdapter)
        resume_path = tmp_path / "records.json"
        generated = await bound_llm.generate_many_records(["hi"], resume_path=resume_path)
        restored = await bound_llm.generate_many_records(["hi"], resume_path=resume_path)
        generated_record = generated[0]
        restored_record = restored[0]
        assert isinstance(generated_record, ResponseRecord)
        assert isinstance(restored_record, ResponseRecord)
        assert generated_record.output == answer
        assert restored_record.output == answer
        assert isinstance(restored_record.output, _Answer)
        assert structured_adapter.open_count == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_records_validates_serialized_bytes_before_replacing_the_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A result that cannot round-trip through JSON leaves the prepared resume document unchanged."""
    resume_path = tmp_path / "records.json"
    original_replace = Path.replace
    replaced_document_json: list[bytes] = []

    def observed_replace(path: Path, target: Path) -> Path:
        replaced_path = original_replace(path, target)
        if target == resume_path:
            replaced_document_json.append(target.read_bytes())
        return replaced_path

    monkeypatch.setattr(Path, "replace", observed_replace)

    async def scenario() -> None:
        answer = _NonRoundTrippableAnswer(value=float("nan"))
        outcome: AdapterResult[_NonRoundTrippableAnswer] = AdapterResult(
            output=answer,
            assistant_message=AssistantMessage(turn=(TextPart(text=answer.model_dump_json()),)),
            stop_reason="end_turn",
        )
        bound_llm = LLM(_FakeAdapter()).bind(response_format=_NonRoundTrippableAnswer)
        bound_llm._bound_adapter = _ScriptedStructuredBoundAdapter(outcome)
        with pytest.raises(ValueError, match="Input should be a valid number"):
            await bound_llm.generate_many_records(["hi"], resume_path=resume_path)
        assert len(replaced_document_json) == 1
        assert resume_path.read_bytes() == replaced_document_json[0]

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_records_restores_a_structured_tool_call_turn(tmp_path: Path) -> None:
    """A resumed tool-call record retains `output=None` and its tool calls."""

    async def scenario() -> None:
        """Generate one tool-call record, then restore it without another request."""
        bound_llm = _structured_tool_bound_llm(_STRUCTURED_TOOL_CALL_TURN)
        structured_adapter = bound_llm._bound_adapter
        assert isinstance(structured_adapter, _ScriptedStructuredBoundAdapter)
        resume_path = tmp_path / "records.json"
        generated = await bound_llm.generate_many_records(["hi"], resume_path=resume_path)
        restored = await bound_llm.generate_many_records(["hi"], resume_path=resume_path)
        generated_record = generated[0]
        restored_record = restored[0]
        assert isinstance(generated_record, ToolCallTurnRecord)
        assert isinstance(restored_record, ToolCallTurnRecord)
        assert restored_record.output is None
        assert restored_record.tool_calls == (_FAKE_TOOL_CALL,)
        assert structured_adapter.open_count == 1

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


def test_automatic_cache_breakpoints_participates_in_binding_equality() -> None:
    """`automatic_cache_breakpoints` participates in `Binding` equality."""
    adapter = _FakeAdapter()
    bound_llm = LLM(adapter).bind(automatic_cache_breakpoints=True)
    flipped = bound_llm.bind(automatic_cache_breakpoints=False)
    assert flipped.binding != bound_llm.binding
    assert flipped._bound_adapter is not bound_llm._bound_adapter
    unchanged = bound_llm.bind(automatic_cache_breakpoints=True)
    assert unchanged.binding == bound_llm.binding
    assert unchanged._bound_adapter is not bound_llm._bound_adapter


@pytest.mark.parametrize("default_value", [False, True])
def test_bind_resolves_automatic_cache_breakpoints_default(*, default_value: bool) -> None:
    """None resolves before `Binding` reaches the adapter."""
    llm = LLM(_FakeAdapter(automatic_cache_breakpoints_default=default_value))
    override_value = not default_value
    assert llm.bind().binding.automatic_cache_breakpoints is default_value
    assert (
        llm.bind(automatic_cache_breakpoints=override_value).binding.automatic_cache_breakpoints
        is override_value
    )


@pytest.mark.parametrize("default_value", [False, True])
def test_bind_omission_preserves_and_none_resets_automatic_cache_breakpoints(
    *, default_value: bool
) -> None:
    """Omission preserves the value. None resolves `automatic_cache_breakpoints_default`."""
    override_value = not default_value
    bound = LLM(_FakeAdapter(automatic_cache_breakpoints_default=default_value)).bind(
        automatic_cache_breakpoints=override_value
    )
    assert bound.bind().binding.automatic_cache_breakpoints is override_value
    assert (
        bound.bind(automatic_cache_breakpoints=None).binding.automatic_cache_breakpoints
        is default_value
    )


def test_initial_and_replacement_bind_carry_extra_body_by_reference() -> None:
    """LLM.bind stores extra_body unchanged. BoundLLM.bind keeps, replaces, or clears it."""
    adapter = _FakeAdapter()
    extra_body = {"safety_identifier": "user-7"}
    bound_llm = LLM(adapter).bind(extra_body=extra_body)
    assert bound_llm.binding.extra_body is extra_body
    assert bound_llm.bind(system_prompt="s").binding.extra_body is extra_body
    replacement = {"safety_identifier": "user-8"}
    assert bound_llm.bind(extra_body=replacement).binding.extra_body is replacement
    assert bound_llm.bind(extra_body=None).binding.extra_body is None


def test_config_fingerprint_has_a_fixed_digest() -> None:
    """Pin the canonical encoding and SHA-256 result for the default fake binding."""
    fingerprint = LLM(_FakeAdapter()).bind().config_fingerprint()
    assert fingerprint == "sha256:5ed4b01ba8c91b5caf398531228f76e30777147deade680d3b1d9379dad2031b"


def test_config_fingerprint_ignores_mapping_insertion_order() -> None:
    """Mappings with equal entries produce one configuration fingerprint."""
    first = LLM(_FakeAdapter()).bind(extra_body={"alpha": 1, "beta": 2})
    second = LLM(_FakeAdapter()).bind(extra_body={"beta": 2, "alpha": 1})
    assert first.config_fingerprint() == second.config_fingerprint()


def test_config_fingerprint_preserves_sequence_order_and_container_types() -> None:
    """Sequence order and stored container types distinguish configurations."""
    ordered = LLM(_FakeAdapter()).bind(extra_body={"values": [1, 2]})
    reversed_order = LLM(_FakeAdapter()).bind(extra_body={"values": [2, 1]})
    tuple_value = LLM(_FakeAdapter()).bind(extra_body={"values": (1, 2)})
    set_value = LLM(_FakeAdapter()).bind(extra_body={"values": {1, 2}})
    frozen_set_value = LLM(_FakeAdapter()).bind(extra_body={"values": frozenset({1, 2})})
    assert ordered.config_fingerprint() != reversed_order.config_fingerprint()
    assert ordered.config_fingerprint() != tuple_value.config_fingerprint()
    assert set_value.config_fingerprint() != frozen_set_value.config_fingerprint()


def test_config_fingerprint_tracks_ignored_binding_values() -> None:
    """A changed stored automatic-cache setting changes configuration identity."""
    enabled = LLM(_FakeAdapter()).bind(automatic_cache_breakpoints=True)
    disabled = LLM(_FakeAdapter()).bind(automatic_cache_breakpoints=False)
    assert enabled.config_fingerprint() != disabled.config_fingerprint()


def test_config_fingerprint_treats_an_omitted_resolved_default_as_its_explicit_value() -> None:
    """Binding stores the resolved automatic-cache value instead of the bind call form."""
    omitted = LLM(_FakeAdapter(automatic_cache_breakpoints_default=True)).bind()
    explicit = LLM(_FakeAdapter(automatic_cache_breakpoints_default=True)).bind(
        automatic_cache_breakpoints=True
    )
    assert omitted.config_fingerprint() == explicit.config_fingerprint()


def test_config_fingerprint_tracks_adapter_model_provider_class_and_configuration() -> None:
    """Each stored adapter identity category participates in configuration identity."""
    baseline_adapter = _FakeAdapter()
    baseline = LLM(baseline_adapter).bind()
    changed_model_adapter = _FakeAdapter()
    changed_model_adapter.model = "other-model"
    changed_provider_adapter = _FakeAdapter()
    changed_provider_adapter.provider_name = "other-provider"
    changed_config_adapter = _FakeAdapter()
    changed_config_adapter.automatic_cache_breakpoints_default = True
    assert baseline.config_fingerprint() != LLM(changed_model_adapter).bind().config_fingerprint()
    assert (
        baseline.config_fingerprint() != LLM(changed_provider_adapter).bind().config_fingerprint()
    )
    assert baseline.config_fingerprint() != LLM(_OtherFakeAdapter()).bind().config_fingerprint()
    assert baseline.config_fingerprint() != LLM(changed_config_adapter).bind().config_fingerprint()


def test_config_fingerprint_reads_binding_values_and_captures_adapter_values() -> None:
    """Binding references stay current, while adapter configuration is captured during binding."""
    extra_body: dict[str, object] = {"value": 1}
    adapter = _FakeAdapter()
    bound = LLM(adapter).bind(extra_body=extra_body)
    initial = bound.config_fingerprint()
    extra_body["value"] = 2
    after_binding_mutation = bound.config_fingerprint()
    adapter.model = "changed-model"
    adapter.provider_name = "changed-provider"
    adapter.automatic_cache_breakpoints_default = True
    assert initial != after_binding_mutation
    assert after_binding_mutation == bound.config_fingerprint()
    assert after_binding_mutation != bound.bind().config_fingerprint()


def test_config_fingerprint_tracks_response_format_and_bind() -> None:
    """Structured response configuration and replacement fields participate."""
    text = LLM(_FakeAdapter()).bind()
    answer = LLM(_FakeAdapter()).bind(response_format=_Answer)
    other_answer = LLM(_FakeAdapter()).bind(response_format=_OtherAnswer)
    replacement_bound = answer.bind(system_prompt="changed")
    assert text.config_fingerprint() != answer.config_fingerprint()
    assert answer.config_fingerprint() != other_answer.config_fingerprint()
    assert answer.config_fingerprint() != replacement_bound.config_fingerprint()


def test_config_fingerprint_excludes_retry_admission_and_tool_functions() -> None:
    """Execution policy and application functions do not enter configuration identity."""

    async def first_function(_args: _Answer) -> str:
        return "first"

    async def second_function(_args: _Answer) -> str:
        return "second"

    first_tool = PydanticTool(
        name="answer",
        description="Answer.",
        args_model=_Answer,
        function=first_function,
    )
    second_tool = PydanticTool(
        name="answer",
        description="Answer.",
        args_model=_Answer,
        function=second_function,
    )
    first = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
        tools=[first_tool], max_attempts=1
    )
    second = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind(
        tools=[second_tool], max_attempts=9
    )
    assert first.config_fingerprint() == second.config_fingerprint()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (object(), "binding.extra_body['value'] has unsupported type builtins.object"),
        (float("inf"), "binding.extra_body['value'] contains a non-finite float"),
    ],
)
def test_config_fingerprint_rejects_values_without_a_deterministic_encoding(
    value: object, message: str
) -> None:
    """The TypeError names the unsupported value's path."""
    bound = LLM(_FakeAdapter()).bind(extra_body={"value": value})
    with pytest.raises(TypeError) as raised:
        _ = bound.config_fingerprint()
    assert str(raised.value) == message


def test_config_fingerprint_rejects_a_container_cycle() -> None:
    """A cycle raises TypeError with the recursive value's path."""
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    bound = LLM(_FakeAdapter()).bind(extra_body=cyclic)
    message = "binding.extra_body['self'] contains a cycle"
    with pytest.raises(TypeError) as raised:
        _ = bound.config_fingerprint()
    assert str(raised.value) == message


def test_config_fingerprint_encodes_an_unpaired_surrogate() -> None:
    """ASCII JSON escaping gives every Python string a UTF-8 fingerprint input."""
    fingerprint = LLM(_FakeAdapter()).bind(extra_body={"value": "\ud800"}).config_fingerprint()
    assert fingerprint.startswith("sha256:")


def test_config_fingerprint_normalizes_a_response_schema_failure() -> None:
    """Pydantic schema failures become the documented TypeError at the response-format path."""
    bound = LLM(_FakeAdapter()).bind(response_format=_UnschematizableAnswer)
    with pytest.raises(TypeError) as raised:
        _ = bound.config_fingerprint()
    assert str(raised.value).startswith(
        "response_format.model_json_schema() cannot be serialized deterministically:"
    )


def test_config_fingerprint_captures_the_response_schema_during_binding() -> None:
    """A bound fingerprint retains the schema that existed during binding."""
    first = LLM(_FakeAdapter()).bind(response_format=_MutableSchemaAnswer)
    field = _MutableSchemaAnswer.model_fields["value"]
    field.annotation = str
    _ = _MutableSchemaAnswer.model_rebuild(force=True)
    try:
        second = LLM(_FakeAdapter()).bind(response_format=_MutableSchemaAnswer)
        assert first.config_fingerprint() != second.config_fingerprint()
    finally:
        field.annotation = int
        _ = _MutableSchemaAnswer.model_rebuild(force=True)


def test_bind_keeps_replaces_and_removes_provider_executed_tools() -> None:
    """Preserve omitted provider_executed_tools and replace or clear specified values."""
    first_tool = {"type": "web_search"}
    second_tool = {"type": "file_search", "vector_store_ids": ["vs_1"]}
    bound = LLM(_FakeAdapter()).bind(
        provider_executed_tools=(first_tool,),
        max_attempts=5,
    )
    assert bound.binding.provider_executed_tools == (first_tool,)
    assert bound.bind(system_prompt="s").binding.provider_executed_tools == (first_tool,)
    replaced = bound.bind(provider_executed_tools=(second_tool,))
    assert replaced.binding.provider_executed_tools == (second_tool,)
    assert replaced.max_attempts == 5
    assert bound.bind(provider_executed_tools=()).binding.provider_executed_tools == ()


def test_generate_many_aligns_results_with_inputs() -> None:
    """Result i belongs to generation_inputs[i], preserving input order."""

    async def scenario() -> None:
        """Run a two-item batch whose fake echoes each item's first turn."""
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind()
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

        One permit runs the items in submission order.
        The scripted failure lands on the first item.
        The remaining items succeed at their input indexes.
        """
        adapter = _FakeAdapter(echo=True, scripted_attempts=[TransientError("x")])
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            max_attempts=1,
        )
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
            [UserMessage(content="c")],
        ])
        first, second, third = results
        assert isinstance(first, GenerationError)
        assert isinstance(second, Response)
        assert second.output == "b"
        assert isinstance(third, Response)
        assert third.output == "c"

    asyncio.run(scenario())


def test_generate_many_returns_a_refusal_at_its_index() -> None:
    """An item whose attempt reports Refusal comes back as the GenerationError at its index, siblings succeed."""

    async def scenario() -> None:
        """Serialize a two-item batch (max_concurrent_requests=1) whose first attempt reports Refusal."""
        adapter = _FakeAdapter(
            echo=True,
            scripted_attempts=[_billed(_REFUSAL)],
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, GenerationError)
        assert first.stop_reason == "refusal"
        assert first.usage.cost_in_usd == 0.25
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(scenario())


def test_generate_many_records_persists_and_reuses_records_in_input_order(tmp_path: Path) -> None:
    """A complete batch without `sample_ids` persists records that the next binding reuses."""

    async def scenario() -> None:
        """Generate two records, inspect the JSON document, then resume without another request."""
        resume_path = tmp_path / "records.json"
        first_adapter = _FakeAdapter(echo=True)
        first_bound = LLM(first_adapter).bind()
        first_records = await first_bound.generate_many_records(
            ["a", "b"], resume_path=resume_path
        )
        assert _record_outputs(first_records) == ["a", "b"]
        assert first_adapter.bound_adapters[0].open_count == 2
        resume_data = _resume_json_object(resume_path)
        assert resume_data["format_version"] == 1
        assert resume_data["identity_mode"] == "position"
        resume_items = TypeAdapter(list[dict[str, object]]).validate_python(resume_data["items"])
        assert len(resume_items) == 2

        resumed_adapter = _FakeAdapter(echo=True)
        resumed_bound = LLM(resumed_adapter).bind()
        resumed_records = await resumed_bound.generate_many_records(
            ["a", "b"], resume_path=resume_path
        )
        assert _record_outputs(resumed_records) == ["a", "b"]
        assert resumed_adapter.bound_adapters[0].open_count == 0

    asyncio.run(scenario())


def test_generate_many_records_uses_one_background_thread_for_resume_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Assert `Path.resolve` and `Path.replace` use one background thread."""
    resume_path = tmp_path / "records.json"
    original_resolve = Path.resolve
    original_replace = Path.replace
    first_result_replace_started = threading.Event()
    release_first_result_replace = threading.Event()
    resolve_thread_ids: list[int] = []
    replace_thread_ids: list[int] = []

    def observed_resolve(path: Path, *, strict: bool = False) -> Path:
        """Record resolution for the test's `resume_path`."""
        if path == resume_path:
            resolve_thread_ids.append(threading.get_ident())
        return original_resolve(path, strict=strict)

    def observed_replace(path: Path, target: Path) -> Path:
        """Block the first result replacement and record every document replacement."""
        if target == resume_path:
            replace_thread_ids.append(threading.get_ident())
            if len(replace_thread_ids) == 2:
                first_result_replace_started.set()
                assert release_first_result_replace.wait(timeout=5.0)
        return original_replace(path, target)

    monkeypatch.setattr(Path, "resolve", observed_resolve)
    monkeypatch.setattr(Path, "replace", observed_replace)

    async def scenario() -> None:
        """Let a second request start while the first result replacement remains blocked."""
        event_loop_thread_id = threading.get_ident()
        adapter = _FakeAdapter(echo=True)
        bound = LLM(adapter).bind()
        task = asyncio.create_task(
            bound.generate_many_records(["a", "b"], resume_path=resume_path)
        )
        assert await asyncio.to_thread(first_result_replace_started.wait, 5.0)
        while adapter.bound_adapters[0].open_count < 2:
            await asyncio.sleep(0)
        assert not task.done()
        release_first_result_replace.set()
        records = await task
        assert _record_outputs(records) == ["a", "b"]
        assert len(resolve_thread_ids) == 1
        assert len(replace_thread_ids) == 3
        assert set(resolve_thread_ids) == set(replace_thread_ids)
        assert resolve_thread_ids[0] != event_loop_thread_id

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_records_retries_a_retries_exhausted_record(tmp_path: Path) -> None:
    """A saved RetriesExhaustedErrorRecord remains pending until a later call succeeds."""

    async def scenario() -> None:
        """Exhaust one request, resume to success, then reuse that success."""
        adapter = _FakeAdapter(echo=True, scripted_attempts=[TransientError("try again")])
        bound = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(max_attempts=1)
        resume_path = tmp_path / "records.json"
        first = await bound.generate_many_records(["a"], resume_path=resume_path)
        assert isinstance(first[0], RetriesExhaustedErrorRecord)
        second = await bound.generate_many_records(["a"], resume_path=resume_path)
        assert _record_outputs(second) == ["a"]
        third = await bound.generate_many_records(["a"], resume_path=resume_path)
        assert _record_outputs(third) == ["a"]
        assert adapter.bound_adapters[0].open_count == 2

    asyncio.run(scenario())


def test_generate_many_records_retries_a_timed_out_record(tmp_path: Path) -> None:
    """A later call with more working time regenerates a saved `TimedOutErrorRecord`."""

    async def scenario() -> None:
        """Time out one request, then resume the same input without a working-time limit."""
        resume_path = tmp_path / "records.json"
        timed_out_adapter = _FakeAdapter(echo=True, hang_from_open=1)
        timed_out_bound = LLM(
            timed_out_adapter,
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        timed_out = await timed_out_bound.generate_many_records(
            ["a"],
            resume_path=resume_path,
            max_working_seconds_per_item=0.01,
        )
        assert isinstance(timed_out[0], TimedOutErrorRecord)

        resumed_adapter = _FakeAdapter(echo=True)
        resumed_bound = LLM(resumed_adapter).bind()
        resumed = await resumed_bound.generate_many_records(["a"], resume_path=resume_path)
        assert _record_outputs(resumed) == ["a"]
        assert resumed_adapter.bound_adapters[0].open_count == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_records_reuses_a_terminal_error_record(tmp_path: Path) -> None:
    """A saved terminal error record prevents another provider request."""

    async def scenario() -> None:
        """Save one refusal and restore the same record on the next call."""
        adapter = _FakeAdapter(
            echo=True,
            scripted_attempts=[_billed(_REFUSAL)],
        )
        bound = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        resume_path = tmp_path / "records.json"
        generated = await bound.generate_many_records(["a"], resume_path=resume_path)
        restored = await bound.generate_many_records(["a"], resume_path=resume_path)
        assert generated[0].kind == "refusal_error"
        assert restored[0].kind == "refusal_error"
        record_adapter = TypeAdapter(list[ResponseRecord[str] | GenerationErrorRecord])
        assert record_adapter.dump_json(restored) == record_adapter.dump_json(generated)
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_generate_many_records_without_sample_ids_replaces_changed_input_lists(
    tmp_path: Path,
) -> None:
    """`sample_ids=None` replaces the batch after an input is reordered, added, or deleted."""

    async def scenario() -> None:
        """Change the ordered input list three ways and count the replacement requests."""
        adapter = _FakeAdapter(echo=True)
        bound = LLM(adapter).bind()
        resume_path = tmp_path / "records.json"
        first = await bound.generate_many_records(["a", "b"], resume_path=resume_path)
        reordered = await bound.generate_many_records(["b", "a"], resume_path=resume_path)
        added = await bound.generate_many_records(["b", "a", "c"], resume_path=resume_path)
        deleted = await bound.generate_many_records(["b"], resume_path=resume_path)
        assert _record_outputs(first) == ["a", "b"]
        assert _record_outputs(reordered) == ["b", "a"]
        assert _record_outputs(added) == ["b", "a", "c"]
        assert _record_outputs(deleted) == ["b"]
        assert adapter.bound_adapters[0].open_count == 8

    asyncio.run(scenario())


def test_generate_many_records_replaces_a_changed_binding_or_identity_mode(tmp_path: Path) -> None:
    """A `binding_fingerprint` or `identity_mode` change replaces the saved batch."""

    async def scenario() -> None:
        """Change the binding, provide `sample_ids`, then omit `sample_ids` again."""
        adapter = _FakeAdapter(echo=True)
        position_bound = LLM(adapter).bind()
        resume_path = tmp_path / "records.json"
        await position_bound.generate_many_records(["a"], resume_path=resume_path)
        changed_bound = position_bound.bind(system_prompt="changed")
        await changed_bound.generate_many_records(["a"], resume_path=resume_path)
        await changed_bound.generate_many_records(
            ["a"], resume_path=resume_path, sample_ids=["sample-a"]
        )
        await changed_bound.generate_many_records(["a"], resume_path=resume_path)
        assert adapter.bound_adapters[0].open_count == 1
        assert adapter.bound_adapters[1].open_count == 3
        assert _resume_json_object(resume_path)["identity_mode"] == "position"

    asyncio.run(scenario())


def test_generate_many_records_sample_ids_support_batch_edits_and_equal_inputs(
    tmp_path: Path,
) -> None:
    """`sample_ids` preserves unchanged records across reorder, add, and delete operations."""

    async def scenario() -> None:
        """Track equal inputs, delete one `sample_ids` entry, add it again, and change another input."""
        adapter = _FakeAdapter(echo=True)
        bound = LLM(adapter).bind()
        resume_path = tmp_path / "records.json"
        first = await bound.generate_many_records(
            ["same", "same"],
            resume_path=resume_path,
            sample_ids=["left", "right"],
        )
        reordered = await bound.generate_many_records(
            ["same", "same"],
            resume_path=resume_path,
            sample_ids=["right", "left"],
        )
        added = await bound.generate_many_records(
            ["same", "third", "same"],
            resume_path=resume_path,
            sample_ids=["right", "third", "left"],
        )
        deleted = await bound.generate_many_records(
            ["third", "same"],
            resume_path=resume_path,
            sample_ids=["third", "left"],
        )
        restored = await bound.generate_many_records(
            ["third", "same", "same"],
            resume_path=resume_path,
            sample_ids=["third", "left", "right"],
        )
        changed = await bound.generate_many_records(
            ["third", "changed", "same"],
            resume_path=resume_path,
            sample_ids=["third", "left", "right"],
        )
        assert _record_outputs(first) == ["same", "same"]
        assert _record_outputs(reordered) == ["same", "same"]
        assert _record_outputs(added) == ["same", "third", "same"]
        assert _record_outputs(deleted) == ["third", "same"]
        assert _record_outputs(restored) == ["third", "same", "same"]
        assert _record_outputs(changed) == ["third", "changed", "same"]
        assert adapter.bound_adapters[0].open_count == 5
        items = TypeAdapter(list[dict[str, object]]).validate_python(
            _resume_json_object(resume_path)["items"]
        )
        assert [item["sample_id"] for item in items] == ["third", "left", "right"]

    asyncio.run(scenario())


def test_generate_many_records_rejects_invalid_sample_ids_before_requests(tmp_path: Path) -> None:
    """Duplicate or misaligned `sample_ids` raises before the file or provider changes."""

    async def scenario() -> None:
        """Pass each invalid `sample_ids` sequence to a fresh path."""
        adapter = _FakeAdapter(echo=True)
        bound = LLM(adapter).bind()
        resume_path = tmp_path / "records.json"
        with pytest.raises(ValueError, match="sample_ids"):
            await bound.generate_many_records(
                ["a", "b"], resume_path=resume_path, sample_ids=["same", "same"]
            )
        with pytest.raises(ValueError, match="sample_ids"):
            await bound.generate_many_records(
                ["a", "b"], resume_path=resume_path, sample_ids=["only-one"]
            )
        assert not resume_path.exists()
        assert adapter.bound_adapters[0].open_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "resume_text",
    [
        "not json\n",
        '{"format_version": 999, "binding_fingerprint": "x", "identity_mode": "position", "items": []}\n',
        '{"unrelated": true}\n',
    ],
    ids=["malformed", "unsupported-version", "unrelated-json"],
)
def test_generate_many_records_preserves_an_unrecognized_file(
    tmp_path: Path, resume_text: str
) -> None:
    """An unrecognized existing file raises before changing the file or sending a request."""

    async def scenario() -> None:
        """Attempt to resume from the supplied text and verify its bytes remain unchanged."""
        resume_path = tmp_path / "records.json"
        _ = resume_path.write_text(resume_text)
        adapter = _FakeAdapter(echo=True)
        bound = LLM(adapter).bind()
        with pytest.raises(ValueError, match="resume"):
            await bound.generate_many_records(["a"], resume_path=resume_path)
        assert resume_path.read_text() == resume_text
        assert adapter.bound_adapters[0].open_count == 0

    asyncio.run(scenario())


def test_generate_many_records_persists_a_settled_item_before_batch_cancellation(
    tmp_path: Path,
) -> None:
    """A settled item survives cancellation while a later item is still running."""

    async def scenario() -> None:
        """Cancel after the second request starts, then resume only the missing second item."""
        resume_path = tmp_path / "records.json"
        first_adapter = _FakeAdapter(echo=True, hang_from_open=2)
        first_bound = LLM(
            first_adapter,
            shared_backoff=_fast_shared_backoff(max_concurrent_requests=1),
        ).bind()
        task = asyncio.create_task(
            first_bound.generate_many_records(["a", "b"], resume_path=resume_path)
        )
        while first_adapter.bound_adapters[0].open_count < 2:
            await asyncio.sleep(0)
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        resumed_adapter = _FakeAdapter(echo=True)
        resumed_bound = LLM(resumed_adapter).bind()
        resumed = await resumed_bound.generate_many_records(["a", "b"], resume_path=resume_path)
        assert _record_outputs(resumed) == ["a", "b"]
        assert resumed_adapter.bound_adapters[0].open_count == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_records_rejects_concurrent_use_of_one_path(tmp_path: Path) -> None:
    """A second active call with the same resume path raises before sending a request."""

    async def scenario() -> None:
        """Hold the first request open while the second call uses the same path."""
        adapter = _FakeAdapter(echo=True, hang_from_open=1)
        bound = LLM(adapter).bind()
        resume_path = tmp_path / "records.json"
        first_call = asyncio.create_task(
            bound.generate_many_records(["a"], resume_path=resume_path)
        )
        while adapter.bound_adapters[0].open_count < 1:
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="active generate_many_records"):
            await bound.generate_many_records(["b"], resume_path=resume_path)
        assert adapter.bound_adapters[0].open_count == 1
        _ = first_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_call

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_invalid_request_fails_only_its_item() -> None:
    """A rejected item comes back as its GenerationError at its index. The sibling still succeeds.

    Nothing a single item does reaches a sibling, so the batch returns one outcome per generation_input.
    """

    async def scenario() -> None:
        """Serialize a two-item batch (max_concurrent_requests=1) whose first build_request refuses."""
        adapter = _FakeAdapter(
            echo=True, invalid_requests=[InvalidRequest(reason="misconfigured")]
        )
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        first, second = results
        assert isinstance(first, GenerationError)
        assert isinstance(first.record, InvalidRequestErrorRecord)
        assert first.record.reason == "misconfigured"
        assert isinstance(second, Response)
        assert second.output == "b"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_many_warm_cache_runs_the_first_item_alone_then_the_rest_together() -> None:
    """warm_cache completes generation_inputs[0] before any sibling starts. The rest run at normal concurrency."""

    async def scenario() -> None:
        """Run controlled three-item batches and compare peak_in_flight.

        The warmed first open completes before blocked siblings overlap.
        The unwarmed opens remain blocked until all three overlap.
        """
        generation_inputs = [[UserMessage(content=str(index))] for index in range(3)]
        warmed_barrier = asyncio.Barrier(3)
        warmed_adapter = _FakeAdapter(
            echo=True,
            open_barrier=warmed_barrier,
            open_barrier_from_call=2,
        )
        warmed_bound_llm = LLM(
            warmed_adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=8)
        ).bind()
        warmed_task = asyncio.create_task(
            warmed_bound_llm.generate_many(generation_inputs, warm_cache=True)
        )
        await asyncio.wait_for(warmed_barrier.wait(), timeout=1.0)
        assert warmed_adapter.bound_adapters[0].peak_in_flight == 2
        warmed = await warmed_task
        assert _batch_outputs(warmed) == ["0", "1", "2"]
        control_barrier = asyncio.Barrier(4)
        control_adapter = _FakeAdapter(
            echo=True,
            open_barrier=control_barrier,
        )
        control_bound_llm = LLM(
            control_adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=8)
        ).bind()
        control_task = asyncio.create_task(control_bound_llm.generate_many(generation_inputs))
        await asyncio.wait_for(control_barrier.wait(), timeout=1.0)
        assert control_adapter.bound_adapters[0].peak_in_flight == 3
        control = await control_task
        assert _batch_outputs(control) == ["0", "1", "2"]

    asyncio.run(scenario())


def test_generate_many_warm_cache_first_failure_still_admits_the_rest() -> None:
    """A first item ending in a GenerationError stays at its index and the siblings still run."""

    async def scenario() -> None:
        """Fail the deterministic first attempt under a one-attempt budget. The other two succeed."""
        adapter = _FakeAdapter(echo=True, scripted_attempts=[TransientError("x")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            max_attempts=1,
        )
        results = await bound_llm.generate_many(
            [[UserMessage(content="a")], [UserMessage(content="b")], [UserMessage(content="c")]],
            warm_cache=True,
        )
        first, second, third = results
        assert isinstance(first, GenerationError)
        assert isinstance(second, Response)
        assert second.output == "b"
        assert isinstance(third, Response)
        assert third.output == "c"

    asyncio.run(scenario())


def test_generate_many_warm_cache_empty_batch_returns_empty() -> None:
    """An empty batch under warm_cache returns [] instead of indexing a first item."""

    async def scenario() -> None:
        """Run the empty batch."""
        bound_llm = LLM(_FakeAdapter()).bind()
        assert await bound_llm.generate_many([], warm_cache=True) == []

    asyncio.run(scenario())


class _ClassifyRaisesAdapter(_FakeAdapter):
    """Raise a scripted defect from classify."""

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Raise instead of classifying.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError("classify defect")


def test_a_defect_becomes_one_items_failure_and_leaves_the_batch_complete() -> None:
    """An Exception from langchaint's own machinery fails its item and no sibling.

    generate_many returns the defect as one item without discarding settled siblings.
    """

    async def scenario() -> None:
        """Raise past the retry loop on the one item whose attempt fails, and let the other succeed."""
        adapter = _ClassifyRaisesAdapter(scripted_attempts=[ValueError("defect")], echo=True)
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        results = await bound_llm.generate_many([
            [UserMessage(content="a")],
            [UserMessage(content="b")],
        ])
        failures = [result for result in results if isinstance(result, GenerationError)]
        responses = [result for result in results if isinstance(result, Response)]
        assert len(failures) == 1
        assert len(responses) == 1
        failure = failures[0]
        assert isinstance(failure.record, EscapedExceptionErrorRecord)
        assert isinstance(failure.__cause__, RuntimeError)
        assert "classify defect" in failure.error_text
        assert failure.request is None
        # The attempt was in flight when the defect escaped, so no attempt is settled.
        assert failure.call.attempt_records == ()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_generate_one_raises_a_defect_as_a_generation_error() -> None:
    """`generate_one` raises a defect as `GenerationError`."""

    async def scenario() -> None:
        """Fail the one attempt and let classify raise past the retry loop."""
        adapter = _ClassifyRaisesAdapter(scripted_attempts=[ValueError("defect")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        assert str(raised.value) == "an exception escaped langchaint: classify defect"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_parse_contract_violation_surfaces_as_langchaints_defect_not_a_provider_outcome() -> (
    None
):
    """generate_one raises GenerationError whose error is the ParserContractError.

    A parse contract violation must bypass the transport-failure path.
    `classify` would produce `UnknownExceptionErrorRecord` on that path.
    """

    async def scenario() -> None:
        """Fail the one attempt with a TransientError whose parse raises."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("boom")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff(parse=_parse_raises)).bind()
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        assert isinstance(raised.value.record, EscapedExceptionErrorRecord)
        assert isinstance(raised.value.__cause__, ParserContractError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_parse_contract_violation_on_a_stream_open_reaches_the_caller() -> None:
    """Entering stream_one raises the ParserContractError itself. No stream-path frame wraps it."""

    async def scenario() -> None:
        """Fail the one open with a TransientError whose parse raises."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("boom")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff(parse=_parse_raises)).bind()
        with pytest.raises(ParserContractError):
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


class _ClassifyRaisesOverStagedResponseAdapter(_ClassifyRaisesAdapter):
    """A _ClassifyRaisesAdapter whose bound adapters price a response and then raise reading it."""

    _bound_adapter_class = _InterpretRaisesBoundAdapter


def test_a_defect_over_a_staged_response_keeps_the_attempt_and_its_billing() -> None:
    """A defect after response arrival preserves the attempt and Billing."""

    async def scenario() -> None:
        """Stage the response, raise from interpret, then raise again from classify placing it."""
        adapter = _ClassifyRaisesOverStagedResponseAdapter()
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="a")])
        (record,) = _settled_attempt_records(raised.value.attempt_records)
        assert record.usage == _USAGE
        assert raised.value.usage == _USAGE
        assert raised.value.attempts == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_bare_str_is_shorthand_for_one_user_message() -> None:
    """A bare str reaches the adapter as a Sequence[Message] of one UserMessage."""

    async def scenario() -> None:
        """Drive each generate method with a bare str against the echo fake.

        The echo fake returns content from a UserMessage with str content.
        An echoed value proves that coercion built a UserMessage.
        """
        adapter = _FakeAdapter(echo=True)
        bound_llm = LLM(adapter).bind()
        response = await bound_llm.generate_one("hi")
        assert response.output == "hi"
        results = await bound_llm.generate_many(["a", [UserMessage(content="b")]])
        assert _batch_outputs(results) == ["a", "b"]

    asyncio.run(scenario())


def test_stream_one_accepts_a_bare_str() -> None:
    """stream_one coerces a bare str to a Sequence[Message] of one UserMessage."""

    async def scenario() -> None:
        """Build a handle from a bare str and check the stored messages."""
        bound_llm = LLM(_FakeAdapter()).bind()
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
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
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
        bound_llm = LLM(_FakeAdapter(hang_from_open=1), shared_backoff=shared_backoff).bind()

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns. The wait_for below cancels this."""
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        async with shared_backoff.admitted(budget=1.0):
            pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_inside_the_block_sets_its_abandoned() -> None:
    """Cancellation records the stream's available in-flight state."""

    async def scenario() -> None:
        """Time out a consumer suspended on a hanging stream, then read the handle."""
        adapter = _FakeAdapter(stream=_HangingStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def drain() -> None:
            """Enter and iterate into the hang. The wait_for below cancels this."""
            async with handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)
        abandoned = handle.abandoned
        assert abandoned is not None
        (cut_off,) = abandoned.attempt_records
        assert isinstance(cut_off, CutOffAttemptRecord)
        assert cut_off.billing is None
        assert abandoned.usage == ZERO_USAGE
        assert abandoned.model == adapter.model

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_cancelled_stream_reports_what_it_billed_before_the_cancellation() -> None:
    """Abandoned stream Usage includes reported in-flight Billing."""

    async def scenario() -> None:
        """Time out a consumer on a hanging stream that reports a running spend."""
        stream = _HangingStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def drain() -> None:
            """Enter and iterate into the hang. The wait_for below cancels this."""
            async with handle:
                async for _item in handle:
                    pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(), timeout=0.05)
        abandoned = handle.abandoned
        assert abandoned is not None
        (cut_off,) = abandoned.attempt_records
        assert isinstance(cut_off, CutOffAttemptRecord)
        assert cut_off.billing is not None
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
        ).bind()
        # Leaving after one item keeps admission held until __aexit__.
        # close must return that admission.
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
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def enter_and_leave() -> None:
            """Enter the handle whose open never returns. The wait_for below cancels this."""
            async with handle:
                pass

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(enter_and_leave(), timeout=0.02)
        assert handle.abandoned is not None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_completed_or_left_early_sets_no_abandoned() -> None:
    """final() completing, or a consumer leaving the block voluntarily, leaves abandoned None.

    Voluntary exit after `Response` creates no `AbandonedCallErrorRecord`.
    """

    async def scenario() -> None:
        """Consume one stream to final(), leave a second before its first item."""
        bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind()
        completed = bound_llm.stream_one([UserMessage(content="hi")])
        async with completed:
            await completed.final()
        second_adapter = _FakeAdapter(stream=_HangingStream())
        second_bound_llm = LLM(second_adapter, shared_backoff=_fast_shared_backoff()).bind()
        left_early = second_bound_llm.stream_one([UserMessage(content="hi")])
        async with left_early:
            pass
        assert completed.abandoned is None
        assert left_early.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_final_raised_sets_no_abandoned() -> None:
    """A cancellation after final() raised its GenerationError leaves abandoned None.

    `RefusalErrorRecord` concludes the call without `AbandonedCallErrorRecord`.
    """

    async def scenario() -> None:
        """Absorb a refusal from final() inside the block, then hang into the caller's deadline."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_REFUSAL))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() report Refusal, then sleep. The wait_for below cancels this inside the block."""
            async with handle:
                with pytest.raises(GenerationError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_a_protocol_error_sets_its_abandoned() -> None:
    """A StreamProtocolError accounts for nothing, so a cancellation after one still records the call.

    The record accounts for the opened stream.
    StreamProtocolError carries no model, attempt records, or usage.
    """

    async def scenario() -> None:
        """Absorb the protocol error inside the block, then hang into the caller's deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_ProtocolErrorStream()), shared_backoff=_fast_shared_backoff()
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() hit the protocol error, then sleep. The wait_for below cancels this."""
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
        ).bind()
        before_monotonic_seconds = time.monotonic()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as raised:
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
    """Yield one item before raising a classified TransientError."""

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
        ).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as raised:
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
    """A BaseException during close preserves the abandoned record."""

    async def scenario() -> None:
        """Cancel the block, then let the close raise on the way out."""
        bound_llm = LLM(
            _FakeAdapter(stream=_CloseRaisesBaseExceptionStream(), classify_result="transient"),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Hang inside the block. The wait_for cancels this."""
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
    """A dropped stream records its running Billing."""

    async def scenario() -> None:
        """Let the iteration fail after one item, then read the error's usage."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = usage_reported
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result="transient"),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as raised:
                async for _item in handle:
                    pass
        (record,) = _settled_attempt_records(raised.value.attempt_records)
        assert record.usage == expected_usage
        assert raised.value.usage == expected_usage

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_a_mid_stream_failure_sets_no_abandoned() -> None:
    """Cancellation after GenerationError creates no abandoned error."""

    async def scenario() -> None:
        """Absorb the mid-stream failure inside the block, then hang into the caller's deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream(), classify_result="transient"),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let the iteration fail after one item, then sleep. The wait_for cancels this."""
            async with handle:
                with pytest.raises(GenerationError) as raised:
                    async for _item in handle:
                        pass
                (record,) = _settled_attempt_records(raised.value.attempt_records)
                assert isinstance(record.error, TransientErrorRecord)
                assert "open stream failed during iteration" in str(record.error)
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_cancelled_after_a_drain_failure_sets_no_abandoned() -> None:
    """A GenerationError raised while draining carries the call, so a cancellation sets nothing.

    A concluded stream failure creates no `AbandonedCallErrorRecord`.
    """

    async def scenario() -> None:
        """Absorb an unplaceable item failure inside the block, then hang into the deadline."""
        bound_llm = LLM(
            _FakeAdapter(stream=_UnnamedItemErrorStream(), classify_result="unknown_exception"),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() hit the unplaceable failure, then sleep. The wait_for cancels this."""
            async with handle:
                with pytest.raises(GenerationError):
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
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
        bound_llm = LLM(_FakeAdapter()).bind()
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
        assert record.elapsed_seconds >= 0.0

    asyncio.run(scenario())


def test_stream_final_refusal_raises_without_retry() -> None:
    """A structured refusal detected in the stream's final() surfaces as a GenerationError.

    final() records the one 200 that produced no output and raises the GenerationError carrying that record.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports Refusal, then read the raised GenerationError."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_REFUSAL))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as refusal:
                await handle.final()
        failure = refusal.value
        assert failure.attempts == 1
        assert failure.stop_reason == "refusal"
        assert failure.usage.cost_in_usd == 0.25
        assert failure.usage.output_tokens == _USAGE.output_tokens
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None
        assert record.usage.cost_in_usd == 0.25

    asyncio.run(scenario())


def test_stream_final_unfinished_turn_raises_carrying_the_adapter_s_reason() -> None:
    """Stream final preserves UnfinishedTurn.reason."""

    async def scenario() -> None:
        """Drain a stream whose final() reports UnfinishedTurn, then read the raised error."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_UNFINISHED_TURN))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as unfinished_turn:
                await handle.final()
        failure = unfinished_turn.value
        assert failure.attempts == 1
        assert "pause_turn" in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None

    asyncio.run(scenario())


def test_stream_final_schema_violation_raises_carrying_the_rejection() -> None:
    """A SchemaViolation from the stream's final() fails the call, carrying pydantic's rejection.

    Stream SchemaViolation preserves ValidationError outside error_text.
    """

    async def scenario() -> None:
        """Drain a stream whose final() reports SchemaViolation, then read the raised error."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_SCHEMA_VIOLATION))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as schema_violation:
                await handle.final()
        failure = schema_violation.value
        assert failure.attempts == 1
        assert isinstance(failure.record, SchemaViolationErrorRecord)
        assert failure.record.validation_error_json == _VALIDATION_ERROR_JSON
        assert "SENTINEL" not in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("stream", "expected_error"),
    [
        (_FakeStream(outcome=_MAX_COMPLETION_TOKENS_EXCEEDED), GenerationError),
        (_FakeStream(outcome=_EMPTY_TURN), GenerationError),
        (_FakeStream(outcome=_CONTEXT_WINDOW_EXCEEDED), GenerationError),
    ],
    ids=["max_completion_tokens_exceeded", "empty_turn", "context_window_exceeded"],
)
def test_stream_final_reports_each_no_output_outcome_as_its_own_error(
    stream: _FakeStream, expected_error: type[GenerationError]
) -> None:
    """Final maps each terminal outcome to its GenerationError without retrying."""

    async def scenario() -> None:
        """Drain a stream whose final() reports the outcome, then read the raised error."""
        adapter = _FakeAdapter(stream=stream)
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(expected_error) as caught:
                await handle.final()
        failure = caught.value
        assert failure.attempts == 1
        assert failure.usage.cost_in_usd == 0.25
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None

    asyncio.run(scenario())


def test_stream_final_provider_failed_transiently_fails_the_item_with_retry_unavailable() -> None:
    """A transient terminal outcome raises GenerationError without reopening."""

    async def scenario() -> None:
        """Drain a stream whose final() reports the failure, then read the raised error."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_PROVIDER_FAILED_TRANSIENTLY))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as retry_unavailable:
                await handle.final()
        failure = retry_unavailable.value
        assert failure.attempts == 1
        assert _PROVIDER_FAILURE_REASON in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert str(record.error) == _PROVIDER_FAILURE_REASON
        assert record.assistant_message == _REJECTED_TURN
        assert failure.provider_attempts[0].raw is not None
        assert isinstance(failure.__cause__, TransientError)
        assert str(failure.__cause__) == str(record.error)
        assert adapter.bound_adapters[0].open_count == 1

    asyncio.run(scenario())


def test_stream_final_provider_failed_terminally_raises_carrying_the_providers_reason() -> None:
    """The outcome fails the call once, and the provider's own text is the error's message."""

    async def scenario() -> None:
        """Drain a stream whose final() reports the terminal failure, then read the raised error."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_PROVIDER_FAILED_TERMINALLY))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as provider_failure:
                await handle.final()
        failure = provider_failure.value
        assert failure.attempts == 1
        assert _PROVIDER_FAILURE_REASON in failure.error_text
        assert failure.usage.cost_in_usd == 0.25
        (record,) = _settled_attempt_records(failure.attempt_records)
        assert record.error is None

    asyncio.run(scenario())


def test_a_stream_cancelled_after_absorbing_a_provider_failure_sets_no_abandoned() -> None:
    """A cancellation after final() raised its GenerationError sets nothing.

    `RetryUnavailableErrorRecord` concludes the call without `AbandonedCallErrorRecord`.
    """

    async def scenario() -> None:
        """Absorb the failure from final() inside the block, then hang into the caller's deadline."""
        adapter = _FakeAdapter(stream=_FakeStream(outcome=_PROVIDER_FAILED_TRANSIENTLY))
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")])

        async def consume() -> None:
            """Let final() report the failure, then sleep. The wait_for below cancels in the block."""
            async with handle:
                with pytest.raises(GenerationError):
                    await handle.final()
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume(), timeout=0.05)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_that_drops_mid_turn_records_the_id_its_open_response_carried() -> None:
    """A dropped connection raises an error naming no request, and the open stream still has the header.

    Stream failures preserve the request ID for provider support.
    """

    async def scenario() -> None:
        """Read a failure raised before the first item."""
        adapter = _FakeAdapter(stream=_FailsBeforeFirstItemStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as raised:
                await handle.final()
        (record,) = _settled_attempt_records(raised.value.attempt_records)
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as raised:
                await anext(handle)
        (record,) = _settled_attempt_records(raised.value.attempt_records)
        assert record.request_id == "req-from-items-error"

    asyncio.run(scenario())


def test_streaming_reads_the_adapter_stream_request_id() -> None:
    """StreamHandle.final records AdapterStream.request_id()."""

    async def scenario() -> None:
        """Drain a stream whose assembled response names its own request."""
        stream = _FakeStream()
        stream.raw = _FakeRawResponse(id="fake-final", request_id="req-from-assembled")
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.request_id == "req-fake-stream"

    asyncio.run(scenario())


def test_stream_retry_populates_attempt_records() -> None:
    """A retried open failure precedes a successful stream attempt."""

    async def scenario() -> None:
        """Open a stream whose first open_stream call fails, then drain it."""
        adapter = _FakeAdapter(
            scripted_attempts=[_RequestIdError("connection reset", "req-from-open-failure")],
            classify_result="transient",
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
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
        assert failed.started_after_seconds + failed.elapsed_seconds <= (
            succeeded.started_after_seconds
        )

    asyncio.run(scenario())


def test_a_stream_stamps_its_first_item_and_not_a_later_one() -> None:
    """first_item_at_monotonic_seconds records the first item."""

    async def scenario() -> None:
        """Drain a stream that waits between its items and read the record it froze."""
        gap_seconds = _SlowAfterFirstItemStream.gap_seconds
        adapter = _FakeAdapter(stream=_SlowAfterFirstItemStream())
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            _ = [item async for item in handle]
            response = await handle.final()
        (record,) = response.attempt_records
        assert record.seconds_to_first_item is not None
        seconds_to_first_item = record.seconds_to_first_item
        assert 0.0 <= seconds_to_first_item < gap_seconds
        assert record.elapsed_seconds > gap_seconds

    asyncio.run(scenario())


def test_a_non_stream_attempt_leaves_its_first_item_stamp_unset() -> None:
    """generate_one yields no items, so the column it feeds stays null for every non-stream call."""

    async def scenario() -> None:
        """Run one generate_one and read the record it froze."""
        bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        (record,) = response.attempt_records
        assert record.seconds_to_first_item is None

    asyncio.run(scenario())


def test_stream_open_classified_invalid_request_carries_the_prior_attempts_records() -> None:
    """A rejected open raises GenerationError from the entry with the prior transient record."""

    async def scenario() -> None:
        """Enter a handle whose first open fails transiently and whose second is rejected."""
        adapter = _FakeAdapter(
            scripted_attempts=[TransientError("connection reset"), ValueError("boom")],
            classify_result="invalid_request",
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as rejected:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert isinstance(rejected.value.__cause__, ValueError)
        assert isinstance(rejected.value.record, InvalidRequestErrorRecord)
        assert rejected.value.record.reason == "the provider rejected the request: boom"
        transient_record, rejected_record = _settled_attempt_records(
            rejected.value.attempt_records
        )
        assert str(transient_record.error) == "connection reset"
        assert rejected_record.error is None
        assert rejected_record.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_a_stream_whose_build_request_refuses_fails_the_item_with_nothing_opened() -> None:
    """build_request refusing the messages fails the item before any stream is opened.

    GenerationError carries model and attempt_records without a request.
    """

    async def scenario() -> None:
        """Enter a handle whose build_request refuses."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="nope")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as rejected:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 0
        assert rejected.value.request is None
        assert isinstance(rejected.value.record, InvalidRequestErrorRecord)
        assert rejected.value.record.reason == "nope"
        assert rejected.value.model == adapter.model
        assert rejected.value.attempt_records == ()

    asyncio.run(scenario())


def test_stream_open_classified_unknown_exception_raises_the_items_failure() -> None:
    """An open failure classified unknown_exception raises GenerationError from the entry, unretried."""

    async def scenario() -> None:
        """Enter a handle whose open raises a classify-unknown_exception exception."""
        adapter = _FakeAdapter(
            scripted_attempts=[ValueError("boom")], classify_result="unknown_exception"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as unplaceable:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(unplaceable.value.record, UnknownExceptionErrorRecord)
        assert isinstance(unplaceable.value.__cause__, ValueError)
        assert unplaceable.value.error_text == "langchaint could not place this exception: boom"
        assert unplaceable.value.attempt_records == ()

    asyncio.run(scenario())


def test_stream_open_classified_declared_final_raises_the_items_failure() -> None:
    """An open failure the provider declared final raises GenerationError, unretried.

    The open reached the provider, which answered, so that attempt has a record.
    """

    async def scenario() -> None:
        """Enter a handle whose open raises a classify-declared_final exception."""
        adapter = _FakeAdapter(
            scripted_attempts=[ValueError("boom")], classify_result="declared_final"
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as declared_final:
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        assert isinstance(declared_final.value.record, ProviderDeclaredFinalErrorRecord)
        assert isinstance(declared_final.value.__cause__, ValueError)
        assert declared_final.value.error_text == "a final error from the provider: boom"
        (record,) = _settled_attempt_records(declared_final.value.attempt_records)
        assert record.error is None
        assert record.usage == ZERO_USAGE

    asyncio.run(scenario())


def test_terminal_pause_stops_stream_open_and_pauses_rate_limit_quota() -> None:
    """A terminal open failure raises without reopening the stream."""

    async def scenario() -> None:
        """Enter a handle whose open failure parses to PauseAllDoNotRetry."""
        shared_backoff = _fast_shared_backoff(parse=_parse_pause_all_do_not_retry)
        adapter = _FakeAdapter(
            scripted_attempts=[TransientError("throttled")], classify_result="invalid_request"
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        with pytest.raises(GenerationError):
            async with bound_llm.stream_one([UserMessage(content="hi")]):
                pass
        assert adapter.bound_adapters[0].open_count == 1
        # Only a pausing record moves _pause_until off the sentinel, so this is the pause arriving.
        assert shared_backoff._pause_until != _NEVER

    asyncio.run(scenario())


def test_a_pause_all_do_not_retry_verdict_ends_an_open_stream_with_the_items_failure() -> None:
    """A terminal mid-stream failure raises GenerationError."""

    async def scenario() -> None:
        """Read a first-item failure that parses to PauseAllDoNotRetry."""
        adapter = _FakeAdapter(
            stream=_FailsBeforeFirstItemStream(), classify_result="invalid_request"
        )
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(parse=_parse_pause_all_do_not_retry)
        ).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as declared_final:
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as raised:
                await anext(handle)
        assert adapter.bound_adapters[0].open_count == 1
        assert raised.value.attempts == 1
        (record,) = _settled_attempt_records(raised.value.attempt_records)
        assert str(record.error) == "dropped before the first item"
        assert record.seconds_to_first_item is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("classify_result", "expected_error"),
    [
        ("invalid_request", GenerationError),
        ("declared_final", GenerationError),
        ("unknown_exception", GenerationError),
    ],
    ids=["invalid_request", "declared_final", "unknown_exception"],
)
def test_a_terminal_mid_stream_error_records_what_the_stream_reported(
    classify_result: ErrorClassification, expected_error: type[GenerationError]
) -> None:
    """Terminal stream errors preserve reported Usage for each classification."""

    async def scenario() -> None:
        """Let the iteration hit the failure after one item, then read the record."""
        stream = _FailsAfterFirstItemStream()
        stream._usage_reported = _USAGE_STREAM
        bound_llm = LLM(
            _FakeAdapter(stream=stream, classify_result=classify_result),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(expected_error) as caught:
                async for _item in handle:
                    pass
        (record,) = _settled_attempt_records(caught.value.attempt_records)
        assert record.usage == _USAGE_STREAM
        assert caught.value.usage == _USAGE_STREAM

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_an_unplaceable_mid_stream_error_records_the_attempt_with_nothing_reported() -> None:
    """An open stream failure records the attempt without reported Billing."""

    async def scenario() -> None:
        """Let the iteration hit the failure after one item on a stream reporting no counters."""
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream(), classify_result="unknown_exception"),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(GenerationError) as caught:
                async for _item in handle:
                    pass
        (record,) = _settled_attempt_records(caught.value.attempt_records)
        assert record.billing is None
        assert record.usage == ZERO_USAGE

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_open_exhaustion_raises_retries_exhausted() -> None:
    """Opens that keep failing past the budget raise GenerationError with the table fields set."""

    async def scenario() -> None:
        """Open a stream under a two-attempt budget whose every open_stream fails transiently."""
        adapter = _FakeAdapter(scripted_attempts=[TransientError("e1"), TransientError("e2")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind(
            max_attempts=2,
        )
        with pytest.raises(GenerationError) as exhausted:
            async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
                await handle.final()
        failure = exhausted.value
        assert failure.attempts == 2
        assert isinstance(failure.record, RetriesExhaustedErrorRecord)
        assert [str(error) for error in failure.record.errors_from_attempts] == ["e1", "e2"]
        assert failure.model == "fake-model"
        assert failure.provider_name == "fake"
        assert failure.elapsed_seconds >= 0.0

    asyncio.run(scenario())


def test_stream_record_and_elapsed_end_at_exhaustion_not_at_final() -> None:
    """Idle time between draining the stream and calling final() lands in neither measurement."""
    idle_seconds = 0.02

    async def scenario() -> None:
        """Drain the stream, idle, then call final()."""
        bound_llm = LLM(_FakeAdapter()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            async for _item in handle:
                pass
            await asyncio.sleep(idle_seconds)
            response = await handle.final()
        assert response.elapsed_seconds < idle_seconds

    asyncio.run(scenario())


def test_stream_final_is_idempotent() -> None:
    """A second final() returns the same cached Response object."""

    async def scenario() -> None:
        """Call final() twice on one drained stream."""
        bound_llm = LLM(_FakeAdapter()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            first: Response[str] = await handle.final()
            second: Response[str] = await handle.final()
        assert first is second

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("stream", "expected_error"),
    [
        (_FakeStream(outcome=_REFUSAL), GenerationError),
        (_FakeStream(outcome=_PROVIDER_FAILED_TRANSIENTLY), GenerationError),
        (_ProtocolErrorStream(), StreamProtocolError),
    ],
    ids=["refusal", "provider_failed_transiently", "protocol_error"],
)
def test_stream_final_replays_every_error_that_concluded_the_call(
    stream: _FakeStream, expected_error: type[Exception]
) -> None:
    """Repeated final raises the stored error without another attempt."""

    async def scenario() -> None:
        """Call final() twice on a stream whose call cannot end in a Response."""
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
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

    The stored exception prevents a second adapter-stream assembly.
    """

    async def scenario() -> None:
        """Call final() twice on a stream whose own final() raises."""
        stream = _FinalRaisesStream()
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=_fast_shared_backoff()).bind()
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
        bound_llm = LLM(_FakeAdapter()).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            collected_items = [item async for item in handle]
        assert collected_items == ["ok", _FAKE_TOOL_CALL]

    asyncio.run(scenario())


def test_stream_closes_on_context_exit() -> None:
    """Leaving the async with block closes the underlying adapter stream."""

    async def scenario() -> None:
        """Open the stream, consume one item, then leave the context."""
        stream = _FakeStream()
        bound_llm = LLM(_FakeAdapter(stream=stream)).bind()
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
            scripted_attempts=[
                TransientError("slow down", retry_after_seconds=retry_after_seconds)
            ]
        )
        shared_backoff = _fast_shared_backoff(longest_wait_seconds=1.0)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
            max_attempts=2,
        )
        started_at = time.monotonic()
        response = await bound_llm.generate_one([UserMessage(content="hi")])
        elapsed_seconds = time.monotonic() - started_at
        assert response.output == "ok"
        assert response.attempts == 2
        assert elapsed_seconds >= retry_after_seconds
        # Only a PauseAll record moves _pause_until off the sentinel. A RetryThisOne must not.
        assert shared_backoff._pause_until == _NEVER

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_max_concurrent_requests_bounds_batch_concurrency() -> None:
    """A five-item batch under max_concurrent_requests=2 never overlaps more than two requests."""

    async def scenario() -> None:
        """Run the batch on a slow fake and read the recorded peak."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.01)
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=2)
        ).bind()
        generation_inputs = [[UserMessage(content=str(index))] for index in range(5)]
        results = await bound_llm.generate_many(generation_inputs)
        assert _batch_outputs(results) == ["0", "1", "2", "3", "4"]
        assert adapter.bound_adapters[0].peak_in_flight == 2

    asyncio.run(scenario())


def test_backoff_sleep_does_not_hold_the_permit() -> None:
    """Private backoff releases a request permit.

    retry_after_seconds outlasts the second request.
    The second request finishes while the first request waits.
    """

    async def scenario() -> None:
        """Interleave a retrying item with a clean one under one permit."""
        retry_after_seconds = 0.2
        adapter = _FakeAdapter(
            scripted_attempts=[TransientError("boom", retry_after_seconds=retry_after_seconds)]
        )
        shared_backoff = _fast_shared_backoff(
            max_concurrent_requests=1,
            longest_wait_seconds=1.0,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind(
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
        bound_llm = LLM(_FakeAdapter(stream=stream), shared_backoff=shared_backoff).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            with pytest.raises(StreamProtocolError):
                await handle.final()
            assert stream.closed is True
            async with shared_backoff.admitted(budget=1.0):
                pass

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_stream_releases_its_permit_when_exhausted() -> None:
    """Stream exhaustion releases its request permit before block exit."""

    async def scenario() -> None:
        """Drain one stream under max_concurrent_requests=1, then re-admit inside the still-open block."""
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(_FakeAdapter(), shared_backoff=shared_backoff).bind()
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
            scripted_attempts=[
                TransientError("rate limited", retry_after_seconds=0.001, is_rate_limit=True)
            ]
        )
        shared_backoff = _fast_shared_backoff()
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        async with bound_llm.stream_one([UserMessage(content="hi")]) as handle:
            response = await handle.final()
        # Only a PauseAll record moves _pause_until off the sentinel, so this is the open's failure arriving.
        assert shared_backoff._pause_until != _NEVER
        assert response.output == "ok"
        assert response.attempts == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_bind_coerces_system_prompt_parts_to_a_tuple() -> None:
    """A list of system parts freezes to a tuple on the binding. A str passes through."""
    parts = [TextPart(text="stable", cache_breakpoint=True), TextPart(text="context")]
    bound_llm = LLM(_FakeAdapter()).bind(system_prompt=parts)
    assert bound_llm.binding.system_prompt == tuple(parts)
    assert isinstance(bound_llm.binding.system_prompt, tuple)


def test_bind_rejects_an_empty_system_prompt_parts_sequence() -> None:
    """Empty parts are a configuration error. None is the way to bind no system prompt."""
    with pytest.raises(ValueError, match="empty"):
        _ = LLM(_FakeAdapter()).bind(system_prompt=[])


@pytest.mark.parametrize("settled_attempts", [1, 0])
def test_a_deadline_expiring_mid_request_counts_the_request_it_cut_off(
    settled_attempts: int,
) -> None:
    """GenerationError counts the in-flight request after settled attempts."""

    async def scenario() -> None:
        """Fail settled_attempts requests transiently, then hang the next past a deadline."""
        adapter = _FakeAdapter(
            scripted_attempts=[TransientError("settled attempt")] * settled_attempts,
            # 1-based, so the request that hangs is the one after the settled ones.
            hang_from_open=settled_attempts + 1,
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="hi")], timeout_seconds=0.05)
        timed_out = raised.value
        assert timed_out.attempts == settled_attempts + 1
        assert len(timed_out.attempt_records) == settled_attempts + 1
        assert isinstance(timed_out.attempt_records[-1], CutOffAttemptRecord)
        assert timed_out.attempt_records[-1].billing is None
        assert timed_out.usage == ZERO_USAGE
        assert str(timed_out) == "the call timed out before it produced a result"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_deadline_expiring_before_admission_reports_no_attempts() -> None:
    """Admission timeout reports attempts=0."""

    async def scenario() -> None:
        """Hold the only permit, then run a second call under a deadline it cannot outlast."""
        adapter = _FakeAdapter(hang_from_open=1)
        bound_llm = LLM(
            adapter, shared_backoff=_fast_shared_backoff(max_concurrent_requests=1)
        ).bind()
        holder = asyncio.create_task(bound_llm.generate_one([UserMessage(content="held")]))
        await asyncio.sleep(0.02)
        with pytest.raises(GenerationError) as raised:
            await bound_llm.generate_one([UserMessage(content="queued")], timeout_seconds=0.05)
        timed_out = raised.value
        assert timed_out.attempts == 0
        assert timed_out.attempt_records == ()
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
        ).bind()
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
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="fast")], [UserMessage(content="slow")]],
            max_working_seconds_per_item=0.05,
        )
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "fast"
        assert isinstance(second, GenerationError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_batch_item_spends_no_budget_waiting_for_a_permit() -> None:
    """Permit waits do not consume max_working_seconds_per_item."""

    async def scenario() -> None:
        """Queue four items behind one permit, each item working well inside its budget."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.15)
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content=str(index))] for index in range(4)],
            max_working_seconds_per_item=0.30,
        )
        assert _batch_outputs(results) == ["0", "1", "2", "3"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_batch_item_spends_its_budget_once_it_is_admitted() -> None:
    """Work after admission consumes max_working_seconds_per_item."""

    async def scenario() -> None:
        """Let the first item answer, then hang the second one after it is admitted."""
        adapter = _FakeAdapter(echo=True, open_seconds=0.15, hang_from_open=2)
        shared_backoff = _fast_shared_backoff(max_concurrent_requests=1)
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="answered")], [UserMessage(content="hangs")]],
            max_working_seconds_per_item=0.30,
        )
        first, second = results
        assert isinstance(first, Response)
        assert first.output == "answered"
        assert isinstance(second, GenerationError)

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
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
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
            scripted_attempts=[
                TransientError("slow down", retry_after_seconds=0.3, is_rate_limit=True)
            ],
        )
        shared_backoff = _fast_shared_backoff(
            max_concurrent_requests=None,
            longest_wait_seconds=0.5,
        )
        bound_llm = LLM(adapter, shared_backoff=shared_backoff).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="paused")]], max_working_seconds_per_item=0.15
        )
        assert _batch_outputs(results) == ["paused"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_batch_item_banks_what_is_left_of_its_budget_across_a_retry() -> None:
    """Retries share one max_working_seconds_per_item budget."""

    async def scenario() -> None:
        """Fail the first attempt transiently, then time out inside the retry."""
        adapter = _FakeAdapter(
            echo=True,
            open_seconds=0.12,
            scripted_attempts=[TransientError("try again")],
        )
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        results = await bound_llm.generate_many(
            [[UserMessage(content="retries")]], max_working_seconds_per_item=0.20
        )
        assert isinstance(results[0], GenerationError)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))


def test_a_stream_deadline_raises_and_leaves_abandoned_unset() -> None:
    """One cut-off call gets one account: the raise, not the handle field as well.

    Setting both would put the same call in the archive twice.
    """

    async def scenario() -> None:
        """Enter a stream whose open never returns, under a deadline."""
        bound_llm = LLM(
            _FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        with pytest.raises(GenerationError):
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
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        seen: list[StreamItem] = []

        async def drain() -> None:
            """Consume the stream, recording each item as it arrives.

            The deadline expires mid-iteration, so each item has to reach seen on its own.
            A comprehension would build a list the expiry discards.
            """
            async with handle:
                async for item in handle:
                    seen.append(item)  # noqa: PERF401

        with pytest.raises(GenerationError):
            await drain()
        assert seen, "the items delivered before the deadline stay delivered"
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_deadline_stops_at_the_calls_conclusion() -> None:
    """Final closes the call deadline before caller work."""

    async def scenario() -> None:
        """Take the Response, then outlive the deadline inside the block."""
        bound_llm = LLM(_FakeAdapter(), shared_backoff=_fast_shared_backoff()).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        async with handle:
            response = await handle.final()
            await asyncio.sleep(0.1)
        assert response.output == "ok"
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_stream_deadline_stops_at_a_conclusion_an_item_pull_raised() -> None:
    """A terminal item pull closes the call deadline."""

    async def scenario() -> None:
        """Take the GenerationError mid-iteration, then outlive the deadline in the block."""
        bound_llm = LLM(
            _FakeAdapter(stream=_FailsAfterFirstItemStream()),
            shared_backoff=_fast_shared_backoff(),
        ).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.05)
        async with handle:
            with pytest.raises(GenerationError):
                async for _ in handle:
                    pass
            await asyncio.sleep(0.1)
        assert handle.abandoned is None

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_a_failed_stream_entry_leaves_no_armed_deadline() -> None:
    """A failing __aenter__ closes its deadline."""

    async def scenario() -> None:
        """Fail an entry under a short deadline, then outlive that deadline uncancelled."""
        adapter = _FakeAdapter(invalid_requests=[InvalidRequest(reason="no")])
        bound_llm = LLM(adapter, shared_backoff=_fast_shared_backoff()).bind()
        handle = bound_llm.stream_one([UserMessage(content="hi")], timeout_seconds=0.02)
        with pytest.raises(GenerationError):
            async with handle:
                pass
        # Outlast the deadline the failed entry opened. A leaked timer cancels this sleep.
        await asyncio.sleep(0.1)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_no_deadline_lets_a_callers_own_scope_expire() -> None:
    """The default claims nothing, so a caller's own scope converts its cancellation as it would."""

    async def scenario() -> None:
        """Run a call with no deadline and let an outer scope cut it."""
        bound_llm = LLM(
            _FakeAdapter(hang_from_open=1), shared_backoff=_fast_shared_backoff()
        ).bind()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await bound_llm.generate_one([UserMessage(content="hi")])

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
