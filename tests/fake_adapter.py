"""Fake adapters, streams, and scripted responses shared by the generation test modules.

The fakes implement the adapter contract without an SDK, so tests drive BoundLLM, StreamHandle, and tracing offline.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, override

from pydantic import BaseModel

from langchaint import (
    AssistantMessage,
    DoNotRetry,
    Message,
    SharedBackoff,
    StreamItem,
    TextPart,
    ToolCall,
    TransientError,
    Usage,
    Verdict,
)
from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    BoundAdapter,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    ProviderBilling,
    Refusal,
    RequestParams,
    ResponseIdentity,
    ResponseOutcome,
    verdict_from_transient_error,
)
from tests.helpers import stated_provider_billing, yield_until

USAGE: Usage = Usage(
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


USAGE_BILLED: Usage = USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.25})
"""The billing a 200 that produced no output (a refusal or truncation) carries."""


USAGE_STREAM: Usage = USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.001})
"""The stream final()'s assembled usage, distinct so a stream cost is visible."""


def parse_fake(failure: Exception) -> Verdict:
    """Map TransientError with verdict_from_transient_error."""
    if isinstance(failure, TransientError):
        return verdict_from_transient_error(failure)
    return DoNotRetry()


def fast_shared_backoff(
    *,
    max_concurrent_requests: int | None = 8,
    parse: Callable[[Exception], Verdict] = parse_fake,
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


class FakeRawResponse(BaseModel):
    """Identify one fake raw response and its optional request ID."""

    id: str
    request_id: str | None = None


def as_fake_raw(raw: BaseModel) -> FakeRawResponse:
    """Narrow a raw response to the fake one.

    Raises:
        TypeError: raw is not a FakeRawResponse, which the real adapters raise for the same reason.
    """
    if not isinstance(raw, FakeRawResponse):
        raise TypeError(f"expected a FakeRawResponse, got {type(raw).__name__}")
    return raw


@dataclass(frozen=True, kw_only=True)
class ScriptedResponse:
    """One response the fake hands back: what interpret reads off it, and what it billed."""

    outcome: ResponseOutcome[str]
    usage: Usage


def billed(outcome: ResponseOutcome[str]) -> ScriptedResponse:
    """Script one 200 the provider billed, whatever interpret goes on to make of it."""
    return ScriptedResponse(outcome=outcome, usage=USAGE_BILLED)


REJECTED_TURN: AssistantMessage = AssistantMessage(
    turn=(TextPart(text="what the rejected 200 carried"),)
)
"""The turn a 200 that produced no output still carried, which every such variant takes."""


REFUSAL: Refusal = Refusal(assistant_message=REJECTED_TURN)
MAX_COMPLETION_TOKENS_EXCEEDED: MaxCompletionTokensExceeded = MaxCompletionTokensExceeded(
    assistant_message=REJECTED_TURN
)


def success_result(content: str) -> AdapterResult[str]:
    """Build a successful text AdapterResult carrying the given content."""
    return AdapterResult(
        output=content,
        assistant_message=AssistantMessage(turn=(TextPart(text=content),)),
        stop_reason="end_turn",
    )


FAKE_TOOL_CALL: ToolCall = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')


class FakeStream(AdapterStream):
    """Provide fixed stream items and an assembled response."""

    def __init__(self, *, outcome: ResponseOutcome[str] | None = None) -> None:
        """Script the assembled outcome, defaulting to a success whose output is "ab"."""
        self.closed: bool = False
        self.raw: FakeRawResponse = FakeRawResponse(id="fake-final")
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

    def scripted_response(self) -> ScriptedResponse:
        """Return the assembled result the SDK would produce, and what the stream billed."""
        if self._outcome is not None:
            return billed(self._outcome)
        return ScriptedResponse(outcome=success_result("ab"), usage=USAGE_STREAM)

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        yield "a"
        yield "b"
        yield FAKE_TOOL_CALL

    @override
    async def final(self) -> BaseModel:
        """Return the response the stream's events assembled into."""
        return self.raw

    @override
    async def close(self) -> None:
        self.closed = True


class HangsAfterFirstItemStream(FakeStream):
    """Yield one item before suspending until cancellation."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        """Yield one item, then wait on an event that never fires.

        Yields:
            One item, and nothing after it.
        """
        yield "a"
        await asyncio.Event().wait()


type ScriptedAttempt = Exception | ScriptedResponse
"""One open_stream exception or assembled response."""


class ScriptedAttemptStream(FakeStream):
    """Stream one scripted response.

    Each attempt has a fresh raw response and request ID.
    A success yields its content and FAKE_TOOL_CALL.
    """

    def __init__(self, *, raw: FakeRawResponse, content: str | None) -> None:
        """Store final output and optional streamed content."""
        super().__init__()
        self.raw: FakeRawResponse = raw
        self._content = content

    @override
    def request_id(self) -> str | None:
        """Derive the request ID from raw.id."""
        return f"req-{self.raw.id}"

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        if self._content is not None:
            yield self._content
            yield FAKE_TOOL_CALL


@dataclass(frozen=True, kw_only=True)
class FakeRequest(RequestParams):
    """Store messages for a fake request."""

    messages: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        """Serialize messages as JSON."""
        return json.dumps([message.model_dump(mode="json") for message in self.messages])


def as_fake_request(request: RequestParams) -> FakeRequest:
    """Narrow a request to the fake one.

    Raises:
        TypeError: request is not a FakeRequest, which the real adapters raise for the same reason.
    """
    if not isinstance(request, FakeRequest):
        raise TypeError(f"expected a FakeRequest, got {type(request).__name__}")
    return request


class FakeBoundAdapter(BoundAdapter[str]):
    """A bound adapter whose open_stream follows the behavior its FakeAdapter was given.

    Each bound adapter consumes its own copy of the adapter's scripts and keeps its own counters.
    """

    def __init__(self, adapter: "FakeAdapter") -> None:
        """Copy the adapter's scripts so this binding consumes them independently."""
        self._adapter = adapter
        self._scripted_attempts = list(adapter.scripted_attempts)
        self._invalid_requests = list(adapter.invalid_requests)
        self._scripted_by_raw_id: dict[str, ScriptedResponse] = {}
        self.final_raws: list[FakeRawResponse] = []
        """The raw response of every attempt stream this binding built, in order."""
        self.build_count: int = 0
        self.open_count: int = 0
        self.hang_reached: asyncio.Event = asyncio.Event()
        """Set when an open_stream call suspends because of the adapter's hang_from_open."""

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Return what the response under this raw was scripted to have billed."""
        return stated_provider_billing(self._scripted_by_raw_id[as_fake_raw(raw).id].usage)

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Name the fake model, take the response id from the raw's own, and the request id as it came."""
        fake_raw = as_fake_raw(raw)
        return ResponseIdentity(
            model_served="fake-model-served",
            response_id=fake_raw.id,
            request_id=request_id,
        )

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str]:
        """Return what the response under this raw was scripted to produce."""
        return self._scripted_by_raw_id[as_fake_raw(raw).id].outcome

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Report the scripted refusal, else carry messages into the request."""
        if self._invalid_requests:
            return self._invalid_requests.pop(0)
        self.build_count += 1
        return FakeRequest(messages=tuple(messages))

    def _attempt_stream(
        self, scripted_response: ScriptedResponse, *, content: str | None
    ) -> ScriptedAttemptStream:
        """Register the scripted response under a fresh raw and wrap it in this attempt's stream."""
        raw = FakeRawResponse(id=f"fake-response-{self.open_count}")
        self._scripted_by_raw_id[raw.id] = scripted_response
        self.final_raws.append(raw)
        return ScriptedAttemptStream(raw=raw, content=content)

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Count the attempt, suspend, then raise or return the next scripted attempt's stream.

        Raises:
            TypeError: request is not a FakeRequest.
            Exception: the next scripted failure.
        """
        messages = as_fake_request(request).messages
        self.open_count += 1
        open_call = self.open_count
        adapter = self._adapter
        if adapter.open_barrier is not None and open_call >= adapter.open_barrier_from_call:
            await adapter.open_barrier.wait()
        if adapter.hang_from_open is not None and open_call >= adapter.hang_from_open:
            self.hang_reached.set()
            await asyncio.Event().wait()
        if adapter.open_seconds:
            await asyncio.sleep(adapter.open_seconds)
        if self._scripted_attempts:
            scripted_attempt = self._scripted_attempts.pop(0)
            if isinstance(scripted_attempt, Exception):
                raise scripted_attempt
            return self._attempt_stream(scripted_attempt, content=None)
        if adapter.stream is not None:
            stream = adapter.stream
            self._scripted_by_raw_id[stream.raw.id] = stream.scripted_response()
            return stream
        first = messages[0]
        content = (
            first.content
            if adapter.echo and first.kind == "user" and isinstance(first.content, str)
            else "ok"
        )
        return self._attempt_stream(
            ScriptedResponse(outcome=success_result(content), usage=USAGE), content=content
        )


class FakeStructuredBoundAdapter[ModelT: BaseModel](BoundAdapter[ModelT]):
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


class RequestIdError(RuntimeError):
    """An error carrying the request-id header its response had, as both SDKs' APIStatusError does."""

    def __init__(self, message: str, request_id: str) -> None:
        """Store the message and the request id."""
        super().__init__(message)
        self.request_id: str = request_id


class TransientRequestIdError(TransientError):
    """The same id, on the class an adapter raises to retry an attempt without going through classify."""

    def __init__(self, message: str, request_id: str) -> None:
        """Store the message and the request id."""
        super().__init__(message)
        self.request_id: str = request_id


class FakeAdapter(Adapter):
    """An adapter whose bind_text hands out fake bound adapters."""

    _bound_adapter_class: ClassVar[type[FakeBoundAdapter]] = FakeBoundAdapter
    """The class bind_text hands out; a subclass names its own to vary what interpret does."""

    def __init__(
        self,
        *,
        scripted_attempts: Sequence[ScriptedAttempt] = (),
        invalid_requests: Sequence[InvalidRequest] = (),
        echo: bool = False,
        stream: FakeStream | None = None,
        classify_result: ErrorClassification = "unknown_exception",
        open_seconds: float = 0.0,
        hang_from_open: int | None = None,
        open_barrier: asyncio.Barrier | None = None,
        open_barrier_from_call: int = 1,
        automatic_cache_breakpoints_default: bool = False,
    ) -> None:
        """Store the behavior every bound adapter reads, and the classify verdict.

        From call number open_barrier_from_call on, open_stream first waits at open_barrier.
        From call number hang_from_open on, it then suspends until cancelled.
        It then sleeps open_seconds, and raises or streams the next of scripted_attempts.
        With none left, it returns stream.
        Without a stream, it streams a success whose output is "ok".
        With echo set and a first message that is a user message with str content, the output is that content.
        build_request returns the next of invalid_requests while any remain.
        """
        # This adapter reaches no SDK, so it passes client=None.
        # The empty provider_name_by_client_class preserves the stated "fake" provider_name.
        super().__init__(
            client=None,
            model="fake-model",
            provider_name="fake",
            automatic_cache_breakpoints_default=automatic_cache_breakpoints_default,
        )
        self.scripted_attempts: Sequence[ScriptedAttempt] = scripted_attempts
        self.invalid_requests: Sequence[InvalidRequest] = invalid_requests
        self.echo: bool = echo
        self.stream: FakeStream | None = stream
        self.open_seconds: float = open_seconds
        self.hang_from_open: int | None = hang_from_open
        self.open_barrier: asyncio.Barrier | None = open_barrier
        self.open_barrier_from_call: int = open_barrier_from_call
        self._classify_result = classify_result
        self.bound_adapters: list[FakeBoundAdapter] = []
        self.structured_bind_count: int = 0

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return the fake adapter's stored request configuration."""
        return {"automatic_cache_breakpoints_default": self.automatic_cache_breakpoints_default}

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        bound = self._bound_adapter_class(self)
        self.bound_adapters.append(bound)
        return bound

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT]:
        """Build a structured bound adapter and count the call."""
        self.structured_bind_count += 1
        bound: BoundAdapter[ModelT] = FakeStructuredBoundAdapter()
        return bound

    failure_types: ClassVar[tuple[type[Exception], ...]] = (TransientError,)

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Delegate to the module-level rule fast_shared_backoff also parses with."""
        return parse_fake(failure)

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Return the fixed verdict for every exception classify sees."""
        return self._classify_result

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        """Read the request id off the errors that carry one, as each SDK's adapter does."""
        if isinstance(error, (RequestIdError, TransientRequestIdError)):
            return error.request_id
        return None


SIBLING_OPEN_YIELDS = 100
"""Event loop turns, far more than an unblocked sibling generation needs to reach open_stream."""


async def assert_the_first_open_runs_alone(
    adapter: FakeAdapter, opens_pair_up: asyncio.Barrier
) -> None:
    """Hold the first open at opens_pair_up, assert that no sibling opens meanwhile, then release it.

    adapter's open_barrier must be opens_pair_up with two parties, so later opens release each other.
    A batch that starts its siblings with the first item opens a second request within SIBLING_OPEN_YIELDS turns.
    """
    bound_adapter = adapter.bound_adapters[0]
    await yield_until(lambda: opens_pair_up.n_waiting == 1)
    for _ in range(SIBLING_OPEN_YIELDS):
        await asyncio.sleep(0)
    assert bound_adapter.open_count == 1
    _ = await opens_pair_up.wait()
