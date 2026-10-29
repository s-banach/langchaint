"""Drive generation and tracing tests without an SDK."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, override

from pydantic import BaseModel, ValidationError

from langchaint import (
    AssistantMessage,
    Message,
    SharedBackoff,
    StreamItem,
    TextPart,
    ToolCall,
    Usage,
)
from langchaint.adapter import (
    Adapter,
    AdapterStream,
    Binding,
    BoundAdapter,
    MaxCompletionTokensExceeded,
    ProviderBilling,
    Refusal,
    RejectedMessages,
    RequestFailure,
    RequestParams,
    ResponseIdentity,
    ResponseOutcome,
    SchemaViolation,
    TransientError,
    UsableResponse,
    text_output,
)
from tests.helpers import stated_provider_billing, terminal, yield_until

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
"""Billing for a refusal or truncation."""


USAGE_STREAM: Usage = USAGE.model_copy(update={"output_tokens_cost_in_usd": 0.001})
"""Distinct stream billing makes a stream's cost visible in assertions."""


_UNKNOWN_EXCEPTION = terminal("unknown_exception")


def fast_shared_backoff(
    *,
    max_concurrent_requests: int | None = 8,
    max_wait_seconds: float = 0.002,
    max_request_starts_per_second: float = 10_000.0,
) -> SharedBackoff:
    """Create one SharedBackoff per event loop with millisecond waits."""
    return SharedBackoff(
        max_concurrent_requests=max_concurrent_requests,
        min_wait_ceiling_seconds=0.001,
        max_wait_seconds=max_wait_seconds,
        max_request_starts_per_second=max_request_starts_per_second,
    )


class FakeRawResponse(BaseModel):
    id: str
    request_id: str | None = None


def as_fake_raw(raw: BaseModel) -> FakeRawResponse:
    if not isinstance(raw, FakeRawResponse):
        raise TypeError(f"expected a FakeRawResponse, got {type(raw).__name__}")
    return raw


@dataclass(frozen=True, kw_only=True)
class ScriptedResponse:
    outcome: ResponseOutcome[str | None]
    usage: Usage


def billed(outcome: ResponseOutcome[str | None]) -> ScriptedResponse:
    return ScriptedResponse(outcome=outcome, usage=USAGE_BILLED)


REJECTED_ASSISTANT_MESSAGE: AssistantMessage = AssistantMessage(
    content=(TextPart(text="what the rejected 200 carried"),)
)
"""Recoverable content on billed refusals and truncations."""


REFUSAL: Refusal = Refusal(assistant_message=REJECTED_ASSISTANT_MESSAGE)
MAX_COMPLETION_TOKENS_EXCEEDED: MaxCompletionTokensExceeded = MaxCompletionTokensExceeded(
    assistant_message=REJECTED_ASSISTANT_MESSAGE
)


def usable_text_response(content: str) -> UsableResponse[str]:
    return UsableResponse(
        output=content,
        assistant_message=AssistantMessage(content=(TextPart(text=content),)),
        stop_reason="stop",
    )


FAKE_TOOL_CALL: ToolCall = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')


class FakeStream(AdapterStream):
    def __init__(self, *, outcome: ResponseOutcome[str | None] | None = None) -> None:
        self.closed: bool = False
        self.raw: FakeRawResponse = FakeRawResponse(id="fake-final")
        self._usage_reported: Usage | None = None
        """None represents unavailable provider_billing."""
        self._outcome = outcome

    @override
    def provider_billing(self) -> ProviderBilling | None:
        return (
            None if self._usage_reported is None else stated_provider_billing(self._usage_reported)
        )

    @override
    def request_id(self) -> str | None:
        return "req-fake-stream"

    def scripted_response(self) -> ScriptedResponse:
        if self._outcome is not None:
            return billed(self._outcome)
        return ScriptedResponse(outcome=usable_text_response("ab"), usage=USAGE_STREAM)

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        yield "a"
        yield "b"
        yield FAKE_TOOL_CALL

    @override
    async def final(self) -> BaseModel:
        return self.raw

    @override
    async def close(self) -> None:
        self.closed = True


class HangsAfterFirstItemStream(FakeStream):
    """Yield one item before suspending until cancellation."""

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        yield "a"
        await asyncio.Event().wait()


type ScriptedRequest = Exception | ScriptedResponse


class ScriptedRequestStream(FakeStream):
    def __init__(self, *, raw: FakeRawResponse, content: str | None) -> None:
        super().__init__()
        self.raw: FakeRawResponse = raw
        self._content = content

    @override
    def request_id(self) -> str | None:
        return f"req-{self.raw.id}"

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        if self._content is not None:
            yield self._content
            yield FAKE_TOOL_CALL


@dataclass(frozen=True, kw_only=True)
class FakeRequestParams(RequestParams):
    messages: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        return json.dumps([message.model_dump(mode="json") for message in self.messages])


def as_fake_request_params(request_params: RequestParams) -> FakeRequestParams:
    if not isinstance(request_params, FakeRequestParams):
        raise TypeError(f"expected a FakeRequestParams, got {type(request_params).__name__}")
    return request_params


class FakeBoundAdapter(BoundAdapter[str | None]):
    """Consume scripts and count requests independently for each binding."""

    def __init__(self, adapter: "FakeAdapter") -> None:
        self._adapter = adapter
        self._scripted_requests = list(adapter.scripted_requests)
        self._rejected_messages = list(adapter.rejected_messages)
        self._scripted_by_raw_id: dict[str, ScriptedResponse] = {}
        self.final_raws: list[FakeRawResponse] = []
        """The raw response of every request stream this binding built, in order."""
        self.build_count: int = 0
        self.open_count: int = 0
        self.hang_reached: asyncio.Event = asyncio.Event()
        """Set when an open_stream call suspends because of the adapter's hang_from_open."""

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        return stated_provider_billing(self._scripted_by_raw_id[as_fake_raw(raw).id].usage)

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        fake_raw = as_fake_raw(raw)
        return ResponseIdentity(
            model_served="fake-model-served",
            response_id=fake_raw.id,
            request_id=request_id,
        )

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[str | None]:
        return self._scripted_by_raw_id[as_fake_raw(raw).id].outcome

    @override
    def build_request_params(
        self, messages: Sequence[Message]
    ) -> RequestParams | RejectedMessages:
        if self._rejected_messages:
            return self._rejected_messages.pop(0)
        self.build_count += 1
        return FakeRequestParams(messages=tuple(messages))

    def _request_stream(
        self, scripted_response: ScriptedResponse, *, content: str | None
    ) -> ScriptedRequestStream:
        raw = FakeRawResponse(id=f"fake-response-{self.open_count}")
        self._scripted_by_raw_id[raw.id] = scripted_response
        self.final_raws.append(raw)
        return ScriptedRequestStream(raw=raw, content=content)

    @override
    async def open_stream(self, request_params: RequestParams) -> AdapterStream:
        messages = as_fake_request_params(request_params).messages
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
        if self._scripted_requests:
            scripted_request = self._scripted_requests.pop(0)
            if isinstance(scripted_request, Exception):
                raise scripted_request
            return self._request_stream(scripted_request, content=None)
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
        return self._request_stream(
            ScriptedResponse(outcome=usable_text_response(content), usage=USAGE), content=content
        )


class FakeStructuredBoundAdapter[ModelT: BaseModel](BoundAdapter[ModelT | None]):
    def __init__(
        self, text_bound_adapter: FakeBoundAdapter, response_format: type[ModelT]
    ) -> None:
        self._text_bound_adapter = text_bound_adapter
        self._response_format = response_format

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        return self._text_bound_adapter.billing_from_raw(raw)

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        return self._text_bound_adapter.identity_from_raw(raw, request_id=request_id)

    @override
    def interpret(self, raw: BaseModel) -> ResponseOutcome[ModelT | None]:
        outcome = self._text_bound_adapter.interpret(raw)
        if outcome.kind != "usable_response":
            return outcome
        text = text_output(outcome.assistant_message)
        try:
            output = self._response_format.model_validate_json(text) if text is not None else None
        except ValidationError as error:
            return SchemaViolation(
                assistant_message=outcome.assistant_message,
                validation_error_json=error.json(include_url=False),
            )
        return UsableResponse(
            output=output,
            assistant_message=outcome.assistant_message,
            stop_reason=outcome.stop_reason,
        )

    @override
    def build_request_params(
        self, messages: Sequence[Message]
    ) -> RequestParams | RejectedMessages:
        return self._text_bound_adapter.build_request_params(messages)

    @override
    async def open_stream(self, request_params: RequestParams) -> AdapterStream:
        return await self._text_bound_adapter.open_stream(request_params)


class RequestIdError(RuntimeError):
    def __init__(self, message: str, request_id: str) -> None:
        super().__init__(message)
        self.request_id: str = request_id


class TransientRequestIdError(TransientError):
    def __init__(self, message: str, request_id: str) -> None:
        super().__init__(message)
        self.request_id: str = request_id


class FakeAdapter(Adapter):
    _bound_adapter_class: ClassVar[type[FakeBoundAdapter]] = FakeBoundAdapter

    def __init__(
        self,
        *,
        scripted_requests: Sequence[ScriptedRequest] = (),
        rejected_messages: Sequence[RejectedMessages] = (),
        echo: bool = False,
        stream: FakeStream | None = None,
        request_failure: RequestFailure = _UNKNOWN_EXCEPTION,
        open_seconds: float = 0.0,
        hang_from_open: int | None = None,
        open_barrier: asyncio.Barrier | None = None,
        open_barrier_from_call: int = 1,
        automatic_cache_breakpoints_default: bool = False,
    ) -> None:
        """Wait at open_barrier starting with call open_barrier_from_call.

        open_stream then suspends starting with call hang_from_open; otherwise it waits open_seconds.
        Requests consume scripted_requests, then use stream, then produce "ok".
        With echo=True, the default response echoes the first message when it is a user message with string content.
        build_request_params consumes rejected_messages before constructing request params.
        """
        super().__init__(
            client=None,
            model="fake-model",
            provider_name="fake",
            automatic_cache_breakpoints_default=automatic_cache_breakpoints_default,
        )
        self.scripted_requests: Sequence[ScriptedRequest] = scripted_requests
        self.rejected_messages: Sequence[RejectedMessages] = rejected_messages
        self.echo: bool = echo
        self.stream: FakeStream | None = stream
        self.open_seconds: float = open_seconds
        self.hang_from_open: int | None = hang_from_open
        self.open_barrier: asyncio.Barrier | None = open_barrier
        self.open_barrier_from_call: int = open_barrier_from_call
        self._request_failure = request_failure
        self.bound_adapters: list[FakeBoundAdapter] = []
        self.structured_bind_count: int = 0

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        return {"automatic_cache_breakpoints_default": self.automatic_cache_breakpoints_default}

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str | None]:
        bound = self._bound_adapter_class(self)
        self.bound_adapters.append(bound)
        return bound

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        self.structured_bind_count += 1
        return FakeStructuredBoundAdapter(self._bound_adapter_class(self), response_format)

    @override
    def request_failure(self, error: Exception) -> RequestFailure:
        return self._request_failure

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        if isinstance(error, (RequestIdError, TransientRequestIdError)):
            return error.request_id
        return None


SIBLING_OPEN_YIELDS = 100
"""Event loop iterations, far more than an unblocked sibling generation needs to reach open_stream."""


async def assert_the_first_open_runs_alone(
    adapter: FakeAdapter, opens_pair_up: asyncio.Barrier
) -> None:
    """Hold the first open at opens_pair_up, assert that no sibling opens meanwhile, then release it.

    adapter's open_barrier must be opens_pair_up with two parties, so later opens release each other.
    A batch that starts its siblings with the first item opens a second request within SIBLING_OPEN_YIELDS iterations.
    """
    bound_adapter = adapter.bound_adapters[0]
    await yield_until(lambda: opens_pair_up.n_waiting == 1)
    for _ in range(SIBLING_OPEN_YIELDS):
        await asyncio.sleep(0)
    assert bound_adapter.open_count == 1
    _ = await opens_pair_up.wait()
