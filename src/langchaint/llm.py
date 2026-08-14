"""Provider-neutral `LLM` construction and binding.

`LLM.bind` freezes a prompt prefix and returns `BoundLLM`.
Each request attempt runs inside `SharedBackoff.admitted`.
`PauseAll` pauses shared requests; `RetryThisOne` retries only the current request.
"""

import asyncio
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple, Protocol, overload

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    Binding,
    BoundAdapter,
    ErrorClassification,
    InvalidRequest,
    RequestParams,
    ResponseOutcome,
    ToolChoice,
)
from langchaint.call import _CallLedger
from langchaint.exceptions import (
    ContextWindowExceededError,
    EmptyTurnError,
    EscapedExceptionError,
    GenerationError,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    ParserContractError,
    ProviderDeclaredFinalError,
    ProviderFailedTerminallyError,
    RefusalError,
    RetriesExhaustedError,
    SchemaViolationError,
    StreamProtocolError,
    TimedOutError,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
)
from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, TextPart, UserMessage
from langchaint.pricing import Billing
from langchaint.response import (
    CallResult,
    GenerateResult,
    Response,
    ToolCallTurn,
    _abandoned_call_error,
    _success_variant,
)
from langchaint.run_many import max_pending_for_requests, run_many
from langchaint.sequence_not_str import SequenceNotStr
from langchaint.shared_backoff import (
    Admission,
    DoNotRetry,
    PauseAllDoNotRetry,
    PrivateBackoff,
    SharedBackoff,
    Verdict,
)
from langchaint.streaming import StreamHandle, _close_stream_quietly
from langchaint.tools import Tool, ToolManager


class _StreamObservations(NamedTuple):
    """Billing, request ID, and open status captured before a failed stream closes."""

    billing: Billing | None
    request_id: str | None
    opened: bool


class Unchanged:
    """Sentinel type for rebind parameters the caller leaves as bound.

    Not in __all__: a caller never constructs or passes it, since omitting the keyword is the interface;
    it appears only in the rebind signature the caller reads.
    """

    def __repr__(self) -> str:
        """Render the default as UNCHANGED in signatures and help() output."""
        return "UNCHANGED"


UNCHANGED: Unchanged = Unchanged()


type GenerationInput = str | Sequence[Message]
"""What one request is generated from: a bare str is shorthand for a Sequence[Message] of one UserMessage."""


class Deadline(Protocol):
    """The scope one call runs inside, told when the call waits to be admitted and when it is.

    Admission waits include the `SharedBackoff` permit and admission queue.
    Implementations differ only in whether that wait counts against the call.
    """

    @property
    def scope(self) -> asyncio.Timeout:
        """The scope to enter around the retry loop, expiring when the call is out of time."""
        ...

    def suspend_until_admitted(self) -> None:
        """Answer an attempt about to wait for admission."""
        ...

    def resume_on_admission(self) -> None:
        """Answer an attempt now admitted, free to send its request."""
        ...


class WallClockDeadline:
    """A deadline that runs from construction to the result, whatever the call waits on.

    `generate_one.timeout_seconds` includes admission waits.
    """

    def __init__(self, timeout_seconds: float | None) -> None:
        """Create the scope with timeout_seconds; None disables expiration."""
        self.scope: asyncio.Timeout = asyncio.timeout(timeout_seconds)

    def suspend_until_admitted(self) -> None:
        """Keep the clock running."""

    def resume_on_admission(self) -> None:
        """Keep the clock running."""


class WorkingTimeDeadline:
    """A deadline that stops while the call waits to be admitted and runs the rest of the time.

    This is what generate_many's max_working_seconds_per_item asks for.
    """

    def __init__(self, max_working_seconds: float | None) -> None:
        """Create a scope without expiration; resume_on_admission schedules the budget."""
        self.scope: asyncio.Timeout = asyncio.timeout(None)
        self._seconds_left = max_working_seconds

    def suspend_until_admitted(self) -> None:
        """Stop the clock, banking what is left for the resume that follows."""
        if self._seconds_left is None:
            return
        expires_at = self.scope.when()
        if expires_at is not None:
            self._seconds_left = expires_at - asyncio.get_running_loop().time()
        self.scope.reschedule(None)

    def resume_on_admission(self) -> None:
        """Start the clock again with what is banked, which on the first attempt is the budget."""
        if self._seconds_left is None:
            return
        self.scope.reschedule(asyncio.get_running_loop().time() + self._seconds_left)


def _as_messages(generation_input: GenerationInput) -> Sequence[Message]:
    if isinstance(generation_input, str):
        return (UserMessage(content=generation_input),)
    return generation_input


def _build_binding(  # noqa: PLR0913 (every parameter becomes one Binding field)
    *,
    system_prompt: str | Sequence[TextPart] | None,
    tool_manager: ToolManager | None,
    provider_executed_tools: Sequence[Mapping[str, object]],
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    inference_params: InferenceParams,
    automatic_cache_breakpoints: bool,
    extra_body: Mapping[str, object] | None,
) -> Binding:
    """Convert bind arguments to the frozen Binding.

    Tool schema conversion happens here, once per binding.

    Raises:
        ValueError: system_prompt is an empty sequence of parts; pass None to bind no system prompt.
    """
    if system_prompt is not None and not isinstance(system_prompt, str):
        if not system_prompt:
            raise ValueError(
                "system_prompt is an empty sequence of parts; pass None to bind no system prompt"
            )
        system_prompt = tuple(system_prompt)
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=() if tool_manager is None else tool_manager.schemas(),
        provider_executed_tools=tuple(provider_executed_tools),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        inference_params=inference_params,
        automatic_cache_breakpoints=automatic_cache_breakpoints,
        extra_body=extra_body,
    )


def _bind_adapter(
    adapter: Adapter, binding: Binding, response_format: type[Any] | None
) -> BoundAdapter[Any]:
    """Dispatch to the adapter bind method the response_format selects.

    The caller-visible output type comes from the bind / rebind overloads,
    so this returns BoundAdapter[Any] and the Any is confined here.
    The parameter is type[Any] | None, not type[BaseModel] | None,
    because rebind feeds it the stored response_format typed type[OutputT] | None:
    type[OutputT] with OutputT unbounded is not assignable to a BaseModel-bounded parameter,
    and narrowing with is None narrows the value, not OutputT.
    """
    if response_format is None:
        return adapter.bind_text(binding)
    return adapter.bind_structured(binding, response_format)


def _resolve_tool_manager(
    tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]] | None,
) -> ToolManager | None:
    """Raise `ValueError` when a `tools` sequence contains duplicate names."""
    if isinstance(tools, ToolManager) or tools is None:
        return tools
    return ToolManager(tools)


class GenerateItem[OutputT](Protocol):
    """Run one batch item."""

    async def __call__(
        self, generation_input: GenerationInput, *, deadline: Deadline
    ) -> GenerateResult[OutputT | None]:
        """Run the call, raising its GenerationError rather than returning it.

        Raises:
            GenerationError: the call failed; the batch turns it into that item's result.
        """
        ...


class LLM:
    """The un-bound client; holds what is shared across bindings."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        shared_backoff: SharedBackoff | None = None,
    ) -> None:
        """Store the shared pieces.

        `shared_backoff=None` creates a `SharedBackoff` from the adapter.
        It uses `max_concurrent_requests=8` and other `SharedBackoff` defaults.
        Pass one instance to every `LLM` sharing a rate-limit quota.
        """
        self.adapter: Adapter = adapter
        self.shared_backoff: SharedBackoff = (
            shared_backoff
            if shared_backoff is not None
            else SharedBackoff(
                parse=adapter.parse, failure_types=adapter.failure_types, max_concurrent_requests=8
            )
        )

    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[ModelT, ToolManager]": ...
    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[ModelT, None]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[str, ToolManager]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[str, None]": ...
    def bind(  # noqa: PLR0913 (the binding states every choice: prompt, tools, format, params, caching, extra_body)
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]] | None = None,
        provider_executed_tools: Sequence[Mapping[str, object]] = (),
        response_format: type[BaseModel] | None = None,
        inference_params: InferenceParams | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_attempts: int = 3,
        automatic_cache_breakpoints: bool | None = None,
    ) -> "BoundLLM[Any, Any]":
        """Freeze the prompt prefix and fix the output type.

        `response_format` sets `OutputT`; its absence uses `str`.
        A `tools` sequence constructs `ToolManager`; an existing `ToolManager` retains its identity.
        `automatic_cache_breakpoints=None` uses `Adapter.automatic_cache_breakpoints_default`.
        `max_attempts` counts requests including the first.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `automatic_cache_breakpoints` is unsupported.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_attempts` is boolean or below one.
        """
        tool_manager = _resolve_tool_manager(tools)
        binding = _build_binding(
            system_prompt=system_prompt,
            tool_manager=tool_manager,
            provider_executed_tools=provider_executed_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            inference_params=(
                inference_params if inference_params is not None else InferenceParams()
            ),
            automatic_cache_breakpoints=(
                self.adapter.automatic_cache_breakpoints_default
                if automatic_cache_breakpoints is None
                else automatic_cache_breakpoints
            ),
            extra_body=extra_body,
        )
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, binding, response_format),
            response_format=response_format,
            binding=binding,
            tool_manager=tool_manager,
            shared_backoff=self.shared_backoff,
            max_attempts=max_attempts,
        )


class BoundLLM[OutputT, ToolManagerT: ToolManager | None = None]:
    """A frozen prompt prefix with generation and streaming methods.

    `OutputT` is `str` or the validated `response_format` type.
    A structured binding with `ToolManager` returns `ToolCallTurn` for tool-call turns.
    `tool_manager` preserves the bound `ToolManager` for application dispatch.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT | None],
        response_format: type[OutputT] | None,
        binding: Binding,
        tool_manager: ToolManagerT,
        shared_backoff: SharedBackoff,
        max_attempts: int,
    ) -> None:
        """Store the frozen pieces; called by `LLM.bind` and `rebind` only.

        Raises:
            ValueError: `max_attempts` is a bool or below one.
        """
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive int, got {max_attempts!r}")
        self.adapter: Adapter = adapter
        self.binding: Binding = binding
        self.response_format: type[OutputT] | None = response_format
        self.shared_backoff: SharedBackoff = shared_backoff
        self.max_attempts: int = max_attempts
        self._bound_adapter = bound_adapter
        self._tool_manager = tool_manager

    @property
    def tool_manager(self) -> ToolManagerT:
        """The bound ToolManager, or None where none was bound.

        A property, not an attribute, so ToolManagerT stays covariant.
        An attribute is written as well as read, so its parameter would be invariant.
        A structured binding then named by a ToolManager subclass would match no request method.
        """
        return self._tool_manager

    @property
    def _splits_tool_call_turns(self) -> bool:
        """Whether this binding's tool-call turns are ToolCallTurn: it is structured and tool-bound."""
        return self.response_format is not None and self._tool_manager is not None

    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolManager]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, None]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: type[NewModelT],
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolManagerT]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[str, ToolManager]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[str, None]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: None,
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[str, ToolManagerT]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: ToolManager | Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManager]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[OutputT, None]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManagerT]": ...
    def rebind(  # noqa: PLR0913 (rebind takes every field bind takes, each replaceable alone)
        self,
        *,
        response_format: type[BaseModel] | None | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tools: (
            ToolManager
            | Sequence[Tool[BaseModel | Mapping[str, object] | None]]
            | None
            | Unchanged
        ) = UNCHANGED,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        inference_params: InferenceParams | Unchanged = UNCHANGED,
        extra_body: Mapping[str, object] | None | Unchanged = UNCHANGED,
        max_attempts: int | Unchanged = UNCHANGED,
        automatic_cache_breakpoints: bool | None | Unchanged = UNCHANGED,
    ) -> "BoundLLM[Any, Any]":
        """Return a new `BoundLLM` with specified fields replaced.

        Omitting a field preserves its value.
        `response_format` sets `OutputT`; `tools` sets `ToolManagerT`.
        `tools=None` removes `ToolManager`; an existing `ToolManager` retains its identity.
        `inference_params` replaces the complete value.
        `automatic_cache_breakpoints=None` reads `Adapter.automatic_cache_breakpoints_default`.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `automatic_cache_breakpoints` is unsupported.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_attempts` is boolean or below one.
        """
        if isinstance(tools, Unchanged):
            tool_manager = self.tool_manager
        else:
            tool_manager = _resolve_tool_manager(tools)
        new_binding = _build_binding(
            system_prompt=(
                self.binding.system_prompt
                if isinstance(system_prompt, Unchanged)
                else system_prompt
            ),
            tool_manager=tool_manager,
            provider_executed_tools=(
                self.binding.provider_executed_tools
                if isinstance(provider_executed_tools, Unchanged)
                else provider_executed_tools
            ),
            tool_choice=(
                self.binding.tool_choice if isinstance(tool_choice, Unchanged) else tool_choice
            ),
            parallel_tool_calls=(
                self.binding.parallel_tool_calls
                if isinstance(parallel_tool_calls, Unchanged)
                else parallel_tool_calls
            ),
            inference_params=(
                self.binding.inference_params
                if isinstance(inference_params, Unchanged)
                else inference_params
            ),
            automatic_cache_breakpoints=(
                self.binding.automatic_cache_breakpoints
                if isinstance(automatic_cache_breakpoints, Unchanged)
                else (
                    self.adapter.automatic_cache_breakpoints_default
                    if automatic_cache_breakpoints is None
                    else automatic_cache_breakpoints
                )
            ),
            extra_body=(
                self.binding.extra_body if isinstance(extra_body, Unchanged) else extra_body
            ),
        )
        new_response_format = (
            self.response_format if isinstance(response_format, Unchanged) else response_format
        )
        new_max_attempts = (
            self.max_attempts if isinstance(max_attempts, Unchanged) else max_attempts
        )
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, new_binding, new_response_format),
            response_format=new_response_format,
            binding=new_binding,
            tool_manager=tool_manager,
            shared_backoff=self.shared_backoff,
            max_attempts=new_max_attempts,
        )

    def _terminal_error(
        self,
        exc: Exception,
        *,
        verdict: Verdict | None,
        ledger: _CallLedger,
        request: RequestParams,
        observations: _StreamObservations,
    ) -> GenerationError:
        """Convert a terminal verdict or `classify` result to `GenerationError`.

        Attempt records preserve in-flight billing and the staged response.
        """
        classification: ErrorClassification = (
            "declared_final"
            if isinstance(verdict, PauseAllDoNotRetry)
            else self.adapter.classify(exc)
        )
        if classification == "invalid_request":
            # Adapter.classify returns invalid_request only for a request the provider rejected,
            # so it went out and gets a record.
            ledger.record(error=None, assistant_message=None, billing=observations.billing)
            return InvalidRequestError(
                reason=f"the provider rejected the request: {exc}",
                call=ledger.freeze(),
                request=request,
            )
        if classification == "declared_final":
            # The provider answered, so the attempt gets a record.
            ledger.record(error=None, assistant_message=None, billing=observations.billing)
            return ProviderDeclaredFinalError(error=exc, call=ledger.freeze(), request=request)
        if observations.opened:
            # The stream was open, so langchaint can say the attempt reached the provider and
            # what that provider reported for it; nothing is recorded where it cannot.
            ledger.record(error=None, assistant_message=None, billing=observations.billing)
        return UnknownExceptionError(error=exc, call=ledger.freeze(), request=request)

    def _request_id_for_failure(
        self, exc: Exception, observations: _StreamObservations
    ) -> str | None:
        """Name the failed attempt's request id: the error's own header, else the open stream's."""
        request_id = self.adapter.request_id_from_error(exc)
        return request_id if request_id is not None else observations.request_id

    async def _pace_after_verdict(
        self,
        exc: Exception,
        *,
        admission: Admission,
        private_backoff: PrivateBackoff,
        assistant_message: AssistantMessage | None,
        ledger: _CallLedger,
        request: RequestParams,
        observations: _StreamObservations,
    ) -> None:
        """Record a verdicted attempt and apply its retry delay.

        `RetryThisOne` uses `PrivateBackoff`; shared pauses apply at the next admission.
        No delay follows the final attempt.

        Raises:
            GenerationError: A terminal verdict stops this request.
        """
        ledger.note_request_id(self._request_id_for_failure(exc, observations))
        verdict = admission.verdict
        if verdict is None or isinstance(verdict, DoNotRetry | PauseAllDoNotRetry):
            raise self._terminal_error(
                exc,
                verdict=verdict,
                ledger=ledger,
                request=request,
                observations=observations,
            ) from exc
        if isinstance(exc, TransientError):
            error = exc
        else:
            error = TransientError(
                str(exc),
                retry_after_seconds=verdict.retry_after,
                is_rate_limit=verdict.kind == "pause_all",
            )
            error.__cause__ = exc
        ledger.record(
            error=error, assistant_message=assistant_message, billing=observations.billing
        )
        if verdict.kind == "retry_this_one" and ledger.attempts < self.max_attempts:
            await asyncio.sleep(private_backoff.next_wait(verdict.retry_after))

    async def _pace_after_transport_failure(
        self,
        exc: Exception,
        *,
        private_backoff: PrivateBackoff,
        ledger: _CallLedger,
        request: RequestParams,
        observations: _StreamObservations,
    ) -> GenerationError | None:
        """Retry a transient transport failure or return its terminal error.

        `StreamProtocolError` also retries because generated items have not reached the caller.
        No delay follows the final attempt.
        """
        ledger.note_request_id(self._request_id_for_failure(exc, observations))
        if not isinstance(exc, StreamProtocolError) and self.adapter.classify(exc) != "transient":
            return self._terminal_error(
                exc, verdict=None, ledger=ledger, request=request, observations=observations
            )
        error = TransientError(str(exc))
        error.__cause__ = exc
        ledger.record(error=error, assistant_message=None, billing=observations.billing)
        if ledger.attempts < self.max_attempts:
            await asyncio.sleep(private_backoff.next_wait(None))
        return None

    def _staged_interpretation(
        self, raw: BaseModel, *, request_id: str | None, ledger: _CallLedger
    ) -> ResponseOutcome[OutputT | None]:
        """Stage an arrived response with its billing, then read what it produced.

        Staging first is what makes the attempt and its billing survive a raise from interpret:
        freeze closes a still-staged response, so the error that raise becomes carries the record.
        request_id is the open stream's, filling in where the assembled response carries none.

        Raises:
            Exception: whatever interpret raises, for Adapter.classify to sort.
        """
        ledger.stage_response(
            raw=raw,
            billing=self._bound_adapter.billing_from_raw(raw),
            identity=self._bound_adapter.identity_from_raw(raw, request_id=request_id),
        )
        return self._bound_adapter.interpret(raw)

    async def _generate_with_retries(
        self,
        messages: Sequence[Message],
        *,
        ledger: _CallLedger,
        deadline: Deadline,
    ) -> GenerateResult[OutputT | None]:
        """Run generation attempts under `deadline` and record each outcome.

        Each request runs inside one `SharedBackoff.admitted` block.
        Billing is recorded before response interpretation.
        Backoff waits run outside admission.

        Raises:
            InvalidRequestError: The adapter rejects the request.
            ProviderDeclaredFinalError: The provider declares a terminal failure.
            UnknownExceptionError: The adapter cannot classify an exception.
            RefusalError: The provider refuses structured output.
            MaxCompletionTokensExceededError: Structured output reaches its token limit.
            EmptyTurnError: The model produces no structured output or tool call.
            SchemaViolationError: The output fails `response_format` validation.
            ContextWindowExceededError: The request exceeds the context window.
            UnfinishedTurnError: The provider returns an unfinished turn.
            ProviderFailedTerminallyError: The response reports a terminal provider failure.
            RetriesExhaustedError: Transient failures consume `max_attempts`.
            TimedOutError: `deadline` expires.
            ParserContractError: `Adapter.parse` violates its contract.
        """
        ledger.start_call()
        timeout_scope = deadline.scope
        try:
            async with timeout_scope:
                return await self._attempt_until_budget_runs_out(
                    messages, ledger=ledger, deadline=deadline
                )
        except TimeoutError:
            if not timeout_scope.expired():
                raise
            # The ledger's noted in-flight billing is what the cut-off attempt's stream had
            # reported, noted before the loop's frame unwound; None where a record settled it.
            raise _abandoned_call_error(TimedOutError, ledger, ledger.billing_in_flight) from None

    async def _attempt_until_budget_runs_out(
        self, messages: Sequence[Message], *, ledger: _CallLedger, deadline: Deadline
    ) -> GenerateResult[OutputT | None]:
        """Send requests until success, a terminal failure, or `max_attempts`.

        Each attempt drains its stream before exposing output.
        A failed stream records current billing and `request_id` before closing.

        Raises:
            GenerationError: The call reaches a terminal failure.
            ParserContractError: `Adapter.parse` violates its contract.
        """
        request = self._request_for_messages(messages, ledger=ledger)
        private_backoff = PrivateBackoff(self.shared_backoff)
        while ledger.attempts < self.max_attempts:
            deadline.suspend_until_admitted()
            admission = self.shared_backoff.admitted()
            assistant_message: AssistantMessage | None = None
            observations = _StreamObservations(billing=None, request_id=None, opened=False)
            try:
                async with admission:
                    # Entering the block is the admission, so the clock starts on the first
                    # statement inside it.
                    deadline.resume_on_admission()
                    ledger.start_attempt()
                    adapter_stream = await self._bound_adapter.open_stream(request)
                    observations = observations._replace(opened=True)
                    try:
                        async for _ in adapter_stream.items():
                            pass
                        raw = await adapter_stream.final()
                        observations = observations._replace(
                            request_id=adapter_stream.request_id()
                        )
                    except BaseException:
                        observations = observations._replace(
                            billing=adapter_stream.billing_reported(),
                            request_id=adapter_stream.request_id(),
                        )
                        ledger.note_billing_in_flight(observations.billing)
                        raise
                    finally:
                        await _close_stream_quietly(
                            adapter_stream,
                            failure_log_message=(
                                "closing the provider stream raised; the attempt's outcome stands"
                            ),
                        )
                    outcome = self._staged_interpretation(
                        raw, request_id=observations.request_id, ledger=ledger
                    )
                    if outcome.kind == "provider_failed_transiently":
                        # Raised inside the block so the exit parses it: a billable 200 whose
                        # body reports a transient failure is still a provider failure, and a
                        # rate-limit body pauses the rate-limit quota exactly as a 429 status does.
                        assistant_message = outcome.assistant_message
                        raise TransientError(  # noqa: TRY301 (the admitted() block's exit is the parser, so the raise must sit inside it)
                            outcome.reason, is_rate_limit=outcome.is_rate_limit
                        )
            except ParserContractError:
                # A parse contract violation is langchaint's defect, not the attempt's failure:
                # it skips the verdict handling below and reaches the caller inside EscapedExceptionError.
                raise
            except self.shared_backoff.failure_types as exc:
                await self._pace_after_verdict(
                    exc,
                    admission=admission,
                    private_backoff=private_backoff,
                    assistant_message=assistant_message,
                    ledger=ledger,
                    request=request,
                    observations=observations,
                )
            except Exception as exc:
                terminal = await self._pace_after_transport_failure(
                    exc,
                    private_backoff=private_backoff,
                    ledger=ledger,
                    request=request,
                    observations=observations,
                )
                if terminal is not None:
                    raise terminal from exc
            else:
                # error is None on every variant reaching here: the request succeeded, and what the
                # adapter made of the response is the item's outcome, not this attempt's failure.
                ledger.record(error=None, assistant_message=outcome.assistant_message)
                match outcome.kind:
                    case "adapter_result":
                        return _success_variant(
                            splits_tool_call_turns=self._splits_tool_call_turns,
                            output=outcome.output,
                            call=ledger.freeze(),
                            raw=raw,
                            stop_reason=outcome.stop_reason,
                            assistant_message=outcome.assistant_message,
                        )
                    case "refusal":
                        raise RefusalError(call=ledger.freeze(), request=request)
                    case "max_completion_tokens_exceeded":
                        raise MaxCompletionTokensExceededError(
                            call=ledger.freeze(), request=request
                        )
                    case "empty_turn":
                        raise EmptyTurnError(call=ledger.freeze(), request=request)
                    case "schema_violation":
                        raise SchemaViolationError(
                            validation_error_json=outcome.validation_error_json,
                            call=ledger.freeze(),
                            request=request,
                        )
                    case "context_window_exceeded":
                        raise ContextWindowExceededError(call=ledger.freeze(), request=request)
                    case "unfinished_turn":
                        raise UnfinishedTurnError(
                            reason=outcome.reason, call=ledger.freeze(), request=request
                        )
                    case "provider_failed_terminally":
                        raise ProviderFailedTerminallyError(
                            reason=outcome.reason, call=ledger.freeze(), request=request
                        )
        raise RetriesExhaustedError(call=ledger.freeze(), request=request)

    def _request_for_messages(
        self, messages: Sequence[Message], *, ledger: _CallLedger
    ) -> RequestParams:
        """Build one provider request.

        Raises:
            InvalidRequestError: The adapter rejects `messages` before any request.
        """
        built = self._bound_adapter.build_request(messages)
        if isinstance(built, InvalidRequest):
            raise InvalidRequestError(reason=built.reason, call=ledger.freeze(), request=None)
        return built

    @overload
    async def generate_one(
        self: "BoundLLM[str, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> Response[str]: ...
    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, ToolManager]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> GenerateResult[OutputT]: ...
    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> Response[OutputT]: ...
    async def generate_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> GenerateResult[Any]:
        """Generate one response with retries.

        A structured tool-bound binding can return `ToolCallTurn`.
        `timeout_seconds` bounds admission, requests, and backoff waits.
        Generation failures raise a `GenerationError` subclass.

        Raises:
            asyncio.CancelledError: The caller cancels this call.
        """
        return await self._generate_one_any_binding(
            generation_input, deadline=WallClockDeadline(timeout_seconds)
        )

    async def _generate_one_any_binding(
        self, generation_input: GenerationInput, *, deadline: Deadline
    ) -> GenerateResult[OutputT | None]:
        """Run one call at the widest output type and record escaped `Exception` values.

        Raises:
            GenerationError: Generation fails or an escaped `Exception` becomes `EscapedExceptionError`.
            BaseException: A non-`Exception` value interrupts the call.
        """
        ledger = _CallLedger(model=self.adapter.model, provider_name=self.adapter.provider_name)
        try:
            return await self._generate_with_retries(
                _as_messages(generation_input), ledger=ledger, deadline=deadline
            )
        except GenerationError:
            raise
        except Exception as escaped:
            raise EscapedExceptionError(error=escaped, call=ledger.freeze()) from escaped

    async def _generate_or_failure(
        self,
        generation_input: GenerationInput,
        *,
        generate_item: "GenerateItem[OutputT]",
        deadline: Deadline,
    ) -> CallResult[OutputT | None]:
        """One batch item: the success variant or the GenerationError.

        Every terminal item failure becomes `GenerationError` before reaching `run_many`.
        One item's expired deadline therefore cannot cancel a sibling.

        Raises:
            BaseException: whatever cut the item off, propagating unobserved.
        """
        try:
            return await generate_item(generation_input, deadline=deadline)
        except GenerationError as failure:
            return failure

    @overload
    async def generate_many(
        self: "BoundLLM[str, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_item: float | None = ...,
    ) -> list[Response[str] | GenerationError]: ...
    @overload
    async def generate_many(
        self: "BoundLLM[OutputT, ToolManager]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_item: float | None = ...,
    ) -> list[CallResult[OutputT]]: ...
    @overload
    async def generate_many(
        self: "BoundLLM[OutputT, None]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_item: float | None = ...,
    ) -> list[Response[OutputT] | GenerationError]: ...
    async def generate_many(
        self,
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = False,
        max_working_seconds_per_item: float | None = None,
        # list is invariant, so no single element union is assignable from all three overloads;
        # a union of list types would restate the overloads without replacing this Any.
    ) -> list[Any]:
        """Generate an input-aligned batch.

        Each `GenerationError` becomes that input's result and does not cancel sibling calls.
        `SharedBackoff.max_concurrent_requests` limits request starts and pending items.
        `warm_cache` completes the first input before starting the rest.
        `max_working_seconds_per_item` excludes admission and shared-pause waits.

        Raises:
            asyncio.CancelledError: The caller cancels the batch.
            BaseException: An item raises a non-`Exception` value.
        """
        return await self._generate_many_any_binding(
            generation_inputs,
            warm_cache=warm_cache,
            generate_item=self._generate_one_any_binding,
            max_working_seconds_per_item=max_working_seconds_per_item,
        )

    async def _generate_many_any_binding(
        self,
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool,
        generate_item: "GenerateItem[OutputT]",
        max_working_seconds_per_item: float | None,
    ) -> list[CallResult[OutputT | None]]:
        """Run a batch at the widest output type.

        Each item receives its own `WorkingTimeDeadline` when it starts.

        Raises:
            asyncio.CancelledError: The caller cancels the batch.
            BaseException: An item raises a non-`Exception` value.
        """
        # The slices also convert the SequenceNotStr protocol to the Sequence _run_items takes.
        if warm_cache and generation_inputs:
            first_result = await self._generate_or_failure(
                generation_inputs[0],
                generate_item=generate_item,
                deadline=WorkingTimeDeadline(max_working_seconds_per_item),
            )
            rest = await self._run_items(
                generation_inputs[1:],
                generate_item=generate_item,
                max_working_seconds_per_item=max_working_seconds_per_item,
            )
            return [first_result, *rest]
        return await self._run_items(
            generation_inputs[0:],
            generate_item=generate_item,
            max_working_seconds_per_item=max_working_seconds_per_item,
        )

    async def _run_items(
        self,
        generation_inputs: Sequence[GenerationInput],
        *,
        generate_item: "GenerateItem[OutputT]",
        max_working_seconds_per_item: float | None,
    ) -> list[CallResult[OutputT | None]]:
        """Run the items through run_many, which returns them in input order.

        The pending bound follows `SharedBackoff.max_concurrent_requests`.
        `run_many` therefore receives a positive integer.

        Raises:
            asyncio.CancelledError: an outer scope cancelled generate_many.
            BaseException: An item raised a `BaseException` outside `Exception`.
                `run_many` cancels and awaits started items before propagation.
        """

        async def run_one(
            generation_input: GenerationInput,
        ) -> CallResult[OutputT | None]:
            """Run one batch item under a deadline of its own.

            Raises:
                BaseException: _generate_or_failure propagated it.
            """
            return await self._generate_or_failure(
                generation_input,
                generate_item=generate_item,
                deadline=WorkingTimeDeadline(max_working_seconds_per_item),
            )

        run_ones = tuple(
            partial(run_one, generation_input) for generation_input in generation_inputs
        )
        max_pending = max_pending_for_requests(self.shared_backoff.max_concurrent_requests)
        return await run_many(run_ones, max_pending=max_pending)

    @overload
    def stream_one(
        self: "BoundLLM[str, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> StreamHandle[str]: ...
    @overload
    def stream_one(
        self: "BoundLLM[OutputT, ToolManager]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> StreamHandle[OutputT, ToolCallTurn[OutputT]]: ...
    @overload
    def stream_one(
        self: "BoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> StreamHandle[OutputT]: ...
    def stream_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> StreamHandle[Any, Any]:
        """Build a `StreamHandle` that opens on context-manager entry.

        `timeout_seconds` starts on entry and covers request opening, iteration, and caller work before conclusion.
        """
        return self._stream_one_any_binding(generation_input, timeout_seconds=timeout_seconds)

    def _stream_one_any_binding(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None
    ) -> StreamHandle[OutputT | None, ToolCallTurn[OutputT | None]]:
        """Build the handle at the widest output type; _generate_one_any_binding says why."""
        return StreamHandle(
            adapter=self.adapter,
            bound_adapter=self._bound_adapter,
            messages=_as_messages(generation_input),
            shared_backoff=self.shared_backoff,
            max_attempts=self.max_attempts,
            timeout_seconds=timeout_seconds,
            splits_tool_call_turns=self._splits_tool_call_turns,
        )
