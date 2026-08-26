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

from langchaint._config_fingerprint import (
    bound_llm_config_fingerprint,
    capture_response_format_fingerprint_data,
)
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
    ContextWindowExceededErrorRecord,
    EmptyTurnErrorRecord,
    EscapedExceptionErrorRecord,
    GenerationError,
    InvalidRequestErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    ParserContractError,
    ProviderDeclaredFinalErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    RefusalErrorRecord,
    RetriesExhaustedErrorRecord,
    SchemaViolationErrorRecord,
    StreamProtocolError,
    TimedOutErrorRecord,
    TransientError,
    UnfinishedTurnErrorRecord,
    UnknownExceptionErrorRecord,
)
from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, TextPart, UserMessage
from langchaint.pricing import ProviderBilling
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
from langchaint.tools import Tool, ToolManager, ToolSchema


class _StreamObservations(NamedTuple):
    billing: ProviderBilling | None
    request_id: str | None
    opened: bool


class Unchanged:
    """Sentinel type for an omitted `rebind` keyword."""

    def __repr__(self) -> str:
        """Render the default as `UNCHANGED` in signatures and `help()` output."""
        return "UNCHANGED"


UNCHANGED: Unchanged = Unchanged()


type GenerationInput = str | Sequence[Message]
"""A bare `str` is shorthand for one `UserMessage`."""


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

    `timeout_seconds` includes admission waits.
    """

    def __init__(self, timeout_seconds: float | None) -> None:
        """Create the scope.

        `None` disables expiration.

        Args:
            timeout_seconds: The wall-clock budget in seconds, or `None`.
        """
        self.scope: asyncio.Timeout = asyncio.timeout(timeout_seconds)

    def suspend_until_admitted(self) -> None:
        """Keep the clock running."""

    def resume_on_admission(self) -> None:
        """Keep the clock running."""


class WorkingTimeDeadline:
    """A deadline that stops while the call waits to be admitted and runs the rest of the time.

    `max_working_seconds_per_item` selects this deadline.
    """

    def __init__(self, max_working_seconds: float | None) -> None:
        """Create a scope without expiration.

        Args:
            max_working_seconds: The working-time budget in seconds, or `None`.
        """
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
    tool_schemas: tuple[ToolSchema, ...],
    provider_executed_tools: Sequence[Mapping[str, object]],
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    inference_params: InferenceParams,
    automatic_cache_breakpoints: bool,
    extra_body: Mapping[str, object] | None,
) -> Binding:
    """Convert bind arguments to the frozen Binding.

    Raises:
        ValueError: system_prompt is an empty sequence of parts; pass None to bind no system prompt.
        ValueError: `tool_choice` contains a name absent from `tool_schemas`.
    """
    if system_prompt is not None and not isinstance(system_prompt, str):
        if not system_prompt:
            raise ValueError(
                "system_prompt is an empty sequence of parts; pass None to bind no system prompt"
            )
        system_prompt = tuple(system_prompt)
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
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
    """Hold the client state shared across bindings."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        shared_backoff: SharedBackoff | None = None,
    ) -> None:
        """Store the shared pieces.

        `shared_backoff=None` uses `max_concurrent_requests=8` and other `SharedBackoff` defaults.
        Pass one instance to every `LLM` sharing a rate-limit quota.

        Args:
            adapter: The provider SDK adapter.
            shared_backoff: The request admission state, or `None` to create one.
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

        A `tools` sequence constructs `ToolManager`; an existing `ToolManager` retains its identity.
        `max_attempts` counts requests including the first.

        Args:
            system_prompt: The bound system prompt, or `None`.
            tools: The application tools or an existing `ToolManager`.
            provider_executed_tools: The provider-shaped tool definitions executed by the provider.
            response_format: The pydantic model for structured output, or `None` for text.
            inference_params: The inference parameters, or `None` for defaults.
            tool_choice: The provider-neutral tool choice.
            parallel_tool_calls: Whether the provider may request parallel tool calls.
            extra_body: Additional provider wire-body fields, or `None`.
            max_attempts: The maximum requests for one generation call.
            automatic_cache_breakpoints: The automatic cache setting, or `None` for the adapter default.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `tool_choice` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `automatic_cache_breakpoints` is unsupported.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_attempts` is boolean or below one.
            TypeError: The adapter does not support `tool_choice`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        tool_manager = _resolve_tool_manager(tools)
        binding = _build_binding(
            system_prompt=system_prompt,
            tool_schemas=() if tool_manager is None else tool_manager.schemas(),
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
        """Store the frozen pieces.

        Args:
            adapter: The provider SDK adapter.
            bound_adapter: The adapter bound to the prompt prefix.
            response_format: The pydantic model for structured output, or `None` for text.
            binding: The frozen provider-neutral prompt prefix.
            tool_manager: The bound `ToolManager`, or `None`.
            shared_backoff: The request admission state.
            max_attempts: The maximum requests for one generation call.

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
        self._adapter_class = type(adapter)
        self._adapter_model = adapter.model
        self._adapter_provider_name = adapter.provider_name
        self._adapter_config_fingerprint_data = adapter.config_fingerprint_data()
        self._response_format_fingerprint_data = capture_response_format_fingerprint_data(
            response_format
        )

    @property
    def tool_manager(self) -> ToolManagerT:
        """Return the bound `ToolManager` or `None`."""
        return self._tool_manager

    def config_fingerprint(self) -> str:
        """Return a versioned SHA-256 fingerprint of the current stored request configuration.

        The fingerprint captures adapter and response-format configuration during binding.
        The fingerprint includes the binding.
        The fingerprint excludes per-call messages, retry configuration, and admission configuration.
        The fingerprint excludes pricing, credentials, SDK client state, and tool functions.
        It identifies stored configuration, not semantic or provider-wire equivalence.
        Each call reads current values referenced by `Binding`.

        Mapping insertion order does not affect the fingerprint. Sequence order and container types do.
        Class identity uses `__module__` and `__qualname__`.
        Dynamically created classes that reuse both values require distinct serialized configuration.

        Raises:
            TypeError: A configuration value has no deterministic encoding or contains a cycle.
        """
        return bound_llm_config_fingerprint(
            adapter_class=self._adapter_class,
            adapter_model=self._adapter_model,
            adapter_provider_name=self._adapter_provider_name,
            adapter_config_fingerprint_data=self._adapter_config_fingerprint_data,
            binding=self.binding,
            response_format_fingerprint_data=self._response_format_fingerprint_data,
        )

    @property
    def _splits_tool_call_turns(self) -> bool:
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

        `tools=None` removes `ToolManager`; an existing `ToolManager` retains its identity.
        `automatic_cache_breakpoints=None` reads `Adapter.automatic_cache_breakpoints_default`.

        Args:
            response_format: The replacement output model, `None` for text, or `UNCHANGED`.
            system_prompt: The replacement system prompt, `None`, or `UNCHANGED`.
            tools: The replacement tools, `None`, or `UNCHANGED`.
            provider_executed_tools: The replacement provider-shaped tools or `UNCHANGED`.
            tool_choice: The replacement tool choice or `UNCHANGED`.
            parallel_tool_calls: The replacement parallel-tool setting or `UNCHANGED`.
            inference_params: The replacement inference parameters or `UNCHANGED`.
            extra_body: The replacement provider wire-body fields, `None`, or `UNCHANGED`.
            max_attempts: The replacement request limit or `UNCHANGED`.
            automatic_cache_breakpoints: The replacement automatic cache setting or `UNCHANGED`.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `tool_choice` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `automatic_cache_breakpoints` is unsupported.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_attempts` is boolean or below one.
            TypeError: The adapter does not support `tool_choice`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        if isinstance(tools, Unchanged):
            tool_manager = self.tool_manager
            tool_schemas = self.binding.tool_schemas
        else:
            tool_manager = _resolve_tool_manager(tools)
            tool_schemas = () if tool_manager is None else tool_manager.schemas()
        new_binding = _build_binding(
            system_prompt=(
                self.binding.system_prompt
                if isinstance(system_prompt, Unchanged)
                else system_prompt
            ),
            tool_schemas=tool_schemas,
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
        """Convert a terminal verdict or `classify` result to `GenerationError`."""
        classification: ErrorClassification = (
            "declared_final"
            if isinstance(verdict, PauseAllDoNotRetry)
            else self.adapter.classify(exc)
        )
        if classification == "invalid_request":
            # `Adapter.classify` returns `invalid_request` only for a provider rejection.
            # The provider received this request, so the ledger records the attempt.
            ledger.record(error=None, assistant_message=None, billing=observations.billing)
            return GenerationError(
                record=InvalidRequestErrorRecord(
                    reason=f"the provider rejected the request: {exc}",
                    call=ledger.freeze(),
                ),
                request=request,
                provider_attempts=ledger.provider_attempts,
            )
        if classification == "declared_final":
            # The provider answered, so the attempt gets a record.
            ledger.record(error=None, assistant_message=None, billing=observations.billing)
            return GenerationError(
                record=ProviderDeclaredFinalErrorRecord(reason=str(exc), call=ledger.freeze()),
                request=request,
                provider_attempts=ledger.provider_attempts,
            )
        if observations.opened:
            ledger.record(error=None, assistant_message=None, billing=observations.billing)
        return GenerationError(
            record=UnknownExceptionErrorRecord(reason=str(exc), call=ledger.freeze()),
            request=request,
            provider_attempts=ledger.provider_attempts,
        )

    def _request_id_for_failure(
        self, exc: Exception, observations: _StreamObservations
    ) -> str | None:
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
        """Retry a transient transport failure or return its terminal error."""
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

        Raises:
            GenerationError: The adapter rejects the request.
                The provider declares a terminal failure.
                The adapter cannot classify an exception.
                The completed response reports a terminal result.
                Transient failures consume `max_attempts`.
                `deadline` expires.
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
            # The ledger retains billing that the interrupted stream reported.
            # A settled attempt record clears this value to `None`.
            raise _abandoned_call_error(
                TimedOutErrorRecord, ledger, ledger.billing_in_flight
            ) from None

    async def _attempt_until_budget_runs_out(
        self, messages: Sequence[Message], *, ledger: _CallLedger, deadline: Deadline
    ) -> GenerateResult[OutputT | None]:
        """Send requests until success, a terminal failure, or `max_attempts`.

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
                        # Raise inside the block so its exit parses the failure.
                        # A billable 200 body can report a transient provider failure.
                        # A rate-limit body pauses the rate-limit quota like a 429 status.
                        assistant_message = outcome.assistant_message
                        raise TransientError(  # noqa: TRY301 (the admitted() block's exit is the parser, so the raise must sit inside it)
                            outcome.reason, is_rate_limit=outcome.is_rate_limit
                        )
            except ParserContractError:
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
                ledger.record(error=None, assistant_message=outcome.assistant_message)
                match outcome.kind:
                    case "adapter_result":
                        return _success_variant(
                            splits_tool_call_turns=self._splits_tool_call_turns,
                            output=outcome.output,
                            call=ledger.freeze(),
                            provider_attempts=ledger.provider_attempts,
                            stop_reason=outcome.stop_reason,
                        )
                    case "refusal":
                        raise GenerationError(
                            record=RefusalErrorRecord(call=ledger.freeze()),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
                    case "max_completion_tokens_exceeded":
                        raise GenerationError(
                            record=MaxCompletionTokensExceededErrorRecord(call=ledger.freeze()),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
                    case "empty_turn":
                        raise GenerationError(
                            record=EmptyTurnErrorRecord(call=ledger.freeze()),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
                    case "schema_violation":
                        raise GenerationError(
                            record=SchemaViolationErrorRecord(
                                validation_error_json=outcome.validation_error_json,
                                call=ledger.freeze(),
                            ),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
                    case "context_window_exceeded":
                        raise GenerationError(
                            record=ContextWindowExceededErrorRecord(call=ledger.freeze()),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
                    case "unfinished_turn":
                        raise GenerationError(
                            record=UnfinishedTurnErrorRecord(
                                reason=outcome.reason, call=ledger.freeze()
                            ),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
                    case "provider_failed_terminally":
                        raise GenerationError(
                            record=ProviderFailedTerminallyErrorRecord(
                                reason=outcome.reason, call=ledger.freeze()
                            ),
                            request=request,
                            provider_attempts=ledger.provider_attempts,
                        )
        raise GenerationError(
            record=RetriesExhaustedErrorRecord(call=ledger.freeze()),
            request=request,
            provider_attempts=ledger.provider_attempts,
        )

    def _request_for_messages(
        self, messages: Sequence[Message], *, ledger: _CallLedger
    ) -> RequestParams:
        """Build one provider request.

        Raises:
            GenerationError: The adapter rejects `messages` before any request.
        """
        built = self._bound_adapter.build_request(messages)
        if isinstance(built, InvalidRequest):
            raise GenerationError(
                record=InvalidRequestErrorRecord(reason=built.reason, call=ledger.freeze()),
                request=None,
                provider_attempts=ledger.provider_attempts,
            )
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

        `timeout_seconds` bounds admission, requests, and backoff waits.

        Args:
            generation_input: The text or messages to send.
            timeout_seconds: The wall-clock budget in seconds, or `None`.

        Raises:
            GenerationError: Generation fails.
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
            GenerationError: Generation fails or an escaped `Exception` becomes `GenerationError`.
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
            raise GenerationError(
                record=EscapedExceptionErrorRecord(reason=str(escaped), call=ledger.freeze()),
                request=None,
                provider_attempts=ledger.provider_attempts,
            ) from escaped

    async def _generate_or_failure(
        self,
        generation_input: GenerationInput,
        *,
        generate_item: "GenerateItem[OutputT]",
        deadline: Deadline,
    ) -> CallResult[OutputT | None]:
        """Return one batch item as a success variant or `GenerationError`.

        Raises:
            BaseException: `generate_item` raises a value other than `GenerationError`.
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
        # `list` is invariant.
        # No single value union is assignable from each overload's list type.
        # A union of list types would restate the overloads without replacing this `Any`.
    ) -> list[Any]:
        """Generate an input-aligned batch.

        Each `GenerationError` becomes that input's result and does not cancel sibling calls.
        `SharedBackoff.max_concurrent_requests` limits request starts and pending items.
        `max_working_seconds_per_item` excludes admission and shared-pause waits.

        Args:
            generation_inputs: The input-aligned text or message values.
            warm_cache: Whether to finish the first input before starting the remaining inputs.
            max_working_seconds_per_item: The per-item working-time budget, or `None`.

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

        Raises:
            asyncio.CancelledError: The caller cancels the batch.
            BaseException: An item raises a non-`Exception` value.
        """
        # The slices convert `SequenceNotStr` to the `Sequence` that `_run_items` takes.
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

        Raises:
            asyncio.CancelledError: An outer scope cancels `generate_many` after started items settle.
            BaseException: An item raises a value outside `Exception` after started items settle.
        """

        async def run_one(
            generation_input: GenerationInput,
        ) -> CallResult[OutputT | None]:
            """Run one batch item under a deadline of its own.

            Raises:
                BaseException: Generation raises a value other than `GenerationError`.
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

        Args:
            generation_input: The text or messages to send.
            timeout_seconds: The wall-clock budget in seconds, or `None`.
        """
        return self._stream_one_any_binding(generation_input, timeout_seconds=timeout_seconds)

    def _stream_one_any_binding(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None
    ) -> StreamHandle[OutputT | None, ToolCallTurn[OutputT | None]]:
        return StreamHandle(
            adapter=self.adapter,
            bound_adapter=self._bound_adapter,
            messages=_as_messages(generation_input),
            shared_backoff=self.shared_backoff,
            max_attempts=self.max_attempts,
            timeout_seconds=timeout_seconds,
            splits_tool_call_turns=self._splits_tool_call_turns,
        )
