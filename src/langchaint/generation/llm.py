"""Provider-neutral LLM binding, request admission, retries, and batch generation."""

import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Final, NamedTuple, Protocol, overload

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    Binding,
    BoundAdapter,
    RejectedMessages,
    RequestParams,
    ResponseOutcome,
    ToolChoice,
)
from langchaint.billing.pricing import ProviderBilling
from langchaint.common.exceptions import TransientError
from langchaint.common.messages import AssistantMessage, Message, TextPart, UserMessage
from langchaint.common.observed_operation import (
    UNOBSERVED_OPERATION,
    ObservedOperation,
    start_observed_operation,
)
from langchaint.common.request_failure import RequestFailure, _request_failure_of
from langchaint.common.sequence_not_str import SequenceNotStr
from langchaint.concurrency.run_many import max_pending_for_requests, run_many
from langchaint.concurrency.shared_backoff import PrivateBackoff, SharedBackoff
from langchaint.generation._config_fingerprint import (
    ResponseFormatFingerprintData,
    bound_llm_config_fingerprint,
    capture_response_format_fingerprint_data,
    input_fingerprint,
)
from langchaint.generation._generate_many_records import (
    _run_resume_io,
    claim_resume_path,
    prepare_resume_state,
)
from langchaint.generation.errors import (
    GenerationError,
    GenerationErrorRecord,
    PlainErrorRecord,
    _terminal_generation_error,
    _transient_error_for,
)
from langchaint.generation.observer import GenerationStart, Observer
from langchaint.generation.request_history import AbandonedStreamRecord, _RequestLedger
from langchaint.generation.response import (
    Generation,
    GenerationOutcome,
    GenerationOutcomeRecord,
    PlainGeneration,
    PlainGenerationRecord,
    ToolCallGeneration,
    _escaped_error,
    _generation_outcome_from_response_outcome,
    _generation_outcome_record,
    _timed_out_error,
)
from langchaint.generation.streaming import StreamHandle, _close_stream_quietly
from langchaint.tools import ToolManager, ToolSchema, ToolSequence


class _StreamObservations(NamedTuple):
    provider_billing: ProviderBilling | None
    request_id: str | None
    opened: bool


class Unchanged:
    """Sentinel type for an omitted `bind` keyword."""

    def __repr__(self) -> str:
        """Render the default as `UNCHANGED` in signatures and `help()` output."""
        return "UNCHANGED"


UNCHANGED: Unchanged = Unchanged()


def _resolved_replacement[T](replacement: T | Unchanged, current: T) -> T:
    if isinstance(replacement, Unchanged):
        return current
    return replacement


type GenerationInput = str | Sequence[Message]
"""A bare `str` is shorthand for one `UserMessage`."""


class Deadline(Protocol):
    """Control whether request admission waits consume an input's deadline.

    Admission waits include the `SharedBackoff` permit and admission queue.
    """

    @property
    def scope(self) -> asyncio.Timeout:
        """The scope to enter around the retry loop, expiring when the input is out of time."""
        ...

    def suspend_until_admitted(self) -> None:
        """Pause the deadline before waiting for admission, if admission time is excluded."""
        ...

    def resume_on_admission(self) -> None:
        """Resume the deadline upon admission."""
        ...


class WallClockDeadline:
    """Run the deadline from construction through the outcome, including admission waits."""

    def __init__(self, timeout_seconds: float | None) -> None:
        """Set the wall-clock budget.

        Args:
            timeout_seconds: The wall-clock budget in seconds, or `None`.
        """
        self.scope: asyncio.Timeout = asyncio.timeout(timeout_seconds)

    def suspend_until_admitted(self) -> None:
        """Keep the clock running."""

    def resume_on_admission(self) -> None:
        """Keep the clock running."""


class WorkingTimeDeadline:
    """Run the deadline from first admission, excluding subsequent admission waits."""

    def __init__(self, max_working_seconds: float | None) -> None:
        """Create a scope without expiration.

        Args:
            max_working_seconds: The working-time budget in seconds, or `None`.
        """
        self.scope: asyncio.Timeout = asyncio.timeout(None)
        self._seconds_left = max_working_seconds

    def suspend_until_admitted(self) -> None:
        """Preserve the remaining budget while admission waits."""
        if self._seconds_left is None:
            return
        expires_at = self.scope.when()
        if expires_at is not None:
            self._seconds_left = expires_at - asyncio.get_running_loop().time()
        self.scope.reschedule(None)

    def resume_on_admission(self) -> None:
        """Resume the remaining budget upon admission."""
        if self._seconds_left is None:
            return
        self.scope.reschedule(asyncio.get_running_loop().time() + self._seconds_left)


def _as_messages(generation_input: GenerationInput) -> Sequence[Message]:
    if isinstance(generation_input, str):
        return (UserMessage(content=generation_input),)
    return generation_input


def _snapshot_generation_input(generation_input: GenerationInput) -> tuple[Message, ...]:
    return tuple(message.model_copy(deep=True) for message in _as_messages(generation_input))


async def _run_many_with_warm_cache[OutputT](
    run_ones: tuple[Callable[[], Coroutine[object, object, OutputT]], ...],
    *,
    warm_cache: bool,
    max_pending: int,
) -> list[OutputT]:
    if not warm_cache or not run_ones:
        return await run_many(run_ones, max_pending=max_pending)
    first = await run_ones[0]()
    remaining = await run_many(run_ones[1:], max_pending=max_pending)
    return [first, *remaining]


def _build_binding(
    *,
    system_prompt: str | Sequence[TextPart] | None,
    tool_schemas: tuple[ToolSchema, ...],
    provider_executed_tools: Sequence[Mapping[str, object]],
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    max_completion_tokens: int | None,
    reasoning_level: str | None,
    temperature: float | None,
    automatic_cache_breakpoints: bool,
    extra_body: Mapping[str, object] | None,
) -> Binding:
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
        max_completion_tokens=max_completion_tokens,
        reasoning_level=reasoning_level,
        temperature=temperature,
        automatic_cache_breakpoints=automatic_cache_breakpoints,
        extra_body=extra_body,
    )


@overload
def _bind_adapter(
    adapter: Adapter, binding: Binding, response_format: type[str]
) -> BoundAdapter[str | None]: ...
@overload
def _bind_adapter[ModelT: BaseModel](
    adapter: Adapter, binding: Binding, response_format: type[ModelT]
) -> BoundAdapter[ModelT | None]: ...
@overload
def _bind_adapter(
    adapter: Adapter, binding: Binding, response_format: type[str] | type[BaseModel]
) -> BoundAdapter[str | None] | BoundAdapter[BaseModel | None]: ...
def _bind_adapter(
    adapter: Adapter, binding: Binding, response_format: type[str] | type[BaseModel]
) -> BoundAdapter[str | None] | BoundAdapter[BaseModel | None]:
    if response_format is str:
        return adapter.bind_text(binding)
    if not issubclass(response_format, BaseModel):
        raise TypeError(
            f"response_format must be str or a pydantic.BaseModel subclass, got {response_format!r}"
        )
    return adapter.bind_structured(binding, response_format)


def _resolve_tool_manager(
    tools: ToolManager | ToolSequence | None, *, observer: Observer | None
) -> ToolManager | None:
    if isinstance(tools, ToolManager) or tools is None:
        return tools
    return ToolManager(tools, observer=observer)


class LLM:
    """Hold the client state shared across bindings."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Store the adapter, request admission state, and observer.

        Share `shared_backoff` across every `LLM` using one rate-limit quota.

        Args:
            adapter: The provider SDK adapter.
            shared_backoff: The request admission state, or `None` to create one with default settings.
            observer: The observer that follows every input and tool dispatch, or `None` to follow none.
        """
        self.adapter: Adapter = adapter
        self.shared_backoff: SharedBackoff = (
            shared_backoff if shared_backoff is not None else SharedBackoff()
        )
        self.observer: Observer | None = observer

    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[ModelT],
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
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
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[ModelT, None]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[str, ToolManager]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> "BoundLLM[str, None]": ...
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tools: ToolManager | ToolSequence | None = None,
        provider_executed_tools: Sequence[Mapping[str, object]] = (),
        response_format: type[str] | type[BaseModel] = str,
        max_completion_tokens: int | None = None,
        reasoning_level: str | None = None,
        temperature: float | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_requests: int = 3,
        automatic_cache_breakpoints: bool | None = None,
    ) -> "BoundLLM[BaseModel | str, ToolManager | None]":
        """Freeze the prompt prefix and fix the output type.

        A `tools` sequence constructs `ToolManager` with this `LLM`'s observer.
        An existing `ToolManager` retains its identity and its own observer.

        Args:
            system_prompt: The bound system prompt, or `None`.
            tools: The application tools or an existing `ToolManager`.
            provider_executed_tools: The provider-shaped tool definitions executed by the provider.
            response_format: The pydantic model for structured output, or `str` for text.
                With `str`, `output` is the kept assistant message's final text.
                Without text, `PlainGeneration.output` is `""` and `ToolCallGeneration.output` is `None`.
                To treat an empty output as a failure, bind a `response_format` that rejects empty text.
            max_completion_tokens: The maximum generated tokens, or `None` to omit the limit.
            reasoning_level: The exact reasoning-level string sent to the provider, or `None`.
            temperature: The sampling temperature, or `None` for the provider default.
            tool_choice: The provider-neutral tool choice.
            parallel_tool_calls: Whether the provider may request parallel tool calls.
            extra_body: Additional provider wire-body fields, or `None`.
                Declare tools only through `tools`, because result types follow the bound tools.
            max_requests: The maximum number of requests sent for one input; i.e., max_requests = max_retries + 1.
            automatic_cache_breakpoints: The automatic cache setting, or `None` for the adapter default.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `AllowedToolsChoice.tool_names` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one.
            ValueError: The binding breaks a rule of the adapter.
                The docstring of the backend's `llm()` lists the rules.
            TypeError: The adapter does not support `tool_choice`.
            TypeError: `response_format` is a proper subclass of `str`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        tool_manager = _resolve_tool_manager(tools, observer=self.observer)
        binding = _build_binding(
            system_prompt=system_prompt,
            tool_schemas=() if tool_manager is None else tool_manager.schemas(),
            provider_executed_tools=provider_executed_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            max_completion_tokens=max_completion_tokens,
            reasoning_level=reasoning_level,
            temperature=temperature,
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
            max_requests=max_requests,
            observer=self.observer,
        )


class BoundLLM[OutputT: str | BaseModel, ToolManagerT: ToolManager | None = None]:
    """A frozen prompt prefix with generation and streaming methods.

    `OutputT` is the `response_format` type: `str`, or the pydantic model that validates structured output.
    Both type parameters are covariant.
    `BoundLLM[Answer, ToolManager]` satisfies `BoundLLM[BaseModel, ToolManager | None]`.
    With a `ToolManager`, tool calls produce `ToolCallGeneration` whose output is `None` when no output exists.
    `tool_manager` preserves the bound `ToolManager` for application dispatch.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT | None],
        response_format: type[OutputT],
        binding: Binding,
        tool_manager: ToolManagerT,
        shared_backoff: SharedBackoff,
        max_requests: int,
        observer: Observer | None,
    ) -> None:
        """Store the frozen pieces.

        Args:
            adapter: The provider SDK adapter.
            bound_adapter: `adapter` bound to `binding` for `response_format` output.
            response_format: The pydantic model for structured output, or `str` for text.
            binding: The frozen provider-neutral prompt prefix.
            tool_manager: The bound `ToolManager`, or `None`.
            shared_backoff: The request admission state.
            max_requests: The maximum number of requests sent for one input; i.e., max_requests = max_retries + 1.
            observer: The originating `LLM.observer`.

        Raises:
            ValueError: `max_requests` is a bool or below one.
        """
        if isinstance(max_requests, bool) or max_requests < 1:
            raise ValueError(f"max_requests must be a positive int, got {max_requests!r}")
        # Final attributes keep both type parameters covariant.
        self.adapter: Final[Adapter] = adapter
        self.binding: Final[Binding] = binding
        self.response_format: Final[type[OutputT]] = response_format
        self.tool_manager: Final[ToolManagerT] = tool_manager
        self.shared_backoff: Final[SharedBackoff] = shared_backoff
        self.max_requests: Final[int] = max_requests
        self.observer: Final[Observer | None] = observer
        self._bound_adapter: Final[BoundAdapter[OutputT | None]] = bound_adapter
        self._adapter_class: Final[type[Adapter]] = type(adapter)
        self._adapter_model: Final[str] = adapter.model
        self._adapter_provider_name: Final[str] = adapter.provider_name
        self._adapter_config_fingerprint_data: Final[Mapping[str, object]] = (
            adapter.config_fingerprint_data()
        )
        self._response_format_fingerprint_data: Final[ResponseFormatFingerprintData] = (
            capture_response_format_fingerprint_data(response_format)
        )

    def config_fingerprint(self) -> str:
        """Return a SHA-256 fingerprint of the current stored request configuration.

        `BoundLLM` captures adapter and response-format configuration during binding.
        Each call reads current `Binding` values.
        It excludes per-input messages, retry and admission configuration, pricing, and credentials.
        It also excludes SDK client state and tool functions.
        It identifies stored configuration without establishing semantic or provider-wire equivalence.

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
    def _splits_on_tool_calls(self) -> bool:
        return self.tool_manager is not None

    @overload
    def bind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolManager]": ...
    @overload
    def bind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, None]": ...
    @overload
    def bind[NewModelT: BaseModel](
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: type[NewModelT],
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolManagerT]": ...
    @overload
    def bind(
        self,
        *,
        response_format: type[str],
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[str, ToolManager]": ...
    @overload
    def bind(
        self,
        *,
        response_format: type[str],
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[str, None]": ...
    @overload
    def bind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: type[str],
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[str, ToolManagerT]": ...
    @overload
    def bind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManager]": ...
    @overload
    def bind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[OutputT, None]": ...
    @overload
    def bind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManagerT]": ...
    def bind(
        self,
        *,
        response_format: type[str] | type[BaseModel] | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tools: ToolManager | ToolSequence | None | Unchanged = UNCHANGED,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        max_completion_tokens: int | None | Unchanged = UNCHANGED,
        reasoning_level: str | None | Unchanged = UNCHANGED,
        temperature: float | None | Unchanged = UNCHANGED,
        extra_body: Mapping[str, object] | None | Unchanged = UNCHANGED,
        max_requests: int | Unchanged = UNCHANGED,
        automatic_cache_breakpoints: bool | None | Unchanged = UNCHANGED,
    ) -> "BoundLLM[BaseModel | str, ToolManager | None]":
        """Return a new `BoundLLM` with specified fields replaced.

        `tools=None` removes `ToolManager`.
        A `tools` sequence constructs `ToolManager` with this binding's observer.
        An existing `ToolManager` retains its identity and its own observer.
        `automatic_cache_breakpoints=None` reads `Adapter.automatic_cache_breakpoints_default`.

        Args:
            response_format: The replacement output model, `str` for text, or `UNCHANGED`.
            system_prompt: The replacement system prompt, `None`, or `UNCHANGED`.
            tools: The replacement tools, `None`, or `UNCHANGED`.
            provider_executed_tools: The replacement provider-shaped tools or `UNCHANGED`.
            tool_choice: The replacement tool choice or `UNCHANGED`.
            parallel_tool_calls: The replacement parallel-tool setting or `UNCHANGED`.
            max_completion_tokens: The replacement token limit, `None` to omit the limit, or `UNCHANGED`.
            reasoning_level: The replacement exact provider string, `None`, or `UNCHANGED`.
            temperature: The replacement sampling temperature, `None`, or `UNCHANGED`.
            extra_body: The replacement provider wire-body fields, `None`, or `UNCHANGED`.
                Declare tools only through `tools`, because result types follow the bound tools.
            max_requests: The replacement request limit or `UNCHANGED`.
            automatic_cache_breakpoints: The replacement automatic cache setting or `UNCHANGED`.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `AllowedToolsChoice.tool_names` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one.
            ValueError: The binding breaks a rule of the adapter.
                The docstring of the backend's `llm()` lists the rules.
            TypeError: The adapter does not support `tool_choice`.
            TypeError: `response_format` is a proper subclass of `str`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        if isinstance(tools, Unchanged):
            tool_manager = self.tool_manager
            tool_schemas = self.binding.tool_schemas
        else:
            tool_manager = _resolve_tool_manager(tools, observer=self.observer)
            tool_schemas = () if tool_manager is None else tool_manager.schemas()
        resolved_automatic_cache_breakpoints = _resolved_replacement(
            automatic_cache_breakpoints, self.binding.automatic_cache_breakpoints
        )
        binding = _build_binding(
            system_prompt=_resolved_replacement(system_prompt, self.binding.system_prompt),
            tool_schemas=tool_schemas,
            provider_executed_tools=_resolved_replacement(
                provider_executed_tools, self.binding.provider_executed_tools
            ),
            tool_choice=_resolved_replacement(tool_choice, self.binding.tool_choice),
            parallel_tool_calls=_resolved_replacement(
                parallel_tool_calls, self.binding.parallel_tool_calls
            ),
            max_completion_tokens=_resolved_replacement(
                max_completion_tokens, self.binding.max_completion_tokens
            ),
            reasoning_level=_resolved_replacement(reasoning_level, self.binding.reasoning_level),
            temperature=_resolved_replacement(temperature, self.binding.temperature),
            automatic_cache_breakpoints=(
                self.adapter.automatic_cache_breakpoints_default
                if resolved_automatic_cache_breakpoints is None
                else resolved_automatic_cache_breakpoints
            ),
            extra_body=_resolved_replacement(extra_body, self.binding.extra_body),
        )
        resolved_response_format = _resolved_replacement(response_format, self.response_format)
        max_requests = _resolved_replacement(max_requests, self.max_requests)
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, binding, resolved_response_format),
            response_format=resolved_response_format,
            binding=binding,
            tool_manager=tool_manager,
            shared_backoff=self.shared_backoff,
            max_requests=max_requests,
            observer=self.observer,
        )

    def _request_id_for_failure(
        self, exc: Exception, observations: _StreamObservations
    ) -> str | None:
        request_id = self.adapter.request_id_from_error(exc)
        return request_id if request_id is not None else observations.request_id

    async def _settle_failed_request(
        self,
        exc: Exception,
        *,
        request_failure: RequestFailure | None,
        private_backoff: PrivateBackoff,
        assistant_message: AssistantMessage | None,
        ledger: _RequestLedger,
        request_params: RequestParams,
        observations: _StreamObservations,
    ) -> None:
        """Record a failed request and wait before the next one, as its `RequestFailure` decides.

        `request_failure` is `None` when the adapter's `request_failure` raised `exc`.
        A transient failure that pauses the quota relies on the next `admitted()` wait.
        Another transient failure waits for `private_backoff`, only while `max_requests` permits another request.

        Raises:
            GenerationError: The failure is not transient.
            Exception: `exc`, when `request_failure` is `None`.
        """
        ledger.note_request_id(self._request_id_for_failure(exc, observations))
        if request_failure is None:
            raise exc
        if request_failure.kind != "transient":
            raise _terminal_generation_error(
                request_failure.kind,
                error_text=str(exc),
                ledger=ledger,
                provider_billing=observations.provider_billing,
                request_params=request_params,
                stream_opened=observations.opened,
            ) from exc
        ledger.record(
            error=_transient_error_for(exc, str(exc), request_failure),
            assistant_message=assistant_message,
            provider_billing=observations.provider_billing,
        )
        if not request_failure.pauses_quota and ledger.request_count < self.max_requests:
            await asyncio.sleep(
                private_backoff.next_wait_seconds(request_failure.retry_after_seconds)
            )

    def _staged_interpretation(
        self, raw: BaseModel, *, request_id: str | None, ledger: _RequestLedger
    ) -> ResponseOutcome[OutputT | None]:
        ledger.stage_response(
            raw=raw,
            provider_billing=self._bound_adapter.billing_from_raw(raw),
            identity=self._bound_adapter.identity_from_raw(raw, request_id=request_id),
        )
        return self._bound_adapter.interpret(raw)

    async def _generate_with_retries(
        self,
        messages: Sequence[Message],
        *,
        ledger: _RequestLedger,
        deadline: Deadline,
    ) -> Generation[OutputT]:
        timeout_scope = deadline.scope
        try:
            async with timeout_scope:
                return await self._request_until_budget_runs_out(
                    messages, ledger=ledger, deadline=deadline
                )
        except TimeoutError:
            if not timeout_scope.expired():
                raise
            # The ledger retains billing that the interrupted stream reported.
            # A settled request record clears this value to `None`.
            raise _timed_out_error(ledger, ledger.provider_billing_in_flight) from None

    async def _request_until_budget_runs_out(
        self, messages: Sequence[Message], *, ledger: _RequestLedger, deadline: Deadline
    ) -> Generation[OutputT]:
        request_params = self._request_params_for_messages(messages, ledger=ledger)
        private_backoff = PrivateBackoff(self.shared_backoff)
        last_failure: Exception | None = None
        while ledger.request_count < self.max_requests:
            deadline.suspend_until_admitted()
            admission = self.shared_backoff.admitted()
            assistant_message: AssistantMessage | None = None
            observations = _StreamObservations(
                provider_billing=None, request_id=None, opened=False
            )
            request_failure: RequestFailure | None = None
            try:
                async with admission:
                    try:
                        deadline.resume_on_admission()
                        ledger.start_request()
                        adapter_stream = await self._bound_adapter.open_stream(request_params)
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
                                provider_billing=adapter_stream.provider_billing(),
                                request_id=adapter_stream.request_id(),
                            )
                            ledger.note_provider_billing_in_flight(observations.provider_billing)
                            raise
                        finally:
                            await _close_stream_quietly(
                                adapter_stream,
                                failure_log_message=(
                                    "closing the provider stream raised; the request's outcome stands"
                                ),
                            )
                        outcome = self._staged_interpretation(
                            raw, request_id=observations.request_id, ledger=ledger
                        )
                        if outcome.kind == "provider_failed_transiently":
                            # Raise inside the try so the handler below records its `RequestFailure`.
                            # A billable 200 body can report a transient provider failure.
                            # A rate-limit body pauses the rate-limit quota like a 429 status.
                            assistant_message = outcome.assistant_message
                            raise TransientError(  # noqa: TRY301 (the handler below records it)
                                outcome.error_text, pauses_quota=outcome.pauses_quota
                            )
                    except Exception as error:
                        request_failure = admission.record(
                            _request_failure_of(error, self.adapter.request_failure)
                        )
                        raise
            except Exception as exc:  # noqa: BLE001 (_settle_failed_request raises every terminal failure)
                last_failure = exc
                await self._settle_failed_request(
                    exc,
                    request_failure=request_failure,
                    private_backoff=private_backoff,
                    assistant_message=assistant_message,
                    ledger=ledger,
                    request_params=request_params,
                    observations=observations,
                )
            else:
                ledger.record(error=None, assistant_message=outcome.assistant_message)
                generation_outcome = _generation_outcome_from_response_outcome(
                    outcome,
                    request_history=ledger.freeze(),
                    request_provider_data=ledger.request_provider_data,
                    request_params=request_params,
                    splits_on_tool_calls=self._splits_on_tool_calls,
                )
                if isinstance(generation_outcome, GenerationError):
                    raise generation_outcome
                return generation_outcome
        raise GenerationError(
            record=PlainErrorRecord(
                kind="retries_exhausted_error", request_history=ledger.freeze()
            ),
            request_params=request_params,
            request_provider_data=ledger.request_provider_data,
        ) from last_failure

    def _request_params_for_messages(
        self, messages: Sequence[Message], *, ledger: _RequestLedger
    ) -> RequestParams:
        built = self._bound_adapter.build_request_params(messages)
        if isinstance(built, RejectedMessages):
            raise GenerationError(
                record=PlainErrorRecord(
                    kind="rejected_error",
                    error_text=built.error_text,
                    request_history=ledger.freeze(),
                ),
                request_params=None,
                request_provider_data=ledger.request_provider_data,
            )
        return built

    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> PlainGeneration[OutputT]: ...
    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> Generation[OutputT]: ...
    async def generate_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> Generation[OutputT]:
        """Return the `Generation` for one input, retrying transient failures while `max_requests` allows.

        `timeout_seconds` bounds admission, requests, and backoff waits.

        Args:
            generation_input: The text or messages to send.
            timeout_seconds: The wall-clock budget in seconds, or `None`.

        Raises:
            GenerationError: Generation fails.
            asyncio.CancelledError: The caller cancels this coroutine.
        """
        return await self._generate_one_any_binding(
            generation_input, deadline=WallClockDeadline(timeout_seconds)
        )

    async def _generate_one_any_binding(
        self, generation_input: GenerationInput, *, deadline: Deadline
    ) -> Generation[OutputT]:
        messages = _as_messages(generation_input)
        operation = self._generation_started(messages, stream=False)
        try:
            with operation.current():
                generation = await self._generate_with_escapes_recorded(
                    messages, deadline=deadline
                )
        except GenerationError as failure:
            operation.conclude(failure)
            raise
        else:
            operation.conclude(generation)
            return generation
        finally:
            operation.end()

    async def _generate_with_escapes_recorded(
        self, messages: Sequence[Message], *, deadline: Deadline
    ) -> Generation[OutputT]:
        ledger = _RequestLedger(model=self.adapter.model, provider_name=self.adapter.provider_name)
        try:
            return await self._generate_with_retries(messages, ledger=ledger, deadline=deadline)
        except GenerationError:
            raise
        except Exception as escaped:
            raise _escaped_error(ledger, escaped) from escaped

    def _generation_started(
        self, messages: Sequence[Message], *, stream: bool
    ) -> ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord]:
        if self.observer is None:
            return UNOBSERVED_OPERATION
        start = GenerationStart(
            provider_name=self.adapter.provider_name,
            model=self.adapter.model,
            binding=self.binding,
            response_format=self.response_format,
            messages=messages,
            stream=stream,
        )
        return start_observed_operation(partial(self.observer.generation_started, start))

    async def _generate_one_or_failure(
        self, generation_input: GenerationInput, *, deadline: Deadline
    ) -> GenerationOutcome[OutputT]:
        try:
            return await self._generate_one_any_binding(generation_input, deadline=deadline)
        except GenerationError as failure:
            return failure

    @overload
    async def generate_many[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, None]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_input: float | None = ...,
    ) -> list[PlainGeneration[CallOutputT]]: ...
    @overload
    async def generate_many[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManager]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_input: float | None = ...,
    ) -> list[Generation[CallOutputT]]: ...
    @overload
    async def generate_many[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        max_working_seconds_per_input: float | None = ...,
    ) -> list[PlainGeneration[CallOutputT]] | list[Generation[CallOutputT]]: ...
    async def generate_many[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = False,
        max_working_seconds_per_input: float | None = None,
    ) -> list[PlainGeneration[CallOutputT]] | list[Generation[CallOutputT]]:
        """Generate an input-aligned batch, raising on failure.

        Each input retries transient failures as `generate_one` does.
        A failure cancels and awaits unfinished inputs before propagating.
        Failures observed together propagate by the lowest input index.
        `SharedBackoff.max_concurrent_requests` limits request starts and pending items.
        `max_working_seconds_per_input` excludes admission and shared-pause waits.

        Args:
            generation_inputs: The input-aligned text or message values.
            warm_cache: Whether to finish the first input before starting the remaining inputs.
            max_working_seconds_per_input: The per-input working-time budget, or `None`.

        Raises:
            GenerationError: Handling an input fails.
            asyncio.CancelledError: The caller cancels the batch.
            BaseException: An item raises a non-`Exception` value.
        """

        async def run_one(generation_input: GenerationInput) -> Generation[CallOutputT]:
            return await self._generate_one_any_binding(
                generation_input, deadline=WorkingTimeDeadline(max_working_seconds_per_input)
            )

        run_ones = tuple(
            partial(run_one, generation_input) for generation_input in generation_inputs
        )
        return await _run_many_with_warm_cache(
            run_ones,
            warm_cache=warm_cache,
            max_pending=max_pending_for_requests(self.shared_backoff.max_concurrent_requests),
        )

    # `CallOutputT` is the output type of the `BoundLLM` type the caller holds.
    # Typing the invariant `list` result with it keeps `OutputT` covariant.
    # That is sound only because the caller holds the only reference to the returned `list`.
    @overload
    async def generate_many_records[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, None]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        resume_path: Path | None = None,
        input_ids: SequenceNotStr[str] | None = ...,
        warm_cache: bool = ...,
        max_working_seconds_per_input: float | None = ...,
    ) -> list[PlainGenerationRecord[CallOutputT] | GenerationErrorRecord]: ...
    @overload
    async def generate_many_records[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManager]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        resume_path: Path | None = None,
        input_ids: SequenceNotStr[str] | None = ...,
        warm_cache: bool = ...,
        max_working_seconds_per_input: float | None = ...,
    ) -> list[GenerationOutcomeRecord[CallOutputT]]: ...
    @overload
    async def generate_many_records[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        resume_path: Path | None = None,
        input_ids: SequenceNotStr[str] | None = ...,
        warm_cache: bool = ...,
        max_working_seconds_per_input: float | None = ...,
    ) -> (
        list[PlainGenerationRecord[CallOutputT] | GenerationErrorRecord]
        | list[GenerationOutcomeRecord[CallOutputT]]
    ): ...
    async def generate_many_records[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        resume_path: Path | None = None,
        input_ids: SequenceNotStr[str] | None = None,
        warm_cache: bool = False,
        max_working_seconds_per_input: float | None = None,
    ) -> (
        list[PlainGenerationRecord[CallOutputT] | GenerationErrorRecord]
        | list[GenerationOutcomeRecord[CallOutputT]]
    ):
        """Return an input-aligned batch of records, including generation failures.

        With `resume_path=None`, generate every input without file I/O or fingerprinting.
        With `resume_path` provided, restore reusable records and generate the remaining records.

        Each input retries transient failures as `generate_one` does.
        Each `GenerationError` becomes that input's record and does not cancel sibling inputs.
        The JSON file stores one item per current input.
        It stores input fingerprints instead of `GenerationInput` values.
        It stores normalized records without live provider SDK objects.
        The returned list follows `generation_inputs` order.
        When `input_ids=None`, each input is identified by its position in the complete ordered input list.
        Any ordered input change then replaces a valid resume file.
        Providing `input_ids` identifies inputs by those strings.
        Both cases use one JSON format whose `identity_mode` records whether `input_ids` was provided.
        Adding, deleting, or reordering inputs preserves records for unchanged `input_ids` entries.
        Deleting an `input_ids` entry removes its saved item.
        Reusing an `input_ids` entry with a changed input generates that item again.
        Repeated equal inputs remain separate items by position or by distinct `input_ids` entries.
        Changing the binding or switching whether `input_ids` is provided replaces a valid resume file.
        The file's `config_fingerprint` is the value from `config_fingerprint()`.
        Changes excluded by `config_fingerprint()` do not replace the file.
        A malformed file or unsupported format raises before any provider request and remains unchanged.
        A file in the supported format holds the records this langchaint version would write.
        Those are the records for the binding and inputs that produced the file.
        That equality assumes the provider answers the same way both times.
        A langchaint version that writes different records for the same binding and inputs uses a new format.
        Missing records generate again.
        Error records with kind `retries_exhausted_error`, `timed_out_error`, or `auth_error` generate again.
        The fingerprints exclude the causes of those errors, so the same input can succeed later.
        The new record replaces the saved error record.
        The replacement record excludes the earlier requests and billing for that input.
        Every other saved record is reused.
        Each generated record is written with an atomic file replacement before its item finishes.
        A process failure before file replacement can cause a repeated request after restart.
        Separate processes must not use the same `resume_path` concurrently.

        Args:
            generation_inputs: The input-aligned text or message values.
            resume_path: The JSON file whose parent directory already exists, or `None` to disable persistence.
            input_ids: Stable unique strings aligned with `generation_inputs`, or `None` for position identity.
                Used only when `resume_path` is provided.
            warm_cache: Whether to finish the first input requiring generation before starting the remaining inputs.
            max_working_seconds_per_input: The per-input working-time budget, or `None`.

        Raises:
            ValueError: `resume_path` is provided and `input_ids` has the wrong length or contains a duplicate.
            ValueError: `resume_path` contains malformed data or an unsupported format.
            ValueError: A generated outcome record cannot be serialized as resume JSON.
            RuntimeError: Another `generate_many_records` call in this process is using `resume_path`.
            TypeError: `resume_path` is provided and the binding or an input has no deterministic fingerprint encoding.
            OSError: The resume file cannot be read, written, or replaced.
            asyncio.CancelledError: The caller cancels the batch after started items settle.
            BaseException: An item raises a non-`Exception` value after started items settle.
        """
        if resume_path is None:

            async def run_record(
                generation_input: GenerationInput,
            ) -> GenerationOutcomeRecord[CallOutputT]:
                generation_outcome = await self._generate_one_or_failure(
                    generation_input, deadline=WorkingTimeDeadline(max_working_seconds_per_input)
                )
                return _generation_outcome_record(generation_outcome)

            return await _run_many_with_warm_cache(
                tuple(
                    partial(run_record, generation_input) for generation_input in generation_inputs
                ),
                warm_cache=warm_cache,
                max_pending=max_pending_for_requests(self.shared_backoff.max_concurrent_requests),
            )

        input_snapshots = tuple(
            _snapshot_generation_input(generation_input) for generation_input in generation_inputs
        )
        input_fingerprints = tuple(
            input_fingerprint(input_snapshot) for input_snapshot in input_snapshots
        )
        input_ids = None if input_ids is None else tuple(input_ids)
        config_fingerprint = self.config_fingerprint()
        resume_path = await _run_resume_io(resume_path.resolve)
        with claim_resume_path(resume_path):
            resume_state = await _run_resume_io(
                partial(
                    prepare_resume_state,
                    resume_path=resume_path,
                    output_type=self.response_format,
                    config_fingerprint=config_fingerprint,
                    input_fingerprints=input_fingerprints,
                    input_ids=input_ids,
                )
            )
            pending_indices = resume_state.pending_indices()

            async def run_one(outcome_index: int) -> None:
                generation_outcome = await self._generate_one_or_failure(
                    input_snapshots[outcome_index],
                    deadline=WorkingTimeDeadline(max_working_seconds_per_input),
                )
                await _run_resume_io(
                    partial(
                        resume_state.store_outcome_record,
                        outcome_index,
                        _generation_outcome_record(generation_outcome),
                    )
                )

            _ = await _run_many_with_warm_cache(
                tuple(partial(run_one, outcome_index) for outcome_index in pending_indices),
                warm_cache=warm_cache,
                max_pending=max_pending_for_requests(self.shared_backoff.max_concurrent_requests),
            )
            return resume_state.outcome_records()

    # `CallOutputT` is the output type of the `BoundLLM` type the caller holds.
    # Typing the invariant `StreamHandle` result with it keeps `OutputT` covariant.
    # That is sound only because the caller holds the only reference to the returned `StreamHandle`.
    @overload
    def stream_one[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> StreamHandle[CallOutputT]: ...
    @overload
    def stream_one[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> StreamHandle[CallOutputT, ToolCallGeneration]: ...
    def stream_one[CallOutputT: str | BaseModel](
        self: "BoundLLM[CallOutputT, ToolManagerT]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = None,
    ) -> StreamHandle[CallOutputT, ToolCallGeneration]:
        """Build a `StreamHandle` that opens on context-manager entry.

        `timeout_seconds` starts on entry and covers request opening, iteration, and caller work.
        It ends when the stream stores a `GenerationOutcome` or the block exits.

        Args:
            generation_input: The text or messages to send.
            timeout_seconds: The wall-clock budget in seconds, or `None`.
        """
        messages = _as_messages(generation_input)
        return StreamHandle(
            adapter=self.adapter,
            bound_adapter=self._bound_adapter,
            messages=messages,
            shared_backoff=self.shared_backoff,
            max_requests=self.max_requests,
            timeout_seconds=timeout_seconds,
            splits_on_tool_calls=self._splits_on_tool_calls,
            generation_started=partial(self._generation_started, messages, stream=True),
        )
