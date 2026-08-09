"""The client; generation happens only through a binding.

LLM composes an adapter and a SharedBackoff.
LLM has no generate methods.
bind() freezes everything that determines the cacheable prompt prefix,
fixes the output type, and precomputes SDK keyword arguments once;
the returned BoundLLM takes only the per-request GenerationInput.
There are no per-call parameter overrides; changing parameters is rebind().
The SharedBackoff admitted() block gates every request start on every path, retries included,
and one block spans one attempt.
The retry loop raises each provider failure inside the block, so the exit parses and records it,
and acts on the verdict the block leaves on Admission.verdict:
a PauseAll holds every request in the domain at entry until the shared pause ends,
a RetryThisOne waits out a PrivateBackoff between blocks,
and a DoNotRetry becomes the item's GenerationError, named by Adapter.classify.
PauseAllDoNotRetry is both: the pause it recorded holds the domain, and this item stops as
ProviderDeclaredFinalError.
"""

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple, Protocol, SupportsIndex, overload

from pydantic import BaseModel

from langchaint.account_state import AccountClosedError, AccountState
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
from langchaint.run_many import run_many
from langchaint.shared_backoff import (
    Admission,
    DoNotRetry,
    PauseAllDoNotRetry,
    PrivateBackoff,
    SharedBackoff,
    Verdict,
)
from langchaint.streaming import StreamHandle, _close_stream_quietly
from langchaint.tools import ToolManager


class _StreamObservations(NamedTuple):
    """What the retry loop saw of one attempt's stream, read before the close dropped it.

    billing is what the provider had reported when a failure cut the stream off: None where the
    stream never opened, where it reported nothing, and where it concluded normally, a staged
    response then stating the attempt's billing itself. A counter the provider sends late is
    missing from it.
    request_id is the request-id header the stream carried, None where it had none or never opened.
    opened says whether open_stream returned, which is what lets a failure nobody can classify
    still record that the attempt reached the provider.
    """

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


UNCHANGED = Unchanged()


type GenerationInput = str | Sequence[Message]
"""What one request is generated from: a bare str is shorthand for a Sequence[Message] of one UserMessage."""


_PENDING_TASKS_PER_CONCURRENT_REQUEST = 2
"""Pending tasks generate_many holds for each request its SharedBackoff admits at once.

A task holds no permit while it sleeps between attempts, so permits stay fed only while the awake
tasks outnumber the permits. The share of tasks that may sleep at once without leaving a permit
idle is at least one minus the reciprocal of this ratio, so 2 tolerates half of them asleep.
A transient failure rate puts far fewer than half to sleep, so this ratio is a margin rather than
a measured rate.
"""

_SPARE_PENDING_TASKS = 8
"""Pending tasks generate_many holds on top of what _PENDING_TASKS_PER_CONCURRENT_REQUEST gives.

That ratio holds in the steady state, and a small max_concurrent_requests leaves too few tasks for
the sleeping count to vary within: at one permit the ratio gives one spare, whose sleep idles the
permit. A pending task costs a coroutine frame, so spares are cheap where the ratio is thin.
"""

_MAX_PENDING_TASKS_WITHOUT_A_CONCURRENCY_BOUND = 1000
"""The most pending tasks generate_many holds when max_concurrent_requests is None.

No permit gates a request start in that configuration, so this is what caps requests in flight.
"""


class Deadline(Protocol):
    """The scope one call runs inside, told when the call waits to be admitted and when it is.

    Waiting to be admitted is waiting behind everything else sharing the SharedBackoff, first for a
    permit and then for the admission queue. Whether that time counts against the call is the only
    thing the implementations disagree on.
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

    This is what generate_one's timeout_seconds asks for: a caller blocked on one call wants an
    answer or a failure within that many seconds, and a wait for admission is time it spent waiting.
    """

    def __init__(self, timeout_seconds: float | None) -> None:
        """Arm the scope now, or open one that never expires when timeout_seconds is None."""
        self.scope = asyncio.timeout(timeout_seconds)

    def suspend_until_admitted(self) -> None:
        """Keep the clock running."""

    def resume_on_admission(self) -> None:
        """Keep the clock running."""


class WorkingTimeDeadline:
    """A deadline that stops while the call waits to be admitted and runs the rest of the time.

    This is what generate_many's max_working_seconds_per_item asks for.
    """

    def __init__(self, max_working_seconds: float | None) -> None:
        """Open the scope unarmed; the first resume_on_admission arms it with the budget."""
        self.scope = asyncio.timeout(None)
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


class SequenceNotStr[T_co](Protocol):
    """A Sequence that a type checker rejects a bare str for.

    str satisfies Sequence[GenerationInput] (a str is a sequence of str),
    so a plain Sequence batch parameter statically accepts generate_many("hi"),
    which would run one request per character.
    This protocol structurally matches list and tuple but not str,
    because typeshed's str.__contains__ accepts only str while the protocol requires __contains__(value: object).
    Being covariant, it also accepts a caller's list[str] or list[list[UserMessage]],
    which the invariant list[GenerationInput] would reject.
    Same shape as openai._types.SequenceNotStr, originally from the useful_types library;
    index() and count() are omitted deliberately, matching it.
    If typeshed ever widens str.__contains__,
    the static rejection lapses and _reject_bare_str_batch remains the backstop.
    """

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[T_co]: ...
    def __contains__(self, value: object, /) -> bool:
        """Accept object, which str's str-only __contains__ cannot satisfy."""
        ...

    def __len__(self) -> int:
        """Match Sequence."""
        ...

    def __iter__(self) -> Iterator[T_co]:
        """Match Sequence."""
        ...

    def __reversed__(self) -> Iterator[T_co]:
        """Match Sequence."""
        ...


def _reject_bare_str_batch(generation_inputs: SequenceNotStr[GenerationInput]) -> None:
    """Reject a bare str passed as the whole batch.

    The SequenceNotStr parameter type makes the type checker reject a bare str;
    this runtime guard is the backstop for untyped callers.

    Raises:
        TypeError: generation_inputs is a bare str.
    """
    if isinstance(generation_inputs, str):
        raise TypeError(
            "generation_inputs is a bare str; wrap it in a list, or use generate_one"
            " for a single generation_input"
        )


def _as_messages(generation_input: GenerationInput) -> Sequence[Message]:
    if isinstance(generation_input, str):
        return (UserMessage(content=generation_input),)
    return generation_input


def _build_binding(
    *,
    system_prompt: str | Sequence[TextPart] | None,
    tool_manager: ToolManager | None,
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    inference_params: InferenceParams,
    automatic_prompt_caching: bool,
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
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        inference_params=inference_params,
        automatic_prompt_caching=automatic_prompt_caching,
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


class GenerateItem[OutputT](Protocol):
    """Runs one item of a batch.

    BoundLLM.generate_many passes its own _generate_one_any_binding. A wrapper passes an
    implementation that calls the same method and does its own work around it, which is how one call
    of a batch gets treated exactly as generate_one treats one call.
    Pass deadline through: an implementation that dropped it would silently give its items no
    deadline at all. It belongs to this item alone, so hand it to one call and no other.
    """

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

        shared_backoff None builds a private domain from the adapter's parse and failure_types,
        at the SharedBackoff defaults with max_concurrent_requests 8. One instance is one
        backpressure domain, so pass the same instance to every LLM whose requests share a
        provider quota.
        """
        self.adapter = adapter
        self.shared_backoff = (
            shared_backoff
            if shared_backoff is not None
            else SharedBackoff(
                parse=adapter.parse, failure_types=adapter.failure_types, max_concurrent_requests=8
            )
        )
        self._account_state: AccountState | None = None

    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: ToolManager,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_prompt_caching: bool,
    ) -> "BoundLLM[ModelT, ToolManager]": ...
    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: None = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_prompt_caching: bool,
    ) -> "BoundLLM[ModelT, None]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: ToolManager,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_prompt_caching: bool,
    ) -> "BoundLLM[str, ToolManager]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: None = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_attempts: int = ...,
        automatic_prompt_caching: bool,
    ) -> "BoundLLM[str, None]": ...
    def bind(  # noqa: PLR0913 (the binding states every choice: prompt, tools, format, params, caching, extra_body)
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tool_manager: ToolManager | None = None,
        response_format: type[BaseModel] | None = None,
        inference_params: InferenceParams | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_attempts: int = 3,
        automatic_prompt_caching: bool,
    ) -> "BoundLLM[Any, Any]":
        """Freeze the prompt prefix and fix the output type.

        response_format=Model gives BoundLLM[Model] whose output is a validated Model.
        Absent, bind gives BoundLLM[str] whose output is the assistant text.
        Passing a tool_manager gives the BoundLLM[Model, ToolManager] form, whose structured request
        methods type output as optional because a tool-call turn parses no instance; see BoundLLM.
        A caller holding a ToolManager | None gets the union of the two forms, whose request methods
        return the optional type, which is what a caller who does not know can act on.
        automatic_prompt_caching has no default: caching changes billing,
        so langchaint never chooses a caching configuration for the caller.
        max_attempts counts requests sent including the first, so 1 means no retrying.
        Ad-hoc use is llm.bind(automatic_prompt_caching=False).generate_one(...).
        Binding.extra_body documents extra_body: the merge precedence and the colliding-key raise.

        Raises:
            ValueError: system_prompt is an empty sequence of parts; pass None to bind no system
                prompt. Also raised by the adapter, which refuses an automatic_prompt_caching its
                model cannot honor and an extra_body key the adapter itself populates. Also raised
                when max_attempts is a bool or an int below 1.
        """
        binding = _build_binding(
            system_prompt=system_prompt,
            tool_manager=tool_manager,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            inference_params=(
                inference_params if inference_params is not None else InferenceParams()
            ),
            automatic_prompt_caching=automatic_prompt_caching,
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
            account_state=self._account_state,
        )


class BoundLLM[OutputT, ToolManagerT: ToolManager | None = None]:
    """One frozen prefix plus the request methods; constructed by LLM.bind.

    OutputT is what the binding asks the model for: str, or the response_format instance.
    ToolManagerT is the bound tool_manager's type, ToolManager or None.
    The tool_manager property returns it.
    A tool loop therefore dispatches through the binding it was handed.
    ToolManagerT is also what the request methods overload on.
    A structured BoundLLM[Model, ToolManager] generates GenerateResult[Model]: a tool-call turn is
    the ToolCallTurn variant, so the Response variant's output is never None.
    Every other combination generates Response[OutputT] alone.
    Keeping the variants out of OutputT is what lets rebind add and remove a tool_manager.
    The parameter defaults to None, so BoundLLM[Model] annotates the common binding.
    A tool-bound one names both, BoundLLM[Model, ToolManager].
    bind writes ToolManager as the type argument for every manager, subclasses included.

    tool_manager is kept for tool dispatch;
    the provider only ever sees the converted schemas inside the binding.
    """

    def __init__(  # noqa: PLR0913 (the binding carries every bound field and account state)
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT | None],
        response_format: type[OutputT] | None,
        binding: Binding,
        tool_manager: ToolManagerT,
        shared_backoff: SharedBackoff,
        max_attempts: int,
        account_state: AccountState | None,
    ) -> None:
        """Store the frozen pieces; called by `LLM.bind` and `rebind` only.

        Raises:
            ValueError: `max_attempts` is a bool or below one.
        """
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive int, got {max_attempts!r}")
        self.adapter = adapter
        self.binding = binding
        self.response_format = response_format
        self.shared_backoff = shared_backoff
        self.max_attempts = max_attempts
        self._account_state = account_state
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

    def _ensure_account_open(self) -> None:
        """Raise when this binding's account is closed.

        Raises:
            RuntimeError: This binding's account is closed.
        """
        if self._account_state is not None:
            self._account_state.ensure_open()

    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tool_manager: ToolManager,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolManager]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        tool_manager: None,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, None]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: type[NewModelT],
        tool_manager: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolManagerT]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        tool_manager: ToolManager,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[str, ToolManager]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        tool_manager: None,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[str, None]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: None,
        tool_manager: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[str, ToolManagerT]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tool_manager: ToolManager,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManager]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tool_manager: None,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, None]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tool_manager: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_attempts: int | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManagerT]": ...
    def rebind(  # noqa: PLR0913 (rebind takes every field bind takes, each replaceable alone)
        self,
        *,
        response_format: type[BaseModel] | None | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tool_manager: ToolManager | None | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        inference_params: InferenceParams | Unchanged = UNCHANGED,
        extra_body: Mapping[str, object] | None | Unchanged = UNCHANGED,
        max_attempts: int | Unchanged = UNCHANGED,
        automatic_prompt_caching: bool | Unchanged = UNCHANGED,
    ) -> "BoundLLM[Any, Any]":
        """Return a new BoundLLM with these fields replaced; a left-out field keeps its value.

        response_format and tool_manager are the two fields whose change alters the static output
        type, so they drive the overload return type: the first sets OutputT, the second sets
        ToolManagerT, and leaving either out keeps what this binding has. Every combination is exact, including
        dropping a tool_manager, which is what returns a structured binding to a non-optional output.
        Replace semantics: a passed inference_params replaces the bound one whole, never field-wise.
        Every rebind converts the binding to SDK keyword arguments again, a pure conversion with no I/O.
        Whether a rebind preserves the provider's prompt cache is provider-specific and partly undocumented
        (Anthropic documents the prefix order tools -> system -> messages),
        and it depends on which field a rebind changes and on which value that field moves between,
        so measure it on the deployment you ship on.
        langchaint owns no cache-safety matrix over this.
        A matrix carried in the code goes stale the moment a provider changes a model.

        Raises:
            ValueError: system_prompt is an empty sequence of parts; pass None to bind no system
                prompt. Also raised by the adapter, which refuses an automatic_prompt_caching its
                model cannot honor and an extra_body key the adapter itself populates. Also raised
                when max_attempts is a bool or an int below 1.
        """
        new_tool_manager = (
            self.tool_manager if isinstance(tool_manager, Unchanged) else tool_manager
        )
        new_binding = _build_binding(
            system_prompt=(
                self.binding.system_prompt
                if isinstance(system_prompt, Unchanged)
                else system_prompt
            ),
            tool_manager=new_tool_manager,
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
            automatic_prompt_caching=(
                self.binding.automatic_prompt_caching
                if isinstance(automatic_prompt_caching, Unchanged)
                else automatic_prompt_caching
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
            tool_manager=new_tool_manager,
            shared_backoff=self.shared_backoff,
            max_attempts=new_max_attempts,
            account_state=self._account_state,
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
        """Name this item's terminal failure from the verdict, else from classify's reading of exc.

        A PauseAllDoNotRetry is declared_final without consulting classify: only a provider
        directive that this request will not succeed produces one, which is what
        ProviderDeclaredFinalError names. Reading the status instead would call a throttled
        account's 429 a rejection of the request.
        Every other failure takes classify. Reached on a terminal verdict and on an exception
        outside failure_types that classify did not call transient, so the "transient" value cannot
        arrive; if a classify defect produces one anyway, it lands on the unknown_exception default
        with everything else out of place.
        Every record written here bills observations.billing, what the failure's stream had
        reported in flight, so a terminal failure's spend still reaches the caller; a staged
        response's own billing wins where one arrived.

        StreamHandle carries its own copy of this mapping; what the two retry loops share is the ledger in call.py.
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
        """Record one verdicted attempt, then wait whatever the verdict asks before the next.

        exc is a failure_types exception, so the admitted() block's exit parsed it and left the
        verdict on admission.verdict. A verdict of None is folded into the terminal branch: the
        exit parses every failure_types exception, so a None reaching here has no verdict to act
        on.
        The attempt's record carries a TransientError: exc itself when it is one, otherwise one
        wrapping exc with the verdict's capped retry_after. It bills observations.billing, what
        the attempt's stream had reported when the failure cut it off, so a retried attempt's
        spend reaches the caller; a staged response's own billing wins where one arrived.
        On RetryThisOne the wait is the PrivateBackoff's, floored by the verdict's retry_after;
        on PauseAll there is no wait of our own, because the next admitted() entry already holds
        until the shared pause ends. Neither waits after the last attempt.
        assistant_message is the turn a 200 the provider filled with a failure still carried, and
        None where the attempt received no response.

        Raises:
            GenerationError: the verdict is terminal; _terminal_error names which. A
                PauseAllDoNotRetry raises like any other terminal verdict, its pause already
                recorded by the block's exit, so the domain keeps waiting while this item stops.
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
        """Record one transport failure and wait, or return the terminal error for it.

        exc is outside failure_types, so it exited the admitted() block unparsed and unrecorded
        there. Two failures are retried here, as RetryThisOne with no retry_after and with no wait
        after the last attempt. One is a failure classify calls "transient", a transport failure
        that produced nothing parseable. The other is a StreamProtocolError: a stream the
        transport ended without its terminal event and without any provider-reported error, which
        classify cannot place because the class is langchaint's own. No item from the drained
        stream reached any caller, so a resend is safe, and a violation that persists ends as
        RetriesExhaustedError whose attempt records each carry this text.
        For anything else this returns the GenerationError _terminal_error names,
        and the caller raises it so the raise sits beside the except clause that caught exc.
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
            identity=self._bound_adapter.identity_from_raw(raw).with_request_id_fallback(
                request_id
            ),
        )
        return self._bound_adapter.interpret(raw)

    async def _generate_with_retries(
        self,
        messages: Sequence[Message],
        *,
        ledger: _CallLedger,
        deadline: Deadline,
    ) -> GenerateResult[OutputT | None]:
        """Run the retry loop every generate method shares, under the caller's deadline.

        ledger is the caller's own empty ledger (the retry budget counts its attempts), recorded
        into as each attempt settles. Every GenerationError and the success variant are built from
        ledger.freeze(), the one site a call's elapsed_seconds is computed.

        deadline bounds this whole loop, and which of the loop's waits spend it is the deadline's
        own question to answer. Expiring raises TimedOutError, whose docstring says why the scope
        has to sit in this frame.
        A cancellation from any scope but this one is the caller's own order and propagates
        untouched. expired() is what tells the two apart: a TimeoutError this scope did not raise
        came from under the loop unclassified, and re-raising it hands it to the same wrapping every
        other unclassified exception gets.

        The adapter reports one attempt as a ResponseOutcome variant and never as a GenerationError,
        so this loop matches the variant and constructs the item's GenerationError here, where the
        attempts and the timing are known.
        Each arrived response is staged on the ledger with its billing before anything is read off it,
        so an exception from that read still leaves the attempt and its billing on the record.
        Each attempt spans one admitted() block, held for the request only;
        backoff sleeps sit outside the block so a waiting task does not hold a permit.
        Each provider failure is raised inside the block, so the exit records its verdict before
        anyone else is admitted and a rate-limit error pauses the whole domain.
        Every attempt is timed onto an AttemptRecord whose bracket is the request only,
        excluding the admission wait and the backoff sleep,
        so a slow request is distinguishable from time spent rate limited.

        Raises:
            RuntimeError: This binding's account closes before a request starts.
            InvalidRequestError: build_request returned InvalidRequest, or the adapter classified
                an attempt's error as a rejection of the request; terminal for this item, without a retry.
            ProviderDeclaredFinalError: the adapter classified an attempt's error as one the provider
                declared final; terminal for this item, without a retry.
            UnknownExceptionError: the adapter could not place an attempt's exception;
                terminal for this item, without a retry.
            RefusalError: the adapter reported a Refusal attempt (no structured output: the model
                refused or a provider filter blocked the turn); terminal for this item, without a retry.
            MaxCompletionTokensExceededError: the adapter reported a MaxCompletionTokensExceeded attempt (the structured
                response hit the token cap); terminal for this item, without a retry.
            EmptyTurnError: the adapter reported an EmptyTurn attempt (the model finished and produced
                nothing); terminal for this item, without a retry.
            SchemaViolationError: the adapter reported a SchemaViolation attempt (the model finished
                and its text is not an instance of the bound response_format); terminal for this
                item, without a retry.
            ContextWindowExceededError: the adapter reported a ContextWindowExceeded attempt;
                terminal for this item, without a retry.
            UnfinishedTurnError: the adapter reported an UnfinishedTurn attempt (a 200 langchaint
                cannot continue); terminal for this item, without a retry.
            ProviderFailedTerminallyError: the adapter reported a ProviderFailedTerminally attempt
                (the 200's body reports that generating the response failed, for a reason a resend
                would hit again); terminal for this item, without a retry.
            RetriesExhaustedError: every attempt failed transiently and the budget ran out.
            TimedOutError: the deadline expired before the call produced a result.
            ParserContractError: the adapter's parse violated its contract on an attempt's failure.
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

    async def _attempt_until_budget_runs_out(  # noqa: PLR0912 (account closure bypasses provider error handling)
        self, messages: Sequence[Message], *, ledger: _CallLedger, deadline: Deadline
    ) -> GenerateResult[OutputT | None]:
        """Send the request until it succeeds, fails terminally, or the retry budget runs out.

        Runs inside the deadline opened by _generate_with_retries, its only caller.

        Each attempt opens one adapter stream and drains it privately: no item reaches any caller,
        which is what makes retrying a mid-stream failure safe, where stream_one, whose items do,
        never retries an open stream. The drain runs inside the attempt's admitted() block, so a
        mid-stream failure exits the block with its verdict exactly as an open failure does.
        A failure that cuts the stream off has its billing_reported() and request_id() read before
        the close drops them, into the _StreamObservations the failure handlers record from; the
        billing is also noted on the ledger, where the deadline account finds it if this frame
        unwinds instead.

        Raises:
            GenerationError: every failure _generate_with_retries names but TimedOutError, which its
                scope raises.
            ParserContractError: the adapter's parse violated its contract on an attempt's failure.
            RuntimeError: This binding's account closed before an admitted request started.
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
                    self._ensure_account_open()
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
                        # rate-limit body pauses the domain exactly as a 429 status does.
                        assistant_message = outcome.assistant_message
                        raise TransientError(  # noqa: TRY301 (the admitted() block's exit is the parser, so the raise must sit inside it)
                            outcome.reason, is_rate_limit=outcome.is_rate_limit
                        )
            except AccountClosedError:
                raise
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
        """Build one request after confirming its account remains open.

        Raises:
            RuntimeError: This binding's account is closed.
            InvalidRequestError: The adapter rejects `messages` before any request.
        """
        self._ensure_account_open()
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
        """Generate one response under the retry loop.

        A structured tool-bound binding returns GenerateResult[OutputT], and a match on its kind
        tells the final Response from the ToolCallTurn owing tool results.
        Every other binding returns Response alone, its output never None, a text turn's being "".
        Every non-success outcome propagates, all of them sharing the GenerationError base a caller
        can catch at once: RetriesExhaustedError on transient exhaustion, InvalidRequestError on a
        rejected request, ProviderDeclaredFinalError or UnknownExceptionError on an error the adapter
        placed as final or could not place at all, and one of RefusalError,
        MaxCompletionTokensExceededError, EmptyTurnError, SchemaViolationError,
        ContextWindowExceededError, UnfinishedTurnError, or ProviderFailedTerminallyError on a 200
        that produced no output; _generate_with_retries names the condition for each.
        EscapedExceptionError joins them on an Exception that escaped langchaint's own machinery,
        raised by the guard around that loop.

        timeout_seconds bounds the whole call, admission waits and backoff sleeps included, and expiring
        raises TimedOutError, which carries what the cut-off call spent. None is no deadline.
        A cancellation from anywhere else (a caller's own asyncio.timeout, a TaskGroup sibling
        failing, shutdown) cuts the call off and propagates, so this call's settled attempts are lost
        with the frame. Ask for the deadline here to keep that account.

        Raises:
            RuntimeError: This binding's account is closed.
            asyncio.CancelledError: an outer scope cancelled this call.
        """
        return await self._generate_one_any_binding(
            generation_input, deadline=WallClockDeadline(timeout_seconds)
        )

    async def _generate_one_any_binding(
        self, generation_input: GenerationInput, *, deadline: Deadline
    ) -> GenerateResult[OutputT | None]:
        """Run one call under a ledger of its own, reporting every Exception as its failure.

        What generate_one does, at the widest output type, callable from a frame whose binding is not
        statically concrete: generate_one's overloads are keyed on the binding, so they match no
        generic self. The tracing wrapper reaches the request through here, and so does every batch
        item, this being the GenerateItem generate_many passes.

        The GenerationError clause re-raises the failures the retry loop already reported, which the
        Exception clause below it would otherwise wrap a second time. TimedOutError is one of them,
        so a deadline is never rewrapped as an escaped exception.

        Raises:
            GenerationError: whatever _generate_with_retries failed the call with, or
                EscapedExceptionError wrapping any other Exception that reached here.
            RuntimeError: This binding's account is closed.
            BaseException: whatever cut the call off, propagating unobserved.
        """
        self._ensure_account_open()
        ledger = _CallLedger(model=self.adapter.model, provider_name=self.adapter.provider_name)
        try:
            return await self._generate_with_retries(
                _as_messages(generation_input), ledger=ledger, deadline=deadline
            )
        except AccountClosedError:
            raise
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

        Every terminal per-item outcome is a GenerationError, so nothing a request produces escapes
        into run_many and reaches a sibling. An expired deadline is one of them, so one item's
        deadline never cuts a sibling.

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
        """Run an order-aligned batch: result i belongs to generation_inputs[i].

        Each success uses generate_one's result type for this binding.
        A bare str batch is rejected: str satisfies the item Sequence type, so it would silently
        become one request per character.

        A GenerationError is returned in place of that item's result and never cancels a sibling.
        It names retries exhausted, a rejected request, an error langchaint does not retry, a 200
        that produced no output, or a defect in langchaint itself.
        to_tables renders each GenerationError as one failure row, so the batch stays table-ready.
        The returned list is therefore always complete.

        SharedBackoff.max_concurrent_requests sets the batch's throughput, gating every request
        start across everything sharing that instance.
        The batch separately bounds how many items are pending, meaning started and not settled,
        so a batch of a million inputs does not hold a million tasks.

        warm_cache runs generation_inputs[0] to completion before starting the rest, because a
        provider cache entry is readable only once the response that writes it begins.
        Without warming, a batch sharing a cached prefix pays one cold cache write per pending item.
        Warming costs one item of serial latency, and it runs whether or not the binding places a
        cache marker.
        A first item ending in a GenerationError still admits the rest: a 200 that produced no
        output (a refusal, a truncation) wrote the prefix on the provider side, and after a
        transport failure the rest simply run against a cold cache.
        There is no second warmer.

        max_working_seconds_per_item is how long one item may spend able to work, its clock stopped
        for as long as that item waits to be admitted. An item waits behind the batch's other items
        for a permit and then for its turn in the admission queue, and it waits out a shared pause
        without being free to send anything, so charging any of that to the item would expire items
        that never ran. What spends it is the request and the sleeps between attempts.
        Use generate_one's timeout_seconds when what you need bounded is wall clock.
        An item that expires is returned as a TimedOutError while its siblings run on.
        Bound the batch this way rather than with a scope of your own: a cancellation from outside
        discards the returned list, settled results and all, because the list is this frame's and the
        frame is what unwinds.
        Neither an outer cancellation nor an item's BaseException starts an item that had not
        started.

        Raises:
            TypeError: generation_inputs is a bare str (from _reject_bare_str_batch).
            RuntimeError: This binding's account is closed.
            asyncio.CancelledError: an outer scope cancelled the batch.
            BaseException: an item raised a BaseException that is not an Exception, which langchaint
                does not catch; the started items are cancelled and awaited, and it propagates.
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
        """Run the batch at the widest output type; _generate_one_any_binding says why this exists.

        generate_item runs one item, so a caller that wraps each call wraps every item of a batch
        alike, whichever branch below started it.
        Every item gets a WorkingTimeDeadline of its own, built where that item starts.

        Raises:
            TypeError: generation_inputs is a bare str (from _reject_bare_str_batch).
            RuntimeError: This binding's account is closed.
            asyncio.CancelledError: an outer scope cancelled the batch.
            BaseException: an item raised a BaseException that is not an Exception; the started
                items are cancelled and awaited, and it propagates.
        """
        self._ensure_account_open()
        _reject_bare_str_batch(generation_inputs)
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

        The pending bound follows the domain's max_concurrent_requests, which is read-only, so
        run_many always receives a positive int here.

        Raises:
            asyncio.CancelledError: an outer scope cancelled generate_many.
            RuntimeError: This binding's account closes before a request starts.
            BaseException: an item raised a BaseException that is not an Exception; run_many cancels
                and awaits the started items, and it propagates.
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

        max_concurrent_requests = self.shared_backoff.max_concurrent_requests
        if max_concurrent_requests is None:
            max_pending = _MAX_PENDING_TASKS_WITHOUT_A_CONCURRENCY_BOUND
        else:
            max_pending = (
                max_concurrent_requests * _PENDING_TASKS_PER_CONCURRENT_REQUEST
                + _SPARE_PENDING_TASKS
            )
        return await run_many(generation_inputs, run_one, max_pending=max_pending)

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
        """Build the stream handle; entering it with `async with` opens the request.

        The handle's final() result is typed the way generate_one types it, per binding.
        Sync because nothing suspends until the handle is entered;
        see StreamHandle for the retry, close, deadline, and abandoned contracts.
        timeout_seconds bounds the block from entry until the call concludes, so it covers the open,
        the item pulls, and whatever the block does between them. Its clock starts at entry, not
        here, so a handle held before entering loses none of it. Work the block does after the call
        concludes is the caller's own time.
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
            account_state=self._account_state,
            timeout_seconds=timeout_seconds,
            splits_tool_call_turns=self._splits_tool_call_turns,
        )
