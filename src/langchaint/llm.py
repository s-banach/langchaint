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
and a DoNotRetry becomes the item's terminal GenerationError, named by Adapter.classify.
"""

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple, Protocol, SupportsIndex, overload

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    Binding,
    BoundAdapter,
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
    TimedOutError,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
)
from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, TextPart, UserMessage
from langchaint.response import (
    CallResult,
    Response,
    _abandoned_call_error,
)
from langchaint.shared_backoff import (
    Admission,
    DoNotRetry,
    PrivateBackoff,
    SharedBackoff,
)
from langchaint.streaming import StreamHandle
from langchaint.tools import ToolManager


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
    Pass timeout_seconds through: an implementation that dropped it would silently give its items no
    deadline at all.
    """

    async def __call__(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None
    ) -> Response[OutputT | None]:
        """Run the call, raising its GenerationError rather than returning it.

        Raises:
            GenerationError: the call failed; the batch turns it into that item's row.
        """
        ...


class _Interpretation[OutputT](NamedTuple):
    """One arrived response and what interpret read off it.

    The two travel together because the retry loop needs both: raw is what a Response carries, and
    outcome is what decides the item's fate.
    """

    raw: BaseModel
    outcome: ResponseOutcome[OutputT]


class LLM:
    """The un-bound client; holds what is shared across bindings."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        shared_backoff: SharedBackoff | None = None,
        max_attempts: int = 3,
    ) -> None:
        """Store the shared pieces.

        shared_backoff None builds a private domain from the adapter's parse and failure_types,
        at the SharedBackoff defaults with capacity 8. One instance is one backpressure domain,
        so pass the same instance to every LLM whose requests share a provider quota.
        max_attempts counts requests sent including the first, so 1 means no retrying.

        Raises:
            ValueError: max_attempts is not a positive non-bool int, a defect to report before
                any request rather than a retry budget to misread.
        """
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive int, got {max_attempts!r}")
        self.adapter = adapter
        self.shared_backoff = (
            shared_backoff
            if shared_backoff is not None
            else SharedBackoff(
                parse=adapter.parse, failure_types=adapter.failure_types, capacity=8
            )
        )
        self.max_attempts = max_attempts

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
        automatic_prompt_caching: bool,
    ) -> "BoundLLM[str, None]": ...
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tool_manager: ToolManager | None = None,
        response_format: type[BaseModel] | None = None,
        inference_params: InferenceParams | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
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
        Ad-hoc use is llm.bind(automatic_prompt_caching=False).generate_one(...).

        Raises:
            ValueError: system_prompt is an empty sequence of parts; pass None to bind no system
                prompt. Also raised by the adapter for a binding its model cannot be sent, which is
                where an automatic_prompt_caching the model cannot honor is refused.
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
        )
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, binding, response_format),
            response_format=response_format,
            binding=binding,
            tool_manager=tool_manager,
            shared_backoff=self.shared_backoff,
            max_attempts=self.max_attempts,
        )


class BoundLLM[OutputT, ToolManagerT: ToolManager | None = None]:
    """One frozen prefix plus the request methods; constructed by LLM.bind.

    OutputT is what the binding asks the model for: str, or the response_format instance.
    ToolManagerT is the bound tool_manager's type, ToolManager or None.
    The tool_manager property returns it.
    A tool loop therefore dispatches through the binding it was handed.
    ToolManagerT is also what the request methods overload on.
    A structured BoundLLM[Model, ToolManager] types its output OutputT | None.
    That None is the tool-call turn; every other combination types the output OutputT.
    Keeping the None out of OutputT is what lets rebind add and remove a tool_manager.
    The output type is then right both ways.
    The parameter defaults to None, so BoundLLM[Model] annotates the common binding.
    A tool-bound one names both, BoundLLM[Model, ToolManager].
    bind writes ToolManager as the type argument for every manager, subclasses included.

    tool_manager is kept for tool dispatch;
    the provider only ever sees the converted schemas inside the binding.
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
        """Store the frozen pieces; called by LLM.bind and rebind only."""
        self.adapter = adapter
        self.binding = binding
        self.response_format = response_format
        self.shared_backoff = shared_backoff
        self.max_attempts = max_attempts
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
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolManagerT]": ...
    def rebind(
        self,
        *,
        response_format: type[BaseModel] | None | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tool_manager: ToolManager | None | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        inference_params: InferenceParams | Unchanged = UNCHANGED,
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
                prompt. Also raised by the adapter for a binding its model cannot be sent, which is
                where an automatic_prompt_caching the model cannot honor is refused.
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
        )
        new_response_format = (
            self.response_format if isinstance(response_format, Unchanged) else response_format
        )
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, new_binding, new_response_format),
            response_format=new_response_format,
            binding=new_binding,
            tool_manager=new_tool_manager,
            shared_backoff=self.shared_backoff,
            max_attempts=self.max_attempts,
        )

    def _terminal_error(
        self, exc: Exception, *, ledger: _CallLedger, request: RequestParams
    ) -> GenerationError:
        """Name this item's terminal failure from the adapter's classification of exc.

        Reached on a DoNotRetry verdict and on an exception outside failure_types that classify
        did not call transient, so the "transient" member cannot arrive; if a classify defect
        produces one anyway, it lands on the unknown_exception default with everything else out
        of place.

        StreamHandle carries its own copy of this mapping; what the two retry loops share is the ledger in call.py.
        """
        classification = self.adapter.classify(exc)
        if classification == "invalid_request":
            # Adapter.classify returns invalid_request only for a request the provider rejected,
            # so it went out and gets a record. A rejection carries no response, so the record bills
            # nothing unless a response was staged, which is the exception raised while reading one.
            ledger.record(error=None, assistant_message=None)
            return InvalidRequestError(
                reason=f"the provider rejected the request: {exc}",
                call=ledger.freeze(),
                request=request,
            )
        if classification == "declared_final":
            # The provider answered, so the attempt gets a record; its answer was an error, which
            # reports no billing, so the record bills nothing unless a response was staged.
            ledger.record(error=None, assistant_message=None)
            return ProviderDeclaredFinalError(error=exc, call=ledger.freeze(), request=request)
        return UnknownExceptionError(error=exc, call=ledger.freeze(), request=request)

    async def _pace_after_verdict(
        self,
        exc: Exception,
        *,
        admission: Admission,
        private_backoff: PrivateBackoff,
        assistant_message: AssistantMessage | None,
        ledger: _CallLedger,
        request: RequestParams,
    ) -> None:
        """Record one verdicted attempt, then wait whatever the verdict asks before the next.

        exc is a failure_types exception, so the admitted() block's exit parsed it and left the
        verdict on admission.verdict. A verdict of None is folded into the terminal branch: the
        exit parses every failure_types exception, so a None reaching here has no verdict to act
        on.
        The attempt's record carries a TransientError: exc itself when it is one, otherwise one
        wrapping exc with the verdict's capped retry_after.
        On RetryThisOne the wait is the PrivateBackoff's, floored by the verdict's retry_after;
        on PauseAll there is no wait of our own, because the next admitted() entry already holds
        until the shared pause ends. Neither waits after the last attempt.
        assistant_message is the turn a 200 the provider filled with a failure still carried, and
        None where the attempt received no response.

        Raises:
            GenerationError: the verdict is DoNotRetry; _terminal_error names which.
        """
        ledger.note_request_id(self.adapter.request_id_from_error(exc))
        verdict = admission.verdict
        if verdict is None or isinstance(verdict, DoNotRetry):
            raise self._terminal_error(exc, ledger=ledger, request=request) from exc
        if isinstance(exc, TransientError):
            error = exc
        else:
            error = TransientError(
                str(exc),
                retry_after_seconds=verdict.retry_after,
                is_rate_limit=verdict.kind == "pause_all",
            )
            error.__cause__ = exc
        ledger.record(error=error, assistant_message=assistant_message)
        if verdict.kind == "retry_this_one" and ledger.attempts < self.max_attempts:
            await asyncio.sleep(private_backoff.next_wait(verdict.retry_after))

    async def _pace_after_transport_failure(
        self,
        exc: Exception,
        *,
        private_backoff: PrivateBackoff,
        ledger: _CallLedger,
        request: RequestParams,
    ) -> GenerationError | None:
        """Record one transport failure and wait, or return the terminal error for it.

        exc is outside failure_types, so it exited the admitted() block unparsed and unrecorded
        there. classify's "transient" is a transport failure that produced nothing parseable: the
        loop retries it alone, as RetryThisOne with no retry_after, and does not wait after the
        last attempt. For anything else this returns the GenerationError _terminal_error names,
        and the caller raises it so the raise sits beside the except clause that caught exc.
        """
        ledger.note_request_id(self.adapter.request_id_from_error(exc))
        if self.adapter.classify(exc) != "transient":
            return self._terminal_error(exc, ledger=ledger, request=request)
        error = TransientError(str(exc))
        error.__cause__ = exc
        ledger.record(error=error, assistant_message=None)
        if ledger.attempts < self.max_attempts:
            await asyncio.sleep(private_backoff.next_wait(None))
        return None

    def _staged_interpretation(
        self, sent: BaseModel, *, ledger: _CallLedger
    ) -> _Interpretation[OutputT | None]:
        """Stage an arrived response with its billing, then read what it produced.

        Staging first is what makes the attempt and its billing survive a raise from interpret:
        freeze closes a still-staged response, so the error that raise becomes carries the record.

        Raises:
            Exception: whatever interpret raises, for Adapter.classify to sort.
        """
        ledger.stage_response(
            raw=sent,
            billing=self._bound_adapter.billing_from_raw(sent),
            identity=self._bound_adapter.identity_from_raw(sent),
        )
        return _Interpretation(raw=sent, outcome=self._bound_adapter.interpret(sent))

    async def _generate_with_retries(
        self,
        messages: Sequence[Message],
        *,
        ledger: _CallLedger,
        timeout_seconds: float | None,
    ) -> Response[OutputT | None]:
        """Run the retry loop every generate method shares, under the caller's deadline.

        ledger is the caller's own empty ledger (the retry budget counts its attempts), recorded
        into as each attempt settles. Every GenerationError and the Response are built from
        ledger.freeze(), the one site a call's elapsed_seconds is computed.

        timeout_seconds bounds this whole loop, admission waits and backoff sleeps included, and
        None opens a scope that never expires. Expiring raises TimedOutError, whose docstring says
        why the scope has to sit in this frame.
        A cancellation from any scope but this one is the caller's own order and propagates
        untouched. expired() is what tells the two apart: a TimeoutError this scope did not raise
        came from under the loop unclassified, and re-raising it hands it to the same wrapping every
        other unclassified exception gets.

        The adapter reports one attempt as a ResponseOutcome member and never as a GenerationError,
        so this loop matches the member and constructs the item's GenerationError here, where the
        attempts and the timing are known.
        Each arrived response is staged on the ledger with its billing before anything is read off it,
        so an exception from that read still leaves the attempt and its billing on the record.
        Each attempt spans one admitted() block, held for the request only;
        backoff sleeps sit outside the block so a waiting task does not hold capacity.
        Each provider failure is raised inside the block, so the exit records its verdict before
        anyone else is admitted and a rate-limit error pauses the whole domain.
        Every attempt is timed onto an AttemptRecord whose bracket is the send only,
        excluding the admission wait and the backoff sleep,
        so a slow request is distinguishable from time spent rate limited.

        Raises:
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
            TimedOutError: timeout_seconds expired before the call produced a result.
            ParserContractError: the adapter's parse violated its contract on an attempt's failure.
        """
        ledger.start_call()
        timeout_scope = asyncio.timeout(timeout_seconds)
        try:
            async with timeout_scope:
                return await self._attempt_until_budget_runs_out(messages, ledger=ledger)
        except TimeoutError:
            if not timeout_scope.expired():
                raise
            raise _abandoned_call_error(TimedOutError, ledger) from None

    async def _attempt_until_budget_runs_out(
        self, messages: Sequence[Message], *, ledger: _CallLedger
    ) -> Response[OutputT | None]:
        """Send the request until it succeeds, fails terminally, or the retry budget runs out.

        Runs inside the deadline opened by _generate_with_retries, its only caller.

        Raises:
            GenerationError: every failure _generate_with_retries names but TimedOutError, which its
                scope raises.
            ParserContractError: the adapter's parse violated its contract on an attempt's failure.
        """
        built = self._bound_adapter.build_request(messages)
        if isinstance(built, InvalidRequest):
            raise InvalidRequestError(reason=built.reason, call=ledger.freeze(), request=None)
        request = built
        private_backoff = PrivateBackoff(self.shared_backoff)
        while ledger.attempts < self.max_attempts:
            admission = self.shared_backoff.admitted()
            assistant_message: AssistantMessage | None = None
            try:
                async with admission:
                    ledger.start_attempt()
                    raw, outcome = self._staged_interpretation(
                        await self._bound_adapter.send(request), ledger=ledger
                    )
                    if outcome.kind == "provider_failed_transiently":
                        # Raised inside the block so the exit parses it: a billable 200 whose
                        # body reports a transient failure is still a provider failure, and a
                        # rate-limit body pauses the domain exactly as a 429 status does.
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
                )
            except Exception as exc:
                terminal = await self._pace_after_transport_failure(
                    exc, private_backoff=private_backoff, ledger=ledger, request=request
                )
                if terminal is not None:
                    raise terminal from exc
            else:
                # error is None on every member reaching here: the request succeeded, and what the
                # adapter made of the response is the item's outcome, not this attempt's failure.
                ledger.record(error=None, assistant_message=outcome.assistant_message)
                match outcome.kind:
                    case "adapter_result":
                        return Response(
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
    ) -> Response[OutputT | None]: ...
    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> Response[OutputT]: ...
    async def generate_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> Response[Any]:
        """Generate one response under the retry loop.

        output is None only on a structured tool-bound binding, where the turn parsed no instance;
        the overloads type it away everywhere else, a text turn's output being "" and not None.
        Response.output states what a None means and what to branch on for a pending tool call.
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
            asyncio.CancelledError: an outer scope cancelled this call.
        """
        return await self._generate_one_any_binding(
            generation_input, timeout_seconds=timeout_seconds
        )

    async def _generate_one_any_binding(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None
    ) -> Response[OutputT | None]:
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
            BaseException: whatever cut the call off, propagating unobserved.
        """
        ledger = _CallLedger(model=self.adapter.model, provider_name=self.adapter.provider_name)
        try:
            return await self._generate_with_retries(
                _as_messages(generation_input), ledger=ledger, timeout_seconds=timeout_seconds
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
        timeout_seconds: float | None,
    ) -> CallResult[OutputT | None]:
        """One batch item: the Response or the GenerationError.

        Every terminal per-item outcome is a GenerationError, so nothing a request produces escapes
        into the gather and reaches a sibling. An expired timeout_seconds is one of them, so one
        item's deadline never cuts a sibling.

        Raises:
            BaseException: whatever cut the item off, propagating unobserved.
        """
        try:
            return await generate_item(generation_input, timeout_seconds=timeout_seconds)
        except GenerationError as failure:
            return failure

    @overload
    async def generate_many(
        self: "BoundLLM[str, ToolManagerT]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        timeout_seconds: float | None = ...,
    ) -> list[CallResult[str]]: ...
    @overload
    async def generate_many(
        self: "BoundLLM[OutputT, ToolManager]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        timeout_seconds: float | None = ...,
    ) -> list[CallResult[OutputT | None]]: ...
    @overload
    async def generate_many(
        self: "BoundLLM[OutputT, None]",
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = ...,
        timeout_seconds: float | None = ...,
    ) -> list[CallResult[OutputT]]: ...
    async def generate_many(
        self,
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool = False,
        timeout_seconds: float | None = None,
    ) -> list[CallResult[Any]]:
        """Order-aligned batch: result i belongs to generation_inputs[i].

        A Response's output is typed the way generate_one types it, per binding.
        A bare str as the whole batch is rejected: str satisfies the item Sequence type,
        so it would silently become one request per character.
        Every item ends in its own slot: a Response, or the GenerationError it failed with
        (retries exhausted, a rejected request, an error langchaint does not retry, a 200 that
        produced no output, or a defect in langchaint itself), which to_tables renders to a failure
        row so the batch stays table-ready.
        No item's failure reaches a sibling, so the returned list is always complete.
        Concurrency is bounded by shared_backoff.capacity, which gates every request start and is
        shared with everything else using the same SharedBackoff instance;
        a capacity of None leaves the bound to whatever spawns the work.

        warm_cache runs generation_inputs[0] to completion before starting the rest,
        because a provider cache entry is readable only after the response that writes it begins,
        so a batch sharing a cached prefix otherwise pays one cold cache write per in-flight item.
        It costs one item of serial latency and warms unconditionally,
        whether or not the binding places any cache marker.
        A first item ending in a GenerationError still admits the rest:
        a 200 that produced no output (a refusal, a truncation) wrote the prefix on the provider side,
        and after a transport failure the rest simply run against a cold cache; there is no second warmer.
        There is no warmup ladder: after the first item settles, every remaining item is admitted at once.

        timeout_seconds is each item's own deadline, started when that item starts, and an item that
        expires gets a TimedOutError row while its siblings run on. Bound the batch this way rather
        than with a scope of your own: a cancellation from outside discards the returned list,
        settled rows and all, because the list is this frame's and the frame is what unwinds.

        Raises:
            TypeError: generation_inputs is a bare str (from _reject_bare_str_batch).
            asyncio.CancelledError: an outer scope cancelled the batch.
            BaseException: an item raised a BaseException that is not an Exception, which langchaint
                does not catch; _gather cancels the remaining items and it propagates.
        """
        return await self._generate_many_any_binding(
            generation_inputs,
            warm_cache=warm_cache,
            generate_item=self._generate_one_any_binding,
            timeout_seconds=timeout_seconds,
        )

    async def _generate_many_any_binding(
        self,
        generation_inputs: SequenceNotStr[GenerationInput],
        *,
        warm_cache: bool,
        generate_item: "GenerateItem[OutputT]",
        timeout_seconds: float | None,
    ) -> list[CallResult[OutputT | None]]:
        """Run the batch at the widest output type; _generate_one_any_binding says why this exists.

        generate_item runs one item, so a caller that wraps each call wraps every item of a batch
        alike, whichever branch below started it.
        timeout_seconds is each item's own deadline, started when that item starts.

        Raises:
            TypeError: generation_inputs is a bare str (from _reject_bare_str_batch).
            asyncio.CancelledError: an outer scope cancelled the batch.
            BaseException: an item raised a BaseException that is not an Exception; _gather cancels
                the remaining items and it propagates.
        """
        _reject_bare_str_batch(generation_inputs)
        # The slices also convert the SequenceNotStr protocol to the Sequence _gather takes.
        if warm_cache and generation_inputs:
            first_result = await self._generate_or_failure(
                generation_inputs[0],
                generate_item=generate_item,
                timeout_seconds=timeout_seconds,
            )
            rest = await self._gather(
                generation_inputs[1:],
                generate_item=generate_item,
                timeout_seconds=timeout_seconds,
            )
            return [first_result, *rest]
        return await self._gather(
            generation_inputs[0:],
            generate_item=generate_item,
            timeout_seconds=timeout_seconds,
        )

    async def _gather(
        self,
        generation_inputs: Sequence[GenerationInput],
        *,
        generate_item: "GenerateItem[OutputT]",
        timeout_seconds: float | None,
    ) -> list[CallResult[OutputT | None]]:
        """Run the items concurrently and return the settled list, order-aligned.

        The cancelled tasks are awaited before it raises, because gather returns here with siblings
        still running on both paths: a KeyboardInterrupt or a SystemExit from an item propagates
        immediately, and a cancellation propagates as soon as the first item completes as cancelled.

        Raises:
            asyncio.CancelledError: an outer scope cancelled generate_many.
            BaseException: an item raised a BaseException that is not an Exception, which langchaint
                does not catch; the remaining tasks are cancelled and it propagates.
        """
        tasks = [
            asyncio.create_task(
                self._generate_or_failure(
                    generation_input,
                    generate_item=generate_item,
                    timeout_seconds=timeout_seconds,
                )
            )
            for generation_input in generation_inputs
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                _ = task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
    ) -> StreamHandle[OutputT | None]: ...
    @overload
    def stream_one(
        self: "BoundLLM[OutputT, None]",
        generation_input: GenerationInput,
        *,
        timeout_seconds: float | None = ...,
    ) -> StreamHandle[OutputT]: ...
    def stream_one(
        self, generation_input: GenerationInput, *, timeout_seconds: float | None = None
    ) -> StreamHandle[Any]:
        """Build the stream handle; entering it with `async with` opens the request.

        The handle's final() Response types output the way generate_one types it, per binding.
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
    ) -> StreamHandle[OutputT | None]:
        """Build the handle at the widest output type; _generate_one_any_binding says why."""
        return StreamHandle(
            adapter=self.adapter,
            bound_adapter=self._bound_adapter,
            messages=_as_messages(generation_input),
            shared_backoff=self.shared_backoff,
            max_attempts=self.max_attempts,
            timeout_seconds=timeout_seconds,
        )
