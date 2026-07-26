"""The client; generation happens only through a binding.

LLM composes an adapter and a RateLimiter.
LLM has no generate methods.
bind() freezes everything that determines the cacheable prompt prefix,
fixes the output type, and precomputes SDK keyword arguments once;
the returned BoundLLM takes only the per-request conversation.
There are no per-call parameter overrides; changing parameters is rebind().
The RateLimiter slot gates every request start on every path, retries included;
the retry loop feeds every failure and every success back
so a rate-limit error pauses admission account-wide until a request succeeds again.
"""

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple, Protocol, SupportsIndex, assert_never, overload

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    AdapterResult,
    Binding,
    BoundAdapter,
    ContextWindowExceeded,
    EmptyTurn,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutput,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    Refusal,
    ResponseOutcome,
    SchemaViolation,
    ToolChoice,
    UnfinishedTurn,
)
from langchaint.call import _CallLedger
from langchaint.exceptions import (
    ContextWindowExceededError,
    EmptyTurnError,
    GenerationError,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    ProviderFailedTerminallyError,
    RefusalError,
    RetriesExhaustedError,
    SchemaViolationError,
    TransientError,
    UnfinishedTurnError,
    UnrecognizedError,
    _extract_transient_errors,
)
from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, TextPart, UserMessage
from langchaint.rate_limiter import Admission, RateLimiter
from langchaint.response import AbandonedCallLog, Response, _append_abandoned_call
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


class SequenceNotStr[T_co](Protocol):
    """A Sequence that a type checker rejects a bare str for.

    str satisfies Sequence[str | Sequence[Message]] (a str is a sequence of str),
    so a plain Sequence batch parameter statically accepts generate_many("hi"),
    which would run one request per character.
    This protocol structurally matches list and tuple but not str,
    because typeshed's str.__contains__ accepts only str while the protocol requires __contains__(value: object).
    Being covariant, it also accepts a caller's list[str] or list[list[UserMessage]],
    which the invariant list[str | Sequence[Message]] would reject.
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


def _reject_bare_str_batch(conversations: SequenceNotStr[str | Sequence[Message]]) -> None:
    """Reject a bare str passed as the whole batch.

    The SequenceNotStr parameter type makes the type checker reject a bare str;
    this runtime guard is the backstop for untyped callers.

    Raises:
        TypeError: conversations is a bare str.
    """
    if isinstance(conversations, str):
        raise TypeError(
            "conversations is a bare str; wrap it in a list, or use generate_one"
            " for a single conversation"
        )


def _as_conversation(conversation: str | Sequence[Message]) -> Sequence[Message]:
    if isinstance(conversation, str):
        return (UserMessage(content=conversation),)
    return conversation


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


class NoTools:
    """Type-level marker: a binding that bound no ToolManager, so no turn can be a tool call.

    Never instantiated. It and HasTools are the two values of BoundLLM's second type parameter,
    which is what lets the request methods type output as the instance itself rather than as
    optional: a binding that cannot receive a tool call always produces the output it was bound for.
    """


class HasTools:
    """Type-level marker: a binding that bound a ToolManager, so a turn may be a tool call.

    Never instantiated; the counterpart of NoTools, whose docstring states what the pair is for.
    A structured binding marked this way types its output optional, None being the tool-call turn.
    """


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
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Store the shared pieces; rate_limiter None means the defaults."""
        self.adapter = adapter
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()

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
    ) -> "BoundLLM[ModelT, HasTools]": ...
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
    ) -> "BoundLLM[ModelT, NoTools]": ...
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
    ) -> "BoundLLM[str, HasTools]": ...
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
    ) -> "BoundLLM[str, NoTools]": ...
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
        Passing a tool_manager gives the HasTools form, whose structured request methods type output
        as optional because a tool-call turn parses no instance; see BoundLLM.
        A caller holding a ToolManager | None gets the union of the two forms, whose request methods
        return the optional type, which is what a caller who does not know can act on.
        automatic_prompt_caching has no default: caching changes billing,
        so langchaint never chooses a caching configuration for the caller.
        Ad-hoc use is llm.bind(automatic_prompt_caching=False).generate_one(...).
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
            rate_limiter=self.rate_limiter,
        )


class BoundLLM[OutputT, ToolsT = NoTools]:
    """One frozen prefix plus the request methods; constructed by LLM.bind.

    OutputT is what the binding asks the model for: str, or the response_format instance.
    ToolsT is NoTools or HasTools, and says whether a tool_manager was bound. It is what the request
    methods overload on: a structured HasTools binding types its output OutputT | None, None being
    the tool-call turn, and every other combination types it OutputT. Keeping the None out of OutputT
    is what lets rebind add and remove a tool_manager and get the output type right both ways.
    The parameter defaults to NoTools, so BoundLLM[Model] annotates the common binding and a
    tool-bound one names both, BoundLLM[Model, HasTools].

    tool_manager is kept for tool dispatch (the manual tool loop reads it);
    the provider only ever sees the converted schemas inside the binding.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT | None],
        response_format: type[OutputT] | None,
        binding: Binding,
        tool_manager: ToolManager | None,
        rate_limiter: RateLimiter,
    ) -> None:
        """Store the frozen pieces; called by LLM.bind and rebind only."""
        self.adapter = adapter
        self.binding = binding
        self.response_format = response_format
        self.tool_manager = tool_manager
        self.rate_limiter = rate_limiter
        self._bound_adapter = bound_adapter

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
    ) -> "BoundLLM[NewModelT, HasTools]": ...
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
    ) -> "BoundLLM[NewModelT, NoTools]": ...
    @overload
    def rebind[NewModelT: BaseModel](
        self: "BoundLLM[OutputT, ToolsT]",
        *,
        response_format: type[NewModelT],
        tool_manager: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[NewModelT, ToolsT]": ...
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
    ) -> "BoundLLM[str, HasTools]": ...
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
    ) -> "BoundLLM[str, NoTools]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolsT]",
        *,
        response_format: None,
        tool_manager: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[str, ToolsT]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolsT]",
        *,
        response_format: Unchanged = ...,
        tool_manager: ToolManager,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, HasTools]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolsT]",
        *,
        response_format: Unchanged = ...,
        tool_manager: None,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, NoTools]": ...
    @overload
    def rebind(
        self: "BoundLLM[OutputT, ToolsT]",
        *,
        response_format: Unchanged = ...,
        tool_manager: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "BoundLLM[OutputT, ToolsT]": ...
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
        """Replace bound fields; a left-out field keeps its current value.

        response_format and tool_manager are the two fields whose change alters the static output
        type, so they drive the overload return type: the first sets OutputT, the second sets ToolsT,
        and leaving either out keeps what this binding has. Every combination is exact, including
        dropping a tool_manager, which is what returns a structured binding to a non-optional output.
        Replace semantics: a passed inference_params replaces the bound one whole, never field-wise.
        Every rebind converts the binding to SDK keyword arguments again, a pure conversion with no I/O.
        Whether a rebind preserves the provider's prompt cache is provider-specific and partly undocumented
        (Anthropic documents the prefix order tools -> system -> messages),
        and it depends on which field a rebind changes and on which value that field moves between,
        so measure it on the deployment you ship on.
        langchaint owns no cache-safety matrix over this.
        A matrix carried in the code goes stale the moment a provider changes a model.
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
            rate_limiter=self.rate_limiter,
        )

    def _new_ledger(self) -> _CallLedger:
        """Open a ledger against this binding's adapter; the only place llm.py reads its identity."""
        return _CallLedger(model=self.adapter.model, provider_name=self.adapter.provider_name)

    def _classified_error(
        self, exc: Exception, *, ledger: _CallLedger
    ) -> TransientError | GenerationError:
        """Sort one attempt's exception into the error to retry or this item's terminal failure.

        Reached only for exceptions, which by the adapter contract are attempts the adapter read no
        outcome from: what it did read it reports as an AttemptOutcome arm, which the loop matches
        instead. The returned TransientError carries the adapter's retry-after reading and whether
        the error was a rate limit, the two things the limiter needs to pace the next attempt;
        every other return is terminal for this item, and the caller raises it.

        StreamHandle carries its own copy of this mapping; what the two retry loops share is the ledger in call.py.
        """
        classification = self.adapter.classify(exc)
        if classification == "invalid_request":
            # Adapter.classify returns invalid_request only for a request the provider rejected,
            # so it went out and gets a record. A rejection carries no response, so the record is
            # ZERO_USAGE unless a receipt was staged, which is the exception raised while reading one.
            ledger.record(error=None, assistant_message=None)
            return InvalidRequestError(
                reason=f"the provider rejected the request: {exc}", call=ledger.freeze()
            )
        if classification == "unrecognized":
            return UnrecognizedError(error=exc, call=ledger.freeze())
        error = TransientError(
            str(exc),
            retry_after_seconds=self.adapter.retry_after_seconds(exc),
            is_rate_limit=classification == "rate_limit",
        )
        error.__cause__ = exc
        return error

    def _record_completed_attempt(
        self,
        outcome: AdapterResult[OutputT | None] | NoOutput,
        *,
        admission: Admission,
        ledger: _CallLedger,
    ) -> None:
        """Register the completed request with the limiter and close its attempt.

        error is None on every member reaching here: the request succeeded, and what the adapter
        made of the response is the item's outcome, not this attempt's failure.
        Called while the attempt's slot is still held, so a completed request ends the limiter's
        recovery before anyone else is admitted. Every 200 counts as completed, including one that
        produced no output: the provider served the request, which is what the recovery probe asks.
        """
        self.rate_limiter.register_success(admission)
        ledger.record(error=None, assistant_message=outcome.assistant_message)

    def _record_transient_error(
        self,
        error: TransientError,
        *,
        assistant_message: AssistantMessage | None,
        ledger: _CallLedger,
    ) -> float:
        """Close the failed attempt and register it with the limiter, while its slot is still held.

        assistant_message is the turn a 200 the provider filled with a failure still carried, and
        None where the attempt received no response.

        Returns:
            The backoff delay to sleep before the next attempt, in seconds;
            register_transient_error draws it once so it equals any account-wide pause it set.
        """
        ledger.record(error=error, assistant_message=assistant_message)
        return self.rate_limiter.register_transient_error(
            _extract_transient_errors(ledger.attempt_records)
        )

    def _staged_interpretation(
        self, sent: BaseModel | InvalidRequest, *, ledger: _CallLedger
    ) -> _Interpretation[OutputT | None] | InvalidRequest:
        """Record the receipt of an arrived response, then read what it produced.

        Staging first is what makes the attempt and its billing survive a raise from interpret:
        freeze closes a still-staged receipt, so the error that raise becomes carries the record.
        An InvalidRequest passes through, having received nothing to record.

        Raises:
            Exception: whatever interpret raises, for Adapter.classify to sort.
        """
        if isinstance(sent, InvalidRequest):
            return sent
        ledger.stage_receipt(raw=sent, usage=self._bound_adapter.usage_from_raw(sent))
        return _Interpretation(raw=sent, outcome=self._bound_adapter.interpret(sent))

    async def _generate_with_retries(
        self, conversation: Sequence[Message], *, ledger: _CallLedger
    ) -> Response[OutputT | None]:
        """Run the retry loop every generate method shares.

        ledger is the caller's own empty ledger (the retry budget counts its attempts), recorded
        into as each attempt settles, so a cancellation that kills this frame leaves the settled
        attempts readable outside it; generate_one freezes it to build its AbandonedCall.
        Every GenerationError and the Response are built from ledger.freeze(), the one site a call's
        elapsed_seconds is computed.

        The adapter reports one attempt as an AttemptOutcome member and never as a GenerationError,
        so this loop matches the member and constructs the item's GenerationError here, where the
        attempts and the timing are known.
        Each arrived response is staged on the ledger with its price before anything is read off it,
        so an exception from that read still leaves the attempt and its billing on the record.
        Every exception, whether the attempt reached a response or not, goes to Adapter.classify.
        Each attempt holds a RateLimiter slot for the request only;
        backoff sleeps outside the slot so a waiting task does not hold capacity.
        Every failure and every success is registered with the limiter while the slot is still held,
        so a rate-limit error pauses admission account-wide before anyone else is admitted and a completed request
        ends recovery. Every 200 counts as completed, including one that produced no output:
        the provider served the request, which is what the recovery probe asks.
        Every attempt is timed onto an AttemptRecord whose bracket is the send only,
        excluding the slot wait and the backoff sleep,
        so a slow request is distinguishable from time spent rate limited.

        Raises:
            InvalidRequestError: the adapter reported the conversation as InvalidRequest, or classified
                an attempt's error as a rejection of the request; terminal for this item, without a retry.
            UnrecognizedError: the adapter classified an attempt's error as unrecognized;
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
        """
        ledger.start_call()
        while ledger.attempts < self.rate_limiter.max_attempts:
            async with self.rate_limiter.slot() as admission:
                ledger.start_attempt()
                assistant_message: AssistantMessage | None = None
                try:
                    interpreted = self._staged_interpretation(
                        await self._bound_adapter.send(conversation), ledger=ledger
                    )
                except TransientError as exc:
                    error: TransientError = exc
                except Exception as exc:
                    classified = self._classified_error(exc, ledger=ledger)
                    if not isinstance(classified, TransientError):
                        raise classified from exc
                    error = classified
                else:
                    if isinstance(interpreted, InvalidRequest):
                        raise InvalidRequestError(reason=interpreted.reason, call=ledger.freeze())
                    raw, outcome = interpreted
                    match outcome:
                        case AdapterResult():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            return Response(
                                output=outcome.output,
                                call=ledger.freeze(),
                                raw=raw,
                                stop_reason=outcome.stop_reason,
                                assistant_message=outcome.assistant_message,
                            )
                        case Refusal():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise RefusalError(call=ledger.freeze())
                        case MaxCompletionTokensExceeded():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise MaxCompletionTokensExceededError(call=ledger.freeze())
                        case EmptyTurn():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise EmptyTurnError(call=ledger.freeze())
                        case SchemaViolation():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise SchemaViolationError(
                                validation_error_json=outcome.validation_error_json,
                                call=ledger.freeze(),
                            )
                        case ContextWindowExceeded():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise ContextWindowExceededError(call=ledger.freeze())
                        case UnfinishedTurn():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise UnfinishedTurnError(reason=outcome.reason, call=ledger.freeze())
                        case ProviderFailedTerminally():
                            self._record_completed_attempt(
                                outcome, admission=admission, ledger=ledger
                            )
                            raise ProviderFailedTerminallyError(
                                reason=outcome.reason, call=ledger.freeze()
                            )
                        case ProviderFailedTransiently():
                            self.rate_limiter.register_success(admission)
                            error = TransientError(
                                outcome.reason, is_rate_limit=outcome.is_rate_limit
                            )
                            assistant_message = outcome.assistant_message
                        case _ as unhandled:
                            assert_never(unhandled)
                delay_seconds = self._record_transient_error(
                    error, assistant_message=assistant_message, ledger=ledger
                )
            if ledger.attempts < self.rate_limiter.max_attempts:
                await asyncio.sleep(delay_seconds)
        raise RetriesExhaustedError(call=ledger.freeze())

    @overload
    async def generate_one(
        self: "BoundLLM[str, ToolsT]",
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> Response[str]: ...
    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, HasTools]",
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> Response[OutputT | None]: ...
    @overload
    async def generate_one(
        self: "BoundLLM[OutputT, NoTools]",
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> Response[OutputT]: ...
    async def generate_one(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = None,
    ) -> Response[Any]:
        """Generate one response under the retry loop.

        output is None only on a structured tool-bound binding, where the turn parsed no instance;
        the overloads type it away everywhere else, a text turn's output being "" and not None.
        Response.output states what a None means and what to branch on for a pending tool call.
        A bare str is shorthand for a conversation of one UserMessage holding that text.
        Every non-success outcome propagates, all of them sharing the GenerationError base a caller
        can catch at once: RetriesExhaustedError on transient exhaustion, InvalidRequestError on a
        rejected request, UnrecognizedError on an unrecognized error, and one of RefusalError,
        MaxCompletionTokensExceededError, EmptyTurnError, SchemaViolationError,
        ContextWindowExceededError, UnfinishedTurnError, or ProviderFailedTerminallyError on a 200
        that produced no output.
        _generate_with_retries names the condition for each.

        abandoned_call_log, when given, receives one AbandonedCall if a cancellation (a caller's
        asyncio.timeout, a TaskGroup sibling failing, shutdown) cuts this call off. The append is
        the only channel that path has: no value reaches the caller, and the settled attempts'
        records live in this frame, which the cancellation unwinds. Every other outcome appends
        nothing, because its usage travels on the Response or the raised GenerationError, which the
        caller records itself.

        Raises:
            asyncio.CancelledError: an outer scope cancelled this call; when abandoned_call_log is
                given, the AbandonedCall is appended first.
        """
        return await self._generate_one_any_binding(
            conversation, abandoned_call_log=abandoned_call_log
        )

    async def _generate_one_any_binding(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None,
    ) -> Response[OutputT | None]:
        """Run one call, appending its AbandonedCall if a cancellation cuts the call off.

        What generate_one does, at the widest output type, callable from a frame whose binding is not
        statically concrete: generate_one's overloads are keyed on the binding, so they match no
        generic self. The batch path and the tracing wrapper reach the request through here.

        Raises:
            GenerationError: whatever _generate_with_retries failed the item with.
            asyncio.CancelledError: an outer scope cancelled this call; the AbandonedCall is
                appended first when abandoned_call_log is given.
        """
        ledger = self._new_ledger()
        try:
            return await self._generate_with_retries(_as_conversation(conversation), ledger=ledger)
        except asyncio.CancelledError:
            _append_abandoned_call(abandoned_call_log, ledger.freeze())
            raise

    async def _generate_or_failure(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None,
    ) -> Response[OutputT | None] | GenerationError:
        """One batch item: the Response, or the GenerationError caught as the failure row.

        Every terminal per-item outcome is a GenerationError, so nothing a request produces
        escapes into the gather and reaches a sibling.
        A cancellation is the one outcome that is not a row, so it runs through
        _generate_one_any_binding, whose CancelledError handler appends this item's AbandonedCall
        before re-raising.
        """
        try:
            return await self._generate_one_any_binding(
                conversation, abandoned_call_log=abandoned_call_log
            )
        except GenerationError as failure:
            return failure

    @overload
    async def generate_many(
        self: "BoundLLM[str, ToolsT]",
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = ...,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> list[Response[str] | GenerationError]: ...
    @overload
    async def generate_many(
        self: "BoundLLM[OutputT, HasTools]",
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = ...,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> list[Response[OutputT | None] | GenerationError]: ...
    @overload
    async def generate_many(
        self: "BoundLLM[OutputT, NoTools]",
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = ...,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> list[Response[OutputT] | GenerationError]: ...
    async def generate_many(
        self,
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = False,
        abandoned_call_log: AbandonedCallLog | None = None,
    ) -> list[Response[Any] | GenerationError]:
        """Order-aligned batch: result i belongs to conversations[i].

        A Response's output is typed the way generate_one types it, per binding.
        Each conversation may be a bare str, shorthand for a conversation of one UserMessage holding that text.
        A bare str as the whole batch is rejected: str satisfies the item Sequence type,
        so it would silently become one request per character.
        Every item ends in its own slot: a Response, or the GenerationError it failed with
        (retries exhausted, a rejected request, an unrecognized error, or a 200 that produced no
        output), which to_row renders to a failure row so the batch stays table-ready.
        No item's failure reaches a sibling, so the returned list is always complete.
        Concurrency is bounded by rate_limiter.max_in_flight,
        which gates every request start and is shared with everything else using the same RateLimiter instance.

        warm_cache runs conversations[0] to completion before starting the rest,
        because a provider cache entry is readable only after the response that writes it begins,
        so a batch sharing a cached prefix otherwise pays one cold cache write per in-flight item.
        It costs one item of serial latency and warms unconditionally,
        whether or not the binding places any cache marker.
        A first item ending in a GenerationError still admits the rest:
        a 200 that produced no output (a refusal, a truncation) wrote the prefix on the provider side,
        and after a transport failure the rest simply run against a cold cache; there is no second warmer.
        There is no warmup ladder: after the first item settles, every remaining item is admitted at once.

        abandoned_call_log, when given, receives one AbandonedCall for each item that had started
        when a cancellation (a caller's asyncio.timeout, a TaskGroup sibling failing, shutdown) cuts
        the batch off; under warm_cache the items after the warming one have not started. An item
        raising past the GenerationError arms discards the batch the same way, and every item that
        did not itself raise is recorded. The returned list is lost with the frame on both paths, an
        already-settled item's row with it, so the appends are the account of what the recorded
        items spent.

        Raises:
            TypeError: conversations is a bare str (from _reject_bare_str_batch).
            asyncio.CancelledError: an outer scope cancelled the batch; when abandoned_call_log is
                given, each started item's AbandonedCall is appended first.
            BaseException: an item raised something that is not a GenerationError, a defect in
                langchaint itself; _gather cancels the remaining items and it propagates.
        """
        return await self._generate_many_any_binding(
            conversations, warm_cache=warm_cache, abandoned_call_log=abandoned_call_log
        )

    async def _generate_many_any_binding(
        self,
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool,
        abandoned_call_log: AbandonedCallLog | None,
    ) -> list[Response[OutputT | None] | GenerationError]:
        """Run the batch at the widest output type; _generate_one_any_binding says why this exists.

        Raises:
            TypeError: conversations is a bare str (from _reject_bare_str_batch).
            asyncio.CancelledError: an outer scope cancelled the batch; when abandoned_call_log is
                given, each started item's AbandonedCall is appended first.
            BaseException: an item raised something that is not a GenerationError; _gather cancels
                the remaining items and it propagates.
        """
        _reject_bare_str_batch(conversations)
        # The slices also convert the SequenceNotStr protocol to the Sequence _gather takes.
        if warm_cache and conversations:
            first_result = await self._generate_or_failure(
                conversations[0], abandoned_call_log=abandoned_call_log
            )
            try:
                rest = await self._gather(conversations[1:], abandoned_call_log=abandoned_call_log)
            except BaseException:
                # The warming item settled in this frame rather than in a _gather task,
                # so its record reaches the log here or not at all.
                _append_abandoned_call(abandoned_call_log, first_result.call)
                raise
            return [first_result, *rest]
        return await self._gather(conversations[0:], abandoned_call_log=abandoned_call_log)

    async def _gather(
        self,
        conversations: Sequence[str | Sequence[Message]],
        *,
        abandoned_call_log: AbandonedCallLog | None,
    ) -> list[Response[OutputT | None] | GenerationError]:
        """Run the items concurrently and return the settled list, order-aligned.

        Nothing leaves this frame on the raising path, so every item's history reaches
        abandoned_call_log or is lost. An item still running appends its own record while it unwinds
        its cancellation; an item that already settled cannot, so its record is appended here from
        the task's result, which carries it whether the item ended in a Response or a
        GenerationError.

        The cancelled item tasks are awaited before those appends because gather returns here with
        siblings still running on both arms: an item's non-GenerationError raise propagates
        immediately, and a cancellation propagates as soon as the first item completes as cancelled.
        Without the await a sibling's own append would land after the raise reached the caller, and
        the task.exception() reads below would hit tasks that are not done.

        Raises:
            asyncio.CancelledError: an outer scope cancelled generate_many; the items are cancelled
                and their outcomes are lost with the frame.
            BaseException: an item raised something that is not a GenerationError, a defect in
                langchaint itself; the remaining tasks are cancelled and it propagates.
        """
        tasks = [
            asyncio.create_task(
                self._generate_or_failure(conversation, abandoned_call_log=abandoned_call_log)
            )
            for conversation in conversations
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                if not task.cancelled() and task.exception() is None:
                    _append_abandoned_call(abandoned_call_log, task.result().call)
            raise

    @overload
    def stream_one(
        self: "BoundLLM[str, ToolsT]",
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> StreamHandle[str]: ...
    @overload
    def stream_one(
        self: "BoundLLM[OutputT, HasTools]",
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> StreamHandle[OutputT | None]: ...
    @overload
    def stream_one(
        self: "BoundLLM[OutputT, NoTools]",
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = ...,
    ) -> StreamHandle[OutputT]: ...
    def stream_one(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = None,
    ) -> StreamHandle[Any]:
        """Build the stream handle; entering it with `async with` opens the request.

        The handle's final() Response types output the way generate_one types it, per binding.
        A bare str is shorthand for a conversation of one UserMessage holding that text.
        Sync because nothing suspends until the handle is entered;
        see StreamHandle for the retry, close, and abandoned_call_log contracts.
        """
        return self._stream_one_any_binding(conversation, abandoned_call_log=abandoned_call_log)

    def _stream_one_any_binding(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None,
    ) -> StreamHandle[OutputT | None]:
        """Build the handle at the widest output type; _generate_one_any_binding says why."""
        return StreamHandle(
            adapter=self.adapter,
            bound_adapter=self._bound_adapter,
            conversation=_as_conversation(conversation),
            rate_limiter=self.rate_limiter,
            abandoned_call_log=abandoned_call_log,
        )
