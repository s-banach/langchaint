"""Exception vocabulary.

One property decides an error's fate: whether a retry may fix it.
A TransientError is retried, everything else is not.
No NonRetriableError class exists; "non-retriable" simply means "not a TransientError".

No item's failure cancels a sibling: a non-retriable outcome fails the one item it belongs to.
GenerationError says which of its subclasses each generate method delivers.
There is no error that dooms a whole batch, because langchaint cannot tell one apart from
an item's own rejection: a provider states a status, never whether the binding or this one
request caused it. A binding defect langchaint can detect raises at construction or
bind time instead, before any request is sent.

TransientError is a per-attempt control signal between an adapter and a retry loop, raised to no
application: a loop that cannot retry one reports it as RetryUnavailableError, carrying the call.
The GenerationError subclasses are terminal per-item results a to_tables failure row is built from:
RetriesExhaustedError, RetryUnavailableError, RefusalError, MaxCompletionTokensExceededError,
EmptyTurnError, SchemaViolationError, ContextWindowExceededError, UnfinishedTurnError,
ProviderFailedTerminallyError, InvalidRequestError, ProviderDeclaredFinalError,
UnknownExceptionError, EscapedExceptionError, AbandonedCallError, and TimedOutError.

Classification of raw SDK exceptions into these lives in the adapter (Adapter.classify);
a 200 that produced no output is a normal response that never reaches classify,
so the adapter reports it as a ResponseOutcome variant where it reads the response.
Every GenerationError is constructed by a scope holding the call's ledger, the only scope that knows
its attempts and timing; an adapter reports one attempt and never a GenerationError.

The following exceptions sit outside this axis and are not `GenerationError` subclasses.
DispatchExceptionGroup and InvalidToolArgsError belong to the tool layer, not the generate loop.
ToolManager.dispatch_many raises the group after every sibling dispatch settled.
It groups the tool-function defects and carries the settled calls' outcomes.
PydanticTool.validate_and_run raises InvalidToolArgsError when a tool call's args fail validation.
StreamProtocolError says a stream did not follow the event contract.
GaveUpWaiting and ParserContractError belong to SharedBackoff.
The first reports an admitted() entry whose budget expired; the second reports a parse that violated its contract.
EmbeddingOutputError reports a malformed successful embedding response.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self, override

from pydantic import ValidationError

from langchaint.call import AttemptRecord, CallRecord, _CallCarrier
from langchaint.messages import AssistantMessage, StopReason
from langchaint.pricing import Billing
from langchaint.usage import Usage

if TYPE_CHECKING:
    # Type-only: tools.py imports this module at runtime, so importing the dispatch outcome types
    # here at runtime would be a cycle, and adapter.py imports tools.py, so RequestParams is one too.
    # The annotations below quote both. Nothing here constructs either: a retry loop hands
    # GenerationError the request it built, and ToolManager builds the dispatch outcomes.
    from langchaint.adapter import RequestParams
    from langchaint.tools import DispatchManyOutcome


class TransientError(Exception):
    """One failed attempt that a retry may fix.

    `__cause__` holds the original provider exception when one exists.
    Retry loops raise `TransientError` inside `SharedBackoff.admitted()`.
    Attempt records store the capped wait on their `TransientError`.
    Billing remains on the same `AttemptRecord`.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        is_rate_limit: bool = False,
    ) -> None:
        """Store the server-stated wait and the rate-limit classification."""
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.is_rate_limit = is_rate_limit


class EmbeddingOutputError(RuntimeError):
    """A provider returned unusable embedding vectors."""


def _extract_transient_errors(
    attempt_records: Sequence[AttemptRecord],
) -> tuple[TransientError, ...]:
    """Return the errors of the failed attempts, in order.

    The fold RetriesExhaustedError consumes;
    on a failure this is every record's error, on a success all but the last.
    """
    return tuple(record.error for record in attempt_records if record.error is not None)


def _join_error_text(attempt_records: Sequence[AttemptRecord]) -> str:
    return "; ".join(
        f"attempt {index + 1}: {record.error}" for index, record in enumerate(attempt_records)
    )


class GenerationError(_CallCarrier, Exception):
    """A terminal per-item generate result that becomes a to_tables failure row.

    The base for the non-retriable per-item outcomes:
    RetriesExhaustedError (the retry budget ran out on transient errors),
    RetryUnavailableError (a transient failure from an open stream),
    RefusalError (no structured output: the model refused or a provider filter blocked the turn),
    MaxCompletionTokensExceededError (the structured response hit the token cap before its JSON parsed),
    EmptyTurnError (the model finished the turn and produced nothing),
    SchemaViolationError (the model finished the turn and its text is not an instance of the
    bound response_format),
    ContextWindowExceededError (the request overflowed the model's context window),
    UnfinishedTurnError (the 200 is not a finished turn, so langchaint cannot report it as the answer),
    ProviderFailedTerminallyError (the 200's body reports that generating the response failed),
    InvalidRequestError (the request was rejected, by the provider or by the adapter before sending),
    ProviderDeclaredFinalError (the provider answered with a final error),
    UnknownExceptionError (the adapter could not place the attempt's exception),
    EscapedExceptionError (an exception escaped langchaint's own machinery),
    AbandonedCallError (the call was cut off before its result reached the caller), and
    TimedOutError (the call was cut off by the deadline the caller asked for).
    generate_one raises every one of them but two: RetryUnavailableError, which comes only from a
    stream_one block, and AbandonedCallError itself, which is never raised.
    generate_many returns each of the ones generate_one raises at that item's index,
    so to_tables renders a uniform failure row.

    call is this call's history; model, provider_name, attempt_records, started_at_monotonic_seconds,
    and elapsed_seconds read off it, and with stop_reason they mirror the fields a success Response carries
    so to_tables fills the same columns from either.
    request is what every attempt of the call sent, which is what someone reading a failure asks for
    next. None where no request is in scope to carry: where build_request returned
    InvalidRequest, having built none, on EscapedExceptionError, whose exception can come from
    anywhere, and on AbandonedCallError, a cancellation being nobody's request defect.
    No success carries one: build_request is a function of the Sequence[Message] and the binding the
    caller still holds, so a call that came back is reconstructible without archiving what it sent.
    It holds the whole prompt, so it stays off error_text and __str__.
    usage (carrying cost_in_usd) is the paid total summed from the records
    (a refusal or truncation reads its one completed attempt;
    a retry-exhausted item sums its records, near zero when they were transport failures);
    attempts and error_text are derived from the records too.
    A caller recovers each attempt's turn and its raw provider response from attempt_records.

    Only a scope holding the call's ledger constructs one of these, because only it knows the
    attempts and the timing, and every field is set in the constructor. An adapter reports what one
    attempt produced (a ResponseOutcome variant) and never a GenerationError, so none exists in a
    half-built state.
    """

    call: CallRecord
    usage: Usage
    request: "RequestParams | None"

    def __init__(self, *, call: CallRecord, request: "RequestParams | None") -> None:
        """Store the call and what it sent, and fold the paid total from the records."""
        super().__init__()
        self.call = call
        self.request = request
        self.usage = Usage.sum_of(record.usage for record in call.attempt_records)

    @property
    def stop_reason(self) -> StopReason | None:
        """None: no turn completed. Five subclasses override it, because a turn did.

        RefusalError, MaxCompletionTokensExceededError, EmptyTurnError, SchemaViolationError, and
        ContextWindowExceededError each fix their own value; every other subclass inherits None.

        Fixed by the class rather than taken as a constructor argument, because a raise site must
        not choose a value the subclass already fixes. to_tables and gen_ai_attributes both read it
        off Response | GenerationError, so this property is what spares each an isinstance ladder
        over the subclasses.
        """
        return None

    def _summary(self) -> str:
        """Return the exception message; each subclass overrides this with its own reason."""
        return "generation failed"

    @override
    def __str__(self) -> str:
        """Render the reason, computed on demand so it never depends on when the fields were set."""
        return self._summary()

    @property
    def assistant_message(self) -> AssistantMessage | None:
        """The turn of the last attempt that produced one, None where no attempt did.

        Every attempt that reached a billable 200 records the turn it carried, so a failure the
        provider generated content for reaches that content here, and a call whose attempts all
        failed before a 200 has none.
        It is the field a success Response carries under the same name, so the tracing layer reads
        one name off either.
        Content, so it stays off error_text and __str__.
        """
        for record in reversed(self.attempt_records):
            if record.assistant_message is not None:
                return record.assistant_message
        return None

    @property
    def attempts(self) -> int:
        """Requests langchaint observed going out: one attempt record each."""
        return len(self.attempt_records)

    @property
    def error_text(self) -> str:
        """The whole failure as one string, which the tracing layer sets as the call span's status.

        RetriesExhaustedError overrides it to fold its attempt chain in, because a chain of transient
        errors is the failure and no one of them is.
        """
        return str(self)


class RetriesExhaustedError(GenerationError):
    """Every attempt failed with a transient error, and the budget ran out.

    generate_one raises it;
    generate_many returns it at the index of the item that exhausted its retries,
    so the same object is both the raised failure and the failure result of a batch.
    errors_from_attempts is derived from attempt_records.
    """

    @override
    def _summary(self) -> str:
        errors = _extract_transient_errors(self.attempt_records)
        last = str(errors[-1]) if errors else "no attempts recorded"
        return f"{len(errors)} attempts failed; last: {last}"

    @property
    def errors_from_attempts(self) -> tuple[TransientError, ...]:
        """The failed attempts' errors, in order."""
        return _extract_transient_errors(self.attempt_records)

    @property
    @override
    def error_text(self) -> str:
        """The folded failure chain, one entry per attempt."""
        return _join_error_text(self.attempt_records)


class RetryUnavailableError(GenerationError):
    """A transient failure a retry may have fixed, on a call where no retry was available.

    Raised only inside a stream_one block.
    An iterator failure can raise it after open_stream() returns.
    An assembled response can also report a transient failure.
    Either ends the call regardless of the remaining retry budget.
    Call stream_one again with the same GenerationInput to retry.

    The failure itself is the TransientError on the last attempt record, and is __cause__ as well:
    it carries the adapter's own message, retry_after_seconds, and is_rate_limit.
    """

    @override
    def _summary(self) -> str:
        errors = _extract_transient_errors(self.attempt_records)
        last = str(errors[-1]) if errors else "no attempts recorded"
        return f"no retry was available for a transient failure: {last}"


class RefusalError(GenerationError):
    """No structured output: the model refused, or a provider content filter blocked the turn.

    Fires only on the structured path, where neither leaves an instance to return;
    the text path surfaces a refusal as a Response with stop_reason "refusal".
    Not retried, by policy:
    a refusal can flip under sampling,
    but retrying spends the full input tokens
    (cache-read rate when warm, never zero) on an expected-value bet langchaint does not take by default.
    An app whose economics differ overrides the adapter's _parsed_output.
    """

    @property
    @override
    def stop_reason(self) -> Literal["refusal"]:
        """The turn the provider completed and the adapter rejected ended in a refusal."""
        return "refusal"

    @override
    def _summary(self) -> str:
        return "no structured output: the model refused or a provider filter blocked the turn"


class MaxCompletionTokensExceededError(GenerationError):
    """The structured response reached max_completion_tokens before its JSON parsed.

    Fires only on the structured path; the text path surfaces the cap as a Response with stop_reason "max_tokens".
    Not retried, unconditionally:
    the attempt already generated the full token cap,
    the most expensive possible response, and a resample under the same cap truncates again.
    The fix is a larger max_completion_tokens via rebind.
    """

    @property
    @override
    def stop_reason(self) -> Literal["max_tokens"]:
        """The turn the provider completed and the adapter rejected hit the token cap."""
        return "max_tokens"

    @override
    def _summary(self) -> str:
        return "the structured response reached max_completion_tokens before its JSON parsed"


class EmptyTurnError(GenerationError):
    """The model finished its turn and produced no instance and no tool call.

    Fires only on the structured path; a text binding returns "" for the same turn.
    Not retried: the model stopped on its own terms, so a resend is a fresh sample rather than a
    recovery, and it bills the full input again.
    A turn that spent its budget on reasoning and emitted no text lands here, and what it generated
    is on the attempt record.
    """

    @property
    @override
    def stop_reason(self) -> Literal["end_turn"]:
        """The provider completed the turn; it was empty of anything the binding could return."""
        return "end_turn"

    @override
    def _summary(self) -> str:
        return "the model completed its turn without producing output"


class SchemaViolationError(GenerationError):
    """The model finished its turn and its text is not an instance of the bound response_format.

    Fires only on the structured path; a text binding returns the same turn's text as its output.
    Not retried: the text failed the caller's own model, so what to do about it is the caller's
    decision, as it is when a tool call's args fail the tool's args_model.
    The fix is a model the schema can express, or validation moved out of the model and into the
    caller's own code, where a rejected turn is data rather than a failed generation.

    validation_error_json is pydantic's rejection as JSON, the rejected values included.
    None of it is in the message: pydantic writes the caller's own validator text into each msg, so a
    validator naming the value it rejected would put generated content in every span.
    """

    validation_error_json: str

    def __init__(
        self, *, validation_error_json: str, call: CallRecord, request: "RequestParams | None"
    ) -> None:
        """Store what the validation rejected, then the call."""
        self.validation_error_json = validation_error_json
        super().__init__(call=call, request=request)

    @property
    @override
    def stop_reason(self) -> Literal["end_turn"]:
        """The provider completed the turn; what it wrote is not the bound model."""
        return "end_turn"

    @override
    def _summary(self) -> str:
        return "the turn's text is not an instance of the bound response_format"


class ContextWindowExceededError(GenerationError):
    """The request overflowed the model's context window.

    Not retried: the same request overflows identically on every attempt.
    The fix is a shorter GenerationInput or a model with a larger window.
    """

    @property
    @override
    def stop_reason(self) -> Literal["context_window_exceeded"]:
        """The provider reported the overflow as the turn's stop reason."""
        return "context_window_exceeded"

    @override
    def _summary(self) -> str:
        return "the request exceeded the model's context window"


class UnfinishedTurnError(GenerationError):
    """The provider returned a 200 that is not a finished turn, so its content is not the answer.

    Continuing such a turn means sending its content back for the provider to resume, which langchaint
    has no code for. Returning the partial content as the answer would be silently wrong data, so the
    item fails instead.
    Not retried: nothing about the next attempt makes langchaint able to continue the turn.

    reason names what the provider reported, in the provider's own word, and is the whole message.
    """

    reason: str

    def __init__(self, *, reason: str, call: CallRecord, request: "RequestParams | None") -> None:
        """Store what the provider reported, then the call."""
        self.reason = reason
        super().__init__(call=call, request=request)

    @override
    def _summary(self) -> str:
        return self.reason


class ProviderFailedTerminallyError(GenerationError):
    """The provider returned a 200 whose body reports that generating the response failed.

    A 200 the provider bills and fills with a failure rather than a turn. Whatever the body names is
    a property of the request, so the item fails once instead of buying the same body for the whole
    retry budget.

    reason is the provider's own description of the failure, carried unabridged in the message after
    a prefix naming what failed.
    stop_reason is None: no turn completed.
    """

    reason: str

    def __init__(self, *, reason: str, call: CallRecord, request: "RequestParams | None") -> None:
        """Store what the provider reported, then the call."""
        self.reason = reason
        super().__init__(call=call, request=request)

    @override
    def _summary(self) -> str:
        return f"the provider reported that generating the response failed: {self.reason}"


class InvalidRequestError(GenerationError):
    """The provider or the adapter rejected this one request; the item fails on its own.

    Two sources, both meaning the request as sent (or as it would have been sent) is not acceptable:
    the provider's own rejection, which Adapter.classify returns "invalid_request" for, and
    build_request returning InvalidRequest, because the messages cannot be put on the wire
    with the meaning they state.
    Not retried: the same request would be rejected the same way.
    request is None on the second source, where none was built, and holds what went out on the first.

    The provider's rejection is every 4xx the retry policy declines, not only a rejected GenerationInput.
    A bad API key, a permission failure, and an unknown model id land here too.
    A caller separating them reads status_code off __cause__, which holds the exception classify saw.
    Both shipped adapters return "invalid_request" only for an APIStatusError (anthropic 0.120.0, openai 2.45.0).
    __cause__ is None on the InvalidRequest source, where nothing went out.

    Behaviorally this is UnknownExceptionError (one failed item, no retry); it is a separate class because
    Adapter.classify's contract is that "unknown_exception" means the adapter could not place the
    exception, and a rejection it does place is not that.

    reason states what was rejected, and is the whole error message.
    """

    reason: str

    def __init__(self, *, reason: str, call: CallRecord, request: "RequestParams | None") -> None:
        """Store the rejection, then the call."""
        self.reason = reason
        super().__init__(call=call, request=request)

    @override
    def _summary(self) -> str:
        return self.reason


class ProviderDeclaredFinalError(GenerationError):
    """The provider answered with a final error; the item fails on its own.

    Two failures arrive this way: an error response the provider marked x-should-retry: false,
    which states the disposition and never what failed,
    and a mid-stream error event, raised carrying the live response's 200 status.
    Either way the provider's own message is the only description of the condition.
    Not retried: Adapter.parse returned a terminal verdict.
    error is the SDK exception carrying that message, also chained as __cause__.
    The attempt has a record: the request reached the provider, which answered.
    """

    error: Exception

    def __init__(
        self, *, error: Exception, call: CallRecord, request: "RequestParams | None"
    ) -> None:
        """Store the provider's exception, then the call."""
        self.error = error
        super().__init__(call=call, request=request)

    @override
    def _summary(self) -> str:
        return f"a final error from the provider: {self.error}"


class UnknownExceptionError(GenerationError):
    """An exception the adapter could not place; the item fails on its own.

    Adapter.classify's default: not a known transient or rate-limit condition (which retry), not a
    rejection of this request (which is InvalidRequestError), and not a final error from the
    provider (which is ProviderDeclaredFinalError), so the safe treatment is to fail this item visibly.
    It may be a defect, in langchaint, the SDK, or the provider.
    Not retried: a defect must surface rather than be retried silently at billing expense.
    error is the exception classify could not place, also chained as __cause__.
    The attempt it ends has a record wherever langchaint can account for it: a response had arrived
    and staged its billing, or an open stream had reported what the provider billed.
    Where nothing arrived, both usage and attempts are short by that attempt: langchaint cannot say
    what it billed, nor even that it reached the provider.
    """

    error: Exception

    def __init__(
        self, *, error: Exception, call: CallRecord, request: "RequestParams | None"
    ) -> None:
        """Store the unplaceable exception, then the call."""
        self.error = error
        super().__init__(call=call, request=request)

    @override
    def _summary(self) -> str:
        return f"langchaint could not place this exception: {self.error}"


class EscapedExceptionError(GenerationError):
    """An exception escaped langchaint's own machinery; the item fails on its own.

    The outermost guard around one call, so a defect anywhere between the caller and the adapter
    ends as that call's failure rather than as a raise past it. In a batch that is what keeps one
    item's defect from discarding the settled results and billing of its siblings.
    Report it as a defect in langchaint.
    Nothing is silenced: error is the escaped exception, also chained as __cause__, and error_text
    carries its text, which is the only description of a condition langchaint does not model.

    The attempt in flight when the exception escaped has a record where a response had arrived and
    staged its billing, which is what langchaint can account for.
    """

    error: Exception

    def __init__(self, *, error: Exception, call: CallRecord) -> None:
        """Store the escaped exception, then the call."""
        self.error = error
        super().__init__(call=call, request=None)

    @override
    def _summary(self) -> str:
        return f"an exception escaped langchaint: {self.error}"


class AbandonedCallError(GenerationError):
    """The account of one call whose result never reached the caller.

    A cancelled call returns no value, and the BaseException must propagate for the cancelling
    scope's teardown to run, so this is the only carrier of what the call did. StreamHandle sets one
    as its abandoned.
    A GenerationError like every other, so to_tables renders it to a failure row and error_summary
    names it.

    in_flight_attempt_started_at_monotonic_seconds says a request was in flight when the cut hit and
    when that request started, and is None where the cut fell between attempts. That attempt gets no
    AttemptRecord: an ending nobody observed would make the ending fields conditional on every
    record.
    billing_in_flight is what the provider had reported for that attempt by the time of the cut,
    and is None wherever it had reported nothing.
    A counter the provider sends late is missing from it.
    usage is the settled records' fold plus billing_in_flight's usage, the paid total this call is
    known to have incurred, which is usage's scope on every carrier. What the cut-off attempt billed
    beyond the provider's last report is unobservable client-side and none of it is fabricated.
    Reconciliation closes that gap from the provider's side, which model and provider_name identify.
    """

    billing_in_flight: Billing | None
    in_flight_attempt_started_at_monotonic_seconds: float | None

    def __init__(
        self,
        *,
        call: CallRecord,
        billing_in_flight: Billing | None,
        in_flight_attempt_started_at_monotonic_seconds: float | None,
    ) -> None:
        """Store the cut-off attempt's account, then add what it billed to the settled total."""
        super().__init__(call=call, request=None)
        self.billing_in_flight = billing_in_flight
        self.in_flight_attempt_started_at_monotonic_seconds = (
            in_flight_attempt_started_at_monotonic_seconds
        )
        if billing_in_flight is not None:
            self.usage = Usage.sum_of([self.usage, billing_in_flight.usage])

    @property
    @override
    def attempts(self) -> int:
        """Requests langchaint observed going out, counting the one cut off without a record.

        That request went out, so leaving it out would put a smaller number on the calls row than
        the attempts rows to_tables writes for the same call.
        """
        if self.in_flight_attempt_started_at_monotonic_seconds is None:
            return len(self.attempt_records)
        return len(self.attempt_records) + 1

    @override
    def _summary(self) -> str:
        return "the call was cut off before its result reached the caller"


class TimedOutError(AbandonedCallError):
    """The deadline the caller asked for expired before the call produced a result.

    Raised, where its base is only ever stored: langchaint owns the scope that expired, so the
    expiry surfaces inside the frame holding the call's ledger and leaves as an ordinary
    GenerationError. A scope the caller owns ends the call from outside that frame, so whatever it
    raises there carries no account of what the call spent.

    generate_one's timeout_seconds covers the whole call, so it can expire while a request is in
    flight, during a backoff sleep, or while waiting for SharedBackoff admission. That last case
    reports attempts == 0, which is how a caller tells a call that never sent from one the provider
    answered slowly.
    generate_many's max_working_seconds_per_item stops for the wait for admission, so it expires
    only with a request in flight or during a backoff sleep, and a batch item carrying it has
    always sent at least once.
    """

    @override
    def _summary(self) -> str:
        return "the call timed out before it produced a result"


class InvalidToolArgsError(Exception):
    """A tool call's args_json failed validation against the tool's args_model.

    Raised only by PydanticTool._validated_args, never by langchaint from the function,
    so catching it cannot swallow a function defect.
    This is model data the model can correct:
    ToolManager.dispatch catches it and returns a DispatchInvalidToolArgs
    holding the neutral InvalidToolArgsDetail tuple and an is_error ToolMessage.
    """

    def __init__(self, validation_error: ValidationError) -> None:
        """Hold the ValidationError by reference; __str__ derives the message from it."""
        super().__init__()
        self.validation_error = validation_error

    @override
    def __str__(self) -> str:
        """Render the held ValidationError as its own multi-line string."""
        return str(self.validation_error)


class DispatchExceptionGroup(ExceptionGroup[Exception]):
    """One or more tool functions raised during ToolManager.dispatch_many.

    Raised only after every sibling dispatch settled, so it carries what the batch still produced:
    completed_outcomes holds the settled calls' outcomes ordered by tool_calls position,
    a call answered through dispatch_many's precomputed argument included as its DispatchPrecomputed,
    each naming its call via tool_message.tool_call_id,
    so app_data a completed sibling produced (a billing record for money the tool spent) survives the raise,
    the same principle as GenerationError preserving on attempt_records what a 200 with no output billed.
    The grouped exceptions are user-code defects, dispatch's exceptions-propagate rule extended to a batch,
    ordered by tool_calls position; the ExceptionGroup base keeps every traceback in the report
    and supports except* handling.
    A CancelledError is never grouped here: ExceptionGroup rejects a BaseException that is not an Exception,
    and dispatch_many re-raises cancellation bare to keep its semantics.
    When defects co-occur with such a bare re-raise, this group still carries them,
    chained as the re-raised exception's __cause__ instead of being the raise itself.
    """

    completed_outcomes: "tuple[DispatchManyOutcome, ...]"

    def __new__(
        cls,
        message: str,
        exceptions: Sequence[Exception],
        *,
        completed_outcomes: "tuple[DispatchManyOutcome, ...]",
    ) -> Self:
        """Pass message and exceptions to the base __new__, which takes nothing else; __init__ stores the keyword."""
        group = super().__new__(cls, message, exceptions)
        group.completed_outcomes = completed_outcomes
        return group

    def __init__(
        self,
        message: str,
        exceptions: Sequence[Exception],
        *,
        completed_outcomes: "tuple[DispatchManyOutcome, ...]",
    ) -> None:
        """Store completed_outcomes and set args on the base.

        BaseException.__init__ takes only positional args, so without this override the keyword
        the constructor call carries would TypeError there.
        """
        super().__init__(message, exceptions)
        self.completed_outcomes = completed_outcomes

    @override
    # pyrefly: ignore[bad-override]  # typeshed types derive as generic per call
    # ([_ExceptionT](Sequence[_ExceptionT], /) -> ExceptionGroup[_ExceptionT]), which no concrete
    # subclass override can satisfy; this is the override pattern PEP 654 itself documents.
    def derive(self, excs: Sequence[Exception], /) -> "DispatchExceptionGroup":
        """Rebuild a subgroup carrying the same completed_outcomes.

        except* and split call this; without the override they would build a plain ExceptionGroup
        and the subgroup would silently lose completed_outcomes.
        """
        return DispatchExceptionGroup(
            self.message, excs, completed_outcomes=self.completed_outcomes
        )


class StreamProtocolError(Exception):
    """A stream did not follow the event contract.

    Raised where a stream ends without the terminal event carrying its result
    (no stop reason on the Messages API, no terminal response on the Responses API,
    or a StreamHandle that finished iterating with no adapter stream left to ask),
    and where final() is called before items() is exhausted, so no terminal response was captured.
    """


class GaveUpWaiting(Exception):  # noqa: N818 (the interface names the outcome, not an error kind)
    """The `budget` expired before SharedBackoff.admitted's entry admitted the request.

    Entry holds nothing: no permit, no queue place, and nothing recorded.
    A fresh attempt joins the same queue behind the same pause, so do not resubmit at once.
    """


class ParserContractError(Exception):
    """A SharedBackoff `parse` raised, or returned something that is not a verdict.

    A defect in `parse`, not a provider classification: fix `parse` rather than retrying through it.
    __cause__ holds what `parse` raised, where it raised.
    Behind that as context sits the provider failure `parse` was given.
    Nothing was recorded, and Admission.verdict is None.
    """
