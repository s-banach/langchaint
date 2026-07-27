"""Exception vocabulary.

One property decides an error's fate: whether a retry may fix it.
A TransientError is retried, everything else is not.
No NonRetriableError class exists; "non-retriable" simply means "not a TransientError".

Every non-retriable outcome is one item's result: generate_one raises it,
and generate_many returns it in that item's slot, so no item's failure cancels a sibling.
There is no error that dooms a whole batch, because langchaint cannot tell one apart from
an item's own rejection: a provider states a status, never whether the binding or this one
conversation caused it. A binding defect langchaint can detect raises at construction or
bind time instead, before any request is sent.

TransientError is a per-attempt control signal between an adapter and a retry loop, raised to no
application: a loop that cannot retry one reports it as RetryUnavailableError, carrying the call.
The GenerationError subclasses are terminal per-item results a to_row failure row is built from:
RetriesExhaustedError, RetryUnavailableError, RefusalError, MaxCompletionTokensExceededError,
EmptyTurnError, SchemaViolationError, ContextWindowExceededError, UnfinishedTurnError,
ProviderFailedTerminallyError, InvalidRequestError, and UnrecognizedError.

Classification of raw SDK exceptions into these lives in the adapter (Adapter.classify);
a 200 that produced no output is a normal response that never reaches classify,
so the adapter reports it as an AttemptOutcome member where it reads the response.
Every GenerationError is constructed by a retry loop, which is the only scope that knows a call's
attempts and timing; an adapter reports one attempt and never a GenerationError.

Three exceptions sit outside this axis, none of them a GenerationError.
DispatchExceptionGroup and InvalidToolArgsError belong to the tool layer, not the generate loop.
ToolManager.dispatch_many raises the group after every sibling dispatch settled.
It groups the tool-function defects and carries the settled calls' outcomes.
PydanticTool.validate_and_run raises InvalidToolArgsError when a tool call's args fail validation.
StreamProtocolError says a stream did not follow the event contract.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self, override

from pydantic import ValidationError

from langchaint.call import AttemptRecord, CallRecord, _CallCarrier
from langchaint.messages import StopReason
from langchaint.usage import Usage

if TYPE_CHECKING:
    # Type-only: tools.py imports this module at runtime, so importing the dispatch outcome types
    # here at runtime would be a cycle. The annotations below quote them.
    from langchaint.tools import DispatchManyOutcome


class TransientError(Exception):
    """One failed attempt that a retry may fix.

    __cause__ holds the original provider exception when one exists.
    retry_after_seconds is the server-stated wait parsed from the response's retry-after headers,
    when the provider sent one;
    RateLimiter honors it up to a 60-second cap and uses it to pause admission account-wide.
    is_rate_limit marks the errors Adapter.classify returned "rate_limit" for;
    RateLimiter pauses admission on them and requires a successful probe request before resuming full admission.
    No billing fields: the attempt that reached a billable 200 is recorded carrying this error, so
    what it cost and the response it was read from are on that AttemptRecord.
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


def _extract_transient_errors(
    attempt_records: Sequence[AttemptRecord],
) -> tuple[TransientError, ...]:
    """Return the errors of the failed attempts, in order.

    The fold RetriesExhaustedError and RateLimiter.delay_seconds consume;
    on a failure this is every record's error, on a success all but the last.
    """
    return tuple(record.error for record in attempt_records if record.error is not None)


def _join_error_text(attempt_records: Sequence[AttemptRecord]) -> str:
    return "; ".join(
        f"attempt {index + 1}: {record.error}" for index, record in enumerate(attempt_records)
    )


class GenerationError(_CallCarrier, Exception):
    """A terminal per-item generate result that becomes a to_row failure row.

    The base for the non-retriable per-item outcomes:
    RetriesExhaustedError (the retry budget ran out on transient errors),
    RetryUnavailableError (a transient failure on a stream, which retries only failures raised
    before its first item),
    RefusalError (no structured output: the model refused or a provider filter blocked the turn),
    MaxCompletionTokensExceededError (the structured response hit the token cap before its JSON parsed),
    EmptyTurnError (the model finished the turn and produced nothing),
    SchemaViolationError (the model finished the turn and its text is not an instance of the
    bound response_format),
    ContextWindowExceededError (the conversation overflowed the model's context window),
    UnfinishedTurnError (the 200 is not a finished turn, so langchaint cannot report it as the answer),
    ProviderFailedTerminallyError (the 200's body reports that generating the response failed),
    InvalidRequestError (the request was rejected, by the provider or by the adapter before sending), and
    UnrecognizedError (the adapter did not recognize the attempt's error).
    generate_one raises any of them;
    generate_many returns each in the slot of the item it belongs to,
    so to_row renders a uniform failure row.

    call is this call's history; model, provider_name, attempt_records, and elapsed_seconds read
    off it, and with stop_reason they mirror the fields a success Response carries
    so to_row fills the same row shape from either.
    usage (carrying cost_in_usd) is the paid total summed from the records
    (a refusal or truncation reads its one completed attempt;
    a retry-exhausted item sums its records, near zero when they were transport failures);
    attempts and error_text are derived from the records too.
    A caller recovers each attempt's turn and its raw provider response from attempt_records.

    Only a retry loop constructs one of these, because only a loop knows the attempts and the timing,
    and every field is set in the constructor. An adapter reports what one attempt produced (an
    AttemptOutcome arm) and never a GenerationError, so none exists in a half-built state.
    """

    call: CallRecord
    usage: Usage

    def __init__(self, *, call: CallRecord) -> None:
        """Store the call and fold the paid total from its records."""
        super().__init__()
        self.call = call
        self.usage = Usage.sum_of(record.usage for record in call.attempt_records)

    @property
    def stop_reason(self) -> StopReason | None:
        """None: no turn completed. Five subclasses override it, because a turn did.

        RefusalError, MaxCompletionTokensExceededError, EmptyTurnError, SchemaViolationError, and
        ContextWindowExceededError each fix their own value; every other subclass inherits None.

        Fixed by the class rather than taken as a constructor argument, because a raise site must
        not choose a value the subclass already fixes. to_row and gen_ai_attributes both read it
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
    def attempts(self) -> int:
        """Requests langchaint observed going out: one attempt record each.

        Below the requests sent when the adapter could not read the last attempt's error, which is
        the count an UnrecognizedError costs.
        """
        return len(self.attempt_records)

    @property
    def error_text(self) -> str:
        """The failure-row error cell; RetriesExhaustedError folds its attempt chain instead."""
        return str(self)


class RetriesExhaustedError(GenerationError):
    """Every attempt failed with a transient error, and the budget ran out.

    generate_one raises it;
    generate_many returns it in the row where an item exhausted its retries,
    so the same object is both the raised failure and the failure row of a batch.
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

    Raised only inside a stream_one block, on the two transient failures a stream does not retry:
    one an item pull raised after the caller already held part of the turn, and one the assembled
    response reported, which arrives when the stream is already over. Either ends the call whatever
    the retry budget still holds. A stream that failed before its first item is reopened instead,
    under that budget. Reopen after this error by calling stream_one again with the same conversation.

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
    Not retried: the turn completed on the terms it reports, so nothing about the attempt was
    transient.
    The fix is a model the schema can express, or validation moved out of the model and into the
    caller's own code, where a rejected turn is data rather than a failed generation.

    validation_error_json is pydantic's rejection as JSON, the rejected values included.
    None of it is in the message: pydantic writes the caller's own validator text into each msg, so a
    validator naming the value it rejected would put generated content in every span.
    """

    validation_error_json: str

    def __init__(self, *, validation_error_json: str, call: CallRecord) -> None:
        """Store what the validation rejected, then the call."""
        self.validation_error_json = validation_error_json
        super().__init__(call=call)

    @property
    @override
    def stop_reason(self) -> Literal["end_turn"]:
        """The provider completed the turn; what it wrote is not the bound model."""
        return "end_turn"

    @override
    def _summary(self) -> str:
        return "the turn's text is not an instance of the bound response_format"


class ContextWindowExceededError(GenerationError):
    """The conversation overflowed the model's context window.

    Not retried: the same conversation overflows identically on every attempt.
    The fix is a shorter conversation or a model with a larger window.
    """

    @property
    @override
    def stop_reason(self) -> Literal["context_window_exceeded"]:
        """The provider reported the overflow as the turn's stop reason."""
        return "context_window_exceeded"

    @override
    def _summary(self) -> str:
        return "the conversation exceeded the model's context window"


class UnfinishedTurnError(GenerationError):
    """The provider returned a 200 that is not a finished turn, so its content is not the answer.

    Continuing such a turn means sending its content back for the provider to resume, which langchaint
    has no code for. Returning the partial content as the answer would be silently wrong data, so the
    item fails instead.
    Not retried: nothing about the next attempt makes langchaint able to continue the turn.

    reason names what the provider reported, in the provider's own word, and is the whole message.
    """

    reason: str

    def __init__(self, *, reason: str, call: CallRecord) -> None:
        """Store what the provider reported, then the call."""
        self.reason = reason
        super().__init__(call=call)

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

    def __init__(self, *, reason: str, call: CallRecord) -> None:
        """Store what the provider reported, then the call."""
        self.reason = reason
        super().__init__(call=call)

    @override
    def _summary(self) -> str:
        return f"the provider reported that generating the response failed: {self.reason}"


class InvalidRequestError(GenerationError):
    """The provider or the adapter rejected this one request; the item fails as a row.

    Two sources, both meaning the request as sent (or as it would have been sent) is not acceptable:
    the provider's own rejection, which Adapter.classify returns "invalid_request" for, and the
    adapter reporting the conversation as a InvalidRequest outcome, because it cannot be put on the
    wire with the meaning the message states.
    Not retried: the same request would be rejected the same way.

    The provider's rejection is every 4xx the retry policy declines, not only a rejected conversation.
    A bad API key, a permission failure, and an unknown model id land here too.
    A caller separating them reads status_code off __cause__, which holds the exception classify saw.
    Both shipped adapters return "invalid_request" only for an APIStatusError (anthropic 0.120.0, openai 2.45.0).
    __cause__ is None on the InvalidRequest source, where nothing went out.

    Behaviorally this is UnrecognizedError (one row, no retry); it is a separate class because
    Adapter.classify's contract is that "unrecognized" means the adapter could not name the error,
    and a rejection it does name is not that.

    reason states what was rejected, and is the whole error message.
    """

    reason: str

    def __init__(self, *, reason: str, call: CallRecord) -> None:
        """Store the rejection, then the call."""
        self.reason = reason
        super().__init__(call=call)

    @override
    def _summary(self) -> str:
        return self.reason


class UnrecognizedError(GenerationError):
    """A provider error the adapter cannot name; the item fails as a row.

    Adapter.classify's default: not a known transient or rate-limit condition (which retry), and not
    a rejection of this request (which is InvalidRequestError), so the safe treatment is to fail this
    item visibly.
    It covers both an error langchaint has no account of, which may be a defect (in langchaint, the
    SDK, or the provider), and one whose disposition is known while its identity is not: a server
    error the provider declares final by responding x-should-retry: false.
    Not retried: a defect must surface rather than be retried silently at billing expense,
    and a failure the provider calls final would not survive a retry either.
    error is the unrecognized exception, also chained as __cause__.
    attempt_records covers the prior attempts; the unrecognized attempt itself has no record,
    because its billing is unobservable through an exception the adapter cannot read.
    So both usage and attempts are short by that attempt: langchaint cannot say what it billed,
    nor even that it reached the provider.
    """

    def __init__(self, *, error: Exception, call: CallRecord) -> None:
        """Store the unrecognized exception, then the call."""
        self.error = error
        super().__init__(call=call)

    @override
    def _summary(self) -> str:
        return f"unrecognized provider error: {self.error}"


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
    A CancelledError is never a member: ExceptionGroup rejects a BaseException that is not an Exception,
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
