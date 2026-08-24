"""Exception vocabulary.

`TransientError` marks a retriable attempt.
`GenerationError` carries one terminal call outcome without cancelling sibling calls.
Adapters classify SDK exceptions; retry loops construct `GenerationError` values from call records.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self, override

from pydantic import ValidationError

from langchaint.call import AttemptRecord, CallRecord, _CallCarrier
from langchaint.messages import AssistantMessage, StopReason
from langchaint.pricing import Billing
from langchaint.usage import Usage

if TYPE_CHECKING:
    # Runtime imports of these types would create cycles through `tools.py` and `adapter.py`.
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
        self.retry_after_seconds: float | None = retry_after_seconds
        self.is_rate_limit: bool = is_rate_limit


class EmbeddingOutputError(RuntimeError):
    """A provider returned unusable embedding vectors."""


def _extract_transient_errors(
    attempt_records: Sequence[AttemptRecord],
) -> tuple[TransientError, ...]:
    """Return the errors of the failed attempts, in order."""
    return tuple(record.error for record in attempt_records if record.error is not None)


def _join_error_text(attempt_records: Sequence[AttemptRecord]) -> str:
    return "; ".join(
        f"attempt {index + 1}: {record.error}" for index, record in enumerate(attempt_records)
    )


class GenerationError(_CallCarrier, Exception):
    """A terminal per-item generation result.

    `call` preserves attempt timing, billing, turns, and raw responses.
    `request` preserves the sent request when one exists and stays outside `error_text`.
    `usage` sums paid usage across recorded attempts.
    `generate_one` raises subclasses.
    `generate_many` returns subclasses at their input positions.
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
        """Return `None`."""
        return None

    def _summary(self) -> str:
        return "generation failed"

    @override
    def __str__(self) -> str:
        """Render the reason, computed on demand so it never depends on when the fields were set."""
        return self._summary()

    @property
    def assistant_message(self) -> AssistantMessage | None:
        """Return the last recorded `assistant_message`, or `None`."""
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

        `RetriesExhaustedError` includes its complete transient-error chain.
        """
        return str(self)


class RetriesExhaustedError(GenerationError):
    """Every attempt failed with a transient error, and the budget ran out."""

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
    """A transient failure from an open stream that `stream_one` cannot retry.

    Call `stream_one` again to retry.
    The final attempt's `TransientError` is also `__cause__`.
    """

    @override
    def _summary(self) -> str:
        errors = _extract_transient_errors(self.attempt_records)
        last = str(errors[-1]) if errors else "no attempts recorded"
        return f"no retry was available for a transient failure: {last}"


class RefusalError(GenerationError):
    """No structured output: the model refused, or a provider content filter blocked the turn.

    This error occurs only on the structured path because no validated instance exists.
    The text path returns a `Response` with `stop_reason="refusal"`.
    langchaint does not retry refusals.
    Retrying resamples and spends nonzero input tokens, including cache reads when warm.
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
    """The structured response reached `max_completion_tokens` before its JSON parsed.

    This error occurs only on the structured path.
    The text path returns a `Response` with `stop_reason="max_tokens"`.
    langchaint does not retry this error.
    Use `rebind` with a larger `max_completion_tokens`.
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
    """A structured turn produces no instance or tool call."""

    @property
    @override
    def stop_reason(self) -> Literal["end_turn"]:
        """Return the stop reason for a turn with no value the binding could return."""
        return "end_turn"

    @override
    def _summary(self) -> str:
        return "the model completed its turn without producing output"


class SchemaViolationError(GenerationError):
    """The turn's text fails the bound `response_format` validation.

    `validation_error_json` contains pydantic's errors and rejected values.
    It stays outside the exception message because validator text can contain generated content.
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
        """Return the stop reason for text that does not validate as the bound model."""
        return "end_turn"

    @override
    def _summary(self) -> str:
        return "the turn's text is not an instance of the bound response_format"


class ContextWindowExceededError(GenerationError):
    """The request overflowed the model's context window.

    langchaint does not retry this error because the same request overflows on every attempt.
    Use a shorter `GenerationInput` or a model with a larger context window.
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
    """The provider returns a partial turn that langchaint cannot continue.

    `reason` preserves the provider's description and becomes the exception message.
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
    """A billable response reports a terminal generation failure.

    `reason` preserves the provider's description in the exception message.
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
    """The adapter or provider rejects one request.

    `request` is `None` when `build_request` rejects messages before constructing a request.
    Provider rejections preserve their SDK exception as `__cause__`.
    `reason` is the exception message.
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
    """The provider marks an error terminal.

    `error` preserves the provider's message and is also chained as `__cause__`.
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
    """An exception `Adapter.classify` cannot place.

    `error` is also chained as `__cause__`.
    The attempt record contains billing only when a response or open stream reported it.
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
    """An exception that escapes langchaint's generation handling.

    `error` is also chained as `__cause__` and supplies `error_text`.
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
    """A call whose result did not reach the caller.

    `in_flight_attempt_started_at_monotonic_seconds` identifies an unfinished request.
    `billing_in_flight` contains billing reported before interruption.
    `usage` sums settled attempts and `billing_in_flight`.
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

        Counting the request keeps the calls row consistent with the `to_tables` attempts rows.
        """
        if self.in_flight_attempt_started_at_monotonic_seconds is None:
            return len(self.attempt_records)
        return len(self.attempt_records) + 1

    @override
    def _summary(self) -> str:
        return "the call was cut off before its result reached the caller"


class TimedOutError(AbandonedCallError):
    """A langchaint deadline expires before the call returns.

    `generate_one.timeout_seconds` includes admission waits.
    `generate_many.max_working_seconds_per_item` excludes admission waits.
    """

    @override
    def _summary(self) -> str:
        return "the call timed out before it produced a result"


class InvalidToolArgsError(Exception):
    """A tool call's `args_json` failed validation against the tool's `args_model`.

    `PydanticTool._validated_args` is the only langchaint function that raises this error.
    A tool function does not cause langchaint to raise this error.
    `ToolManager.dispatch` catches it and returns `DispatchInvalidToolArgs`.
    """

    def __init__(self, validation_error: ValidationError) -> None:
        """Hold `validation_error` by reference.

        Args:
            validation_error: The tool argument validation failure.
        """
        super().__init__()
        self.validation_error: ValidationError = validation_error

    @override
    def __str__(self) -> str:
        """Render the held ValidationError as its own multi-line string."""
        return str(self.validation_error)


class DispatchExceptionGroup(ExceptionGroup[Exception]):
    """Tool function exceptions from `ToolManager.dispatch_many`.

    `completed_outcomes` preserves settled outcomes in input order.
    The grouped exceptions also preserve input order and their tracebacks.
    Cancellation propagates separately with this group as its cause when both occur.
    """

    completed_outcomes: "tuple[DispatchManyOutcome, ...]"

    def __new__(
        cls,
        message: str,
        exceptions: Sequence[Exception],
        *,
        completed_outcomes: "tuple[DispatchManyOutcome, ...]",
    ) -> Self:
        """Pass `message` and `exceptions` to the base `__new__`."""
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

        `BaseException.__init__` accepts only positional arguments.
        This override stores the keyword-only `completed_outcomes`.
        """
        super().__init__(message, exceptions)
        self.completed_outcomes = completed_outcomes

    @override
    # pyrefly: ignore[bad-override]  # Typeshed makes `derive` generic for each call.
    # No concrete subclass override can satisfy the generic signature.
    def derive(self, excs: Sequence[Exception], /) -> "DispatchExceptionGroup":
        """Rebuild a subgroup carrying the same completed_outcomes.

        Args:
            excs: The subgroup exceptions.
        """
        return DispatchExceptionGroup(
            self.message, excs, completed_outcomes=self.completed_outcomes
        )


class StreamProtocolError(Exception):
    """A stream did not follow the event contract.

    Raised when a stream ends without a terminal result.
    This includes a missing Messages API stop reason or Responses API terminal response.
    It also includes a `StreamHandle` that ends without an adapter stream.
    `AdapterStream.final()` may raise it when called before `AdapterStream.items()` is exhausted.
    """


class GaveUpWaiting(Exception):  # noqa: N818 (the interface names the outcome, not an error kind)
    """The `budget` expired before SharedBackoff.admitted's entry admitted the request.

    Entry holds nothing: no permit, no queue place, and nothing recorded.
    A fresh attempt joins the same queue behind the same pause, so do not resubmit at once.
    """


class ParserContractError(Exception):
    """A SharedBackoff parse raised.

    A defect in `parse`, not a provider classification: fix `parse` rather than retrying through it.
    __cause__ holds what `parse` raised, where it raised.
    Behind that as context sits the provider failure `parse` was given.
    Nothing was recorded, and Admission.verdict is None.
    """
