"""Provider-neutral exception and normalized generation error records."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal, Self, override

from pydantic import ConfigDict, Field, ValidationError, model_validator

from langchaint.call import (
    AttemptProviderData,
    CallRecord,
    CutOffAttemptRecord,
    SettledAttemptRecord,
    TransientErrorRecord,
    _require_completed_model_turn,
    _settled_attempts,
)
from langchaint.checked_copy import CheckedCopyModel
from langchaint.messages import AssistantMessage, StopReason
from langchaint.usage import Usage

if TYPE_CHECKING:
    from langchaint.adapter import ErrorClassification, RequestParams
    from langchaint.tools import DispatchManyOutcome


class TransientError(Exception):
    """One failed attempt that a retry may fix.

    `__cause__` holds the original provider exception when one exists.
    Retry loops raise `TransientError` inside `SharedBackoff.admitted()`.
    `SettledAttemptRecord.error` preserves normalized failure data.
    `SettledAttemptRecord.billing` preserves billing from the same request.
    """

    retry_after_seconds: float | None
    is_rate_limit: bool

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        is_rate_limit: bool = False,
    ) -> None:
        """Store the server-stated wait and rate-limit classification."""
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.is_rate_limit = is_rate_limit


class EmbeddingOutputError(RuntimeError):
    """A provider returned unusable embedding vectors."""


_RECORD_CONFIG = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")


class _GenerationErrorRecordBase(CheckedCopyModel):
    """Shared normalized data and properties for terminal generation errors.

    Validation rejects unknown fields.
    """

    model_config = _RECORD_CONFIG

    call: CallRecord

    @property
    def attempt_records(
        self,
    ) -> tuple[SettledAttemptRecord | CutOffAttemptRecord, ...]:
        """Return the call's normalized attempt records."""
        return self.call.attempt_records

    @property
    def attempts(self) -> int:
        """Return requests that langchaint observed going out."""
        return len(self.call.attempt_records)

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every recorded request."""
        return Usage.sum_of(attempt.usage for attempt in self.call.attempt_records)

    @property
    def model(self) -> str:
        """Return the requested model id."""
        return self.call.model

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.call.provider_name

    @property
    def elapsed_seconds(self) -> float:
        """Return the complete call duration."""
        return self.call.elapsed_seconds

    @property
    def assistant_message(self) -> AssistantMessage | None:
        """Return the last recorded assistant message."""
        for attempt in reversed(self.call.attempt_records):
            if isinstance(attempt, SettledAttemptRecord) and attempt.assistant_message is not None:
                return attempt.assistant_message
        return None

    @property
    def stop_reason(self) -> StopReason | None:
        """Return the normalized stop reason when this variant defines one."""
        return None

    @property
    def error_summary(self) -> str:
        """Return the tracing-safe terminal failure summary."""
        return "generation failed"

    @property
    def error_text(self) -> str:
        """Return the tracing-safe complete failure text."""
        return self.error_summary

    @override
    def __str__(self) -> str:
        """Return `error_summary`."""
        return self.error_summary


class _CompletedModelTurnErrorRecordBase(_GenerationErrorRecordBase):
    """Shared validation for errors that require a completed model turn.

    Validation rejects unknown fields.
    """

    @model_validator(mode="after")
    def _validate_completed_model_turn(self) -> Self:
        _require_completed_model_turn(self.call)
        return self


def _require_retry_failures(call: CallRecord) -> tuple[SettledAttemptRecord, ...]:
    attempts = _settled_attempts(call)
    if not attempts:
        raise ValueError("call must contain at least one settled attempt")
    if any(attempt.error is None for attempt in attempts):
        raise ValueError("every attempt must contain a transient error")
    return attempts


def _require_terminal_provider_result(call: CallRecord, *, permit_empty: bool) -> None:
    attempts = _settled_attempts(call)
    if not attempts:
        if permit_empty:
            return
        raise ValueError("call must contain a final provider result")
    if any(attempt.error is None for attempt in attempts[:-1]):
        raise ValueError("every attempt before the final attempt must contain an error")
    final = attempts[-1]
    if final.error is not None:
        raise ValueError("the final attempt must be error-free")
    if final.assistant_message is not None:
        raise ValueError("the final provider result must not contain an assistant message")


def _require_abandoned_shape(call: CallRecord) -> None:
    settled = tuple(
        attempt for attempt in call.attempt_records if isinstance(attempt, SettledAttemptRecord)
    )
    final_is_cut_off = bool(call.attempt_records) and isinstance(
        call.attempt_records[-1], CutOffAttemptRecord
    )
    if final_is_cut_off:
        settled_prefix = settled
    elif settled and settled[-1].error is None:
        final = settled[-1]
        if final.assistant_message is not None:
            raise ValueError("the final settled request must be a terminal provider result")
        settled_prefix = settled[:-1]
    else:
        settled_prefix = settled
    if any(attempt.error is None for attempt in settled_prefix):
        raise ValueError("settled attempts before the terminal request must contain errors")


class RetriesExhaustedErrorRecord(_GenerationErrorRecordBase):
    """Every request failed transiently and the retry budget ended.

    Validation rejects unknown fields.
    """

    kind: Literal["retries_exhausted_error"] = "retries_exhausted_error"

    @model_validator(mode="after")
    def _validate_retry_failures(self) -> Self:
        _ = _require_retry_failures(self.call)
        return self

    @property
    def errors_from_attempts(self) -> tuple[TransientErrorRecord, ...]:
        """Return each attempt's normalized transient error."""
        return tuple(
            attempt.error for attempt in _settled_attempts(self.call) if attempt.error is not None
        )

    @property
    @override
    def error_summary(self) -> str:
        errors = self.errors_from_attempts
        return f"{len(errors)} attempts failed; last: {errors[-1]}"

    @property
    @override
    def error_text(self) -> str:
        """Return one transient error entry per request."""
        return "; ".join(
            f"attempt {index + 1}: {attempt.error}"
            for index, attempt in enumerate(_settled_attempts(self.call))
        )


class RetryUnavailableErrorRecord(_GenerationErrorRecordBase):
    """An open stream failed transiently after retry became unavailable.

    Validation rejects unknown fields.
    """

    kind: Literal["retry_unavailable_error"] = "retry_unavailable_error"

    @model_validator(mode="after")
    def _validate_retry_failures(self) -> Self:
        _ = _require_retry_failures(self.call)
        return self

    @property
    @override
    def error_summary(self) -> str:
        """Return the final transient error in the stable summary."""
        error = _require_retry_failures(self.call)[-1].error
        return f"no retry was available for a transient failure: {error}"


class RefusalErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response ended in a refusal.

    Validation rejects unknown fields.
    """

    kind: Literal["refusal_error"] = "refusal_error"

    @property
    @override
    def stop_reason(self) -> Literal["refusal"]:
        return "refusal"

    @property
    @override
    def error_summary(self) -> str:
        return "no structured output: the model refused or a provider filter blocked the turn"


class MaxCompletionTokensExceededErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response reached its token limit before parsing.

    Validation rejects unknown fields.
    """

    kind: Literal["max_completion_tokens_exceeded_error"] = "max_completion_tokens_exceeded_error"

    @property
    @override
    def stop_reason(self) -> Literal["max_tokens"]:
        return "max_tokens"

    @property
    @override
    def error_summary(self) -> str:
        return "the structured response reached max_completion_tokens before its JSON parsed"


class EmptyTurnErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response produced no output or tool call.

    Validation rejects unknown fields.
    """

    kind: Literal["empty_turn_error"] = "empty_turn_error"

    @property
    @override
    def stop_reason(self) -> Literal["end_turn"]:
        return "end_turn"

    @property
    @override
    def error_summary(self) -> str:
        return "the model completed its turn without producing output"


class SchemaViolationErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response failed the caller's model validation.

    Validation rejects unknown fields.
    """

    validation_error_json: str
    kind: Literal["schema_violation_error"] = "schema_violation_error"

    @property
    @override
    def stop_reason(self) -> Literal["end_turn"]:
        return "end_turn"

    @property
    @override
    def error_summary(self) -> str:
        return "the turn's text is not an instance of the bound response_format"


class ContextWindowExceededErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A response reported that the request exceeded the context window.

    Validation rejects unknown fields.
    """

    kind: Literal["context_window_exceeded_error"] = "context_window_exceeded_error"

    @property
    @override
    def stop_reason(self) -> Literal["context_window_exceeded"]:
        return "context_window_exceeded"

    @property
    @override
    def error_summary(self) -> str:
        return "the request exceeded the model's context window"


class UnfinishedTurnErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A provider returned a partial turn that langchaint cannot continue.

    Validation rejects unknown fields.
    """

    reason: str
    kind: Literal["unfinished_turn_error"] = "unfinished_turn_error"

    @property
    @override
    def error_summary(self) -> str:
        return self.reason


class ProviderFailedTerminallyErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A billable response reported a terminal generation failure.

    Validation rejects unknown fields.
    """

    reason: str
    kind: Literal["provider_failed_terminally_error"] = "provider_failed_terminally_error"

    @property
    @override
    def error_summary(self) -> str:
        return f"the provider reported that generating the response failed: {self.reason}"


class InvalidRequestErrorRecord(_GenerationErrorRecordBase):
    """An adapter or provider rejected one request.

    Validation rejects unknown fields.
    """

    reason: str
    kind: Literal["invalid_request_error"] = "invalid_request_error"

    @model_validator(mode="after")
    def _validate_terminal_provider_result(self) -> Self:
        _require_terminal_provider_result(self.call, permit_empty=True)
        return self

    @property
    @override
    def error_summary(self) -> str:
        return self.reason


class ProviderDeclaredFinalErrorRecord(_GenerationErrorRecordBase):
    """A provider marked one request error as terminal.

    Validation rejects unknown fields.
    """

    reason: str
    kind: Literal["provider_declared_final_error"] = "provider_declared_final_error"

    @model_validator(mode="after")
    def _validate_terminal_provider_result(self) -> Self:
        _require_terminal_provider_result(self.call, permit_empty=False)
        return self

    @property
    @override
    def error_summary(self) -> str:
        return f"a final error from the provider: {self.reason}"


class UnknownExceptionErrorRecord(_GenerationErrorRecordBase):
    """An exception `Adapter.classify` could not place.

    Validation rejects unknown fields.
    """

    reason: str
    kind: Literal["unknown_exception_error"] = "unknown_exception_error"

    @property
    @override
    def error_summary(self) -> str:
        return f"langchaint could not place this exception: {self.reason}"


class EscapedExceptionErrorRecord(_GenerationErrorRecordBase):
    """An exception escaped langchaint's generation handling.

    Validation rejects unknown fields.
    """

    reason: str
    kind: Literal["escaped_exception_error"] = "escaped_exception_error"

    @property
    @override
    def error_summary(self) -> str:
        return f"an exception escaped langchaint: {self.reason}"


class AbandonedCallErrorRecord(_GenerationErrorRecordBase):
    """A call ended before its result reached the caller.

    Validation rejects unknown fields.
    """

    kind: Literal["abandoned_call_error"] = "abandoned_call_error"

    @model_validator(mode="after")
    def _validate_abandoned_shape(self) -> Self:
        _require_abandoned_shape(self.call)
        return self

    @property
    @override
    def error_summary(self) -> str:
        return "the call was cut off before its result reached the caller"


class TimedOutErrorRecord(_GenerationErrorRecordBase):
    """A langchaint deadline expired before the call returned.

    Validation rejects unknown fields.
    """

    kind: Literal["timed_out_error"] = "timed_out_error"

    @model_validator(mode="after")
    def _validate_abandoned_shape(self) -> Self:
        _require_abandoned_shape(self.call)
        return self

    @property
    @override
    def error_summary(self) -> str:
        return "the call timed out before it produced a result"


type GenerationErrorRecord = Annotated[
    RetriesExhaustedErrorRecord
    | RetryUnavailableErrorRecord
    | RefusalErrorRecord
    | MaxCompletionTokensExceededErrorRecord
    | EmptyTurnErrorRecord
    | SchemaViolationErrorRecord
    | ContextWindowExceededErrorRecord
    | UnfinishedTurnErrorRecord
    | ProviderFailedTerminallyErrorRecord
    | InvalidRequestErrorRecord
    | ProviderDeclaredFinalErrorRecord
    | UnknownExceptionErrorRecord
    | EscapedExceptionErrorRecord
    | AbandonedCallErrorRecord
    | TimedOutErrorRecord,
    Field(discriminator="kind"),
]

_GENERATION_ERROR_RECORD_CLASSES = (
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    RefusalErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    EmptyTurnErrorRecord,
    SchemaViolationErrorRecord,
    ContextWindowExceededErrorRecord,
    UnfinishedTurnErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    InvalidRequestErrorRecord,
    ProviderDeclaredFinalErrorRecord,
    UnknownExceptionErrorRecord,
    EscapedExceptionErrorRecord,
    AbandonedCallErrorRecord,
    TimedOutErrorRecord,
)


def _terminal_error_record(
    classification: "ErrorClassification", *, reason: str, call: CallRecord
) -> GenerationErrorRecord:
    if classification == "invalid_request":
        return InvalidRequestErrorRecord(
            reason=f"the provider rejected the request: {reason}", call=call
        )
    if classification == "declared_final":
        return ProviderDeclaredFinalErrorRecord(reason=reason, call=call)
    return UnknownExceptionErrorRecord(reason=reason, call=call)


class GenerationError(Exception):
    """A live terminal generation failure with one normalized record."""

    record: GenerationErrorRecord
    request: "RequestParams | None"
    provider_attempts: tuple[AttemptProviderData, ...]

    def __init__(
        self,
        *,
        record: GenerationErrorRecord,
        request: "RequestParams | None",
        provider_attempts: tuple[AttemptProviderData, ...],
    ) -> None:
        """Store normalized and live-only failure data."""
        if type(record) not in _GENERATION_ERROR_RECORD_CLASSES:
            raise TypeError(f"unsupported generation error record: {type(record).__name__}")
        if len(provider_attempts) != len(record.call.attempt_records):
            raise ValueError("provider_attempts must align with call.attempt_records")
        super().__init__()
        self.record = record
        self.request = request
        self.provider_attempts = provider_attempts

    @property
    def call(self) -> CallRecord:
        """Return the normalized call record."""
        return self.record.call

    @property
    def attempt_records(
        self,
    ) -> tuple[SettledAttemptRecord | CutOffAttemptRecord, ...]:
        """Return normalized request records in order."""
        return self.record.attempt_records

    @property
    def attempts(self) -> int:
        """Return requests that langchaint observed going out."""
        return self.record.attempts

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every request."""
        return self.record.usage

    @property
    def model(self) -> str:
        """Return the requested model id."""
        return self.record.model

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.record.provider_name

    @property
    def elapsed_seconds(self) -> float:
        """Return the complete call duration."""
        return self.record.elapsed_seconds

    @property
    def assistant_message(self) -> AssistantMessage | None:
        """Return the last recorded assistant message."""
        return self.record.assistant_message

    @property
    def stop_reason(self) -> StopReason | None:
        """Return the normalized stop reason when present."""
        return self.record.stop_reason

    @property
    def error_text(self) -> str:
        """Return the tracing-safe complete failure text."""
        return self.record.error_text

    @override
    def __str__(self) -> str:
        """Return `record.error_summary`."""
        return self.record.error_summary


class InvalidToolArgsError(Exception):
    """A tool call's `args_json` failed validation against the tool's `args_model`.

    `PydanticTool._validated_args` raises this error.
    A tool function does not cause langchaint to raise this error.
    `ToolManager.dispatch` catches this error and returns `DispatchInvalidToolArgs`.
    """

    validation_error: ValidationError

    def __init__(self, validation_error: ValidationError) -> None:
        """Hold `validation_error` by reference."""
        super().__init__()
        self.validation_error = validation_error

    @override
    def __str__(self) -> str:
        """Render the held validation error."""
        return str(self.validation_error)


class DispatchExceptionGroup(ExceptionGroup[Exception]):
    """Tool function exceptions from `ToolManager.dispatch_many`.

    `completed_outcomes` preserves settled outcomes in input order.
    The grouped exceptions preserve input order and their tracebacks.
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
        """Build a group carrying the completed dispatch outcomes."""
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
        """Store grouped exceptions and completed dispatch outcomes."""
        super().__init__(message, exceptions)
        self.completed_outcomes = completed_outcomes

    @override
    # pyrefly: ignore[bad-override]  # Typeshed makes `derive` generic for each call.
    def derive(self, excs: Sequence[Exception], /) -> "DispatchExceptionGroup":
        """Return a subgroup with the same `completed_outcomes`."""
        return DispatchExceptionGroup(
            self.message, excs, completed_outcomes=self.completed_outcomes
        )


class StreamProtocolError(Exception):
    """A stream did not follow the event contract.

    A stream that ends without a terminal result raises this error.
    A missing Messages API stop reason or Responses API terminal response raises this error.
    A `StreamHandle` that ends without an adapter stream raises this error.
    `AdapterStream.final()` may raise this error before `AdapterStream.items()` is exhausted.
    """


class GaveUpWaiting(Exception):  # noqa: N818
    """A budget expired before `SharedBackoff.admitted()` admitted the request.

    The admission holds no permit or queue position and records no request.
    A new attempt joins the same queue behind the same pause.
    """


class ParserContractError(Exception):
    """A `SharedBackoff` parse function raised.

    This error identifies a defect in `parse` instead of a provider classification.
    `__cause__` holds the exception from `parse`.
    The provider failure passed to `parse` remains as exception context.
    `SharedBackoff` records no request outcome for this error.
    """
