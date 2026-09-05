"""Normalized generation error records and live generation failures."""

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, override

from pydantic import Field, model_validator

from langchaint.billing.usage import Usage
from langchaint.common.messages import AssistantMessage, StopReason
from langchaint.generation.call import (
    AttemptProviderData,
    CallRecord,
    CutOffAttemptRecord,
    SettledAttemptRecord,
    TransientErrorRecord,
    _CallResultRecordBase,
    _require_completed_model_turn,
    _settled_attempts,
)

if TYPE_CHECKING:
    from langchaint.adapter import ErrorClassification, RequestParams


class _GenerationErrorRecordBase(_CallResultRecordBase):
    """Shared normalized data and properties for terminal generation errors.

    Validation rejects unknown fields.
    """

    @property
    def attempt_records(
        self,
    ) -> tuple[SettledAttemptRecord | CutOffAttemptRecord, ...]:
        """Return the call's normalized attempt records."""
        return self.call.attempt_records

    @property
    def assistant_message(self) -> AssistantMessage | None:
        """Return the last recorded assistant message."""
        for attempt in reversed(self.call.attempt_records):
            if isinstance(attempt, SettledAttemptRecord) and attempt.assistant_message is not None:
                return attempt.assistant_message
        return None

    stop_reason: ClassVar[StopReason | None] = None

    error_text: str

    @override
    def __str__(self) -> str:
        """Return `error_text`."""
        return self.error_text


class _CompletedModelTurnErrorRecordBase(_GenerationErrorRecordBase):
    """Shared validation for errors that require a completed model turn.

    Validation rejects unknown fields.
    """

    @model_validator(mode="after")
    def _validate_completed_model_turn(self) -> Self:
        _require_completed_model_turn(self.call)
        return self


def _require_retry_failures(call: CallRecord) -> tuple[TransientErrorRecord, ...]:
    attempts = _settled_attempts(call)
    if not attempts:
        raise ValueError("call must contain at least one settled attempt")
    errors: list[TransientErrorRecord] = []
    for attempt in attempts:
        if attempt.error is None:
            raise ValueError("every attempt must contain a transient error")
        errors.append(attempt.error)
    return tuple(errors)


def _retries_exhausted_error_text(call: CallRecord) -> str:
    entries: list[str] = []
    for attempt_number, error in enumerate(_require_retry_failures(call), start=1):
        indented_message = error.message.replace("\n", "\n  ")
        entries.append(f"attempt {attempt_number}: {indented_message}")
    return "\n".join(entries)


def _retry_unavailable_error_text(call: CallRecord) -> str:
    return _require_retry_failures(call)[-1].message


def _set_or_validate_retry_error_text(
    record: _GenerationErrorRecordBase, expected_error_text: str
) -> None:
    if "error_text" not in record.model_fields_set:
        object.__setattr__(record, "error_text", expected_error_text)
    elif record.error_text != expected_error_text:
        raise ValueError("error_text must match the call's transient errors")


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

    error_text: str = ""
    kind: Literal["retries_exhausted_error"] = "retries_exhausted_error"

    @model_validator(mode="after")
    def _validate_retry_failures(self) -> Self:
        expected_error_text = _retries_exhausted_error_text(self.call)
        _set_or_validate_retry_error_text(self, expected_error_text)
        return self

    @property
    def errors_from_attempts(self) -> tuple[TransientErrorRecord, ...]:
        """Return each attempt's normalized transient error."""
        return _require_retry_failures(self.call)


class RetryUnavailableErrorRecord(_GenerationErrorRecordBase):
    """An open stream failed transiently after retry became unavailable.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["retry_unavailable_error"] = "retry_unavailable_error"

    @model_validator(mode="after")
    def _validate_retry_failures(self) -> Self:
        expected_error_text = _retry_unavailable_error_text(self.call)
        _set_or_validate_retry_error_text(self, expected_error_text)
        return self


class RefusalErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response ended in a refusal.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["refusal_error"] = "refusal_error"

    stop_reason: ClassVar[Literal["refusal"]] = "refusal"


class MaxCompletionTokensExceededErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response reached its token limit before parsing.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["max_completion_tokens_exceeded_error"] = "max_completion_tokens_exceeded_error"

    stop_reason: ClassVar[Literal["max_tokens"]] = "max_tokens"


class EmptyTurnErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response produced no output or tool call.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["empty_turn_error"] = "empty_turn_error"

    stop_reason: ClassVar[Literal["end_turn"]] = "end_turn"


class SchemaViolationErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A structured response failed the caller's model validation.

    Validation rejects unknown fields.
    """

    validation_error_json: str
    error_text: str = ""
    kind: Literal["schema_violation_error"] = "schema_violation_error"

    stop_reason: ClassVar[Literal["end_turn"]] = "end_turn"


class ContextWindowExceededErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A response reported that the request exceeded the context window.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["context_window_exceeded_error"] = "context_window_exceeded_error"

    stop_reason: ClassVar[Literal["context_window_exceeded"]] = "context_window_exceeded"


class UnfinishedTurnErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A provider returned a partial turn that langchaint cannot continue.

    Validation rejects unknown fields.
    """

    kind: Literal["unfinished_turn_error"] = "unfinished_turn_error"


class ProviderFailedTerminallyErrorRecord(_CompletedModelTurnErrorRecordBase):
    """A billable response reported a terminal generation failure.

    Validation rejects unknown fields.
    """

    kind: Literal["provider_failed_terminally_error"] = "provider_failed_terminally_error"


class InvalidRequestErrorRecord(_GenerationErrorRecordBase):
    """An adapter or provider rejected one request.

    Validation rejects unknown fields.
    """

    kind: Literal["invalid_request_error"] = "invalid_request_error"

    @model_validator(mode="after")
    def _validate_terminal_provider_result(self) -> Self:
        _require_terminal_provider_result(self.call, permit_empty=True)
        return self


class ProviderDeclaredFinalErrorRecord(_GenerationErrorRecordBase):
    """A provider marked one request error as terminal.

    Validation rejects unknown fields.
    """

    kind: Literal["provider_declared_final_error"] = "provider_declared_final_error"

    @model_validator(mode="after")
    def _validate_terminal_provider_result(self) -> Self:
        _require_terminal_provider_result(self.call, permit_empty=False)
        return self


class UnknownExceptionErrorRecord(_GenerationErrorRecordBase):
    """An exception `Adapter.classify` could not place.

    Validation rejects unknown fields.
    """

    kind: Literal["unknown_exception_error"] = "unknown_exception_error"


class EscapedExceptionErrorRecord(_GenerationErrorRecordBase):
    """An exception escaped langchaint's generation handling.

    Validation rejects unknown fields.
    """

    kind: Literal["escaped_exception_error"] = "escaped_exception_error"


class AbandonedCallErrorRecord(_GenerationErrorRecordBase):
    """A call ended before its result reached the caller.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["abandoned_call_error"] = "abandoned_call_error"

    @model_validator(mode="after")
    def _validate_abandoned_shape(self) -> Self:
        _require_abandoned_shape(self.call)
        return self


class TimedOutErrorRecord(_GenerationErrorRecordBase):
    """A langchaint deadline expired before the call returned.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["timed_out_error"] = "timed_out_error"

    @model_validator(mode="after")
    def _validate_abandoned_shape(self) -> Self:
        _require_abandoned_shape(self.call)
        return self


type GenerationErrorKind = Literal[
    "retries_exhausted_error",
    "retry_unavailable_error",
    "refusal_error",
    "max_completion_tokens_exceeded_error",
    "empty_turn_error",
    "schema_violation_error",
    "context_window_exceeded_error",
    "unfinished_turn_error",
    "provider_failed_terminally_error",
    "invalid_request_error",
    "provider_declared_final_error",
    "unknown_exception_error",
    "escaped_exception_error",
    "abandoned_call_error",
    "timed_out_error",
]


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
        return InvalidRequestErrorRecord(error_text=reason, call=call)
    if classification == "declared_final":
        return ProviderDeclaredFinalErrorRecord(error_text=reason, call=call)
    return UnknownExceptionErrorRecord(error_text=reason, call=call)


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
    def kind(self) -> GenerationErrorKind:
        """Return the normalized failure category."""
        return self.record.kind

    @property
    def error_text(self) -> str:
        """Return the complete failure text."""
        return self.record.error_text

    @override
    def __str__(self) -> str:
        """Return `error_text`."""
        return self.error_text
