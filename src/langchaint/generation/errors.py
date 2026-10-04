"""Normalized generation error records and live generation failures."""

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, override

from pydantic import Field, model_validator

from langchaint.billing.usage import Usage
from langchaint.common.messages import AssistantMessage, StopReason
from langchaint.generation.request_history import (
    CutOffRequestRecord,
    RequestHistory,
    RequestProviderData,
    SettledRequestRecord,
    TransientErrorRecord,
    _InputOutcomeRecordBase,
    _RequestLedger,
    _require_abandoned_shape,
    _require_completed_assistant_message,
    _settled_request_records,
)

if TYPE_CHECKING:
    from langchaint.adapter import ErrorClassification, RequestParams
    from langchaint.billing.pricing import ProviderBilling
    from langchaint.failure_step import _Terminal


class _GenerationErrorRecordBase(_InputOutcomeRecordBase):
    """Shared normalized data and properties for terminal generation errors.

    Validation rejects unknown fields.
    """

    stop_reason: ClassVar[StopReason | None] = None

    error_text: str

    @override
    def __str__(self) -> str:
        """Return `error_text`."""
        return self.error_text


class _CompletedAssistantMessageErrorRecordBase(_GenerationErrorRecordBase):
    """Shared validation for errors whose final request is error-free, billed, and holds an assistant message.

    Validation rejects unknown fields.
    """

    @model_validator(mode="after")
    def _validate_completed_assistant_message(self) -> Self:
        _require_completed_assistant_message(self.request_history)
        return self


def _require_retry_failures(request_history: RequestHistory) -> tuple[TransientErrorRecord, ...]:
    settled_request_records = _settled_request_records(request_history)
    if not settled_request_records:
        raise ValueError("request history must contain at least one settled request")
    errors: list[TransientErrorRecord] = []
    for request_record in settled_request_records:
        if request_record.error is None:
            raise ValueError("every request must contain a transient error")
        errors.append(request_record.error)
    return tuple(errors)


def _retries_exhausted_error_text(request_history: RequestHistory) -> str:
    return "\n".join(
        f"request {request_number}: {error.message.replace('\n', '\n  ')}"
        for request_number, error in enumerate(_require_retry_failures(request_history), start=1)
    )


def _retry_unavailable_error_text(request_history: RequestHistory) -> str:
    return _require_retry_failures(request_history)[-1].message


def _set_or_validate_retry_error_text(
    record: _GenerationErrorRecordBase, expected_error_text: str
) -> None:
    if "error_text" not in record.model_fields_set:
        object.__setattr__(record, "error_text", expected_error_text)
    elif record.error_text != expected_error_text:
        raise ValueError("error_text must match the request history's transient errors")


def _require_final_terminal_response(
    request_history: RequestHistory, *, permit_empty: bool
) -> None:
    settled_request_records = _settled_request_records(request_history)
    if not settled_request_records:
        if permit_empty:
            return
        raise ValueError("request history must contain a final response")
    if any(request_record.error is None for request_record in settled_request_records[:-1]):
        raise ValueError("every request before the final request must contain an error")
    final = settled_request_records[-1]
    if final.error is not None:
        raise ValueError("the final request must be error-free")
    if final.assistant_message is not None:
        raise ValueError("the final response must not contain an assistant message")


class RetriesExhaustedErrorRecord(_GenerationErrorRecordBase):
    """Every request failed transiently and the retry budget ended.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["retries_exhausted_error"] = "retries_exhausted_error"

    @model_validator(mode="after")
    def _validate_retry_failures(self) -> Self:
        expected_error_text = _retries_exhausted_error_text(self.request_history)
        _set_or_validate_retry_error_text(self, expected_error_text)
        return self

    @property
    def errors_from_requests(self) -> tuple[TransientErrorRecord, ...]:
        """Return each request's normalized transient error."""
        return _require_retry_failures(self.request_history)


class RetryUnavailableErrorRecord(_GenerationErrorRecordBase):
    """An open stream failed transiently after retry became unavailable.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["retry_unavailable_error"] = "retry_unavailable_error"

    @model_validator(mode="after")
    def _validate_retry_failures(self) -> Self:
        expected_error_text = _retry_unavailable_error_text(self.request_history)
        _set_or_validate_retry_error_text(self, expected_error_text)
        return self


class RefusalErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A refusal ended a response that produced no output.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["refusal_error"] = "refusal_error"

    stop_reason: ClassVar[Literal["refusal"]] = "refusal"


class MaxCompletionTokensExceededErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A structured response reached its token limit before parsing.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["max_completion_tokens_exceeded_error"] = "max_completion_tokens_exceeded_error"

    stop_reason: ClassVar[Literal["max_tokens"]] = "max_tokens"


class EmptyAssistantMessageErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A structured response produced no output or tool call.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["empty_assistant_message_error"] = "empty_assistant_message_error"

    stop_reason: ClassVar[Literal["end_turn"]] = "end_turn"


class SchemaViolationErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A structured response failed the caller's model validation.

    Validation rejects unknown fields.
    """

    validation_error_json: str
    error_text: str = ""
    kind: Literal["schema_violation_error"] = "schema_violation_error"

    stop_reason: ClassVar[Literal["end_turn"]] = "end_turn"


class ContextWindowExceededErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A response reported that the request exceeded the context window.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["context_window_exceeded_error"] = "context_window_exceeded_error"

    stop_reason: ClassVar[Literal["context_window_exceeded"]] = "context_window_exceeded"


class UnfinishedAssistantMessageErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A provider returned an unfinished assistant message.

    Validation rejects unknown fields.
    """

    kind: Literal["unfinished_assistant_message_error"] = "unfinished_assistant_message_error"


class ProviderFailedTerminallyErrorRecord(_CompletedAssistantMessageErrorRecordBase):
    """A billable response reported a terminal generation failure.

    Validation rejects unknown fields.
    """

    kind: Literal["provider_failed_terminally_error"] = "provider_failed_terminally_error"


class AuthErrorRecord(_GenerationErrorRecordBase):
    """A provider rejected the client's credentials or permissions for one request.

    The same request can succeed after the caller repairs the credentials or permissions.
    Validation rejects unknown fields.
    """

    kind: Literal["auth_error"] = "auth_error"

    @model_validator(mode="after")
    def _validate_final_terminal_response(self) -> Self:
        _require_final_terminal_response(self.request_history, permit_empty=False)
        return self


class RejectedErrorRecord(_GenerationErrorRecordBase):
    """The adapter returned `RefusedMessages`, so `request_count == 0`, or the provider rejected a request.

    Validation rejects unknown fields.
    """

    kind: Literal["rejected_error"] = "rejected_error"

    @model_validator(mode="after")
    def _validate_final_terminal_response(self) -> Self:
        _require_final_terminal_response(self.request_history, permit_empty=True)
        return self


class ProviderDeclaredFinalErrorRecord(_GenerationErrorRecordBase):
    """A provider marked one request error as terminal.

    Validation rejects unknown fields.
    """

    kind: Literal["provider_declared_final_error"] = "provider_declared_final_error"

    @model_validator(mode="after")
    def _validate_final_terminal_response(self) -> Self:
        _require_final_terminal_response(self.request_history, permit_empty=False)
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


class TimedOutErrorRecord(_GenerationErrorRecordBase):
    """A langchaint deadline expired before handling the input ended.

    Validation rejects unknown fields.
    """

    error_text: str = ""
    kind: Literal["timed_out_error"] = "timed_out_error"

    @model_validator(mode="after")
    def _validate_abandoned_shape(self) -> Self:
        _require_abandoned_shape(self.request_history)
        return self


type GenerationErrorKind = Literal[
    "retries_exhausted_error",
    "retry_unavailable_error",
    "refusal_error",
    "max_completion_tokens_exceeded_error",
    "empty_assistant_message_error",
    "schema_violation_error",
    "context_window_exceeded_error",
    "unfinished_assistant_message_error",
    "provider_failed_terminally_error",
    "auth_error",
    "rejected_error",
    "provider_declared_final_error",
    "unknown_exception_error",
    "escaped_exception_error",
    "timed_out_error",
]


type GenerationErrorRecord = Annotated[
    RetriesExhaustedErrorRecord
    | RetryUnavailableErrorRecord
    | RefusalErrorRecord
    | MaxCompletionTokensExceededErrorRecord
    | EmptyAssistantMessageErrorRecord
    | SchemaViolationErrorRecord
    | ContextWindowExceededErrorRecord
    | UnfinishedAssistantMessageErrorRecord
    | ProviderFailedTerminallyErrorRecord
    | AuthErrorRecord
    | RejectedErrorRecord
    | ProviderDeclaredFinalErrorRecord
    | UnknownExceptionErrorRecord
    | EscapedExceptionErrorRecord
    | TimedOutErrorRecord,
    Field(discriminator="kind"),
]

_GENERATION_ERROR_RECORD_CLASSES = (
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    RefusalErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    EmptyAssistantMessageErrorRecord,
    SchemaViolationErrorRecord,
    ContextWindowExceededErrorRecord,
    UnfinishedAssistantMessageErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    AuthErrorRecord,
    RejectedErrorRecord,
    ProviderDeclaredFinalErrorRecord,
    UnknownExceptionErrorRecord,
    EscapedExceptionErrorRecord,
    TimedOutErrorRecord,
)


_PROVIDER_ANSWERED_CLASSIFICATIONS = ("auth", "invalid_request", "declared_final")
"""The classifications that show the provider answered the terminal request.

`_terminal_generation_error` records a settled request for that answer.
`_require_final_terminal_response` validates that request on the records for these classifications.
"""


def _terminal_error_record(
    classification: "ErrorClassification", *, reason: str, request_history: RequestHistory
) -> GenerationErrorRecord:
    if classification == "auth":
        return AuthErrorRecord(error_text=reason, request_history=request_history)
    if classification == "invalid_request":
        return RejectedErrorRecord(error_text=reason, request_history=request_history)
    if classification == "declared_final":
        return ProviderDeclaredFinalErrorRecord(error_text=reason, request_history=request_history)
    return UnknownExceptionErrorRecord(error_text=reason, request_history=request_history)


class GenerationError(Exception):
    """A live terminal generation failure with one normalized record."""

    record: GenerationErrorRecord
    request_params: "RequestParams | None"
    request_provider_data: tuple[RequestProviderData, ...]

    def __init__(
        self,
        *,
        record: GenerationErrorRecord,
        request_params: "RequestParams | None",
        request_provider_data: tuple[RequestProviderData, ...],
    ) -> None:
        """Store normalized and live-only failure data."""
        if type(record) not in _GENERATION_ERROR_RECORD_CLASSES:
            raise TypeError(f"unsupported generation error record: {type(record).__name__}")
        if len(request_provider_data) != len(record.request_history.request_records):
            raise ValueError(
                "request_provider_data must align with request_history.request_records"
            )
        super().__init__()
        self.record = record
        self.request_params = request_params
        self.request_provider_data = request_provider_data

    @property
    def request_history(self) -> RequestHistory:
        """Return the normalized request history."""
        return self.record.request_history

    @property
    def request_records(
        self,
    ) -> tuple[SettledRequestRecord | CutOffRequestRecord, ...]:
        """Return normalized request records in order."""
        return self.record.request_records

    @property
    def request_count(self) -> int:
        """Return requests that langchaint observed going out."""
        return self.record.request_count

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
        """Return the seconds from the start of handling the input until its request history was frozen.

        It includes admission and backoff waits.
        """
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


def _terminal_generation_error(
    step: "_Terminal",
    *,
    reason: str,
    ledger: _RequestLedger,
    billing: "ProviderBilling | None",
    request_params: "RequestParams | None",
    stream_opened: bool,
) -> GenerationError:
    """Settle the failed request when the provider answered or the stream opened, then build the input's error.

    `billing` is the provider-reported billing of the failed request.
    `stream_opened` is true when the provider stream opened before the failure.
    """
    if step.classification in _PROVIDER_ANSWERED_CLASSIFICATIONS or stream_opened:
        ledger.record(error=None, assistant_message=None, billing=billing)
    return GenerationError(
        record=_terminal_error_record(
            step.classification, reason=reason, request_history=ledger.freeze()
        ),
        request_params=request_params,
        request_provider_data=ledger.request_provider_data,
    )
