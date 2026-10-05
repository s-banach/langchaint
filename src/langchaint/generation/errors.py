"""Normalized generation error records and live generation failures."""

from typing import TYPE_CHECKING, Literal, Self, override

from pydantic import model_validator

from langchaint.billing.usage import Usage
from langchaint.common.exceptions import TransientError
from langchaint.common.messages import AssistantMessage, StopReason
from langchaint.common.request_failure import RequestFailure, _TerminalRequestFailureKind
from langchaint.generation.request_history import (
    CutOffRequestRecord,
    RequestHistory,
    RequestProviderData,
    SettledRequestRecord,
    TransientErrorRecord,
    _InputOutcomeRecordBase,
    _RequestLedger,
    _require_abandoned_shape,
    _require_finished_assistant_message,
    _settled_request_records,
)

if TYPE_CHECKING:
    from langchaint.adapter import RequestParams
    from langchaint.billing.pricing import ProviderBilling


class _GenerationErrorRecordBase(_InputOutcomeRecordBase):
    """Shared normalized data and properties for terminal generation errors.

    Validation rejects unknown fields.
    """

    error_text: str = ""

    @override
    def __str__(self) -> str:
        """Return `error_text`."""
        return self.error_text


def _require_retry_failures(request_history: RequestHistory) -> tuple[TransientErrorRecord, ...]:
    settled_request_records = _settled_request_records(request_history)
    if not settled_request_records:
        raise ValueError("request history must contain at least one settled request")
    if any(request_record.error is None for request_record in settled_request_records):
        raise ValueError("every request must contain a transient error")
    return request_history.errors_from_requests


def _retries_exhausted_error_text(request_history: RequestHistory) -> str:
    return "\n".join(
        f"request {request_number}: {error.error_text.replace('\n', '\n  ')}"
        for request_number, error in enumerate(_require_retry_failures(request_history), start=1)
    )


def _retry_unavailable_error_text(request_history: RequestHistory) -> str:
    return _require_retry_failures(request_history)[-1].error_text


def _set_or_validate_retry_error_text(
    record: _GenerationErrorRecordBase, expected_error_text: str
) -> None:
    if "error_text" not in record.model_fields_set:
        object.__setattr__(record, "error_text", expected_error_text)
    elif record.error_text != expected_error_text:
        raise ValueError("error_text must match the request history's transient errors")


def _require_error_text(record: "GenerationErrorRecord") -> None:
    if "error_text" not in record.model_fields_set:
        raise ValueError(f"{record.kind} requires error_text")


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


def _require_provider_failed_terminally_shape(request_history: RequestHistory) -> None:
    settled_request_records = _settled_request_records(request_history)
    if settled_request_records and settled_request_records[-1].assistant_message is not None:
        _require_finished_assistant_message(request_history)
    else:
        _require_final_terminal_response(request_history, permit_empty=False)


type _GenerationErrorRecordKind = Literal[
    "retries_exhausted_error",
    "retry_unavailable_error",
    "refusal_error",
    "max_completion_tokens_exceeded_error",
    "empty_assistant_message_error",
    "context_window_exceeded_error",
    "unfinished_assistant_message_error",
    "provider_failed_terminally_error",
    "auth_error",
    "rejected_error",
    "unknown_exception_error",
    "timed_out_error",
]
"""The kinds a `GenerationErrorRecord` reports, which `GenerationErrorKind` documents."""


type GenerationErrorKind = _GenerationErrorRecordKind | Literal["schema_violation_error"]
"""The outcome a `GenerationError` reports.

- `retries_exhausted_error`: every request failed transiently and the retry budget ended.
- `retry_unavailable_error`: an open stream failed transiently after retry became unavailable.
- `refusal_error`: a refusal ended a response that produced no output.
- `max_completion_tokens_exceeded_error`: a structured response reached its token limit before parsing.
- `empty_assistant_message_error`: a structured response produced no output or tool call.
- `schema_violation_error`: a structured response failed the caller's model validation.
  `SchemaViolationErrorRecord` reports it, and every other kind is a `GenerationErrorRecord`.
- `context_window_exceeded_error`: a response reported that the request exceeded the context window.
- `unfinished_assistant_message_error`: a provider returned an unfinished assistant message.
- `provider_failed_terminally_error`: a provider reported a failure that sending the request again would repeat.
  A response body or an error response can report it.
- `auth_error`: a provider rejected the client's credentials or permissions for one request.
  The same request can succeed after the caller repairs the credentials or permissions.
- `rejected_error`: the adapter returned `RejectedMessages`, or the provider rejected a request.
  After `RejectedMessages`, `request_count == 0`.
- `unknown_exception_error`: an exception escaped generation handling, or `Adapter.request_failure` could not place it.
- `timed_out_error`: a langchaint deadline expired before handling the input ended.
"""


class GenerationErrorRecord(_GenerationErrorRecordBase):
    """One terminal generation failure other than a schema violation.

    Validation requires of `request_history`, by `kind`:
    - `retries_exhausted_error` and `retry_unavailable_error`: every request has a transient error.
      `error_text` defaults to text built from those errors and must equal it.
    - `timed_out_error`: every request before the final one has an error.
      The final request has an error, is cut off, or is error-free without an assistant message.
      The history may hold no requests.
    - `unknown_exception_error`: nothing.
    - Every other kind: every request before the final one has an error, and the final request is error-free.
      - `auth_error` and `rejected_error`: the final request holds no assistant message.
        `rejected_error` also accepts a history without requests.
      - `provider_failed_terminally_error`: the final request may hold an assistant message, and then is billed.
        It holds one when the adapter read the failure from a finished response, and none when the SDK raised it.
      - Every remaining kind: the final request is billed and holds an assistant message.

    `unknown_exception_error`, `auth_error`, and `rejected_error` require `error_text`.
    So do `unfinished_assistant_message_error` and `provider_failed_terminally_error`.
    Validation rejects unknown fields.
    """

    kind: _GenerationErrorRecordKind

    @property
    def stop_reason(self) -> StopReason | None:
        """Return the stop reason that `kind` implies, or `None` when `kind` implies none."""
        match self.kind:
            case "refusal_error":
                return "refusal"
            case "max_completion_tokens_exceeded_error":
                return "max_completion_tokens"
            case "empty_assistant_message_error":
                return "stop"
            case "context_window_exceeded_error":
                return "context_window_exceeded"
            case _:
                return None

    @model_validator(mode="after")
    def _validate_request_history(self) -> Self:
        match self.kind:
            case "retries_exhausted_error":
                _set_or_validate_retry_error_text(
                    self, _retries_exhausted_error_text(self.request_history)
                )
            case "retry_unavailable_error":
                _set_or_validate_retry_error_text(
                    self, _retry_unavailable_error_text(self.request_history)
                )
            case (
                "refusal_error"
                | "max_completion_tokens_exceeded_error"
                | "empty_assistant_message_error"
                | "context_window_exceeded_error"
            ):
                _require_finished_assistant_message(self.request_history)
            case "unfinished_assistant_message_error":
                _require_error_text(self)
                _require_finished_assistant_message(self.request_history)
            case "provider_failed_terminally_error":
                _require_error_text(self)
                _require_provider_failed_terminally_shape(self.request_history)
            case "auth_error":
                _require_error_text(self)
                _require_final_terminal_response(self.request_history, permit_empty=False)
            case "rejected_error":
                _require_error_text(self)
                _require_final_terminal_response(self.request_history, permit_empty=True)
            case "unknown_exception_error":
                _require_error_text(self)
            case "timed_out_error":
                _require_abandoned_shape(self.request_history)
        return self


class SchemaViolationErrorRecord(_GenerationErrorRecordBase):
    """A structured response failed the caller's model validation.

    `validation_error_json` is pydantic's error JSON for the rejected text.
    The final request is error-free, billed, and holds an assistant message.
    Validation rejects unknown fields.
    """

    validation_error_json: str
    kind: Literal["schema_violation_error"] = "schema_violation_error"

    @property
    def stop_reason(self) -> Literal["stop"]:
        """Return `stop`, the stop reason of a finished structured response."""
        return "stop"

    @model_validator(mode="after")
    def _validate_finished_assistant_message(self) -> Self:
        _require_finished_assistant_message(self.request_history)
        return self


_GENERATION_ERROR_RECORD_CLASSES = (GenerationErrorRecord, SchemaViolationErrorRecord)
"""The record classes a `GenerationError` holds."""


_PROVIDER_ANSWERED_KINDS = ("auth", "rejected", "provider_failed_terminally")
"""The `RequestFailure` kinds that show the provider answered the terminal request.

`_terminal_generation_error` records a settled request for that answer.
`_require_final_terminal_response` validates that request on the records of these kinds.
"""


def _terminal_error_record(
    request_failure_kind: _TerminalRequestFailureKind,
    *,
    error_text: str,
    request_history: RequestHistory,
) -> GenerationErrorRecord:
    match request_failure_kind:
        case "auth":
            kind = "auth_error"
        case "rejected":
            kind = "rejected_error"
        case "provider_failed_terminally":
            kind = "provider_failed_terminally_error"
        case "unknown_exception":
            kind = "unknown_exception_error"
    return GenerationErrorRecord(kind=kind, error_text=error_text, request_history=request_history)


class GenerationError(Exception):
    """A live terminal generation failure with one normalized record."""

    record: GenerationErrorRecord | SchemaViolationErrorRecord
    request_params: "RequestParams | None"
    request_provider_data: tuple[RequestProviderData, ...]

    def __init__(
        self,
        *,
        record: GenerationErrorRecord | SchemaViolationErrorRecord,
        request_params: "RequestParams | None",
        request_provider_data: tuple[RequestProviderData, ...],
    ) -> None:
        """Store normalized and live-only failure data."""
        if type(record) not in _GENERATION_ERROR_RECORD_CLASSES:
            raise TypeError(f"unsupported generation error record: {type(record).__name__}")
        if len(request_provider_data) != len(record.request_history.records):
            raise ValueError("request_provider_data must align with request_history.records")
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
    request_failure_kind: _TerminalRequestFailureKind,
    *,
    error_text: str,
    ledger: _RequestLedger,
    provider_billing: "ProviderBilling | None",
    request_params: "RequestParams | None",
    stream_opened: bool,
) -> GenerationError:
    """Settle the failed request when the provider answered or the stream opened, then build the input's error.

    `provider_billing` is the provider-reported billing of the failed request.
    `stream_opened` is true when the provider stream opened before the failure.
    """
    if request_failure_kind in _PROVIDER_ANSWERED_KINDS or stream_opened:
        ledger.record(error=None, assistant_message=None, provider_billing=provider_billing)
    return GenerationError(
        record=_terminal_error_record(
            request_failure_kind, error_text=error_text, request_history=ledger.freeze()
        ),
        request_params=request_params,
        request_provider_data=ledger.request_provider_data,
    )


def _transient_error_for(
    error: Exception, message: str, request_failure: RequestFailure
) -> TransientError:
    """Wrap one transiently failed request's exception as the `TransientError` its request record carries.

    Return an existing `TransientError` unchanged, which keeps its message, `retry_after_seconds`, and `pauses_quota`.
    A wrap takes `retry_after_seconds` and `pauses_quota` from `request_failure`.
    """
    if isinstance(error, TransientError):
        return error
    wrapped = TransientError(
        message,
        retry_after_seconds=request_failure.retry_after_seconds,
        pauses_quota=request_failure.pauses_quota,
    )
    wrapped.__cause__ = error
    return wrapped
