"""Live generations, their records, and the outcome unions."""

from dataclasses import dataclass
from typing import Annotated, Generic, Literal, Self, TypeVar, override

from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from langchaint.adapter import RequestParams, ResponseOutcome
from langchaint.billing.pricing import ProviderBilling
from langchaint.billing.usage import Usage
from langchaint.common.messages import AssistantMessage, StopReason, ToolCall
from langchaint.generation.errors import (
    _GENERATION_ERROR_RECORD_CLASSES,
    GenerationError,
    GenerationErrorRecord,
    SchemaViolationErrorRecord,
)
from langchaint.generation.request_history import (
    AbandonedStreamRecord,
    RequestHistory,
    RequestProviderData,
    SettledRequestRecord,
    _InputOutcomeRecordBase,
    _RequestLedger,
    _require_finished_assistant_message,
    _settled_request_records,
)

# Declared covariant because pyrefly infers a PEP 695 parameter invariant through the record's
# `output` field, and `AttributeMapper` takes `GenerationOutcome[object]`.
_OutputT_co = TypeVar("_OutputT_co", covariant=True)


class _GenerationRecordBase(_InputOutcomeRecordBase):
    """Properties and invariants shared by the two generation records.

    Validation rejects unknown fields.
    """

    stop_reason: StopReason

    @model_validator(mode="after")
    def _validate_generation(self) -> Self:
        _require_finished_assistant_message(self.request_history)
        return self

    @property
    def request_records(self) -> tuple[SettledRequestRecord, ...]:
        """Return `request_history.records`, which validation guarantees are settled."""
        return _settled_request_records(self.request_history)

    @property
    def kept_request(self) -> SettledRequestRecord:
        """Return the final request, whose response holds the kept assistant message."""
        final = self.request_history.records[-1]
        assert final.kind == "settled"
        return final

    @property
    @override
    def assistant_message(self) -> AssistantMessage:
        """Return the kept assistant message, from the final request."""
        assistant_message = self.kept_request.assistant_message
        assert assistant_message is not None
        return assistant_message

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the kept assistant message's tool calls."""
        return self.assistant_message.tool_calls


class GenerationWithoutToolCallsRecord[OutputT](_GenerationRecordBase):
    """The normalized record of one `GenerationWithoutToolCalls`.

    Validation rejects unknown fields.
    """

    output: OutputT
    kind: Literal["without_tool_calls"] = "without_tool_calls"


class GenerationWithToolCallsRecord[OutputT](_GenerationRecordBase):
    """The normalized record of one `GenerationWithToolCalls`.

    `output` is `None` when a structured binding's kept assistant message has no validated caller model.

    Validation rejects unknown fields.
    """

    output: OutputT
    kind: Literal["with_tool_calls"] = "with_tool_calls"

    @model_validator(mode="after")
    def _validate_tool_call(self) -> Self:
        if not self.assistant_message.tool_calls:
            raise ValueError("a GenerationWithToolCallsRecord must contain at least one tool call")
        return self


@dataclass(frozen=True, kw_only=True)
class _LiveGenerationBase[RecordT: _GenerationRecordBase]:
    """Delegate live generation properties to one normalized record."""

    record: RecordT
    request_provider_data: tuple[RequestProviderData, ...]

    def __post_init__(self) -> None:
        """Require aligned provider data with a final provider response."""
        _validate_live_generation(self.record, self.request_provider_data)

    @property
    def raw(self) -> BaseModel:
        """Return the final provider SDK response."""
        return _final_raw(self.request_provider_data)

    @property
    def request_history(self) -> RequestHistory:
        """Return the normalized request history."""
        return self.record.request_history

    @property
    def stop_reason(self) -> StopReason:
        """Return the normalized stop reason."""
        return self.record.stop_reason

    @property
    def assistant_message(self) -> AssistantMessage:
        """Return the kept assistant message."""
        return self.record.assistant_message

    @property
    def request_records(self) -> tuple[SettledRequestRecord, ...]:
        """Return `request_history.records`, which validation guarantees are settled."""
        return self.record.request_records

    @property
    def kept_request(self) -> SettledRequestRecord:
        """Return the final request, whose response holds the kept assistant message."""
        return self.record.kept_request

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every request."""
        return self.record.usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the kept assistant message's tool calls."""
        return self.record.tool_calls


@dataclass(frozen=True, kw_only=True)
class GenerationWithoutToolCalls(
    _LiveGenerationBase[GenerationWithoutToolCallsRecord[_OutputT_co]],
    Generic[_OutputT_co],  # noqa: UP046
):
    """A generation whose kept assistant message has no tool calls, with provider SDK values."""

    kind: Literal["without_tool_calls"] = "without_tool_calls"

    @property
    def output(self) -> _OutputT_co:
        """Return the assistant text or validated caller model.

        A text binding returns the joined `TextPart` text of `assistant_message`, which is `""` without a `TextPart`.
        To continue the conversation, replay `assistant_message`.
        """
        return self.record.output


@dataclass(frozen=True, kw_only=True)
class GenerationWithToolCalls(
    _LiveGenerationBase[GenerationWithToolCallsRecord[_OutputT_co]],
    Generic[_OutputT_co],  # noqa: UP046
):
    """A generation whose kept assistant message has tool calls, with provider SDK values.

    Only a binding with tools produces one.
    """

    kind: Literal["with_tool_calls"] = "with_tool_calls"

    @property
    def output(self) -> _OutputT_co:
        """Return the kept assistant message's text or validated caller model.

        A text binding returns the joined `TextPart` text of `assistant_message`, which is `""` without a `TextPart`.
        A structured binding returns `None` when the kept assistant message has no validated caller model.
        To continue the conversation, replay `assistant_message`.
        """
        return self.record.output


def _validate_live_generation(
    record: _GenerationRecordBase, request_provider_data: tuple[RequestProviderData, ...]
) -> None:
    if len(request_provider_data) != len(record.request_history.records):
        raise ValueError("request_provider_data must align with request_history.records")
    _ = _final_raw(request_provider_data)


def _final_raw(request_provider_data: tuple[RequestProviderData, ...]) -> BaseModel:
    final_raw = request_provider_data[-1].raw
    if final_raw is None:
        raise ValueError("a live generation requires a final provider response")
    return final_raw


type Generation[OutputT, WithToolCallsOutputT = OutputT] = (
    GenerationWithoutToolCalls[OutputT] | GenerationWithToolCalls[WithToolCallsOutputT]
)
"""What an input produces when it succeeds. `WithToolCallsOutputT` is `OutputT | None` for a structured binding."""
type GenerationRecord[OutputT, WithToolCallsOutputT] = (
    GenerationWithoutToolCallsRecord[OutputT] | GenerationWithToolCallsRecord[WithToolCallsOutputT]
)
"""The normalized record of one `GenerationWithoutToolCalls` or `GenerationWithToolCalls`.

`WithToolCallsOutputT` has no default: pydantic 2.13.5 ignores a `type` alias default that names another type parameter.
pydantic then validates that argument as `Any`.
"""
type GenerationOutcome[OutputT, WithToolCallsOutputT = OutputT] = (
    Generation[OutputT, WithToolCallsOutputT] | GenerationError
)
"""Every live outcome of an input: a `Generation` or a `GenerationError`.

`InputOutcomeRecord` also covers abandonment, which has no live form.
"""

type GenerationOutcomeRecord[OutputT, WithToolCallsOutputT] = Annotated[
    SerializeAsAny[GenerationWithoutToolCallsRecord[OutputT]]
    | SerializeAsAny[GenerationWithToolCallsRecord[WithToolCallsOutputT]]
    | GenerationErrorRecord
    | SchemaViolationErrorRecord,
    Field(discriminator="kind"),
]
"""The normalized record of one `GenerationOutcome`.

`WithToolCallsOutputT` has no default: pydantic 2.13.5 ignores a `type` alias default that names another type parameter.
pydantic then validates that argument as `Any`.
"""

type InputOutcomeRecord[OutputT, WithToolCallsOutputT] = (
    GenerationOutcomeRecord[OutputT, WithToolCallsOutputT] | AbandonedStreamRecord
)
"""Every record of an input's outcome, including a stream the application abandoned."""


def _generation_outcome_record[OutputT, WithToolCallsOutputT](
    generation_outcome: GenerationOutcome[OutputT, WithToolCallsOutputT]
    | GenerationOutcomeRecord[OutputT, WithToolCallsOutputT],
) -> GenerationOutcomeRecord[OutputT, WithToolCallsOutputT]:
    if isinstance(
        generation_outcome, (GenerationWithoutToolCalls, GenerationWithToolCalls, GenerationError)
    ):
        if type(generation_outcome) not in (
            GenerationWithoutToolCalls,
            GenerationWithToolCalls,
            GenerationError,
        ):
            raise TypeError(f"unsupported generation outcome: {type(generation_outcome).__name__}")
        record = generation_outcome.record
        if (
            isinstance(record, _GenerationRecordBase)
            or type(record) in _GENERATION_ERROR_RECORD_CLASSES
        ):
            return record
        raise TypeError(f"unsupported generation outcome record: {type(record).__name__}")
    if isinstance(generation_outcome, _GenerationRecordBase):
        return generation_outcome
    if type(generation_outcome) in _GENERATION_ERROR_RECORD_CLASSES:
        return generation_outcome
    raise TypeError(f"unsupported generation outcome: {type(generation_outcome).__name__}")


def _generation_variant[OutputT](
    *,
    splits_on_tool_calls: bool,
    output: OutputT,
    request_history: RequestHistory,
    request_provider_data: tuple[RequestProviderData, ...],
    stop_reason: StopReason,
) -> Generation[OutputT]:
    """Build one live generation and its normalized record."""
    final = request_history.records[-1]
    assert final.kind == "settled"
    assert final.assistant_message is not None
    if splits_on_tool_calls and final.assistant_message.tool_calls:
        return GenerationWithToolCalls(
            record=GenerationWithToolCallsRecord(
                output=output, request_history=request_history, stop_reason=stop_reason
            ),
            request_provider_data=request_provider_data,
        )
    return GenerationWithoutToolCalls(
        record=GenerationWithoutToolCallsRecord(
            output=output, request_history=request_history, stop_reason=stop_reason
        ),
        request_provider_data=request_provider_data,
    )


def _generation_outcome_from_response_outcome[OutputT](
    outcome: ResponseOutcome[OutputT],
    *,
    request_history: RequestHistory,
    request_provider_data: tuple[RequestProviderData, ...],
    request_params: RequestParams | None,
    splits_on_tool_calls: bool,
) -> GenerationOutcome[OutputT]:
    match outcome.kind:
        case "usable_response":
            return _generation_variant(
                splits_on_tool_calls=splits_on_tool_calls,
                output=outcome.output,
                request_history=request_history,
                request_provider_data=request_provider_data,
                stop_reason=outcome.stop_reason,
            )
        case "refusal":
            record = GenerationErrorRecord(kind="refusal_error", request_history=request_history)
        case "max_completion_tokens_exceeded":
            record = GenerationErrorRecord(
                kind="max_completion_tokens_exceeded_error", request_history=request_history
            )
        case "empty_assistant_message":
            record = GenerationErrorRecord(
                kind="empty_assistant_message_error", request_history=request_history
            )
        case "schema_violation":
            record = SchemaViolationErrorRecord(
                validation_error_json=outcome.validation_error_json,
                request_history=request_history,
            )
        case "context_window_exceeded":
            record = GenerationErrorRecord(
                kind="context_window_exceeded_error", request_history=request_history
            )
        case "unfinished_assistant_message":
            record = GenerationErrorRecord(
                kind="unfinished_assistant_message_error",
                error_text=outcome.error_text,
                request_history=request_history,
            )
        case "provider_failed_terminally":
            record = GenerationErrorRecord(
                kind="provider_failed_terminally_error",
                error_text=outcome.error_text,
                request_history=request_history,
            )
        case "provider_failed_transiently":
            raise ValueError("ProviderFailedTransiently requires the caller's retry policy")
    return GenerationError(
        record=record,
        request_params=request_params,
        request_provider_data=request_provider_data,
    )


def _timed_out_error(
    ledger: _RequestLedger, provider_billing_in_flight: ProviderBilling | None = None
) -> GenerationError:
    """Build the expired deadline's failure with one normalized cut-off request."""
    request_history, request_provider_data = ledger.freeze_with_cut_off(provider_billing_in_flight)
    return GenerationError(
        record=GenerationErrorRecord(kind="timed_out_error", request_history=request_history),
        request_params=None,
        request_provider_data=request_provider_data,
    )


def _escaped_error(ledger: _RequestLedger, escaped: Exception) -> GenerationError:
    """Build the failure for an `Exception` that escaped failure handling, with one normalized cut-off request.

    The cut-off request carries the billing noted on `ledger` for the request in flight.
    The caller sets `escaped` as the cause.
    """
    request_history, request_provider_data = ledger.freeze_with_cut_off()
    return GenerationError(
        record=GenerationErrorRecord(
            kind="unknown_exception_error",
            error_text=str(escaped),
            request_history=request_history,
        ),
        request_params=None,
        request_provider_data=request_provider_data,
    )
