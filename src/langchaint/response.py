"""Live generation results, normalized result records, and table conversion."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Generic, Literal, NamedTuple, Self, TypeVar

from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from langchaint.adapter import RequestParams, ResponseOutcome
from langchaint.call import (
    AttemptProviderData,
    CallRecord,
    CutOffAttemptRecord,
    SettledAttemptRecord,
    _CallLedger,
    _CallResultRecordBase,
    _require_completed_model_turn,
    _settled_attempts,
)
from langchaint.exceptions import (
    _GENERATION_ERROR_RECORD_CLASSES,
    AbandonedCallErrorRecord,
    ContextWindowExceededErrorRecord,
    EmptyTurnErrorRecord,
    GenerationError,
    GenerationErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    RefusalErrorRecord,
    SchemaViolationErrorRecord,
    TimedOutErrorRecord,
    UnfinishedTurnErrorRecord,
    _GenerationErrorRecordBase,
)
from langchaint.messages import AssistantMessage, StopReason, ToolCall
from langchaint.pricing import Billing, ProviderBilling
from langchaint.usage import Usage

_OutputT_co = TypeVar("_OutputT_co", covariant=True)


class _SuccessRecordBase(_CallResultRecordBase):
    """Properties and invariants shared by normalized success records.

    Validation rejects unknown fields.
    """

    @model_validator(mode="after")
    def _validate_success(self) -> Self:
        _require_completed_model_turn(self.call)
        return self

    @property
    def attempt_records(
        self,
    ) -> tuple[SettledAttemptRecord, ...]:
        """Return the call's normalized request records."""
        return _settled_attempts(self.call)

    @property
    def assistant_message(self) -> AssistantMessage:
        """Return the successful attempt's assistant message."""
        final = self.call.attempt_records[-1]
        assert isinstance(final, SettledAttemptRecord)
        assert final.assistant_message is not None
        return final.assistant_message

    @property
    def usage_successful_attempt(self) -> Usage:
        """Return normalized usage for the successful request."""
        return self.call.attempt_records[-1].usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the final turn's tool calls."""
        return self.assistant_message.tool_calls


class ResponseRecord[OutputT](_SuccessRecordBase):
    """One normalized successful generation result.

    Validation rejects unknown fields.
    """

    output: OutputT
    stop_reason: StopReason
    kind: Literal["response"] = "response"


class ToolCallTurnRecord[OutputT](_SuccessRecordBase):
    """One normalized structured result whose turn called tools.

    Validation rejects unknown fields.
    """

    output: OutputT | None
    stop_reason: StopReason
    kind: Literal["tool_call_turn"] = "tool_call_turn"

    @model_validator(mode="after")
    def _validate_tool_call(self) -> Self:
        if not self.assistant_message.tool_calls:
            raise ValueError("a ToolCallTurnRecord must contain at least one tool call")
        return self


class _LiveSuccess(Generic[_OutputT_co]):  # noqa: UP046
    """Delegate live success properties to one normalized record."""

    record: ResponseRecord[_OutputT_co] | ToolCallTurnRecord[_OutputT_co]
    provider_attempts: tuple[AttemptProviderData, ...]

    def __post_init__(self) -> None:
        """Require aligned provider data with a final provider response."""
        _validate_live_success(self.record, self.provider_attempts)

    @property
    def raw(self) -> BaseModel:
        """Return the final provider SDK response."""
        return _final_raw(self.provider_attempts)

    @property
    def call(self) -> CallRecord:
        """Return the normalized call record."""
        return self.record.call

    @property
    def stop_reason(self) -> StopReason:
        """Return the normalized stop reason."""
        return self.record.stop_reason

    @property
    def assistant_message(self) -> AssistantMessage:
        """Return the final assistant message."""
        return self.record.assistant_message

    @property
    def attempt_records(
        self,
    ) -> tuple[SettledAttemptRecord, ...]:
        """Return normalized attempt records in request order."""
        return self.record.attempt_records

    @property
    def attempts(self) -> int:
        """Return the observed request count."""
        return self.record.attempts

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every request."""
        return self.record.usage

    @property
    def usage_successful_attempt(self) -> Usage:
        """Return normalized usage for the successful request."""
        return self.record.usage_successful_attempt

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
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the final assistant message's tool calls."""
        return self.record.tool_calls


@dataclass(frozen=True, kw_only=True)
class Response(_LiveSuccess[_OutputT_co], Generic[_OutputT_co]):  # noqa: UP046
    """A live success with a normalized record and provider SDK values."""

    # pyrefly: ignore[bad-override]  # The frozen live value makes this field read-only.
    record: ResponseRecord[_OutputT_co]
    # pyrefly: ignore[bad-override]  # The frozen live value makes this field read-only.
    provider_attempts: tuple[AttemptProviderData, ...]
    kind: Literal["response"] = "response"

    @property
    def output(self) -> _OutputT_co:
        """Return the assistant text or validated caller model."""
        return self.record.output


@dataclass(frozen=True, kw_only=True)
class ToolCallTurn(_LiveSuccess[_OutputT_co], Generic[_OutputT_co]):  # noqa: UP046
    """A live tool-call turn with normalized and provider SDK values."""

    # pyrefly: ignore[bad-override]  # The frozen live value makes this field read-only.
    record: ToolCallTurnRecord[_OutputT_co]
    # pyrefly: ignore[bad-override]  # The frozen live value makes this field read-only.
    provider_attempts: tuple[AttemptProviderData, ...]
    kind: Literal["tool_call_turn"] = "tool_call_turn"

    @property
    def output(self) -> _OutputT_co | None:
        """Return the validated caller model or `None` for a tool-only turn."""
        return self.record.output


def _validate_live_success(
    record: _SuccessRecordBase, provider_attempts: tuple[AttemptProviderData, ...]
) -> None:
    if len(provider_attempts) != len(record.call.attempt_records):
        raise ValueError("provider_attempts must align with call.attempt_records")
    _ = _final_raw(provider_attempts)


def _final_raw(provider_attempts: tuple[AttemptProviderData, ...]) -> BaseModel:
    final_raw = provider_attempts[-1].raw
    if final_raw is None:
        raise ValueError("a live success requires a final provider response")
    return final_raw


type GenerateResult[OutputT] = Response[OutputT] | ToolCallTurn[OutputT]
type CallResult[OutputT] = GenerateResult[OutputT] | GenerationError

type CallResultRecord[OutputT] = Annotated[
    SerializeAsAny[ResponseRecord[OutputT]]
    | SerializeAsAny[ToolCallTurnRecord[OutputT]]
    | GenerationErrorRecord,
    Field(discriminator="kind"),
]


def _result_record[OutputT](
    result: CallResult[OutputT] | CallResultRecord[OutputT],
) -> CallResultRecord[OutputT]:
    if isinstance(result, (Response, ToolCallTurn, GenerationError)):
        if type(result) not in (Response, ToolCallTurn, GenerationError):
            raise TypeError(f"unsupported call result: {type(result).__name__}")
        record = result.record
        if (
            isinstance(record, _SuccessRecordBase)
            or type(record) in _GENERATION_ERROR_RECORD_CLASSES
        ):
            return record
        raise TypeError(f"unsupported call result record: {type(record).__name__}")
    if isinstance(result, _SuccessRecordBase):
        return result
    if (
        isinstance(result, _GenerationErrorRecordBase)
        and type(result) in _GENERATION_ERROR_RECORD_CLASSES
    ):
        return result
    raise TypeError(f"unsupported call result: {type(result).__name__}")


def _success_variant[OutputT](
    *,
    splits_tool_call_turns: bool,
    output: OutputT,
    call: CallRecord,
    provider_attempts: tuple[AttemptProviderData, ...],
    stop_reason: StopReason,
) -> GenerateResult[OutputT]:
    """Build one live success and its single normalized record."""
    final = call.attempt_records[-1]
    assert isinstance(final, SettledAttemptRecord)
    assert final.assistant_message is not None
    if splits_tool_call_turns and final.assistant_message.tool_calls:
        return ToolCallTurn(
            record=ToolCallTurnRecord(output=output, call=call, stop_reason=stop_reason),
            provider_attempts=provider_attempts,
        )
    return Response(
        record=ResponseRecord(output=output, call=call, stop_reason=stop_reason),
        provider_attempts=provider_attempts,
    )


def _call_result_from_response_outcome[OutputT](
    outcome: ResponseOutcome[OutputT],
    *,
    call: CallRecord,
    provider_attempts: tuple[AttemptProviderData, ...],
    request: RequestParams | None,
    splits_tool_call_turns: bool,
) -> CallResult[OutputT]:
    match outcome.kind:
        case "adapter_result":
            return _success_variant(
                splits_tool_call_turns=splits_tool_call_turns,
                output=outcome.output,
                call=call,
                provider_attempts=provider_attempts,
                stop_reason=outcome.stop_reason,
            )
        case "refusal":
            record = RefusalErrorRecord(call=call)
        case "max_completion_tokens_exceeded":
            record = MaxCompletionTokensExceededErrorRecord(call=call)
        case "empty_turn":
            record = EmptyTurnErrorRecord(call=call)
        case "schema_violation":
            record = SchemaViolationErrorRecord(
                validation_error_json=outcome.validation_error_json, call=call
            )
        case "context_window_exceeded":
            record = ContextWindowExceededErrorRecord(call=call)
        case "unfinished_turn":
            record = UnfinishedTurnErrorRecord(error_text=outcome.reason, call=call)
        case "provider_failed_terminally":
            record = ProviderFailedTerminallyErrorRecord(error_text=outcome.reason, call=call)
        case "provider_failed_transiently":
            raise ValueError("ProviderFailedTransiently requires the caller's retry policy")
    return GenerationError(
        record=record,
        request=request,
        provider_attempts=provider_attempts,
    )


def _abandoned_call_error(
    record_class: type[AbandonedCallErrorRecord] | type[TimedOutErrorRecord],
    ledger: _CallLedger,
    billing_in_flight: ProviderBilling | None = None,
) -> GenerationError:
    """Build a live interrupted failure with one normalized cut-off request."""
    call, provider_attempts = ledger.freeze_with_cut_off(billing_in_flight)
    return GenerationError(
        record=record_class(call=call), request=None, provider_attempts=provider_attempts
    )


type RowValue = str | int | float | bool | None


class Tables(NamedTuple):
    """The calls and attempts tables joined on `call_id`."""

    calls: list[dict[str, RowValue]]
    attempts: list[dict[str, RowValue]]


def _output_cell(output: object) -> str | None:
    if output is None:
        return None
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


def _billing_cells(
    billing: Billing | None, provider_data: AttemptProviderData | None
) -> dict[str, RowValue]:
    usage = None if billing is None else billing.usage
    usage_raw = None if provider_data is None else provider_data.usage_raw
    return {
        "service_tier": None if billing is None else billing.service_tier,
        "usage_raw_json": None if usage_raw is None else usage_raw.model_dump_json(),
        "input_tokens_cache_read": None if usage is None else usage.input_tokens_cache_read,
        "input_tokens_cache_read_cost_in_usd": None
        if usage is None
        else usage.input_tokens_cache_read_cost_in_usd,
        "input_tokens_cache_write": None if usage is None else usage.input_tokens_cache_write,
        "input_tokens_cache_write_cost_in_usd": None
        if usage is None
        else usage.input_tokens_cache_write_cost_in_usd,
        "input_tokens_cache_none": None if usage is None else usage.input_tokens_cache_none,
        "input_tokens_cache_none_cost_in_usd": None
        if usage is None
        else usage.input_tokens_cache_none_cost_in_usd,
        "input_tokens_total": None if usage is None else usage.input_tokens_total,
        "output_tokens": None if usage is None else usage.output_tokens,
        "output_tokens_cost_in_usd": None if usage is None else usage.output_tokens_cost_in_usd,
        "output_tokens_reasoning": None if usage is None else usage.output_tokens_reasoning,
        "provider_executed_tool_cost_in_usd": None
        if usage is None
        else usage.provider_executed_tool_cost_in_usd,
        "cost_in_usd": None if usage is None else usage.cost_in_usd,
        "input_cache_none_usd_per_million_tokens": None
        if billing is None
        else billing.input_cache_none_usd_per_million_tokens,
        "cache_read_usd_per_million_tokens": None
        if billing is None
        else billing.cache_read_usd_per_million_tokens,
        "cache_write_usd_per_million_tokens": None
        if billing is None
        else billing.cache_write_usd_per_million_tokens,
        "output_usd_per_million_tokens": None
        if billing is None
        else billing.output_usd_per_million_tokens,
    }


def _attempt_row(
    *,
    call_id: int,
    attempt_index: int,
    kept: bool,
    attempt: SettledAttemptRecord | CutOffAttemptRecord,
    provider_data: AttemptProviderData | None,
) -> dict[str, RowValue]:
    common = _billing_cells(attempt.billing, provider_data) | {
        "call_id": call_id,
        "attempt_index": attempt_index,
        "kept": kept,
        "started_after_seconds": attempt.started_after_seconds,
    }
    if isinstance(attempt, CutOffAttemptRecord):
        return common | {
            "elapsed_seconds": None,
            "seconds_to_first_item": None,
            "model_served": None,
            "response_id": None,
            "request_id": None,
            "error_text": None,
            "assistant_message_json": None,
        }
    return common | {
        "elapsed_seconds": attempt.elapsed_seconds,
        "seconds_to_first_item": attempt.seconds_to_first_item,
        "model_served": attempt.model_served,
        "response_id": attempt.response_id,
        "request_id": attempt.request_id,
        "error_text": None if attempt.error is None else str(attempt.error),
        "assistant_message_json": None
        if attempt.assistant_message is None
        else attempt.assistant_message.model_dump_json(),
    }


def to_tables[OutputT](
    results: CallResult[OutputT]
    | CallResultRecord[OutputT]
    | Iterable[CallResult[OutputT] | CallResultRecord[OutputT]],
) -> Tables:
    """Flatten live or normalized results into calls and attempts tables."""
    values = (
        list(results)
        if isinstance(results, Iterable) and not isinstance(results, BaseModel)
        else [results]
    )
    calls: list[dict[str, RowValue]] = []
    attempts: list[dict[str, RowValue]] = []
    for call_id, value in enumerate(values):
        record = _result_record(value)
        live_error = value if isinstance(value, GenerationError) else None
        live_success = value if isinstance(value, (Response, ToolCallTurn)) else None
        provider_attempts = (
            live_error.provider_attempts
            if live_error is not None
            else live_success.provider_attempts
            if live_success is not None
            else ()
        )
        is_error = isinstance(record, _GenerationErrorRecordBase)
        calls.append({
            "call_id": call_id,
            "model": record.model,
            "provider_name": record.provider_name,
            "elapsed_seconds": record.elapsed_seconds,
            "attempts": record.attempts,
            "stop_reason": record.stop_reason,
            "error_text": record.error_text if is_error else None,
            "request_json": None
            if live_error is None or live_error.request is None
            else live_error.request.as_json(),
            "output": None if is_error else _output_cell(record.output),
        })
        kept_index = None if is_error else len(record.attempt_records) - 1
        for attempt_index, attempt in enumerate(record.attempt_records):
            attempts.append(
                _attempt_row(
                    call_id=call_id,
                    attempt_index=attempt_index,
                    kept=attempt_index == kept_index,
                    attempt=attempt,
                    provider_data=(
                        provider_attempts[attempt_index] if provider_attempts else None
                    ),
                )
            )
    return Tables(calls=calls, attempts=attempts)
