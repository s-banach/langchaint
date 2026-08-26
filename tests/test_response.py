"""Test normalized generation records, live wrappers, result normalization, and tables."""

import math
from collections.abc import Callable
from typing import override

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint import (
    ZERO_USAGE,
    AbandonedCallErrorRecord,
    AssistantMessage,
    AttemptProviderData,
    Billing,
    CallRecord,
    CallResultRecord,
    ContextWindowExceededErrorRecord,
    CutOffAttemptRecord,
    EmptyTurnErrorRecord,
    EscapedExceptionErrorRecord,
    GenerationError,
    GenerationErrorRecord,
    InvalidRequestErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    ProviderDeclaredFinalErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    RefusalErrorRecord,
    Response,
    ResponseRecord,
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    SchemaViolationErrorRecord,
    SettledAttemptRecord,
    TextPart,
    TimedOutErrorRecord,
    ToolCall,
    ToolCallTurn,
    ToolCallTurnRecord,
    TransientErrorRecord,
    UnfinishedTurnErrorRecord,
    UnknownExceptionErrorRecord,
    Usage,
    to_tables,
)
from langchaint.adapter import ProviderBilling, RequestParams
from langchaint.call import ResponseIdentity, _CallLedger, _less_than_or_ulp_close
from langchaint.response import _abandoned_call_error, _success_variant
from tests.helpers import StubRaw


class Report(BaseModel):
    """One caller-supplied structured output."""

    value: int


class DifferentReport(BaseModel):
    """A caller output type incompatible with `Report` JSON."""

    text: str


class ProviderUsage(BaseModel):
    """One provider-specific usage value."""

    billed_units: int


class StubRequest(RequestParams):
    """One live-only request for table tests."""

    @override
    def as_json(self) -> str:
        """Return fixed request JSON."""
        return '{"prompt":"hi"}'


_USAGE = Usage(
    input_tokens_cache_read=2,
    input_tokens_cache_write=3,
    input_tokens_cache_none=5,
    output_tokens=7,
    output_tokens_reasoning=1,
    input_tokens_cache_read_cost_in_usd=0.2,
    input_tokens_cache_write_cost_in_usd=0.3,
    input_tokens_cache_none_cost_in_usd=0.5,
    output_tokens_cost_in_usd=0.7,
    provider_executed_tool_cost_in_usd=0.0,
)

_BILLING = Billing(
    usage=_USAGE,
    service_tier="standard",
    input_cache_none_usd_per_million_tokens=1.0,
    cache_read_usd_per_million_tokens=math.nan,
    cache_write_usd_per_million_tokens=math.inf,
    output_usd_per_million_tokens=-math.inf,
)

_TURN = AssistantMessage(turn=(TextPart(text="done"),))
_TOOL_TURN = AssistantMessage(turn=(ToolCall(id="call-1", name="lookup", args_json="{}"),))


def _settled(
    *,
    started_after_seconds: float = 0.0,
    elapsed_seconds: float = 1.0,
    error: TransientErrorRecord | None = None,
    billing: Billing | None = _BILLING,
    assistant_message: AssistantMessage | None = _TURN,
) -> SettledAttemptRecord:
    return SettledAttemptRecord(
        started_after_seconds=started_after_seconds,
        elapsed_seconds=elapsed_seconds,
        seconds_to_first_item=None,
        error=error,
        billing=billing,
        assistant_message=assistant_message,
        model_served="served" if error is None else None,
        response_id="response" if error is None else None,
        request_id="request",
    )


def _call(*attempts: SettledAttemptRecord | CutOffAttemptRecord) -> CallRecord:
    elapsed_seconds = 0.0
    for attempt in attempts:
        attempt_end = attempt.started_after_seconds
        if isinstance(attempt, SettledAttemptRecord):
            attempt_end += attempt.elapsed_seconds
        elapsed_seconds = max(elapsed_seconds, attempt_end)
    return CallRecord(
        model="model",
        provider_name="provider",
        attempt_records=attempts,
        elapsed_seconds=elapsed_seconds,
    )


def _failed_call() -> CallRecord:
    return _call(
        _settled(
            elapsed_seconds=0.5,
            error=TransientErrorRecord(message="retry", retry_after_seconds=0.25),
            billing=None,
            assistant_message=None,
        )
    )


def _completed_turn_call(*, assistant_message: AssistantMessage = _TURN) -> CallRecord:
    return _call(_settled(assistant_message=assistant_message))


def _provider_attempts(raw: BaseModel | None = None) -> tuple[AttemptProviderData, ...]:
    return (
        AttemptProviderData(
            raw=StubRaw() if raw is None else raw,
            usage_raw=ProviderUsage(billed_units=17),
        ),
    )


def test_usage_and_billing_nonfinite_values_round_trip_as_strings() -> None:
    """Normalized non-finite floats serialize as reconstructible strings."""
    billing_json = _BILLING.model_dump_json()
    assert '"NaN"' in billing_json
    assert '"Infinity"' in billing_json
    assert '"-Infinity"' in billing_json
    restored = Billing.model_validate_json(billing_json)
    assert math.isnan(restored.cache_read_usd_per_million_tokens)
    assert restored.cache_write_usd_per_million_tokens == math.inf
    assert restored.output_usd_per_million_tokens == -math.inf


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_usage_nonfinite_cost_round_trips(value: float) -> None:
    """Each supported non-finite usage cost reconstructs its float value."""
    usage = ZERO_USAGE.model_copy(update={"output_tokens_cost_in_usd": value})
    restored = Usage.model_validate_json(usage.model_dump_json())
    if math.isnan(value):
        assert math.isnan(restored.output_tokens_cost_in_usd)
    else:
        assert restored.output_tokens_cost_in_usd == value


def test_transient_error_record_rejects_invalid_retry_delays() -> None:
    """Retry delays must be finite and nonnegative."""
    for value in (-1.0, math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            _ = TransientErrorRecord(message="retry", retry_after_seconds=value)


def test_attempt_timing_rejects_invalid_values_and_first_item_order() -> None:
    """Attempt timing rejects negative durations and late first items."""
    with pytest.raises(ValidationError):
        _ = _settled(elapsed_seconds=-1.0)
    with pytest.raises(ValidationError):
        _ = SettledAttemptRecord(
            started_after_seconds=0.0,
            elapsed_seconds=1.0,
            seconds_to_first_item=2.0,
            error=None,
            billing=_BILLING,
            assistant_message=_TURN,
            model_served=None,
            response_id=None,
            request_id=None,
        )


def test_less_than_or_ulp_close_accepts_the_documented_rounding_boundary() -> None:
    """The timing comparison accepts four ULPs and rejects five ULPs."""
    right = 1.0
    assert _less_than_or_ulp_close(right + 4 * math.ulp(right), right)
    assert not _less_than_or_ulp_close(right + 5 * math.ulp(right), right)


def test_call_record_rejects_overlap_out_of_bounds_and_cut_off_placement() -> None:
    """Call validation enforces ordering, bounds, and final cut-off placement."""
    first = _settled(elapsed_seconds=1.0)
    overlap = _settled(started_after_seconds=0.5, elapsed_seconds=0.5)
    with pytest.raises(ValidationError, match="overlap"):
        _ = CallRecord(
            model="m",
            provider_name="p",
            attempt_records=(first, overlap),
            elapsed_seconds=1.0,
        )
    with pytest.raises(ValidationError, match="within the call"):
        _ = CallRecord(
            model="m",
            provider_name="p",
            attempt_records=(first,),
            elapsed_seconds=0.5,
        )
    with pytest.raises(ValidationError, match="must be final"):
        _ = CallRecord(
            model="m",
            provider_name="p",
            attempt_records=(
                CutOffAttemptRecord(started_after_seconds=0.0, billing=None),
                _settled(started_after_seconds=1.0),
            ),
            elapsed_seconds=2.0,
        )


def test_response_record_round_trips_with_concrete_output_type_and_message_bytes() -> None:
    """A concrete caller model reconstructs through `ResponseRecord` JSON."""
    record = ResponseRecord(
        output=Report(value=3), call=_completed_turn_call(), stop_reason="end_turn"
    )
    response_json = record.model_dump_json()
    restored = ResponseRecord[Report].model_validate_json(response_json)
    assert restored.model_dump_json() == response_json
    assert isinstance(restored.output, Report)


def test_response_record_rejects_a_mismatched_output_type() -> None:
    """`model_validate_json` validates the concrete caller output type."""
    record = ResponseRecord(
        output=Report(value=3), call=_completed_turn_call(), stop_reason="end_turn"
    )
    with pytest.raises(ValidationError):
        _ = ResponseRecord[DifferentReport].model_validate_json(record.model_dump_json())


def test_caller_output_with_nonreconstructible_json_fails_on_validation() -> None:
    """A caller model that serializes NaN as null fails reconstruction."""

    class NonReconstructibleOutput(BaseModel):
        value: float

    record = ResponseRecord(
        output=NonReconstructibleOutput(value=math.nan),
        call=_completed_turn_call(),
        stop_reason="end_turn",
    )
    with pytest.raises(ValidationError):
        _ = ResponseRecord[NonReconstructibleOutput].model_validate_json(record.model_dump_json())


def test_success_record_rejects_unknown_fields_and_invalid_attempt_shapes() -> None:
    """Success records reject extra fields and incomplete successful requests."""
    valid = ResponseRecord(output=1, call=_completed_turn_call(), stop_reason="end_turn")
    payload = valid.model_dump()
    with pytest.raises(ValidationError, match="extra"):
        _ = ResponseRecord[int].model_validate(payload | {"extra": True})
    with pytest.raises(TypeError, match="not a field"):
        _ = valid.model_copy(update={"extra": True})
    with pytest.raises(ValidationError, match="final attempt must contain billing"):
        _ = ResponseRecord(
            output=1,
            call=_call(_settled(billing=None)),
            stop_reason="end_turn",
        )
    with pytest.raises(ValidationError, match="cut-off"):
        _ = ResponseRecord(
            output=1,
            call=_call(CutOffAttemptRecord(started_after_seconds=0.0, billing=None)),
            stop_reason="end_turn",
        )


def test_tool_call_turn_record_requires_a_tool_call() -> None:
    """A `ToolCallTurnRecord` requires a tool call in its final turn."""
    with pytest.raises(ValidationError, match="tool call"):
        _ = ToolCallTurnRecord(output=None, call=_completed_turn_call(), stop_reason="tool_use")
    record = ToolCallTurnRecord(
        output=Report(value=4),
        call=_completed_turn_call(assistant_message=_TOOL_TURN),
        stop_reason="tool_use",
    )
    restored = ToolCallTurnRecord[Report].model_validate_json(record.model_dump_json())
    assert restored.tool_calls == _TOOL_TURN.tool_calls
    assert isinstance(restored.output, Report)


def test_live_success_delegates_normalized_properties_and_preserves_provider_data() -> None:
    """A live success delegates normalized values and retains provider values."""
    record = ResponseRecord(
        output=Report(value=5), call=_completed_turn_call(), stop_reason="end_turn"
    )
    provider_attempts = _provider_attempts()
    response = Response(record=record, provider_attempts=provider_attempts)
    assert response.raw is provider_attempts[0].raw
    assert response.output is record.output
    assert response.call == record.call
    assert response.attempt_records == record.attempt_records
    assert response.attempts == record.attempts
    assert response.usage == record.usage
    assert response.model == record.model
    assert response.provider_name == record.provider_name
    assert response.elapsed_seconds == record.elapsed_seconds
    assert response.assistant_message == record.assistant_message
    assert response.usage_successful_attempt == record.usage_successful_attempt
    assert response.stop_reason == record.stop_reason
    assert response.tool_calls == record.tool_calls
    with pytest.raises(ValueError, match="align"):
        _ = Response(record=record, provider_attempts=())


def test_live_success_requires_provider_data_from_the_final_attempt() -> None:
    """A live success rejects an earlier provider response when its final attempt has none."""
    failed_attempt = _settled(
        elapsed_seconds=0.5,
        error=TransientErrorRecord(message="retry"),
        billing=None,
        assistant_message=None,
    )
    successful_attempt = _settled(started_after_seconds=0.5, elapsed_seconds=0.5)
    record = ResponseRecord(
        output=Report(value=5),
        call=_call(failed_attempt, successful_attempt),
        stop_reason="end_turn",
    )
    provider_attempts = (
        AttemptProviderData(raw=StubRaw(), usage_raw=None),
        AttemptProviderData(raw=None, usage_raw=None),
    )
    with pytest.raises(ValueError, match="final provider response"):
        _ = Response(record=record, provider_attempts=provider_attempts)


def test_success_variant_constructs_one_normalized_record() -> None:
    """The success factory stores one normalized tool-call record by reference."""
    call = _completed_turn_call(assistant_message=_TOOL_TURN)
    result = _success_variant(
        splits_tool_call_turns=True,
        output=Report(value=6),
        call=call,
        provider_attempts=_provider_attempts(),
        stop_reason="tool_use",
    )
    assert isinstance(result, ToolCallTurn)
    assert isinstance(result.record, ToolCallTurnRecord)
    assert result.record.call is call


def _error_records() -> list[GenerationErrorRecord]:
    completed = _completed_turn_call()
    failed = _failed_call()
    terminal = _call(_settled(billing=None, assistant_message=None))
    cut_off = _call(CutOffAttemptRecord(started_after_seconds=0.0, billing=_BILLING))
    return [
        RetriesExhaustedErrorRecord(call=failed),
        RetryUnavailableErrorRecord(call=failed),
        RefusalErrorRecord(call=completed),
        MaxCompletionTokensExceededErrorRecord(call=completed),
        EmptyTurnErrorRecord(call=completed),
        SchemaViolationErrorRecord(call=completed, validation_error_json="[]"),
        ContextWindowExceededErrorRecord(call=completed),
        UnfinishedTurnErrorRecord(call=completed, reason="unfinished"),
        ProviderFailedTerminallyErrorRecord(call=completed, reason="failed"),
        InvalidRequestErrorRecord(
            call=CallRecord(
                model="model", provider_name="provider", attempt_records=(), elapsed_seconds=0.0
            ),
            reason="invalid",
        ),
        ProviderDeclaredFinalErrorRecord(call=terminal, reason="terminal"),
        UnknownExceptionErrorRecord(
            call=CallRecord(
                model="model", provider_name="provider", attempt_records=(), elapsed_seconds=0.0
            ),
            reason="unknown",
        ),
        EscapedExceptionErrorRecord(
            call=CallRecord(
                model="model", provider_name="provider", attempt_records=(), elapsed_seconds=0.0
            ),
            reason="escaped",
        ),
        AbandonedCallErrorRecord(call=cut_off),
        TimedOutErrorRecord(call=cut_off),
    ]


def test_every_error_record_round_trips_through_the_closed_union() -> None:
    """The closed error union reconstructs each built-in error record."""
    adapter = TypeAdapter(GenerationErrorRecord)
    for record in _error_records():
        assert record.model_config.get("frozen") is True
        record_json_text = record.model_dump_json()
        restored_direct = type(record).model_validate_json(record_json_text)
        assert restored_direct.model_dump_json() == record_json_text
        record_json_bytes = adapter.dump_json(record)
        restored = adapter.validate_json(record_json_bytes)
        assert type(restored) is type(record)
        assert adapter.dump_json(restored) == record_json_bytes
        assert str(restored) == restored.error_summary


def test_closed_result_unions_reject_unknown_kinds() -> None:
    """Unknown discriminators fail validation for error and mixed result records."""
    record = RefusalErrorRecord(call=_completed_turn_call())
    payload: dict[str, object] = TypeAdapter(GenerationErrorRecord).dump_python(record)
    payload["kind"] = "future_error"
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        _ = TypeAdapter(GenerationErrorRecord).validate_python(payload)
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        _ = TypeAdapter(CallResultRecord[Report]).validate_python(payload)


def test_error_record_properties_and_stable_text() -> None:
    """Error records retain normalized properties and tracing-safe text."""
    exhausted = RetriesExhaustedErrorRecord(call=_failed_call())
    assert [str(error) for error in exhausted.errors_from_attempts] == ["retry"]
    assert exhausted.error_text == "attempt 1: retry"
    assert exhausted.assistant_message is None
    refusal = RefusalErrorRecord(call=_completed_turn_call())
    assert refusal.stop_reason == "refusal"
    assert refusal.assistant_message == _TURN
    assert refusal.usage == _USAGE


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: RetriesExhaustedErrorRecord(call=_completed_turn_call()), "transient error"),
        (lambda: RefusalErrorRecord(call=_failed_call()), "final attempt must be error-free"),
        (
            lambda: ProviderDeclaredFinalErrorRecord(
                call=CallRecord(
                    model="m", provider_name="p", attempt_records=(), elapsed_seconds=0.0
                ),
                reason="terminal",
            ),
            "final provider result",
        ),
    ],
)
def test_error_records_reject_invalid_call_shapes(
    factory: Callable[[], GenerationErrorRecord], match: str
) -> None:
    """Each constrained error group rejects a call shape outside its contract."""
    with pytest.raises(ValidationError, match=match):
        _ = factory()


def test_generation_error_delegates_record_and_keeps_live_only_state() -> None:
    """`GenerationError` delegates normalized values and retains live-only state."""
    record = RefusalErrorRecord(call=_completed_turn_call())
    provider_attempts = _provider_attempts()
    failure = GenerationError(
        record=record, request=StubRequest(), provider_attempts=provider_attempts
    )
    assert failure.request is not None
    assert failure.provider_attempts is provider_attempts
    assert str(failure) == record.error_summary
    assert failure.error_text == record.error_text
    assert failure.call == record.call
    assert failure.attempt_records == record.attempt_records
    assert failure.attempts == record.attempts
    assert failure.usage == record.usage
    assert failure.model == record.model
    assert failure.provider_name == record.provider_name
    assert failure.elapsed_seconds == record.elapsed_seconds
    assert failure.assistant_message == record.assistant_message
    assert failure.stop_reason == record.stop_reason
    with pytest.raises(ValueError, match="align"):
        _ = GenerationError(record=record, request=None, provider_attempts=())


def test_mixed_normalized_result_list_round_trips() -> None:
    """A mixed normalized result list reconstructs through its concrete output type."""
    records: list[CallResultRecord[Report]] = [
        ResponseRecord(
            output=Report(value=9), call=_completed_turn_call(), stop_reason="end_turn"
        ),
        RefusalErrorRecord(call=_completed_turn_call()),
    ]
    adapter = TypeAdapter(list[CallResultRecord[Report]])
    records_json = adapter.dump_json(records)
    restored = adapter.validate_json(records_json)
    assert adapter.dump_json(restored) == records_json
    assert to_tables(restored).calls[0]["output"] == '{"value":9}'


def test_to_tables_reads_live_only_request_and_provider_usage() -> None:
    """Tables read request and provider usage only from live results."""
    record = RefusalErrorRecord(call=_completed_turn_call())
    live = GenerationError(
        record=record, request=StubRequest(), provider_attempts=_provider_attempts()
    )
    live_tables = to_tables(live)
    normalized_tables = to_tables(record)
    assert live_tables.calls[0]["request_json"] == '{"prompt":"hi"}'
    assert normalized_tables.calls[0]["request_json"] is None
    assert live_tables.attempts[0]["usage_raw_json"] == '{"billed_units":17}'
    assert normalized_tables.attempts[0]["usage_raw_json"] is None
    assert live_tables.attempts[0]["started_after_seconds"] == 0.0


def test_to_tables_emits_one_row_for_a_cut_off_attempt() -> None:
    """A cut-off request produces one attempt row with no fabricated ending."""
    record = TimedOutErrorRecord(
        call=_call(CutOffAttemptRecord(started_after_seconds=0.25, billing=_BILLING))
    )
    tables = to_tables(record)
    assert tables.calls[0]["attempts"] == 1
    assert tables.attempts == [
        tables.attempts[0]
        | {
            "started_after_seconds": 0.25,
            "elapsed_seconds": None,
            "seconds_to_first_item": None,
        }
    ]
    assert tables.attempts[0]["cost_in_usd"] == _USAGE.cost_in_usd


def test_abandoned_call_error_appends_one_cut_off_request_with_live_usage() -> None:
    """A live timeout aligns its cut-off record with provider usage."""
    ledger = _CallLedger(model="model", provider_name="provider")
    ledger.start_call()
    ledger.start_attempt()
    provider_billing = ProviderBilling(billing=_BILLING, usage_raw=ProviderUsage(billed_units=23))
    failure = _abandoned_call_error(TimedOutErrorRecord, ledger, provider_billing)
    assert failure.record.kind == "timed_out_error"
    assert len(failure.attempt_records) == 1
    assert isinstance(failure.attempt_records[0], CutOffAttemptRecord)
    assert failure.provider_attempts[0].usage_raw == provider_billing.usage_raw
    assert failure.usage == _USAGE


@pytest.mark.parametrize("record_class", [AbandonedCallErrorRecord, TimedOutErrorRecord])
def test_interrupted_error_record_accepts_a_transient_prefix_without_a_cut_off(
    record_class: type[AbandonedCallErrorRecord] | type[TimedOutErrorRecord],
) -> None:
    """An interruption during retry backoff retains its transient settled attempt."""
    record = record_class(call=_failed_call())
    assert record.call == _failed_call()


def test_interruption_after_a_staged_response_records_no_cut_off_request() -> None:
    """A staged provider response settles before an interrupted call freezes."""
    ledger = _CallLedger(model="model", provider_name="provider")
    ledger.start_call()
    ledger.start_attempt()
    raw = StubRaw()
    provider_billing = ProviderBilling(billing=_BILLING, usage_raw=None)
    ledger.stage_response(
        raw=raw,
        billing=provider_billing,
        identity=ResponseIdentity(
            model_served="served", response_id="response", request_id="request"
        ),
    )
    failure = _abandoned_call_error(TimedOutErrorRecord, ledger)
    assert len(failure.attempt_records) == 1
    assert isinstance(failure.attempt_records[0], SettledAttemptRecord)
    assert failure.provider_attempts[0].raw is raw
