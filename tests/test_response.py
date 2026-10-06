"""Test normalized generation records, live wrappers, outcome normalization, and tables."""

import json
import math
from collections.abc import Callable
from typing import override

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint import (
    ZERO_USAGE,
    AbandonedStreamRecord,
    AssistantMessage,
    Billing,
    CutOffRequestRecord,
    GenerationError,
    GenerationErrorKind,
    GenerationErrorRecord,
    GenerationOutcomeRecord,
    PlainErrorRecord,
    PlainGeneration,
    PlainGenerationRecord,
    RequestHistory,
    RequestProviderData,
    SchemaViolationErrorRecord,
    SettledRequestRecord,
    TextPart,
    TokenRates,
    ToolCall,
    ToolCallGenerationRecord,
    TransientErrorRecord,
    Usage,
    to_tables,
)
from langchaint.adapter import ProviderBilling, RequestParams, ResponseIdentity
from langchaint.generation.request_history import _less_than_or_ulp_close, _RequestLedger
from langchaint.generation.response import _generation_variant, _timed_out_error
from tests.helpers import StubRaw


class Report(BaseModel):
    """One caller-supplied structured output."""

    value: int


class ProviderUsage(BaseModel):
    """One provider-specific usage value."""

    billed_units: int


class StubRequestParams(RequestParams):
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
    usd_per_million_tokens=TokenRates(
        input_tokens_cache_read=float("nan"),
        input_tokens_cache_write=float("inf"),
        input_tokens_cache_none=1.0,
        output_tokens=float("-inf"),
    ),
)

_TEXT_ASSISTANT_MESSAGE = AssistantMessage(parts=(TextPart(text="done"),))
_TOOL_CALL_ASSISTANT_MESSAGE = AssistantMessage(
    parts=(ToolCall(id="call-1", name="lookup", args_json="{}"),)
)


def _settled(
    *,
    started_after_seconds: float = 0.0,
    elapsed_seconds: float = 1.0,
    error: TransientErrorRecord | None = None,
    billing: Billing | None = _BILLING,
    assistant_message: AssistantMessage | None = _TEXT_ASSISTANT_MESSAGE,
) -> SettledRequestRecord:
    return SettledRequestRecord(
        started_after_seconds=started_after_seconds,
        elapsed_seconds=elapsed_seconds,
        first_item_after_seconds=None,
        error=error,
        billing=billing,
        assistant_message=assistant_message,
        model_served="served" if error is None else None,
        response_id="response" if error is None else None,
        request_id="request",
    )


def _request_history(
    *records: SettledRequestRecord | CutOffRequestRecord,
) -> RequestHistory:
    elapsed_seconds = 0.0
    for request_record in records:
        request_end = request_record.started_after_seconds
        if request_record.kind == "settled":
            request_end += request_record.elapsed_seconds
        elapsed_seconds = max(elapsed_seconds, request_end)
    return RequestHistory(
        model="model",
        provider_name="provider",
        records=records,
        elapsed_seconds=elapsed_seconds,
    )


def _failed_request_history() -> RequestHistory:
    return _request_history(
        _settled(
            elapsed_seconds=0.5,
            error=TransientErrorRecord(error_text="retry", retry_after_seconds=0.25),
            billing=None,
            assistant_message=None,
        )
    )


def _completed_request_history(
    *, assistant_message: AssistantMessage = _TEXT_ASSISTANT_MESSAGE
) -> RequestHistory:
    return _request_history(_settled(assistant_message=assistant_message))


def _request_provider_data(raw: BaseModel | None = None) -> tuple[RequestProviderData, ...]:
    return (
        RequestProviderData(
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
    assert math.isnan(restored.usd_per_million_tokens.input_tokens_cache_read)
    assert restored.usd_per_million_tokens.input_tokens_cache_write == float("inf")
    assert restored.usd_per_million_tokens.output_tokens == float("-inf")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_usage_nonfinite_cost_round_trips(value: float) -> None:
    """Each supported non-finite usage cost reconstructs its float value."""
    usage = ZERO_USAGE.model_copy(update={"output_tokens_cost_in_usd": value})
    restored = Usage.model_validate_json(usage.model_dump_json())
    if math.isnan(value):
        assert math.isnan(restored.output_tokens_cost_in_usd)
    else:
        assert restored.output_tokens_cost_in_usd == value


def test_request_rejects_first_item_after_its_end() -> None:
    """Reject first_item_after_seconds exceeding elapsed_seconds."""
    with pytest.raises(ValidationError):
        _ = SettledRequestRecord(
            started_after_seconds=0.0,
            elapsed_seconds=1.0,
            first_item_after_seconds=2.0,
            error=None,
            billing=_BILLING,
            assistant_message=_TEXT_ASSISTANT_MESSAGE,
            model_served=None,
            response_id=None,
            request_id=None,
        )


def test_less_than_or_ulp_close_accepts_the_documented_rounding_boundary() -> None:
    """The timing comparison accepts four ULPs and rejects five ULPs."""
    right = 1.0
    assert _less_than_or_ulp_close(right + 4 * math.ulp(right), right)
    assert not _less_than_or_ulp_close(right + 5 * math.ulp(right), right)


def test_request_history_rejects_overlap_out_of_bounds_and_cut_off_placement() -> None:
    """Request history validation enforces ordering, bounds, and final cut-off placement."""
    first = _settled(elapsed_seconds=1.0)
    overlap = _settled(started_after_seconds=0.5, elapsed_seconds=0.5)
    with pytest.raises(ValidationError, match="overlap"):
        _ = RequestHistory(
            model="m",
            provider_name="p",
            records=(first, overlap),
            elapsed_seconds=1.0,
        )
    with pytest.raises(ValidationError, match="within the request history"):
        _ = RequestHistory(
            model="m",
            provider_name="p",
            records=(first,),
            elapsed_seconds=0.5,
        )
    with pytest.raises(ValidationError, match="must be final"):
        _ = RequestHistory(
            model="m",
            provider_name="p",
            records=(
                CutOffRequestRecord(started_after_seconds=0.0, billing=None),
                _settled(started_after_seconds=1.0),
            ),
            elapsed_seconds=2.0,
        )


def test_response_record_round_trips_with_concrete_output_type() -> None:
    """A concrete caller model reconstructs through `PlainGenerationRecord` JSON."""
    record = PlainGenerationRecord(
        output=Report(value=3),
        request_history=_completed_request_history(),
        stop_reason="stop",
    )
    response_json = record.model_dump_json()
    restored = PlainGenerationRecord[Report].model_validate_json(response_json)
    assert restored.model_dump_json() == response_json
    assert isinstance(restored.output, Report)


def test_generation_record_rejects_invalid_request_shapes() -> None:
    """Reject generation records with missing billing or a cut-off request."""
    with pytest.raises(ValidationError, match="final request must contain billing"):
        _ = PlainGenerationRecord(
            output=1,
            request_history=_request_history(_settled(billing=None)),
            stop_reason="stop",
        )
    with pytest.raises(ValidationError, match="cut-off"):
        _ = PlainGenerationRecord(
            output=1,
            request_history=_request_history(
                CutOffRequestRecord(started_after_seconds=0.0, billing=None)
            ),
            stop_reason="stop",
        )


def test_tool_call_generation_record_requires_a_tool_call() -> None:
    """A `ToolCallGenerationRecord` requires a tool call in its kept assistant message."""
    with pytest.raises(ValidationError, match="tool call"):
        _ = ToolCallGenerationRecord(
            output=None, request_history=_completed_request_history(), stop_reason="tool_call"
        )
    record = ToolCallGenerationRecord(
        output=Report(value=4),
        request_history=_completed_request_history(assistant_message=_TOOL_CALL_ASSISTANT_MESSAGE),
        stop_reason="tool_call",
    )
    restored = ToolCallGenerationRecord[Report].model_validate_json(record.model_dump_json())
    assert restored.tool_calls == _TOOL_CALL_ASSISTANT_MESSAGE.tool_calls
    assert isinstance(restored.output, Report)


def test_live_generation_preserves_provider_data_and_requires_alignment() -> None:
    """Preserve raw provider data and reject misaligned request_provider_data."""
    record = PlainGenerationRecord(
        output=Report(value=5),
        request_history=_completed_request_history(),
        stop_reason="stop",
    )
    request_provider_data = _request_provider_data()
    response = PlainGeneration(record=record, request_provider_data=request_provider_data)
    assert response.raw is request_provider_data[0].raw
    with pytest.raises(ValueError, match="align"):
        _ = PlainGeneration(record=record, request_provider_data=())


def test_live_generation_requires_provider_data_from_the_final_request() -> None:
    """A live generation rejects an earlier provider response when its final request has none."""
    failed_request = _settled(
        elapsed_seconds=0.5,
        error=TransientErrorRecord(error_text="retry"),
        billing=None,
        assistant_message=None,
    )
    successful_request = _settled(started_after_seconds=0.5, elapsed_seconds=0.5)
    record = PlainGenerationRecord(
        output=Report(value=5),
        request_history=_request_history(failed_request, successful_request),
        stop_reason="stop",
    )
    request_provider_data = (
        RequestProviderData(raw=StubRaw(), usage_raw=None),
        RequestProviderData(raw=None, usage_raw=None),
    )
    with pytest.raises(ValueError, match="final provider response"):
        _ = PlainGeneration(record=record, request_provider_data=request_provider_data)


def test_generation_variant_constructs_one_normalized_record() -> None:
    """`_generation_variant` stores one normalized tool-call record by reference."""
    request_history = _completed_request_history(assistant_message=_TOOL_CALL_ASSISTANT_MESSAGE)
    generation = _generation_variant(
        splits_on_tool_calls=True,
        output=Report(value=6),
        request_history=request_history,
        request_provider_data=_request_provider_data(),
        stop_reason="tool_call",
    )
    assert generation.kind == "tool_call"
    assert generation.record.kind == "tool_call"
    assert generation.record.request_history is request_history


def _error_record_cases() -> list[tuple[GenerationErrorRecord, GenerationErrorKind, str]]:
    completed = _completed_request_history()
    failed = _failed_request_history()
    multiline_failed = _request_history(
        _settled(
            elapsed_seconds=0.5,
            error=TransientErrorRecord(error_text="connection\nreset"),
            billing=None,
            assistant_message=None,
        ),
        _settled(
            started_after_seconds=0.5,
            elapsed_seconds=0.5,
            error=TransientErrorRecord(error_text=""),
            billing=None,
            assistant_message=None,
        ),
    )
    terminal = _request_history(_settled(billing=None, assistant_message=None))
    cut_off = _request_history(CutOffRequestRecord(started_after_seconds=0.0, billing=_BILLING))
    return [
        (
            PlainErrorRecord(kind="retries_exhausted_error", request_history=multiline_failed),
            "retries_exhausted_error",
            "request 1: connection\n  reset\nrequest 2: ",
        ),
        (
            PlainErrorRecord(kind="retry_unavailable_error", request_history=failed),
            "retry_unavailable_error",
            "retry",
        ),
        (
            PlainErrorRecord(kind="refusal_error", request_history=completed),
            "refusal_error",
            "",
        ),
        (
            PlainErrorRecord(
                kind="max_completion_tokens_exceeded_error", request_history=completed
            ),
            "max_completion_tokens_exceeded_error",
            "",
        ),
        (
            PlainErrorRecord(kind="empty_assistant_message_error", request_history=completed),
            "empty_assistant_message_error",
            "",
        ),
        (
            SchemaViolationErrorRecord(request_history=completed, validation_error_json="[]"),
            "schema_violation_error",
            "",
        ),
        (
            PlainErrorRecord(kind="context_window_exceeded_error", request_history=completed),
            "context_window_exceeded_error",
            "",
        ),
        (
            PlainErrorRecord(
                kind="unfinished_assistant_message_error",
                request_history=completed,
                error_text="unfinished",
            ),
            "unfinished_assistant_message_error",
            "unfinished",
        ),
        (
            PlainErrorRecord(
                kind="provider_failed_terminally_error",
                request_history=completed,
                error_text="failed",
            ),
            "provider_failed_terminally_error",
            "failed",
        ),
        (
            PlainErrorRecord(
                kind="provider_failed_terminally_error",
                request_history=terminal,
                error_text="terminal",
            ),
            "provider_failed_terminally_error",
            "terminal",
        ),
        (
            PlainErrorRecord(kind="auth_error", request_history=terminal, error_text="auth"),
            "auth_error",
            "auth",
        ),
        (
            PlainErrorRecord(
                kind="rejected_error",
                request_history=RequestHistory(
                    model="model",
                    provider_name="provider",
                    records=(),
                    elapsed_seconds=0.0,
                ),
                error_text="invalid",
            ),
            "rejected_error",
            "invalid",
        ),
        (
            PlainErrorRecord(
                kind="unknown_exception_error",
                request_history=RequestHistory(
                    model="model",
                    provider_name="provider",
                    records=(),
                    elapsed_seconds=0.0,
                ),
                error_text="unknown",
            ),
            "unknown_exception_error",
            "unknown",
        ),
        (
            PlainErrorRecord(
                kind="unknown_exception_error", request_history=cut_off, error_text="escaped"
            ),
            "unknown_exception_error",
            "escaped",
        ),
        (
            PlainErrorRecord(kind="timed_out_error", request_history=cut_off),
            "timed_out_error",
            "",
        ),
    ]


def test_every_error_record_round_trips_through_the_closed_union() -> None:
    """The closed error union reconstructs each built-in error record."""
    adapter = TypeAdapter(GenerationErrorRecord)
    for record, expected_kind, expected_error_text in _error_record_cases():
        assert record.error_text == expected_error_text
        record_json_text = record.model_dump_json()
        assert json.loads(record_json_text)["error_text"] == expected_error_text
        record_json_bytes = adapter.dump_json(record)
        restored = adapter.validate_json(record_json_bytes)
        assert type(restored) is type(record)
        assert adapter.dump_json(restored) == record_json_bytes
        assert restored.kind == expected_kind
        assert restored.error_text == expected_error_text
        assert str(restored) == restored.error_text


def test_error_record_properties_and_error_text() -> None:
    """Error records retain normalized properties and error_text."""
    exhausted = PlainErrorRecord(
        kind="retries_exhausted_error", request_history=_failed_request_history()
    )
    assert [str(error) for error in exhausted.request_history.errors_from_requests] == ["retry"]
    assert exhausted.error_text == "request 1: retry"
    assert exhausted.assistant_message is None
    refusal = PlainErrorRecord(kind="refusal_error", request_history=_completed_request_history())
    assert refusal.stop_reason == "refusal"
    assert refusal.assistant_message == _TEXT_ASSISTANT_MESSAGE
    assert refusal.usage == _USAGE


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PlainErrorRecord(
            kind="retries_exhausted_error",
            request_history=_failed_request_history(),
            error_text="different text",
        ),
        lambda: PlainErrorRecord(
            kind="retry_unavailable_error",
            request_history=_failed_request_history(),
            error_text="different text",
        ),
    ],
)
def test_retry_error_records_reject_error_text_that_disagrees_with_request_history(
    factory: Callable[[], PlainErrorRecord],
) -> None:
    """Retry records derive error_text from the request history."""
    with pytest.raises(ValidationError, match="error_text must match"):
        _ = factory()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: PlainErrorRecord(
                kind="retries_exhausted_error", request_history=_completed_request_history()
            ),
            "transient error",
        ),
        (
            lambda: PlainErrorRecord(
                kind="refusal_error", request_history=_failed_request_history()
            ),
            "final request must be error-free",
        ),
        (
            lambda: PlainErrorRecord(
                kind="auth_error",
                request_history=_request_history(_settled(billing=None, assistant_message=None)),
            ),
            "auth_error requires error_text",
        ),
        (
            lambda: PlainErrorRecord(
                kind="provider_failed_terminally_error",
                request_history=RequestHistory(
                    model="m", provider_name="p", records=(), elapsed_seconds=0.0
                ),
                error_text="terminal",
            ),
            "must contain a final response",
        ),
    ],
)
def test_error_records_reject_invalid_request_history_shapes(
    factory: Callable[[], PlainErrorRecord], match: str
) -> None:
    """Each constrained error group rejects a request history shape outside its contract."""
    with pytest.raises(ValidationError, match=match):
        _ = factory()


def test_generation_error_requires_provider_request_alignment() -> None:
    """Reject request_provider_data that do not align with the record."""
    record = PlainErrorRecord(kind="refusal_error", request_history=_completed_request_history())
    with pytest.raises(ValueError, match="align"):
        _ = GenerationError(record=record, request_params=None, request_provider_data=())


def test_mixed_normalized_outcome_list_round_trips() -> None:
    """A mixed normalized outcome list reconstructs through its concrete output type."""
    records: list[GenerationOutcomeRecord[Report, Report | None]] = [
        PlainGenerationRecord(
            output=Report(value=9),
            request_history=_completed_request_history(),
            stop_reason="stop",
        ),
        PlainErrorRecord(kind="refusal_error", request_history=_completed_request_history()),
    ]
    adapter = TypeAdapter(list[GenerationOutcomeRecord[Report, Report | None]])
    records_json = adapter.dump_json(records)
    restored = adapter.validate_json(records_json)
    assert adapter.dump_json(restored) == records_json
    assert to_tables(restored).outcomes[0]["output"] == '{"value":9}'


def test_to_tables_reads_live_only_request_params_and_provider_usage() -> None:
    """Tables read request params and provider usage only from live outcomes."""
    record = PlainErrorRecord(kind="refusal_error", request_history=_completed_request_history())
    live = GenerationError(
        record=record,
        request_params=StubRequestParams(),
        request_provider_data=_request_provider_data(),
    )
    live_tables = to_tables(live)
    normalized_tables = to_tables(record)
    assert live_tables.outcomes[0]["request_params_json"] == '{"prompt":"hi"}'
    assert normalized_tables.outcomes[0]["request_params_json"] is None
    assert live_tables.outcomes[0]["error_text"] == ""
    assert live_tables.requests[0]["usage_raw_json"] == '{"billed_units":17}'
    assert normalized_tables.requests[0]["usage_raw_json"] is None
    assert live_tables.requests[0]["started_after_seconds"] == 0.0
    response = PlainGeneration(
        record=PlainGenerationRecord(
            output=Report(value=5),
            request_history=_completed_request_history(),
            stop_reason="stop",
        ),
        request_provider_data=_request_provider_data(),
    )
    assert to_tables(response).requests[0]["usage_raw_json"] == '{"billed_units":17}'


def test_to_tables_emits_one_row_for_a_cut_off_request() -> None:
    """A cut-off request produces one request row with no fabricated ending."""
    record = PlainErrorRecord(
        kind="timed_out_error",
        request_history=_request_history(
            CutOffRequestRecord(started_after_seconds=0.25, billing=_BILLING)
        ),
    )
    tables = to_tables(record)
    assert tables.outcomes[0]["request_count"] == 1
    assert len(tables.requests) == 1
    assert tables.requests[0]["started_after_seconds"] == 0.25
    assert tables.requests[0]["elapsed_seconds"] is None
    assert tables.requests[0]["first_item_after_seconds"] is None
    assert tables.requests[0]["cost_in_usd"] == _USAGE.cost_in_usd


def test_to_tables_writes_an_abandoned_stream_without_output_error_text_or_kept_request() -> None:
    """An abandoned stream row has neither output nor error_text, and no kept request."""
    tables = to_tables(AbandonedStreamRecord(request_history=_failed_request_history()))
    assert (tables.outcomes[0]["output"], tables.outcomes[0]["error_text"]) == (None, None)
    assert [request_record["kept"] for request_record in tables.requests] == [False]


class _TickingClock:
    """A `time` stand-in whose `monotonic` returns 0.0, 1.0, 2.0, ... on successive calls."""

    def __init__(self) -> None:
        self._ticks = 0

    def monotonic(self) -> float:
        """Return the number of earlier calls as seconds."""
        seconds = float(self._ticks)
        self._ticks += 1
        return seconds


def test_the_ledger_stamps_the_first_item_and_not_a_later_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`first_item_after_seconds` measures the first `stamp_first_item`, and later stamps leave it unchanged.

    The clock ticks on every read: 0.0 builds the ledger, 1.0 starts the request, 2.0 stamps the first item.
    A second stamp reads no time, so the record ends at 3.0.
    """
    monkeypatch.setattr("langchaint.generation.request_history.time", _TickingClock())
    ledger = _RequestLedger(model="model", provider_name="provider")
    ledger.start_request()
    ledger.stamp_first_item()
    ledger.stamp_first_item()
    ledger.record(error=None, assistant_message=None)
    (record,) = ledger.freeze().records
    assert record.kind == "settled"
    assert record.first_item_after_seconds == 1.0
    assert record.elapsed_seconds == 2.0


def test_timed_out_error_appends_one_cut_off_request_with_live_usage() -> None:
    """A live timeout aligns its cut-off record with provider usage."""
    ledger = _RequestLedger(model="model", provider_name="provider")
    ledger.start_request()
    provider_billing = ProviderBilling(billing=_BILLING, usage_raw=ProviderUsage(billed_units=23))
    failure = _timed_out_error(ledger, provider_billing)
    assert failure.record.kind == "timed_out_error"
    assert len(failure.request_history.records) == 1
    assert failure.request_history.records[0].kind == "cut_off"
    assert failure.request_provider_data[0].usage_raw == provider_billing.usage_raw
    assert failure.usage == _USAGE


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AbandonedStreamRecord(request_history=_failed_request_history()),
        lambda: PlainErrorRecord(
            kind="timed_out_error", request_history=_failed_request_history()
        ),
    ],
)
def test_an_interrupted_outcome_record_accepts_a_transient_prefix_without_a_cut_off(
    factory: Callable[[], AbandonedStreamRecord | PlainErrorRecord],
) -> None:
    """An interruption during retry backoff retains its transient settled request."""
    record = factory()
    assert record.request_history == _failed_request_history()


def test_interruption_after_a_staged_response_records_no_cut_off_request() -> None:
    """A staged provider response settles before an interrupted input's request history freezes."""
    ledger = _RequestLedger(model="model", provider_name="provider")
    ledger.start_request()
    raw = StubRaw()
    provider_billing = ProviderBilling(billing=_BILLING, usage_raw=None)
    ledger.stage_response(
        raw=raw,
        provider_billing=provider_billing,
        identity=ResponseIdentity(
            model_served="served", response_id="response", request_id="request"
        ),
    )
    failure = _timed_out_error(ledger)
    assert len(failure.request_history.records) == 1
    assert failure.request_history.records[0].kind == "settled"
    assert failure.request_provider_data[0].raw is raw
