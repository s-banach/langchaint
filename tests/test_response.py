"""Response constructor invariants, the two usage scopes, and to_tables over both result types.

The retry loops in llm.py and streaming.py are the only production constructors of Response,
so these invariants are what stops a refactor of either loop from building a success row whose records disagree;
the retry tests in test_bound_llm.py pin the record values themselves.
usage is the paid total folded from attempt_records and usage_successful_attempt is the last record's own,
so a retried billed 200 makes them diverge; both are exercised here.
to_tables is the table boundary, exercised here over a success and over each GenerationError arm,
because what each arm puts in the shared columns (output, error_summary, stop_reason, and the
per-attempt billing columns) differs per arm and is what a mixed batch's tables show.
"""

import json
import math
from dataclasses import dataclass
from typing import override

import pytest
from pydantic import BaseModel

from langchaint import (
    ZERO_USAGE,
    AbandonedCallError,
    AssistantMessage,
    AttemptRecord,
    Billing,
    CallRecord,
    MaxCompletionTokensExceededError,
    RefusalError,
    Response,
    RetriesExhaustedError,
    StopReason,
    TextPart,
    TransientError,
    Usage,
    to_tables,
)
from langchaint.adapter import RequestParams
from langchaint.call import ResponseIdentity, _CallLedger
from langchaint.response import _abandoned_call_error
from tests.helpers import stated_billing


class _Raw(BaseModel):
    """Stand-in for the SDK's own response model held on Response.raw."""


@dataclass(frozen=True, kw_only=True)
class _Request(RequestParams):
    """Stand-in for the request an adapter builds, held on a GenerationError."""

    prompt: str

    @override
    def as_json(self) -> str:
        """Render the one field as a JSON object, as a real adapter renders its own."""
        return json.dumps({"prompt": self.prompt})


_REQUEST = _Request(prompt="the-prompt-text")


class _ProviderUsage(BaseModel):
    """Stand-in for the SDK's own usage object, holding detail the neutral Usage has no field for."""

    cache_creation_5m_input_tokens: int
    server_tool_use_requests: int


_USAGE = Usage(
    input_tokens_cache_read=2,
    input_tokens_cache_write=3,
    input_tokens_cache_none=5,
    output_tokens=7,
    output_tokens_reasoning=2,
    input_tokens_cache_read_cost_in_usd=0.1,
    input_tokens_cache_write_cost_in_usd=0.1,
    input_tokens_cache_none_cost_in_usd=0.1,
    output_tokens_cost_in_usd=0.2,
)
"""One attempt's billing: four category costs summing to 0.5."""


_CALL_STARTED_AT = 1000.0
"""The fixed time.monotonic() origin every record here is placed against."""


def _record(
    *,
    error: TransientError | None,
    usage: Usage = ZERO_USAGE,
    reported_billing: bool = True,
    input_cache_none_usd_per_million_tokens: float = math.nan,
    usage_raw: BaseModel | None = None,
    started_after_seconds: float = 0.0,
    elapsed_seconds: float = 0.0,
    seconds_to_first_item: float | None = None,
    turn: AssistantMessage | None = None,
    model_served: str | None = None,
    response_id: str | None = None,
    request_id: str | None = None,
) -> AttemptRecord:
    """Build one record on the fixed origin; reported_billing False is an attempt the provider never billed."""
    started_at_monotonic_seconds = _CALL_STARTED_AT + started_after_seconds
    return AttemptRecord(
        started_at_monotonic_seconds=started_at_monotonic_seconds,
        ended_at_monotonic_seconds=started_at_monotonic_seconds + elapsed_seconds,
        first_item_at_monotonic_seconds=(
            None
            if seconds_to_first_item is None
            else started_at_monotonic_seconds + seconds_to_first_item
        ),
        error=error,
        billing=(
            stated_billing(
                usage,
                input_cache_none_usd_per_million_tokens=input_cache_none_usd_per_million_tokens,
                usage_raw=usage_raw,
            )
            if reported_billing
            else None
        ),
        assistant_message=turn,
        raw=None,
        model_served=model_served,
        response_id=response_id,
        request_id=request_id,
    )


def _call(attempt_records: tuple[AttemptRecord, ...], *, elapsed_seconds: float) -> CallRecord:
    """Build a CallRecord over the records under test; the identity fields are fixed filler."""
    return CallRecord(
        model="fake-model",
        provider_name="fake",
        attempt_records=attempt_records,
        started_at_monotonic_seconds=_CALL_STARTED_AT,
        elapsed_seconds=elapsed_seconds,
    )


def _response[OutputT](
    *,
    output: OutputT,
    attempt_records: tuple[AttemptRecord, ...],
    stop_reason: StopReason = "end_turn",
) -> Response[OutputT]:
    """Build a Response with the fields under test; everything else is fixed filler."""
    return Response(
        output=output,
        call=_call(attempt_records, elapsed_seconds=1.5),
        raw=_Raw(),
        stop_reason=stop_reason,
        assistant_message=AssistantMessage(turn=(TextPart(text=str(output)),)),
    )


def _failure(*, attempt_records: tuple[AttemptRecord, ...]) -> RetriesExhaustedError:
    """Build a RetriesExhaustedError with the table fields set."""
    return RetriesExhaustedError(
        call=_call(attempt_records, elapsed_seconds=2.5), request=_REQUEST
    )


def test_response_rejects_empty_attempt_records() -> None:
    """A Response without a single record has no history and is rejected."""
    with pytest.raises(ValueError, match="at least one record"):
        _ = _response(output="ok", attempt_records=())


def test_response_rejects_an_error_free_record_before_the_last() -> None:
    """A success record can only be last: the loop stops on the attempt that succeeded."""
    with pytest.raises(ValueError, match="only the last"):
        _ = _response(
            output="ok",
            attempt_records=(_record(error=None), _record(error=TransientError("e"))),
        )


def test_response_rejects_a_failed_last_record() -> None:
    """A Response is a success, so its final record must be the one that succeeded."""
    with pytest.raises(ValueError, match="must be error-free"):
        _ = _response(output="ok", attempt_records=(_record(error=TransientError("e")),))


def test_retries_exhausted_error_derives_from_its_records() -> None:
    """errors_from_attempts, attempts, and error_text are folds over the records, not stored copies."""
    failure = _failure(
        attempt_records=(
            _record(error=TransientError("e1")),
            _record(error=TransientError("e2")),
        )
    )
    assert failure.attempts == 2
    assert [str(error) for error in failure.errors_from_attempts] == ["e1", "e2"]
    assert failure.error_text == "attempt 1: e1; attempt 2: e2"


def test_usage_successful_attempt_is_the_last_record() -> None:
    """usage_successful_attempt reads the single kept answer's own usage."""
    response = _response(
        output="ok",
        attempt_records=(
            _record(error=TransientError("e"), usage=ZERO_USAGE),
            _record(error=None, usage=_USAGE),
        ),
    )
    assert response.usage_successful_attempt is response.attempt_records[-1].usage
    assert response.usage_successful_attempt == _USAGE


@pytest.mark.parametrize(
    ("failed_attempt_usage", "expected_cost_in_usd", "expected_output_tokens"),
    [(_USAGE, 1.0, 14), (ZERO_USAGE, 0.5, 7)],
    ids=["failed_attempt_billed", "failed_attempt_billed_nothing"],
)
def test_usage_is_the_paid_total_across_attempts(
    failed_attempt_usage: Usage, expected_cost_in_usd: float, expected_output_tokens: int
) -> None:
    """Usage folds every attempt's billing; usage_successful_attempt stays the kept answer's own.

    A retried billed 200 makes the two diverge, which is what says the fold is over the records
    rather than a copy of the last one.
    """
    response = _response(
        output="ok",
        attempt_records=(
            _record(error=TransientError("empty parse"), usage=failed_attempt_usage),
            _record(error=None, usage=_USAGE),
        ),
    )
    assert response.usage.cost_in_usd == pytest.approx(expected_cost_in_usd)
    assert response.usage.output_tokens == expected_output_tokens
    assert response.usage_successful_attempt.cost_in_usd == pytest.approx(0.5)


def test_to_tables_success_writes_one_call_row_and_one_attempt_row() -> None:
    """A success names its output and reason on the call row and its billing on the attempt row.

    usage_raw_json is None on a billed attempt whose provider sent no usage object, which is what says
    the column tracks the provider's report rather than whether the attempt billed.
    """
    turn = AssistantMessage(turn=(TextPart(text="hello"),))
    calls, attempts = to_tables(
        _response(output="hello", attempt_records=(_record(error=None, usage=_USAGE, turn=turn),))
    )
    (call_row,) = calls
    assert call_row == {
        "call_id": 0,
        "model": "fake-model",
        "provider_name": "fake",
        "elapsed_seconds": 1.5,
        "attempts": 1,
        "stop_reason": "end_turn",
        "error_summary": None,
        "request_json": None,
        "output": "hello",
    }
    (attempt_row,) = attempts
    assert attempt_row["call_id"] == 0
    assert attempt_row["attempt_index"] == 0
    assert attempt_row["kept"] is True
    assert attempt_row["service_tier"] == "stub"
    assert attempt_row["error_text"] is None
    assert attempt_row["assistant_message_json"] == turn.model_dump_json()
    assert attempt_row["usage_raw_json"] is None
    assert attempt_row["cost_in_usd"] == pytest.approx(0.5)
    assert attempt_row["input_tokens_cache_none"] == 5
    assert attempt_row["input_tokens_cache_none_cost_in_usd"] == 0.1
    assert attempt_row["input_tokens_total"] == 10
    assert attempt_row["output_tokens"] == 7
    assert attempt_row["output_tokens_cost_in_usd"] == 0.2
    assert attempt_row["output_tokens_reasoning"] == 2


def test_to_tables_bills_each_attempt_in_its_own_row() -> None:
    """A retried billed 200 gets a row each, so the call's spend is the caller's sum over them.

    kept marks only the attempt whose turn became the answer, which is what makes the kept row's
    tokens the single answer's own and the sum over both rows the paid total. A single cost column
    on a call row could state one or the other and not both.
    """
    _, attempts = to_tables(
        _response(
            output="ok",
            attempt_records=(
                _record(error=TransientError("empty parse"), usage=_USAGE),
                _record(error=None, usage=_USAGE),
            ),
        )
    )
    assert [row["attempt_index"] for row in attempts] == [0, 1]
    assert [row["kept"] for row in attempts] == [False, True]
    assert [row["error_text"] for row in attempts] == ["empty parse", None]
    assert [row["cost_in_usd"] for row in attempts] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert [row["output_tokens"] for row in attempts] == [7, 7]


def test_to_tables_carries_an_unpriced_cost_as_nan() -> None:
    """A response no rate table could price reaches its row as NaN, output intact."""
    unpriced = Usage(
        input_tokens_cache_read=2,
        input_tokens_cache_write=3,
        input_tokens_cache_none=5,
        output_tokens=7,
        output_tokens_reasoning=2,
        input_tokens_cache_read_cost_in_usd=float("nan"),
        input_tokens_cache_write_cost_in_usd=float("nan"),
        input_tokens_cache_none_cost_in_usd=float("nan"),
        output_tokens_cost_in_usd=float("nan"),
    )
    calls, attempts = to_tables(
        _response(output="hello", attempt_records=(_record(error=None, usage=unpriced),))
    )
    assert calls[0]["output"] == "hello"
    assert isinstance(attempts[0]["cost_in_usd"], float)
    assert math.isnan(attempts[0]["cost_in_usd"])
    assert isinstance(attempts[0]["output_tokens_cost_in_usd"], float)
    assert math.isnan(attempts[0]["output_tokens_cost_in_usd"])
    assert attempts[0]["output_tokens"] == 7


def test_to_tables_nulls_every_billing_column_where_the_provider_reported_nothing() -> None:
    """An attempt with no billing writes None, not zero: nothing reported is not a zero bill.

    Zeros would add an attempt whose spend is unknown into a sum as if it had been free.
    """
    _, attempts = to_tables(
        _failure(
            attempt_records=(_record(error=TransientError("no response"), reported_billing=False),)
        )
    )
    (attempt_row,) = attempts
    assert attempt_row["service_tier"] is None
    assert attempt_row["cost_in_usd"] is None
    assert attempt_row["input_tokens_total"] is None
    assert attempt_row["output_tokens"] is None
    assert attempt_row["output_usd_per_million_tokens"] is None
    assert attempt_row["error_text"] == "no response"


def test_to_tables_pins_the_prices_that_applied_beside_the_counters() -> None:
    """The attempt row carries its own rates, so counter times rate reproduces the stored cost."""
    _, attempts = to_tables(
        _response(
            output="ok",
            attempt_records=(
                _record(
                    error=None,
                    usage=_USAGE,
                    input_cache_none_usd_per_million_tokens=20.0,
                ),
            ),
        )
    )
    assert attempts[0]["input_cache_none_usd_per_million_tokens"] == 20.0
    assert isinstance(attempts[0]["output_usd_per_million_tokens"], float)
    assert math.isnan(attempts[0]["output_usd_per_million_tokens"])


def test_to_tables_dumps_the_provider_usage_object_the_neutral_counters_cannot_hold() -> None:
    """usage_raw_json carries what Usage merges or drops, so the archive keeps it past the process.

    A caller reaches the cache-write TTL split and the server-tool counters with their query
    engine's JSON functions; no column of langchaint's own exists for either.
    """
    _, attempts = to_tables(
        _response(
            output="ok",
            attempt_records=(
                _record(
                    error=None,
                    usage=_USAGE,
                    usage_raw=_ProviderUsage(
                        cache_creation_5m_input_tokens=3, server_tool_use_requests=1
                    ),
                ),
            ),
        )
    )
    assert json.loads(str(attempts[0]["usage_raw_json"])) == {
        "cache_creation_5m_input_tokens": 3,
        "server_tool_use_requests": 1,
    }


def test_to_tables_places_each_attempt_on_the_calls_timeline() -> None:
    """started_after_seconds is measured from the call's start, so the first wait is visible.

    The gap between one row's start plus its elapsed_seconds and the next row's start is the
    RateLimiter wait or backoff sleep between the two, which no attempt bracket covers.
    """
    _, attempts = to_tables(
        _failure(
            attempt_records=(
                _record(
                    error=TransientError("e1"), started_after_seconds=3.0, elapsed_seconds=0.5
                ),
                _record(
                    error=TransientError("e2"), started_after_seconds=9.0, elapsed_seconds=0.25
                ),
            )
        )
    )
    assert [row["started_after_seconds"] for row in attempts] == [3.0, 9.0]
    assert [row["elapsed_seconds"] for row in attempts] == [0.5, 0.25]


def test_to_tables_measures_time_to_first_item_from_the_attempts_own_start() -> None:
    """The column is the gap from this attempt's start, not from the call's, and null with no stamp.

    Reading it against the call's start would charge a reopened stream with every earlier attempt
    and the waits between them.
    """
    _, attempts = to_tables(
        _failure(
            attempt_records=(
                _record(
                    error=TransientError("e1"), started_after_seconds=3.0, elapsed_seconds=0.5
                ),
                _record(
                    error=TransientError("e2"),
                    started_after_seconds=9.0,
                    elapsed_seconds=2.0,
                    seconds_to_first_item=0.75,
                ),
            )
        )
    )
    assert [row["seconds_to_first_item"] for row in attempts] == [None, 0.75]


def test_to_tables_writes_the_ids_and_served_model_each_attempt_carries() -> None:
    """Every column is the record's own value.

    model_served and response_id are null on the attempt that received no response; request_id is
    not, being the one the error channel fills.
    """
    _, attempts = to_tables(
        _failure(
            attempt_records=(
                _record(error=TransientError("e1"), request_id="req_1"),
                _record(
                    error=TransientError("e2"),
                    model_served="fake-model-2026-01-01",
                    response_id="msg_9",
                    request_id="req_2",
                ),
            )
        )
    )
    assert [row["model_served"] for row in attempts] == [None, "fake-model-2026-01-01"]
    assert [row["response_id"] for row in attempts] == [None, "msg_9"]
    assert [row["request_id"] for row in attempts] == ["req_1", "req_2"]


def test_to_tables_structured_output_becomes_json() -> None:
    """A pydantic output instance is flattened to its JSON, not its repr."""
    calls, _ = to_tables(_response(output=_USAGE, attempt_records=(_record(error=None),)))
    assert calls[0]["output"] == _USAGE.model_dump_json()


def test_to_tables_writes_a_tool_call_success_as_a_null_output_with_no_error_summary() -> None:
    """A structured tool-bound binding's tool-call turn is a success whose output cell is None.

    str(None) would write the string "None" into a column readers scan for real output, and the
    error_summary and stop_reason cells are what tell this row from the failure row beside it.
    """
    calls, _ = to_tables(
        _response(output=None, attempt_records=(_record(error=None),), stop_reason="tool_use")
    )
    assert calls[0]["output"] is None
    assert calls[0]["error_summary"] is None
    assert calls[0]["stop_reason"] == "tool_use"


def test_to_tables_failure_summarizes_the_call_and_rows_each_attempts_own_error() -> None:
    """error_summary states how the call ended; each attempt's own error is its row's error_text.

    The two differ on this class: error_text folds every attempt's error into one string, and
    error_summary states the outcome without restating what the attempt rows already carry.
    """
    calls, attempts = to_tables(
        _failure(
            attempt_records=(
                _record(error=TransientError("e1")),
                _record(error=TransientError("e2")),
            )
        )
    )
    assert calls[0]["output"] is None
    assert calls[0]["stop_reason"] is None
    assert calls[0]["attempts"] == 2
    assert calls[0]["error_summary"] == "2 attempts failed; last: e2"
    assert [row["error_text"] for row in attempts] == ["e1", "e2"]
    assert all(row["kept"] is False for row in attempts)


def test_to_tables_writes_the_request_a_failed_call_sent_and_nothing_else() -> None:
    """A failure's row carries what every attempt of the call sent, rendered by the adapter.

    The prompt reaches the table through this column alone: error_summary is the failure's own text,
    so a caller who drops request_json is left with no message content in the calls table.
    """
    calls, _attempts = to_tables(_failure(attempt_records=(_record(error=TransientError("e1")),)))
    assert calls[0]["request_json"] == json.dumps({"prompt": "the-prompt-text"})
    assert "the-prompt-text" not in str(calls[0]["error_summary"])


def test_to_tables_writes_no_request_where_the_call_built_none() -> None:
    """A failure the adapter declared before building a request writes a null cell, not an empty one."""
    calls, _attempts = to_tables(
        RefusalError(
            call=_call((_record(error=None, usage=_USAGE),), elapsed_seconds=1.0), request=None
        )
    )
    assert calls[0]["request_json"] is None


def test_a_failures_text_carries_no_part_of_the_request() -> None:
    """error_text and __str__ stay free of the prompt, which the tracing layer writes unconditionally.

    A caller who set capture_message_content False would otherwise find the GenerationInput in a span
    anyway, through the error's own text.
    """
    failure = _failure(attempt_records=(_record(error=TransientError("e1")),))
    assert failure.request == _REQUEST
    assert "the-prompt-text" not in failure.error_text
    assert "the-prompt-text" not in str(failure)


@pytest.mark.parametrize(
    ("error_class", "expected_stop_reason", "expected_error_summary"),
    [
        (
            RefusalError,
            "refusal",
            "no structured output: the model refused or a provider filter blocked the turn",
        ),
        (
            MaxCompletionTokensExceededError,
            "max_tokens",
            "the structured response reached max_completion_tokens before its JSON parsed",
        ),
    ],
    ids=["refusal", "truncation"],
)
def test_to_tables_a_failed_200_reports_its_billing_and_reason(
    error_class: type[RefusalError | MaxCompletionTokensExceededError],
    expected_stop_reason: StopReason,
    expected_error_summary: str,
) -> None:
    """A 200 that produced no output still carries its cost and usage, plus the call's reason.

    Its attempt has no error of its own: the failure is the call's, so error_summary carries it and
    the attempt's error_text is None.
    """
    calls, attempts = to_tables(
        error_class(
            call=_call((_record(error=None, usage=_USAGE),), elapsed_seconds=1.0),
            request=_REQUEST,
        )
    )
    assert calls[0]["output"] is None
    assert calls[0]["stop_reason"] == expected_stop_reason
    assert calls[0]["error_summary"] == expected_error_summary
    assert calls[0]["attempts"] == 1
    assert attempts[0]["kept"] is False
    assert attempts[0]["error_text"] is None
    assert attempts[0]["cost_in_usd"] == pytest.approx(0.5)
    assert attempts[0]["input_tokens_total"] == 10
    assert attempts[0]["output_tokens"] == 7


def test_to_tables_numbers_a_mixed_batch_and_joins_its_attempts_back() -> None:
    """Every attempt row names the call it belongs to, which is what the two tables join on."""
    calls, attempts = to_tables([
        _response(output="ok", attempt_records=(_record(error=None, usage=_USAGE),)),
        _failure(
            attempt_records=(
                _record(error=TransientError("e1")),
                _record(error=TransientError("e2")),
            )
        ),
    ])
    assert [row["call_id"] for row in calls] == [0, 1]
    assert [row["output"] for row in calls] == ["ok", None]
    assert [row["call_id"] for row in attempts] == [0, 1, 1]
    assert [row["attempt_index"] for row in attempts] == [0, 0, 1]


def _abandoned(
    *,
    attempt_records: tuple[AttemptRecord, ...],
    billing_in_flight: Billing | None,
    in_flight_started_after_seconds: float | None,
) -> AbandonedCallError:
    """Build an AbandonedCallError placing its cut-off request on the fixed origin."""
    return AbandonedCallError(
        call=_call(attempt_records, elapsed_seconds=3.0),
        billing_in_flight=billing_in_flight,
        in_flight_attempt_started_at_monotonic_seconds=(
            None
            if in_flight_started_after_seconds is None
            else _CALL_STARTED_AT + in_flight_started_after_seconds
        ),
    )


def test_to_tables_rows_the_request_that_was_in_flight_when_the_call_was_cut_off() -> None:
    """The cut-off request gets an attempt row carrying what the provider had billed for it.

    Its ending was never observed, so every column describing how it ended is null, which is what
    distinguishes it from an attempt that failed. It sits after the settled records under the next
    attempt_index, so one SUM over cost_in_usd reaches the total the error reports.
    """
    calls, attempts = to_tables(
        _abandoned(
            attempt_records=(_record(error=TransientError("e1"), started_after_seconds=0.0),),
            billing_in_flight=stated_billing(_USAGE),
            in_flight_started_after_seconds=2.0,
        )
    )
    assert calls[0]["attempts"] == 2
    assert calls[0]["error_summary"] == "the call was cut off before its result reached the caller"
    assert [row["attempt_index"] for row in attempts] == [0, 1]
    cut_off = attempts[1]
    assert cut_off["started_after_seconds"] == pytest.approx(2.0)
    assert cut_off["kept"] is False
    assert cut_off["cost_in_usd"] == pytest.approx(0.5)
    assert cut_off["input_tokens_total"] == 10
    assert cut_off["elapsed_seconds"] is None
    assert cut_off["error_text"] is None
    assert cut_off["response_id"] is None
    assert sum(float(row["cost_in_usd"] or 0.0) for row in attempts) == pytest.approx(0.5)


def test_to_tables_nulls_the_billing_of_a_cut_off_request_the_provider_never_reported_on() -> None:
    """A non-stream call is cut off with nothing reported, and no spend is fabricated for it."""
    _calls, attempts = to_tables(
        _abandoned(
            attempt_records=(),
            billing_in_flight=None,
            in_flight_started_after_seconds=0.5,
        )
    )
    (cut_off,) = attempts
    assert cut_off["attempt_index"] == 0
    assert cut_off["cost_in_usd"] is None
    assert cut_off["input_tokens_total"] is None
    assert cut_off["usage_raw_json"] is None


def test_to_tables_writes_no_extra_row_for_a_call_cut_off_between_attempts() -> None:
    """No request was in flight, so the settled records are the whole attempts table for the call."""
    _calls, attempts = to_tables(
        _abandoned(
            attempt_records=(_record(error=TransientError("e1")),),
            billing_in_flight=None,
            in_flight_started_after_seconds=None,
        )
    )
    assert [row["error_text"] for row in attempts] == ["e1"]


def test_a_cut_off_call_counts_a_staged_response_once() -> None:
    """A response that arrived and was never read is one record, not also an attempt in flight.

    freeze closes it into an ordinary record carrying its own billing, so reporting the same request
    as in flight too would put it in the attempts table twice and add its billing to usage again.
    """
    ledger = _CallLedger(model="fake-model", provider_name="fake")
    ledger.start_call()
    ledger.start_attempt()
    ledger.stage_response(
        raw=_Raw(),
        billing=stated_billing(_USAGE),
        identity=ResponseIdentity(
            model_served="fake-model", response_id="resp-1", request_id="req-1"
        ),
    )
    # What an open stream reports of the attempt it is on, which this response has already billed.
    error = _abandoned_call_error(AbandonedCallError, ledger, stated_billing(_USAGE))
    assert error.in_flight_attempt_started_at_monotonic_seconds is None
    assert error.billing_in_flight is None
    assert error.attempts == 1
    assert error.usage.cost_in_usd == pytest.approx(0.5)
    _calls, attempts = to_tables(error)
    assert [row["response_id"] for row in attempts] == ["resp-1"]


def test_a_cut_off_call_reports_the_attempt_that_got_no_response() -> None:
    """An attempt open with nothing staged is the in-flight one, and takes the caller's billing report."""
    ledger = _CallLedger(model="fake-model", provider_name="fake")
    ledger.start_call()
    ledger.start_attempt()
    error = _abandoned_call_error(AbandonedCallError, ledger, stated_billing(_USAGE))
    assert error.in_flight_attempt_started_at_monotonic_seconds is not None
    assert error.billing_in_flight is not None
    assert error.attempts == 1
    assert error.usage.cost_in_usd == pytest.approx(0.5)
