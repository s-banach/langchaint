"""Test response variants, Usage scopes, and to_tables.

Constructor tests verify CallRecord invariants.
_success_variant tests verify variant selection.
to_tables tests verify rows for successes and GenerationError variants.
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
    MaxCompletionTokensExceededError,
    RefusalError,
    Response,
    RetriesExhaustedError,
    StopReason,
    TextPart,
    ToolCall,
    ToolCallTurn,
    TransientError,
    Usage,
    to_tables,
)
from langchaint.adapter import RequestParams
from langchaint.call import ResponseIdentity, _CallLedger
from langchaint.response import _abandoned_call_error, _success_variant
from tests.helpers import CALL_STARTED_AT, StubRaw, attempt_record, call_record, stated_billing


@dataclass(frozen=True, kw_only=True)
class _Request(RequestParams):
    """Provide a request for GenerationError tests."""

    prompt: str

    @override
    def as_json(self) -> str:
        """Serialize prompt as JSON."""
        return json.dumps({"prompt": self.prompt})


_REQUEST = _Request(prompt="the-prompt-text")


class _ProviderUsage(BaseModel):
    """Provide provider-specific usage fields."""

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
    provider_executed_tool_cost_in_usd=0.0,
)
"""One attempt's billing: four category costs summing to 0.5."""


def _response[OutputT](
    *,
    output: OutputT,
    attempt_records: tuple[AttemptRecord, ...],
) -> Response[OutputT]:
    """Build a Response for constructor tests."""
    return Response(
        output=output,
        call=call_record(attempt_records, elapsed_seconds=1.5),
        raw=StubRaw(),
        stop_reason="end_turn",
        assistant_message=AssistantMessage(turn=(TextPart(text=str(output)),)),
    )


_TOOL_CALL = ToolCall(id="call1", name="lookup", args_json='{"q": "tide"}')

_TOOL_CALL_TURN_MESSAGE = AssistantMessage(turn=(_TOOL_CALL,))
"""Provide one ToolCall for ToolCallTurn tests."""


def _tool_call_turn[OutputT](
    *,
    output: OutputT | None,
    attempt_records: tuple[AttemptRecord, ...],
    assistant_message: AssistantMessage = _TOOL_CALL_TURN_MESSAGE,
) -> ToolCallTurn[OutputT]:
    """Build a ToolCallTurn for constructor tests."""
    return ToolCallTurn(
        output=output,
        call=call_record(attempt_records, elapsed_seconds=1.5),
        raw=StubRaw(),
        stop_reason="tool_use",
        assistant_message=assistant_message,
    )


def _failure(*, attempt_records: tuple[AttemptRecord, ...]) -> RetriesExhaustedError:
    """Build a RetriesExhaustedError with the table fields set."""
    return RetriesExhaustedError(
        call=call_record(attempt_records, elapsed_seconds=2.5), request=_REQUEST
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
            attempt_records=(
                attempt_record(error=None),
                attempt_record(error=TransientError("e")),
            ),
        )


def test_response_rejects_a_failed_last_record() -> None:
    """A Response is a success, so its final record must be the one that succeeded."""
    with pytest.raises(ValueError, match="must be error-free"):
        _ = _response(output="ok", attempt_records=(attempt_record(error=TransientError("e")),))


def test_tool_call_turn_rejects_a_turn_without_a_tool_call() -> None:
    """The variant exists to say the turn called tools, so a turn without a call is a construction defect."""
    with pytest.raises(ValueError, match="at least one tool call"):
        _ = _tool_call_turn(
            output=None,
            attempt_records=(attempt_record(error=None),),
            assistant_message=AssistantMessage(turn=(TextPart(text="no calls"),)),
        )


def test_tool_call_turn_enforces_the_shared_success_record_invariants() -> None:
    """ToolCallTurn runs the shared checks before its own, so a failed last record is rejected."""
    with pytest.raises(ValueError, match="must be error-free"):
        _ = _tool_call_turn(
            output=None, attempt_records=(attempt_record(error=TransientError("e")),)
        )


@pytest.mark.parametrize(
    ("splits_tool_call_turns", "assistant_message", "output", "expected_class"),
    [
        (True, _TOOL_CALL_TURN_MESSAGE, None, ToolCallTurn),
        (True, AssistantMessage(turn=(TextPart(text="done"),)), "done", Response),
        (False, _TOOL_CALL_TURN_MESSAGE, None, Response),
        (
            True,
            AssistantMessage(turn=(TextPart(text="parsed"), _TOOL_CALL)),
            "parsed",
            ToolCallTurn,
        ),
    ],
    ids=[
        "split_binding_tool_call_turn",
        "split_binding_final_turn",
        "unsplit_binding_tool_call_turn",
        "split_binding_tool_call_turn_with_instance",
    ],
)
def test_success_variant_is_tool_call_turn_only_where_a_split_bindings_turn_called_tools(
    *,
    splits_tool_call_turns: bool,
    assistant_message: AssistantMessage,
    output: str | None,
    expected_class: type[Response[object] | ToolCallTurn[object]],
) -> None:
    """The split needs both: the binding splits and the turn called tools; either alone is a Response.

    output rides whichever variant results, so a turn carrying both an instance and tool calls keeps it.
    """
    result = _success_variant(
        splits_tool_call_turns=splits_tool_call_turns,
        output=output,
        call=call_record((attempt_record(error=None),), elapsed_seconds=1.0),
        raw=StubRaw(),
        stop_reason="tool_use",
        assistant_message=assistant_message,
    )
    assert isinstance(result, expected_class)
    assert result.assistant_message is assistant_message
    assert result.output is output


def test_to_tables_failure_writes_complete_call_and_attempt_rows() -> None:
    """A failed call retains its request, summary, attempt errors, and record-derived fields."""
    failure = _failure(
        attempt_records=(
            attempt_record(error=TransientError("e1")),
            attempt_record(error=TransientError("e2")),
        )
    )
    assert failure.attempts == 2
    assert [str(error) for error in failure.errors_from_attempts] == ["e1", "e2"]
    assert failure.error_text == "attempt 1: e1; attempt 2: e2"
    assert failure.request == _REQUEST
    assert "the-prompt-text" not in failure.error_text
    assert "the-prompt-text" not in str(failure)

    calls, attempts = to_tables(failure)
    assert calls == [
        {
            "call_id": 0,
            "model": "fake-model",
            "provider_name": "fake",
            "elapsed_seconds": 2.5,
            "attempts": 2,
            "stop_reason": None,
            "error_summary": "2 attempts failed; last: e2",
            "request_json": json.dumps({"prompt": "the-prompt-text"}),
            "output": None,
        }
    ]
    assert [row["error_text"] for row in attempts] == ["e1", "e2"]
    assert [row["kept"] for row in attempts] == [False, False]


@pytest.mark.parametrize(
    ("failed_attempt_usage", "expected_cost_in_usd", "expected_output_tokens"),
    [(_USAGE, 1.0, 14), (ZERO_USAGE, 0.5, 7)],
    ids=["failed_attempt_billed", "failed_attempt_billed_nothing"],
)
def test_usage_is_the_paid_total_across_attempts(
    failed_attempt_usage: Usage, expected_cost_in_usd: float, expected_output_tokens: int
) -> None:
    """Usage folds every attempt's billing; usage_successful_attempt stays the kept answer's own.

    A retried billed response separates total Usage from successful-attempt Usage.
    """
    response = _response(
        output="ok",
        attempt_records=(
            attempt_record(error=TransientError("empty parse"), usage=failed_attempt_usage),
            attempt_record(error=None, usage=_USAGE),
        ),
    )
    assert response.usage.cost_in_usd == pytest.approx(expected_cost_in_usd)
    assert response.usage.output_tokens == expected_output_tokens
    assert response.usage_successful_attempt.cost_in_usd == pytest.approx(0.5)


def test_to_tables_success_writes_complete_call_and_attempt_rows() -> None:
    """A retried success retains output, billing, rates, provider usage, timing, and identifiers."""
    turn = AssistantMessage(turn=(TextPart(text="hello"),))
    calls, attempts = to_tables(
        _response(
            output=_USAGE,
            attempt_records=(
                attempt_record(
                    error=TransientError("empty parse"),
                    usage=_USAGE,
                    started_after_seconds=3.0,
                    elapsed_seconds=0.5,
                    request_id="req_1",
                ),
                attempt_record(
                    error=None,
                    usage=_USAGE,
                    input_cache_none_usd_per_million_tokens=20.0,
                    usage_raw=_ProviderUsage(
                        cache_creation_5m_input_tokens=3,
                        server_tool_use_requests=1,
                    ),
                    started_after_seconds=9.0,
                    elapsed_seconds=2.0,
                    seconds_to_first_item=0.75,
                    turn=turn,
                    model_served="fake-model-2026-01-01",
                    response_id="msg_9",
                    request_id="req_2",
                ),
            ),
        )
    )
    (call_row,) = calls
    assert call_row == {
        "call_id": 0,
        "model": "fake-model",
        "provider_name": "fake",
        "elapsed_seconds": 1.5,
        "attempts": 2,
        "stop_reason": "end_turn",
        "error_summary": None,
        "request_json": None,
        "output": _USAGE.model_dump_json(),
    }
    assert [row["attempt_index"] for row in attempts] == [0, 1]
    assert [row["kept"] for row in attempts] == [False, True]
    assert [row["error_text"] for row in attempts] == ["empty parse", None]
    assert [row["cost_in_usd"] for row in attempts] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert [row["output_tokens"] for row in attempts] == [7, 7]
    assert [row["started_after_seconds"] for row in attempts] == [3.0, 9.0]
    assert [row["elapsed_seconds"] for row in attempts] == [0.5, 2.0]
    assert [row["seconds_to_first_item"] for row in attempts] == [None, 0.75]
    assert [row["model_served"] for row in attempts] == [None, "fake-model-2026-01-01"]
    assert [row["response_id"] for row in attempts] == [None, "msg_9"]
    assert [row["request_id"] for row in attempts] == ["req_1", "req_2"]
    assert attempts[0]["usage_raw_json"] is None
    kept = attempts[1]
    assert kept["service_tier"] == "stub"
    assert kept["assistant_message_json"] == turn.model_dump_json()
    assert json.loads(str(kept["usage_raw_json"])) == {
        "cache_creation_5m_input_tokens": 3,
        "server_tool_use_requests": 1,
    }
    assert kept["input_tokens_cache_none"] == 5
    assert kept["input_tokens_cache_none_cost_in_usd"] == 0.1
    assert kept["input_tokens_total"] == 10
    assert kept["output_tokens_cost_in_usd"] == 0.2
    assert kept["output_tokens_reasoning"] == 2
    assert kept["input_cache_none_usd_per_million_tokens"] == 20.0
    assert isinstance(kept["output_usd_per_million_tokens"], float)
    assert math.isnan(kept["output_usd_per_million_tokens"])


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
        provider_executed_tool_cost_in_usd=float("nan"),
    )
    calls, attempts = to_tables(
        _response(output="hello", attempt_records=(attempt_record(error=None, usage=unpriced),))
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
            attempt_records=(
                attempt_record(error=TransientError("no response"), reported_billing=False),
            )
        )
    )
    (attempt_row,) = attempts
    assert attempt_row["service_tier"] is None
    assert attempt_row["cost_in_usd"] is None
    assert attempt_row["input_tokens_total"] is None
    assert attempt_row["output_tokens"] is None
    assert attempt_row["output_usd_per_million_tokens"] is None
    assert attempt_row["error_text"] == "no response"


def test_to_tables_writes_a_tool_call_turn_as_a_null_output_with_no_error_summary() -> None:
    """A ToolCallTurn that parsed no instance is a success whose output cell is None.

    A ToolCallTurn without parsed output stores None in the output column.
    """
    calls, _ = to_tables(
        _tool_call_turn(output=None, attempt_records=(attempt_record(error=None),))
    )
    assert calls[0]["output"] is None
    assert calls[0]["error_summary"] is None
    assert calls[0]["stop_reason"] == "tool_use"


def test_to_tables_writes_no_request_where_the_call_built_none() -> None:
    """A failure the adapter declared before building a request writes a null cell, not an empty one."""
    calls, _attempts = to_tables(
        RefusalError(
            call=call_record((attempt_record(error=None, usage=_USAGE),), elapsed_seconds=1.0),
            request=None,
        )
    )
    assert calls[0]["request_json"] is None


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

    Call failure uses error_summary while its attempt error_text remains None.
    """
    calls, attempts = to_tables(
        error_class(
            call=call_record((attempt_record(error=None, usage=_USAGE),), elapsed_seconds=1.0),
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
        _response(output="ok", attempt_records=(attempt_record(error=None, usage=_USAGE),)),
        _failure(
            attempt_records=(
                attempt_record(error=TransientError("e1")),
                attempt_record(error=TransientError("e2")),
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
        call=call_record(attempt_records, elapsed_seconds=3.0),
        billing_in_flight=billing_in_flight,
        in_flight_attempt_started_at_monotonic_seconds=(
            None
            if in_flight_started_after_seconds is None
            else CALL_STARTED_AT + in_flight_started_after_seconds
        ),
    )


def test_to_tables_rows_the_request_that_was_in_flight_when_the_call_was_cut_off() -> None:
    """Attempt rows include the abandoned request after settled records."""
    calls, attempts = to_tables(
        _abandoned(
            attempt_records=(
                attempt_record(error=TransientError("e1"), started_after_seconds=0.0),
            ),
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
            attempt_records=(attempt_record(error=TransientError("e1")),),
            billing_in_flight=None,
            in_flight_started_after_seconds=None,
        )
    )
    assert [row["error_text"] for row in attempts] == ["e1"]


def test_a_cut_off_call_counts_a_staged_response_once() -> None:
    """A settled response does not remain in flight."""
    ledger = _CallLedger(model="fake-model", provider_name="fake")
    ledger.start_call()
    ledger.start_attempt()
    ledger.stage_response(
        raw=StubRaw(),
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
