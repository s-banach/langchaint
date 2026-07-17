"""Response constructor invariants, the two usage scopes, and to_row over both result types.

The retry loops in llm.py and streaming.py are the only production constructors of Response,
so these invariants are what stops a refactor of either loop from building a success row whose records disagree;
the retry tests in test_bound_llm.py pin the record values themselves.
usage is the paid total folded from attempt_records and usage_successful_attempt is the last record's own,
so a retried billed 200 makes them diverge; both are exercised here.
to_row is the table seam: a success and a RetriesExhaustedError must flatten to the same keys,
or a mixed batch could not become one table.
"""

import time

import pytest
from pydantic import BaseModel

from langchaint import (
    ZERO_USAGE,
    AssistantMessage,
    AttemptRecord,
    ExceededMaxCompletionTokensError,
    RefusalError,
    Response,
    RetriesExhaustedError,
    TextPart,
    TransientError,
    Usage,
    to_row,
)


class _Raw(BaseModel):
    """Stand-in for the SDK's own response model held on Response.raw."""


class _RawUsage(BaseModel):
    """Stand-in for the SDK's own usage object held on AttemptRecord.usage_raw."""


_USAGE = Usage(
    input_tokens_cache_read=2,
    input_tokens_cache_write=3,
    input_tokens_cache_none=5,
    output_tokens=7,
    output_tokens_reasoning=2,
    cost_in_usd=0.5,
)


def _record(
    *,
    error: TransientError | None,
    usage: Usage = ZERO_USAGE,
    usage_raw: BaseModel | None = None,
) -> AttemptRecord:
    """Build one record whose bracket is a single instant; a non-billing attempt defaults to ZERO_USAGE."""
    now = time.monotonic()
    return AttemptRecord(
        started_at_monotonic_seconds=now,
        ended_at_monotonic_seconds=now,
        error=error,
        usage=usage,
        usage_raw=usage_raw,
    )


def _response[OutputT](
    *, output: OutputT, attempt_records: tuple[AttemptRecord, ...]
) -> Response[OutputT]:
    """Build a Response with the fields under test; everything else is fixed filler."""
    return Response(
        output=output,
        model="fake-model",
        provider_name="fake",
        attempt_records=attempt_records,
        elapsed_seconds=1.5,
        raw=_Raw(),
        stop_reason="end_turn",
        assistant_message=AssistantMessage(turn=(TextPart(text=str(output)),)),
    )


def _failure(*, attempt_records: tuple[AttemptRecord, ...]) -> RetriesExhaustedError:
    """Build a RetriesExhaustedError with the table fields set."""
    return RetriesExhaustedError(
        attempt_records=attempt_records,
        model="fake-model",
        provider_name="fake",
        elapsed_seconds=2.5,
        stop_reason=None,
    )


def test_response_rejects_empty_attempt_records() -> None:
    """A Response without a single record has no history and is rejected."""
    with pytest.raises(ValueError, match="at least one record"):
        _response(output="ok", attempt_records=())


def test_response_rejects_an_error_free_record_before_the_last() -> None:
    """A success record can only be last: the loop stops on the attempt that succeeded."""
    with pytest.raises(ValueError, match="only the last"):
        _response(
            output="ok",
            attempt_records=(_record(error=None), _record(error=TransientError("e"))),
        )


def test_response_rejects_a_failed_last_record() -> None:
    """A Response is a success, so its final record must be the one that succeeded."""
    with pytest.raises(ValueError, match="must be error-free"):
        _response(output="ok", attempt_records=(_record(error=TransientError("e")),))


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


def test_usage_is_the_paid_total_across_attempts() -> None:
    """A retried billed 200 makes usage exceed usage_successful_attempt by the retried attempt's billing."""
    response = _response(
        output="ok",
        attempt_records=(
            _record(error=TransientError("empty parse"), usage=_USAGE),
            _record(error=None, usage=_USAGE),
        ),
    )
    assert response.usage.cost_in_usd == pytest.approx(1.0)
    assert response.usage.output_tokens == 14
    assert response.usage_successful_attempt.cost_in_usd == pytest.approx(0.5)


def test_usage_equals_successful_attempt_when_only_transport_failures_billed() -> None:
    """Transport retries bill nothing (ZERO_USAGE), so the paid total equals the kept answer's usage."""
    response = _response(
        output="ok",
        attempt_records=(
            _record(error=TransientError("timeout"), usage=ZERO_USAGE),
            _record(error=None, usage=_USAGE),
        ),
    )
    assert response.usage == response.usage_successful_attempt == _USAGE


def test_usage_raw_is_recoverable_by_reference_from_the_records() -> None:
    """The raw SDK usage object travels on the record and is recoverable by identity."""
    raw_usage = _RawUsage()
    response = _response(
        output="ok",
        attempt_records=(_record(error=None, usage=_USAGE, usage_raw=raw_usage),),
    )
    assert response.attempt_records[-1].usage_raw is raw_usage


def test_to_row_success_flattens_output_and_usage() -> None:
    """A success row carries the output, no error_text, and the real usage counters."""
    row = to_row(_response(output="hello", attempt_records=(_record(error=None, usage=_USAGE),)))
    assert row["output"] == "hello"
    assert row["error_text"] is None
    assert row["stop_reason"] == "end_turn"
    assert row["cost_in_usd"] == 0.5
    assert row["attempts"] == 1
    assert row["input_tokens_cache_none"] == 5
    assert row["input_tokens_total"] == 10
    assert row["output_tokens"] == 7
    assert row["output_tokens_reasoning"] == 2


def test_to_row_cost_is_the_paid_total_across_attempts() -> None:
    """to_row's cost_in_usd sums every attempt's billing, not just the kept answer's."""
    row = to_row(
        _response(
            output="ok",
            attempt_records=(
                _record(error=TransientError("empty parse"), usage=_USAGE),
                _record(error=None, usage=_USAGE),
            ),
        )
    )
    assert row["cost_in_usd"] == pytest.approx(1.0)
    assert row["output_tokens"] == 14


def test_to_row_structured_output_becomes_json() -> None:
    """A pydantic output instance is flattened to its JSON, not its repr."""
    row = to_row(_response(output=_USAGE, attempt_records=(_record(error=None, usage=_USAGE),)))
    assert row["output"] == _USAGE.model_dump_json()


def test_to_row_failure_is_none_and_zero_with_the_error_chain() -> None:
    """A failure row nulls output and stop_reason, zeroes cost and usage, and carries error_text."""
    row = to_row(
        _failure(
            attempt_records=(
                _record(error=TransientError("e1")),
                _record(error=TransientError("e2")),
            )
        )
    )
    assert row["output"] is None
    assert row["stop_reason"] is None
    assert row["cost_in_usd"] == 0.0
    assert row["error_text"] == "attempt 1: e1; attempt 2: e2"
    assert row["attempts"] == 2
    assert row["input_tokens_total"] == 0
    assert row["output_tokens"] == 0


def test_to_row_refusal_reports_its_billing_and_reason() -> None:
    """A refusal row carries the rejected 200's cost and usage, not zeros, and stop_reason "refusal"."""
    row = to_row(
        RefusalError(
            attempt_records=(_record(error=None, usage=_USAGE),),
            model="fake-model",
            provider_name="fake",
            elapsed_seconds=1.0,
            stop_reason="refusal",
        )
    )
    assert row["output"] is None
    assert row["stop_reason"] == "refusal"
    assert row["cost_in_usd"] == 0.5
    assert row["error_text"] == "the model refused to produce structured output"
    assert row["attempts"] == 1
    assert row["input_tokens_total"] == 10
    assert row["output_tokens"] == 7


def test_to_row_truncation_reports_its_billing_and_reason() -> None:
    """A truncation row carries the rejected 200's cost and usage and stop_reason "max_tokens"."""
    row = to_row(
        ExceededMaxCompletionTokensError(
            attempt_records=(_record(error=None, usage=_USAGE),),
            model="fake-model",
            provider_name="fake",
            elapsed_seconds=1.0,
            stop_reason="max_tokens",
        )
    )
    assert row["output"] is None
    assert row["stop_reason"] == "max_tokens"
    assert row["cost_in_usd"] == 0.5
    assert (
        row["error_text"]
        == "the structured response reached max_completion_tokens before its JSON parsed"
    )
    assert row["attempts"] == 1
    assert row["input_tokens_total"] == 10
    assert row["output_tokens"] == 7


def test_to_row_success_and_failure_share_the_same_keys() -> None:
    """The whole point of the split: a mixed batch converts to one table."""
    success = to_row(_response(output="ok", attempt_records=(_record(error=None, usage=_USAGE),)))
    failure = to_row(_failure(attempt_records=(_record(error=TransientError("e")),)))
    assert success.keys() == failure.keys()
