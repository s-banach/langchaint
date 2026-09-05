"""Convert generation results into calls and attempts tables."""

from collections.abc import Iterable
from typing import NamedTuple

from pydantic import BaseModel

from langchaint.billing.pricing import Billing
from langchaint.generation.call import (
    AttemptProviderData,
    CutOffAttemptRecord,
    SettledAttemptRecord,
)
from langchaint.generation.errors import GenerationError, _GenerationErrorRecordBase
from langchaint.generation.response import (
    CallResult,
    CallResultRecord,
    Response,
    ToolCallTurn,
    _result_record,
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
