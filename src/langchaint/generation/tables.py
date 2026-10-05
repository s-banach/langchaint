"""Convert outcomes into outcome and request tables."""

from collections.abc import Iterable
from typing import NamedTuple

from pydantic import BaseModel

from langchaint.billing.pricing import Billing
from langchaint.generation.errors import GenerationError, _GenerationErrorRecordBase
from langchaint.generation.request_history import (
    CutOffRequestRecord,
    RequestProviderData,
    SettledRequestRecord,
)
from langchaint.generation.response import (
    GenerationOutcome,
    GenerationWithoutToolCalls,
    GenerationWithToolCalls,
    InputOutcomeRecord,
    _GenerationRecordBase,
)

type RowValue = str | int | float | bool | None


class Tables(NamedTuple):
    """The outcome and request tables joined on `outcome_index`."""

    outcomes: list[dict[str, RowValue]]
    requests: list[dict[str, RowValue]]


def _output_cell(output: object) -> str | None:
    if output is None:
        return None
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


def _billing_cells(
    billing: Billing | None, request_provider_data: RequestProviderData | None
) -> dict[str, RowValue]:
    usage = None if billing is None else billing.usage
    usage_raw = None if request_provider_data is None else request_provider_data.usage_raw
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
        "input_tokens_cache_none_usd_per_million_tokens": None
        if billing is None
        else billing.usd_per_million_tokens.input_tokens_cache_none,
        "input_tokens_cache_read_usd_per_million_tokens": None
        if billing is None
        else billing.usd_per_million_tokens.input_tokens_cache_read,
        "input_tokens_cache_write_usd_per_million_tokens": None
        if billing is None
        else billing.usd_per_million_tokens.input_tokens_cache_write,
        "output_tokens_usd_per_million_tokens": None
        if billing is None
        else billing.usd_per_million_tokens.output_tokens,
    }


def _request_row(
    *,
    outcome_index: int,
    request_index: int,
    kept: bool,
    request_record: SettledRequestRecord | CutOffRequestRecord,
    request_provider_data: RequestProviderData | None,
) -> dict[str, RowValue]:
    common = _billing_cells(request_record.billing, request_provider_data) | {
        "outcome_index": outcome_index,
        "request_index": request_index,
        "kept": kept,
        "started_after_seconds": request_record.started_after_seconds,
    }
    if request_record.kind == "cut_off":
        return common | {
            "elapsed_seconds": None,
            "first_item_after_seconds": request_record.first_item_after_seconds,
            "model_served": None,
            "response_id": None,
            "request_id": None,
            "error_text": None,
            "assistant_message_json": None,
        }
    return common | {
        "elapsed_seconds": request_record.elapsed_seconds,
        "first_item_after_seconds": request_record.first_item_after_seconds,
        "model_served": request_record.model_served,
        "response_id": request_record.response_id,
        "request_id": request_record.request_id,
        "error_text": None if request_record.error is None else str(request_record.error),
        "assistant_message_json": None
        if request_record.assistant_message is None
        else request_record.assistant_message.model_dump_json(),
    }


def to_tables[OutputT, WithToolCallsOutputT](
    outcomes: GenerationOutcome[OutputT, WithToolCallsOutputT]
    | InputOutcomeRecord[OutputT, WithToolCallsOutputT]
    | Iterable[
        GenerationOutcome[OutputT, WithToolCallsOutputT]
        | InputOutcomeRecord[OutputT, WithToolCallsOutputT]
    ],
) -> Tables:
    """Flatten live outcomes or outcome records into outcome and request tables.

    `outcome_index` is a value's position in `outcomes`.
    An `AbandonedStreamRecord` row has neither `output` nor `error_text`, and no kept request.
    """
    values = (
        list(outcomes)
        if isinstance(outcomes, Iterable) and not isinstance(outcomes, BaseModel)
        else [outcomes]
    )
    outcome_rows: list[dict[str, RowValue]] = []
    request_rows: list[dict[str, RowValue]] = []
    for outcome_index, value in enumerate(values):
        if isinstance(
            value, (GenerationWithoutToolCalls, GenerationWithToolCalls, GenerationError)
        ):
            record = value.record
            live_request_provider_data = value.request_provider_data
        else:
            record = value
            live_request_provider_data = ()
        live_error = value if isinstance(value, GenerationError) else None
        is_error = isinstance(record, _GenerationErrorRecordBase)
        is_generation = isinstance(record, _GenerationRecordBase)
        request_history = record.request_history
        outcome_rows.append({
            "outcome_index": outcome_index,
            "model": request_history.model,
            "provider_name": request_history.provider_name,
            "elapsed_seconds": request_history.elapsed_seconds,
            "request_count": len(request_history.records),
            "stop_reason": record.stop_reason,
            "error_text": record.error_text if is_error else None,
            "request_params_json": None
            if live_error is None or live_error.request_params is None
            else live_error.request_params.as_json(),
            "output": _output_cell(record.output) if is_generation else None,
        })
        kept_index = len(request_history.records) - 1 if is_generation else None
        for request_index, request_record in enumerate(request_history.records):
            request_rows.append(
                _request_row(
                    outcome_index=outcome_index,
                    request_index=request_index,
                    kept=request_index == kept_index,
                    request_record=request_record,
                    request_provider_data=(
                        live_request_provider_data[request_index]
                        if live_request_provider_data
                        else None
                    ),
                )
            )
    return Tables(outcomes=outcome_rows, requests=request_rows)
