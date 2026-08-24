"""Generation results and their calls-table and attempts-table conversion.

`Response`, `ToolCallTurn`, and `GenerationError` carry the call's `CallRecord`.
`to_tables` emits one row per call and one row per recorded attempt.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, NamedTuple

from pydantic import BaseModel

from langchaint.call import AttemptRecord, CallRecord, _CallCarrier, _CallLedger
from langchaint.exceptions import AbandonedCallError, GenerationError
from langchaint.messages import AssistantMessage, StopReason, ToolCall
from langchaint.pricing import Billing
from langchaint.usage import Usage

type GenerateResult[OutputT] = Response[OutputT] | ToolCallTurn[OutputT]
"""One successful generate result.

A match on `kind` selects the variant without importing the classes.
Only a structured tool-bound binding returns both variants.
Every other binding returns `Response` alone.
"""

type CallResult[OutputT] = GenerateResult[OutputT] | GenerationError
"""One call's terminal result: a success variant, or the failure carrier a batch returns at its item's index."""

type RowValue = str | int | float | bool | None
"""The scalar cell types to_tables emits."""


class _SuccessCarrier(_CallCarrier):
    """Provide the invariants and folds shared by the success variants.

    `_SuccessCarrier` is not a dataclass, so each frozen subclass declares its fields.
    """

    assistant_message: AssistantMessage

    def __post_init__(self) -> None:
        """Require retries before one final successful attempt record.

        Raises:
            ValueError: `attempt_records` is empty.
            ValueError: A non-final record has no error.
            ValueError: The final record carries an error.
        """
        if not self.attempt_records:
            raise ValueError("attempt_records must hold at least one record")
        if any(record.error is None for record in self.attempt_records[:-1]):
            raise ValueError("only the last attempt record may be error-free")
        if self.attempt_records[-1].error is not None:
            raise ValueError("the last attempt record of a success must be error-free")

    @property
    def attempts(self) -> int:
        """Count requests that langchaint observed going out."""
        return len(self.attempt_records)

    @property
    def usage(self) -> Usage:
        """Return the paid total across every attempt of the call.

        `cost_in_usd` is the amount to bill.
        Every billed retry contributes usage.
        This total can exceed the final output's `usage_successful_attempt`.
        `GenerationError.usage` uses the same paid-total scope.
        """
        return Usage.sum_of(record.usage for record in self.attempt_records)

    @property
    def usage_successful_attempt(self) -> Usage:
        """Return usage for the single kept answer.

        This usage matches `output`, `assistant_message`, and `raw.usage`.
        This value equals `usage` when no failed attempt was billed.
        """
        return self.attempt_records[-1].usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The turn's tool calls."""
        return self.assistant_message.tool_calls


@dataclass(frozen=True, kw_only=True)
class Response[OutputT](_SuccessCarrier):
    """One successful generation result.

    `output` contains assistant text or a validated `response_format` instance.
    `assistant_message` preserves the complete ordered turn for reuse.
    `raw` holds the mutable SDK response by reference.
    Copy `raw` before mutation.
    `usage` covers every attempt.
    `usage_successful_attempt` covers the final attempt.
    """

    output: OutputT
    # pyrefly: ignore[bad-override]  # _CallCarrier.call is writable. This frozen field is read-only.
    call: CallRecord
    raw: BaseModel
    stop_reason: StopReason
    # pyrefly: ignore[bad-override]  # _SuccessCarrier.assistant_message is writable. This frozen field is read-only.
    assistant_message: AssistantMessage
    kind: Literal["response"] = "response"


@dataclass(frozen=True, kw_only=True)
class ToolCallTurn[OutputT](_SuccessCarrier):
    """A structured result that requires one `ToolMessage` per tool call.

    `output` contains a parsed instance or `None` for a tool-only turn.
    Dispatch `tool_calls`, then append `assistant_message` and the results before generating again.
    """

    output: OutputT | None
    # pyrefly: ignore[bad-override]  # _CallCarrier.call is writable. This frozen field is read-only.
    call: CallRecord
    raw: BaseModel
    stop_reason: StopReason
    # pyrefly: ignore[bad-override]  # _SuccessCarrier.assistant_message is writable. This frozen field is read-only.
    assistant_message: AssistantMessage
    kind: Literal["tool_call_turn"] = "tool_call_turn"

    def __post_init__(self) -> None:
        """Enforce the shared success invariants. Require the turn to hold a tool call.

        Raises:
            ValueError: A check in `_SuccessCarrier.__post_init__` failed.
            ValueError: The turn has no tool call.
        """
        super().__post_init__()
        if not self.assistant_message.tool_calls:
            raise ValueError("a ToolCallTurn must hold at least one tool call")


def _success_variant[OutputT](
    *,
    splits_tool_call_turns: bool,
    output: OutputT,
    call: CallRecord,
    raw: BaseModel,
    stop_reason: StopReason,
    assistant_message: AssistantMessage,
) -> GenerateResult[OutputT]:
    """Build the success variant one adapter_result outcome concludes its call with.

    `splits_tool_call_turns` identifies a structured tool-bound binding.
    It returns `ToolCallTurn` for tool calls and `Response` otherwise.

    Raises:
        ValueError: The variant's `__post_init__` rejected the records or turn.
            Retry loops pass a fresh success, so this error indicates a langchaint defect.
    """
    result_class: type[Response[OutputT]] | type[ToolCallTurn[OutputT]] = (
        ToolCallTurn if splits_tool_call_turns and assistant_message.tool_calls else Response
    )
    return result_class(
        output=output,
        call=call,
        raw=raw,
        stop_reason=stop_reason,
        assistant_message=assistant_message,
    )


def _abandoned_call_error[ErrorT: AbandonedCallError](
    error_class: type[ErrorT], ledger: _CallLedger, billing_in_flight: Billing | None = None
) -> ErrorT:
    """Freeze an interrupted call into `AbandonedCallError` or `TimedOutError`.

    Include `billing_in_flight` only when the ledger still has an unrecorded request.
    """
    call = ledger.freeze()
    started_at_monotonic_seconds = ledger.in_flight_attempt_started_at_monotonic_seconds
    return error_class(
        call=call,
        billing_in_flight=(
            billing_in_flight if started_at_monotonic_seconds is not None else None
        ),
        in_flight_attempt_started_at_monotonic_seconds=started_at_monotonic_seconds,
    )


class Tables(NamedTuple):
    """The two tables that `to_tables` builds, joined on `call_id`."""

    calls: list[dict[str, RowValue]]
    attempts: list[dict[str, RowValue]]


def _output_cell(output: object) -> str | None:
    if output is None:
        return None
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


def _request_cell(result: CallResult[object]) -> str | None:
    if not isinstance(result, GenerationError) or result.request is None:
        return None
    return result.request.as_json()


def _call_row(*, call_id: int, result: CallResult[object]) -> dict[str, RowValue]:
    """Build one call-level row.

    `error_summary` describes a terminal error; attempt errors stay in the attempts table.
    `request_json` contains the full failed request when available.
    `elapsed_seconds` includes admission and backoff waits.
    """
    return {
        "call_id": call_id,
        "model": result.model,
        "provider_name": result.provider_name,
        "elapsed_seconds": result.elapsed_seconds,
        "attempts": result.attempts,
        "stop_reason": result.stop_reason,
        "error_summary": str(result) if isinstance(result, GenerationError) else None,
        "request_json": _request_cell(result),
        "output": None if isinstance(result, GenerationError) else _output_cell(result.output),
    }


def _attempt_row(
    *,
    call_id: int,
    attempt_index: int,
    kept: bool,
    record: AttemptRecord,
    call_started_at_monotonic_seconds: float,
) -> dict[str, RowValue]:
    """Build one attempt-level row.

    Timing fields are durations from the call or attempt start.
    Response identifiers are `None` when no response supplied them.
    `error_text` contains this attempt's transient error.
    """
    return _billing_cells(record.billing) | {
        "call_id": call_id,
        "attempt_index": attempt_index,
        "kept": kept,
        "started_after_seconds": (
            record.started_at_monotonic_seconds - call_started_at_monotonic_seconds
        ),
        "elapsed_seconds": record.elapsed_seconds,
        "seconds_to_first_item": (
            None
            if record.first_item_at_monotonic_seconds is None
            else record.first_item_at_monotonic_seconds - record.started_at_monotonic_seconds
        ),
        "model_served": record.model_served,
        "response_id": record.response_id,
        "request_id": record.request_id,
        "error_text": None if record.error is None else str(record.error),
        "assistant_message_json": (
            None
            if record.assistant_message is None
            else record.assistant_message.model_dump_json()
        ),
    }


def _cut_off_attempt_row(
    *,
    call_id: int,
    attempt_index: int,
    started_at_monotonic_seconds: float,
    call_started_at_monotonic_seconds: float,
    billing_in_flight: Billing | None,
) -> dict[str, RowValue]:
    """Build the attempts row for an interrupted in-flight request.

    End, response, and error fields are `None` because the request did not settle.
    """
    return _billing_cells(billing_in_flight) | {
        "call_id": call_id,
        "attempt_index": attempt_index,
        "kept": False,
        "started_after_seconds": (
            started_at_monotonic_seconds - call_started_at_monotonic_seconds
        ),
        "elapsed_seconds": None,
        "seconds_to_first_item": None,
        "model_served": None,
        "response_id": None,
        "request_id": None,
        "error_text": None,
        "assistant_message_json": None,
    }


def _billing_cells(billing: Billing | None) -> dict[str, RowValue]:
    """Build provider-reported billing cells for one attempt.

    Return `None` values when the provider reports no billing.
    `usage_raw_json` preserves provider-specific usage fields.
    """
    usage = None if billing is None else billing.usage
    usage_raw = None if billing is None else billing.usage_raw
    return {
        "service_tier": None if billing is None else billing.service_tier,
        "usage_raw_json": None if usage_raw is None else usage_raw.model_dump_json(),
        "input_tokens_cache_read": None if usage is None else usage.input_tokens_cache_read,
        "input_tokens_cache_read_cost_in_usd": (
            None if usage is None else usage.input_tokens_cache_read_cost_in_usd
        ),
        "input_tokens_cache_write": None if usage is None else usage.input_tokens_cache_write,
        "input_tokens_cache_write_cost_in_usd": (
            None if usage is None else usage.input_tokens_cache_write_cost_in_usd
        ),
        "input_tokens_cache_none": None if usage is None else usage.input_tokens_cache_none,
        "input_tokens_cache_none_cost_in_usd": (
            None if usage is None else usage.input_tokens_cache_none_cost_in_usd
        ),
        "input_tokens_total": None if usage is None else usage.input_tokens_total,
        "output_tokens": None if usage is None else usage.output_tokens,
        "output_tokens_cost_in_usd": None if usage is None else usage.output_tokens_cost_in_usd,
        "output_tokens_reasoning": None if usage is None else usage.output_tokens_reasoning,
        "provider_executed_tool_cost_in_usd": (
            None if usage is None else usage.provider_executed_tool_cost_in_usd
        ),
        "cost_in_usd": None if usage is None else usage.cost_in_usd,
        "input_cache_none_usd_per_million_tokens": (
            None if billing is None else billing.input_cache_none_usd_per_million_tokens
        ),
        "cache_read_usd_per_million_tokens": (
            None if billing is None else billing.cache_read_usd_per_million_tokens
        ),
        "cache_write_usd_per_million_tokens": (
            None if billing is None else billing.cache_write_usd_per_million_tokens
        ),
        "output_usd_per_million_tokens": (
            None if billing is None else billing.output_usd_per_million_tokens
        ),
    }


def to_tables[OutputT](results: CallResult[OutputT] | Iterable[CallResult[OutputT]]) -> Tables:
    """Flatten results into calls and attempts tables joined by `call_id`.

    `call_id` is each result's position in `results`.
    `kept` marks the final attempt of a successful call.
    An interrupted in-flight request gets an extra attempts row.
    A single result is accepted in place of an iterable.

    Args:
        results: One result or the results to flatten.
    """
    call_results = list(results) if isinstance(results, Iterable) else [results]
    calls: list[dict[str, RowValue]] = []
    attempts: list[dict[str, RowValue]] = []
    for call_id, result in enumerate(call_results):
        calls.append(_call_row(call_id=call_id, result=result))
        kept_index = (
            None if isinstance(result, GenerationError) else len(result.attempt_records) - 1
        )
        for attempt_index, record in enumerate(result.attempt_records):
            attempts.append(
                _attempt_row(
                    call_id=call_id,
                    attempt_index=attempt_index,
                    kept=attempt_index == kept_index,
                    record=record,
                    call_started_at_monotonic_seconds=result.started_at_monotonic_seconds,
                )
            )
        if isinstance(result, AbandonedCallError):
            cut_off_started_at = result.in_flight_attempt_started_at_monotonic_seconds
            if cut_off_started_at is not None:
                attempts.append(
                    _cut_off_attempt_row(
                        call_id=call_id,
                        attempt_index=len(result.attempt_records),
                        started_at_monotonic_seconds=cut_off_started_at,
                        call_started_at_monotonic_seconds=result.started_at_monotonic_seconds,
                        billing_in_flight=result.billing_in_flight,
                    )
                )
    return Tables(calls=calls, attempts=attempts)
