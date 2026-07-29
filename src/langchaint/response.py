"""The generate results: the success Response and the terminal GenerationError.

A generate that succeeds returns a Response; one that ends terminally (retries exhausted on transient errors, a refusal,
a truncation at the token cap, or a provider error langchaint does not retry) raises or returns a GenerationError.
Both carry the CallRecord their retry loop froze, because a call's history survives only if the
result carries it: attempt_records is that history.
On a Response every record but the last failed and the last succeeded;
on a GenerationError the records describe the terminal outcome.
to_tables flattens a list of results to two tables of scalars, one row per result and one row per
attempt, joined on call_id.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from pydantic import BaseModel

from langchaint.call import AttemptRecord, CallRecord, _CallCarrier, _CallLedger
from langchaint.exceptions import AbandonedCallError, GenerationError
from langchaint.messages import AssistantMessage, StopReason, ToolCall
from langchaint.pricing import Billing
from langchaint.usage import Usage

type CallResult[OutputT] = Response[OutputT] | GenerationError
"""One call's terminal result: the success carrier, or the failure carrier a batch returns in its slot."""

type RowValue = str | int | float | bool | None
"""The scalar cell types to_tables emits."""


@dataclass(frozen=True, kw_only=True)
class Response[OutputT](_CallCarrier):
    """One successful generate result.

    output is the assistant text, or the response_format instance validated from the turn's text.
    It is None when the turn parsed no instance, which on a structured tool-bound binding means the model called tools.
    Only that binding types output optional, so no other caller has a None to handle.
    A turn can both parse an instance and call tools, setting output and tool_calls at once.
    tool_calls is therefore what says whether this turn owes the model a tool result.
    call is this call's history: model, provider_name, attempt_records, started_at_monotonic_seconds,
    and elapsed_seconds read off it.
    attempts counts its records.
    Every attempt record but the last failed and the last succeeded.
    assistant_message is the adapter-built turn exactly as the provider produced it,
    the whole ordered turn (reasoning, text, and tool calls in emission order),
    held by reference for appending to a Sequence[Message].
    Rebuilding it from output and tool_calls is lossy (it drops reasoning and the element order)
    and is the rewrap this field exists to prevent.
    The last attempt record holds the same object, because the record is where every attempt's turn
    goes and this one is the attempt that succeeded.
    raw is the SDK's own response model, held by reference (no dump, no copy; call raw.model_dump() for a dict);
    on streams it comes from the SDK-assembled final message.
    It is a live, mutable pydantic object shared with the adapter, so despite the frozen dataclass around it,
    treat it read-only and raw.model_copy() before mutating.
    usage and usage_successful_attempt are two scopes, both folded from attempt_records (see their docstrings):
    usage is the paid total across every attempt, usage_successful_attempt the single kept answer's own.
    """

    output: OutputT
    # pyrefly: ignore[bad-override]  # read-only here, read-write on _CallCarrier; see its docstring
    call: CallRecord
    raw: BaseModel
    stop_reason: StopReason
    assistant_message: AssistantMessage

    def __post_init__(self) -> None:
        """Enforce that the records describe a success: retries before, one success last.

        Raises:
            ValueError: attempt_records is empty, a non-final record has no error, or the final record carries an error.
        """
        if not self.attempt_records:
            raise ValueError("attempt_records must hold at least one record")
        if any(record.error is None for record in self.attempt_records[:-1]):
            raise ValueError("only the last attempt record may be error-free")
        if self.attempt_records[-1].error is not None:
            raise ValueError("the last attempt record of a success must be error-free")

    @property
    def attempts(self) -> int:
        """Requests langchaint observed going out: one attempt record each."""
        return len(self.attempt_records)

    @property
    def usage(self) -> Usage:
        """The paid total across every attempt of the call, carrying cost_in_usd, the number to bill on.

        A call that retried a billed 200 (an empty structured parse retried as transient) counts every such
        attempt, so this can exceed the tokens of the single answer in output; usage_successful_attempt is
        that single answer's own usage. This is the same paid-total scope as GenerationError.usage,
        so the two mean the same thing.
        """
        return Usage.sum_of(record.usage for record in self.attempt_records)

    @property
    def usage_successful_attempt(self) -> Usage:
        """The single kept answer's own usage, the one matching output, assistant_message, and raw.usage.

        The last attempt record is the success (__post_init__ enforces it), so this reads it directly.
        It equals usage where no failed attempt billed, and is smaller where one did.
        """
        return self.attempt_records[-1].usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The turn's tool calls."""
        return self.assistant_message.tool_calls


def _abandoned_call_error[ErrorT: AbandonedCallError](
    error_class: type[ErrorT], ledger: _CallLedger, billing_in_flight: Billing | None = None
) -> ErrorT:
    """Freeze a cut-off call's ledger into the error that accounts for it.

    Call while the exception that cut the call off unwinds, after the in-flight slot has been
    returned and, on the stream path, after the connection has closed, so nothing the ledger reports
    is still moving.
    billing_in_flight defaults to None, which is what a caller with no channel for observing an
    in-flight attempt states; only an open stream has one.
    error_class selects which account is built: AbandonedCallError for a cancellation, TimedOutError
    for a deadline langchaint owned.

    The freeze runs before the in-flight attempt is read, because it closes a response that had
    arrived and been staged into an ordinary record carrying that response's own billing. What is in
    flight afterwards is the attempt that got no record at all, so one request is either a record or
    the in-flight attempt and never both, and a report of what a recorded attempt billed is dropped
    here rather than added on top of the record that already states it.
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
    """The two tables to_tables builds, joined on call_id.

    A NamedTuple so the two unpack positionally and read by name, and so neither is mistaken for the
    other at a call site where both are list[dict[str, RowValue]].
    """

    calls: list[dict[str, RowValue]]
    attempts: list[dict[str, RowValue]]


def _output_cell(output: object) -> str | None:
    """Flatten one output to its cell: a pydantic instance as its JSON, anything else as its str.

    None stays None rather than becoming "None", which would write a word into a column readers scan
    for real output.
    """
    if output is None:
        return None
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


def _request_cell(result: CallResult[object]) -> str | None:
    """Render what a failed call sent, None on a success and None where no request was built."""
    if not isinstance(result, GenerationError) or result.request is None:
        return None
    return result.request.as_json()


def _call_row(*, call_id: int, result: CallResult[object]) -> dict[str, RowValue]:
    """One row of the calls table: what is measured at the call rather than summed from its attempts.

    error_summary is how the call ended, the carrier's own __str__, and None on a success. It is not
    error_text: RetriesExhaustedError.error_text folds every attempt's error into one string, and
    those errors have rows of their own in the attempts table.
    output is the parsed result, None on a failure and None on the tool-call turn of a structured
    tool-bound binding, which stop_reason "tool_use" tells apart.
    request_json is what every attempt of a failed call sent, rendered as a JSON object. It is None
    on a success, whose request is reconstructible from the Sequence[Message] and the binding the caller
    still holds, and None on a failure the adapter declared before building a request.
    It holds the whole prompt, so a caller writing this table somewhere the outputs may not go drops
    the column.
    elapsed_seconds belongs here because it spans the RateLimiter waits and backoff sleeps no attempt
    bracket covers, so it is its own measurement rather than a fold. Spend is not: every billing
    column sits in the attempts table and a caller sums what they need.
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
    """One row of the attempts table: what one request billed, produced, and took.

    The timing columns are durations, not clock readings, because AttemptRecord's stamps are raw
    time.monotonic() values that mean nothing outside the process that took them.
    started_after_seconds places the attempt on the call's timeline, so the gap between one row's
    start and the previous row's end is the RateLimiter wait or backoff sleep between them, and the
    first row's own value is the slot wait before the first request went out.
    seconds_to_first_item is how long this attempt's stream took to yield anything, None on a
    non-stream attempt and on a stream that yielded nothing.
    model_served and response_id are what the provider said about the response this attempt
    received, both None where none arrived; the calls table's model beside them is the id langchaint
    sent.
    request_id is the request-id header, which provider support asks for. It is filled on an attempt
    that failed too, read off the SDK's own error there, and on a streamed attempt it comes from the
    open stream. None where none of those had one.
    error_text is this attempt's own error, None where it has none, which includes an attempt whose
    failure was the call's rather than the attempt's (a rejected request, a provider-declared final
    error, an exception the adapter could not place); the calls row's error_summary carries those.
    _billing_cells fills the rest, from what the provider reported.
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
    """Return the attempts row for the request that was in flight when a call was cut off.

    It fills the same columns as a settled attempt's row so one SUM over cost_in_usd closes the
    archive to the total the carriers report.
    The columns describing how the attempt ended are None because nobody observed it ending:
    elapsed_seconds, seconds_to_first_item, error_text, and everything a response would have said
    about itself. kept is false, no turn having become an answer.
    started_after_seconds places the request on the call's timeline the same way a settled row does.
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
    """Return the attempts columns the provider's report fills, all None where it reported nothing.

    None rather than zero keeps "the provider reported nothing" distinct from "the provider billed
    zero"; zeros would sum the two together.
    Shared by a settled attempt's row and a cut-off attempt's, so one definition fixes the names.
    usage_raw_json is the provider's own usage object as it reported it, which is what keeps the
    billing detail the neutral counters merge or drop alive past the process. None where the
    provider reported no usage, which is not the same as there being no billing at all.
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
    """Flatten results to a calls table and an attempts table of scalars, joined on call_id.

    A success and a failure fill the same keys in each table, so a mixed batch is one pair of tables.
    Every column sits at the grain of the thing it is true of: a call fact repeated across attempt
    rows would assert it of attempts it is not true of.

    call_id is the result's position in the results given here, which is what joins an attempt row
    back to the call it belongs to. A caller concatenating the output of two to_tables calls offsets
    the second, since each numbers from zero.
    kept marks the attempt whose turn became the answer: the last attempt of a Response, and no
    attempt of a GenerationError.
    An AbandonedCallError cut off mid-request gets one attempts row past its records, for the
    request that was in flight, so summing cost_in_usd over the archive reaches the total the
    carriers report.
    A single result is accepted in place of an iterable, and is told apart by not being iterable,
    which no Response or GenerationError is.

    Repricing held rows is a join against the caller's own rate table on model, provider_name, and
    service_tier: the first two are call columns and the third an attempt column.
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
