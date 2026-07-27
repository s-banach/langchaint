"""The generate results: the success Response, the terminal GenerationError, and the AbandonedCall record.

A generate that succeeds returns a Response; one that ends terminally (retries exhausted on transient errors, a refusal,
a truncation at the token cap, or a provider error langchaint does not retry) raises or returns a GenerationError;
one whose result never reaches the caller, cut off by a cancellation or discarded when a batch raises,
leaves an AbandonedCall on the caller's abandoned_call_log.
Each of the three carries the CallRecord its retry loop froze, because a call's history survives
only if the result carries it: attempt_records is that history.
On a Response every record but the last failed and the last succeeded;
on a GenerationError the records describe the terminal outcome;
on an AbandonedCall they cover the attempts that settled, with no record for an attempt still in
flight when the call was cut off.
to_row flattens a Response or a GenerationError to one dict of scalars with the same keys,
so a mixed list of successes and failures converts directly to a table.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from langchaint.call import CallRecord, _CallCarrier
from langchaint.exceptions import GenerationError
from langchaint.messages import AssistantMessage, StopReason, ToolCall
from langchaint.usage import ZERO_USAGE, Usage

type RowValue = str | int | float | bool | None
"""The scalar cell types to_row emits."""

_logger = logging.getLogger("langchaint.response")


@dataclass(frozen=True, kw_only=True)
class Response[OutputT](_CallCarrier):
    """One successful generate result.

    output is the assistant text, or the response_format instance validated from the turn's text.
    It is None when the turn parsed no instance, which on a structured tool-bound binding means the model called tools.
    Only that binding types output optional, so no other caller has a None to handle.
    A turn can both parse an instance and call tools, setting output and tool_calls at once.
    tool_calls is therefore what says whether this turn owes the model a tool result.
    call is this call's history: model, provider_name, attempt_records, and elapsed_seconds read off it.
    attempts counts its records.
    Every attempt record but the last failed and the last succeeded.
    assistant_message is the adapter-built turn exactly as the provider produced it,
    the whole ordered turn (reasoning, text, and tool calls in emission order),
    held by reference for appending to a conversation.
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


@dataclass(frozen=True, kw_only=True)
class AbandonedCall(_CallCarrier):
    """The record of one generate call whose result never reached the caller, appended to its log.

    A cancelled call returns no value, and the CancelledError must propagate for the cancelling
    scope's teardown to run, so this record is the only carrier of the call's history:
    generate_one and stream_one append it to their abandoned_call_log before re-raising.
    A batch item is abandoned on the same terms when generate_many raises past it, whether the item
    was still running or had already settled its row into the list the raise discards.
    Its presence in the log is the count of abandoned calls.

    call is this call's history. Where the call was cut off mid-attempt, its attempt_records cover
    only the attempts that settled before that, and the in-flight attempt has no record:
    whatever it billed beyond what usage_in_flight states is unobservable client-side
    and none of it is fabricated ("unbilled" would overclaim).
    Reconciliation closes the gap from the provider's side, which model and provider_name identify.

    usage_in_flight is what the provider had reported for the cut-off attempt by the time the call
    was cut off, and is ZERO_USAGE wherever it had reported nothing, which is true of every
    non-stream call. A counter the provider sends late is missing from it.
    The two usage fields are disjoint and add, so total spend is at least
    usage_settled + usage_in_flight, with no branch to write.
    """

    # pyrefly: ignore[bad-override]  # read-only here, read-write on _CallCarrier; see its docstring
    call: CallRecord
    usage_in_flight: Usage

    @property
    def usage_settled(self) -> Usage:
        """The folded paid total of this call's settled attempts.

        Deliberately not named usage: on the other result carriers, usage is the call's whole paid
        total, and a call cut off mid-attempt lacks that attempt's share here, which
        usage_in_flight carries instead.
        Cancellation correlates with long requests, so the omitted attempt skews expensive, and
        answering to the name the whole-total carriers use would turn a loud AttributeError into a
        silent undercount.
        """
        return Usage.sum_of(record.usage for record in self.attempt_records)


class AbandonedCallLog(Protocol):
    """Where generate_one, generate_many, and stream_one record a call whose result never reached the caller.

    append is the one required method, so a list[AbandonedCall] satisfies it, and so does an
    application's own log of turn records whose element union includes AbandonedCall; total spend is
    then one fold over the log's rows. The log must be append-only: a conversation the application
    trims or rebuilds is the wrong target, because spend is monotone and a trim would shrink it.
    """

    def append(self, abandoned_call: AbandonedCall, /) -> None:
        """Receive one record."""


def _append_abandoned_call(
    abandoned_call_log: AbandonedCallLog | None,
    call: CallRecord,
    usage_in_flight: Usage = ZERO_USAGE,
) -> None:
    """Append one AbandonedCall when a log was given, without letting the append escape.

    usage_in_flight defaults to ZERO_USAGE, which is what a caller with no channel for
    observing an in-flight attempt states; only an open stream has one.
    Every caller runs this while an exception unwinds, after it has returned the in-flight slot
    and, on the stream path, closed the connection.
    A log whose append raises is a defect in application code, but raising it here would replace
    that exception, and where the exception is a CancelledError the substitution costs more than the
    lost record: a task that ends on any other exception is not cancelled, so asyncio.timeout reports
    that exception in place of TimeoutError and a TaskGroup's shutdown sees a substituted error.
    A failed append is logged and the exception already unwinding continues to propagate.
    """
    if abandoned_call_log is None:
        return
    try:
        abandoned_call_log.append(AbandonedCall(call=call, usage_in_flight=usage_in_flight))
    except Exception:
        _logger.warning(
            "abandoned_call_log.append raised; this call's record was lost", exc_info=True
        )


def to_row[OutputT](result: Response[OutputT] | GenerationError) -> dict[str, RowValue]:
    """One flat dict of scalars per result, for table building.

    A success and a failure fill the same keys, so a mixed list becomes one table:
    a failure's output is None and its error_text carries the failure reason a success leaves None.
    A success whose output is None writes None too, that being the tool-call turn of a structured
    tool-bound binding; stop_reason "tool_use" is what tells the two None cells apart.
    The cost and usage-counter columns are the call's paid totals across every attempt, uniform on
    success and failure rows (zero for a retry-exhausted item whose attempts billed nothing, the real
    values for a 200 that produced no output, and above the single answer's tokens when a billed 200
    was retried).
    Usage counters and per-category costs are hoisted to top-level keys named exactly like the Usage
    fields, with cost_in_usd their sum; model output is flattened to its JSON.
    assistant_message_json is the generated turn on either kind of result, None where no attempt
    produced one, so a failure row shows what the provider generated the same way a success row does.
    It is not the output cell: output means the parsed result, which a failure has none of.
    """
    assistant_message = result.assistant_message
    if isinstance(result, GenerationError):
        output_cell: str | None = None
        error_text: str | None = result.error_text
        stop_reason: StopReason | None = result.stop_reason
        usage = result.usage
    else:
        output = result.output
        if output is None:
            output_cell = None
        elif isinstance(output, BaseModel):
            output_cell = output.model_dump_json()
        else:
            output_cell = str(output)
        error_text = None
        stop_reason = result.stop_reason
        usage = result.usage
    return {
        "output": output_cell,
        "assistant_message_json": (
            None if assistant_message is None else assistant_message.model_dump_json()
        ),
        "error_text": error_text,
        "stop_reason": stop_reason,
        "model": result.model,
        "provider_name": result.provider_name,
        "attempts": result.attempts,
        "elapsed_seconds": result.elapsed_seconds,
        "cost_in_usd": usage.cost_in_usd,
        "input_tokens_cache_read": usage.input_tokens_cache_read,
        "input_tokens_cache_read_cost_in_usd": usage.input_tokens_cache_read_cost_in_usd,
        "input_tokens_cache_write": usage.input_tokens_cache_write,
        "input_tokens_cache_write_cost_in_usd": usage.input_tokens_cache_write_cost_in_usd,
        "input_tokens_cache_none": usage.input_tokens_cache_none,
        "input_tokens_cache_none_cost_in_usd": usage.input_tokens_cache_none_cost_in_usd,
        "input_tokens_total": usage.input_tokens_total,
        "output_tokens": usage.output_tokens,
        "output_tokens_cost_in_usd": usage.output_tokens_cost_in_usd,
        "output_tokens_reasoning": usage.output_tokens_reasoning,
    }
