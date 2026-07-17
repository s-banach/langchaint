"""The generate results: the success Response and the terminal GenerationError.

A generate that succeeds returns a Response; one that ends terminally (retries exhausted on transient errors, a refusal,
or a truncation at the token cap) raises or returns a GenerationError.
They share the per-attempt history the package-owned retry loop produces,
because that loop has no callback seam and the history survives only if the result carries it:
attempt_records is that history, one record per request sent.
On a Response every record but the last failed and the last succeeded;
on a GenerationError the records describe the terminal outcome.
attempts is derived from the records on both.
to_row flattens either result to one dict of scalars with the same keys,
so a mixed list of successes and failures converts directly to a table.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from langchaint.exceptions import AttemptRecord, GenerationError
from langchaint.messages import AssistantMessage, StopReason, ToolCall
from langchaint.usage import ZERO_USAGE, Usage

type RowValue = str | int | float | bool | None
"""The scalar cell types to_row emits."""


@dataclass(frozen=True, kw_only=True)
class Response[OutputT]:
    """One successful generate result.

    output is the assistant text, or the SDK-parsed response_format instance.
    attempt_records holds one AttemptRecord per request sent, in order;
    every record but the last failed and the last succeeded, and attempts is derived from the records.
    assistant_message is the adapter-built turn exactly as the provider produced it,
    the whole ordered turn (reasoning, text, and tool calls in emission order),
    held by reference for appending to a conversation.
    Rebuilding it from output and tool_calls is lossy (it drops reasoning and the element order)
    and is the rewrap this field exists to prevent.
    raw is the SDK's own response model, held by reference (no dump, no copy; call raw.model_dump() for a dict);
    on streams it comes from the SDK-assembled final message.
    It is a live, mutable pydantic object shared with the adapter, so despite the frozen dataclass around it,
    treat it read-only and raw.model_copy() before mutating.
    usage and usage_successful_attempt are two scopes, both folded from attempt_records (see their docstrings):
    usage is the paid total across every attempt, usage_successful_attempt the single kept answer's own.
    elapsed_seconds spans first request to completion, RateLimiter slot waits and backoff waits included;
    it is stored rather than derived from the records because the records deliberately exclude those waits.
    """

    output: OutputT
    model: str
    provider_name: str
    attempt_records: tuple[AttemptRecord, ...]
    elapsed_seconds: float
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
        """Requests actually sent: one attempt record each."""
        return len(self.attempt_records)

    @property
    def usage(self) -> Usage:
        """The paid total across every attempt of the call, carrying cost_in_usd, the number to bill on.

        A call that retried a billed 200 (an empty structured parse retried as transient) counts every such
        attempt, so this can exceed the tokens of the single answer in output; usage_successful_attempt is
        that single answer's own usage. This is the same paid-total scope as GenerationError.usage,
        so the two mean the same thing. Transport, 5xx, and rate-limit retries bill nothing (ZERO_USAGE),
        so when every failed attempt was one of those this equals usage_successful_attempt.
        """
        return sum((record.usage for record in self.attempt_records), start=ZERO_USAGE)

    @property
    def usage_successful_attempt(self) -> Usage:
        """The single kept answer's own usage, the one matching output, assistant_message, and raw.usage.

        The last attempt record is the success (__post_init__ enforces it), so this reads it directly.
        It equals usage in the common case where no failed attempt billed (every retry was a transport,
        5xx, or rate-limit failure); it is smaller than usage only when a billed 200 was retried.
        """
        return self.attempt_records[-1].usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The turn's tool calls, from assistant_message."""
        return self.assistant_message.tool_calls


def to_row[OutputT](result: Response[OutputT] | GenerationError) -> dict[str, RowValue]:
    """One flat dict of scalars per result, for table building.

    A success and a failure fill the same keys, so a mixed list becomes one table:
    a failure's output is None and its error_text carries the failure reason a success leaves None.
    The cost_in_usd and usage-counter columns are the call's paid totals across every attempt, uniform on
    success and failure rows (zero for a retry-exhausted item whose attempts billed nothing, the real values
    for a refusal or truncation, and above the single answer's tokens when a billed 200 was retried).
    Usage counters are hoisted to top-level keys named exactly like the Usage fields;
    model output is flattened to its JSON.
    """
    if isinstance(result, GenerationError):
        output_cell: str | None = None
        error_text: str | None = result.error_text
        stop_reason: StopReason | None = result.stop_reason
        usage = result.usage
    else:
        output = result.output
        output_cell = (
            output.model_dump_json() if isinstance(output, BaseModel) else str(output)
        )
        error_text = None
        stop_reason = result.stop_reason
        usage = result.usage
    return {
        "output": output_cell,
        "error_text": error_text,
        "stop_reason": stop_reason,
        "model": result.model,
        "provider_name": result.provider_name,
        "attempts": result.attempts,
        "elapsed_seconds": result.elapsed_seconds,
        "cost_in_usd": usage.cost_in_usd,
        "input_tokens_cache_read": usage.input_tokens_cache_read,
        "input_tokens_cache_write": usage.input_tokens_cache_write,
        "input_tokens_cache_none": usage.input_tokens_cache_none,
        "input_tokens_total": usage.input_tokens_total,
        "output_tokens": usage.output_tokens,
        "output_tokens_reasoning": usage.output_tokens_reasoning,
    }
