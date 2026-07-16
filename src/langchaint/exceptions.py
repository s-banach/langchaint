"""Exception vocabulary.

Two orthogonal properties decide an error's fate, and this module keeps them separate:

- Retry axis (read by the retry loop in llm.py): a TransientError is retried, everything else is not.
  No NonRetriableError class exists; "non-retriable" simply means "not a TransientError".
- Batch axis (read by generate_many): an AbortBatchError cancels the sibling requests,
  while a GenerationError becomes one item's failure row.
  AbortBatchError and the GenerationError leaves are all non-retriable; they differ only on this batch axis.

TransientError and AbortBatchError are per-attempt / control signals.
The GenerationError leaves are terminal per-item results a to_row failure row is built from:
RetriesExhaustedError, RefusalError, and ExceededMaxCompletionTokensError.

Classification of raw SDK exceptions into these lives in the provider adapter (Provider.classify);
a refusal and a token-cap truncation are normal 200 responses that never reach classify,
so the adapter detects them where it reads the response and raises the matching leaf directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, override

from pydantic import ValidationError

from langchaint.messages import StopReason
from langchaint.usage import Usage


class TransientError(Exception):
    """One failed attempt that a retry may fix.

    __cause__ holds the original provider exception when one exists.
    retry_after_seconds is the server-stated wait parsed from the response's retry-after headers,
    when the provider sent one;
    RateLimiter honors it up to a 60-second cap and uses it to pause admission account-wide.
    is_rate_limit marks errors saying the account or service refuses further requests right now
    (Provider.classify returned "rate_limit");
    RateLimiter pauses admission on them and requires a successful probe request before resuming full admission.
    usage, cost_in_usd, and stop_reason describe the attempt's billable completion
    when the failing attempt was a completed 200 the adapter rejected downstream
    (a structured parse that returned no output);
    they are None for a transport failure (timeout, 5xx, connection or rate-limit error) that billed nothing.
    The retry loop copies them onto the attempt's AttemptRecord.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        is_rate_limit: bool = False,
        usage: Usage | None = None,
        cost_in_usd: float | None = None,
        stop_reason: StopReason | None = None,
    ) -> None:
        """Store the server-stated wait, the rate-limit classification, and any attempt billing."""
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.is_rate_limit = is_rate_limit
        self.usage = usage
        self.cost_in_usd = cost_in_usd
        self.stop_reason = stop_reason


class AbortBatchError(Exception):
    """A non-retriable error that cancels the whole batch instead of becoming one item's row.

    generate_many cancels the in-flight siblings and re-raises this, because the batch is misconfigured, not unlucky;
    generate_one just propagates it.
    It is deliberately not a GenerationError: those are per-item rows, this aborts every item.
    Examples: bad credentials, an invalid request, an ImagePart media_type outside the API's set,
    a response with 1-hour cache writes but no cache_write_1h_usd_per_million_tokens, and the classify default.
    Adapters must classify unrecognized exceptions as abort so bugs are not retried silently.
    """


@dataclass(frozen=True, kw_only=True)
class AttemptRecord:
    """One request sent inside the retry loop, success or failure.

    started_at_monotonic_seconds and ended_at_monotonic_seconds are raw time.monotonic() readings:
    only differences are meaningful, and only within one process.
    The package defines no time origin because it does not own the enclosing loop;
    subtract whatever origin the caller holds (an agent-loop start, another record)
    to place records on a shared timeline.
    The bracket spans the request itself and excludes RateLimiter slot waits and backoff sleeps,
    so a slow request is distinguishable from time spent rate limited;
    the gap between consecutive records is that wait.
    On a stream the succeeding record spans opening the stream to its exhaustion, because that is the whole request.
    error is None on the attempt that succeeded and on a completed 200 rejected downstream
    (a refusal or a truncation, which are not transient);
    it holds the TransientError otherwise.
    usage and cost_in_usd are the attempt's billing: set when the attempt reached a billable 200
    (a success, or a rejected 200), None for a transport failure that billed nothing.
    """

    started_at_monotonic_seconds: float
    ended_at_monotonic_seconds: float
    error: TransientError | None
    usage: Usage | None
    cost_in_usd: float | None

    @property
    def elapsed_seconds(self) -> float:
        """ended_at_monotonic_seconds minus started_at_monotonic_seconds."""
        return self.ended_at_monotonic_seconds - self.started_at_monotonic_seconds


def _extract_transient_errors(attempt_records: Sequence[AttemptRecord]) -> tuple[TransientError, ...]:
    """Return the errors of the failed attempts, in order.

    The fold RetriesExhaustedError and RateLimiter.delay_seconds consume;
    on a failure this is every record's error, on a success all but the last.
    """
    return tuple(
        record.error for record in attempt_records if record.error is not None
    )


def _join_error_text(attempt_records: Sequence[AttemptRecord]) -> str:
    """Flatten the failure chain to one line, numbering each attempt."""
    return "; ".join(
        f"attempt {index + 1}: {record.error}"
        for index, record in enumerate(attempt_records)
    )


def _sum_usage(attempt_records: Sequence[AttemptRecord]) -> Usage:
    """Sum the billable usage across records; a record that billed nothing adds zero.

    input_tokens_total_provider_reported stays None:
    the sum spans several requests and no single provider-reported total covers it.
    """
    usages = [record.usage for record in attempt_records if record.usage is not None]
    return Usage(
        input_tokens_cache_read=sum(usage.input_tokens_cache_read for usage in usages),
        input_tokens_cache_write=sum(usage.input_tokens_cache_write for usage in usages),
        input_tokens_cache_none=sum(usage.input_tokens_cache_none for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
    )


def _sum_cost_in_usd(attempt_records: Sequence[AttemptRecord]) -> float:
    """Sum the billed cost across records; a record that billed nothing adds zero."""
    return sum(
        (record.cost_in_usd for record in attempt_records if record.cost_in_usd is not None),
        0.0,
    )


class GenerationError(Exception):
    """A terminal per-item generate result that becomes a to_row failure row.

    The base for the three non-retriable per-item outcomes:
    RetriesExhaustedError (the retry budget ran out on transient errors),
    RefusalError (the model refused on the structured path), and
    ExceededMaxCompletionTokensError (the structured response hit the token cap before its JSON parsed).
    generate_one raises any of them;
    generate_many returns each in the slot of the item it belongs to,
    so to_row renders a uniform failure row and siblings keep running.

    attempt_records holds one AttemptRecord per request sent;
    model, provider_name, elapsed_seconds, and stop_reason mirror the fields a success Response carries
    so to_row fills the same row shape from either.
    usage and cost_in_usd are summed from the records (a refusal or truncation reads its one completed attempt;
    a retry-exhausted item sums its records, near zero when they were transport failures);
    attempts and error_text are derived from the records too.

    The adapter that detects a refusal or truncation cannot know the loop's prior attempts or timing, so
    it raises the leaf through for_rejected_200 carrying only the one attempt's billing;
    the retry loop records that attempt and re-raises the enriched leaf via the normal constructor.
    """

    attempt_records: tuple[AttemptRecord, ...]
    model: str
    provider_name: str
    elapsed_seconds: float
    stop_reason: StopReason | None
    usage: Usage
    cost_in_usd: float

    def __init__(
        self,
        *,
        attempt_records: tuple[AttemptRecord, ...],
        model: str,
        provider_name: str,
        elapsed_seconds: float,
        stop_reason: StopReason | None,
    ) -> None:
        """Fill the row-shape fields; usage and cost_in_usd are summed from the records."""
        self.attempt_records = attempt_records
        self.model = model
        self.provider_name = provider_name
        self.elapsed_seconds = elapsed_seconds
        self.stop_reason = stop_reason
        self.usage = _sum_usage(attempt_records)
        self.cost_in_usd = _sum_cost_in_usd(attempt_records)
        super().__init__(self._summary())

    @classmethod
    def for_rejected_200(
        cls, *, usage: Usage, cost_in_usd: float, stop_reason: StopReason
    ) -> Self:
        """Adapter-side leaf carrying one rejected 200's billing, before the loop fills the row.

        The retry loop catches it, records the attempt from usage and cost_in_usd,
        and re-raises the enriched leaf through the normal constructor;
        no caller ever sees this partial object,
        so its row-shape fields (attempt_records, model, provider_name, elapsed_seconds) stay unset.
        """
        error = cls.__new__(cls)
        error.usage = usage
        error.cost_in_usd = cost_in_usd
        error.stop_reason = stop_reason
        Exception.__init__(error, cls._summary(error))
        return error

    def _summary(self) -> str:
        """Return the exception message; leaves override this with their own reason."""
        return "generation failed"

    @property
    def attempts(self) -> int:
        """Requests actually sent: one attempt record each."""
        return len(self.attempt_records)

    @property
    def error_text(self) -> str:
        """The failure-row error cell; RetriesExhaustedError folds its attempt chain instead."""
        return str(self)


class RetriesExhaustedError(GenerationError):
    """Every attempt failed with a transient error, and the budget ran out.

    generate_one raises it;
    generate_many returns it in the row where an item exhausted its retries,
    so the same object is both the raised failure and the failure row of a batch.
    stop_reason is None: no attempt reached a completed turn to report one.
    errors_from_attempts is derived from attempt_records.
    """

    @override
    def _summary(self) -> str:
        """Name the failed attempts and quote the last error."""
        errors = _extract_transient_errors(self.attempt_records)
        last = str(errors[-1]) if errors else "no attempts recorded"
        return f"{len(errors)} attempts failed; last: {last}"

    @property
    def errors_from_attempts(self) -> tuple[TransientError, ...]:
        """The failed attempts' errors, in order."""
        return _extract_transient_errors(self.attempt_records)

    @property
    @override
    def error_text(self) -> str:
        """The folded failure chain, one entry per attempt."""
        return _join_error_text(self.attempt_records)


class RefusalError(GenerationError):
    """The model refused to produce structured output.

    Fires only on the structured path, where the refusal left no instance to return;
    the text path surfaces a refusal as a Response with stop_reason "refusal".
    Not retried, by policy:
    a refusal can flip under sampling,
    but retrying spends the full input tokens
    (cache-read rate when warm, never zero) on an expected-value bet the library does not take by default.
    An app whose economics differ overrides the adapter's _parsed_output.
    stop_reason is "refusal".
    """

    @override
    def _summary(self) -> str:
        """State the refusal."""
        return "the model refused to produce structured output"


class ExceededMaxCompletionTokensError(GenerationError):
    """The structured response reached max_completion_tokens before its JSON parsed.

    Fires only on the structured path; the text path surfaces the cap as a Response with stop_reason "max_tokens".
    Not retried, unconditionally:
    the attempt already generated the full token cap,
    the most expensive possible response, and a resample under the same cap truncates again.
    The fix is a larger max_completion_tokens via rebind.
    stop_reason is "max_tokens".
    """

    @override
    def _summary(self) -> str:
        """State the truncation."""
        return "the structured response reached max_completion_tokens before its JSON parsed"


class InvalidToolArgsError(Exception):
    """A tool call's args_json failed validation against the tool's args_model.

    Raised only from Tool.validate_and_run's validation step, never from the function,
    so catching it cannot swallow a function defect.
    This is model data the model can correct:
    ToolManager.dispatch catches it and returns a DispatchInvalidToolArgs
    holding the ValidationError and an is_error ToolMessage.
    A tool function must not raise it:
    dispatch's catch spans the whole validate_and_run call,
    so a function raising it is classified as bad model args, not as a defect.
    """

    def __init__(self, validation_error: ValidationError) -> None:
        """Hold the ValidationError by reference; __str__ derives the message from it."""
        super().__init__()
        self.validation_error = validation_error

    @override
    def __str__(self) -> str:
        """Render the held ValidationError as its own multi-line string."""
        return str(self.validation_error)


class StreamProtocolError(Exception):
    """A provider stream violated the event contract.

    Example: the stream ended without a stop event.
    """
