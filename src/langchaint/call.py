"""The per-call history: one attempt's record, the frozen CallRecord, the ledger, and the mixin.

A call is one generate or one stream, from the moment its retry loop begins to the outcome that ends it.
An attempt is one request inside that call. The adapter runs one attempt and reports what came back;
only a retry loop sees the whole call, so only a retry loop builds a CallRecord.
A loop accumulates its call into a _CallLedger and freezes that into the CallRecord it hands out.

Response, GenerationError, and AbandonedCall each hold one, and each derives _CallCarrier, so the
call-level field set is declared once and every carrier answers to the same names.
Imports no error class at runtime: a success carries a CallRecord, so a dependency on the error
vocabulary would run the wrong way.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from langchaint.usage import Usage

if TYPE_CHECKING:
    # Type-only: exceptions.py imports this module at runtime, because GenerationError derives
    # _CallCarrier, so importing TransientError here at runtime would be a cycle.
    # AttemptRecord.error quotes it, and nothing here constructs one: the retry loops and the
    # adapters do.
    from langchaint.exceptions import TransientError


@dataclass(frozen=True, kw_only=True)
class AttemptRecord:
    """One request langchaint observed going out, success or failure.

    started_at_monotonic_seconds and ended_at_monotonic_seconds are raw time.monotonic() readings:
    only differences are meaningful, and only within one process.
    langchaint defines no time origin because it does not own the enclosing loop;
    subtract whatever origin the caller holds (an agent-loop start, another record)
    to place records on a shared timeline.
    The bracket spans the request itself and excludes RateLimiter slot waits and backoff sleeps,
    so a slow request is distinguishable from time spent rate limited;
    the gap between consecutive records is that wait.
    On a stream the succeeding record spans opening the stream to its exhaustion, because that is the whole request.
    error is None on the attempt that succeeded, on a 200 that produced no output and is not
    retried (a refusal, a truncation, a context-window overflow), and on a request the provider
    rejected; it holds the TransientError otherwise.
    usage is the attempt's billing (with cost_in_usd inside): the reported counts when the attempt reached a
    billable 200, whether or not it produced output, ZERO_USAGE for a transport failure or a rejected request.
    A stream that dropped after delivering items was paid for what it delivered,
    and no client-side channel reports the amount.
    usage_raw is the raw SDK usage object usage was normalized from.
    It is None when no usage came back: a transport failure, a rejected request, or an openai 200 reporting no usage.
    """

    started_at_monotonic_seconds: float
    ended_at_monotonic_seconds: float
    error: "TransientError | None"
    usage: Usage
    usage_raw: BaseModel | None

    @property
    def elapsed_seconds(self) -> float:
        """The bracket's length."""
        return self.ended_at_monotonic_seconds - self.started_at_monotonic_seconds


@dataclass(frozen=True, kw_only=True)
class CallRecord:
    """What one call did: every attempt it made, what served them, and how long it took.

    attempt_records holds the call's attempt records, in order.
    Two attempts have no record: the one an UnrecognizedError ends the call on, whose error the
    adapter could not read, and the one in flight when a cancellation cut the call off.
    An InvalidRequestError built from a InvalidRequest outcome has none either, because nothing went out.
    elapsed_seconds spans the call's start to the stamp it was frozen at, RateLimiter slot waits
    and backoff sleeps included;
    it is stored rather than folded from the records, because the records deliberately exclude those waits.
    model and provider_name name what served the call, which is what a caller reconciling spend
    against the provider's own billing asks about.

    No usage field: each carrier folds the records under its own name, because AbandonedCall's fold
    structurally lacks the in-flight attempt's share and must not answer to the name the carriers
    reporting a whole paid total use.
    """

    model: str
    provider_name: str
    attempt_records: tuple[AttemptRecord, ...]
    elapsed_seconds: float


class _CallCarrier:
    """Forwards a held CallRecord's fields, so every result carrier reads the same names.

    A deriving dataclass declares `call: CallRecord` itself: this class is not a dataclass, so its
    annotations are not fields, and without the declaration the subclass gets no such parameter.
    A frozen dataclass's re-declaration is read-only where this one is read-write, which pyrefly
    reports as an inconsistent override; the two frozen carriers suppress it. Declaring `call` here
    as a read-only property instead would break at runtime: a property is a data descriptor, so it
    would shadow the instance attribute every deriving class sets.

    No attempts count is forwarded: Response and GenerationError each define their own, and AbandonedCall reports none.

    CallRecord is the one declaration of the names it forwards, so a carrier adds none of them itself.
    """

    call: CallRecord

    @property
    def model(self) -> str:
        """The model id the call was sent to."""
        return self.call.model

    @property
    def provider_name(self) -> str:
        """The provider that served the call."""
        return self.call.provider_name

    @property
    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        """The call's attempt records, in order."""
        return self.call.attempt_records

    @property
    def elapsed_seconds(self) -> float:
        """The call's wall time, slot waits and backoff sleeps included."""
        return self.call.elapsed_seconds


class _CallLedger:
    """The mutable history one call accumulates; freeze() is its CallRecord.

    The loop's owner constructs the ledger and keeps the reference, so the settled attempts survive
    a cancellation that unwinds the loop's frame: the ledger is the only channel that path has.
    Records are appended between awaits, so it never holds a partial one.
    freeze_ending_at is the one place a call's elapsed_seconds is computed.
    """

    def __init__(self, *, model: str, provider_name: str) -> None:
        """Open a ledger against what will serve the call, stamping its start as now."""
        self._model = model
        self._provider_name = provider_name
        self._attempt_records: list[AttemptRecord] = []
        self._started_at_monotonic_seconds = time.monotonic()
        self._attempt_started_at_monotonic_seconds = self._started_at_monotonic_seconds

    def start_call(self) -> None:
        """Stamp the call's start as now, replacing the constructor's stamp.

        Each retry loop calls this as its first act, because an owner may build the ledger before
        the call begins: StreamHandle is built by stream_one and opens its request on __aenter__,
        and the interval between the two belongs to the caller, not to the call.
        """
        self._started_at_monotonic_seconds = time.monotonic()

    def start_attempt(self) -> None:
        """Stamp the attempt's start as now; the next record closes the bracket it opens."""
        self._attempt_started_at_monotonic_seconds = time.monotonic()

    @property
    def attempts(self) -> int:
        """Records so far, which is what a retry budget counts."""
        return len(self._attempt_records)

    @property
    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        """The records so far, in order."""
        return tuple(self._attempt_records)

    def record(
        self, *, error: "TransientError | None", usage: Usage, usage_raw: BaseModel | None
    ) -> None:
        """Close the attempt started by the last start_attempt(), ending it now."""
        self.record_ending_at(time.monotonic(), error=error, usage=usage, usage_raw=usage_raw)

    def record_ending_at(
        self,
        ended_at_monotonic_seconds: float,
        *,
        error: "TransientError | None",
        usage: Usage,
        usage_raw: BaseModel | None,
    ) -> None:
        """Close the attempt started by the last start_attempt(), at a stamp taken earlier.

        A stream's attempt ends when its item iterator exhausts, several awaits before final() reads
        the assembled outcome, so the stream path stamps the end itself.
        """
        self._attempt_records.append(
            AttemptRecord(
                started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                error=error,
                usage=usage,
                usage_raw=usage_raw,
            )
        )

    def freeze(self) -> CallRecord:
        """Return the call's history as of now, for a result to carry."""
        return self.freeze_ending_at(time.monotonic())

    def freeze_ending_at(self, ended_at_monotonic_seconds: float) -> CallRecord:
        """Return the call's history as of a stamp taken earlier.

        A stream's call ends when its item iterator exhausts; final() may be awaited any time after,
        and that gap is the caller's own work, not the call's.
        """
        return CallRecord(
            model=self._model,
            provider_name=self._provider_name,
            attempt_records=tuple(self._attempt_records),
            elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
        )
