"""The per-call history: one attempt's record, the frozen CallRecord, the ledger, and the mixin.

A call is one generate or one stream, from the moment its retry loop begins to the outcome that ends it.
An attempt is one request inside that call. The adapter runs one attempt and reports what came back;
only a retry loop sees the whole call, so only a retry loop builds a CallRecord.
A loop accumulates its call into a _CallLedger and freezes that into the CallRecord it hands out.

Every success variant and GenerationError hold one, and each derives _CallCarrier, so the call-level field
set is declared once and every carrier answers to the same names.
Imports no error class at runtime: a success carries a CallRecord, so a dependency on the error
vocabulary would run the wrong way.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel

from langchaint.messages import AssistantMessage
from langchaint.pricing import Billing
from langchaint.usage import ZERO_USAGE, Usage

if TYPE_CHECKING:
    # Type-only: exceptions.py imports this module at runtime, because GenerationError derives
    # _CallCarrier, so importing TransientError here at runtime would be a cycle.
    # AttemptRecord.error quotes it, and nothing here constructs one: the retry loops and the
    # adapters do.
    from langchaint.exceptions import TransientError


class ResponseIdentity(NamedTuple):
    """What one response says about itself: its ids and the model that served it.

    Returned by BoundAdapter.identity_from_raw, and the source of the AttemptRecord fields of the
    same names.
    """

    model_served: str
    response_id: str
    request_id: str | None

    def with_request_id_fallback(self, request_id: str | None) -> "ResponseIdentity":
        """Fill request_id from the open stream when the response itself carries none.

        A response the SDK assembled from stream events need not carry the request-id header its
        HTTP response did; the stream is what still has it. A request_id the response does carry
        wins, being the header of the request that came back.
        """
        if self.request_id is not None:
            return self
        return self._replace(request_id=request_id)


@dataclass(frozen=True, kw_only=True)
class AttemptRecord:
    """One attempt langchaint can account for, whether or not its request went out.

    started_at_monotonic_seconds and ended_at_monotonic_seconds are raw time.monotonic() readings:
    only differences are meaningful, and only within one process.
    langchaint defines no time origin because it does not own the enclosing loop;
    subtract whatever origin the caller holds (an agent-loop start, another record)
    to place records on a shared timeline.
    The bracket spans the request itself and excludes SharedBackoff admission waits and backoff sleeps,
    so a slow request is distinguishable from time spent rate limited;
    the gap between consecutive records is that wait.
    A succeeding generate record spans opening the request's stream through reading its assembled response.
    A succeeding stream_one record ends when its item iterator exhausts, because final() waits on the caller.
    first_item_at_monotonic_seconds is langchaint's own stamp of the moment this attempt's stream
    yielded its first item to the caller, on the same clock as the two bracket stamps. Only
    stream_one yields items to a caller, so it is None on every generate attempt and on a stream
    that yielded nothing. Any StreamItem stamps it, a ReasoningDelta, a ToolCallDelta, or a
    ToolCall as much as a text chunk.
    error is None on the attempt that succeeded, on a 200 that produced no output and is not
    retried (a refusal, a truncation, a context-window overflow), on a request the provider
    rejected, on an error response the provider declared final, and on an attempt ended by an
    exception the adapter could not place; it holds the TransientError otherwise.
    billing is what the attempt billed: the counters the provider reported, priced when they
    arrived and before the response was interpreted, so a 200 is accounted for whatever
    langchaint went on to make of it, with the prices that applied pinned on.
    None where no response arrived and nothing was reported in flight, which keeps "the provider
    reported no billing" distinct from "the provider billed zero".
    A stream that dropped carries what the provider had reported by then, which a counter the
    provider sends late is missing from.
    raw is the SDK's own response object, held by reference (no dump, no copy).
    It is None exactly where no response arrived: a transport failure, an error status, or a request
    the adapter would not send. It is a live, mutable pydantic object, so despite the frozen
    dataclass around it, treat it read-only and raw.model_copy() before mutating.
    model_served and response_id are what the response reported about itself, read by the adapter
    the moment it arrived. Both are None exactly where raw is None. model_served is the model the
    provider reported serving the request; CallRecord.model beside it is the id langchaint sent.
    response_id is the provider's own id for the response, which a caller disputing a charge is
    asked for.
    request_id is the request-id header, the id provider support asks for. It reaches a record by
    three routes the two above do not have: the response, the SDK error on an attempt that received
    no response, and the open stream, which is what carries the header on a streamed attempt. None
    where none of the three had one.
    assistant_message is the turn that response carried, whatever langchaint made of it: the answer
    on the attempt that succeeded, and the refusal, the rejected text, or the fragment on a 200 that
    produced no output. It is None where no response arrived, and None where one arrived and reading
    it raised, because then no turn was ever built.
    """

    started_at_monotonic_seconds: float
    ended_at_monotonic_seconds: float
    first_item_at_monotonic_seconds: float | None
    error: "TransientError | None"
    billing: Billing | None
    assistant_message: AssistantMessage | None
    raw: BaseModel | None
    model_served: str | None
    response_id: str | None
    request_id: str | None

    @property
    def elapsed_seconds(self) -> float:
        """The bracket's length."""
        return self.ended_at_monotonic_seconds - self.started_at_monotonic_seconds

    @property
    def usage(self) -> Usage:
        """The billed counters and costs in the neutral summary, ZERO_USAGE where none were reported.

        Never None, so a fold over attempts adds a shape rather than testing for one.
        """
        return ZERO_USAGE if self.billing is None else self.billing.usage


@dataclass(frozen=True, kw_only=True)
class CallRecord:
    """What one call did: every attempt it made, what served them, and how long it took.

    attempt_records holds the call's attempt records, in order.
    Two attempts have no record: the one in flight when a cancellation cut the call off,
    and the one an UnknownExceptionError ends the call on where the stream never opened.
    An InvalidRequestError built from an InvalidRequest outcome has none either,
    because nothing went out.
    started_at_monotonic_seconds is a raw time.monotonic() reading, meaningful only as a difference
    and only within one process, as on AttemptRecord. It is the origin an attempt record's start is
    read against.
    elapsed_seconds spans that stamp to the stamp the call was frozen at, SharedBackoff admission
    waits and backoff sleeps included;
    it is stored rather than folded from the records, because the records deliberately exclude those waits.
    model and provider_name name what served the call, which is what a caller reconciling spend
    against the provider's own billing asks about.

    No usage field: each carrier folds the records under its own name, because AbandonedCallError
    adds to that fold what the attempt cut off mid-flight had billed, which no record holds.
    """

    model: str
    provider_name: str
    attempt_records: tuple[AttemptRecord, ...]
    started_at_monotonic_seconds: float
    elapsed_seconds: float


class _CallCarrier:
    """Forwards a held CallRecord's fields, so every result carrier reads the same names.

    A deriving dataclass declares `call: CallRecord` itself: this class is not a dataclass, so its
    annotations are not fields, and without the declaration the subclass gets no such parameter.
    A frozen dataclass's re-declaration is read-only where this one is read-write, which pyrefly
    reports as an inconsistent override; the frozen carriers suppress it. Declaring `call` here
    as a read-only property instead would break at runtime: a property is a data descriptor, so it
    would shadow the instance attribute every deriving class sets.

    No attempts count is forwarded: _SuccessCarrier and GenerationError each define their own.

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
    def started_at_monotonic_seconds(self) -> float:
        """The call's start, the origin its attempt records' stamps are read against."""
        return self.call.started_at_monotonic_seconds

    @property
    def elapsed_seconds(self) -> float:
        """The call's wall time, permit waits and backoff sleeps included."""
        return self.call.elapsed_seconds


class _StagedResponse(NamedTuple):
    """One arrived response, what it billed, and its ids, held until the attempt around it closes."""

    raw: BaseModel
    billing: Billing
    identity: ResponseIdentity


class _CallLedger:
    """The mutable history one call accumulates; freeze() is its CallRecord.

    The loop's owner constructs the ledger and keeps the reference, so the settled attempts survive
    a cancellation that unwinds the loop's frame: the ledger is the only channel that path has.
    Records are appended between awaits, so it never holds a partial one.
    freeze_ending_at is the one place a call's elapsed_seconds is computed.

    An attempt is written in two steps, stage then close: stage_response the moment a response
    arrives, record or freeze once its fate is decided. Every attempt that received a response
    appears in the records, whatever happens between those two steps.

    start_attempt and record_ending_at also open and close the attempt-in-flight state that
    in_flight_attempt_started_at_monotonic_seconds reports, which is what a call cut off by a
    cancellation carries in place of the record it never got.
    """

    def __init__(self, *, model: str, provider_name: str) -> None:
        """Open a ledger against what will serve the call, stamping its start as now."""
        self._model = model
        self._provider_name = provider_name
        self._attempt_records: list[AttemptRecord] = []
        self._staged_response: _StagedResponse | None = None
        self._started_at_monotonic_seconds = time.monotonic()
        self._attempt_started_at_monotonic_seconds = self._started_at_monotonic_seconds
        self._attempt_in_flight = False
        self._first_item_at_monotonic_seconds: float | None = None
        self._noted_request_id: str | None = None
        self._billing_in_flight: Billing | None = None

    def stage_response(
        self, *, raw: BaseModel, billing: Billing, identity: ResponseIdentity
    ) -> None:
        """Hold the response that just arrived, what it billed, and what it says about itself.

        Call before interpreting the response. The next record or freeze closes the attempt around
        what is staged here, so no raise in between can lose the attempt, its billing, or its ids.
        """
        self._staged_response = _StagedResponse(raw=raw, billing=billing, identity=identity)

    def start_call(self) -> None:
        """Stamp the call's start as now, replacing the constructor's stamp.

        Each retry loop calls this as its first act, because an owner may build the ledger before
        the call begins: StreamHandle is built by stream_one and opens its request on __aenter__,
        and the interval between the two belongs to the caller, not to the call.
        """
        self._started_at_monotonic_seconds = time.monotonic()

    def start_attempt(self) -> None:
        """Stamp the attempt's start as now; the next record closes the bracket it opens.

        Clears the first-item stamp, the noted request id, and the noted in-flight billing too, so
        everything a record carries is the attempt's own.
        """
        self._attempt_started_at_monotonic_seconds = time.monotonic()
        self._attempt_in_flight = True
        self._first_item_at_monotonic_seconds = None
        self._noted_request_id = None
        self._billing_in_flight = None

    def stamp_first_item(self) -> None:
        """Stamp the attempt's first stream item as now, ignoring every item after it."""
        if self._first_item_at_monotonic_seconds is None:
            self._first_item_at_monotonic_seconds = time.monotonic()

    def note_request_id(self, request_id: str | None) -> None:
        """Hold the request id of the attempt in flight.

        Call where the retry loop first sees an exception, so every record closing that attempt
        carries the id whichever path closes it. A staged response's own id wins, being the header
        of the request that came back.
        """
        self._noted_request_id = request_id

    def note_billing_in_flight(self, billing: Billing | None) -> None:
        """Hold what the provider had reported for the attempt in flight when it was cut off.

        Call where a BaseException cuts an attempt's stream off, so a deadline account built after
        the frame unwinds still reports what that attempt had billed.
        The stash lands exactly once: record clears it when it settles the attempt (the record's
        own billing then states it), and start_attempt clears it with the other per-attempt notes.
        """
        self._billing_in_flight = billing

    @property
    def billing_in_flight(self) -> Billing | None:
        """The noted in-flight billing, None once a record settles the attempt it belonged to."""
        return self._billing_in_flight

    @property
    def attempts(self) -> int:
        """Records so far, which is what a retry budget counts."""
        return len(self._attempt_records)

    @property
    def attempt_records(self) -> tuple[AttemptRecord, ...]:
        """The records so far, in order."""
        return tuple(self._attempt_records)

    @property
    def in_flight_attempt_started_at_monotonic_seconds(self) -> float | None:
        """When the open attempt started, None between attempts and before the first.

        An attempt is open from start_attempt until the record that closes it, so a call cut off
        while one was open reports that a request had gone out and when, which is the account no
        AttemptRecord can give: its ending was never observed.
        """
        return self._attempt_started_at_monotonic_seconds if self._attempt_in_flight else None

    def record(
        self,
        *,
        error: "TransientError | None",
        assistant_message: AssistantMessage | None,
        billing: Billing | None = None,
    ) -> None:
        """Close the attempt started by the last start_attempt(), ending it now."""
        self.record_ending_at(
            time.monotonic(), error=error, assistant_message=assistant_message, billing=billing
        )

    def record_ending_at(
        self,
        ended_at_monotonic_seconds: float,
        *,
        error: "TransientError | None",
        assistant_message: AssistantMessage | None,
        billing: Billing | None = None,
    ) -> None:
        """Close the attempt started by the last start_attempt(), at a stamp taken earlier.

        Merges the staged response with what reading it produced, and clears the stage.
        billing is what an attempt that staged no response billed, which a stream that broke
        mid-turn has: it reports counters as it goes and assembles no response to stage.
        A staged response's own billing takes precedence, being the provider's report of the whole
        response, and the two never arrive together.
        A stream's attempt ends when its item iterator exhausts, several awaits before final() reads
        the assembled response, so the stream path stamps the end itself.
        """
        staged = self._staged_response
        self._staged_response = None
        self._attempt_in_flight = False
        self._billing_in_flight = None
        self._attempt_records.append(
            AttemptRecord(
                started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                first_item_at_monotonic_seconds=self._first_item_at_monotonic_seconds,
                error=error,
                billing=staged.billing if staged is not None else billing,
                assistant_message=assistant_message,
                raw=staged.raw if staged is not None else None,
                model_served=staged.identity.model_served if staged is not None else None,
                response_id=staged.identity.response_id if staged is not None else None,
                request_id=(
                    staged.identity.request_id if staged is not None else self._noted_request_id
                ),
            )
        )

    def freeze(self) -> CallRecord:
        """Return the call's history as of now, for a result to carry."""
        return self.freeze_ending_at(time.monotonic())

    def freeze_ending_at(self, ended_at_monotonic_seconds: float) -> CallRecord:
        """Return the call's history as of a stamp taken earlier.

        A response still staged closes first, with no turn and no error: the response arrived and
        whatever was going to read it did not finish, so the attempt and its billing are on the
        record and nothing about the turn is invented.
        A stream's call ends when its item iterator exhausts; final() may be awaited any time after,
        and that gap is the caller's own work, not the call's.
        """
        if self._staged_response is not None:
            self.record_ending_at(ended_at_monotonic_seconds, error=None, assistant_message=None)
        return CallRecord(
            model=self._model,
            provider_name=self._provider_name,
            attempt_records=tuple(self._attempt_records),
            started_at_monotonic_seconds=self._started_at_monotonic_seconds,
            elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
        )
