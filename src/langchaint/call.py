"""Per-attempt and per-call generation history.

Retry loops append `AttemptRecord` values to `_CallLedger` and freeze them into `CallRecord`.
Success and error values expose `CallRecord` fields through `_CallCarrier`.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel

from langchaint.messages import AssistantMessage
from langchaint.pricing import Billing
from langchaint.usage import ZERO_USAGE, Usage

if TYPE_CHECKING:
    # Importing `TransientError` at runtime would create a cycle through `exceptions.py`.
    from langchaint.exceptions import TransientError


class ResponseIdentity(NamedTuple):
    """Identifiers recorded for one response.

    `BoundAdapter.identity_from_raw` returns this value.
    Its fields populate the corresponding `AttemptRecord` fields.
    """

    model_served: str
    response_id: str
    request_id: str | None


@dataclass(frozen=True, kw_only=True)
class AttemptRecord:
    """The observed state of one request attempt.

    Monotonic timestamps are comparable only within one process.
    The attempt interval excludes admission and backoff waits.
    `first_item_at_monotonic_seconds` records the first streamed item exposed to the caller.
    `error` contains only a retriable `TransientError`.
    `billing` is `None` when no response or stream reported billing.
    `raw` holds the mutable SDK response by reference.
    Copy `raw` before mutation.
    Response identifiers and `assistant_message` are `None` when no response supplied them.
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
        """Return the billed counters and costs in the neutral summary.

        Return `ZERO_USAGE` when no billing was reported.
        """
        return ZERO_USAGE if self.billing is None else self.billing.usage


@dataclass(frozen=True, kw_only=True)
class CallRecord:
    """The ordered attempt history and elapsed time of one call.

    `elapsed_seconds` includes admission and backoff waits.
    Interrupted or rejected requests can lack an `AttemptRecord`.
    Each result carrier computes its own `usage` from these records.
    """

    model: str
    provider_name: str
    attempt_records: tuple[AttemptRecord, ...]
    started_at_monotonic_seconds: float
    elapsed_seconds: float


class _CallCarrier:
    """Forward shared result fields from `call`.

    Each subclass declares its own `call` dataclass field.
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
        """Return the call's start timestamp.

        Attempt record timestamps use this timestamp as their origin.
        """
        return self.call.started_at_monotonic_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Return the call's wall time including permit waits and backoff sleeps."""
        return self.call.elapsed_seconds


class _StagedResponse(NamedTuple):
    raw: BaseModel
    billing: Billing
    identity: ResponseIdentity


class _CallLedger:
    """Accumulate attempt state and freeze it into `CallRecord`.

    `stage_response` preserves a response before interpretation.
    `start_attempt` and `record_ending_at` track the in-flight request state used after interruption.
    """

    def __init__(self, *, model: str, provider_name: str) -> None:
        """Stamp the call start with the current monotonic time."""
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

        Call before response interpretation.
        The next record or freeze closes the staged attempt.
        An intervening exception cannot lose its response, billing, or ids.
        """
        self._staged_response = _StagedResponse(raw=raw, billing=billing, identity=identity)

    def start_call(self) -> None:
        """Stamp the call's start as now, replacing the constructor's stamp.

        Each retry loop calls this first.
        `stream_one` constructs `StreamHandle` before `__aenter__` opens its request.
        The intervening time belongs to the caller.
        """
        self._started_at_monotonic_seconds = time.monotonic()

    def start_attempt(self) -> None:
        """Stamp the attempt start with the current monotonic time.

        The next record closes the interval.
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

        Call when the retry loop first sees an exception.
        Every closing path then records the id.
        A staged response id overrides it because that response completed the request.
        """
        self._noted_request_id = request_id

    def note_billing_in_flight(self, billing: Billing | None) -> None:
        """Hold what the provider had reported for the attempt in flight when it was cut off.

        Call when `BaseException` interrupts a stream.
        A later deadline account then includes the reported billing.
        `record` and `start_attempt` clear the billing after use.
        """
        self._billing_in_flight = billing

    @property
    def billing_in_flight(self) -> Billing | None:
        """Return the noted in-flight billing.

        Return `None` after a record settles the corresponding attempt.
        """
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
        """Return when the open attempt started.

        An attempt remains open from `start_attempt` until a record closes it.
        An interrupted open attempt reports when its request started.
        No `AttemptRecord` exists because its ending was unobserved.
        """
        return self._attempt_started_at_monotonic_seconds if self._attempt_in_flight else None

    def record(
        self,
        *,
        error: "TransientError | None",
        assistant_message: AssistantMessage | None,
        billing: Billing | None = None,
    ) -> None:
        """Close the attempt started by the last `start_attempt` call at the current time."""
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
        """Close the current attempt at an existing timestamp.

        A staged response supplies billing, raw data, and identifiers.
        `billing` applies when no complete response was staged.
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
        """Return the call history as of the current time for a result to carry."""
        return self.freeze_ending_at(time.monotonic())

    def freeze_ending_at(self, ended_at_monotonic_seconds: float) -> CallRecord:
        """Freeze call history at an existing timestamp.

        Close a staged response with its billing and no inferred turn.
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
