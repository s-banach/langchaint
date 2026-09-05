"""Provider-neutral per-attempt and per-call generation records."""

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, override

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from langchaint.adapter import ResponseIdentity
from langchaint.billing.pricing import Billing, ProviderBilling
from langchaint.billing.usage import ZERO_USAGE, Usage
from langchaint.common.checked_copy import CheckedCopyModel
from langchaint.common.messages import AssistantMessage

if TYPE_CHECKING:
    from langchaint.common.exceptions import TransientError


type _NonnegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0)]
_RECORD_CONFIG = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")


def _less_than_or_ulp_close(left: float, right: float) -> bool:
    """Accept an ordered pair or a four-ULP rounding difference."""
    return left <= right or math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=4 * max(math.ulp(left), math.ulp(right)),
    )


class TransientErrorRecord(CheckedCopyModel):
    """The normalized retry information from one failed attempt."""

    model_config = _RECORD_CONFIG

    message: str
    retry_after_seconds: _NonnegativeFiniteFloat | None = None
    is_rate_limit: bool = False

    @override
    def __str__(self) -> str:
        """Return the retry failure message."""
        return self.message


class AttemptRecord(CheckedCopyModel):
    """The normalized base for settled and cut-off request records."""

    model_config = _RECORD_CONFIG


class SettledAttemptRecord(AttemptRecord):
    """One request whose ending langchaint observed."""

    started_after_seconds: _NonnegativeFiniteFloat
    elapsed_seconds: _NonnegativeFiniteFloat
    seconds_to_first_item: _NonnegativeFiniteFloat | None
    error: TransientErrorRecord | None
    billing: Billing | None
    assistant_message: AssistantMessage | None
    model_served: str | None
    response_id: str | None
    request_id: str | None
    kind: Literal["settled"] = "settled"

    @property
    def usage(self) -> Usage:
        """Return normalized billing usage or `ZERO_USAGE`."""
        return ZERO_USAGE if self.billing is None else self.billing.usage

    @model_validator(mode="after")
    def _validate_first_item_timing(self) -> "SettledAttemptRecord":
        if self.seconds_to_first_item is not None and not _less_than_or_ulp_close(
            self.seconds_to_first_item, self.elapsed_seconds
        ):
            raise ValueError("seconds_to_first_item must not exceed elapsed_seconds")
        return self


class CutOffAttemptRecord(AttemptRecord):
    """One request whose ending langchaint did not observe."""

    started_after_seconds: _NonnegativeFiniteFloat
    billing: Billing | None
    kind: Literal["cut_off"] = "cut_off"

    @property
    def usage(self) -> Usage:
        """Return normalized billing usage or `ZERO_USAGE`."""
        return ZERO_USAGE if self.billing is None else self.billing.usage


type _AttemptRecordVariant = Annotated[
    SettledAttemptRecord | CutOffAttemptRecord, Field(discriminator="kind")
]


class CallRecord(CheckedCopyModel):
    """The normalized ordered request records and elapsed time of one call."""

    model_config = _RECORD_CONFIG

    model: str
    provider_name: str
    attempt_records: tuple[_AttemptRecordVariant, ...]
    elapsed_seconds: _NonnegativeFiniteFloat

    @model_validator(mode="after")
    def _validate_attempt_timing(self) -> "CallRecord":
        cut_off_indexes = [
            index
            for index, attempt in enumerate(self.attempt_records)
            if isinstance(attempt, CutOffAttemptRecord)
        ]
        if len(cut_off_indexes) > 1:
            raise ValueError("attempt_records may contain at most one cut-off record")
        if cut_off_indexes and cut_off_indexes[0] != len(self.attempt_records) - 1:
            raise ValueError("a cut-off attempt record must be final")

        previous_end = 0.0
        for attempt in self.attempt_records:
            if not _less_than_or_ulp_close(previous_end, attempt.started_after_seconds):
                raise ValueError("attempt records must not overlap")
            if not _less_than_or_ulp_close(attempt.started_after_seconds, self.elapsed_seconds):
                raise ValueError("an attempt start must fall within the call")
            if isinstance(attempt, SettledAttemptRecord):
                attempt_end = attempt.started_after_seconds + attempt.elapsed_seconds
                if not _less_than_or_ulp_close(attempt_end, self.elapsed_seconds):
                    raise ValueError("a settled attempt end must fall within the call")
                previous_end = attempt_end
        return self


class _CallResultRecordBase(CheckedCopyModel):
    """Share call-derived properties across normalized result records.

    Validation rejects unknown fields.
    """

    model_config = _RECORD_CONFIG

    call: CallRecord

    @property
    def attempts(self) -> int:
        """Return the observed request count."""
        return len(self.call.attempt_records)

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every request."""
        return Usage.sum_of(attempt.usage for attempt in self.call.attempt_records)

    @property
    def model(self) -> str:
        """Return the requested model id."""
        return self.call.model

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.call.provider_name

    @property
    def elapsed_seconds(self) -> float:
        """Return the complete call duration."""
        return self.call.elapsed_seconds


def _settled_attempts(call: CallRecord) -> tuple[SettledAttemptRecord, ...]:
    attempts = tuple(
        attempt for attempt in call.attempt_records if isinstance(attempt, SettledAttemptRecord)
    )
    if len(attempts) != len(call.attempt_records):
        raise ValueError("this record does not permit a cut-off attempt")
    return attempts


def _require_completed_model_turn(call: CallRecord) -> None:
    attempts = _settled_attempts(call)
    if not attempts:
        raise ValueError("call must contain at least one settled attempt")
    if any(attempt.error is None for attempt in attempts[:-1]):
        raise ValueError("every attempt before the final attempt must contain an error")
    final = attempts[-1]
    if final.error is not None:
        raise ValueError("the final attempt must be error-free")
    if final.billing is None:
        raise ValueError("the final attempt must contain billing")
    if final.assistant_message is None:
        raise ValueError("the final attempt must contain an assistant message")


@dataclass(frozen=True, kw_only=True)
class AttemptProviderData:
    """Live provider values aligned with one normalized attempt record."""

    raw: BaseModel | None
    usage_raw: BaseModel | None


class _StagedResponse(NamedTuple):
    raw: BaseModel
    provider_billing: ProviderBilling
    identity: ResponseIdentity


class _CallLedger:
    """Accumulate live attempt state and freeze provider-neutral records."""

    def __init__(self, *, model: str, provider_name: str) -> None:
        self._model = model
        self._provider_name = provider_name
        self._attempt_records: list[SettledAttemptRecord] = []
        self._provider_attempts: list[AttemptProviderData] = []
        self._staged_response: _StagedResponse | None = None
        self._started_at_monotonic_seconds = time.monotonic()
        self._attempt_started_at_monotonic_seconds = self._started_at_monotonic_seconds
        self._attempt_in_flight = False
        self._first_item_at_monotonic_seconds: float | None = None
        self._noted_request_id: str | None = None
        self._billing_in_flight: ProviderBilling | None = None

    def stage_response(
        self,
        *,
        raw: BaseModel,
        billing: ProviderBilling,
        identity: ResponseIdentity,
    ) -> None:
        """Hold a complete response before interpretation records its outcome."""
        self._staged_response = _StagedResponse(
            raw=raw, provider_billing=billing, identity=identity
        )

    def start_call(self) -> None:
        """Set the call origin immediately before generation starts."""
        self._started_at_monotonic_seconds = time.monotonic()

    def start_attempt(self) -> None:
        """Start one request attempt."""
        self._attempt_started_at_monotonic_seconds = time.monotonic()
        self._attempt_in_flight = True
        self._first_item_at_monotonic_seconds = None
        self._noted_request_id = None
        self._billing_in_flight = None

    def stamp_first_item(self) -> None:
        """Record the first streamed item once."""
        if self._first_item_at_monotonic_seconds is None:
            self._first_item_at_monotonic_seconds = time.monotonic()

    def note_request_id(self, request_id: str | None) -> None:
        """Store the current attempt's request id."""
        self._noted_request_id = request_id

    def note_billing_in_flight(self, billing: ProviderBilling | None) -> None:
        """Store billing reported before an interruption."""
        self._billing_in_flight = billing

    @property
    def billing_in_flight(self) -> ProviderBilling | None:
        """Return billing for the current request, when reported."""
        return self._billing_in_flight

    @property
    def attempts(self) -> int:
        """Return the settled request count."""
        return len(self._attempt_records)

    @property
    def attempt_records(self) -> tuple[SettledAttemptRecord, ...]:
        """Return the settled normalized attempt records."""
        return tuple(self._attempt_records)

    @property
    def provider_attempts(self) -> tuple[AttemptProviderData, ...]:
        """Return live provider data aligned with settled attempts."""
        return tuple(self._provider_attempts)

    def record(
        self,
        *,
        error: "TransientError | None",
        assistant_message: AssistantMessage | None,
        billing: ProviderBilling | None = None,
    ) -> None:
        """Close the current attempt at the current monotonic time."""
        self.record_ending_at(
            time.monotonic(), error=error, assistant_message=assistant_message, billing=billing
        )

    def record_ending_at(
        self,
        ended_at_monotonic_seconds: float,
        *,
        error: "TransientError | None",
        assistant_message: AssistantMessage | None,
        billing: ProviderBilling | None = None,
    ) -> None:
        """Close the current attempt at an existing monotonic timestamp."""
        staged = self._staged_response
        self._staged_response = None
        self._attempt_in_flight = False
        self._billing_in_flight = None
        provider_billing = staged.provider_billing if staged is not None else billing
        started_after_seconds = (
            self._attempt_started_at_monotonic_seconds - self._started_at_monotonic_seconds
        )
        elapsed_seconds = ended_at_monotonic_seconds - self._attempt_started_at_monotonic_seconds
        self._attempt_records.append(
            SettledAttemptRecord(
                started_after_seconds=started_after_seconds,
                elapsed_seconds=elapsed_seconds,
                seconds_to_first_item=(
                    None
                    if self._first_item_at_monotonic_seconds is None
                    else self._first_item_at_monotonic_seconds
                    - self._attempt_started_at_monotonic_seconds
                ),
                error=(
                    None
                    if error is None
                    else TransientErrorRecord(
                        message=str(error),
                        retry_after_seconds=error.retry_after_seconds,
                        is_rate_limit=error.is_rate_limit,
                    )
                ),
                billing=None if provider_billing is None else provider_billing.billing,
                assistant_message=assistant_message,
                model_served=staged.identity.model_served if staged is not None else None,
                response_id=staged.identity.response_id if staged is not None else None,
                request_id=(
                    staged.identity.request_id if staged is not None else self._noted_request_id
                ),
            )
        )
        self._provider_attempts.append(
            AttemptProviderData(
                raw=staged.raw if staged is not None else None,
                usage_raw=None if provider_billing is None else provider_billing.usage_raw,
            )
        )

    def freeze(self) -> CallRecord:
        """Freeze settled call state at the current monotonic time."""
        return self.freeze_ending_at(time.monotonic())

    def freeze_ending_at(self, ended_at_monotonic_seconds: float) -> CallRecord:
        """Freeze settled call state at an existing monotonic timestamp."""
        if self._staged_response is not None:
            self.record_ending_at(ended_at_monotonic_seconds, error=None, assistant_message=None)
        return CallRecord(
            model=self._model,
            provider_name=self._provider_name,
            attempt_records=tuple(self._attempt_records),
            elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
        )

    def freeze_with_cut_off(
        self, billing: ProviderBilling | None = None
    ) -> tuple[CallRecord, tuple[AttemptProviderData, ...]]:
        """Freeze the call and append one cut-off record for an open request."""
        ended_at_monotonic_seconds = time.monotonic()
        cut_off_in_flight = self._attempt_in_flight and self._staged_response is None
        attempt_started_at_monotonic_seconds = self._attempt_started_at_monotonic_seconds
        call = self.freeze_ending_at(ended_at_monotonic_seconds)
        provider_attempts = self.provider_attempts
        if not cut_off_in_flight:
            return call, provider_attempts
        provider_billing = billing if billing is not None else self._billing_in_flight
        cut_off = CutOffAttemptRecord(
            started_after_seconds=attempt_started_at_monotonic_seconds
            - self._started_at_monotonic_seconds,
            billing=None if provider_billing is None else provider_billing.billing,
        )
        return (
            CallRecord(
                model=call.model,
                provider_name=call.provider_name,
                attempt_records=(*call.attempt_records, cut_off),
                elapsed_seconds=call.elapsed_seconds,
            ),
            (
                *provider_attempts,
                AttemptProviderData(
                    raw=None,
                    usage_raw=None if provider_billing is None else provider_billing.usage_raw,
                ),
            ),
        )
