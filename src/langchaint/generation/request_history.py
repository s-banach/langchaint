"""Provider-neutral request records and the request history of one input."""

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, NamedTuple, override

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from langchaint.adapter import ResponseIdentity
from langchaint.billing.pricing import Billing, ProviderBilling
from langchaint.billing.usage import ZERO_USAGE, Usage
from langchaint.common.checked_copy import CheckedCopyModel
from langchaint.common.messages import AssistantMessage, StopReason

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
    """The normalized retry information from one failed request."""

    model_config = _RECORD_CONFIG

    message: str
    retry_after_seconds: _NonnegativeFiniteFloat | None = None
    is_rate_limit: bool = False

    @override
    def __str__(self) -> str:
        """Return the retry failure message."""
        return self.message


class RequestRecord(CheckedCopyModel):
    """The normalized base for settled and cut-off request records."""

    model_config = _RECORD_CONFIG


class SettledRequestRecord(RequestRecord):
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
    def _validate_first_item_timing(self) -> "SettledRequestRecord":
        if self.seconds_to_first_item is not None and not _less_than_or_ulp_close(
            self.seconds_to_first_item, self.elapsed_seconds
        ):
            raise ValueError("seconds_to_first_item must not exceed elapsed_seconds")
        return self


class CutOffRequestRecord(RequestRecord):
    """One request whose ending langchaint did not observe.

    `seconds_to_first_item` is `None` when no streamed item arrived before the cut-off.
    Its default keeps records saved without it valid.
    """

    started_after_seconds: _NonnegativeFiniteFloat
    seconds_to_first_item: _NonnegativeFiniteFloat | None = None
    billing: Billing | None
    kind: Literal["cut_off"] = "cut_off"

    @property
    def usage(self) -> Usage:
        """Return normalized billing usage or `ZERO_USAGE`."""
        return ZERO_USAGE if self.billing is None else self.billing.usage


type _RequestRecordVariant = Annotated[
    SettledRequestRecord | CutOffRequestRecord, Field(discriminator="kind")
]


class RequestHistory(CheckedCopyModel):
    """The normalized ordered request records and elapsed time of one input."""

    model_config = _RECORD_CONFIG

    model: str
    provider_name: str
    request_records: tuple[_RequestRecordVariant, ...]
    elapsed_seconds: _NonnegativeFiniteFloat

    @model_validator(mode="after")
    def _validate_request_timing(self) -> "RequestHistory":
        cut_off_indexes = [
            index
            for index, request_record in enumerate(self.request_records)
            if request_record.kind == "cut_off"
        ]
        if len(cut_off_indexes) > 1:
            raise ValueError("request_records may contain at most one cut-off record")
        if cut_off_indexes and cut_off_indexes[0] != len(self.request_records) - 1:
            raise ValueError("a cut-off request record must be final")

        previous_end = 0.0
        for request_record in self.request_records:
            if not _less_than_or_ulp_close(previous_end, request_record.started_after_seconds):
                raise ValueError("request records must not overlap")
            if not _less_than_or_ulp_close(
                request_record.started_after_seconds, self.elapsed_seconds
            ):
                raise ValueError("a request start must fall within the request history")
            if request_record.kind == "settled":
                request_end = request_record.started_after_seconds + request_record.elapsed_seconds
                if not _less_than_or_ulp_close(request_end, self.elapsed_seconds):
                    raise ValueError("a settled request end must fall within the request history")
                previous_end = request_end
        return self


class _InputOutcomeRecordBase(CheckedCopyModel):
    """Share properties derived from `request_history` across every `InputOutcomeRecord` variant.

    Validation rejects unknown fields.
    """

    model_config = _RECORD_CONFIG

    request_history: RequestHistory

    @property
    def request_count(self) -> int:
        """Return the observed request count."""
        return len(self.request_history.request_records)

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every request."""
        return Usage.sum_of(
            request_record.usage for request_record in self.request_history.request_records
        )

    @property
    def model(self) -> str:
        """Return the requested model id."""
        return self.request_history.model

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.request_history.provider_name

    @property
    def elapsed_seconds(self) -> float:
        """Return the seconds from the start of handling the input until its request history was frozen.

        It includes admission and backoff waits.
        """
        return self.request_history.elapsed_seconds

    @property
    def request_records(
        self,
    ) -> tuple[SettledRequestRecord | CutOffRequestRecord, ...]:
        """Return the input's normalized request records."""
        return self.request_history.request_records

    @property
    def assistant_message(self) -> AssistantMessage | None:
        """Return the last recorded assistant message."""
        for request_record in reversed(self.request_history.request_records):
            if request_record.kind == "settled" and request_record.assistant_message is not None:
                return request_record.assistant_message
        return None


def _require_abandoned_shape(request_history: RequestHistory) -> None:
    settled = tuple(
        request_record
        for request_record in request_history.request_records
        if request_record.kind == "settled"
    )
    final_is_cut_off = (
        bool(request_history.request_records)
        and request_history.request_records[-1].kind == "cut_off"
    )
    if final_is_cut_off:
        settled_prefix = settled
    elif settled and settled[-1].error is None:
        final = settled[-1]
        if final.assistant_message is not None:
            raise ValueError("the final settled request must not contain an assistant message")
        settled_prefix = settled[:-1]
    else:
        settled_prefix = settled
    if any(request_record.error is None for request_record in settled_prefix):
        raise ValueError("settled requests before the final request must contain errors")


class AbandonedStreamRecord(_InputOutcomeRecordBase):
    """A `stream_one` input whose `async with` block exited before a `Generation` or `GenerationError` recorded it.

    The application ended the stream, so handling the input did not fail.
    The record keeps the billing and first-item time of the request the exit cut off.
    Validation rejects unknown fields.
    """

    stop_reason: ClassVar[StopReason | None] = None
    kind: Literal["abandoned_stream"] = "abandoned_stream"

    @model_validator(mode="after")
    def _validate_abandoned_shape(self) -> "AbandonedStreamRecord":
        _require_abandoned_shape(self.request_history)
        return self


def _settled_request_records(request_history: RequestHistory) -> tuple[SettledRequestRecord, ...]:
    settled_request_records = tuple(
        request_record
        for request_record in request_history.request_records
        if request_record.kind == "settled"
    )
    if len(settled_request_records) != len(request_history.request_records):
        raise ValueError("this record does not permit a cut-off request")
    return settled_request_records


def _require_completed_assistant_message(request_history: RequestHistory) -> None:
    settled_request_records = _settled_request_records(request_history)
    if not settled_request_records:
        raise ValueError("request history must contain at least one settled request")
    if any(request_record.error is None for request_record in settled_request_records[:-1]):
        raise ValueError("every request before the final request must contain an error")
    final = settled_request_records[-1]
    if final.error is not None:
        raise ValueError("the final request must be error-free")
    if final.billing is None:
        raise ValueError("the final request must contain billing")
    if final.assistant_message is None:
        raise ValueError("the final request must contain an assistant message")


@dataclass(frozen=True, kw_only=True)
class RequestProviderData:
    """Live provider values aligned with one normalized request record."""

    raw: BaseModel | None
    usage_raw: BaseModel | None


class _StagedResponse(NamedTuple):
    raw: BaseModel
    provider_billing: ProviderBilling
    identity: ResponseIdentity


class _RequestLedger:
    """Accumulate live request state and freeze provider-neutral records.

    Building a ledger starts the input's request history.
    """

    def __init__(self, *, model: str, provider_name: str) -> None:
        self._model = model
        self._provider_name = provider_name
        self._request_records: list[SettledRequestRecord] = []
        self._request_provider_data: list[RequestProviderData] = []
        self._staged_response: _StagedResponse | None = None
        self._started_at_monotonic_seconds = time.monotonic()
        self._request_started_at_monotonic_seconds = self._started_at_monotonic_seconds
        self._request_in_flight = False
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

    def start_request(self) -> None:
        """Start one request."""
        self._request_started_at_monotonic_seconds = time.monotonic()
        self._request_in_flight = True
        self._first_item_at_monotonic_seconds = None
        self._noted_request_id = None
        self._billing_in_flight = None

    def stamp_first_item(self) -> None:
        """Record the first streamed item once."""
        if self._first_item_at_monotonic_seconds is None:
            self._first_item_at_monotonic_seconds = time.monotonic()

    def note_request_id(self, request_id: str | None) -> None:
        """Store the current request's request id."""
        self._noted_request_id = request_id

    def note_billing_in_flight(self, billing: ProviderBilling | None) -> None:
        """Store billing reported before an interruption."""
        self._billing_in_flight = billing

    @property
    def billing_in_flight(self) -> ProviderBilling | None:
        """Return billing for the current request, when reported."""
        return self._billing_in_flight

    @property
    def request_count(self) -> int:
        """Return the settled request count."""
        return len(self._request_records)

    @property
    def request_records(self) -> tuple[SettledRequestRecord, ...]:
        """Return the settled normalized request records."""
        return tuple(self._request_records)

    @property
    def request_provider_data(self) -> tuple[RequestProviderData, ...]:
        """Return live provider data aligned with settled request records."""
        return tuple(self._request_provider_data)

    def record(
        self,
        *,
        error: "TransientError | None",
        assistant_message: AssistantMessage | None,
        billing: ProviderBilling | None = None,
    ) -> None:
        """Close the current request at the current monotonic time."""
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
        """Close the current request at an existing monotonic timestamp."""
        staged = self._staged_response
        self._staged_response = None
        self._request_in_flight = False
        self._billing_in_flight = None
        provider_billing = staged.provider_billing if staged is not None else billing
        started_after_seconds = (
            self._request_started_at_monotonic_seconds - self._started_at_monotonic_seconds
        )
        elapsed_seconds = ended_at_monotonic_seconds - self._request_started_at_monotonic_seconds
        self._request_records.append(
            SettledRequestRecord(
                started_after_seconds=started_after_seconds,
                elapsed_seconds=elapsed_seconds,
                seconds_to_first_item=(
                    None
                    if self._first_item_at_monotonic_seconds is None
                    else self._first_item_at_monotonic_seconds
                    - self._request_started_at_monotonic_seconds
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
        self._request_provider_data.append(
            RequestProviderData(
                raw=staged.raw if staged is not None else None,
                usage_raw=None if provider_billing is None else provider_billing.usage_raw,
            )
        )

    def freeze(self) -> RequestHistory:
        """Freeze the settled request history at the current monotonic time."""
        return self.freeze_ending_at(time.monotonic())

    def freeze_ending_at(self, ended_at_monotonic_seconds: float) -> RequestHistory:
        """Freeze the settled request history at an existing monotonic timestamp."""
        if self._staged_response is not None:
            self.record_ending_at(ended_at_monotonic_seconds, error=None, assistant_message=None)
        return RequestHistory(
            model=self._model,
            provider_name=self._provider_name,
            request_records=tuple(self._request_records),
            elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
        )

    def freeze_with_cut_off(
        self, billing: ProviderBilling | None = None
    ) -> tuple[RequestHistory, tuple[RequestProviderData, ...]]:
        """Freeze the request history and append one cut-off record for an open request."""
        ended_at_monotonic_seconds = time.monotonic()
        cut_off_in_flight = self._request_in_flight and self._staged_response is None
        request_started_at_monotonic_seconds = self._request_started_at_monotonic_seconds
        first_item_at_monotonic_seconds = self._first_item_at_monotonic_seconds
        request_history = self.freeze_ending_at(ended_at_monotonic_seconds)
        request_provider_data = self.request_provider_data
        if not cut_off_in_flight:
            return request_history, request_provider_data
        provider_billing = billing if billing is not None else self._billing_in_flight
        cut_off = CutOffRequestRecord(
            started_after_seconds=request_started_at_monotonic_seconds
            - self._started_at_monotonic_seconds,
            seconds_to_first_item=(
                None
                if first_item_at_monotonic_seconds is None
                else first_item_at_monotonic_seconds - request_started_at_monotonic_seconds
            ),
            billing=None if provider_billing is None else provider_billing.billing,
        )
        return (
            RequestHistory(
                model=request_history.model,
                provider_name=request_history.provider_name,
                request_records=(*request_history.request_records, cut_off),
                elapsed_seconds=request_history.elapsed_seconds,
            ),
            (
                *request_provider_data,
                RequestProviderData(
                    raw=None,
                    usage_raw=None if provider_billing is None else provider_billing.usage_raw,
                ),
            ),
        )
