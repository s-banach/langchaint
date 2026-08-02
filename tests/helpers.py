"""Helpers shared by more than one test module.

A helper lands here when a second module needs it; one used by a single module stays in that module.
"""

import importlib
import math
import pkgutil
from collections.abc import Iterator
from types import ModuleType

from pydantic import BaseModel

import langchaint
from langchaint import (
    ZERO_USAGE,
    AssistantMessage,
    AttemptRecord,
    Billing,
    CallRecord,
    TransientError,
    Usage,
)

CALL_STARTED_AT = 1000.0
"""The fixed time.monotonic() origin every record attempt_record and call_record build sits on."""


class StubRaw(BaseModel):
    """Stand-in for the SDK's own response model a result carries on raw."""


def stated_billing(
    usage: Usage,
    *,
    input_cache_none_usd_per_million_tokens: float = math.nan,
    usage_raw: BaseModel | None = None,
) -> Billing:
    """Carry a Usage a test stated outright as the Billing an attempt record holds.

    The tier is filler and every rate defaults to NaN: a test that states its costs on the Usage is
    exercising what the tree does with a Billing, not how a rate table built one. A test of
    cache_savings_in_usd states the one rate that property reads.
    usage_raw defaults to None, the provider having reported no usage object.
    """
    return Billing(
        usage=usage,
        service_tier="stub",
        usage_raw=usage_raw,
        input_cache_none_usd_per_million_tokens=input_cache_none_usd_per_million_tokens,
        cache_read_usd_per_million_tokens=math.nan,
        cache_write_usd_per_million_tokens=math.nan,
        output_usd_per_million_tokens=math.nan,
    )


def attempt_record(
    *,
    error: TransientError | None,
    usage: Usage = ZERO_USAGE,
    reported_billing: bool = True,
    input_cache_none_usd_per_million_tokens: float = math.nan,
    usage_raw: BaseModel | None = None,
    started_after_seconds: float = 0.0,
    elapsed_seconds: float = 0.0,
    seconds_to_first_item: float | None = None,
    turn: AssistantMessage | None = None,
    model_served: str | None = None,
    response_id: str | None = None,
    request_id: str | None = None,
) -> AttemptRecord:
    """Build one record on the fixed origin; reported_billing False is an attempt the provider never billed."""
    started_at_monotonic_seconds = CALL_STARTED_AT + started_after_seconds
    return AttemptRecord(
        started_at_monotonic_seconds=started_at_monotonic_seconds,
        ended_at_monotonic_seconds=started_at_monotonic_seconds + elapsed_seconds,
        first_item_at_monotonic_seconds=(
            None
            if seconds_to_first_item is None
            else started_at_monotonic_seconds + seconds_to_first_item
        ),
        error=error,
        billing=(
            stated_billing(
                usage,
                input_cache_none_usd_per_million_tokens=input_cache_none_usd_per_million_tokens,
                usage_raw=usage_raw,
            )
            if reported_billing
            else None
        ),
        assistant_message=turn,
        raw=None,
        model_served=model_served,
        response_id=response_id,
        request_id=request_id,
    )


def call_record(
    attempt_records: tuple[AttemptRecord, ...], *, elapsed_seconds: float
) -> CallRecord:
    """Build a CallRecord over the records under test; the identity fields are fixed filler."""
    return CallRecord(
        model="fake-model",
        provider_name="fake",
        attempt_records=attempt_records,
        started_at_monotonic_seconds=CALL_STARTED_AT,
        elapsed_seconds=elapsed_seconds,
    )


def package_modules() -> Iterator[ModuleType]:
    """Import every module under langchaint, backend subpackages included.

    Importing the backend subpackages and tracing requires both SDKs and opentelemetry-api,
    which the dev environment installs for the adapter and tracing tests.

    Yields:
        Each imported module, the package itself first.
    """
    yield langchaint
    for module_info in pkgutil.walk_packages(langchaint.__path__, prefix="langchaint."):
        yield importlib.import_module(module_info.name)


def random_returns_zero() -> float:
    """Stand in for random.random, returning zero.

    Patched over the random.random shared_backoff draws waits from, this makes every drawn wait its
    ceiling. A test can then state the delay the retry loop waits.
    """
    return 0.0
