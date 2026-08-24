"""Helpers shared by more than one test module.

A helper lands here when a second module needs it. One used by a single module stays in that module.
"""

import importlib
import math
import pkgutil
from collections.abc import Iterator, Mapping
from types import ModuleType

import httpx2
import openai
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
from langchaint.adapter import ErrorClassification
from langchaint.shared_backoff import (
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    RetryThisOne,
    Verdict,
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
    """Build Billing from test-stated Usage and optional cache rate."""
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
    """Build one record on the fixed origin. reported_billing False is an attempt the provider never billed."""
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
    """Build a CallRecord over the records under test. The identity fields are fixed filler."""
    return CallRecord(
        model="fake-model",
        provider_name="fake",
        attempt_records=attempt_records,
        started_at_monotonic_seconds=CALL_STARTED_AT,
        elapsed_seconds=elapsed_seconds,
    )


def package_modules() -> Iterator[ModuleType]:
    """Import every module under langchaint, backend subpackages included.

    Backend imports require their provider SDKs.
    Tracing imports require opentelemetry-api.
    The development environment installs those dependencies.

    Yields:
        Each imported module, the package itself first.
    """
    yield langchaint
    for module_info in pkgutil.walk_packages(langchaint.__path__, prefix="langchaint."):
        yield importlib.import_module(module_info.name)


def random_returns_zero() -> float:
    """Stand in for random.random, returning zero.

    Patching random.random with this function makes each wait equal its ceiling.
    """
    return 0.0


def status_error[ErrorT: openai.APIStatusError](
    error_class: type[ErrorT],
    status_code: int,
    headers: dict[str, str] | None = None,
    error_code: str | None = None,
) -> ErrorT:
    """Build an openai status exception with optional error_code."""
    response = httpx2.Response(
        status_code,
        request=httpx2.Request("POST", "https://api.openai.com"),
        headers=headers,
    )
    body = (
        None
        if error_code is None
        else {"code": error_code, "type": "insufficient_quota", "message": "boom"}
    )
    return error_class("boom", response=response, body=body)


def connection_error() -> openai.APIConnectionError:
    """Build an openai transport exception."""
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com"))


def openai_sdk_errors_and_classifications() -> Mapping[Exception, ErrorClassification]:
    """Return shared openai error classification cases."""
    return {
        connection_error(): "transient",
        openai.APITimeoutError(httpx2.Request("POST", "https://api.openai.com")): "transient",
        status_error(openai.RateLimitError, 429): "invalid_request",
        status_error(openai.ConflictError, 409): "invalid_request",
        status_error(openai.BadRequestError, 400): "invalid_request",
        status_error(openai.AuthenticationError, 401): "invalid_request",
        status_error(openai.PermissionDeniedError, 403): "invalid_request",
        status_error(openai.NotFoundError, 404): "invalid_request",
        status_error(openai.UnprocessableEntityError, 422): "invalid_request",
        status_error(openai.APIStatusError, 413): "invalid_request",
        status_error(openai.APIStatusError, 408): "invalid_request",
        status_error(openai.BadRequestError, 400, {"x-should-retry": "false"}): "invalid_request",
        status_error(
            openai.InternalServerError, 500, {"x-should-retry": "false"}
        ): "declared_final",
        status_error(openai.InternalServerError, 500): "unknown_exception",
        status_error(openai.InternalServerError, 503): "unknown_exception",
        status_error(openai.APIStatusError, 302): "unknown_exception",
        status_error(openai.APIStatusError, 200, error_code="invalid_prompt"): "declared_final",
        ValueError("boom"): "unknown_exception",
    }


def openai_sdk_errors_and_verdicts() -> Mapping[Exception, Verdict]:
    """Return shared openai error verdict cases."""
    return {
        status_error(openai.RateLimitError, 429, {"retry-after": "7"}): PauseAll(retry_after=7.0),
        status_error(
            openai.RateLimitError, 429, error_code="organization_spend_limit_exceeded"
        ): DoNotRetry(),
        status_error(openai.InternalServerError, 503): PauseAll(retry_after=None),
        status_error(openai.InternalServerError, 500): RetryThisOne(retry_after=None),
        status_error(openai.APIStatusError, 408): RetryThisOne(retry_after=None),
        status_error(openai.ConflictError, 409): RetryThisOne(retry_after=None),
        status_error(openai.BadRequestError, 400): DoNotRetry(),
        status_error(openai.AuthenticationError, 401): DoNotRetry(),
        status_error(openai.PermissionDeniedError, 403): DoNotRetry(),
        status_error(openai.NotFoundError, 404): DoNotRetry(),
        status_error(openai.UnprocessableEntityError, 422): DoNotRetry(),
        status_error(openai.APIStatusError, 451): DoNotRetry(),
        status_error(openai.InternalServerError, 599): RetryThisOne(retry_after=None),
        status_error(openai.RateLimitError, 429, {"x-should-retry": "false"}): PauseAllDoNotRetry(
            retry_after=None
        ),
        TransientError("throttled body", retry_after_seconds=3.0, is_rate_limit=True): PauseAll(
            retry_after=3.0
        ),
        TransientError("failed body"): RetryThisOne(retry_after=None),
    }
