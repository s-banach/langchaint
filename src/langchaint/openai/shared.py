"""What both adapters over the openai SDK share.

The Responses adapter and the Chat Completions adapter wrap the same SDK client classes, raise the
same exception family, and price the same service-tier vocabulary, so the failure tables, the
pricing table, and the client-class map live here once. This module imports neither adapter module
and nothing private to the SDK.
"""

import base64
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import openai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI
from pydantic import BaseModel

from langchaint.adapter import (
    ErrorClassification,
    record_parse_fallthrough,
    retry_after_seconds_from_headers,
    terminal_classification_from_response,
    verdict_from_transient_error,
    verdict_under_retry_directive,
)
from langchaint.exceptions import TransientError
from langchaint.messages import ImagePart
from langchaint.pricing import Billing, category_cost
from langchaint.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict
from langchaint.usage import Usage

_PAUSE_STATUSES = frozenset({429, 503})
"""429 rate limits and documented 503 forms throttle the rate-limit quota.

Every request sharing the rate-limit quota pauses.
"""

_SPEND_LIMIT_CODES = frozenset({
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
})
"""The error.code values whose 429 no wait restores: credits or a set spend limit ran out."""

_RETRY_THIS_ONE_STATUSES = frozenset({500, 408, 409})
"""One request's server-side failure or collision, retried without pausing siblings."""

_DO_NOT_RETRY_STATUSES = frozenset({400, 401, 403, 404, 422})
"""The statuses that reject this request; a resend fails the same way.

404 and 422 are not in the error-code guide: AsyncOpenAI raises NotFoundError and
UnprocessableEntityError for them (openai 2.51.0).
"""

PARSE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""How often parse_openai fell to a status-family default, keyed by status and error type.

A diagnostic surface, read by no decision: a growing key names a status or error type the
tables above should learn.
"""

type OpenAIServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]
"""What a request may ask for, and what a response reports (openai 2.45.0 types both with this literal).

The API documents the response value as the processing mode actually used and says it may differ
from the value the request set, so the tier is read off each response rather than assumed.
"""

type _OpenAINormalizedServiceTier = Literal["default", "flex", "scale", "priority", "fast"]
"""Normalized OpenAI response service tiers."""

_DEFAULT_TIER: _OpenAINormalizedServiceTier = "default"

type _FailureDisposition = Literal["transient", "terminal"]


def client_without_retries[ClientT: openai.AsyncOpenAI](client: ClientT) -> ClientT:
    """Return one client whose SDK retries are disabled."""
    if client.max_retries == 0:
        return client
    return client.with_options(max_retries=0)


_DISPOSITION_BY_ERROR_CODE: Mapping[str, _FailureDisposition] = {
    "server_error": "transient",
    "rate_limit_exceeded": "transient",
    "vector_store_timeout": "transient",
    "invalid_prompt": "terminal",
    "data_residency_mismatch": "terminal",
    "bio_policy": "terminal",
    "invalid_image": "terminal",
    "invalid_image_format": "terminal",
    "invalid_base64_image": "terminal",
    "invalid_image_url": "terminal",
    "image_too_large": "terminal",
    "image_too_small": "terminal",
    "image_parse_error": "terminal",
    "image_content_policy_violation": "terminal",
    "invalid_image_mode": "terminal",
    "image_file_too_large": "terminal",
    "unsupported_image_media_type": "terminal",
    "empty_image_file": "terminal",
    "failed_to_download_image": "terminal",
    "image_file_not_found": "terminal",
}
"""Whether a resend may get past the failure each ResponseError.code names (openai 2.48.0).

Every value of the SDK's code literal is a key, which tests/test_provider_facts.py pins, so the
unknown-code path below is reached only by a code newer than the installed SDK.
The three transient codes are read off their names: the SDK documents none of the codes, and these
three name a condition of the moment while every other names a property of the request.
failed_to_download_image is terminal for that reason: a URL the caller got wrong fails identically
on every resend.
"""


def _image_data_uri(image_part: ImagePart) -> str:
    encoded_data = base64.b64encode(image_part.data).decode("ascii")
    return f"data:{image_part.media_type};base64,{encoded_data}"


def _priced_tier(service_tier: OpenAIServiceTier | None) -> _OpenAINormalizedServiceTier:
    """Normalize one response's `service_tier`."""
    if service_tier is None or service_tier == "auto":
        return _DEFAULT_TIER
    return service_tier


@dataclass(frozen=True, kw_only=True)
class OpenAIRates:
    """OpenAI token rates for one service tier."""

    input_cache_none_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_usd_per_million_tokens: float

    def price(  # noqa: PLR0913 (each normalized category arrives separately)
        self,
        *,
        service_tier: str,
        usage_raw: BaseModel | None,
        input_tokens_cache_read: int,
        input_tokens_cache_write: int,
        input_tokens_cache_none: int,
        output_tokens: int,
        output_tokens_reasoning: int,
        provider_executed_tool_cost_in_usd: float,
    ) -> Billing:
        """Price one response's counters at these rates.

        The counters arrive as arguments rather than in a counts object,
        which would exist only to be unpacked again one call later.
        output_tokens_reasoning is the reasoning share of output_tokens, billed at the output rate;
        it is a parameter because the returned Usage carries it.

        Token cost is the sum of four token category costs.
        provider_executed_tool_cost_in_usd adds the provider-executed tool charge.

        Raises:
            pydantic.ValidationError: a counter is negative.
        """
        return Billing(
            usage=Usage(
                input_tokens_cache_read=input_tokens_cache_read,
                input_tokens_cache_write=input_tokens_cache_write,
                input_tokens_cache_none=input_tokens_cache_none,
                output_tokens=output_tokens,
                output_tokens_reasoning=output_tokens_reasoning,
                input_tokens_cache_read_cost_in_usd=category_cost(
                    input_tokens_cache_read, self.cache_read_usd_per_million_tokens
                ),
                input_tokens_cache_write_cost_in_usd=category_cost(
                    input_tokens_cache_write, self.cache_write_usd_per_million_tokens
                ),
                input_tokens_cache_none_cost_in_usd=category_cost(
                    input_tokens_cache_none, self.input_cache_none_usd_per_million_tokens
                ),
                output_tokens_cost_in_usd=category_cost(
                    output_tokens, self.output_usd_per_million_tokens
                ),
                provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
            ),
            service_tier=service_tier,
            usage_raw=usage_raw,
            input_cache_none_usd_per_million_tokens=self.input_cache_none_usd_per_million_tokens,
            cache_read_usd_per_million_tokens=self.cache_read_usd_per_million_tokens,
            cache_write_usd_per_million_tokens=self.cache_write_usd_per_million_tokens,
            output_usd_per_million_tokens=self.output_usd_per_million_tokens,
        )

    def multiplied(self, *, input_multiplier: float, output_multiplier: float) -> "OpenAIRates":
        """Return rates multiplied by category."""
        return OpenAIRates(
            input_cache_none_usd_per_million_tokens=(
                self.input_cache_none_usd_per_million_tokens * input_multiplier
            ),
            output_usd_per_million_tokens=(self.output_usd_per_million_tokens * output_multiplier),
            cache_read_usd_per_million_tokens=(
                self.cache_read_usd_per_million_tokens * input_multiplier
            ),
            cache_write_usd_per_million_tokens=(
                self.cache_write_usd_per_million_tokens * input_multiplier
            ),
        )


_UNPRICED_RATES = OpenAIRates(
    input_cache_none_usd_per_million_tokens=float("nan"),
    output_usd_per_million_tokens=float("nan"),
    cache_read_usd_per_million_tokens=float("nan"),
    cache_write_usd_per_million_tokens=float("nan"),
)


@dataclass(frozen=True, kw_only=True)
class OpenAILongContextPricing:
    """OpenAI rate multipliers above one input-token threshold."""

    input_tokens_above: int
    input_multiplier: float
    output_multiplier: float

    def __post_init__(self) -> None:
        """Reject invalid thresholds and multipliers.

        Raises:
            ValueError: A threshold or multiplier is invalid.
        """
        if isinstance(self.input_tokens_above, bool) or self.input_tokens_above <= 0:
            raise ValueError("input_tokens_above must be a positive int")
        for name, multiplier in (
            ("input_multiplier", self.input_multiplier),
            ("output_multiplier", self.output_multiplier),
        ):
            if isinstance(multiplier, bool) or not isfinite(multiplier) or multiplier <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class OpenAIPricingTable:
    """OpenAI rates and modifiers for one model."""

    default: OpenAIRates
    flex: OpenAIRates | None = None
    fast: OpenAIRates | None = None
    scale: OpenAIRates | None = None
    long_context: OpenAILongContextPricing | None = None
    regional_processing_multiplier: float | None = None
    web_search_usd_per_invocation: float | None = None
    file_search_usd_per_invocation: float | None = None

    def __post_init__(self) -> None:
        """Reject an invalid regional multiplier.

        Raises:
            ValueError: The regional multiplier is invalid.
        """
        multiplier = self.regional_processing_multiplier
        if multiplier is None:
            return
        if isinstance(multiplier, bool) or not isfinite(multiplier) or multiplier <= 0:
            raise ValueError("regional_processing_multiplier must be finite and positive")

    def rates_for(
        self,
        *,
        service_tier: OpenAIServiceTier | None,
        input_tokens_total: int,
        regional_processing: bool,
    ) -> OpenAIRates:
        """Select token rates using reported response metadata."""
        priced_tier = _priced_tier(service_tier)
        if priced_tier == "default":
            rates = self.default
        elif priced_tier == "flex":
            rates = self.flex
        elif priced_tier in ("fast", "priority"):
            rates = self.fast
        else:
            rates = self.scale
        if rates is None:
            return _UNPRICED_RATES
        long_context = self.long_context
        if long_context is not None and input_tokens_total > long_context.input_tokens_above:
            rates = rates.multiplied(
                input_multiplier=long_context.input_multiplier,
                output_multiplier=long_context.output_multiplier,
            )
        if not regional_processing:
            return rates
        multiplier = self.regional_processing_multiplier
        if multiplier is None:
            return _UNPRICED_RATES
        return rates.multiplied(input_multiplier=multiplier, output_multiplier=multiplier)


def require_prompt_cache_options_support(
    *, model: str, automatic_prompt_caching: bool, supports_prompt_cache_options: bool
) -> None:
    """Require the prompt_cache_options parameter where a binding declines automatic caching.

    Both adapters' _precompute_fields call this: that parameter is the only thing carrying
    automatic_prompt_caching False to the wire, so a model built without it must refuse the
    binding rather than cache anyway at whatever the model charges for it.

    Raises:
        ValueError: the binding declines automatic caching and the model was built with
            supports_prompt_cache_options False.
    """
    if not automatic_prompt_caching and not supports_prompt_cache_options:
        raise ValueError(
            f"model {model!r} was built with supports_prompt_cache_options False, "
            "so prompt_cache_options is never sent. "
            "prompt_cache_options is what carries automatic_prompt_caching False to the wire. "
            "Bind automatic_prompt_caching=True, or set supports_prompt_cache_options True "
            "if the model accepts it."
        )


def parse_openai(failure: Exception) -> Verdict:
    """Map one OPENAI_FAILURE_TYPES exception to its verdict.

    A status 200 is a mid-stream error an adapter stream raised on a response the provider
    accepted, so _verdict_from_openai_error_code reads its code and the response's headers say
    nothing about this failure.
    Every other status goes to _verdict_from_openai_status, and the provider's own x-should-retry
    directive then overrides that verdict, through verdict_under_retry_directive, which states the
    rule.
    A retry-after header only fills a verdict's retry_after.
    A TransientError takes verdict_from_transient_error's shared mapping.
    Never raises: an Exception outside OPENAI_FAILURE_TYPES is DoNotRetry, counted as a fallthrough.
    """
    if isinstance(failure, TransientError):
        return verdict_from_transient_error(failure)
    if not isinstance(failure, openai.APIStatusError):
        record_parse_fallthrough(
            PARSE_FALLTHROUGH_COUNTS,
            parse_name="parse_openai",
            status_code=None,
            error_type=type(failure).__name__,
        )
        return DoNotRetry()
    retry_after = retry_after_seconds_from_headers(failure.response.headers)
    if failure.status_code == 200:
        return _verdict_from_openai_error_code(failure, retry_after)
    return verdict_under_retry_directive(
        _verdict_from_openai_status(failure, retry_after),
        headers=failure.response.headers,
        retry_after=retry_after,
    )


def _verdict_from_openai_status(
    failure: openai.APIStatusError, retry_after: float | None
) -> Verdict:
    """Return the verdict the status and the error code alone give one error-status failure.

    The listed rows come from openai's error-code guide,
    https://developers.openai.com/api/docs/guides/error-codes (read 2026-08-01):
    _PAUSE_STATUSES are PauseAll, except one whose error.code is in _SPEND_LIMIT_CODES, which is
    DoNotRetry because the guide states retrying those will not restore access;
    _RETRY_THIS_ONE_STATUSES are RetryThisOne and _DO_NOT_RETRY_STATUSES are DoNotRetry.
    Some rows come from the SDK rather than the guide; each table's docstring names which and why.
    error.code separates the spend-limit 429s; the guide notes the accompanying error.type can
    still read insufficient_quota, so the type separates nothing.
    Failures outside the rows take a default, counted in PARSE_FALLTHROUGH_COUNTS and logged:
    an unlisted 5xx is RetryThisOne, one attempt's server-side failure; any other unlisted
    status is DoNotRetry.
    """
    if failure.status_code in _PAUSE_STATUSES:
        if failure.code in _SPEND_LIMIT_CODES:
            return DoNotRetry()
        return PauseAll(retry_after=retry_after)
    if failure.status_code in _RETRY_THIS_ONE_STATUSES:
        return RetryThisOne(retry_after=retry_after)
    if failure.status_code in _DO_NOT_RETRY_STATUSES:
        return DoNotRetry()
    record_parse_fallthrough(
        PARSE_FALLTHROUGH_COUNTS,
        parse_name="parse_openai",
        status_code=failure.status_code,
        error_type=failure.type,
    )
    if failure.status_code >= 500:
        return RetryThisOne(retry_after=retry_after)
    return DoNotRetry()


def _verdict_from_openai_error_code(
    failure: openai.APIStatusError, retry_after: float | None
) -> Verdict:
    """Return the verdict a status-200 mid-stream error's code gives it.

    The code picks the verdict through _DISPOSITION_BY_ERROR_CODE: rate_limit_exceeded is
    PauseAll as a 429 is, any other transient code is RetryThisOne, and a terminal code is
    DoNotRetry, as is a code outside the table, counted in PARSE_FALLTHROUGH_COUNTS and logged.
    """
    disposition = None if failure.code is None else _DISPOSITION_BY_ERROR_CODE.get(failure.code)
    if disposition == "transient":
        if failure.code == "rate_limit_exceeded":
            return PauseAll(retry_after=retry_after)
        return RetryThisOne(retry_after=retry_after)
    if disposition is None:
        record_parse_fallthrough(
            PARSE_FALLTHROUGH_COUNTS,
            parse_name="parse_openai",
            status_code=failure.status_code,
            error_type=failure.code,
        )
    return DoNotRetry()


OPENAI_FAILURE_TYPES: tuple[type[Exception], ...] = (openai.APIStatusError, TransientError)
"""The exceptions parse_openai maps to a verdict; both adapters' failure_types.

APIStatusError catches every error status, not only the ones with their own subclass:
_make_status_error returns a specific subclass only for the statuses it lists and the bare
APIStatusError for every other one (openai 2.51.0), so a subclass list would silently drop
413, which openai maps to no class, and whatever status the provider adds next.
"""


def classify_openai(error: Exception) -> ErrorClassification:
    """Sort an exception parse_openai gave no verdict, or name the terminal error for a DoNotRetry.

    APIConnectionError, which APITimeoutError subclasses, carries no response: a transport
    failure that produced nothing parseable, transient.
    An APIStatusError only arrives here after parse_openai verdicted DoNotRetry, since every one is
    in OPENAI_FAILURE_TYPES; terminal_classification_from_response names what it becomes.
    Anything else the SDK raises is unknown_exception, which fails this item without a retry.
    """
    if isinstance(error, openai.APIConnectionError):
        return "transient"
    if not isinstance(error, openai.APIStatusError):
        return "unknown_exception"
    return terminal_classification_from_response(
        status_code=error.response.status_code,
        headers=error.response.headers,
    )


def request_id_from_openai_error(error: Exception) -> str | None:
    """Read the request-id header off the SDK exception.

    APIStatusError is the only openai exception carrying request_id, which it reads off the
    error response's headers, None where that response carried none (openai 2.48.0).
    """
    if isinstance(error, openai.APIStatusError):
        return error.request_id
    return None


PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS: Mapping[type, str] = {
    AsyncBedrockOpenAI: "aws.bedrock",
    AsyncAzureOpenAI: "azure.ai.openai",
}
"""Both adapters' provider_name_by_client_class. AsyncOpenAI is deliberately absent: it reaches
whatever its base_url points at.

The two classes here each speak one platform's auth and URL scheme, so the class fixes the
provider. A plain AsyncOpenAI does not: pointing its base_url at another vendor's
OpenAI-compatible endpoint is how Groq, DeepSeek, and xAI are reached, all of them
gen_ai.provider.name values.
Mapping AsyncOpenAI to "openai" would make Adapter.__init__ raise for every one of them.
"""
