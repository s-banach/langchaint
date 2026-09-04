"""Share openai clients, errors, and pricing.

The Responses and Chat Completions adapters use the same clients, exceptions, and service tiers.
This module imports neither adapter nor private SDK modules.
"""

import base64
from abc import ABC
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Literal, override

import openai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI
from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    ErrorClassification,
    record_parse_fallthrough,
    retry_after_seconds_from_headers,
    terminal_classification_from_response,
    verdict_from_transient_error,
    verdict_under_retry_directive,
)
from langchaint.exceptions import TransientError
from langchaint.messages import ImagePart
from langchaint.pricing import Billing, ProviderBilling, category_cost
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
"""The statuses that reject this request.

A resend fails the same way.
"""

PARSE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""`record_parse_fallthrough` increments this counter for each status-family default."""

type OpenAIServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]
"""What a Chat Completions request may ask for (openai 3.1.0)."""

type OpenAIResponsesServiceTier = OpenAIServiceTier | Literal["ultrafast"]
"""What a Responses request may ask for and what a Response may report (openai 3.1.0).

The reported value selects pricing because it may differ from the requested value.
"""

type _OpenAINormalizedServiceTier = Literal[
    "default", "flex", "scale", "priority", "fast", "ultrafast"
]
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
    "misalignment_policy_violation": "terminal",
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


def _image_data_uri(image_part: ImagePart) -> str:
    encoded_data = base64.b64encode(image_part.data).decode("ascii")
    return f"data:{image_part.media_type};base64,{encoded_data}"


def _priced_tier(
    service_tier: OpenAIResponsesServiceTier | None,
) -> _OpenAINormalizedServiceTier:
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
    ) -> ProviderBilling:
        """Price one response's counters at these rates.

        output_tokens_reasoning is the reasoning share of output_tokens.
        The output rate applies to output_tokens_reasoning.
        The returned Usage carries output_tokens_reasoning.

        Token cost is the sum of four token category costs.
        provider_executed_tool_cost_in_usd adds the provider-executed tool charge.

        Raises:
            pydantic.ValidationError: a counter is negative.
        """
        return ProviderBilling(
            billing=Billing(
                usage=Usage(
                    input_tokens_cache_read=input_tokens_cache_read,
                    input_tokens_cache_write=input_tokens_cache_write,
                    input_tokens_cache_none=input_tokens_cache_none,
                    output_tokens=output_tokens,
                    output_tokens_reasoning=output_tokens_reasoning,
                    input_tokens_cache_read_cost_in_usd=category_cost(
                        input_tokens_cache_read,
                        usd_per_million_tokens=self.cache_read_usd_per_million_tokens,
                    ),
                    input_tokens_cache_write_cost_in_usd=category_cost(
                        input_tokens_cache_write,
                        usd_per_million_tokens=self.cache_write_usd_per_million_tokens,
                    ),
                    input_tokens_cache_none_cost_in_usd=category_cost(
                        input_tokens_cache_none,
                        usd_per_million_tokens=self.input_cache_none_usd_per_million_tokens,
                    ),
                    output_tokens_cost_in_usd=category_cost(
                        output_tokens,
                        usd_per_million_tokens=self.output_usd_per_million_tokens,
                    ),
                    provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
                ),
                service_tier=service_tier,
                input_cache_none_usd_per_million_tokens=self.input_cache_none_usd_per_million_tokens,
                cache_read_usd_per_million_tokens=self.cache_read_usd_per_million_tokens,
                cache_write_usd_per_million_tokens=self.cache_write_usd_per_million_tokens,
                output_usd_per_million_tokens=self.output_usd_per_million_tokens,
            ),
            usage_raw=usage_raw,
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
    ultrafast: OpenAIRates | None = None
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
        service_tier: OpenAIResponsesServiceTier | None,
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
        elif priced_tier == "scale":
            rates = self.scale
        else:
            rates = self.ultrafast
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
    *, model: str, automatic_cache_breakpoints: bool, supports_prompt_cache_options: bool
) -> None:
    """Require `prompt_cache_options` when `automatic_cache_breakpoints=False`.

    `prompt_cache_options` carries `automatic_cache_breakpoints=False` to the request.
    Both OpenAI adapters call this before building fields.

    Raises:
        ValueError: `automatic_cache_breakpoints=False` lacks `prompt_cache_options` support.
    """
    if not automatic_cache_breakpoints and not supports_prompt_cache_options:
        raise ValueError(
            f"model {model!r} was built with supports_prompt_cache_options=False, "
            "so prompt_cache_options is never sent. "
            "prompt_cache_options carries automatic_cache_breakpoints=False to the request. "
            "Bind automatic_cache_breakpoints=True, or set supports_prompt_cache_options=True "
            "if the model accepts it."
        )


def parse_openai(failure: Exception) -> Verdict:
    """Map one OPENAI_FAILURE_TYPES exception to its verdict.

    Status 200 identifies a mid-stream error, so its code selects the verdict.
    Other statuses use `_verdict_from_openai_status` before `x-should-retry` overrides it.
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

    Source: https://developers.openai.com/api/docs/guides/error-codes.
    Read 2026-08-01.
    `_PAUSE_STATUSES` return `PauseAll` unless `_SPEND_LIMIT_CODES` requires `DoNotRetry`.
    Retrying spend-limit errors cannot restore access.
    `_RETRY_THIS_ONE_STATUSES` return `RetryThisOne`.
    `_DO_NOT_RETRY_STATUSES` return `DoNotRetry`.
    Some rows come from the SDK.
    Each table's docstring names the source and reason.
    `error.code` separates spend-limit 429 errors because `error.type` may still be `insufficient_quota`.
    Failures outside the rows take a default, counted in PARSE_FALLTHROUGH_COUNTS and logged:
    unlisted 5xx statuses return `RetryThisOne`, and other unlisted statuses return `DoNotRetry`.
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

    `rate_limit_exceeded` returns `PauseAll`.
    Other transient codes return `RetryThisOne`.
    Terminal and unknown codes return `DoNotRetry`.
    Unknown codes increment `PARSE_FALLTHROUGH_COUNTS` and are logged.
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
"""The exceptions parse_openai maps to a verdict.

Both adapters use these exceptions as `failure_types`.

APIStatusError catches every error status.
"""


def classify_openai(error: Exception) -> ErrorClassification:
    """Sort an exception parse_openai gave no verdict, or name the terminal error for a DoNotRetry.

    `APIConnectionError` is a transient transport failure without a response.
    `APITimeoutError` is an `APIConnectionError` subclass.
    `APIStatusError` reaches this function only after `parse_openai` returns `DoNotRetry`.
    Other exceptions return `unknown_exception`.
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

    `APIStatusError` alone carries `request_id` from response headers in openai 2.48.0.
    """
    if isinstance(error, openai.APIStatusError):
        return error.request_id
    return None


PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS: Mapping[type, str] = {
    AsyncBedrockOpenAI: "aws.bedrock",
    AsyncAzureOpenAI: "azure.ai.openai",
}
"""The client-class provider map shared by both OpenAI adapters.

`AsyncAzureOpenAI` and `AsyncBedrockOpenAI` determine their providers.
`AsyncOpenAI` is absent so a caller can state the provider for an OpenAI-compatible endpoint.
"""


class _OpenAIGenerationAdapterBase(Adapter, ABC):
    """Share OpenAI provider validation and failure handling."""

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = (
        PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS
    )

    failure_types: ClassVar[tuple[type[Exception], ...]] = OPENAI_FAILURE_TYPES

    @override
    def parse(self, failure: Exception) -> Verdict:
        return parse_openai(failure)

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        return classify_openai(error)

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        return request_id_from_openai_error(error)
