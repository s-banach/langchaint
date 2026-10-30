"""Share openai clients, errors, and pricing.

The Responses and Chat Completions adapters use the same clients, exceptions, and service tiers.
This module imports neither adapter nor private SDK modules.
"""

import base64
from abc import ABC
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, override

import openai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI

from langchaint.adapter import (
    Adapter,
    RequestFailure,
    record_request_failure_fallthrough,
    request_failure_from_response,
)
from langchaint.billing.pricing import UNPRICED_TOKEN_RATES, LongContextPricing, TokenRates
from langchaint.common.messages import ImagePart

_PAUSE_STATUSES = frozenset({429, 503})
"""429 rate limits and documented 503 forms throttle the rate-limit quota.

Every request sharing the rate-limit quota pauses.
The SDK's `api_reference/openapi.transformed.yml` (openai 3.17.0) documents only these
two error statuses on chat completions, responses, and embeddings.
It describes both as retryable after an optional integer-seconds `Retry-After` header.
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

REQUEST_FAILURE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""`record_request_failure_fallthrough` increments this counter for each status-family default."""

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
class OpenAIPricingTable:
    """OpenAI rates and modifiers for one model.

    A `priority` response uses the `fast` rates, and a response that reports no tier or `auto` uses the `default` rates.
    A response at a tier without rates costs NaN in every nonzero token category.
    """

    default: TokenRates
    flex: TokenRates | None = None
    fast: TokenRates | None = None
    ultrafast: TokenRates | None = None
    scale: TokenRates | None = None
    long_context: LongContextPricing | None = None
    web_search_usd_per_invocation: float | None = None
    file_search_usd_per_invocation: float | None = None

    def scaled(self, multiplier: float) -> "OpenAIPricingTable":
        """Return the table with every token rate multiplied by `multiplier`.

        Long-context multipliers, the regional multiplier, and per-invocation prices are unchanged.
        """

        def scaled_rates(rates: TokenRates | None) -> TokenRates | None:
            if rates is None:
                return None
            return rates.scaled(input_multiplier=multiplier, output_multiplier=multiplier)

        return replace(
            self,
            default=self.default.scaled(input_multiplier=multiplier, output_multiplier=multiplier),
            flex=scaled_rates(self.flex),
            fast=scaled_rates(self.fast),
            ultrafast=scaled_rates(self.ultrafast),
            scale=scaled_rates(self.scale),
        )

    def rates_for(
        self, *, service_tier: _OpenAINormalizedServiceTier, input_tokens_total: int
    ) -> TokenRates:
        """Select the token rates for the normalized reported `service_tier` and the request's `input_tokens_total`."""
        match service_tier:
            case "default":
                rates = self.default
            case "flex":
                rates = self.flex
            case "fast" | "priority":
                rates = self.fast
            case "scale":
                rates = self.scale
            case "ultrafast":
                rates = self.ultrafast
            case _:
                rates = None
        if rates is None:
            return UNPRICED_TOKEN_RATES
        if self.long_context is None:
            return rates
        return self.long_context.applied_to(rates, prompt_tokens=input_tokens_total)


def require_prompt_cache_options_support(
    *, model: str, automatic_cache_breakpoints: bool, supports_cache_breakpoints: bool
) -> None:
    """Require `prompt_cache_options` when `automatic_cache_breakpoints=False`.

    `prompt_cache_options` carries `automatic_cache_breakpoints=False` to the request.
    Both OpenAI adapters call this before building fields.

    Raises:
        ValueError: `automatic_cache_breakpoints=False` lacks `prompt_cache_options` support.
    """
    if not automatic_cache_breakpoints and not supports_cache_breakpoints:
        raise ValueError(
            f"model {model!r} was built with supports_cache_breakpoints=False, "
            "so prompt_cache_options is never sent. "
            "prompt_cache_options carries automatic_cache_breakpoints=False to the request. "
            "Bind automatic_cache_breakpoints=True, or set supports_cache_breakpoints=True "
            "if the model accepts it."
        )


def openai_request_failure(error: Exception) -> RequestFailure:
    """Return the `RequestFailure` of one exception an OpenAI request raised.

    `APIConnectionError` is a transient transport failure without a response, and pauses nothing.
    `APITimeoutError` is an `APIConnectionError` subclass.
    Status 200 identifies a mid-stream error, so `_openai_error_code_answers` reads its code.
    `_openai_status_answers` reads every other status.
    `request_failure_from_response` builds the failure from those answers.
    Other exceptions are `unknown_exception`.
    """
    if isinstance(error, openai.APIConnectionError):
        return RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None)
    if not isinstance(error, openai.APIStatusError):
        return RequestFailure(
            kind="unknown_exception", pauses_quota=False, retry_after_seconds=None
        )
    retries, pauses_quota = (
        _openai_error_code_answers(error)
        if error.status_code == 200
        else _openai_status_answers(error)
    )
    return request_failure_from_response(
        status_code=error.status_code,
        headers=error.response.headers,
        retries=retries,
        pauses_quota=pauses_quota,
    )


def _openai_status_answers(error: openai.APIStatusError) -> tuple[bool, bool]:
    """Return whether one error-status failure retries and whether it pauses the quota, by status and code.

    Source: https://developers.openai.com/api/docs/guides/error-codes.
    Read 2026-08-01.
    `_PAUSE_STATUSES` retry and pause unless `_SPEND_LIMIT_CODES` forbids a retry.
    Retrying spend-limit errors cannot restore access.
    `_RETRY_THIS_ONE_STATUSES` retry without a pause.
    `_DO_NOT_RETRY_STATUSES` neither retry nor pause.
    Some rows come from the SDK.
    Each table's docstring names the source and reason.
    `error.code` separates spend-limit 429 errors because `error.type` may still be `insufficient_quota`.
    Failures outside the rows take a default, counted in REQUEST_FAILURE_FALLTHROUGH_COUNTS and logged:
    unlisted 5xx statuses retry without a pause, and other unlisted statuses neither retry nor pause.
    """
    if error.status_code in _PAUSE_STATUSES:
        if error.code in _SPEND_LIMIT_CODES:
            return False, False
        return True, True
    if error.status_code in _RETRY_THIS_ONE_STATUSES:
        return True, False
    if error.status_code in _DO_NOT_RETRY_STATUSES:
        return False, False
    record_request_failure_fallthrough(
        REQUEST_FAILURE_FALLTHROUGH_COUNTS,
        function_name="openai_request_failure",
        status_code=error.status_code,
        error_type=error.type,
    )
    return error.status_code >= 500, False


def _openai_error_code_answers(error: openai.APIStatusError) -> tuple[bool, bool]:
    """Return whether a status-200 mid-stream error retries and whether it pauses the quota, by its code.

    Transient codes retry, and `rate_limit_exceeded` also pauses the quota.
    Terminal and unknown codes neither retry nor pause.
    Unknown codes increment `REQUEST_FAILURE_FALLTHROUGH_COUNTS` and are logged.
    """
    disposition = None if error.code is None else _DISPOSITION_BY_ERROR_CODE.get(error.code)
    if disposition is None:
        record_request_failure_fallthrough(
            REQUEST_FAILURE_FALLTHROUGH_COUNTS,
            function_name="openai_request_failure",
            status_code=error.status_code,
            error_type=error.code,
        )
    if disposition == "transient":
        return True, error.code == "rate_limit_exceeded"
    return False, False


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

    @override
    def request_failure(self, error: Exception) -> RequestFailure:
        return openai_request_failure(error)

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        return request_id_from_openai_error(error)
