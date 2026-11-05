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
    retry_after_seconds_from_headers,
    sdk_retry_decision,
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

REQUEST_FAILURE_FALLTHROUGH_COUNTS: Counter[str] = Counter()
"""Counts mid-stream errors whose code is missing or absent from `_DISPOSITION_BY_ERROR_CODE`."""

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


def openai_request_failure(error: Exception, *, client: openai.AsyncOpenAI) -> RequestFailure:
    """Return the `RequestFailure` of one exception an OpenAI request raised.

    `APIConnectionError` is a transient transport failure without a response, and pauses nothing.
    `APITimeoutError` is an `APIConnectionError` subclass.
    Status 200 identifies a mid-stream error, so `_openai_error_code_answers` reads its code.
    The client's `_should_retry` decides retries for every other status.
    openai 3.46.0 declines retries when `Retry-After` exceeds 120 seconds, and retries spend-limit 429s.
    Statuses 429 and 503 pause the quota independently of the retry decision.
    A failed SDK retry decision is logged and returns `unknown_exception`.
    Other exceptions are `unknown_exception`.
    """
    if isinstance(error, openai.APIConnectionError):
        return RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None)
    if not isinstance(error, openai.APIStatusError):
        return RequestFailure(
            kind="unknown_exception", pauses_quota=False, retry_after_seconds=None
        )
    if error.status_code == 200:
        retries, pauses_quota = _openai_error_code_answers(error)
    else:
        retries = sdk_retry_decision(lambda: client._should_retry(error.response))
        if isinstance(retries, RequestFailure):
            return retries
        pauses_quota = error.status_code in _PAUSE_STATUSES
    return request_failure_from_response(
        status_code=error.status_code,
        retry_after_seconds=retry_after_seconds_from_headers(error.response.headers),
        retries=retries,
        pauses_quota=pauses_quota,
    )


def _openai_error_code_answers(error: openai.APIStatusError) -> tuple[bool, bool]:
    """Return whether a status-200 mid-stream error retries and whether it pauses the quota, by its code.

    Transient codes retry, and `rate_limit_exceeded` also pauses the quota.
    Terminal, unknown, and missing codes neither retry nor pause.
    Unknown and missing codes increment `REQUEST_FAILURE_FALLTHROUGH_COUNTS` and are logged.
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

    `APIStatusError` alone carries `request_id` from response headers in openai 3.44.0.
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

    def __init__(
        self,
        *,
        client: openai.AsyncOpenAI,
        model: str,
        provider_name: str,
        automatic_cache_breakpoints_default: bool,
    ) -> None:
        """Validate the provider and store the SDK client with retries disabled.

        Raises:
            ValueError: The client reaches another known provider.
        """
        super().__init__(
            client=client,
            model=model,
            provider_name=provider_name,
            automatic_cache_breakpoints_default=automatic_cache_breakpoints_default,
        )
        self.client: openai.AsyncOpenAI = client_without_retries(client)

    @override
    def request_failure(self, error: Exception) -> RequestFailure:
        return openai_request_failure(error, client=self.client)

    @override
    def request_id_from_error(self, error: Exception) -> str | None:
        return request_id_from_openai_error(error)
