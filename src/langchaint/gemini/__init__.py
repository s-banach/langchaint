"""Construct Gemini `LLM` values with cataloged pricing.

Importing this subpackage requires `google-genai`.
`Gemini.model` sends the stated model identifier verbatim.
`Gemini.model` reaches the Gemini Developer API and reports `provider_name="gcp.gemini"`.
Vertex AI callers construct `GeminiGenerateContentAdapter` directly.
Use `provider_name="gcp.vertex_ai"` and Vertex pricing there.

Cataloged models receive `ON_DEMAND` rates from `GEMINI_PRICING`.
Uncataloged Gemini models require `pricing`.
Responses from uncataloged traffic types cost NaN.

Token prices use USD per one million tokens.
Google Search prices use USD per query.
Google Maps prices use USD per query.
Source: https://ai.google.dev/gemini-api/docs/pricing, read 2026-08-03.
Maps source: https://ai.google.dev/gemini-api/docs/maps-grounding.
Recheck the sources before relying on a table.
`GEMINI_PRICING` carries text, image, and video rates.
Catalog tool rates estimate post-quota list prices.
`Gemini.model(pricing=...)` replaces cataloged estimates.
langchaint sends no audio.
Explicit cache-resource storage charges have no request `Usage` field.
"""

from collections.abc import Mapping
from typing import Literal, overload

try:
    from google import genai
except ModuleNotFoundError as exc:
    if exc.name not in ("google", "google.genai"):
        raise
    raise ModuleNotFoundError(
        "langchaint's gemini backend requires google-genai; install langchaint[gemini]."
    ) from exc

from langchaint.gemini.generate_content_adapter import (
    GeminiGenerateContentAdapter,
    GeminiPricedServiceTier,
    GeminiPricingTable,
    GeminiRates,
    GeminiServiceTier,
    assembled_response,
    parse_gemini,
)
from langchaint.llm import LLM
from langchaint.shared_backoff import SharedBackoff

type GeminiModelName = Literal[
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]
"""Model identifiers with public prices in GEMINI_PRICING."""

GEMINI_PRICING: dict[GeminiModelName, GeminiPricingTable] = {
    "gemini-3.6-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=1.50,
            cache_read_usd_per_million_tokens=0.15,
            output_usd_per_million_tokens=7.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.5-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=1.50,
            cache_read_usd_per_million_tokens=0.15,
            output_usd_per_million_tokens=9.00,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    # The pricing page lists no cache-read price for gemini-3.5-flash-lite, so the rate is NaN.
    "gemini-3.5-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=0.30,
            cache_read_usd_per_million_tokens=float("nan"),
            output_usd_per_million_tokens=2.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=0.25,
            cache_read_usd_per_million_tokens=0.025,
            output_usd_per_million_tokens=1.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-pro-preview": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=2.00,
            cache_read_usd_per_million_tokens=0.20,
            output_usd_per_million_tokens=12.00,
        ),
        long_prompt_threshold_tokens=200_000,
        long_prompt_rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=4.00,
            cache_read_usd_per_million_tokens=0.40,
            output_usd_per_million_tokens=18.00,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
}
"""Public on-demand prices that `Gemini.model` uses by default."""

_PRICING_BY_MODEL_ID = dict[str, GeminiPricingTable](GEMINI_PRICING.items())
"""`GEMINI_PRICING` with `str` keys for runtime model lookup."""


class Gemini:
    """Create `LLM` values for Gemini."""

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        max_concurrent_requests: int | None = 8,
        max_request_starts_per_second: float = 50.0,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
    ) -> None:
        """Build `Gemini` without sending a request.

        `client=None` constructs `genai.Client(vertexai=False)`.
        A passed `client` must reach the Gemini Developer API.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the initial and minimum wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            ValueError: `client` is absent and no API key is available.
                Also raised when a `SharedBackoff` setting is invalid.
        """
        self._shared_backoff = SharedBackoff(
            parse=parse_gemini,
            failure_types=GeminiGenerateContentAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client: genai.Client = client if client is not None else genai.Client(vertexai=False)

    @overload
    def model(
        self,
        model: GeminiModelName,
        *,
        pricing: Mapping[str, GeminiPricingTable] | None = ...,
        service_tier: GeminiServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def model(
        self,
        model: str,
        *,
        pricing: Mapping[str, GeminiPricingTable],
        service_tier: GeminiServiceTier | None = ...,
    ) -> LLM: ...

    def model(
        self,
        model: str,
        *,
        pricing: Mapping[str, GeminiPricingTable] | None = None,
        service_tier: GeminiServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Gemini Developer API model.

        `model` is sent verbatim.
        Cataloged models receive `ON_DEMAND` rates from `GEMINI_PRICING`.
        Stated `pricing` extends or replaces catalog pricing.
        Uncataloged models require `pricing` with an `"ON_DEMAND"` entry.
        `service_tier` sets the requested Gemini service tier.
        The reported traffic type selects pricing.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when `pricing` lacks `"ON_DEMAND"`.
                Also raised when `client` reaches Vertex AI.
        """
        catalog_table = _PRICING_BY_MODEL_ID.get(model)
        if catalog_table is None:
            if pricing is None:
                raise ValueError(
                    f"model {model!r} is not in GEMINI_PRICING; pass pricing= stating its rates"
                )
        else:
            pricing = {"ON_DEMAND": catalog_table, **(pricing or {})}
        adapter = GeminiGenerateContentAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="gcp.gemini",
            service_tier=service_tier,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)


__all__ = [
    "GEMINI_PRICING",
    "Gemini",
    "GeminiGenerateContentAdapter",
    "GeminiModelName",
    "GeminiPricedServiceTier",
    "GeminiPricingTable",
    "GeminiRates",
    "GeminiServiceTier",
    "assembled_response",
    "parse_gemini",
]
