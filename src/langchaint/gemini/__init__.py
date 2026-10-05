"""Construct Gemini `LLM` values with cataloged pricing.

Importing this subpackage requires `google-genai`.
`Gemini.llm` sends the stated model identifier verbatim.
`Gemini.llm` reaches the Gemini Developer API and reports `provider_name="gcp.gemini"`.
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
`Gemini.llm(pricing=...)` replaces cataloged estimates.
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

from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.gemini.generate_content_adapter import (
    GeminiGenerateContentAdapter,
    GeminiPricedServiceTier,
    GeminiPricingTable,
    GeminiRates,
    GeminiServiceTier,
    assembled_response,
)
from langchaint.generation.llm import LLM
from langchaint.generation.observer import Observer

type GeminiModelId = Literal[
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]
"""Model identifiers with public prices in GEMINI_PRICING."""

GEMINI_PRICING: dict[GeminiModelId, GeminiPricingTable] = {
    "gemini-3.6-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_tokens_cache_none=1.50,
            input_tokens_cache_read=0.15,
            output_tokens=7.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.5-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_tokens_cache_none=1.50,
            input_tokens_cache_read=0.15,
            output_tokens=9.00,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    # The pricing page lists no cache-read price for gemini-3.5-flash-lite, so the rate is NaN.
    "gemini-3.5-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_tokens_cache_none=0.30,
            input_tokens_cache_read=float("nan"),
            output_tokens=2.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_tokens_cache_none=0.25,
            input_tokens_cache_read=0.025,
            output_tokens=1.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-pro-preview": GeminiPricingTable(
        rates=GeminiRates(
            input_tokens_cache_none=2.00,
            input_tokens_cache_read=0.20,
            output_tokens=12.00,
        ),
        long_context_prompt_token_count_above=200_000,
        long_context_rates=GeminiRates(
            input_tokens_cache_none=4.00,
            input_tokens_cache_read=0.40,
            output_tokens=18.00,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
}
"""Public on-demand prices that `Gemini.llm` uses by default."""

_PRICING_BY_MODEL_ID = dict[str, GeminiPricingTable](GEMINI_PRICING.items())
"""`GEMINI_PRICING` with `str` keys for runtime model lookup."""


class Gemini:
    """Create `LLM` values for Gemini."""

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `Gemini` without sending a request.

        `client=None` constructs `genai.Client(vertexai=False)`.
        A passed `client` must reach the Gemini Developer API.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` is absent and no API key is available.
        """
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: genai.Client = client if client is not None else genai.Client(vertexai=False)

    @overload
    def llm(
        self,
        model: GeminiModelId,
        *,
        pricing: Mapping[str, GeminiPricingTable] | None = ...,
        service_tier: GeminiServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: Mapping[str, GeminiPricingTable],
        service_tier: GeminiServiceTier | None = ...,
    ) -> LLM: ...

    def llm(
        self,
        model: str,
        *,
        pricing: Mapping[str, GeminiPricingTable] | None = None,
        service_tier: GeminiServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Gemini Developer API model.

        `model` is sent verbatim.
        Cataloged models receive `ON_DEMAND` rates from `GEMINI_PRICING`.
        Stated `pricing` replaces catalog pricing and must contain an `"ON_DEMAND"` entry.
        Uncataloged models require `pricing`.
        `service_tier` sets the requested Gemini service tier.
        The reported traffic type selects pricing.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when `pricing` lacks `"ON_DEMAND"`.
                Also raised when `client` reaches Vertex AI.
        """
        if pricing is None:
            catalog_table = _PRICING_BY_MODEL_ID.get(model)
            if catalog_table is None:
                raise ValueError(
                    f"model {model!r} is not in GEMINI_PRICING; pass pricing= stating its rates"
                )
            pricing = {"ON_DEMAND": catalog_table}
        adapter = GeminiGenerateContentAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="gcp.gemini",
            service_tier=service_tier,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff, observer=self._observer)


__all__ = [
    "GEMINI_PRICING",
    "Gemini",
    "GeminiGenerateContentAdapter",
    "GeminiModelId",
    "GeminiPricedServiceTier",
    "GeminiPricingTable",
    "GeminiRates",
    "GeminiServiceTier",
    "assembled_response",
]
