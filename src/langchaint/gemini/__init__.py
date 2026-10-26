"""Construct Gemini `LLM` values with cataloged pricing.

Importing this subpackage requires `google-genai`.
`Gemini.llm` sends the stated model identifier verbatim.
`Gemini.llm` reaches the Gemini Developer API and reports `provider_name="gcp.gemini"`.
Vertex AI callers construct `GeminiGenerateContentAdapter` directly.
Use `provider_name="gcp.vertex_ai"` and Vertex pricing there.

Cataloged models receive `GEMINI_PRICING`, which states `ON_DEMAND` rates.
Uncataloged Gemini models require `pricing`.
Responses from traffic types without rates cost NaN.

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

from typing import Literal, overload

try:
    from google import genai
except ModuleNotFoundError as exc:
    if exc.name not in ("google", "google.genai"):
        raise
    raise ModuleNotFoundError(
        "langchaint's gemini backend requires google-genai; install langchaint[gemini]."
    ) from exc

from langchaint.billing.pricing import LongContextPricing
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.gemini.generate_content_adapter import (
    GeminiGenerateContentAdapter,
    GeminiPricingTable,
    GeminiRates,
    GeminiReportedServiceTier,
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
        on_demand=GeminiRates(
            input_tokens_cache_none=1.50,
            input_tokens_cache_read=0.15,
            output_tokens=7.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.5-flash": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=1.50,
            input_tokens_cache_read=0.15,
            output_tokens=9.00,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    # The pricing page lists no cache-read price for gemini-3.5-flash-lite, so the rate is NaN.
    "gemini-3.5-flash-lite": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=0.30,
            input_tokens_cache_read=float("nan"),
            output_tokens=2.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-flash-lite": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=0.25,
            input_tokens_cache_read=0.025,
            output_tokens=1.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-pro-preview": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=2.00,
            input_tokens_cache_read=0.20,
            output_tokens=12.00,
        ),
        long_context=LongContextPricing(
            prompt_tokens_above=200_000, input_multiplier=2.0, output_multiplier=1.5
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
        pricing: GeminiPricingTable | None = ...,
        service_tier: GeminiServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: GeminiPricingTable,
        service_tier: GeminiServiceTier | None = ...,
    ) -> LLM: ...

    def llm(
        self,
        model: str,
        *,
        pricing: GeminiPricingTable | None = None,
        service_tier: GeminiServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Gemini Developer API model.

        `model` is sent verbatim.
        Cataloged models receive `GEMINI_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `service_tier` sets the requested Gemini service tier.
        The reported traffic type selects pricing.

        `bind` on the returned `LLM`, and on every `BoundLLM` derived from it, raises `ValueError` when:
        - `parallel_tool_calls=False`, because `generateContent` cannot disable parallel function calls.
        - A system part sets `cache_breakpoint=True`.
        - `reasoning_level` differs only in letter case from a `types.ThinkingLevel` name, such as `"high"`.
          google-genai 2.33.0 would send the uppercase name instead.
        - `extra_body` has a `generationConfig` key whose value is not a mapping with only string keys.
          Every letter case and underscore spelling of `generationConfig` counts.
        - `provider_executed_tools` is nonempty and `tool_choice` is not `"auto"`.
        - `provider_executed_tools` is nonempty and the model is not a Gemini 3 model.
        - A provider-executed tool sets no field or sets an unsupported field.
          The supported fields are `code_execution`, `file_search`, `google_maps`, `google_search`, and `url_context`.
        - Google Search enables image search.
        - Google Search is bound and `pricing.google_search_usd_per_query` is `None`.
        - Google Maps is bound and `pricing.google_maps_usd_per_query` is `None`.
        `bind` raises `pydantic.ValidationError` when a provider-executed tool fails `types.Tool` validation.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when `client` reaches Vertex AI.
        """
        if pricing is None:
            pricing = _PRICING_BY_MODEL_ID.get(model)
            if pricing is None:
                raise ValueError(
                    f"model {model!r} is not in GEMINI_PRICING; pass pricing= stating its rates"
                )
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
    "GeminiPricingTable",
    "GeminiRates",
    "GeminiReportedServiceTier",
    "GeminiServiceTier",
    "assembled_response",
]
