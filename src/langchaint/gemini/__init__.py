"""The gemini backend: the generateContent adapter, its model catalog, and pricing.

Importing this subpackage requires the google-genai package;
the import below raises a ModuleNotFoundError naming the package to install.

gemini_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the generateContent adapter and wraps it
in an LLM reaching the Gemini Developer API.
Vertex AI callers construct GeminiGenerateContentAdapter directly with genai.Client(vertexai=True)
and provider_name "gcp.vertex_ai"; Vertex prices differ from this catalog, so pass pricing with it.
pricing is a mapping from the traffic_type a response reports to the GeminiPricingTable that prices
it, merged over a cataloged model's public prices under "ON_DEMAND", so a caller adds a tier or
replaces the on-demand rates and omitting it prices at the public rates. A response served at a
traffic_type the mapping does not hold costs NaN.
gemini_model requires pricing for an uncataloged id, having no table to fall back on.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://ai.google.dev/gemini-api/docs/pricing
(read 2026-08-03).
Prices are the one provider fact langchaint cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
The catalog carries the text, image, and video rates; langchaint sends no audio, whose rates differ.
Cache storage per token-hour is a charge on an explicit cache resource, not on a request, so no
table field carries it.
"""

from collections.abc import Mapping
from typing import Literal, overload

try:
    from google import genai
except ModuleNotFoundError as exc:
    if exc.name not in ("google", "google.genai"):
        raise
    raise ModuleNotFoundError(
        "langchaint's gemini backend requires the google-genai package; install google-genai."
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
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
"""Model identifiers with public prices in GEMINI_PRICING."""

GEMINI_PRICING: dict[GeminiModelName, GeminiPricingTable] = {
    "gemini-3.6-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=1.50,
            cache_read_usd_per_million_tokens=0.15,
            output_usd_per_million_tokens=7.50,
        )
    ),
    "gemini-3.5-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=1.50,
            cache_read_usd_per_million_tokens=0.15,
            output_usd_per_million_tokens=9.00,
        )
    ),
    # the pricing page lists no cache-read price for gemini-3.5-flash-lite, so it is NaN
    "gemini-3.5-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=0.30,
            cache_read_usd_per_million_tokens=float("nan"),
            output_usd_per_million_tokens=2.50,
        )
    ),
    "gemini-3.1-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=0.25,
            cache_read_usd_per_million_tokens=0.025,
            output_usd_per_million_tokens=1.50,
        )
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
    ),
    "gemini-2.5-pro": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=1.25,
            cache_read_usd_per_million_tokens=0.125,
            output_usd_per_million_tokens=10.00,
        ),
        long_prompt_threshold_tokens=200_000,
        long_prompt_rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=2.50,
            cache_read_usd_per_million_tokens=0.25,
            output_usd_per_million_tokens=15.00,
        ),
    ),
    "gemini-2.5-flash": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=0.30,
            cache_read_usd_per_million_tokens=0.03,
            output_usd_per_million_tokens=2.50,
        )
    ),
    "gemini-2.5-flash-lite": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=0.10,
            cache_read_usd_per_million_tokens=0.01,
            output_usd_per_million_tokens=0.40,
        )
    ),
}
"""Public on-demand prices per gemini model; the default pricing lookup."""

_PRICING_BY_MODEL_ID = dict[str, GeminiPricingTable](GEMINI_PRICING.items())
"""GEMINI_PRICING under a str key, so gemini_model can look up a possibly-uncataloged model id."""


@overload
def gemini_model(
    model: GeminiModelName,
    *,
    client: genai.Client | None = ...,
    pricing: Mapping[str, GeminiPricingTable] | None = ...,
    service_tier: GeminiServiceTier | None = ...,
    shared_backoff: SharedBackoff | None = ...,
    max_attempts: int = ...,
) -> LLM: ...
@overload
def gemini_model(
    model: str,
    *,
    pricing: Mapping[str, GeminiPricingTable],
    client: genai.Client | None = ...,
    service_tier: GeminiServiceTier | None = ...,
    shared_backoff: SharedBackoff | None = ...,
    max_attempts: int = ...,
) -> LLM: ...
def gemini_model(
    model: str,
    *,
    client: genai.Client | None = None,
    pricing: Mapping[str, GeminiPricingTable] | None = None,
    service_tier: GeminiServiceTier | None = None,
    shared_backoff: SharedBackoff | None = None,
    max_attempts: int = 3,
) -> LLM:
    """Build a ready LLM for one model on the Gemini Developer API's generateContent.

    model is sent verbatim.
    client None constructs genai.Client(vertexai=False), which reads GOOGLE_API_KEY or
    GEMINI_API_KEY from the environment; construction is offline.
    pricing holds one table per traffic_type word a response reports
    (GeminiPricedServiceTier names the vocabulary).
    On a cataloged id it is merged over {"ON_DEMAND": GEMINI_PRICING[model]}, so a caller adds a
    tier or replaces the on-demand rates and omitting it prices at the public rates; on any other
    id it is passed through and needs its "ON_DEMAND" key.
    A response served at a traffic_type this mapping does not hold costs NaN.
    service_tier is what the request asks for, None sending nothing; it is not what prices the
    response, since the request and response tier vocabularies share no word, so a "flex" request
    needs a pricing entry for the traffic_type its responses report ("ON_DEMAND_FLEX").
    shared_backoff and max_attempts have the LLM.__init__ meanings;
    pass one SharedBackoff across models on the same account so a rate limit pauses them together,
    and note one instance serves one event loop.

    Raises:
        ValueError: model is outside the catalog and pricing is missing.
            LLM.__init__ raises it when max_attempts is not a positive int, and
            GeminiGenerateContentAdapter.__init__ when pricing has no "ON_DEMAND" key or client
            was constructed with vertexai=True, which reaches "gcp.vertex_ai" rather than the
            "gcp.gemini" provider this constructor states.
    """
    catalog_table = _PRICING_BY_MODEL_ID.get(model)
    if catalog_table is None:
        if pricing is None:
            raise ValueError(
                f"model {model!r} is not in GEMINI_PRICING; pass pricing= stating its rates"
            )
    else:
        pricing = {"ON_DEMAND": catalog_table, **(pricing or {})}
    return LLM(
        GeminiGenerateContentAdapter(
            client=client if client is not None else genai.Client(vertexai=False),
            model=model,
            pricing=pricing,
            provider_name="gcp.gemini",
            service_tier=service_tier,
        ),
        shared_backoff=shared_backoff,
        max_attempts=max_attempts,
    )


__all__ = [
    "GEMINI_PRICING",
    "GeminiGenerateContentAdapter",
    "GeminiModelName",
    "GeminiPricedServiceTier",
    "GeminiPricingTable",
    "GeminiRates",
    "GeminiServiceTier",
    "assembled_response",
    "gemini_model",
    "parse_gemini",
]
