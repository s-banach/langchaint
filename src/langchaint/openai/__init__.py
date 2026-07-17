"""The openai backend: the Responses adapter, its model catalog, and pricing.

Importing this subpackage requires the openai package (install langchaint[openai]);
the import below raises a ModuleNotFoundError naming the extra to install.

openai_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the Responses adapter and wraps it in an LLM.
client None constructs the native SDK client, which reads credentials from the environment;
Bedrock routing is passing an AsyncBedrockOpenAI client instead.
pricing None selects the model's public prices from OPENAI_PRICING.
Pass your own PricingTable to override, for example when your account bills at a custom rate.
cost_breakdown(usage_raw, pricing) reports the exact per-category cost of one response from its raw
SDK usage, through the same arithmetic that produced the stored Usage.cost_in_usd.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://developers.openai.com/api/docs/pricing.
Prices are the one provider fact this package cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
OpenAI has no 1-hour cache tier, and only the gpt-5.6 family bills cache writes;
earlier models cache automatically with free writes, so their tables carry a zero cache-write rate.
The bare gpt-5.6 model identifier is an alias for gpt-5.6-sol; the catalog uses the explicit identifier.
"""

from typing import Literal

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires the openai package; install langchaint[openai]."
    ) from exc

from langchaint.llm import LLM
from langchaint.openai.responses_provider import OpenAIResponsesProvider, cost_breakdown
from langchaint.provider import PricingTable
from langchaint.rate_limiter import RateLimiter

type OpenAIModelName = Literal[
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]
"""Model identifiers with public prices in OPENAI_PRICING."""

OPENAI_PRICING: dict[OpenAIModelName, PricingTable] = {
    "gpt-5.1": PricingTable(
        input_cache_none_usd_per_million_tokens=1.25,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.125,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.2": PricingTable(
        input_cache_none_usd_per_million_tokens=1.75,
        output_usd_per_million_tokens=14.00,
        cache_read_usd_per_million_tokens=0.175,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.4": PricingTable(
        input_cache_none_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.4-mini": PricingTable(
        input_cache_none_usd_per_million_tokens=0.75,
        output_usd_per_million_tokens=4.50,
        cache_read_usd_per_million_tokens=0.075,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.5": PricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=30.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.6-luna": PricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=6.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=1.25,
    ),
    "gpt-5.6-terra": PricingTable(
        input_cache_none_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_usd_per_million_tokens=3.125,
    ),
    "gpt-5.6-sol": PricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=30.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=6.25,
    ),
}
"""Public prices per openai model; the default pricing lookup."""


def openai_model(
    model: OpenAIModelName,
    *,
    client: AsyncOpenAI | AsyncBedrockOpenAI | None = None,
    pricing: PricingTable | None = None,
    rate_limiter: RateLimiter | None = None,
) -> LLM:
    """Build a ready LLM for one cataloged model on the Responses API.

    client None constructs AsyncOpenAI(), which reads OPENAI_API_KEY from the environment.
    pricing None selects OPENAI_PRICING[model].
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.
    """
    return LLM(
        OpenAIResponsesProvider(
            client=client if client is not None else AsyncOpenAI(),
            model=model,
            pricing=pricing if pricing is not None else OPENAI_PRICING[model],
        ),
        rate_limiter=rate_limiter,
    )


__all__ = [
    "OPENAI_PRICING",
    "OpenAIModelName",
    "OpenAIResponsesProvider",
    "cost_breakdown",
    "openai_model",
]
