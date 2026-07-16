"""The anthropic backend: the Messages adapter, its model catalog, and pricing.

Importing this subpackage requires the anthropic package (install langchaint[anthropic]);
the import below raises a ModuleNotFoundError naming the extra to install.

anthropic_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the Messages adapter and wraps it in an LLM.
client None constructs the native SDK client, which reads credentials from the environment;
Bedrock routing is passing an AsyncAnthropicBedrock client instead.
pricing None selects the model's public prices from ANTHROPIC_PRICING.
Pass your own PricingTable to override, for example when your account bills at a custom rate.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://platform.claude.com/docs/en/about-claude/pricing.
Prices are the one provider fact this package cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
Rates derive from the base input price: cache read 0.1x, 5-minute cache write 1.25x, 1-hour cache write 2x.
"""

from typing import Literal

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
except ModuleNotFoundError as exc:
    if exc.name != "anthropic":
        raise
    raise ModuleNotFoundError(
        "langchaint's anthropic backend requires the anthropic package; install langchaint[anthropic]."
    ) from exc

from langchaint.anthropic.messages_provider import AnthropicMessagesProvider
from langchaint.llm import LLM
from langchaint.provider import PricingTable
from langchaint.rate_limiter import RateLimiter

type AnthropicModelName = Literal[
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
]
"""Model identifiers with public prices in ANTHROPIC_PRICING."""

ANTHROPIC_PRICING: dict[AnthropicModelName, PricingTable] = {
    "claude-sonnet-4-6": PricingTable(
        input_cache_none_usd_per_million_tokens=3.00,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.30,
        cache_write_usd_per_million_tokens=3.75,
        cache_write_1h_usd_per_million_tokens=6.00,
    ),
    # introductory pricing, through 2026-08-31; standard 3.00/15.00 from 2026-09-01
    "claude-sonnet-5": PricingTable(
        input_cache_none_usd_per_million_tokens=2.00,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.20,
        cache_write_usd_per_million_tokens=2.50,
        cache_write_1h_usd_per_million_tokens=4.00,
    ),
    "claude-opus-4-6": PricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-opus-4-7": PricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-opus-4-8": PricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-haiku-4-5-20251001": PricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=5.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=1.25,
        cache_write_1h_usd_per_million_tokens=2.00,
    ),
}
"""Public prices per anthropic model; the default pricing lookup."""


def anthropic_model(
    model: AnthropicModelName,
    *,
    client: AsyncAnthropic | AsyncAnthropicBedrock | None = None,
    pricing: PricingTable | None = None,
    default_max_completion_tokens: int = 4096,
    rate_limiter: RateLimiter | None = None,
) -> LLM:
    """Build a ready LLM for one cataloged model on the Messages API.

    client None constructs AsyncAnthropic(), which reads ANTHROPIC_API_KEY from the environment.
    pricing None selects ANTHROPIC_PRICING[model].
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget.
    """
    return LLM(
        AnthropicMessagesProvider(
            client=client if client is not None else AsyncAnthropic(),
            model=model,
            pricing=pricing if pricing is not None else ANTHROPIC_PRICING[model],
            default_max_completion_tokens=default_max_completion_tokens,
        ),
        rate_limiter=rate_limiter,
    )


__all__ = [
    "ANTHROPIC_PRICING",
    "AnthropicMessagesProvider",
    "AnthropicModelName",
    "anthropic_model",
]
