"""The anthropic backend: the Messages adapter, its model catalog, and pricing.

Importing this subpackage requires the anthropic package (install langchaint[anthropic]);
the import below raises a ModuleNotFoundError naming the extra to install.

anthropic_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the Messages adapter and wraps it in an LLM.
client None constructs the native first-party SDK client, which reads credentials from the environment.
anthropic_bedrock_model is the Bedrock sibling: it names the same catalog model and reads the model's
Bedrock surface (which of two client classes) and its wire model id from ANTHROPIC_BEDROCK,
so the application names neither the client class nor the Bedrock id.
Both Bedrock client classes construct offline from aws_region alone (anthropic 0.116.0),
so building a model object needs no AWS credentials.
pricing None selects the model's public prices from ANTHROPIC_PRICING, shared by both constructors.
Pass your own PricingTable to override, for example when your account bills at a custom rate.
For a custom httpx.AsyncClient (loaded certs, a proxy), anthropic_model takes client=AsyncAnthropic(
http_client=...) since its single client class makes that lossless, while anthropic_bedrock_model takes
http_client= directly so it can still pick the model's Bedrock client class for you.
cost_breakdown(usage_raw, pricing) reports the exact per-category cost of one response from its raw
SDK usage, through the same arithmetic that produced the stored Usage.cost_in_usd.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://platform.claude.com/docs/en/about-claude/pricing.
Prices are the one provider fact this package cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
Rates derive from the base input price: cache read 0.1x, 5-minute cache write 1.25x, 1-hour cache write 2x.
"""

from dataclasses import dataclass
from typing import Literal

import httpx

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
except ModuleNotFoundError as exc:
    if exc.name != "anthropic":
        raise
    raise ModuleNotFoundError(
        "langchaint's anthropic backend requires the anthropic package; install langchaint[anthropic]."
    ) from exc

from langchaint.anthropic.messages_provider import (
    AnthropicMessagesProvider,
    CacheTtl,
    cost_breakdown,
)
from langchaint.llm import LLM
from langchaint.provider import PricingTable
from langchaint.rate_limiter import RateLimiter

type AnthropicModelName = Literal[
    "claude-fable-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
]
"""Model identifiers with public prices in ANTHROPIC_PRICING."""

ANTHROPIC_PRICING: dict[AnthropicModelName, PricingTable] = {
    "claude-fable-5": PricingTable(
        input_cache_none_usd_per_million_tokens=10.00,
        output_usd_per_million_tokens=50.00,
        cache_read_usd_per_million_tokens=1.00,
        cache_write_usd_per_million_tokens=12.50,
        cache_write_1h_usd_per_million_tokens=20.00,
    ),
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
"""Public prices per anthropic model; the default pricing lookup, shared by both constructors."""


@dataclass(frozen=True, kw_only=True)
class BedrockRouting:
    """How one catalog model reaches Bedrock: which surface, and the ready-to-send wire model id.

    surface selects the client class in _BEDROCK_CLIENT_CLASS.
    "mantle" is the "Claude in Amazon Bedrock" Messages-API surface (AsyncAnthropicBedrockMantle),
    "legacy" the InvokeModel surface (AsyncAnthropicBedrock).
    wire_model is the id Bedrock accepts, with any inference-profile prefix already applied;
    it is not derivable from the native name by a rule, so it is stored per model.
    """

    surface: Literal["mantle", "legacy"]
    wire_model: str


ANTHROPIC_BEDROCK: dict[AnthropicModelName, BedrockRouting] = {
    "claude-fable-5": BedrockRouting(surface="mantle", wire_model="anthropic.claude-fable-5"),
    "claude-opus-4-8": BedrockRouting(surface="mantle", wire_model="anthropic.claude-opus-4-8"),
    "claude-opus-4-7": BedrockRouting(surface="mantle", wire_model="anthropic.claude-opus-4-7"),
    "claude-sonnet-5": BedrockRouting(surface="mantle", wire_model="anthropic.claude-sonnet-5"),
    "claude-haiku-4-5-20251001": BedrockRouting(surface="mantle", wire_model="anthropic.claude-haiku-4-5"),
    "claude-opus-4-6": BedrockRouting(surface="legacy", wire_model="us.anthropic.claude-opus-4-6-v1"),
    "claude-sonnet-4-6": BedrockRouting(surface="legacy", wire_model="us.anthropic.claude-sonnet-4-6"),
}
"""Per-model Bedrock routing; total over AnthropicModelName so a new catalog model must add an entry."""

_BEDROCK_CLIENT_CLASS: dict[
    Literal["mantle", "legacy"], type[AsyncAnthropicBedrockMantle | AsyncAnthropicBedrock]
] = {
    "mantle": AsyncAnthropicBedrockMantle,
    "legacy": AsyncAnthropicBedrock,
}


def anthropic_model(
    model: AnthropicModelName,
    *,
    client: AsyncAnthropic | AsyncAnthropicBedrock | None = None,
    pricing: PricingTable | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTtl = "5m",
    rate_limiter: RateLimiter | None = None,
) -> LLM:
    """Build a ready LLM for one cataloged model on the Messages API.

    client None constructs AsyncAnthropic(), which reads ANTHROPIC_API_KEY from the environment.
    pricing None selects ANTHROPIC_PRICING[model].
    cache_ttl applies uniformly to every cache marker the adapter writes:
    "5m" is the API default; "1h" holds entries across longer gaps and bills writes at 2x instead of 1.25x,
    paying off when requests reusing the prefix arrive more than five minutes apart.
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.

    Raises:
        ValueError: cache_ttl is "1h" but pricing has no cache_write_1h_usd_per_million_tokens
            (from AnthropicMessagesProvider.__init__; every ANTHROPIC_PRICING entry carries the
            rate, so only a custom pricing can trip this).
    """
    return LLM(
        AnthropicMessagesProvider(
            client=client if client is not None else AsyncAnthropic(),
            model=model,
            pricing=pricing if pricing is not None else ANTHROPIC_PRICING[model],
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
        ),
        rate_limiter=rate_limiter,
    )


def anthropic_bedrock_model(  # noqa: PLR0913 (Bedrock adds aws_region and http_client to the standard set)
    model: AnthropicModelName,
    *,
    aws_region: str | None = None,
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle | None = None,
    http_client: httpx.AsyncClient | None = None,
    pricing: PricingTable | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTtl = "5m",
    rate_limiter: RateLimiter | None = None,
) -> LLM:
    """Build a ready LLM for one cataloged model on Bedrock.

    The model's Bedrock surface and wire model id come from ANTHROPIC_BEDROCK[model],
    so the application names neither the client class nor the Bedrock id, only the native model.
    client None constructs the surface's client class with aws_region
    (None resolves the region from the AWS credential chain).
    Pass client to supply your own; it must match the model's surface.
    http_client passes a custom httpx.AsyncClient (loaded certs, a proxy) to the surface's client class,
    keeping the class-routing convenience that passing a whole client would forgo;
    it is only for the default-client path, so passing both client and http_client raises
    (a passed client already owns its transport). Unlike anthropic_model and openai_model, whose single
    client class makes client=AsyncAnthropic(http_client=...) lossless, the Bedrock constructor picks one of
    two client classes from the model, so it takes http_client to spare the application naming that class.
    pricing None selects ANTHROPIC_PRICING[model], the same table anthropic_model uses:
    the default is Anthropic's first-party list price, an estimate on Bedrock (AWS sets the real rate),
    corrected by passing pricing.
    cache_ttl has the anthropic_model meaning; Bedrock supports both tiers.
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.

    Raises:
        ValueError: both client and http_client are provided,
            or client is provided but its class does not serve model's Bedrock surface,
            or cache_ttl is "1h" but pricing has no cache_write_1h_usd_per_million_tokens
            (from AnthropicMessagesProvider.__init__; every ANTHROPIC_PRICING entry carries the
            rate, so only a custom pricing can trip this).
    """
    routing = ANTHROPIC_BEDROCK[model]
    if client is None:
        client = _BEDROCK_CLIENT_CLASS[routing.surface](aws_region=aws_region, http_client=http_client)
    else:
        if http_client is not None:
            raise ValueError(
                "Pass at most one of client= or http_client=; a passed client already owns its transport."
            )
        required_class = _BEDROCK_CLIENT_CLASS[routing.surface]
        if not isinstance(client, required_class):
            raise ValueError(
                f"{model!r} is served on the {routing.surface!r} Bedrock surface, which requires a "
                f"{required_class.__name__} client, but a {type(client).__name__} was passed."
            )
    return LLM(
        AnthropicMessagesProvider(
            client=client,
            model=routing.wire_model,
            pricing=pricing if pricing is not None else ANTHROPIC_PRICING[model],
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
        ),
        rate_limiter=rate_limiter,
    )


__all__ = [
    "ANTHROPIC_BEDROCK",
    "ANTHROPIC_PRICING",
    "AnthropicMessagesProvider",
    "AnthropicModelName",
    "BedrockRouting",
    "CacheTtl",
    "anthropic_bedrock_model",
    "anthropic_model",
    "cost_breakdown",
]
