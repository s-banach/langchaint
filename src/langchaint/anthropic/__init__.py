"""The anthropic backend: the Messages adapter, its model catalog, and pricing.

Importing this subpackage requires the anthropic package;
the import below raises a ModuleNotFoundError naming the package to install.

anthropic_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the Messages adapter and wraps it in an LLM.
client None constructs the native first-party SDK client, which reads credentials from the environment.
anthropic_bedrock_model is the Bedrock sibling: it takes the Bedrock wire model id
(AnthropicBedrockModelName) and sends it verbatim, so the id in application code, on the wire,
and in traces is one string; the id's Bedrock API (which of two client classes) and default
pricing come from ANTHROPIC_BEDROCK, so the application never names the client class.
Both Bedrock client classes construct offline from aws_region alone (anthropic 0.120.0),
so building a model object needs no AWS credentials.
pricing is a mapping from the service tier a response reports to the AnthropicPricingTable that
prices it, merged over the model's public standard-tier prices from ANTHROPIC_PRICING, so a caller
adds a tier or replaces the standard rates with a negotiated one and both constructors default to
the public rates. A response served at a tier the mapping does not hold costs NaN.
For a custom httpx.AsyncClient (loaded certs, a proxy), anthropic_model takes client=AsyncAnthropic(
http_client=...) since its single client class makes that lossless, while anthropic_bedrock_model takes
http_client= directly so it can still pick the model's Bedrock client class for you.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://platform.claude.com/docs/en/about-claude/pricing.
Prices are the one provider fact langchaint cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
Rates derive from the base input price: cache read 0.1x, 5-minute cache write 1.25x, 1-hour cache write 2x.
The catalog prices the standard tier; a priority-tier or batch-tier account states its own rates.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import httpx

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
except ModuleNotFoundError as exc:
    if exc.name != "anthropic":
        raise
    raise ModuleNotFoundError(
        "langchaint's anthropic backend requires the anthropic package; install anthropic."
    ) from exc

from langchaint.anthropic.messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicPricedServiceTier,
    AnthropicPricingTable,
    AnthropicServiceTier,
    CacheTTL,
    parse_anthropic,
)
from langchaint.llm import LLM
from langchaint.shared_backoff import SharedBackoff

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

ANTHROPIC_PRICING: dict[AnthropicModelName, AnthropicPricingTable] = {
    "claude-fable-5": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=10.00,
        output_usd_per_million_tokens=50.00,
        cache_read_usd_per_million_tokens=1.00,
        cache_write_5m_usd_per_million_tokens=12.50,
        cache_write_1h_usd_per_million_tokens=20.00,
    ),
    "claude-sonnet-4-6": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=3.00,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.30,
        cache_write_5m_usd_per_million_tokens=3.75,
        cache_write_1h_usd_per_million_tokens=6.00,
    ),
    # introductory pricing, through 2026-08-31; standard 3.00/15.00 from 2026-09-01
    "claude-sonnet-5": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=2.00,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.20,
        cache_write_5m_usd_per_million_tokens=2.50,
        cache_write_1h_usd_per_million_tokens=4.00,
    ),
    "claude-opus-4-6": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-opus-4-7": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-opus-4-8": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-haiku-4-5-20251001": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=5.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_5m_usd_per_million_tokens=1.25,
        cache_write_1h_usd_per_million_tokens=2.00,
    ),
}
"""Public prices per anthropic model; the default pricing lookup, shared by both constructors."""


type AnthropicBedrockModelName = Literal[
    "anthropic.claude-fable-5",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5",
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-sonnet-4-6",
]
"""Bedrock wire model ids anthropic_bedrock_model accepts and sends verbatim.

A us.-prefixed id is a cross-region inference-profile id; the prefix is part of the id the wire accepts.
An id is not derivable from the first-party model identifier by a rule, so the ids are enumerated.
"""


@dataclass(frozen=True, kw_only=True)
class BedrockRouting:
    """How anthropic_bedrock_model serves one Bedrock wire model id.

    api selects the client class in _BEDROCK_CLIENT_CLASS.
    "mantle" is the "Claude in Amazon Bedrock" Messages API (AsyncAnthropicBedrockMantle),
    "legacy" the InvokeModel API (AsyncAnthropicBedrock).
    pricing_key names the catalog model whose ANTHROPIC_PRICING entry is the id's default pricing.
    """

    api: Literal["mantle", "legacy"]
    pricing_key: AnthropicModelName


ANTHROPIC_BEDROCK: dict[AnthropicBedrockModelName, BedrockRouting] = {
    "anthropic.claude-fable-5": BedrockRouting(api="mantle", pricing_key="claude-fable-5"),
    "anthropic.claude-opus-4-8": BedrockRouting(api="mantle", pricing_key="claude-opus-4-8"),
    "anthropic.claude-opus-4-7": BedrockRouting(api="mantle", pricing_key="claude-opus-4-7"),
    "anthropic.claude-sonnet-5": BedrockRouting(api="mantle", pricing_key="claude-sonnet-5"),
    "anthropic.claude-haiku-4-5": BedrockRouting(
        api="mantle", pricing_key="claude-haiku-4-5-20251001"
    ),
    "us.anthropic.claude-opus-4-6-v1": BedrockRouting(api="legacy", pricing_key="claude-opus-4-6"),
    "us.anthropic.claude-sonnet-4-6": BedrockRouting(
        api="legacy", pricing_key="claude-sonnet-4-6"
    ),
}
"""Routing per Bedrock wire model id.

Total over AnthropicBedrockModelName: adding a new id requires an entry giving its api and pricing_key.
"""

_BEDROCK_CLIENT_CLASS: dict[
    Literal["mantle", "legacy"], type[AsyncAnthropicBedrockMantle | AsyncAnthropicBedrock]
] = {
    "mantle": AsyncAnthropicBedrockMantle,
    "legacy": AsyncAnthropicBedrock,
}


def anthropic_model(  # noqa: PLR0913 (the ready-LLM constructor states every choice: client, pricing, caching, tier, and pacing)
    model: AnthropicModelName,
    *,
    client: AsyncAnthropic | None = None,
    pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTTL = "5m",
    service_tier: AnthropicServiceTier | None = None,
    shared_backoff: SharedBackoff | None = None,
    max_attempts: int = 3,
) -> LLM:
    """Build a ready LLM for one cataloged model on the Messages API.

    client None constructs AsyncAnthropic(), which reads ANTHROPIC_API_KEY from the environment.
    pricing holds one table per service tier, keyed by the tier a response reports, and is merged
    over {"standard": ANTHROPIC_PRICING[model]}, so a caller adds a tier or replaces the standard
    rates with a negotiated one and omitting it prices at the public standard rates.
    A response served at a tier this mapping does not hold costs NaN.
    cache_ttl applies uniformly to every cache marker the adapter writes:
    "5m" is the API default; "1h" holds entries across longer gaps and bills writes at 2x instead of 1.25x,
    paying off when requests reusing the prefix arrive more than five minutes apart.
    service_tier is what the request asks for, None sending nothing; it is not what prices the
    response, since anthropic's request and response tier vocabularies share no word.
    shared_backoff and max_attempts have the LLM.__init__ meanings;
    pass one SharedBackoff across models on the same account so a rate limit pauses them together,
    and note one instance serves one event loop.

    Raises:
        ValueError: max_attempts is not a positive int (from LLM.__init__), or
            client is a Bedrock client, which does not reach the "anthropic" provider this
            constructor states (from Adapter.__init__; the narrowed client annotation already
            excludes one at check time, since the Bedrock classes are siblings of AsyncAnthropic
            rather than subclasses).
    """
    return LLM(
        AnthropicMessagesAdapter(
            client=client if client is not None else AsyncAnthropic(),
            model=model,
            pricing={"standard": ANTHROPIC_PRICING[model], **(pricing or {})},
            provider_name="anthropic",
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
            service_tier=service_tier,
        ),
        shared_backoff=shared_backoff,
        max_attempts=max_attempts,
    )


def anthropic_bedrock_model(  # noqa: PLR0913 (Bedrock adds aws_region and http_client to the standard set)
    model: AnthropicBedrockModelName,
    *,
    aws_region: str | None = None,
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle | None = None,
    http_client: httpx.AsyncClient | None = None,
    pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTTL = "5m",
    shared_backoff: SharedBackoff | None = None,
    max_attempts: int = 3,
) -> LLM:
    """Build a ready LLM for one cataloged Bedrock wire model id.

    model is sent verbatim, so the id in application code, on the wire, and in traces is one string;
    its Bedrock API and default pricing come from ANTHROPIC_BEDROCK[model],
    so the application never names the client class.
    client None constructs the API's client class with aws_region.
    aws_region None leaves the client class to resolve the region itself, and the two classes do it
    differently. Pass aws_region to make the region explicit rather than depending on which class
    the model routes to.
    Pass client to supply your own; its class must serve the model's Bedrock API.
    aws_region is only for the default-client path, so passing both it and client raises:
    a passed client already carries its region, and the aws_region beside it would be dropped,
    sending every request to the client's region instead.
    http_client passes a custom httpx.AsyncClient (loaded certs, a proxy) to the API's client class,
    keeping the class-routing convenience that passing a whole client would forgo;
    it is only for the default-client path, so passing both client and http_client raises
    (a passed client already owns its transport). Unlike anthropic_model and openai_model, whose single
    client class makes client=AsyncAnthropic(http_client=...) lossless, the Bedrock constructor picks one of
    two client classes from the model, so it takes http_client to spare the application naming that class.
    pricing is merged over {"standard": ANTHROPIC_PRICING[ANTHROPIC_BEDROCK[model].pricing_key]},
    the same table anthropic_model defaults to:
    that default is Anthropic's first-party list price, an estimate on Bedrock (AWS sets the real rate),
    corrected by passing pricing. A Bedrock response is unlikely to report a service tier at all,
    and one that reports none prices at the "standard" key, so the merged default prices every
    response until a caller adds a tier.
    cache_ttl has the anthropic_model meaning; Bedrock supports both tiers.
    There is no service_tier parameter: Anthropic's service tiers are its own platform's, and this
    constructor reaches Bedrock, so there is no tier here for a request to ask for.
    shared_backoff and max_attempts have the LLM.__init__ meanings;
    pass one SharedBackoff across models on the same account so a rate limit pauses them together,
    and note one instance serves one event loop.

    Raises:
        ValueError: max_attempts is not a positive int (from LLM.__init__);
            client is provided together with http_client or aws_region;
            or client is provided but its class does not serve model's Bedrock API.
        anthropic.AnthropicError: model is served by the "mantle" Bedrock API, client is None, and
            AsyncAnthropicBedrockMantle.__init__ resolves neither a region nor a base URL.
    """
    routing = ANTHROPIC_BEDROCK[model]
    if client is None:
        client = _BEDROCK_CLIENT_CLASS[routing.api](aws_region=aws_region, http_client=http_client)
    else:
        if http_client is not None:
            raise ValueError(
                "Pass at most one of client= or http_client=; a passed client already owns its transport."
            )
        if aws_region is not None:
            raise ValueError(
                "Pass at most one of client= or aws_region=; a passed client already carries its region."
            )
        required_class = _BEDROCK_CLIENT_CLASS[routing.api]
        if not isinstance(client, required_class):
            raise ValueError(
                f"{model!r} is served by the {routing.api!r} Bedrock API, which requires a "
                f"{required_class.__name__} client, but a {type(client).__name__} was passed."
            )
    return LLM(
        AnthropicMessagesAdapter(
            client=client,
            model=model,
            pricing={"standard": ANTHROPIC_PRICING[routing.pricing_key], **(pricing or {})},
            provider_name="aws.bedrock",
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
        ),
        shared_backoff=shared_backoff,
        max_attempts=max_attempts,
    )


__all__ = [
    "ANTHROPIC_BEDROCK",
    "ANTHROPIC_PRICING",
    "AnthropicBedrockModelName",
    "AnthropicMessagesAdapter",
    "AnthropicModelName",
    "AnthropicPricedServiceTier",
    "AnthropicPricingTable",
    "AnthropicServiceTier",
    "BedrockRouting",
    "CacheTTL",
    "anthropic_bedrock_model",
    "anthropic_model",
    "parse_anthropic",
]
