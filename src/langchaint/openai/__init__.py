"""The openai backend: the Responses adapter, its model catalog, and pricing.

Importing this subpackage requires the openai package;
the import below raises a ModuleNotFoundError naming the package to install.

openai_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the Responses adapter and wraps it in an LLM.
client None constructs the native SDK client, which reads credentials from the environment.
openai_model states provider_name="openai" for the adapter,
and the adapter checks that pair against OpenAIResponsesAdapter.provider_name_by_client_class,
which makes Adapter.__init__ raise for AsyncBedrockOpenAI and AsyncAzureOpenAI:
both subclass AsyncOpenAI, so the annotation cannot exclude them on its own.
A base AsyncOpenAI is accepted whatever its base_url,
so reaching an OpenAI-compatible endpoint through openai_model labels it "openai";
a binding that should report the provider it actually reaches (groq and deepseek are gen_ai.provider.name values)
is OpenAIResponsesAdapter(client=..., provider_name="groq", ...) wrapped in an LLM.
openai_bedrock_model is the constructor for OpenAI models served by Bedrock;
Azure is OpenAIResponsesAdapter(client=AsyncAzureOpenAI(...),
provider_name="azure.ai.openai", ...) wrapped in an LLM.
pricing is a mapping from the service tier a response reports to the OpenAIPricingTable that prices it.
openai_model merges it over the model's public default-tier prices from OPENAI_PRICING, so a caller
adds a tier or replaces the default rates with a negotiated one and omitting it prices at the public
rates. A response served at a tier the mapping does not hold costs NaN.
openai_bedrock_model has no catalog to fall back on,
so its pricing and supports_prompt_cache_options are required.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://developers.openai.com/api/docs/pricing.
Prices are the one provider fact langchaint cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
The catalog prices the default tier; a flex, scale, or priority account states its own rates.
OpenAI has no 1-hour cache tier, and only the gpt-5.6 family bills cache writes;
earlier models cache automatically with free writes, so their tables carry a zero cache-write rate.
That family is also the one taking prompt_cache_options, listed in PROMPT_CACHE_OPTIONS_MODELS.
The bare gpt-5.6 model identifier is an alias for gpt-5.6-sol; the catalog uses the explicit identifier.
"""

from collections.abc import Mapping
from typing import Literal

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires the openai package; install openai."
    ) from exc

from langchaint.llm import LLM
from langchaint.openai.responses_adapter import (
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
    OpenAIServiceTier,
    ReasoningSummary,
)
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

OPENAI_PRICING: dict[OpenAIModelName, OpenAIPricingTable] = {
    "gpt-5.1": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.25,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.125,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.2": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.75,
        output_usd_per_million_tokens=14.00,
        cache_read_usd_per_million_tokens=0.175,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.4": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.4-mini": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=0.75,
        output_usd_per_million_tokens=4.50,
        cache_read_usd_per_million_tokens=0.075,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.5": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=30.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=0.00,
    ),
    "gpt-5.6-luna": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=6.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=1.25,
    ),
    "gpt-5.6-terra": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_usd_per_million_tokens=3.125,
    ),
    "gpt-5.6-sol": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=30.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=6.25,
    ),
}
"""Public prices per openai model; the default pricing lookup."""

PROMPT_CACHE_OPTIONS_MODELS: frozenset[OpenAIModelName] = frozenset({
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
})
"""Cataloged models accepting the prompt_cache_options request parameter.

openai documents the parameter as gpt-5.6-and-later (openai 2.45.0), and it is what carries a
binding's automatic_prompt_caching False to the wire, so binding False on a model absent here
raises instead of sending a parameter the model does not take.
Bind True on those models: every one of them bills zero for a cache write and reads cached input
below its uncached rate, so leaving the provider's automatic caching in place costs nothing.
Held apart from those rates rather than derived from them, because a price and a parameter's
availability are two facts openai can change independently.
"""


def openai_model(
    model: OpenAIModelName,
    *,
    client: AsyncOpenAI | None = None,
    pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable] | None = None,
    rate_limiter: RateLimiter | None = None,
    reasoning_summary: ReasoningSummary | None = None,
    service_tier: OpenAIServiceTier | None = None,
) -> LLM:
    """Build a ready LLM for one cataloged model on the Responses API.

    client None constructs AsyncOpenAI(), which reads OPENAI_API_KEY from the environment.
    pricing holds one table per service tier, keyed by the tier a response reports, and is merged
    over {"default": OPENAI_PRICING[model]}, so a caller adds a tier or replaces the default rates
    with a negotiated one and omitting it prices at the public default rates.
    A response served at a tier this mapping does not hold costs NaN.
    Whether the model takes prompt_cache_options comes from PROMPT_CACHE_OPTIONS_MODELS,
    whose docstring gives what a model outside it does with bind(automatic_prompt_caching=False).
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.
    reasoning_summary asks the API for readable text, which reaches ReasoningTrace.text
    where the reasoning item carries no reasoning text of its own;
    None leaves the provider default in place.
    service_tier is what the request asks for, None sending nothing; it is not what prices the
    response, which is priced at the tier the response reports. Leaving it None on a project
    configured for a non-default tier can be served at that tier, and priced at default rates if
    the response reports none: state service_tier on such a project, or state its rates in pricing.

    Raises:
        ValueError: client is an AsyncBedrockOpenAI or AsyncAzureOpenAI, raised by the adapter.
            This constructor states provider_name="openai", which neither client reaches, and both
            subclass AsyncOpenAI, so the annotation alone accepts them. Reach those providers with
            openai_bedrock_model, or by building the adapter directly with the provider_name the
            client reaches.
    """
    return LLM(
        OpenAIResponsesAdapter(
            client=client if client is not None else AsyncOpenAI(),
            model=model,
            pricing={"default": OPENAI_PRICING[model], **(pricing or {})},
            provider_name="openai",
            supports_prompt_cache_options=model in PROMPT_CACHE_OPTIONS_MODELS,
            reasoning_summary=reasoning_summary,
            service_tier=service_tier,
        ),
        rate_limiter=rate_limiter,
    )


def openai_bedrock_model(
    model: str,
    *,
    pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable],
    supports_prompt_cache_options: bool,
    aws_region: str | None = None,
    client: AsyncBedrockOpenAI | None = None,
    rate_limiter: RateLimiter | None = None,
    reasoning_summary: ReasoningSummary | None = None,
) -> LLM:
    """Build a ready LLM for one OpenAI model served by Bedrock, on the Responses API.

    model is the Bedrock wire model id, sent verbatim, so the id in application code, on the wire,
    and in traces is one string. It is a str rather than a Literal catalog, and pricing is required
    rather than defaulted, because both asymmetries with anthropic_bedrock_model come from the same
    absence: langchaint carries no verified list of OpenAI's Bedrock model ids or their AWS rates,
    and prices are the one provider fact langchaint cannot verify by SDK introspection.
    pricing needs its "default" key, which prices every response reporting no service tier, and a
    Bedrock response is unlikely to report one.
    There is no service_tier parameter: OpenAI's service tiers are its own platform's, and this
    constructor reaches Bedrock, so there is no tier here for a request to ask for.
    supports_prompt_cache_options is required for the same absence of a catalog: it says whether
    the model takes the prompt_cache_options request parameter, which openai documents as
    gpt-5.6-and-later (openai 2.45.0), and no Bedrock id maps to that boundary here.
    False makes bind raise for a binding that declines caching, the parameter that would carry it
    being one the model does not take.
    client None constructs AsyncBedrockOpenAI(aws_region=aws_region)
    (None resolves the region from the AWS credential chain).
    There is no http_client parameter, because the Bedrock Responses API has one client class,
    so client=AsyncBedrockOpenAI(http_client=...) loses nothing; anthropic_bedrock_model takes one
    only because it picks between two client classes and would forgo that routing.
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.
    reasoning_summary asks the API for readable text, which reaches ReasoningTrace.text
    where the reasoning item carries no reasoning text of its own;
    None leaves the provider default in place.

    Raises:
        ValueError: both client and aws_region are given. A passed client already carries its
            region, so the aws_region beside it would be dropped and every request would go to
            the client's region instead, silently.
            OpenAIResponsesAdapter.__init__ raises it too when pricing has no "default" key,
            which prices every response reporting no service tier; nothing merges one in here,
            because no catalog maps a Bedrock model id.
    """
    if client is not None and aws_region is not None:
        raise ValueError(
            "Pass at most one of client= or aws_region=; a passed client already carries its region."
        )
    return LLM(
        OpenAIResponsesAdapter(
            client=client if client is not None else AsyncBedrockOpenAI(aws_region=aws_region),
            model=model,
            pricing=pricing,
            provider_name="aws.bedrock",
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
        ),
        rate_limiter=rate_limiter,
    )


__all__ = [
    "OPENAI_PRICING",
    "PROMPT_CACHE_OPTIONS_MODELS",
    "OpenAIModelName",
    "OpenAIPricedServiceTier",
    "OpenAIPricingTable",
    "OpenAIResponsesAdapter",
    "OpenAIServiceTier",
    "ReasoningSummary",
    "openai_bedrock_model",
    "openai_model",
]
