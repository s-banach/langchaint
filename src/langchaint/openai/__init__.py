"""The openai backend: the Responses adapter, its model catalog, and pricing.

Importing this subpackage requires the openai package;
the import below raises a ModuleNotFoundError naming the package to install.

openai_model takes the provider's own model identifier, the same string the wire accepts,
so switching models never changes an import; it constructs the Responses adapter and wraps it in an LLM.
client None constructs the native SDK client, which reads credentials from the environment.
openai_model states provider_name="openai" for the adapter,
and the adapter checks that pair against OpenAIResponsesAdapter.provider_name_by_client_class,
which refuses AsyncBedrockOpenAI and AsyncAzureOpenAI:
both subclass AsyncOpenAI, so the annotation cannot refuse them on its own.
A base AsyncOpenAI is accepted whatever its base_url,
so reaching an OpenAI-compatible endpoint through openai_model labels it "openai";
a binding that should report the provider it actually reaches (groq and deepseek are gen_ai.provider.name values)
is OpenAIResponsesAdapter(client=..., provider_name="groq", ...) wrapped in an LLM.
openai_bedrock_model is the constructor for OpenAI models served by Bedrock;
Azure is OpenAIResponsesAdapter(client=AsyncAzureOpenAI(...),
provider_name="azure.ai.openai", ...) wrapped in an LLM.
openai_model's pricing None selects the model's public prices from OPENAI_PRICING;
pass your own PricingTable to override, for example when your account bills at a custom rate.
openai_bedrock_model has no catalog to fall back on,
so its pricing and supports_prompt_cache_options are required.
cost_breakdown(usage_raw, pricing) reports the exact per-category cost of one response from its raw
SDK usage, through the same arithmetic that produced the stored Usage.cost_in_usd.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://developers.openai.com/api/docs/pricing.
Prices are the one provider fact langchaint cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
OpenAI has no 1-hour cache tier, and only the gpt-5.6 family bills cache writes;
earlier models cache automatically with free writes, so their tables carry a zero cache-write rate.
That family is also the one taking prompt_cache_options, listed in PROMPT_CACHE_OPTIONS_MODELS.
The bare gpt-5.6 model identifier is an alias for gpt-5.6-sol; the catalog uses the explicit identifier.
"""

from typing import Literal

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires the openai package; install openai."
    ) from exc

from langchaint.adapter import PricingTable
from langchaint.llm import LLM
from langchaint.openai.responses_adapter import (
    OpenAIResponsesAdapter,
    ReasoningSummary,
    cost_breakdown,
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

PROMPT_CACHE_OPTIONS_MODELS: frozenset[OpenAIModelName] = frozenset({
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
})
"""Cataloged models accepting the prompt_cache_options request parameter.

openai documents the parameter as gpt-5.6-and-later (openai 2.45.0), and it is what carries a
binding's automatic_prompt_caching False to the wire, so a model absent here keeps caching under
either binding value. At the OPENAI_PRICING rates that costs nothing: every model absent here
bills zero for a cache write and reads cached input below its uncached rate.
Held apart from those rates rather than derived from them, because a price and a parameter's
availability are two facts openai can change independently.
"""


def openai_model(
    model: OpenAIModelName,
    *,
    client: AsyncOpenAI | None = None,
    pricing: PricingTable | None = None,
    rate_limiter: RateLimiter | None = None,
    reasoning_summary: ReasoningSummary | None = None,
) -> LLM:
    """Build a ready LLM for one cataloged model on the Responses API.

    client None constructs AsyncOpenAI(), which reads OPENAI_API_KEY from the environment.
    pricing None selects OPENAI_PRICING[model].
    Whether the model takes prompt_cache_options comes from PROMPT_CACHE_OPTIONS_MODELS,
    whose docstring gives what a model outside it does with bind(automatic_prompt_caching=False).
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.
    reasoning_summary asks the API for readable text, which arrives on each
    ReasoningTrace.text; None leaves the provider default in place.

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
            pricing=pricing if pricing is not None else OPENAI_PRICING[model],
            provider_name="openai",
            supports_prompt_cache_options=model in PROMPT_CACHE_OPTIONS_MODELS,
            reasoning_summary=reasoning_summary,
        ),
        rate_limiter=rate_limiter,
    )


def openai_bedrock_model(
    model: str,
    *,
    pricing: PricingTable,
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
    supports_prompt_cache_options is required for the same absence of a catalog: it says whether
    the model takes the prompt_cache_options request parameter, which openai documents as
    gpt-5.6-and-later (openai 2.45.0), and no Bedrock id maps to that boundary here.
    False leaves the parameter unsent, so the model keeps caching under either binding value;
    pass False only where the pricing beside it charges nothing for a cache write, since otherwise
    a caller binding automatic_prompt_caching False pays for the caching they declined.
    client None constructs AsyncBedrockOpenAI(aws_region=aws_region)
    (None resolves the region from the AWS credential chain).
    There is no http_client parameter, because the Bedrock Responses API has one client class,
    so client=AsyncBedrockOpenAI(http_client=...) loses nothing; anthropic_bedrock_model takes one
    only because it picks between two client classes and would forgo that routing.
    rate_limiter None means the RateLimiter defaults;
    pass one shared instance across models on the same account to share its budget,
    built in the same event loop as the LLMs, since one instance serves one loop.
    reasoning_summary asks the API for readable text, which arrives on each
    ReasoningTrace.text; None leaves the provider default in place.

    Raises:
        ValueError: both client and aws_region are given. A passed client already carries its
            region, so the aws_region beside it would be dropped and every request would go to
            the client's region instead, silently.
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
    "OpenAIResponsesAdapter",
    "ReasoningSummary",
    "cost_breakdown",
    "openai_bedrock_model",
    "openai_model",
]
