"""Pricing: state rates for a model outside the catalog, and read what a response cost."""

from openai import AsyncOpenAI

from langchaint import LLM, Response
from langchaint.openai import OpenAIPricingTable, OpenAIResponsesAdapter


async def price_at_negotiated_rates() -> Response[str]:
    """Price a model id outside the catalog at contract rates, and read both billing scopes.

    Raises:
        GenerationError: any terminal outcome of the generate call.
    """
    negotiated_default_rates = OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=8.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=0.00,
    )
    llm = LLM(
        OpenAIResponsesAdapter(
            client=AsyncOpenAI(),
            model="gpt-5.6",
            pricing={"default": negotiated_default_rates},
            provider_name="openai",
            supports_prompt_cache_options=True,
        )
    )
    bound = llm.bind(system_prompt="Be terse.", automatic_prompt_caching=False)
    response = await bound.generate_one("Name three primary colors.")
    print(f"bill this call at {response.usage.cost_in_usd} USD")
    print(f"the kept answer alone cost {response.usage_successful_attempt.cost_in_usd} USD")
    return response
