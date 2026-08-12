"""Price an uncataloged model and read its response cost."""

from langchaint import Response
from langchaint.openai import OpenAI, OpenAIPricingTable, OpenAIRates


async def price_at_negotiated_rates() -> Response[str]:
    """Price a model id outside the catalog at contract rates, and read both billing scopes.

    Raises:
        OpenAIError: OpenAI credentials are unavailable during `OpenAI` construction.
        GenerationError: any terminal outcome of the generate call.
    """
    negotiated_default_rates = OpenAIRates(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=8.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=0.00,
    )
    pricing = OpenAIPricingTable(default=negotiated_default_rates)
    openai = OpenAI()
    bound = openai.model(
        "gpt-5.6",
        pricing=pricing,
        supports_prompt_cache_options=True,
    ).bind(system_prompt="Be terse.", automatic_prompt_caching=False)
    response = await bound.generate_one("Name three primary colors.")
    print(f"bill this call at {response.usage.cost_in_usd} USD")
    print(f"kept answer cost: {response.usage_successful_attempt.cost_in_usd} USD")
    return response
