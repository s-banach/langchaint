"""Price an uncataloged model and read its response cost."""

from langchaint import Response
from langchaint.openai import OpenAIAccount, OpenAIPricingTable


async def price_at_negotiated_rates() -> Response[str]:
    """Price a model id outside the catalog at contract rates, and read both billing scopes.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable during account construction.
        Exception: An owned resource close operation fails.
        GenerationError: any terminal outcome of the generate call.
    """
    negotiated_default_rates = OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=8.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=0.00,
    )
    async with OpenAIAccount() as openai:
        bound = openai.model(
            "gpt-5.6",
            pricing={"default": negotiated_default_rates},
            supports_prompt_cache_options=True,
        ).bind(system_prompt="Be terse.", automatic_prompt_caching=False)
        response = await bound.generate_one("Name three primary colors.")
        print(f"bill this call at {response.usage.cost_in_usd} USD")
        print(f"the kept answer alone cost {response.usage_successful_attempt.cost_in_usd} USD")
        return response
