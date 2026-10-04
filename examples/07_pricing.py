"""Price an uncataloged model and read the cost of a generation."""

from langchaint import GenerationWithoutToolCalls
from langchaint.openai import OpenAI, OpenAIPricingTable, OpenAIRates


async def price_at_negotiated_rates() -> GenerationWithoutToolCalls[str]:
    """Price an uncataloged model at contract rates.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
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
    ).bind(system_prompt="Be terse.")
    generation = await bound.generate_one("Name three primary colors.")
    print(f"billed {generation.usage.cost_in_usd} USD across every request")
    return generation
