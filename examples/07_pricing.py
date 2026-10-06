"""Price an uncataloged model and read the cost of a generation."""

from langchaint import PlainGeneration
from langchaint.openai import OpenAI, OpenAIPricingTable, OpenAIRates


async def price_at_negotiated_rates() -> PlainGeneration[str]:
    """Price an uncataloged model at contract rates.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
    """
    negotiated_default_rates = OpenAIRates(
        input_tokens_cache_none=1.00,
        output_tokens=8.00,
        input_tokens_cache_read=0.10,
        input_tokens_cache_write=0.00,
    )
    pricing = OpenAIPricingTable(default=negotiated_default_rates)
    openai = OpenAI()
    bound = openai.llm(
        "gpt-5.6",
        pricing=pricing,
        supports_prompt_cache_options=True,
    ).bind(system_prompt="Be terse.")
    generation = await bound.generate_one("Name three primary colors.")
    print(f"billed {generation.usage.cost_in_usd} USD across every request")
    return generation
