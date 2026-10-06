"""Use OpenAI `provider_executed_tools` for web search."""

from langchaint import PlainGeneration
from langchaint.openai import OpenAI


async def search_the_web() -> PlainGeneration[str]:
    """Run provider web search and print its output and cost.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
    """
    openai = OpenAI()
    # Catalog pricing supplies the required web-search invocation rate.
    bound = openai.llm("gpt-5.6-terra").bind(
        provider_executed_tools=({"type": "web_search"},),
        automatic_cache_breakpoints=True,
    )
    generation = await bound.generate_one("Find today's OpenAI developer news.")

    raw_parts = [part.raw for part in generation.assistant_message.parts if part.kind == "raw"]
    print(f"provider output items: {raw_parts}")
    print(f"provider tool cost: {generation.usage.provider_executed_tool_cost_in_usd} USD")
    return generation
