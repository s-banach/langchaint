"""Use OpenAI `provider_executed_tools` for web search."""

from langchaint import RawPart, Response
from langchaint.openai import OpenAI


async def search_the_web() -> Response[str]:
    """Run provider web search and print its output and cost.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
    """
    openai = OpenAI()
    # Catalog pricing supplies the required web-search invocation rate.
    bound = openai.model("gpt-5.6-terra").bind(
        provider_executed_tools=({"type": "web_search"},),
        automatic_cache_breakpoints=True,
    )
    response = await bound.generate_one("Find today's OpenAI developer news.")

    raw_parts = [part.raw for part in response.assistant_message.turn if isinstance(part, RawPart)]
    print(f"provider output items: {raw_parts}")
    print(f"provider tool cost: {response.usage.provider_executed_tool_cost_in_usd} USD")
    return response
