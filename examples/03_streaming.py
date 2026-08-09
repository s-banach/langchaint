"""Streaming: consume answer text and reasoning as they arrive, then read the Response."""

from langchaint import ReasoningDelta, Response
from langchaint.openai import OpenAIAccount


async def stream_answer() -> Response[str]:
    """Print the answer as it arrives, collect the reasoning apart, and return the assembled Response.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable during account construction.
        Exception: An owned resource close operation fails.
        GenerationError: a terminal outcome, raised on entry, while iterating, or at final().
        StreamProtocolError: the provider's event stream ended without a terminal event.
    """
    async with OpenAIAccount() as openai:
        # reasoning_summary asks openai for readable reasoning text.
        bound = openai.model("gpt-5.6-terra", reasoning_summary="auto").bind(
            system_prompt="Answer in a short paragraph.", automatic_prompt_caching=False
        )

        reasoning_chunks: list[str] = []
        # Entering the block is what opens the request.
        async with bound.stream_one("Describe the water cycle.") as handle:
            async for item in handle:
                match item:
                    case str():
                        print(item, end="", flush=True)
                    case ReasoningDelta():
                        reasoning_chunks.append(item.text)
            reasoning = "".join(reasoning_chunks)
            print(f"\nreasoning: {reasoning}")
            # final() returns the assembled Response, where usage and cost are readable.
            return await handle.final()
