"""Streaming: a handle that is an item iterator, a Response source, and a context manager.

stream_one returns a StreamHandle without doing any I/O.
Iterating yields StreamItem = str | ToolCall and nothing else: text chunks are the SDK's own strings passed through,
and each ToolCall is yielded once, complete, when its block closes (there are no partial-argument delta items).
await handle.final() drains the rest silently and returns the assembled Response,
where usage, cost, and stop_reason live.
Use async with so an abandoned stream closes its connection.

The LangChain call-for-call map (stream, astream, astream_events) lives in MIGRATING_FROM_LANGCHAIN.md.
"""

import asyncio

from pydantic import BaseModel

from langchaint import BoundLLM, Message, Tool, ToolCall, ToolManager, UserMessage
from langchaint.openai import openai_model


async def stream_text() -> None:
    """Print text as it arrives, then read usage off the final Response.

    Nothing starts until the first item is requested; the async with block guarantees the connection closes.
    """
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Answer in a short paragraph.",
        automatic_prompt_caching=False,
    )
    handle = bound.stream_one("Describe the water cycle.")
    async with handle:
        async for item in handle:
            if isinstance(item, str):
                print(item, end="", flush=True)
        response = await handle.final()
    print()
    print(f"{response.usage.output_tokens} output tokens, {response.usage.cost_in_usd:.6f} USD")


async def stream_agent(
    bound: BoundLLM[str], tool_manager: ToolManager, prompt: str, max_turns: int = 10
) -> str:
    """Run the streaming ReAct loop: print text live, dispatch the completed tool calls between turns.

    Read the assembled assistant message and tool calls from final() rather than collecting ToolCall items by hand;
    final() is idempotent and returns the same assembled Response.
    This is the non-streaming loop from 02_tool_loop.py with the generate_one call replaced by a stream,
    including the same max_turns ceiling so a tool-looping model cannot stream forever.

    Raises:
        RuntimeError: if the model keeps calling tools for max_turns turns without returning a final answer.
    """
    conversation: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        handle = bound.stream_one(conversation)
        async with handle:
            async for item in handle:
                if isinstance(item, str):
                    print(item, end="", flush=True)
                elif isinstance(item, ToolCall):
                    print(f"\n[calling {item.name}]")
            response = await handle.final()
        conversation.append(response.assistant_message)
        if not response.tool_calls:
            return response.output
        for call in response.tool_calls:
            outcome = await tool_manager.dispatch(call)
            conversation.append(outcome.tool_message)
    raise RuntimeError(f"agent did not finish within {max_turns} turns")


class CityArgs(BaseModel):
    """A placeholder tool's arguments; the tool's specifics do not matter to the streaming point."""

    city: str


async def get_weather(args: CityArgs) -> str:
    """Return a canned weather string for the streaming tool loop to dispatch."""
    return f"It is 18C and clear in {args.city}."


weather_tool = Tool(
    name="get_weather",
    description="Return the current weather for a city.",
    args_model=CityArgs,
    function=get_weather,
)


async def main() -> None:
    """Run the text stream, then the streaming tool loop."""
    await stream_text()
    print()
    tool_manager = ToolManager([weather_tool])
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Use tools to answer questions about the weather.",
        tool_manager=tool_manager,
        automatic_prompt_caching=True,
    )
    answer = await stream_agent(bound, tool_manager, "What is the weather in Oslo?")
    print(f"\nfinal answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
