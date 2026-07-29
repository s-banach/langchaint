"""Streaming: a handle that is an item iterator, a Response source, and a context manager.

stream_one returns a StreamHandle without doing any I/O; entering it with async with opens the request.
Iterating yields StreamItem = str | ReasoningDelta | ToolCall and nothing else.
Answer text chunks are the SDK's own strings passed through;
the model's readable reasoning is wrapped in a ReasoningDelta so it can go somewhere else,
here stderr while the answer goes to stdout, two destinations a shell redirects independently.
Each ToolCall is yielded once, complete, when its block closes (there are no partial-argument delta items).
await handle.final() drains the rest silently and returns the assembled Response,
where usage, cost, and stop_reason live.
The handle is unusable outside its async with block, which closes the connection and bounds the requests it can send.

The LangChain call-for-call map (stream, astream, astream_events) lives in MIGRATING_FROM_LANGCHAIN.md.
"""

import asyncio
import sys

from pydantic import BaseModel

from langchaint import (
    BoundLLM,
    HasTools,
    Message,
    PydanticTool,
    ReasoningDelta,
    ToolCall,
    ToolManager,
    UserMessage,
)
from langchaint.openai import openai_model


async def stream_text() -> None:
    """Print the answer to stdout and the reasoning to stderr as they arrive, then read usage off the final Response.

    The request opens at the async with, which also guarantees the connection closes.
    reasoning_summary is what asks openai for the readable reasoning the stderr branch prints.
    """
    bound = openai_model("gpt-5.6-terra", reasoning_summary="auto").bind(
        system_prompt="Answer in a short paragraph.",
        automatic_prompt_caching=False,
    )
    handle = bound.stream_one("Describe the water cycle.")
    async with handle:
        async for item in handle:
            match item:
                case str():
                    print(item, end="", flush=True)
                case ReasoningDelta():
                    print(item.text, end="", flush=True, file=sys.stderr)
        response = await handle.final()
    print()
    print(f"{response.usage.output_tokens} output tokens, {response.usage.cost_in_usd:.6f} USD")


async def stream_agent(
    bound: BoundLLM[str, HasTools], tool_manager: ToolManager, prompt: str, max_turns: int = 10
) -> str:
    """Run the streaming ReAct loop: print text live, dispatch the completed tool calls between turns.

    Read the assembled assistant message and tool calls from final() rather than collecting ToolCall items by hand;
    final() is idempotent and returns the same assembled Response.
    This is the non-streaming loop from 02_tool_loop.py with the generate_one call replaced by a stream,
    including the same max_turns ceiling so a tool-looping model cannot stream forever.

    Raises:
        RuntimeError: if the model keeps calling tools for max_turns turns without returning a final answer.
    """
    messages: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        handle = bound.stream_one(messages)
        async with handle:
            async for item in handle:
                match item:
                    case str():
                        print(item, end="", flush=True)
                    case ReasoningDelta():
                        print(item.text, end="", flush=True, file=sys.stderr)
                    case ToolCall():
                        print(f"\n[calling {item.name}]")
            response = await handle.final()
        messages.append(response.assistant_message)
        if not response.tool_calls:
            return response.output
        for call in response.tool_calls:
            outcome = await tool_manager.dispatch(call)
            messages.append(outcome.tool_message)
    raise RuntimeError(f"agent did not finish within {max_turns} turns")


class WeatherArgs(BaseModel):
    """Arguments of the weather tool the streaming loop dispatches between turns."""

    city: str


async def get_weather(args: WeatherArgs) -> str:
    """Return a canned weather string for the streaming tool loop to dispatch."""
    return f"It is 18C and clear in {args.city}."


weather_tool = PydanticTool(
    name="get_weather",
    description="Return the current weather for a city.",
    args_model=WeatherArgs,
    function=get_weather,
)


async def main() -> None:
    """Run the text stream, then the streaming tool loop."""
    await stream_text()
    print()
    tool_manager = ToolManager([weather_tool])
    bound = openai_model("gpt-5.6-terra", reasoning_summary="auto").bind(
        system_prompt="Use tools to answer questions about the weather.",
        tool_manager=tool_manager,
        automatic_prompt_caching=True,
    )
    answer = await stream_agent(bound, tool_manager, "What is the weather in Oslo?")
    print(f"\nfinal answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
