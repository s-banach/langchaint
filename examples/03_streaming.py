"""Consume every `StreamItem` variant.

A completed generation's `output` contains the final text part or validated caller model.
`commentary` contains preceding plain text joined with one newline, or `None` when absent.
Application tool calls give a `ToolCallGeneration`, which has no `output`, and make every text part commentary.
Replay `assistant_message` to preserve ordered content, including reasoning.
"""

from pydantic import BaseModel

from langchaint import Generation, SpecificToolChoice, StreamItem, tool
from langchaint.openai import OpenAI


class WeatherArgs(BaseModel):
    """Select the city for a weather report."""

    city: str


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> str:
    """Return a fixed weather report."""
    return f"It is 18C and clear in {args.city}."


def print_stream_item(item: StreamItem) -> None:
    """Print one StreamItem."""
    if isinstance(item, str):
        print(item, end="", flush=True)
        return
    match item.kind:
        case "reasoning_delta":
            print(f"reasoning: {item.text}")
        case "tool_call_delta":
            print(f"{item.name}[{item.id}] arguments: {item.partial_args_json}")
        case "tool_call":
            print(f"completed call: {item.name}({item.args_json})")


async def stream_tool_call() -> Generation[str]:
    """Print stream items and return the assembled generation.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
    """
    bound = OpenAI().llm(
        "gpt-5.6-terra",
        reasoning_summary="auto",
        tools=[get_weather],
        tool_choice=SpecificToolChoice(tool_name=get_weather.name),
    )

    async with bound.stream_one("What is the weather in Oslo?") as handle:
        async for item in handle:
            print_stream_item(item)
        generation = await handle.final()
        print("commentary:", generation.commentary)
        match generation.kind:
            case "plain":
                print("output:", generation.output)
            case "tool_call":
                print("tool calls:", generation.tool_calls)
        return generation
