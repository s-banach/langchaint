"""Consume every `StreamItem` variant."""

from pydantic import BaseModel

from langchaint import (
    ReasoningDelta,
    Response,
    SpecificToolChoice,
    StreamItem,
    ToolCall,
    ToolCallDelta,
    tool,
)
from langchaint.openai import OpenAI


class WeatherArgs(BaseModel):
    """Select the city for a weather report."""

    city: str


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> str:
    """Return a fixed weather report."""
    return f"It is 18C and clear in {args.city}."


def print_stream_item(item: StreamItem) -> None:
    """Print one `StreamItem` using its variant's fields."""
    match item:
        case str():
            print(item, end="", flush=True)
        case ReasoningDelta():
            print(f"reasoning: {item.text}")
        case ToolCallDelta():
            print(f"{item.name}[{item.id}] arguments: {item.partial_args_json}")
        case ToolCall():
            print(f"completed call: {item.name}({item.args_json})")


async def stream_tool_call() -> Response[str]:
    """Print stream items and return the assembled `Response`.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: The stream ended with a generation failure.
        StreamProtocolError: The stream ended without a terminal event.
    """
    openai = OpenAI()
    bound = openai.model("gpt-5.6-terra", reasoning_summary="auto").bind(
        tools=[get_weather],
        tool_choice=SpecificToolChoice(tool_name=get_weather.name),
    )

    async with bound.stream_one("What is the weather in Oslo?") as handle:
        async for item in handle:
            print_stream_item(item)
        return await handle.final()
