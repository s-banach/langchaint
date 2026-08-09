"""Tracing: wrap an LLM in TracedLLM and every generate call opens an OTel span."""

from pydantic import BaseModel

from langchaint import Message, UserMessage, tool
from langchaint.openai import OpenAIAccount
from langchaint.tracing import TracedLLM, TracedToolManager


class WeatherArgs(BaseModel):
    """Which city to report the weather for."""

    city: str


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> str:
    """Return a canned weather report."""
    return f"It is 18C and clear in {args.city}."


async def traced_tool_turn() -> str:
    """Run one tool turn, with every generate call and every dispatch traced.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable during account construction.
        Exception: An owned resource close operation fails.
        GenerationError: a terminal error in bound.generate_one.
        DispatchExceptionGroup: a tool function raised.
    """
    async with OpenAIAccount() as openai:
        # capture_message_content has no default: False keeps message content off the spans.
        traced = TracedLLM(openai.model("gpt-5.6-terra"), capture_message_content=False)
        bound = traced.bind(
            system_prompt="Use tools to answer the user's question.",
            tool_manager=TracedToolManager([get_weather], capture_message_content=False),
            automatic_prompt_caching=False,
        )

        messages: list[Message] = [UserMessage(content="What is the weather in Oslo?")]
        response = await bound.generate_one(messages)
        messages.append(response.assistant_message)
        outcomes = await bound.tool_manager.dispatch_many(response.tool_calls)
        messages.extend(outcome.tool_message for outcome in outcomes)
        return (await bound.generate_one(messages)).output
