"""Run a complete structured tool loop."""

from pydantic import BaseModel

from langchaint import Message, UserMessage, tool
from langchaint.openai import OpenAI


class WeatherArgs(BaseModel):
    """Select the city for a weather report."""

    city: str


class FinalAnswer(BaseModel):
    """Hold the answer returned after tool dispatch."""

    answer: str


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> str:
    """Return a fixed weather report."""
    return f"It is 18C and clear in {args.city}."


async def run_tool_loop(prompt: str, max_turns: int = 10) -> FinalAnswer:
    """Dispatch tools until the model returns FinalAnswer.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
        DispatchExceptionGroup: A tool function raises.
        RuntimeError: The model exceeded `max_turns`.
    """
    openai = OpenAI()
    bound = openai.model("gpt-5.6-terra").bind(
        system_prompt="Use get_weather when needed. Return FinalAnswer when finished.",
        tools=[get_weather],
        response_format=FinalAnswer,
        automatic_cache_breakpoints=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        result = await bound.generate_one(messages)
        match result.kind:
            case "tool_call_turn":
                messages.append(result.assistant_message)
                outcomes = await bound.tool_manager.dispatch_many(result.tool_calls)
                messages.extend(outcome.tool_message for outcome in outcomes)
            case "response":
                return result.output
    raise RuntimeError(f"model did not finish within {max_turns} turns")
