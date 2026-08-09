"""Run a complete structured tool loop."""

from pydantic import BaseModel

from langchaint import Message, Response, ToolCallTurn, ToolManager, UserMessage, tool
from langchaint.openai import OpenAIAccount


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
    """Dispatch tool turns until the model returns `FinalAnswer`.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: A `generate_one` call failed.
        DispatchExceptionGroup: A tool function raised.
        RuntimeError: The model exceeded `max_turns`.
    """
    account = OpenAIAccount()
    tool_manager = ToolManager([get_weather])
    bound = account.model("gpt-5.6-terra").bind(
        system_prompt="Use get_weather when needed. Return FinalAnswer when finished.",
        tool_manager=tool_manager,
        response_format=FinalAnswer,
        automatic_prompt_caching=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        result = await bound.generate_one(messages)
        match result:
            case ToolCallTurn():
                messages.append(result.assistant_message)
                outcomes = await tool_manager.dispatch_many(result.tool_calls)
                messages.extend(outcome.tool_message for outcome in outcomes)
            case Response():
                return result.output
    raise RuntimeError(f"model did not finish within {max_turns} turns")
