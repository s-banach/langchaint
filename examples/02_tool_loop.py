"""A tool loop: deny a call before it runs, dispatch the rest, total their fees."""

from pydantic import BaseModel

from langchaint import (
    Message,
    PydanticTool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    UserMessage,
)
from langchaint.openai import openai_model


# pydantic puts this docstring in model_json_schema, which PydanticTool sends as args_schema.
class WeatherArgs(BaseModel):
    """Which city to report the weather for."""

    city: str


class ApiFee(BaseModel):
    """What one weather lookup cost the application."""

    fee_in_usd: float


async def get_weather(args: WeatherArgs) -> ToolOutputExplicit[ApiFee]:
    """Report the weather to the model, and the lookup's fee to the application."""
    return ToolOutputExplicit(
        content=f"It is 18C and clear in {args.city}.", app_data=ApiFee(fee_in_usd=0.01)
    )


weather_tool = PydanticTool(
    name="get_weather",
    description="Return the current weather for a city.",
    args_model=WeatherArgs,
    function=get_weather,
)


def approve_or_deny(call: ToolCall) -> ToolMessage | None:
    """Deny a transfer_funds call, and approve every other one."""
    if call.name == "transfer_funds":
        return ToolMessage.error(call, "The user declined this action.")
    return None


async def run_agent(
    prompt: str,
    max_turns: int = 10,
) -> tuple[str, float]:
    """Run the tool loop, returning the answer and what the tools charged.

    Raises:
        RuntimeError: the model called tools for max_turns turns without a final answer.
        GenerationError: a bound.generate_one call failed.
        DispatchExceptionGroup: a tool function raised.
    """
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Use tools to answer the user's question.",
        tool_manager=ToolManager([weather_tool]),
        automatic_prompt_caching=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    fees_in_usd = 0.0
    for _ in range(max_turns):
        response = await bound.generate_one(messages)
        messages.append(response.assistant_message)
        if not response.tool_calls:
            return response.output, fees_in_usd

        outcomes = await bound.tool_manager.dispatch_many(
            response.tool_calls, precomputed=approve_or_deny
        )
        for outcome in outcomes:
            if outcome.kind == "handled" and isinstance(outcome.app_data, ApiFee):
                fees_in_usd += outcome.app_data.fee_in_usd
            messages.append(outcome.tool_message)
    raise RuntimeError(f"agent did not finish within {max_turns} turns")
