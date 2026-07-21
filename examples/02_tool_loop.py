"""The ReAct tool loop: the app owns it, langchaint ships no agent class.

Owning the loop is what lets you enforce a budget mid-run, gate a call for approval, or swap the binding between turns;
a fixed agent surface exposes none of that.
The one run_agent below is both the plain loop and the human-in-the-loop loop.
An approval gate is one optional parameter and the four-line block that reads it, not a second copy of the loop.

The LangChain call-for-call map (bind_tools, create_react_agent, interrupts) lives in MIGRATING_FROM_LANGCHAIN.md.
"""

import asyncio
from collections.abc import Callable

from pydantic import BaseModel

from langchaint import (
    BoundLLM,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchUnknownTool,
    Message,
    PydanticTool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    UserMessage,
)
from langchaint.openai import openai_model


class WeatherArgs(BaseModel):
    """The one pydantic model a tool function takes; it is also the schema the provider sees.

    There is no signature introspection and no docstring scraping, so the schema is exactly what this states.
    """

    city: str


async def get_weather(args: WeatherArgs) -> str:
    """Return a canned weather string; a tool function takes one args model and returns model-facing content.

    A bare str (or a Sequence[Part]) is sugar for a successful ToolOutputExplicit with no app_data.
    """
    return f"It is 18C and clear in {args.city}."


weather_tool = PydanticTool(
    name="get_weather",
    description="Return the current weather for a city.",
    args_model=WeatherArgs,
    function=get_weather,
)


async def run_agent(
    bound: BoundLLM[str],
    tool_manager: ToolManager,
    prompt: str,
    approve: Callable[[ToolCall], bool] | None = None,
    max_turns: int = 10,
) -> str:
    """Run the whole ReAct loop.

    Generate a turn and append the assistant message; with no tool calls the run is done and output is the answer.
    Otherwise dispatch each call, append the resulting tool message, and loop.
    dispatch never raises for a bad tool name or bad arguments; it returns an is_error tool message the model corrects,
    so the loop survives a hallucinated call.
    The approve gate is where a between-turns decision goes (a human approval prompt, a budget check, a routing choice):
    a declined call becomes an is_error ToolMessage the model reads and adapts to, exactly like any other tool failure.
    No interrupt machinery and no engine-owned state; the app holds the conversation and the control flow.
    max_turns bounds the run so a model that keeps calling tools cannot loop forever spending tokens;
    that ceiling is a budget the app owns, the kind of control an engine that hides the loop does not give you.
    bound must have been bound with this same tool_manager, so the provider was sent its schemas.

    Raises:
        RuntimeError: if the model keeps calling tools for max_turns turns without returning a final answer.
    """
    conversation: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        response = await bound.generate_one(conversation)
        conversation.append(response.assistant_message)
        if not response.tool_calls:
            return response.output
        for call in response.tool_calls:
            if approve is not None and not approve(call):
                declined = ToolMessage(
                    tool_call_id=call.id, content="The user declined this action.", is_error=True
                )
                conversation.append(declined)
                continue
            outcome = await tool_manager.dispatch(call)
            conversation.append(outcome.tool_message)
    raise RuntimeError(f"agent did not finish within {max_turns} turns")


async def basic_run() -> None:
    """Wire a ToolManager into the binding and run the loop with no approval gate.

    The same ToolManager instance goes into bind (so its schemas are sent) and into the loop (so calls dispatch to it).
    automatic_prompt_caching=True because the loop re-sends the growing conversation every turn,
    so each turn re-reads the cached prefix the previous turn wrote.
    """
    tool_manager = ToolManager([weather_tool])
    llm = openai_model("gpt-5.6-terra")
    bound = llm.bind(
        system_prompt="Use tools to answer questions about the weather.",
        tool_manager=tool_manager,
        automatic_prompt_caching=True,
    )
    answer = await run_agent(bound, tool_manager, "What is the weather in Oslo?")
    print(answer)


async def reading_dispatch_outcomes() -> None:
    """Match the three dispatch arms; every arm carries tool_message, so run_agent above never needs the match.

    Match the arms only when you want more than the reply to append:
    DispatchHandled carries the function's app_data (data the model never sees, e.g. records the tool persisted);
    DispatchInvalidToolArgs carries the neutral InvalidToolArgsDetail tuple as a required details field,
    read with no cast or assert;
    DispatchUnknownTool carries the off-list called_name.
    """
    tool_manager = ToolManager([weather_tool])
    bound = openai_model("gpt-5.6-terra").bind(
        tool_manager=tool_manager, automatic_prompt_caching=False
    )
    conversation: list[Message] = [UserMessage(content="What is the weather in Oslo?")]
    response = await bound.generate_one(conversation)
    conversation.append(response.assistant_message)
    for call in response.tool_calls:
        outcome = await tool_manager.dispatch(call)
        conversation.append(outcome.tool_message)
        match outcome:
            case DispatchHandled(app_data=app_data):
                print("tool ran; app-facing data:", app_data)
            case DispatchInvalidToolArgs(details=details):
                print("model sent bad arguments:", details)
            case DispatchUnknownTool(called_name=called_name):
                print("model called an unknown tool:", called_name)


class TransferArgs(BaseModel):
    """Arguments for a sensitive action the app wants to gate."""

    to_account: str
    amount_usd: float


class TransferReceipt(BaseModel):
    """The app-facing record a tool attaches via app_data; the model never sees it."""

    confirmation_id: str


async def transfer_funds(args: TransferArgs) -> ToolOutputExplicit[TransferReceipt]:
    """Return app_data alongside the model-facing content.

    content is what the model reads; app_data rides through to the application untouched, typed as TransferReceipt,
    so a caller that matches DispatchHandled reads confirmation_id back with no isinstance.
    """
    receipt = TransferReceipt(confirmation_id="TXN-1234")
    return ToolOutputExplicit(
        content=f"Transferred {args.amount_usd} USD to {args.to_account}.",
        app_data=receipt,
    )


transfer_tool = PydanticTool(
    name="transfer_funds",
    description="Transfer money to an account.",
    args_model=TransferArgs,
    function=transfer_funds,
)


def require_approval(call: ToolCall) -> bool:
    """Approve any call except transfer_funds, which a real app would route to a human prompt before proceeding."""
    return call.name != "transfer_funds"


async def approval_run() -> None:
    """Run the same loop with an approval gate that declines the sensitive tool.

    The declined transfer becomes an is_error ToolMessage; the model reads it and answers without the transfer.
    """
    tool_manager = ToolManager([weather_tool, transfer_tool])
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Use tools to help the user.",
        tool_manager=tool_manager,
        automatic_prompt_caching=True,
    )
    answer = await run_agent(
        bound, tool_manager, "Transfer 50 USD to account A-1.", approve=require_approval
    )
    print(answer)


async def main() -> None:
    """Run every snippet in this file."""
    await basic_run()
    await reading_dispatch_outcomes()
    await approval_run()


if __name__ == "__main__":
    asyncio.run(main())
