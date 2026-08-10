"""A tool loop under tool_choice="required", ending when the model calls a CaptureTool."""

from pydantic import BaseModel

from langchaint import (
    CaptureTool,
    DispatchCaptured,
    Message,
    SpecificToolChoice,
    ToolManager,
    UserMessage,
    tool,
)
from langchaint.openai import OpenAI


class SearchArgs(BaseModel):
    """What to search the corpus for."""

    query: str


class FinalResponse(BaseModel):
    """The agent's answer and the sources behind it."""

    answer: str
    sources: list[str]


@tool(description="Search the corpus for a topic.")
async def search(args: SearchArgs) -> str:
    """Return what the model reads as the search result."""
    return f"Three sources discuss {args.query}."


async def run_required_choice_agent(prompt: str, max_turns: int = 10) -> FinalResponse:
    """Loop until the model submits final_response, forcing that call on the last turn.

    Raises:
        OpenAIError: OpenAI credentials are unavailable during `OpenAI` construction.
        RuntimeError: no turn produced a valid capture.
        GenerationError: a generate_one call failed.
    """
    openai = OpenAI()
    final_response_tool = CaptureTool(
        name="final_response",
        description="Submit your final structured answer once.",
        args_model=FinalResponse,
    )
    bound = openai.model("gpt-5.6-terra").bind(
        system_prompt="Research the question, then submit final_response.",
        tool_manager=ToolManager([search, final_response_tool]),
        tool_choice="required",
        automatic_prompt_caching=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for turn in range(max_turns):
        # Force the exit tool on the final turn.
        if turn == max_turns - 1:
            bound = bound.rebind(
                tool_choice=SpecificToolChoice(tool_name=final_response_tool.name)
            )
        response = await bound.generate_one(messages)
        messages.append(response.assistant_message)
        for call in response.tool_calls:
            if call.name != final_response_tool.name:
                messages.append((await bound.tool_manager.dispatch(call)).tool_message)
                continue
            outcome = await final_response_tool.capture(call)
            messages.append(outcome.tool_message)
            if isinstance(outcome, DispatchCaptured):
                return outcome.captured
    raise RuntimeError(f"agent did not submit final_response within {max_turns} turns")
