"""Run a required tool loop until CaptureTool returns output."""

from pydantic import BaseModel

from langchaint import (
    AllowedToolsChoice,
    CaptureTool,
    Message,
    SpecificToolChoice,
    UserMessage,
    tool,
)
from langchaint.openai import OpenAI


class SearchArgs(BaseModel):
    """Define corpus search arguments."""

    query: str


class FinalResponse(BaseModel):
    """Store the answer and its sources."""

    answer: str
    sources: list[str]


@tool(description="Search the corpus for a topic.")
async def search(args: SearchArgs) -> str:
    """Return a corpus search result."""
    return f"Three sources discuss {args.query}."


async def run_required_choice_agent(prompt: str, max_turns: int = 10) -> FinalResponse:
    """Run until final_response captures output.

    The last turn forces final_response.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
        RuntimeError: No turn produced a valid capture.
    """
    openai = OpenAI()
    final_response_tool = CaptureTool(
        name="final_response",
        description="Submit your final structured answer once.",
        args_model=FinalResponse,
    )
    bound = openai.model("gpt-5.6-terra").bind(
        system_prompt="Research the question, then submit final_response.",
        tools=[search, final_response_tool],
        tool_choice=AllowedToolsChoice(mode="required", tool_names=(search.name,)),
        automatic_cache_breakpoints=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for turn in range(max_turns):
        # Force the exit tool on the final turn.
        if turn == max_turns - 1:
            bound = bound.bind(tool_choice=SpecificToolChoice(tool_name=final_response_tool.name))
        response = await bound.generate_one(messages)
        if turn == 0:
            bound = bound.bind(tool_choice="required")
        messages.append(response.assistant_message)
        for call in response.tool_calls:
            if call.name != final_response_tool.name:
                messages.append((await bound.tool_manager.dispatch(call)).tool_message)
                continue
            outcome = await final_response_tool.capture(call)
            messages.append(outcome.tool_message)
            match outcome.kind:
                case "captured":
                    return outcome.captured
                case "invalid_tool_args":
                    continue
    raise RuntimeError(f"agent did not submit final_response within {max_turns} turns")
