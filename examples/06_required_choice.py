"""Run a required tool loop until CaptureTool captures the final answer."""

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


class FinalAnswer(BaseModel):
    """Store the answer and its sources."""

    answer: str
    sources: list[str]


@tool(description="Search the corpus for a topic.")
async def search(args: SearchArgs) -> str:
    """Return a corpus search result."""
    return f"Three sources discuss {args.query}."


async def run_required_choice_agent(prompt: str, max_turns: int = 10) -> FinalAnswer:
    """Run until final_answer captures a FinalAnswer.

    The last turn forces final_answer.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
        RuntimeError: No turn produced a valid capture.
    """
    openai = OpenAI()
    final_answer_tool = CaptureTool(
        name="final_answer",
        description="Submit your final structured answer once.",
        args_model=FinalAnswer,
    )
    bound = openai.llm("gpt-5.6-terra").bind(
        system_prompt="Research the question, then submit final_answer.",
        tools=[search, final_answer_tool],
        tool_choice=AllowedToolsChoice(mode="required", tool_names=(search.name,)),
        automatic_cache_breakpoints=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for turn in range(max_turns):
        # Force the exit tool on the final turn.
        if turn == max_turns - 1:
            bound = bound.bind(tool_choice=SpecificToolChoice(tool_name=final_answer_tool.name))
        generation = await bound.generate_one(messages)
        if turn == 0:
            bound = bound.bind(tool_choice="required")
        messages.append(generation.assistant_message)
        for tool_call in generation.tool_calls:
            if tool_call.name != final_answer_tool.name:
                messages.append((await bound.tool_manager.dispatch(tool_call)).tool_message)
                continue
            outcome = await final_answer_tool.capture(tool_call)
            messages.append(outcome.tool_message)
            match outcome.kind:
                case "captured":
                    return outcome.captured
                case "invalid_tool_args":
                    continue
    raise RuntimeError(f"agent did not submit final_answer within {max_turns} turns")
