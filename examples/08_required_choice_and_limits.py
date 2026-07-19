"""A budgeted tool_choice="required" loop: every run exits through a CaptureTool, forced when the budget is spent.

tool_choice="required" means a turn can never end in plain text, so the loop needs a tool that ends the run.
final_response is that tool, a CaptureTool: capture validates the arguments and returns the args_model instance.
Two budgets bound a run, both owned by the loop.
Spending max_turns rebinds tool_choice to SpecificToolChoice, so the provider forces a final_response call.
Spending tool_budget_in_usd answers further non-final tool calls with a refusal ToolMessage telling the model to stop.
A tool reports its own cost by returning a Usage as app_data; the loop folds it into the run total.
search reports a flat per-call fee that way, and delegate reports a whole sub-agent run's Usage the same way.
Delegation therefore falls under the same cost rule as any billed tool.

The LangChain call-for-call map lives in MIGRATING_FROM_LANGCHAIN.md; this file shows the langchaint calls themselves.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from langchaint import (
    ZERO_USAGE,
    BoundLLM,
    CaptureTool,
    DispatchCaptured,
    DispatchHandled,
    Message,
    PydanticTool,
    Response,
    SpecificToolChoice,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    Usage,
    UserMessage,
)
from langchaint.openai import openai_model

FORCED_TRIES = 3
"""Turns the forcing phase gets to produce a valid final_response call before the run gives up."""

SEARCH_FEE_IN_USD = 0.002
"""The flat fee search reports per call; a search API bills per call, not per token."""


class SearchArgs(BaseModel):
    """Arguments of search; also the schema the provider sees."""

    query: str


async def search(args: SearchArgs) -> ToolOutputExplicit[Usage]:
    """Return a canned search result and report the call's flat fee as a Usage through app_data.

    content is what the model reads; the Usage rides to the loop, which folds it into the run total.
    cost_in_usd is the Usage field that absorbs provider-variant billing structure.
    A per-call fee is therefore a Usage with zero token counters and the fee in cost_in_usd.
    """
    return ToolOutputExplicit(
        content=f"Top result for {args.query!r}: langchaint is a provider-neutral LLM client library.",
        app_data=ZERO_USAGE.model_copy(update={"cost_in_usd": SEARCH_FEE_IN_USD}),
    )


search_tool = PydanticTool(
    name="search",
    description="Search the web and return the top result.",
    args_model=SearchArgs,
    function=search,
)


class FinalResponse(BaseModel):
    """The parent run's structured answer; as final_response's args_model it is also the schema the provider sees."""

    answer: str
    sources: list[str]


def build_final_response_tool[FinalT: BaseModel](response_model: type[FinalT]) -> CaptureTool[FinalT]:
    """Build one loop's final_response tool; only args_model varies between the parent and the sub-agent."""
    return CaptureTool(
        name="final_response",
        description="Submit your final structured answer. Call this exactly once, when you are done.",
        args_model=response_model,
    )


@dataclass(frozen=True, kw_only=True)
class RunResult[FinalT: BaseModel]:
    """A completed run: the captured final response and the run's paid total.

    usage folds the generate usage with every tool-reported Usage: the run's whole spend, sub-agent runs included.
    """

    final_response: FinalT
    usage: Usage


async def run_agent[FinalT: BaseModel](
    bound: BoundLLM[str],
    tool_manager: ToolManager,
    final_response_tool: CaptureTool[FinalT],
    prompt: str,
    *,
    max_turns: int,
    tool_budget_in_usd: float,
) -> RunResult[FinalT]:
    """Run the required-choice loop until a valid final_response call is captured.

    Every turn calls tools, because bound froze tool_choice="required".
    A final_response call is answered by capture, and a valid capture completes the run.
    Every other call is dispatched, unless the tool-reported spend has reached tool_budget_in_usd.
    A call over that budget is answered with a refusal ToolMessage the model reads and adapts to.
    Spending max_turns enters the forcing phase: tool_choice is rebound to SpecificToolChoice.
    The provider then forces final_response, and FORCED_TRIES turns remain to produce a valid call.
    An invalid final_response call carries its field-level corrections back to the model, which retries.
    The loop retains one Response per generate call and one Usage per reporting tool call.
    The budget check and the returned usage fold from those records, so a total and its records cannot desync.
    bound must have been bound with this same tool_manager and with tool_choice="required".
    GenerationError propagates from every generate call: the loop handles model behavior, not generation failure.

    Raises:
        RuntimeError: if no valid final_response call is captured within FORCED_TRIES forced turns.
    """
    conversation: list[Message] = [UserMessage(content=prompt)]
    responses: list[Response[str]] = []
    tool_reported_usages: list[Usage] = []

    async def answer_calls(tool_calls: Sequence[ToolCall], *, forcing: bool) -> FinalT | None:
        """Answer every call of one turn in order; the last valid final_response capture wins.

        In the forcing phase a non-final call is answered with a redirect refusal, so no over-budget work executes.
        Every tool_call therefore gets its ToolMessage, whichever branch answers it.
        """
        final_response: FinalT | None = None
        for call in tool_calls:
            if call.name == final_response_tool.name:
                outcome = await final_response_tool.capture(call)
                if isinstance(outcome, DispatchCaptured):
                    final_response = outcome.captured
                tool_message = outcome.tool_message
            elif forcing:
                redirect = f"Call {final_response_tool.name}."
                tool_message = ToolMessage(tool_call_id=call.id, content=redirect, is_error=True)
            elif sum(usage.cost_in_usd for usage in tool_reported_usages) >= tool_budget_in_usd:
                refusal = "Stop calling tools: the run's tool budget is spent."
                tool_message = ToolMessage(tool_call_id=call.id, content=refusal, is_error=True)
            else:
                dispatch_outcome = await tool_manager.dispatch(call)
                match dispatch_outcome:
                    case DispatchHandled(app_data=Usage() as reported_usage):
                        tool_reported_usages.append(reported_usage)
                tool_message = dispatch_outcome.tool_message
            conversation.append(tool_message)
        return final_response

    async def take_turn(turn_bound: BoundLLM[str], *, forcing: bool) -> FinalT | None:
        """Generate one turn on turn_bound, retain its Response, and answer its calls.

        GenerationError propagates from generate_one.
        """
        response = await turn_bound.generate_one(conversation)
        responses.append(response)
        conversation.append(response.assistant_message)
        return await answer_calls(response.tool_calls, forcing=forcing)

    def completed(final_response: FinalT) -> RunResult[FinalT]:
        """Assemble the RunResult; usage folds from the retained records."""
        generate_usage = Usage.sum_of(response.usage for response in responses)
        return RunResult(final_response=final_response, usage=generate_usage + Usage.sum_of(tool_reported_usages))

    for _ in range(max_turns):
        final_response = await take_turn(bound, forcing=False)
        if final_response is not None:
            return completed(final_response)
    forced = bound.rebind(tool_choice=SpecificToolChoice(tool_name=final_response_tool.name))
    for _ in range(FORCED_TRIES):
        final_response = await take_turn(forced, forcing=True)
        if final_response is not None:
            return completed(final_response)
    raise RuntimeError(f"no valid {final_response_tool.name} call within {FORCED_TRIES} forced turns")


class SubAgentReport(BaseModel):
    """The sub-agent run's structured answer; a completed sub-run's instance rides as JSON in delegate's content."""

    findings: list[str]


class DelegateArgs(BaseModel):
    """Arguments of delegate: the question the parent hands to the sub-agent."""

    query: str


def build_delegate_tool() -> PydanticTool[DelegateArgs, Usage]:
    """Build delegate, the tool whose function runs a whole sub-agent loop.

    content is the sub-run's report as JSON; app_data is the sub-run's whole Usage.
    The parent loop therefore accrues the delegation's real cost under the same rule as search's flat fee.
    The sub-agent is the same run_agent loop on its own binding, with search as its one working tool.
    run_agent's RuntimeError propagates out of delegate if the sub-run gives up; dispatch treats that as a defect.
    A production app would catch it and return an is_error ToolOutputExplicit that still reports the spend.
    """
    sub_agent_final_response_tool = build_final_response_tool(SubAgentReport)
    sub_agent_tool_manager = ToolManager([search_tool, sub_agent_final_response_tool])
    sub_agent_bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Answer the delegated question using search, then submit final_response.",
        tool_manager=sub_agent_tool_manager,
        tool_choice="required",
        automatic_prompt_caching=True,
    )

    async def delegate(args: DelegateArgs) -> ToolOutputExplicit[Usage]:
        """Run the sub-agent loop and report its answer plus its whole spend; run_agent's exceptions propagate."""
        sub_run = await run_agent(
            sub_agent_bound,
            sub_agent_tool_manager,
            sub_agent_final_response_tool,
            args.query,
            max_turns=4,
            tool_budget_in_usd=0.01,
        )
        return ToolOutputExplicit(content=sub_run.final_response.model_dump_json(), app_data=sub_run.usage)

    return PydanticTool(
        name="delegate",
        description="Delegate a focused research question to the sub-agent.",
        args_model=DelegateArgs,
        function=delegate,
    )


async def budgeted_run() -> None:
    """Build the parent agent over search and delegate, then run one budgeted loop end to end.

    The same ToolManager instance goes into bind (so its schemas are sent) and into the loop (so calls dispatch to it).
    final_response_tool is in the manager so bind sends its schema.
    The loop answers its calls through capture, by name, before dispatch is considered.
    automatic_prompt_caching=True because the loop re-sends the growing conversation every turn.
    """
    final_response_tool = build_final_response_tool(FinalResponse)
    tool_manager = ToolManager([search_tool, build_delegate_tool(), final_response_tool])
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Research the question with search, delegating deep sub-questions to delegate.",
        tool_manager=tool_manager,
        tool_choice="required",
        automatic_prompt_caching=True,
    )
    run = await run_agent(
        bound,
        tool_manager,
        final_response_tool,
        "What is langchaint, and who should use it?",
        max_turns=6,
        tool_budget_in_usd=0.02,
    )
    print(run.final_response)
    print(f"paid total: {run.usage.cost_in_usd:.4f} USD")


async def forced_exit_run() -> None:
    """Run with max_turns=1 so the exit goes through the forcing phase.

    The one normal turn nearly always goes to search, so the run enters the forcing phase.
    tool_choice is rebound to SpecificToolChoice and the provider forces final_response.
    The budget changed how the run ended, not what it returned: the result is still a validated FinalResponse.
    """
    final_response_tool = build_final_response_tool(FinalResponse)
    tool_manager = ToolManager([search_tool, final_response_tool])
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Research the question with search.",
        tool_manager=tool_manager,
        tool_choice="required",
        automatic_prompt_caching=True,
    )
    run = await run_agent(
        bound,
        tool_manager,
        final_response_tool,
        "What is langchaint, and who should use it?",
        max_turns=1,
        tool_budget_in_usd=0.02,
    )
    print(run.final_response)


async def main() -> None:
    """Run every snippet in this file."""
    await budgeted_run()
    await forced_exit_run()


if __name__ == "__main__":
    asyncio.run(main())
