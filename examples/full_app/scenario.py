"""Define the example app's tools and scripted scenarios.

research_climate and research_energy run concurrently before synthesize.
research_climate delegates to specialist.
synthesize revises drafts until critique approves one.
search returns its Usage through app_data for inclusion in the run total.

Each binding's system prompt selects an agent script by tag.
Each scenario changes one script to exercise one failure layer.
"""

from events import current_gui_emitter
from harness import Turn, call
from pydantic import BaseModel

from langchaint import PydanticTool, ToolOutputExplicit, Usage, tool

SEARCH_USAGE: Usage = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=200,
    output_tokens=40,
    output_tokens_reasoning=0,
    input_tokens_cache_read_cost_in_usd=0.0,
    input_tokens_cache_write_cost_in_usd=0.0,
    input_tokens_cache_none_cost_in_usd=0.0012,
    output_tokens_cost_in_usd=0.0008,
    provider_executed_tool_cost_in_usd=0.0,
)
"""One search call costs $0.002."""


class SearchArgs(BaseModel):
    """Define the provider schema for search arguments."""

    query: str


class CritiqueArgs(BaseModel):
    """Define the provider schema for critique arguments."""

    draft: str


class DelegateArgs(BaseModel):
    """Define the provider schema for delegate arguments."""

    question: str


@tool(description="Search the corpus for a query.", name="search")
async def search_tool(args: SearchArgs) -> ToolOutputExplicit[Usage]:
    """Return a canned result and its Usage through app_data.

    current_gui_emitter() routes progress to the dispatching run's stream.

    Raises:
        LookupError: No run installed a `GuiEmitter`.
    """
    current_gui_emitter().emit_tool_progress(
        tool_name="search", message=f"searching the corpus for {args.query!r}"
    )
    return ToolOutputExplicit(
        content=f"Top result for {args.query!r}: a paragraph of findings.",
        app_data=SEARCH_USAGE,
    )


_FIRST_VERDICT = "revise: the draft cites no figures."
"""The scripted rejection that exercises revision."""


class CritiqueVerdict(BaseModel):
    """Carry critique approval through app_data."""

    approved: bool


def build_critique_tool() -> PydanticTool[CritiqueArgs, CritiqueVerdict]:
    """Build a critique tool that rejects once and then approves."""
    pending_rejections = [_FIRST_VERDICT]

    @tool(description="Critique a draft; return approval or a revision instruction.")
    async def critique(args: CritiqueArgs) -> ToolOutputExplicit[CritiqueVerdict]:  # noqa: ARG001  # the verdict is scripted, so the draft is deliberately unread
        """Return the next scripted verdict through app_data."""
        if pending_rejections:
            return ToolOutputExplicit(
                content=pending_rejections.pop(0), app_data=CritiqueVerdict(approved=False)
            )
        return ToolOutputExplicit(content="approved", app_data=CritiqueVerdict(approved=True))

    return critique


class SubAgentError(Exception):
    """A scripted provider failure inside the specialist sub-agent."""


def build_scripts(scenario: str) -> dict[str, list[Turn]]:
    """Build each agent's turns for one scenario.

    Scenarios:
        happy: every agent completes.
        call_timeout: one turn exceeds config.generate_one_timeout_seconds.
        app_timeout: both researchers stall on their second turn.
        subagent_error: the specialist's second turn raises.
        unapproved_answer: synthesize answers before critique approves a draft.
    """
    climate_delay = 5.0 if scenario == "call_timeout" else 0.0
    app_delay = 5.0 if scenario == "app_timeout" else 0.0
    specialist_error = (
        SubAgentError("specialist backend fell over") if scenario == "subagent_error" else None
    )

    return {
        "research_climate": [
            Turn(
                tool_calls=(
                    call("search", '{"query": "sea level 2030"}'),
                    call("search", '{"query": "arctic ice extent"}'),
                ),
            ),
            Turn(
                tool_calls=(call("delegate", '{"question": "quantify the ice loss trend"}'),),
                delay_seconds=climate_delay + app_delay,
            ),
            Turn(text="Climate: sea level and ice trends summarized."),
        ],
        "specialist": [
            Turn(tool_calls=(call("search", '{"query": "ice loss gigatonnes per year"}'),)),
            Turn(text="Ice loss is roughly 270 Gt/yr.", error=specialist_error),
        ],
        "research_energy": [
            Turn(tool_calls=(call("search", '{"query": "renewable share 2030"}'),)),
            Turn(
                text="Energy: renewables reach a third of supply.",
                delay_seconds=app_delay,
            ),
        ],
        "synthesize": _synthesize_turns(skips_critique=scenario == "unapproved_answer"),
    }


def _synthesize_turns(*, skips_critique: bool) -> list[Turn]:
    """Build synthesize's turns.

    skips_critique starts with an unapproved answer to exercise self_correction_enabled.
    The scripted rejection requires a second critique before the loop accepts an answer.
    """
    if skips_critique:
        return [
            Turn(text="Synthesis: an answer with no critique behind it."),
            Turn(tool_calls=(call("critique", '{"draft": "the bounced draft"}'),)),
            Turn(text="Synthesis: a revised answer, still unapproved."),
            Turn(tool_calls=(call("critique", '{"draft": "the twice-bounced draft"}'),)),
            Turn(text="Synthesis: climate and energy findings reconciled."),
        ]
    return [
        Turn(tool_calls=(call("critique", '{"draft": "first draft"}'),)),
        Turn(tool_calls=(call("critique", '{"draft": "second draft with figures"}'),)),
        Turn(text="Synthesis: climate and energy findings reconciled."),
    ]
