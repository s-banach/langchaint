"""The example app: its graph, its tools, and the scripted scenarios that exercise each failure layer.

The graph: research_climate and research_energy run concurrently, synthesize starts once both are done.
research_climate delegates a question to a specialist sub-agent, so the tree is three levels deep.
synthesize self-corrects: it drafts, calls critique, is told to revise, drafts again, is approved, answers.
search reports a constant Usage through app_data, standing in for a model call of its own, so a
run's total includes what its tools spent. A sub-agent reports nothing that way: it records its own spend as it goes.

Scripts are keyed by agent tag; a binding's system prompt starts with "[tag]" and the adapter reads it.
Each named scenario perturbs one script to exercise one failure layer, leaving the rest of the graph intact.
"""

from events import current_gui_emitter
from harness import Turn, call
from pydantic import BaseModel

from langchaint import PydanticTool, ToolOutputExplicit, Usage, tool

SEARCH_USAGE = Usage(
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
"""What one search call bills, summing to 0.002, distinct from a turn's 0.01 so a fold can be attributed."""


class SearchArgs(BaseModel):
    """Arguments of search; the schema the provider sees."""

    query: str


class CritiqueArgs(BaseModel):
    """Arguments of critique: the draft the agent wants checked."""

    draft: str


class DelegateArgs(BaseModel):
    """Arguments of delegate: the question the parent hands to a sub-agent."""

    question: str


@tool(description="Search the corpus for a query.", name="search")
async def search_tool(args: SearchArgs) -> ToolOutputExplicit[Usage]:
    """Return a canned result and report what the call spent as a Usage through app_data.

    content is what the model reads; the Usage rides to the loop, which folds it into the run total.
    A tool that calls a model of its own reports that call's Usage, which is what makes the run total
    cover tool spend as well as the agent's own turns.

    The progress report reaches the dispatching run's stream through current_gui_emitter, which is the
    one thing threading cannot do without a parameter on every tool: this function holds no handle on
    the run that dispatched it. The same function called from a parent and from a sub-agent reports to
    two different streams, because each run's task installed its own emitter.

    Raises:
        LookupError: dispatched outside a run's loop, where no emitter is installed.
    """
    current_gui_emitter().emit_tool_progress(
        tool_name="search", message=f"searching the corpus for {args.query!r}"
    )
    return ToolOutputExplicit(
        content=f"Top result for {args.query!r}: a paragraph of findings.",
        app_data=SEARCH_USAGE,
    )


_FIRST_VERDICT = "revise: the draft cites no figures."
"""The one rejection every self-correcting run gets, so the revision path is always exercised."""


class CritiqueVerdict(BaseModel):
    """The verdict critique routes to the loop through app_data, read back by matching the type.

    approved is the release condition of a self-correcting run; the words the model reads ride
    content. A pydantic model rather than a dataclass because ToolManager's app_data channel is
    BaseModel | Mapping[str, object] | None.
    """

    approved: bool


def build_critique_tool() -> PydanticTool[CritiqueArgs, CritiqueVerdict]:
    """Build critique with its own verdict script: the scripted rejection once, then approval forever.

    A fresh tool per tool manager gives each self-correcting run its own script, so no run or
    scenario rewinds another's leftover verdicts.
    """
    pending_rejections = [_FIRST_VERDICT]

    @tool(description="Critique a draft; return approval or a revision instruction.")
    async def critique(args: CritiqueArgs) -> ToolOutputExplicit[CritiqueVerdict]:  # noqa: ARG001  # the verdict is scripted, so the draft is deliberately unread
        """Hand out the next scripted verdict, driving one revision then approval.

        The approval rides app_data as a CritiqueVerdict, so the loop reads the verdict by type
        instead of parsing the prose the model reads.
        """
        if pending_rejections:
            return ToolOutputExplicit(
                content=pending_rejections.pop(0), app_data=CritiqueVerdict(approved=False)
            )
        return ToolOutputExplicit(content="approved", app_data=CritiqueVerdict(approved=True))

    return critique


class SubAgentError(Exception):
    """A scripted provider failure inside the specialist sub-agent."""


def build_scripts(scenario: str) -> dict[str, list[Turn]]:
    """Build the four agents' scripted turns, perturbed for the named scenario.

    Scenarios:
        happy: every agent completes.
        call_timeout: one turn hangs past config.generate_one_timeout_seconds, so that call is
            counted abandoned; the run keeps what it had already spent and answers on its next turn.
        app_timeout: both researchers stall on their second turn, so the whole-app deadline fires
            after each has already billed a turn, which is what makes the surviving fold worth reading.
        subagent_error: the specialist's second turn raises, after it has already billed one turn.
        unapproved_answer: synthesize answers before critiquing, so the self-correction bounce fires.
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
    """Build synthesize's turns, optionally starting with an answer it never critiqued.

    skips_critique exercises the self-correction bounce: the run is configured with
    self_correction_enabled, so a text turn that no critique has approved is sent back with an
    instruction to critique, and only an approved draft is accepted as the answer.
    It takes two bounces here, because the first critique returns the scripted rejection, so the answer
    after it is still unapproved and goes back a second time; that is the loop max_turns bounds.
    Without it the model calls critique on its own and the bounce never fires.
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
