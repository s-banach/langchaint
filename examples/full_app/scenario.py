"""The example app: its graph, its tools, and the scripted scenarios that exercise each failure layer.

The graph: research_climate and research_energy run concurrently, synthesize starts once both are done.
research_climate delegates a question to a specialist sub-agent, so the tree is three levels deep.
synthesize self-corrects: it drafts, calls critique, is told to revise, drafts again, is approved, answers.
search reports a flat per-call fee through app_data, so a run's total includes tool spend as well as
token spend. A sub-agent reports nothing that way: it records its own spend as it goes.

Scripts are keyed by agent tag; a binding's system prompt starts with "[tag]" and the adapter reads it.
Each named scenario perturbs one script to exercise one failure layer, leaving the rest of the graph intact.
"""

from events import ToolProgress, current_gui_emitter
from harness import Turn, call
from pydantic import BaseModel

from langchaint import ZERO_USAGE, PydanticTool, ToolOutputExplicit, Usage

SEARCH_FEE_IN_USD = 0.002
"""What one search call bills, distinct from a turn's 0.01 so a fold can be attributed."""


class SearchArgs(BaseModel):
    """Arguments of search; the schema the provider sees."""

    query: str


class CritiqueArgs(BaseModel):
    """Arguments of critique: the draft the agent wants checked."""

    draft: str


class DelegateArgs(BaseModel):
    """Arguments of delegate: the question the parent hands to a sub-agent."""

    question: str


async def search(args: SearchArgs) -> ToolOutputExplicit[Usage]:
    """Return a canned result and report the call's flat fee as a Usage through app_data.

    content is what the model reads; the Usage rides to the loop, which folds it into the run total.
    A per-call fee is a Usage with zero token counters and the fee in cost_in_usd.

    The progress report reaches the dispatching run's stream through current_gui_emitter, which is the
    one thing threading cannot do without a parameter on every tool: this function holds no handle on
    the run that dispatched it. The same function called from a parent and from a sub-agent reports to
    two different streams, because each run's task installed its own emitter.

    Raises:
        LookupError: dispatched outside a run's loop, where no emitter is installed.
    """
    emitter = current_gui_emitter()
    emitter.emit(
        ToolProgress(
            agent_path=emitter.agent_path,
            tool_name="search",
            message=f"searching the corpus for {args.query!r}",
        )
    )
    return ToolOutputExplicit(
        content=f"Top result for {args.query!r}: a paragraph of findings.",
        app_data=ZERO_USAGE.model_copy(update={"cost_in_usd": SEARCH_FEE_IN_USD}),
    )


_FIRST_VERDICT = "revise: the draft cites no figures."
"""The one rejection every self-correcting run gets, so the revision path is always exercised."""

_pending_verdicts: list[str] = [_FIRST_VERDICT]
"""Verdicts critique has yet to hand out; mutated in place so no rebinding is needed."""


async def critique(args: CritiqueArgs) -> str:  # noqa: ARG001  # the verdict is scripted, so the draft is deliberately unread
    """Hand out the next scripted verdict, driving one revision then approval.

    A bare str is sugar for a successful ToolOutputExplicit with no app_data,
    so a tool that reports no spend of its own needs no ceremony.
    Once the scripted rejections run out every draft is approved, which is what ends the loop.
    """
    return _pending_verdicts.pop(0) if _pending_verdicts else "approved"


def reset_critique() -> None:
    """Rewind the verdict script so every scenario run sees the same self-correction sequence."""
    _pending_verdicts[:] = [_FIRST_VERDICT]


search_tool = PydanticTool(
    name="search",
    description="Search the corpus for a query.",
    args_model=SearchArgs,
    function=search,
)

critique_tool = PydanticTool(
    name="critique",
    description="Critique a draft; returns 'approved' or a revision instruction.",
    args_model=CritiqueArgs,
    function=critique,
)


class SubAgentError(Exception):
    """A scripted provider failure inside the specialist sub-agent."""


def build_scripts(scenario: str) -> dict[str, list[Turn]]:
    """Build the four agents' scripted turns, perturbed for the named scenario.

    Scenarios:
        happy: every agent completes.
        call_timeout: one research_climate turn hangs past the per-call timeout, so that call is
            counted abandoned; the run keeps what it had already spent.
        agent_timeout: research_energy runs an extra turn with every turn delayed to well inside its
            per-call timeout, so no single call trips and the per-agent deadline fires on cumulative
            time. Both scenarios end with one call cancelled in flight, so what separates them is which
            deadline did it: here two calls completed first, each well inside the per-call limit.
        app_timeout: both researchers stall on their second turn, so the whole-app deadline fires
            after each has already billed a turn, which is what makes the surviving fold worth reading.
        subagent_error: the specialist's second turn raises, after it has already billed one turn.
        unapproved_answer: synthesize answers before critiquing, so the self-correction bounce fires.
    """
    climate_delay = 5.0 if scenario == "call_timeout" else 0.0
    # Comfortably under research_energy's per_call_timeout_seconds and, over three turns, past its
    # timeout_seconds; build_configs gives the researchers 1.5 and 2.0 seconds.
    slow_turn_delay = 0.8 if scenario == "agent_timeout" else 0.0
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
        "research_energy": _energy_turns(trailing_delay=app_delay, per_turn_delay=slow_turn_delay),
        "synthesize": _synthesize_turns(skips_critique=scenario == "unapproved_answer"),
    }


def _energy_turns(*, trailing_delay: float, per_turn_delay: float) -> list[Turn]:
    """Build research_energy's turns, with an extra search turn once per_turn_delay is non-zero.

    Three delayed turns are what push cumulative time past the per-agent deadline while every single
    call stays inside the per-call one; at per_turn_delay 0 the extra turn is dropped and the script is
    the two turns every other scenario runs.
    """
    searches = [
        Turn(
            tool_calls=(call("search", '{"query": "renewable share 2030"}'),),
            delay_seconds=per_turn_delay,
        )
    ]
    if per_turn_delay:
        searches.append(
            Turn(
                tool_calls=(call("search", '{"query": "grid storage buildout"}'),),
                delay_seconds=per_turn_delay,
            )
        )
    return [
        *searches,
        Turn(
            text="Energy: renewables reach a third of supply.",
            delay_seconds=trailing_delay + per_turn_delay,
        ),
    ]


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
