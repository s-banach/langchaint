"""Define scripted model turns for full-app tests."""

from collections.abc import Mapping

from config import AgentConfig

from tests.full_app_support.scripted_adapter import Turn, call


class SubAgentError(Exception):
    """A provider failure inside the specialist sub-agent."""


def build_scripts(scenario: str, configs: Mapping[str, AgentConfig]) -> dict[str, list[Turn]]:
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
        configs["research_climate"].system_prompt: [
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
        configs["specialist"].system_prompt: [
            Turn(tool_calls=(call("search", '{"query": "ice loss gigatonnes per year"}'),)),
            Turn(text="Ice loss is roughly 270 Gt/yr.", error=specialist_error),
        ],
        configs["research_energy"].system_prompt: [
            Turn(tool_calls=(call("search", '{"query": "renewable share 2030"}'),)),
            Turn(
                text="Energy: renewables reach a third of supply.",
                delay_seconds=app_delay,
            ),
        ],
        configs["synthesize"].system_prompt: _synthesize_turns(
            skips_critique=scenario == "unapproved_answer"
        ),
    }


def _synthesize_turns(*, skips_critique: bool) -> list[Turn]:
    """Build synthesize's turns.

    skips_critique starts with an unapproved answer to exercise self_correction_enabled.
    The first rejection requires a second critique before the loop accepts an answer.
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
