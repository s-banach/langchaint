"""The per-agent configuration each run is constructed with at launch time.

Every limit an agent is subject to lives here, so a reader answers "what is this agent allowed to do"
from one frozen object rather than from arguments scattered across the launch site.

timeout_seconds is applied inside the run itself (AgentRun._driven), so no caller can launch an agent
without its deadline.

The two timeouts are different failure modes and neither substitutes for the other:
per_call_timeout_seconds bounds one provider request, and the run goes on to its next turn with the
same conversation, since an abandoned request appended nothing; timeout_seconds bounds the whole run,
so an agent that keeps making fast progress toward nothing still stops. Because the per-call deadline is
the inner one, a single hung request always trips it first, so timeout_seconds fires on cumulative time.
A run whose every call hangs is bounded by max_turns rather than looping forever.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AgentConfig:
    """One agent's identity and limits, fixed when the run is constructed.

    name is the last segment of the run's agent_path. The scripted adapter selects a script by the
    "[tag]" prefix of system_prompt instead, so the two are kept equal by convention, not by construction.
    max_tool_calls is a budget across the whole run, not per turn: calls beyond it are refused with an
    is_error tool message the model reads and adapts to, rather than dropped or raised on,
    so the model gets a chance to finish with what it already has.
    self_correction_enabled sends every final answer back for critique until some critique has returned an
    approval, so a run whose critiques keep saying "revise" keeps bouncing and max_turns is what bounds it;
    an agent with it off answers on its first text turn.
    """

    name: str
    system_prompt: str
    max_turns: int = 8
    max_tool_calls: int = 12
    timeout_seconds: float = 30.0
    per_call_timeout_seconds: float = 10.0
    self_correction_enabled: bool = False


def build_configs(*, climate_max_tool_calls: int | None = None) -> dict[str, AgentConfig]:
    """Build the launch-time config of every agent in the graph.

    Each agent gets its own limits: the researchers are allowed more turns than the specialist,
    and only synthesize self-corrects, so the critique pass is stated once here rather than
    inferred from which tools someone happened to pass at the call site.

    climate_max_tool_calls of None leaves AgentConfig's own default in place, so the number has one
    home: a copy written here would drift from it silently, since nothing compares the two.
    """
    climate = AgentConfig(
        name="research_climate",
        system_prompt="[research_climate] Research the climate outlook.",
        max_turns=6,
        timeout_seconds=2.0,
        per_call_timeout_seconds=1.5,
    )
    if climate_max_tool_calls is not None:
        climate = replace(climate, max_tool_calls=climate_max_tool_calls)
    return {
        "research_climate": climate,
        "research_energy": AgentConfig(
            name="research_energy",
            system_prompt="[research_energy] Research the energy outlook.",
            max_turns=6,
            max_tool_calls=6,
            timeout_seconds=2.0,
            per_call_timeout_seconds=1.5,
        ),
        "specialist": AgentConfig(
            name="specialist",
            system_prompt="[specialist] Answer the question with one search.",
            max_turns=3,
            max_tool_calls=2,
            timeout_seconds=1.5,
            per_call_timeout_seconds=1.0,
        ),
        "synthesize": AgentConfig(
            name="synthesize",
            system_prompt="[synthesize] Reconcile the findings.",
            max_turns=6,
            max_tool_calls=4,
            timeout_seconds=3.0,
            per_call_timeout_seconds=1.0,
            self_correction_enabled=True,
        ),
    }
