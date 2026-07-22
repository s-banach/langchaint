"""The per-agent configuration each run is constructed with at launch time.

Every limit an agent is subject to lives here, so a reader answers "what is this agent allowed to do"
from one frozen object rather than from arguments scattered across the launch site.

timeout_seconds is applied inside the run itself (AgentRun.final), so no caller can launch an agent
without its deadline.

The two timeouts are different failure modes and neither substitutes for the other:
per_call_timeout_seconds bounds one provider request, and the run goes on to its next turn with the
same conversation, since an abandoned request appended nothing; timeout_seconds bounds the whole run,
so an agent that keeps making fast progress toward nothing still stops. Because the per-call deadline is
the inner one, a single hung request always trips it first, so timeout_seconds fires on cumulative time.
A run whose every call hangs is bounded by max_turns rather than looping forever.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """One agent's identity and limits, fixed when the run is constructed.

    name is the last segment of the run's agent_path, minus the spawn index a tool-spawned run
    carries. The scripted adapter selects a script by the "[tag]" prefix of system_prompt;
    __post_init__ rejects a prefix that differs from name, so the tag cannot drift.
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

    def __post_init__(self) -> None:
        """Check the tag the scripted adapter reads out of system_prompt.

        A tag that names another agent would silently play that agent's script, so the drift is an
        error at construction instead.

        Raises:
            ValueError: system_prompt does not start with "[name] ".
        """
        tag = f"[{self.name}] "
        if not self.system_prompt.startswith(tag):
            raise ValueError(
                f"system_prompt must start with {tag!r} so the scripted adapter selects "
                f"{self.name}'s script; got {self.system_prompt!r}"
            )


def build_configs() -> dict[str, AgentConfig]:
    """Build the launch-time config of every agent in the graph, keyed by AgentConfig.name.

    Each agent gets its own limits: the researchers are allowed more turns than the specialist,
    and only synthesize self-corrects, so the critique pass is stated once here rather than
    inferred from which tools someone happened to pass at the call site.
    """
    configs = (
        AgentConfig(
            name="research_climate",
            system_prompt="[research_climate] Research the climate outlook.",
            max_turns=6,
            timeout_seconds=2.0,
            per_call_timeout_seconds=1.5,
        ),
        AgentConfig(
            name="research_energy",
            system_prompt="[research_energy] Research the energy outlook.",
            max_turns=6,
            max_tool_calls=6,
            timeout_seconds=2.0,
            per_call_timeout_seconds=1.5,
        ),
        AgentConfig(
            name="specialist",
            system_prompt="[specialist] Answer the question with one search.",
            max_turns=3,
            max_tool_calls=2,
            timeout_seconds=1.5,
            per_call_timeout_seconds=1.0,
        ),
        AgentConfig(
            name="synthesize",
            system_prompt="[synthesize] Reconcile the findings.",
            max_turns=6,
            max_tool_calls=4,
            timeout_seconds=3.0,
            per_call_timeout_seconds=1.0,
            self_correction_enabled=True,
        ),
    )
    return {config.name: config for config in configs}
