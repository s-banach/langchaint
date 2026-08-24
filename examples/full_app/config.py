"""Define each agent's identity and limits.

generate_one_timeout_seconds bounds one generate_one call.
A timed-out call appends no messages, and the next turn uses the same messages.
max_turns bounds repeated timeouts.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Configure one agent.

    name matches the system_prompt tag and the last segment of agent_path before any spawn index.
    max_tool_calls bounds the whole run and declines excess calls with an error ToolMessage.
    max_attempts includes the first request of each generate_one call.
    max_cost_in_usd stops new turns after settled spend reaches the limit.
    self_correction_enabled requires critique approval before a final answer.
    """

    name: str
    system_prompt: str
    automatic_cache_breakpoints: bool
    max_turns: int = 8
    max_tool_calls: int = 12
    max_attempts: int = 2
    generate_one_timeout_seconds: float = 10.0
    max_cost_in_usd: float | None = None
    self_correction_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate the system_prompt tag and max_cost_in_usd.

        Raises:
            ValueError: system_prompt does not start with "[name] ".
                The same exception applies when max_cost_in_usd is not positive and finite.
        """
        tag = f"[{self.name}] "
        if not self.system_prompt.startswith(tag):
            raise ValueError(
                f"system_prompt must start with {tag!r}. "
                f"ScriptedAdapter uses the tag to select {self.name!r}. "
                f"Got {self.system_prompt!r}."
            )
        if self.max_cost_in_usd is not None and (
            not math.isfinite(self.max_cost_in_usd) or self.max_cost_in_usd <= 0
        ):
            raise ValueError("max_cost_in_usd must be positive and finite")


def build_configs() -> dict[str, AgentConfig]:
    """Build each agent's config keyed by AgentConfig.name."""
    configs = (
        AgentConfig(
            name="research_climate",
            system_prompt="[research_climate] Research the climate outlook.",
            automatic_cache_breakpoints=False,
            max_turns=6,
            generate_one_timeout_seconds=1.5,
        ),
        AgentConfig(
            name="research_energy",
            system_prompt="[research_energy] Research the energy outlook.",
            automatic_cache_breakpoints=False,
            max_turns=6,
            max_tool_calls=6,
            generate_one_timeout_seconds=1.5,
        ),
        AgentConfig(
            name="specialist",
            system_prompt="[specialist] Answer the question with one search.",
            automatic_cache_breakpoints=False,
            max_turns=3,
            max_tool_calls=2,
            generate_one_timeout_seconds=1.0,
        ),
        AgentConfig(
            name="synthesize",
            system_prompt="[synthesize] Reconcile the findings.",
            automatic_cache_breakpoints=False,
            max_turns=6,
            max_tool_calls=4,
            generate_one_timeout_seconds=1.0,
            self_correction_enabled=True,
        ),
    )
    return {config.name: config for config in configs}
