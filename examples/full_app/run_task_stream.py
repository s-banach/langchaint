"""Run each task_stream.py scenario and print its events.

SCENARIOS stores the scripts and limits as committed run configuration.
"""

import asyncio
from dataclasses import dataclass, replace

from config import build_configs
from events import Event
from harness import build_llm
from opentelemetry import trace
from render import render
from scenario import build_scripts
from task_stream import App


@dataclass(frozen=True)
class Scenario:
    """Configure one named scenario.

    script_name lets tool_budget reuse the happy script with a smaller limit.
    None for climate_max_tool_calls preserves AgentConfig's default.
    """

    name: str
    script_name: str
    app_timeout_seconds: float = 30.0
    climate_max_tool_calls: int | None = None


SCENARIOS = (
    Scenario(name="happy", script_name="happy"),
    Scenario(name="subagent_error", script_name="subagent_error"),
    Scenario(name="call_timeout", script_name="call_timeout"),
    Scenario(name="app_timeout", script_name="app_timeout", app_timeout_seconds=0.5),
    Scenario(name="tool_budget", script_name="happy", climate_max_tool_calls=1),
    Scenario(name="unapproved_answer", script_name="unapproved_answer"),
)


def print_event(event: Event) -> None:
    """Render and print one event."""
    print(render(event))


def build_app(scenario: Scenario) -> App:
    """Build an App for one scenario."""
    configs = build_configs()
    if scenario.climate_max_tool_calls is not None:
        configs["research_climate"] = replace(
            configs["research_climate"], max_tool_calls=scenario.climate_max_tool_calls
        )
    return App(
        llm=build_llm(build_scripts(scenario.script_name)),
        configs=configs,
        tracer=trace.get_tracer("examples.full_app"),
        on_event=print_event,
        capture_message_content=False,
    )


async def run_scenario(scenario: Scenario) -> None:
    """Run one scenario and print its events and outcome."""
    app = build_app(scenario)
    print(f"\n=== {scenario.name} (app timeout {scenario.app_timeout_seconds}s) ===")
    timed_out = False
    try:
        async with asyncio.timeout(scenario.app_timeout_seconds):
            await app.run()
    except TimeoutError:
        # Cancellation finalizes the turn logs before TimeoutError reaches this frame.
        timed_out = True
        print("!! whole-app timeout fired")
    print(f"--- final answer: {app.final_answer!r}")
    print(f"--- app timed out: {timed_out}")
    # Derive metrics from each run's ordered turn_log.


async def main() -> None:
    """Run every scenario in table order."""
    for scenario in SCENARIOS:
        await run_scenario(scenario)


if __name__ == "__main__":
    asyncio.run(main())
