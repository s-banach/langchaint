"""Drive task_stream.py through every failure layer and print what the UI saw.

SCENARIOS is the run configuration, committed as data: each row names the script to play and the limits
to play it under, so narrowing a run means editing a row that shows in `git diff` rather than passing an
argument that leaves no trace. main() takes no arguments for the same reason.
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
    """One row of the run table: which script to play and the limits to play it under.

    script_name is separate from name because tool_budget is the happy script run under a smaller
    budget, so it perturbs the config rather than the script.
    climate_max_tool_calls of None leaves AgentConfig's own default in place, so the default has one
    home and cannot drift from a copy written here.
    """

    name: str
    script_name: str
    app_timeout_seconds: float = 30.0
    climate_max_tool_calls: int | None = None


SCENARIOS = (
    Scenario(name="happy", script_name="happy"),
    Scenario(name="subagent_error", script_name="subagent_error"),
    Scenario(name="call_timeout", script_name="call_timeout"),
    Scenario(name="agent_timeout", script_name="agent_timeout"),
    Scenario(name="app_timeout", script_name="app_timeout", app_timeout_seconds=0.5),
    Scenario(name="tool_budget", script_name="happy", climate_max_tool_calls=1),
    Scenario(name="unapproved_answer", script_name="unapproved_answer"),
)


def print_event(event: Event) -> None:
    """Render one event as a single line and print it; every scenario's App gets this as on_event."""
    print(render(event))


def build_app(scenario: Scenario) -> App:
    """Build the App over one scenario's scripted LLM and configs.

    Wrapping for tracing is unconditional and happens inside App: with no TracerProvider configured
    this run gets non-recording spans, so tracing costs nothing here and enabling it is SDK
    configuration, never a change to this file.
    """
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
    )


async def run_scenario(scenario: Scenario) -> None:
    """Run one scenario end to end and print the event stream, then the outcome."""
    app = build_app(scenario)
    print(f"\n=== {scenario.name} (app timeout {scenario.app_timeout_seconds}s) ===")
    timed_out = False
    try:
        async with asyncio.timeout(scenario.app_timeout_seconds):
            await app.run()
    except TimeoutError:
        # The turn logs are already final here: the cancellation unwound the whole tree before the
        # TimeoutError reached this frame.
        timed_out = True
        print("!! whole-app timeout fired")
    print(f"--- final answer: {app.final_answer!r}")
    print(f"--- app timed out: {timed_out}")
    # Any metric one could want is a fold over each run's ordered turn_log; none is computed here.


async def main() -> None:
    """Run every scenario in table order."""
    for scenario in SCENARIOS:
        await run_scenario(scenario)


if __name__ == "__main__":
    asyncio.run(main())
