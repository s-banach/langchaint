"""Drive task_stream.py through every failure layer and print what the UI saw and what usage survived.

SCENARIOS is the run configuration, committed as data: each row names the script to play and the limits
to play it under, so narrowing a run means editing a row that shows in `git diff` rather than passing an
argument that leaves no trace. main() takes no arguments for the same reason.
"""

import asyncio
from dataclasses import dataclass

from config import build_configs
from harness import build_llm
from opentelemetry import trace
from render import render
from scenario import build_scripts, reset_critique
from task_stream import App, ReActAgent


@dataclass(frozen=True)
class Scenario:
    """One row of the run table: which script to play and the limits to play it under.

    script_name is separate from name because tool_budget is the happy script run under a smaller
    budget, so it perturbs the config rather than the script.
    climate_max_tool_calls of None leaves build_configs' own default in place, so the default has one
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


def build_app(scenario: Scenario) -> App:
    """Build the App over one scenario's scripted LLM and configs.

    Wrapping for tracing is unconditional and happens inside App: with no TracerProvider configured
    this run gets non-recording spans, so tracing costs nothing here and enabling it is SDK
    configuration, never a change to this file.
    """
    return App(
        llm=build_llm(build_scripts(scenario.script_name)),
        configs=build_configs(climate_max_tool_calls=scenario.climate_max_tool_calls),
        tracer=trace.get_tracer("examples.full_app"),
    )


async def run_scenario(scenario: Scenario) -> None:
    """Run one scenario end to end and print the event stream, then the surviving accounting."""
    reset_critique()
    app = build_app(scenario)
    print(f"\n=== {scenario.name} (app timeout {scenario.app_timeout_seconds}s) ===")
    timed_out = False
    try:
        async with asyncio.timeout(scenario.app_timeout_seconds):
            async for event in app:
                print(render(event))
    except TimeoutError:
        timed_out = True
        print("!! whole-app timeout fired")
    finally:
        # Unconditional: every abandon path (a deadline here, a client disconnect in a server) ends
        # in this one settle(), and after a completed run it does nothing. Cancelling is not the
        # same as having unwound; the counts below are read after this.
        await app.settle()
    print(f"--- final answer: {app.final_answer!r}")
    print(f"--- app timed out: {timed_out}")
    print(
        f"--- total usage: ${app.total_usage.cost_in_usd:.4f} "
        f"({app.total_usage.input_tokens_total}in/{app.total_usage.output_tokens}out)"
    )
    print(f"--- abandoned (cancelled) calls: {app.abandoned_calls}")
    # Each line is that run's subtree, so a parent's figure contains its sub-agents' and the lines
    # deliberately do not add up to the total above.
    for path, run in app.runs.items():
        assert isinstance(run, ReActAgent), "every run this app builds is a ReActAgent"
        print(
            f"---   {path}: ${run.usage.cost_in_usd:.4f}, turns={run.turn_number}, "
            f"tool_calls={run.tool_calls_made}, answered={run.answer is not None}"
        )


async def main() -> None:
    """Run every scenario in table order."""
    for scenario in SCENARIOS:
        await run_scenario(scenario)


if __name__ == "__main__":
    asyncio.run(main())
