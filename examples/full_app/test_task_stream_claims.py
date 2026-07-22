"""Assert the claims task_stream.py's docstrings make, rather than trusting them.

Two of these are the reason the module has its shape.

The first is the accounting claim: the totals are final the moment an app deadline's except runs,
with no settling step, because the cancellation unwinds the whole tree before it propagates and every
record was written where the spend happened.

The second is the reach claim: a tool function reports into the on_event of whichever run dispatched
it through the ambient GuiEmitter, with no emitter parameter on any tool.

Async tests run through asyncio.run in a sync test function, the convention the rest of this repo's
suite uses, so no pytest-asyncio plugin is needed.
"""

import asyncio
from collections.abc import Callable
from typing import override

import pytest
from config import AgentConfig, build_configs
from events import (
    AgentFailed,
    AgentFinished,
    Event,
    ToolProgress,
    current_gui_emitter,
)
from harness import Turn, build_llm
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from scenario import build_scripts
from task_stream import AgentRun, App, ReActAgent, build_delegate_tool

from langchaint import AbandonedCall
from langchaint.tracing import TracedLLM


def _discard(event: Event) -> None:
    """Drop the event; a test using this reads the runs and the totals instead."""


def _abandoned_count(app: App) -> int:
    """Count the AbandonedCall records across every run's turn_log, folded after the run."""
    return sum(
        isinstance(record, AbandonedCall) for run in app.runs.values() for record in run.turn_log
    )


def _total_cost(app: App) -> float:
    """Fold the tree's whole cost from every run's turn_log, after the run."""
    return sum(run.own_usage.cost_in_usd for run in app.runs.values())


def _build_app(
    scenario: str,
    *,
    on_event: Callable[[Event], None] = _discard,
    exporter: InMemorySpanExporter | None = None,
) -> App:
    """Build an app for one scenario under a local TracerProvider."""
    tracer_provider = TracerProvider()
    if exporter is not None:
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return App(
        llm=build_llm(build_scripts(scenario)),
        configs=build_configs(),
        tracer=tracer_provider.get_tracer("full_app.test"),
        on_event=on_event,
    )


def _parent_of(span: ReadableSpan, spans: tuple[ReadableSpan, ...]) -> ReadableSpan | None:
    """Return the span that is the given span's parent, or None when it is a root."""
    if span.parent is None:
        return None
    parent_id = span.parent.span_id
    return next(
        (
            other
            for other in spans
            if other.context is not None and other.context.span_id == parent_id
        ),
        None,
    )


def _named(spans: tuple[ReadableSpan, ...], name: str) -> ReadableSpan:
    """Return the one finished span with the given name, asserting it is unique."""
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
    return matches[0]


def _attribute(span: ReadableSpan, key: str) -> object:
    """Read one attribute off a finished span, asserting the span carries attributes at all."""
    assert span.attributes is not None
    return span.attributes.get(key)


def test_one_on_event_receives_the_whole_tree() -> None:
    """Every run's events reach the one on_event, the specialist's through no forwarding at all.

    delegate constructs its sub-run with the same on_event, so three levels report to one consumer,
    and the parent's total is the prefix fold over the registry: its own log plus the specialist's.
    """
    events: list[Event] = []
    app = _build_app("happy", on_event=events.append)
    asyncio.run(app.run())
    assert {event.agent_path for event in events} == {
        "root/research_climate",
        "root/research_climate/specialist#0",
        "root/research_energy",
        "root/synthesize",
    }
    climate = app.runs["root/research_climate"]
    specialist = app.runs["root/research_climate/specialist#0"]
    assert specialist.usage.cost_in_usd > 0
    assert climate.usage.cost_in_usd == pytest.approx(
        climate.own_usage.cost_in_usd + specialist.own_usage.cost_in_usd
    )


def test_a_tools_progress_lands_in_the_on_event_of_the_run_that_dispatched_it() -> None:
    """ToolProgress from search carries the dispatching run's path, the specialist's included.

    One search function, no emitter parameter: each run's final() installed its own GuiEmitter, and
    the dispatch task copied the context holding it, so the same function reports to whichever run
    called it, and the specialist's progress is stamped with the specialist's path rather than its
    parent's.
    """
    events: list[Event] = []
    app = _build_app("happy", on_event=events.append)
    asyncio.run(app.run())
    progress_paths = [event.agent_path for event in events if isinstance(event, ToolProgress)]
    assert "root/research_climate/specialist#0" in progress_paths
    assert "root/research_energy" in progress_paths


def test_reading_the_emitter_outside_a_run_raises_naming_what_to_install() -> None:
    """current_gui_emitter outside a run's loop raises a LookupError that names the fix.

    The compile-time guarantee threading gives is lost on an ambient value, so the accessor owes the
    reader a message better than the default "no value", which names only a variable.
    """
    with pytest.raises(LookupError, match="inside a run's loop"):
        current_gui_emitter()


def test_sub_agent_span_nests_under_the_delegate_tool_span() -> None:
    """The specialist's span parents to the delegate execute_tool span with no context plumbing.

    The sub-run's final() runs in the frame delegate awaits in, which is the one the tool span is
    current in, so the nesting is ordinary frame nesting; it is asserted rather than assumed because
    moving final() behind a task created elsewhere would silently break it.
    """
    exporter = InMemorySpanExporter()
    app = _build_app("happy", exporter=exporter)
    asyncio.run(app.run())
    spans = exporter.get_finished_spans()
    specialist = _named(spans, "invoke_agent specialist")
    delegate_span = _parent_of(specialist, spans)
    assert delegate_span is not None
    assert _attribute(delegate_span, "gen_ai.tool.name") == "delegate"
    assert _parent_of(delegate_span, spans) is _named(spans, "invoke_agent research_climate")


def test_the_agent_span_parents_its_own_generate_spans() -> None:
    """A run's generate spans sit under its agent span, with the span opened the ordinary way."""
    exporter = InMemorySpanExporter()
    app = _build_app("happy", exporter=exporter)
    asyncio.run(app.run())
    spans = exporter.get_finished_spans()
    energy = _named(spans, "invoke_agent research_energy")
    generate_spans = [
        span
        for span in spans
        if _attribute(span, "gen_ai.operation.name") == "chat"
        and _parent_of(span, spans) is energy
    ]
    # research_energy's script is two turns, so two generate calls belong to it.
    assert len(generate_spans) == 2


def test_the_per_agent_deadline_fires_on_cumulative_time() -> None:
    """An ordinary enclosing timeout cuts a run off on cumulative time, no fixed-instant arithmetic."""
    app = _build_app("agent_timeout")
    asyncio.run(app.run())
    energy = app.runs["root/research_energy"]
    assert isinstance(energy, ReActAgent)
    assert "root/research_energy" in app.failures
    assert "root/research_energy" not in app.answers
    assert energy.usage.cost_in_usd > 0
    assert _abandoned_count(app) == 1
    # More than one call completed before the cut-off, which is the whole difference from the per-call
    # deadline: that one would have ended the run on its first delayed call, at turn 1.
    assert energy.turn_number >= 2


def test_the_accounting_is_final_when_the_app_deadline_lands() -> None:
    """Both researchers are mid-request when the app deadline fires; the except reads final totals.

    The TaskGroup in App.run awaits every child's unwind before the cancellation propagates, and
    generate_one appends each cancelled call's AbandonedCall inside the frame the cancellation
    unwinds, so nothing stands between the except and the totals.
    """
    app = _build_app("app_timeout")
    at_except: list[tuple[int, float]] = []

    async def drive() -> None:
        try:
            async with asyncio.timeout(0.5):
                await app.run()
        except TimeoutError:
            at_except.append((_abandoned_count(app), _total_cost(app)))

    asyncio.run(drive())
    # Climate's first turn ($0.01 plus two searches at $0.002) and energy's ($0.01 plus one search).
    # The two abandoned calls had no settled attempt, and no usage is fabricated for an in-flight
    # attempt, so they add nothing.
    assert at_except == [(2, pytest.approx(0.026))]


def test_a_run_cancelled_from_outside_emits_no_terminal_event() -> None:
    """An outside cancellation ends a run with no AgentFinished or AgentFailed.

    final() lets the CancelledError propagate rather than reporting it, so the run has no outcome to
    report. A UI that treats a terminal event as guaranteed would show these runs in flight forever,
    which is why final() says so.
    """
    events_by_path: dict[str, list[Event]] = {}

    def collect(event: Event) -> None:
        events_by_path.setdefault(event.agent_path, []).append(event)

    app = _build_app("app_timeout", on_event=collect)

    async def drive() -> None:
        try:
            async with asyncio.timeout(0.5):
                await app.run()
        except TimeoutError:
            pass

    asyncio.run(drive())
    climate = events_by_path["root/research_climate"]
    assert climate, "the cancelled run emitted events before the deadline"
    assert not isinstance(climate[-1], AgentFinished | AgentFailed)
    # The accounting survives the missing terminal event, which is the property that matters.
    assert app.runs["root/research_climate"].usage.cost_in_usd > 0


def test_a_failed_sub_agent_becomes_a_tool_message_and_the_parent_still_answers() -> None:
    """A sub-agent failure is data for the parent model, and its spend stands.

    The specialist's final() emits AgentFailed before re-raising, delegate catches the raise and
    returns an is_error tool message, and the records written as the sub-run spent survive the unwind.
    """
    events: list[Event] = []
    app = _build_app("subagent_error", on_event=events.append)
    asyncio.run(app.run())
    specialist_terminals = [
        type(event)
        for event in events
        if event.agent_path == "root/research_climate/specialist#0"
        and isinstance(event, AgentFinished | AgentFailed)
    ]
    assert specialist_terminals == [AgentFailed]
    # One turn at $0.01 and one search at $0.002 before the second turn raised; both records stand.
    assert app.runs["root/research_climate/specialist#0"].usage.cost_in_usd == pytest.approx(0.012)
    assert "root/research_climate" in app.answers


def test_each_delegate_call_registers_a_fresh_spawn_indexed_run() -> None:
    """Two delegate calls register two runs, at spawn indices #0 and #1.

    An agent's name is not unique within a parent, so identity is per spawn: without the index the
    second run would collide with the first in the registry, and a registry row is the run object
    itself, which two spawns cannot share.
    """
    tracer = TracerProvider().get_tracer("full_app.test")
    registry: dict[str, AgentRun] = {}
    # Two text turns: the shared script serves both spawns, one send each.
    llm = TracedLLM(
        build_llm({"specialist": [Turn(text="first"), Turn(text="second")]}),
        capture_message_content=False,
        tracer=tracer,
    )
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        sub_config=AgentConfig(name="specialist", system_prompt="[specialist] answer"),
        tracer=tracer,
        capture_message_content=False,
        registry=registry,
        on_event=_discard,
    )

    async def spawn_twice() -> None:
        await delegate_tool.validate_and_run('{"question": "q1"}')
        await delegate_tool.validate_and_run('{"question": "q2"}')

    asyncio.run(spawn_twice())
    assert set(registry) == {"root/parent/specialist#0", "root/parent/specialist#1"}


def test_a_second_run_under_one_agent_path_is_rejected() -> None:
    """Registering a run under an already-registered path raises instead of replacing the row.

    A registry row is one run held by reference, so a silent replacement would drop the first run's
    turn_log records from every fold; the raise forces a spawner that reuses a name to disambiguate
    the path, as delegate does with its spawn index.
    """

    class NoOpRun(AgentRun):
        """The smallest concrete run: registration is the behavior under test, so run() never runs."""

        @override
        async def run(self) -> str:
            """Return a constant; nothing in this test drives the run."""
            return "unused"

    registry: dict[str, AgentRun] = {}
    config = AgentConfig(name="twin", system_prompt="[twin] p")
    tracer = TracerProvider().get_tracer("full_app.test")
    NoOpRun(
        agent_path="root/twin", config=config, tracer=tracer, registry=registry, on_event=_discard
    )
    with pytest.raises(ValueError, match="already registered"):
        NoOpRun(
            agent_path="root/twin",
            config=config,
            tracer=tracer,
            registry=registry,
            on_event=_discard,
        )
