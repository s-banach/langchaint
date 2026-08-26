"""Verify task_stream.py behavior.

Timed-out calls remain in turn_log, and their runs continue.
GuiEmitter routes tool progress to the dispatching run's on_event.
Sync tests use asyncio.run for async behavior.
"""

import asyncio
import math
from collections.abc import Callable
from dataclasses import replace
from typing import override

import pytest
import task_stream
from config import AgentConfig, build_configs
from events import (
    AgentCancelled,
    AgentFailed,
    AgentFinished,
    Event,
    ToolProgress,
    current_gui_emitter,
)
from harness import Turn, build_llm, call
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from scenario import build_scripts
from task_stream import (
    AgentRun,
    App,
    LlmFailure,
    ReActAgent,
    ToolTurn,
    _check_cost_limit,
    _validate_tool_call_ids,
    build_delegate_tool,
)

from langchaint import ZERO_USAGE, DispatchExceptionGroup, ToolCall, tool
from langchaint.tracing import TracedLLM


def _discard(event: Event) -> None:
    """Discard one event."""


def _timed_out_count(app: App) -> int:
    return sum(
        isinstance(record, LlmFailure) and record.error.record.kind == "timed_out_error"
        for run in app.runs.values()
        for record in run.turn_log
    )


def _total_cost(app: App) -> float:
    """Sum each run's own cost."""
    return sum(run.own_usage.cost_in_usd for run in app.runs.values())


def _build_app(
    scenario: str,
    *,
    on_event: Callable[[Event], None] = _discard,
    exporter: InMemorySpanExporter | None = None,
    climate_max_tool_calls: int | None = None,
) -> App:
    """Build an app for one scenario under a local TracerProvider."""
    tracer_provider = TracerProvider()
    if exporter is not None:
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    configs = build_configs()
    if climate_max_tool_calls is not None:
        configs["research_climate"] = replace(
            configs["research_climate"], max_tool_calls=climate_max_tool_calls
        )
    return App(
        llm=build_llm(build_scripts(scenario)),
        configs=configs,
        tracer=tracer_provider.get_tracer("full_app.test"),
        on_event=on_event,
        capture_message_content=False,
    )


def _parent_of(span: ReadableSpan, spans: tuple[ReadableSpan, ...]) -> ReadableSpan | None:
    """Return span's parent or None for a root span."""
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
    """Return the unique finished span named name."""
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
    return matches[0]


def _attribute(span: ReadableSpan, key: str) -> object:
    """Return one finished span attribute."""
    assert span.attributes is not None
    return span.attributes.get(key)


def test_one_on_event_receives_the_whole_tree() -> None:
    """One on_event receives every run's events."""
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
    """ToolProgress carries the dispatching run's agent_path."""
    events: list[Event] = []
    app = _build_app("happy", on_event=events.append)
    asyncio.run(app.run())
    progress_paths = [event.agent_path for event in events if isinstance(event, ToolProgress)]
    assert "root/research_climate/specialist#0" in progress_paths
    assert "root/research_energy" in progress_paths


def test_reading_without_an_active_emitter_raises() -> None:
    """`current_gui_emitter` raises when no `GuiEmitter` is active."""
    with pytest.raises(LookupError, match="No GuiEmitter is active"):
        _ = current_gui_emitter()


def test_sub_agent_span_nests_under_the_delegate_tool_span() -> None:
    """The specialist span has the delegate tool span as its parent."""
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
    """A run's generate spans have its agent span as parent."""
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
    # research_energy has two scripted turns.
    assert len(generate_spans) == 2


def test_a_call_that_runs_out_of_time_is_recorded_and_the_run_answers_anyway() -> None:
    """A timed-out call is recorded before the next turn."""
    app = _build_app("call_timeout")
    asyncio.run(app.run())
    climate = app.runs["root/research_climate"]
    assert _timed_out_count(app) == 1
    assert "root/research_climate" in app.answers
    assert climate.own_usage.cost_in_usd > 0


def test_the_app_deadline_leaves_every_settled_turn_readable_in_the_except() -> None:
    """The app deadline preserves settled turns and excludes in-flight calls."""
    app = _build_app("app_timeout")
    at_except: list[tuple[int, float]] = []

    async def drive() -> None:
        try:
            async with asyncio.timeout(0.5):
                await app.run()
        except TimeoutError:
            at_except.append((_timed_out_count(app), _total_cost(app)))

    asyncio.run(drive())
    # Climate spends $0.014, and energy spends $0.012.
    assert at_except == [(0, pytest.approx(0.026))]


def test_a_cost_limit_rejects_unknown_cost() -> None:
    """max_cost_in_usd rejects NaN cost."""
    unknown_cost = ZERO_USAGE.model_copy(update={"output_tokens_cost_in_usd": math.nan})
    with pytest.raises(RuntimeError, match="cost_in_usd is NaN"):
        _check_cost_limit(unknown_cost, 1.0)


def test_agent_config_rejects_a_nan_cost_limit() -> None:
    """max_cost_in_usd must be finite."""
    with pytest.raises(ValueError, match="positive and finite"):
        _ = AgentConfig(
            name="limited",
            system_prompt="[limited] answer",
            automatic_cache_breakpoints=False,
            max_cost_in_usd=math.nan,
        )


def test_tool_call_budget_uses_call_positions() -> None:
    """max_tool_calls dispatches the first position and declines the second."""
    app = _build_app("happy", climate_max_tool_calls=1)
    asyncio.run(app.run())
    climate = app.runs["root/research_climate"]
    assert isinstance(climate, ReActAgent)
    first_turn_tools = [
        record
        for record in climate.turn_log
        if isinstance(record, ToolTurn) and record.turn_number == 1
    ]
    assert [record.tool_message.is_error for record in first_turn_tools] == [False, True]
    assert climate.tool_calls_made == 1
    assert climate.bound.binding.tool_choice == "none"


def test_duplicate_tool_call_ids_are_rejected() -> None:
    """Positional budget handling rejects duplicate ToolCall.id values."""
    tool_calls = (
        ToolCall(id="duplicate", name="search", args_json='{"query":"one"}'),
        ToolCall(id="duplicate", name="search", args_json='{"query":"two"}'),
    )
    with pytest.raises(RuntimeError, match="appears twice"):
        _validate_tool_call_ids(tool_calls)


def test_delegate_propagates_a_tool_function_defect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delegate propagates DispatchExceptionGroup from a sub-agent tool."""

    class BrokenSearchArgs(BaseModel):
        """Define broken_search arguments."""

        query: str

    @tool(description="Raise a scripted defect.", name="search")
    async def broken_search(args: BrokenSearchArgs) -> str:
        """Raise a scripted defect.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError(f"search defect for {args.query}")

    monkeypatch.setattr(task_stream, "search_tool", broken_search)
    tracer = TracerProvider().get_tracer("full_app.test")
    registry: dict[str, AgentRun] = {}
    llm = TracedLLM(
        build_llm({"specialist": [Turn(tool_calls=(call("search", '{"query": "q"}'),))]}),
        capture_message_content=False,
        tracer=tracer,
    )
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        sub_config=AgentConfig(
            name="specialist",
            system_prompt="[specialist] answer",
            automatic_cache_breakpoints=False,
        ),
        tracer=tracer,
        registry=registry,
        on_event=_discard,
    )

    async def invoke() -> None:
        with pytest.raises(DispatchExceptionGroup):
            await delegate_tool.validate_and_run('{"question": "q"}')

    asyncio.run(invoke())


def test_settle_node_propagates_a_tool_function_defect() -> None:
    """_settle_node propagates DispatchExceptionGroup."""

    class DefectRun(AgentRun):
        """Raise a tool function defect from run."""

        @override
        async def run(self) -> str:
            """Raise one DispatchExceptionGroup.

            Raises:
                DispatchExceptionGroup: always.
            """
            raise DispatchExceptionGroup(
                "tool function failed",
                [RuntimeError("defect")],
                completed_outcomes=(),
            )

    app = _build_app("happy")
    run = DefectRun(
        agent_path="root/defect",
        config=AgentConfig(
            name="defect",
            system_prompt="[defect] fail",
            automatic_cache_breakpoints=False,
        ),
        tracer=TracerProvider().get_tracer("full_app.test"),
        registry={},
        on_event=_discard,
    )
    with pytest.raises(DispatchExceptionGroup):
        asyncio.run(app._settle_node(run))  # noqa: SLF001 (test the propagation boundary)


def test_a_run_cancelled_from_outside_emits_agent_cancelled() -> None:
    """An outside cancellation emits AgentCancelled before propagating."""
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
    assert isinstance(climate[-1], AgentCancelled)
    assert app.runs["root/research_climate"].usage.cost_in_usd > 0


def test_agent_cancelled_callback_failure_preserves_cancellation() -> None:
    """A callback defect cannot replace the outer TimeoutError."""

    def raise_on_cancelled(event: Event) -> None:
        if isinstance(event, AgentCancelled):
            raise TypeError("AgentCancelled callback failed")

    app = _build_app("app_timeout", on_event=raise_on_cancelled)

    async def drive() -> None:
        async with asyncio.timeout(0.5):
            await app.run()

    with pytest.raises(TimeoutError):
        asyncio.run(drive())


def test_a_failed_sub_agent_becomes_a_tool_message_and_the_parent_still_answers() -> None:
    """A failed sub-agent returns an error ToolMessage and preserves its spend."""
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
    # One turn and one search cost $0.012 before the failure.
    assert app.runs["root/research_climate/specialist#0"].usage.cost_in_usd == pytest.approx(0.012)
    assert "root/research_climate" in app.answers


def test_each_delegate_call_registers_a_fresh_spawn_indexed_run() -> None:
    """Each delegate call registers a distinct spawn-indexed run."""
    tracer = TracerProvider().get_tracer("full_app.test")
    registry: dict[str, AgentRun] = {}
    # The shared script provides one turn to each spawn.
    llm = TracedLLM(
        build_llm({"specialist": [Turn(text="first"), Turn(text="second")]}),
        capture_message_content=False,
        tracer=tracer,
    )
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        sub_config=AgentConfig(
            name="specialist",
            system_prompt="[specialist] answer",
            automatic_cache_breakpoints=False,
        ),
        tracer=tracer,
        registry=registry,
        on_event=_discard,
    )

    async def spawn_twice() -> None:
        await delegate_tool.validate_and_run('{"question": "q1"}')
        await delegate_tool.validate_and_run('{"question": "q2"}')

    asyncio.run(spawn_twice())
    assert set(registry) == {"root/parent/specialist#0", "root/parent/specialist#1"}


def test_a_second_run_under_one_agent_path_is_rejected() -> None:
    """Registering a duplicate agent_path raises ValueError."""

    class NoOpRun(AgentRun):
        """Provide a concrete AgentRun for registration."""

        @override
        async def run(self) -> str:
            """Return an unused constant."""
            return "unused"

    registry: dict[str, AgentRun] = {}
    config = AgentConfig(name="twin", system_prompt="[twin] p", automatic_cache_breakpoints=False)
    tracer = TracerProvider().get_tracer("full_app.test")
    _ = NoOpRun(
        agent_path="root/twin", config=config, tracer=tracer, registry=registry, on_event=_discard
    )
    with pytest.raises(ValueError, match="already registered"):
        _ = NoOpRun(
            agent_path="root/twin",
            config=config,
            tracer=tracer,
            registry=registry,
            on_event=_discard,
        )
