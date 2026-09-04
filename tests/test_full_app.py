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
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
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
from tests.full_app_support.scenarios import build_scripts
from tests.full_app_support.scripted_adapter import Turn, build_llm, call


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
    second_calls_started: tuple[asyncio.Event, asyncio.Event] | None = None,
) -> App:
    """Build an app for one scenario under a local TracerProvider."""
    tracer_provider = TracerProvider()
    if exporter is not None:
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    configs = build_configs()
    if scenario == "call_timeout":
        configs["research_climate"] = replace(
            configs["research_climate"], generate_one_timeout_seconds=0.1
        )
    if climate_max_tool_calls is not None:
        configs["research_climate"] = replace(
            configs["research_climate"], max_tool_calls=climate_max_tool_calls
        )
    scripts = build_scripts(scenario, configs)
    if second_calls_started is not None:
        climate_started, energy_started = second_calls_started
        scripts[configs["research_climate"].system_prompt][1].started = climate_started
        scripts[configs["research_energy"].system_prompt][1].started = energy_started
    return App(
        llm=build_llm(scripts),
        configs=configs,
        tracer=tracer_provider.get_tracer("full_app.test"),
        on_event=on_event,
        capture_message_content=False,
    )


async def _expire_after_second_calls_start(
    app: App, second_calls_started: tuple[asyncio.Event, asyncio.Event]
) -> None:
    """Expire an asyncio deadline after both researchers enter their delayed second calls."""
    deadline = asyncio.timeout(5.0)
    app_task: asyncio.Task[None] | None = None
    try:
        async with deadline, asyncio.TaskGroup() as group:
            app_task = group.create_task(app.run())
            for started in second_calls_started:
                await started.wait()
            deadline.reschedule(asyncio.get_running_loop().time())
    except TimeoutError:
        assert deadline.expired()
        assert all(started.is_set() for started in second_calls_started)
        assert app_task is not None
        assert app_task.cancelled()
        raise


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


def test_application_span_groups_generation_and_tool_spans() -> None:
    """Run the application under an OpenTelemetry span and inspect its children."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer("full_app.test")
    app = _build_app("happy", exporter=exporter)
    with tracer.start_as_current_span("application"):
        asyncio.run(app.run())

    spans = exporter.get_finished_spans()
    application_span = _named(spans, "application")
    delegate_span = _named(spans, "execute_tool delegate")
    assert _parent_of(delegate_span, spans) is application_span
    for parent_span in (application_span, delegate_span):
        children = [span for span in spans if _parent_of(span, spans) is parent_span]
        assert {_attribute(span, "gen_ai.operation.name") for span in children} == {
            "chat",
            "execute_tool",
        }
    assert all(
        _parent_of(span, spans) in (application_span, delegate_span)
        for span in spans
        if span is not application_span
    )


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
    second_calls_started = (asyncio.Event(), asyncio.Event())
    app = _build_app("app_timeout", second_calls_started=second_calls_started)
    at_except: list[tuple[int, float]] = []

    async def drive() -> None:
        try:
            await _expire_after_second_calls_start(app, second_calls_started)
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
            system_prompt="Answer.",
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
    specialist_prompt = "Answer the question."
    llm = TracedLLM(
        build_llm({specialist_prompt: [Turn(tool_calls=(call("search", '{"query": "q"}'),))]}),
        capture_message_content=False,
        tracer=tracer,
    )
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        sub_config=AgentConfig(
            name="specialist",
            system_prompt=specialist_prompt,
            automatic_cache_breakpoints=False,
        ),
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
            system_prompt="Fail.",
            automatic_cache_breakpoints=False,
        ),
        registry={},
        on_event=_discard,
    )
    with pytest.raises(DispatchExceptionGroup):
        asyncio.run(app._settle_node(run))


def test_a_run_cancelled_from_outside_emits_agent_cancelled() -> None:
    """An outside cancellation emits AgentCancelled before propagating."""
    events_by_path: dict[str, list[Event]] = {}

    def collect(event: Event) -> None:
        events_by_path.setdefault(event.agent_path, []).append(event)

    second_calls_started = (asyncio.Event(), asyncio.Event())
    app = _build_app("app_timeout", on_event=collect, second_calls_started=second_calls_started)

    async def drive() -> None:
        try:
            await _expire_after_second_calls_start(app, second_calls_started)
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

    second_calls_started = (asyncio.Event(), asyncio.Event())
    app = _build_app(
        "app_timeout", on_event=raise_on_cancelled, second_calls_started=second_calls_started
    )

    with pytest.raises(TimeoutError):
        asyncio.run(_expire_after_second_calls_start(app, second_calls_started))


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
    specialist_prompt = "Answer the question."
    llm = TracedLLM(
        build_llm({specialist_prompt: [Turn(text="first"), Turn(text="second")]}),
        capture_message_content=False,
        tracer=tracer,
    )
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        sub_config=AgentConfig(
            name="specialist",
            system_prompt=specialist_prompt,
            automatic_cache_breakpoints=False,
        ),
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
    config = AgentConfig(name="twin", system_prompt="Respond.", automatic_cache_breakpoints=False)
    _ = NoOpRun(agent_path="root/twin", config=config, registry=registry, on_event=_discard)
    with pytest.raises(ValueError, match="already registered"):
        _ = NoOpRun(
            agent_path="root/twin",
            config=config,
            registry=registry,
            on_event=_discard,
        )
