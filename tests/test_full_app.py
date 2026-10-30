import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import override

import pytest
from config import AgentConfig, build_configs
from events import Event, current_gui_emitter
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from task_stream import (
    AgentRun,
    App,
    GenerationEntry,
    ReActAgent,
    RunLimitError,
    _check_cost_limit,
    build_delegate_tool,
)
from tools import SearchArgs, search_tool

from langchaint import (
    ZERO_USAGE,
    DispatchExceptionGroup,
    GenerationError,
    ToolCall,
    UserMessage,
    tool,
)
from langchaint.tracing import OtelObserver
from tests.full_app_support.scenarios import build_scripts
from tests.full_app_support.scripted_adapter import Turn, build_llm, call
from tests.helpers import ValidatingSpanExporter, run_with_timeout


def _discard(event: Event) -> None:
    pass


def _timed_out_count(app: App) -> int:
    return sum(
        entry.kind == "generation"
        and isinstance(entry.outcome, GenerationError)
        and entry.outcome.kind == "timed_out_error"
        for run in app.runs.values()
        for entry in run.turn_log
    )


def _total_cost(app: App) -> float:
    return sum(run.own_usage.cost_in_usd for run in app.runs.values())


def _build_app(
    scenario: str,
    *,
    on_event: Callable[[Event], None] = _discard,
    exporter: InMemorySpanExporter | None = None,
    climate_max_tool_calls: int | None = None,
    second_requests_started: tuple[asyncio.Event, asyncio.Event] | None = None,
) -> App:
    """Build an app for one scenario under a local TracerProvider."""
    tracer_provider = TracerProvider()
    if exporter is not None:
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    configs = build_configs()
    if scenario == "input_timeout":
        configs["research_climate"] = replace(
            configs["research_climate"], generate_one_timeout_seconds=0.1
        )
    if scenario == "turn_limit":
        configs["research_climate"] = replace(configs["research_climate"], max_turns=1)
    if climate_max_tool_calls is not None:
        configs["research_climate"] = replace(
            configs["research_climate"], max_tool_calls=climate_max_tool_calls
        )
    scripts = build_scripts(scenario, configs)
    if second_requests_started is not None:
        climate_started, energy_started = second_requests_started
        scripts[configs["research_climate"].system_prompt][1].started = climate_started
        scripts[configs["research_energy"].system_prompt][1].started = energy_started
    observer = OtelObserver(capture_message_content=False, tracer_provider=tracer_provider)
    return App(llm=build_llm(scripts, observer=observer), configs=configs, on_event=on_event)


async def _expire_after_second_requests_start(
    app: App, second_requests_started: tuple[asyncio.Event, asyncio.Event]
) -> None:
    """Expire an asyncio deadline after both researchers enter their delayed second requests."""
    deadline = asyncio.timeout(5.0)
    app_task: asyncio.Task[None] | None = None
    try:
        async with deadline, asyncio.TaskGroup() as group:
            app_task = group.create_task(app.run())
            for started in second_requests_started:
                await started.wait()
            deadline.reschedule(asyncio.get_running_loop().time())
    except TimeoutError:
        assert deadline.expired()
        assert all(started.is_set() for started in second_requests_started)
        assert app_task is not None
        assert app_task.cancelled()
        raise


def _parent_of(span: ReadableSpan, spans: tuple[ReadableSpan, ...]) -> ReadableSpan | None:
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
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
    return matches[0]


def _attribute(span: ReadableSpan, key: str) -> object:
    assert span.attributes is not None
    return span.attributes.get(key)


def test_one_on_event_receives_the_whole_tree() -> None:
    events: list[Event] = []
    app = _build_app("happy", on_event=events.append)
    run_with_timeout(app.run())
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
    progress_paths = [event.agent_path for event in events if event.kind == "tool_progress"]
    assert "root/research_climate/specialist#0" in progress_paths
    assert "root/research_energy" in progress_paths


def test_application_span_groups_generation_and_tool_spans() -> None:
    exporter = ValidatingSpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer("full_app.test")
    app = _build_app("happy", exporter=exporter)
    with tracer.start_as_current_span("application"):
        run_with_timeout(app.run())

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


def test_a_timed_out_input_is_recorded_and_the_run_answers_anyway() -> None:
    app = _build_app("input_timeout")
    run_with_timeout(app.run())
    climate = app.runs["root/research_climate"]
    assert _timed_out_count(app) == 1
    assert "root/research_climate" in app.answers
    assert climate.own_usage.cost_in_usd > 0


def test_the_app_deadline_leaves_every_settled_entry_readable_in_the_except() -> None:
    """The app deadline preserves settled entries and excludes in-flight inputs."""
    events: list[Event] = []
    second_requests_started = (asyncio.Event(), asyncio.Event())
    app = _build_app(
        "app_timeout", on_event=events.append, second_requests_started=second_requests_started
    )
    at_except: list[tuple[int, float]] = []

    async def drive() -> None:
        try:
            await _expire_after_second_requests_start(app, second_requests_started)
        except TimeoutError:
            at_except.append((_timed_out_count(app), _total_cost(app)))

    run_with_timeout(drive())
    # Climate spends $0.014, and energy spends $0.012.
    assert at_except == [(0, pytest.approx(0.026))]
    cancelled = [event for event in events if event.kind == "agent_cancelled"]
    assert {event.agent_path for event in cancelled} == {
        "root/research_climate",
        "root/research_energy",
    }
    assert sorted(event.usage.cost_in_usd for event in cancelled) == pytest.approx([0.012, 0.014])


@pytest.mark.parametrize("cost_in_usd", [float("nan"), 1.0])
def test_a_cost_limit_stops_another_turn(cost_in_usd: float) -> None:
    usage = ZERO_USAGE.model_copy(update={"output_tokens_cost_in_usd": cost_in_usd})
    with pytest.raises(RunLimitError):
        _check_cost_limit(usage, 1.0)


def test_agent_config_rejects_a_nan_cost_limit() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        _ = AgentConfig(
            name="limited",
            system_prompt="Answer.",
            automatic_cache_breakpoints=False,
            max_cost_in_usd=float("nan"),
        )


def test_tool_call_budget_uses_call_positions() -> None:
    app = _build_app("happy", climate_max_tool_calls=1)
    run_with_timeout(app.run())
    climate = app.runs["root/research_climate"]
    assert isinstance(climate, ReActAgent)
    first_turn_dispatches = [
        entry for entry in climate.turn_log if entry.kind == "dispatch" and entry.turn_number == 1
    ]
    assert [entry.outcome.tool_message.is_error for entry in first_turn_dispatches] == [
        False,
        True,
    ]
    assert climate.tool_calls_made == 1
    assert climate.bound.binding.tool_choice == "none"


def test_duplicate_tool_call_ids_are_rejected() -> None:
    events: list[Event] = []
    tool_calls = (
        ToolCall(id="duplicate", name="search", args_json='{"query":"one"}'),
        ToolCall(id="duplicate", name="search", args_json='{"query":"two"}'),
    )
    config = AgentConfig(
        name="duplicate", system_prompt="Search.", automatic_cache_breakpoints=False
    )
    run = ReActAgent(
        agent_path="root/duplicate",
        config=config,
        llm=build_llm({config.system_prompt: [Turn(tool_calls=tool_calls)]}),
        tools=[search_tool],
        prompt="Search twice.",
        registry={},
        on_event=events.append,
    )
    with pytest.raises(RuntimeError, match="appears twice"):
        _ = run_with_timeout(run.run())
    assert not any(event.kind == "tool_progress" for event in events)
    assert run.tool_calls_made == 0


def test_delegate_propagates_a_tool_progress_callback_defect() -> None:
    def raise_on_progress(event: Event) -> None:
        if event.kind == "tool_progress":
            raise RuntimeError("search progress callback defect")

    registry: dict[str, AgentRun] = {}
    specialist_prompt = "Answer the question."
    llm = build_llm({specialist_prompt: [Turn(tool_calls=(call("search", '{"query": "q"}'),))]})
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        config=AgentConfig(
            name="specialist",
            system_prompt=specialist_prompt,
            automatic_cache_breakpoints=False,
        ),
        registry=registry,
        on_event=raise_on_progress,
    )

    async def invoke() -> None:
        with pytest.raises(DispatchExceptionGroup):
            await delegate_tool.validate_and_run('{"question": "q"}')

    run_with_timeout(invoke())


def test_settle_node_propagates_a_tool_function_defect() -> None:
    @tool(description="Raise a scripted defect.", name="broken")
    async def broken(args: SearchArgs) -> str:
        raise RuntimeError(f"search defect for {args.query}")

    app = _build_app("happy")
    config = AgentConfig(name="defect", system_prompt="Fail.", automatic_cache_breakpoints=False)
    run = ReActAgent(
        agent_path="root/defect",
        config=config,
        llm=build_llm({
            config.system_prompt: [
                Turn(tool_calls=(call("search", '{"query":"q"}'), call("broken", '{"query":"q"}')))
            ]
        }),
        tools=[search_tool, broken],
        prompt="Search.",
        registry={},
        on_event=_discard,
    )
    with pytest.raises(DispatchExceptionGroup) as raised:
        run_with_timeout(app._settle_node(run))
    dispatches = [entry for entry in run.turn_log if entry.kind == "dispatch"]
    assert len(dispatches) == 1
    assert dispatches[0].outcome is raised.value.completed_outcomes[0]
    assert run.own_usage.cost_in_usd == pytest.approx(0.012)


def test_agent_cancelled_callback_failure_preserves_cancellation() -> None:
    """A callback defect cannot replace the outer TimeoutError."""

    def raise_on_cancelled(event: Event) -> None:
        if event.kind == "agent_cancelled":
            raise TypeError("AgentCancelled callback failed")

    second_requests_started = (asyncio.Event(), asyncio.Event())
    app = _build_app(
        "app_timeout", on_event=raise_on_cancelled, second_requests_started=second_requests_started
    )

    with pytest.raises(TimeoutError):
        run_with_timeout(_expire_after_second_requests_start(app, second_requests_started))


def test_a_failed_sub_agent_becomes_a_tool_message_and_the_parent_still_answers() -> None:
    events: list[Event] = []
    app = _build_app("subagent_error", on_event=events.append)
    run_with_timeout(app.run())
    specialist_terminals = [
        event.kind
        for event in events
        if event.agent_path == "root/research_climate/specialist#0"
        and event.kind in ("agent_finished", "agent_failed")
    ]
    assert specialist_terminals == ["agent_failed"]
    # One generation and one search cost $0.012 before the failure.
    assert app.runs["root/research_climate/specialist#0"].usage.cost_in_usd == pytest.approx(0.012)
    assert "root/research_climate" in app.answers


def test_each_delegate_call_registers_a_fresh_spawn_indexed_run() -> None:
    registry: dict[str, AgentRun] = {}
    # The shared script provides one turn to each spawn.
    specialist_prompt = "Answer the question."
    llm = build_llm({specialist_prompt: [Turn(text="first"), Turn(text="second")]})
    delegate_tool = build_delegate_tool(
        llm=llm,
        parent_path="root/parent",
        config=AgentConfig(
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

    run_with_timeout(spawn_twice())
    assert set(registry) == {"root/parent/specialist#0", "root/parent/specialist#1"}


def test_a_second_run_under_one_agent_path_is_rejected() -> None:
    app = _build_app("happy")
    _ = app._build_run(name="research_climate", tools=[], prompt="Respond.")
    with pytest.raises(ValueError, match="already registered"):
        _ = app._build_run(name="research_climate", tools=[], prompt="Respond.")


def test_custom_agent_runs_prescribed_steps_with_shared_lifecycle_and_accounting() -> None:
    llm = build_llm({
        "Plan the work.": [Turn(text="Research climate and energy.")],
        "Write the answer.": [Turn(text="Climate and energy findings.")],
    })

    class StructuredAgent(AgentRun):
        @override
        async def _execute(self) -> str:
            plan = await llm.bind(system_prompt="Plan the work.").generate_one([
                UserMessage(content="Prepare a report.")
            ])
            self.turn_log.append(GenerationEntry(turn_number=1, outcome=plan))
            current_gui_emitter().emit_tool_progress(tool_name="plan", message=plan.output)
            answer = await llm.bind(system_prompt="Write the answer.").generate_one([
                UserMessage(content=plan.output)
            ])
            self.turn_log.append(GenerationEntry(turn_number=2, outcome=answer))
            return answer.output

    events: list[Event] = []
    registry: dict[str, AgentRun] = {}
    run = StructuredAgent(agent_path="root/structured", registry=registry, on_event=events.append)
    app = _build_app("happy")
    run_with_timeout(app._settle_node(run))

    assert registry[run.agent_path] is run
    assert app.answers[run.agent_path] == "Climate and energy findings."
    assert run.own_usage.cost_in_usd == pytest.approx(0.02)
    assert [event.kind for event in events] == [
        "agent_started",
        "tool_progress",
        "agent_finished",
    ]
    assert all(event.agent_path == run.agent_path for event in events)
    finished = events[-1]
    assert finished.kind == "agent_finished"
    assert finished.answer == app.answers[run.agent_path]
    assert finished.usage == run.usage
    with pytest.raises(LookupError):
        _ = current_gui_emitter()


def test_a_dispatch_callback_defect_preserves_complete_accounting() -> None:
    def fail(event: Event) -> None:
        if (
            event.kind != "tool_dispatched"
            or event.agent_path != "root/research_climate"
            or event.turn_number != 1
        ):
            return
        assert app.runs[event.agent_path].own_usage.cost_in_usd == pytest.approx(0.014)
        assert event.usage_so_far.cost_in_usd == pytest.approx(0.012)
        raise ValueError("dispatch callback defect")

    app = _build_app("happy", on_event=fail)
    with pytest.raises(ExceptionGroup) as raised:
        run_with_timeout(app.run())
    assert any(isinstance(error, ValueError) for error in raised.value.exceptions)
    dispatches = [
        entry
        for entry in app.runs["root/research_climate"].turn_log
        if entry.kind == "dispatch" and entry.turn_number == 1
    ]
    assert len(dispatches) == 2
    assert all(entry.outcome.kind == "handled" for entry in dispatches)


@pytest.mark.parametrize("scenario", ["input_defect", "subagent_defect"])
def test_unknown_generation_failures_propagate(scenario: str) -> None:
    app = _build_app(scenario)
    with pytest.raises(ExceptionGroup) as raised:
        run_with_timeout(app.run())
    assert (
        raised.value.subgroup(
            lambda error: (
                isinstance(error, GenerationError) and error.kind == "unknown_exception_error"
            )
        )
        is not None
    )


def test_app_recovers_from_a_researcher_limit() -> None:
    app = _build_app("turn_limit")
    run_with_timeout(app.run())
    assert app.failures["root/research_climate"].startswith("RunLimitError:")
    assert app.runs["root/research_climate"].usage.cost_in_usd == pytest.approx(0.014)
    assert app.final_answer is not None


def test_synthesis_finishes_after_scripted_critique_approval() -> None:
    events: list[Event] = []
    app = _build_app("unapproved_answer", on_event=events.append)
    run_with_timeout(app.run())
    answers = [
        event.answer
        for event in events
        if event.agent_path == "root/synthesize" and event.kind == "agent_finished"
    ]
    assert answers == ["Synthesis: climate and energy findings reconciled."]
    assert app.final_answer == answers[0]
    synthesize = app.runs["root/synthesize"]
    assert isinstance(synthesize, ReActAgent)
    assert synthesize.turn_number == 5
