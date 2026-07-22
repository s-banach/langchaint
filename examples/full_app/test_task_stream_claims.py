"""Assert the claims task_stream.py's docstrings make, rather than trusting them.

Two of these are the reason the module has its shape.

The first is the guardrail: an application that writes run() as a generator is rejected at
construction. That is the whole encapsulation argument in one assertion, and if it ever stops holding,
the design has lost its point rather than merely a feature.

The second is the consumer-latency property: a consumer slower than the run's own deadline does not
kill the run, because the loop runs in its own task and the deadline measures the loop's work, not the
consumer's pulls. A design that drives the loop from the consumer's pulls cannot have this property,
which is why run() is a coroutine and not a generator.

Async tests run through asyncio.run in a sync test function, the convention the rest of this repo's
suite uses, so no pytest-asyncio plugin is needed.
"""

import asyncio

import pytest
from config import AgentConfig, build_configs
from events import AgentFailed, AgentFinished, ToolProgress, TurnStarted, current_gui_emitter
from harness import Turn, build_llm, call
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from scenario import build_scripts, reset_critique, search_tool
from task_stream import AgentRun, App, ReActAgent

from langchaint.tracing import TracedLLM, TracedToolManager


def _build_app(scenario: str, *, exporter: InMemorySpanExporter | None = None) -> App:
    """Build an app for one scenario under a local TracerProvider."""
    reset_critique()
    tracer_provider = TracerProvider()
    if exporter is not None:
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return App(
        llm=build_llm(build_scripts(scenario)),
        configs=build_configs(),
        tracer=tracer_provider.get_tracer("full_app.test"),
    )


def _drive(app: App) -> None:
    """Consume an app's whole event stream so every run settles."""

    async def run() -> None:
        async for _ in app:
            pass

    asyncio.run(run())


def _parent_of(span: ReadableSpan, spans: tuple[ReadableSpan, ...]) -> ReadableSpan | None:
    """Return the span that is the given span's parent, or None when it is a root."""
    if span.parent is None:
        return None
    by_id = {other.context.span_id: other for other in spans if other.context is not None}
    return by_id.get(span.parent.span_id)


def _named(spans: tuple[ReadableSpan, ...], name: str) -> ReadableSpan:
    """Return the one finished span with the given name, asserting it is unique."""
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
    return matches[0]


def _attribute(span: ReadableSpan, key: str) -> object:
    """Read one attribute off a finished span, asserting the span carries attributes at all."""
    assert span.attributes is not None
    return span.attributes.get(key)


def test_a_generator_run_is_rejected_at_construction() -> None:
    """Writing run() as an async generator fails immediately, with a message naming the alternative.

    This is the encapsulation: the hazard is not merely absent from application code, it is unwritable.
    Without this check the same mistake surfaces as "'async_generator' object can't be awaited" on the
    first pull, which does not tell the author what to do instead.
    """

    class GeneratorRun(AgentRun):
        """An agent whose author reached for a generator, the shape this class exists to reject.

        Every suppression below is the point of the test: a type checker rejects this class too, and the
        runtime check exists for the authors who do not run one, or who reach for `# type: ignore` to
        quiet it. The check must fire on the class as written here, suppressions and all.
        """

        # pyrefly: ignore[bad-override]
        # pyrefly: ignore[missing-override-decorator]
        # pyrefly: ignore[bad-return]
        async def run(self) -> str:
            """Yield instead of emitting, which is the mistake under test.

            Yields:
                Nothing reachable: constructing this class is what fails, so the body never runs.
            """
            yield "an event"

    with pytest.raises(TypeError, match="must be a coroutine"):
        GeneratorRun(
            agent_path="root/wrong",
            config=AgentConfig(name="wrong", system_prompt="[wrong] p"),
            tracer=TracerProvider().get_tracer("full_app.test"),
            registry={},
        )


SLOW_CONSUMER_PAUSE_SECONDS = 0.4
"""What the consumer sleeps per event, against the 1.0s run deadline the setup below configures."""


def _slow_consumer_setup() -> ReActAgent:
    """Build one agent whose run outlives a consumer slower than its deadline.

    The script is a search turn then an answer, so the loop's own work is milliseconds against a 1.0s
    deadline, while the consumer sleeps 0.4s per event and takes several seconds to drain the run. A
    run whose deadline measured the consumer's pulls would time out; this one finishes, and the margin
    is seconds against milliseconds, so neither outcome is a race.
    """
    reset_critique()
    tracer = TracerProvider().get_tracer("full_app.test")
    scripts = {
        "slow": [
            Turn(tool_calls=(call("search", '{"query": "q"}'),)),
            Turn(text="done"),
        ]
    }
    config = AgentConfig(
        name="slow",
        system_prompt="[slow] answer",
        max_turns=4,
        timeout_seconds=1.0,
        per_call_timeout_seconds=1.0,
    )
    llm = TracedLLM(build_llm(scripts), capture_message_content=False, tracer=tracer)
    tool_manager = TracedToolManager([search_tool], capture_message_content=False, tracer=tracer)
    return ReActAgent(
        agent_path="root/slow",
        config=config,
        tracer=tracer,
        registry={},
        bound=llm.bind(
            system_prompt=config.system_prompt,
            tool_manager=tool_manager,
            automatic_prompt_caching=True,
        ),
        tool_manager=tool_manager,
        prompt="go",
    )


def test_a_consumer_slower_than_the_deadline_does_not_kill_the_run() -> None:
    """The loop runs in its own task, so it finishes long before the slow consumer has drained it."""
    agent = _slow_consumer_setup()

    async def drive_slowly() -> None:
        async for _ in agent:
            await asyncio.sleep(SLOW_CONSUMER_PAUSE_SECONDS)

    asyncio.run(drive_slowly())
    assert agent.answer == "done"
    assert agent.failure is None


def test_the_stream_carries_the_sub_agents_events() -> None:
    """The specialist's events reach the top-level consumer, forwarded through its parent's stream."""
    app = _build_app("happy")
    paths: set[str] = set()

    async def drive() -> None:
        async for event in app:
            paths.add(event.agent_path)

    asyncio.run(drive())
    assert paths == {
        "root/research_climate",
        "root/research_climate/specialist",
        "root/research_energy",
        "root/synthesize",
    }
    specialist = app.runs["root/research_climate/specialist"]
    assert specialist.answer is not None
    assert specialist.usage.cost_in_usd > 0


def test_a_tools_progress_lands_in_the_stream_of_the_run_that_dispatched_it() -> None:
    """ToolProgress from search carries the dispatching run's path, the specialist's included.

    One search function, no emitter parameter: each run's task installed its own GuiEmitter, so the
    same function reports to whichever run called it, and the specialist's progress is stamped with the
    specialist's path rather than its parent's.
    """
    app = _build_app("happy")
    progress_paths: list[str] = []

    async def drive() -> None:
        progress_paths.extend([
            event.agent_path async for event in app if isinstance(event, ToolProgress)
        ])

    asyncio.run(drive())
    assert "root/research_climate/specialist" in progress_paths
    assert "root/research_energy" in progress_paths


def test_reading_the_emitter_outside_a_run_raises_naming_what_to_install() -> None:
    """current_gui_emitter outside a run's loop raises a LookupError that names the fix.

    The compile-time guarantee threading gives is lost on an ambient value, so the accessor owes the
    reader a message better than the default "no value", which names only a variable.
    """
    with pytest.raises(LookupError, match="inside a run's loop"):
        current_gui_emitter()


def test_final_resumes_a_partially_iterated_run_rather_than_restarting_it() -> None:
    """Reading some events and then awaiting final() drives one run, not two."""
    agent = _slow_consumer_setup()

    async def drive() -> str:
        """Take two events, abandon the loop, then finish the same run through final()."""
        seen = 0
        async for _ in agent:
            seen += 1
            if seen == 2:
                break
        return await agent.final()

    assert asyncio.run(drive()) == "done"
    # Two generate turns at $0.01 and one search at $0.002: the script played once, not twice.
    assert agent.usage.cost_in_usd == pytest.approx(0.022)


def test_sub_agent_span_nests_under_the_delegate_tool_span() -> None:
    """The specialist's span parents to the delegate execute_tool span with no context plumbing.

    _drain creates the run's task from the frame delegate iterates in, and create_task copies the
    context active at creation, which is the one the tool span is current in. A task boundary would
    break this if it were created anywhere else, so it is asserted rather than assumed.
    """
    exporter = InMemorySpanExporter()
    app = _build_app("happy", exporter=exporter)
    _drive(app)
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
    _drive(app)
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


def test_no_agent_span_is_current_in_the_consumer() -> None:
    """A consumer receiving an event has no agent span current.

    The span is entered in the run's own task, so it is not in the consumer's context to leak; a shape
    that entered it around a generator's yield would leave it current in the consumer instead.
    """
    app = _build_app("happy")

    async def drive() -> None:
        events = 0
        async for _ in app:
            events += 1
            assert not trace.get_current_span().get_span_context().is_valid
        assert events > 0

    asyncio.run(drive())


def test_the_per_agent_deadline_fires_on_cumulative_time() -> None:
    """An ordinary enclosing timeout cuts a run off on cumulative time, no fixed-instant arithmetic."""
    app = _build_app("agent_timeout")
    _drive(app)
    energy = app.runs["root/research_energy"]
    assert isinstance(energy, ReActAgent)
    assert energy.answer is None
    assert energy.usage.cost_in_usd > 0
    assert app.abandoned_calls == 1
    # More than one call completed before the cut-off, which is the whole difference from the per-call
    # deadline: that one would have ended the run on its first delayed call, at turn 1.
    assert energy.turn_number >= 2


def test_a_whole_app_deadline_counts_the_calls_it_cancelled() -> None:
    """Both researchers are mid-request when the app deadline fires, and both cancellations are counted.

    settle() asks each run to unwind rather than tracking pumps, because each run owns its own task.
    """
    app = _build_app("app_timeout")
    before_settle: list[int] = []

    async def drive() -> None:
        try:
            async with asyncio.timeout(0.5):
                async for _ in app:
                    pass
        except TimeoutError:
            before_settle.append(app.abandoned_calls)
            await app.settle()

    asyncio.run(drive())
    # Read before settle(), the count is short: the cancelled calls have not reached their handler yet.
    # Without this a no-op settle() would pass the assertion below on a tree that never unwound.
    assert before_settle == [0]
    assert app.abandoned_calls == 2
    assert app.total_usage.cost_in_usd > 0


def _pending_run_paths(app: App) -> set[str]:
    """Return the paths of registered runs whose tasks have not finished unwinding."""
    tasks = {path: run._task for path, run in app.runs.items()}  # noqa: SLF001  # done-ness of the run's task is the claim under test
    return {path for path, task in tasks.items() if task is not None and not task.done()}


def test_a_client_disconnect_settles_through_the_registry_not_the_generator_chain() -> None:
    """A consumer that aclose()s the stream mid-run leaves the runs to one settle() in its finally.

    The offline model of a server stream handler whose client disconnected. aclose() throws
    GeneratorExit at App.__aiter__'s suspended yield and closes nothing beneath it: _fan_in is left
    to the event loop's async-generator finalizer, which has not run by the reads below, and the run
    tasks keep running, held by the registry. So right after aclose() both researchers' tasks are
    still pending and the in-flight calls' AbandonedCall rows are missing; settle(), reaching every
    run through the registry, is what unwinds them and makes the accounting final.
    """
    app = _build_app("app_timeout")
    pending_before: list[set[str]] = []
    abandoned_before: list[int] = []

    async def drive() -> None:
        """Consume until both researchers are inside their stalled second call, then disconnect."""
        events = aiter(app)
        second_turns_started = 0
        try:
            async for event in events:
                if isinstance(event, TurnStarted) and event.turn_number == 2:
                    second_turns_started += 1
                    if second_turns_started == 2:
                        break
        finally:
            await events.aclose()
            pending_before.append(_pending_run_paths(app))
            abandoned_before.append(app.abandoned_calls)
            await app.settle()

    asyncio.run(drive())
    assert pending_before == [{"root/research_climate", "root/research_energy"}]
    assert abandoned_before == [0]
    assert _pending_run_paths(app) == set()
    assert app.abandoned_calls == 2
    # Climate's first turn ($0.01 plus two searches at $0.002) and energy's ($0.01 plus one search).
    # The two abandoned calls had no settled attempt, and no usage is fabricated for an in-flight
    # attempt, so they add nothing.
    assert app.total_usage.cost_in_usd == pytest.approx(0.026)


def test_a_run_cancelled_from_outside_ends_its_stream_without_a_terminal_event() -> None:
    """An outside cancellation closes the queue, so iteration ends with no AgentFinished or AgentFailed.

    _driven lets the CancelledError propagate rather than recording it, so the run has no outcome to
    report. The queue still closes, which is what keeps a consumer from waiting on a stopped run. A UI
    that treats a terminal event as guaranteed would hang here, which is why _drain says so.
    """
    app = _build_app("app_timeout")
    events_by_path: dict[str, list[object]] = {}

    async def drive() -> None:
        try:
            async with asyncio.timeout(0.5):
                async for event in app:
                    events_by_path.setdefault(event.agent_path, []).append(event)
        except TimeoutError:
            await app.settle()

    asyncio.run(drive())
    climate = events_by_path["root/research_climate"]
    assert climate, "the cancelled run emitted events before the deadline"
    assert not isinstance(climate[-1], AgentFinished | AgentFailed)
    # The accounting survives the missing terminal event, which is the property that matters.
    assert app.runs["root/research_climate"].usage.cost_in_usd > 0


def test_a_failed_sub_agent_becomes_a_tool_message_and_the_parent_still_answers() -> None:
    """A sub-agent failure is data for the parent model, and its spend stands."""
    app = _build_app("subagent_error")
    _drive(app)
    specialist = app.runs["root/research_climate/specialist"]
    assert specialist.failure is not None
    assert specialist.answer is None
    # One turn at $0.01 and one search at $0.002 before the second turn raised; both rows stand.
    assert specialist.usage.cost_in_usd == pytest.approx(0.012)
    assert app.runs["root/research_climate"].answer is not None
