"""The recommended shape for a streaming multi-agent app on langchaint.

An agent is an async iterator, and the loop that fills it is a coroutine in its own task.

AgentRun splits the two roles. The base class owns the streaming: the queue, the iterator, final(), the
run deadline, and the agent span. The application subclasses it and writes only run(), a plain
coroutine, calling self.emit(event) as it goes. The one generator in the design is _drain, which holds
nothing across its yield and contains no application code.

The split is load-bearing, not taste. The shape an application reaches for first makes the agent itself
an async generator, yielding events from inside the loop. An async generator has no context of its own;
it runs in whichever context its consumer resumes it from, so an asyncio.timeout or a
start_as_current_span entered before a yield and exited after it is open while the consumer runs: the
deadline measures the consumer's pace and cancels into the consumer's frame, and the span is left
current in the consumer, adopting whatever it opens next. Every one of those failures is silent and
needs a slow or abandoned consumer to appear, so it never shows up in development. ruff's ASYNC119
flags the construct, but lint is a prohibition, not a design: it does not say where the deadline goes
instead. Here the application cannot write the hazard, because it never writes a generator: a yield in
run() makes it an async generator, which AgentRun.__init__ rejects at construction with a message
naming the rule and the alternative. The deadline and the span are ordinary enclosing scopes in
_driven, a coroutine with no yield for them to span.

What the task costs: the loop runs ahead of the consumer and its events buffer in the queue. The queue
is unbounded, which is less open-ended than it sounds: one run emits at most a few events per turn and
per tool call, so config.max_turns and config.max_tool_calls already bound it. It becomes a memory
question only when events carry large payloads, which forwarding a stream's text would do. A bounded
queue would give real backpressure at the price of putting the consumer's pace back inside the run's
deadline; take that trade when the consumer is itself the work (writing rows, feeding a socket that
cannot buffer) rather than a reader. The other cost is that a run reports itself finished when its
loop stops, not when the consumer has drained it, which is the right instant for a budget on work and
the wrong one for an end-to-end latency promise.

Two capabilities ride on contextvars, the same mechanism that nests OTel spans across tasks:

    Each run's task installs its own GuiEmitter, so a tool function reports progress into the stream
    of whichever run dispatched it (search does, through events.current_gui_emitter). Threading cannot
    do this without a parameter on every tool, because a tool function holds no handle on its run.

    delegate forwards its sub-run's events to that same emitter, so a three-level tree streams to one
    consumer in real time while delegate stays an ordinary langchaint tool returning a value.

Only the emitter is ambient. The run registry stays a constructor argument, where a missing value is
a TypeError at the call site instead of a LookupError at the first read inside an agent; the emitter
is ambient because for it there is no call site to thread through.

The state a failure must leave behind lives on this object, never in the coroutine frame doing the
work: a timeout cancels the frame, and anything local to it is gone. Spend follows the same rule:
each row goes to the run's own spend_log at the moment it happens (langchaint appends the
AbandonedCall a cancellation would otherwise erase), every run registers itself at construction, and
a parent's total is a fold over the registered runs' logs by path prefix. A CancelledError is a
BaseException that passes through every `except Exception` in the tree, so a design that carries a
sub-run's spend home on a return path loses it when that path never runs; rows already written need
no carrying, so nothing can be dropped.
"""

import asyncio
import contextlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import override

from config import AgentConfig
from events import (
    AgentFailed,
    AgentFinished,
    AgentStarted,
    Event,
    GuiEmitter,
    LlmCallAbandoned,
    LlmResponse,
    ToolCalled,
    ToolResponse,
    TurnStarted,
    content_text,
    current_gui_emitter,
    describe_error,
    gui_emitter_var,
)
from opentelemetry.trace import Tracer
from pydantic import BaseModel
from scenario import DelegateArgs, critique_tool, search_tool

from langchaint import (
    LLM,
    ZERO_USAGE,
    AbandonedCall,
    DispatchExceptionGroup,
    DispatchHandled,
    DispatchOutcome,
    GenerationError,
    Message,
    PydanticTool,
    Tool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    Usage,
    UserMessage,
)
from langchaint.tracing import TracedBoundLLM, TracedLLM, TracedToolManager, agent_span


class AgentRun(ABC):
    """The streaming half of an agent: the queue, the iterator, the deadline and the span.

    A subclass supplies run(), its loop, as a coroutine, and calls emit() as it goes. Everything a UI
    needs to consume that run is here and is not the subclass's to get right.

    Iterating yields the run's events; awaiting final() drives the same run and discards them. Both go
    through one memoized generator, so mixing them resumes rather than restarts.

    AgentStarted, AgentFinished and AgentFailed are emitted here, because this class is what knows how
    the run ended. A subclass emits only what happens inside its loop.

    spend_log is the run's append-only accounting log: the loop appends each settled call's and each
    tool's reported Usage, and langchaint appends an AbandonedCall where a cancellation cut a generate
    call off. Totals are folds over the logs, never a running sum, so there is no second copy of any
    row to drift from.
    """

    def __init__(
        self,
        *,
        agent_path: str,
        config: AgentConfig,
        tracer: Tracer,
        registry: dict[str, "AgentRun"],
    ) -> None:
        """Fix the run's identity and limits; a subclass adds whatever its loop needs.

        The run registers itself in registry here, at construction rather than at start, so a run
        cancelled mid-flight is still in every report folded over the registry.

        Raises:
            TypeError: the subclass wrote run() as an async generator. Awaiting it would fail anyway,
                with asyncio's message rather than one naming the rule, so it is caught here at
                construction where the author is looking at their own class.
        """
        if inspect.isasyncgenfunction(type(self).run):
            raise TypeError(
                f"{type(self).__name__}.run is an async generator; it must be a coroutine. "
                "Send events with self.emit(event) instead of yielding them: a generator hands "
                "control to the consumer with its context managers still open, which is what this "
                "class exists to keep out of application code."
            )
        self.agent_path = agent_path
        self.config = config
        self.tracer = tracer
        self.registry = registry
        self.spend_log: list[Usage | AbandonedCall] = []
        self.answer: str | None = None
        self.failure: Exception | None = None
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._events: AsyncGenerator[Event] | None = None
        self._task: asyncio.Task[None] | None = None
        registry[agent_path] = self

    @abstractmethod
    async def run(self) -> str:
        """Drive this agent's loop to its final answer, calling emit() as it goes.

        A coroutine, never a generator: the base class awaits it under the run's deadline and span, and
        a `yield` here would make it an async generator that cannot be awaited at all.

        Returns:
            The run's final answer. Raising instead is how a run fails; the base class records it.
        """

    @property
    def own_usage(self) -> Usage:
        """The run's own spend, folded from spend_log; a sub-run's rows live on its own log."""
        return Usage.sum_of(
            row.usage_settled if isinstance(row, AbandonedCall) else row for row in self.spend_log
        )

    @property
    def usage(self) -> Usage:
        """The run's own spend plus every sub-run beneath it, folded from the registry by path prefix.

        The prefix fold is what hands a parent its subtree's total without any run having carried a
        number home, and it counts each row exactly once because each row lives on exactly one log.
        """
        return Usage.sum_of(
            run.own_usage
            for path, run in self.registry.items()
            if path == self.agent_path or path.startswith(f"{self.agent_path}/")
        )

    def emit(self, event: Event) -> None:
        """Push one event to whoever is consuming this run.

        Never suspends and never raises, because the queue is unbounded.
        Any frame inside run() can call it.
        """
        self._queue.put_nowait(event)

    def span_attributes(self) -> Mapping[str, str | int | float | bool]:
        """Extra attributes to set on the agent span when the run ends; override to add.

        Called after the loop leaves, so a subclass reports final counters here rather than tracking a
        span it does not own.
        """
        return {}

    def __aiter__(self) -> AsyncIterator[Event]:
        """Yield this run's events, starting the run on the first pull.

        The generator is memoized, so a consumer that stops iterating and then awaits final() resumes
        the same run instead of starting a second one.
        """
        if self._events is None:
            self._events = self._drain()
        return self._events

    async def final(self) -> str:
        """Drive the run to its answer, discarding its events.

        Called after the iteration has already been drained, it reads the recorded outcome without
        driving anything, which is how delegate settles a sub-run it just forwarded.

        Raises:
            Exception: whatever run() failed with, held since _driven caught it.
            RuntimeError: the run neither answered nor failed, which means it was cancelled from
                outside; _driven leaves a CancelledError to propagate rather than recording it.
            asyncio.CancelledError: an outer deadline cancelled the run while this await drove it.
        """
        async for _ in self:
            pass
        if self.failure is not None:
            raise self.failure
        if self.answer is None:
            raise RuntimeError(f"{self.agent_path} ended with neither an answer nor a failure")
        return self.answer

    async def settle(self) -> None:
        """Cancel the loop and wait for it to unwind, so the accounting is final before it is read.

        A consumer that abandons the iteration leaves the task still running, or cancelled but not
        unwound, so generate_one has not yet appended the cancelled call's AbandonedCall. Calling
        this after a completed run does nothing.
        """
        task = self._task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _drain(self) -> AsyncGenerator[Event]:
        """Start the run and yield its events until it signals the end.

        The one generator in this design. It holds no context manager across its yield, which is what
        the whole split exists to guarantee: the deadline, the span and the emitter installation are
        all in _driven, a coroutine.

        The task is created from this frame, and asyncio.create_task copies the context active at
        creation, so the run's span parents to whatever span the consumer had current at the pull that
        started it. For a sub-agent that is the parent's tool span, which is the nesting delegate
        wants. For a top-level run it is whatever the consumer was inside at its first pull, so two
        runs started at different points of one iteration can land under different parents.

        Yields:
            Each emitted event in emission order. A run that reached its own end yields AgentFinished
            or AgentFailed last; one cancelled from outside yields neither, because _driven lets the
            CancelledError propagate. The queue is closed on every path, so iteration always ends.
        """
        self._task = asyncio.create_task(self._driven())
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                yield item
        finally:
            self._task.cancel()

    async def _driven(self) -> None:
        """Run the loop under its deadline, its span and its own emitter, recording how it ended.

        A coroutine, so `asyncio.timeout` and `agent_span` are written the obvious way and are
        correct: there is no yield in this frame for either to span, and a cancellation from the
        deadline lands in this task rather than in whatever the consumer was doing.

        The emitter is installed here rather than in __init__ because this is the run's own task: a
        contextvar set here is visible to everything the loop reaches, tool functions the dispatch runs
        included, and invisible outside. A sub-run's task installs its own, so nesting needs no
        bookkeeping and there is no reset to forget on the way out.

        A CancelledError is left to propagate: an outer deadline that raised it is cancelling the
        consumer too, and the spend logs already hold the rows. The queue is closed either way, so a
        consumer still iterating is never left waiting on a run that has stopped.
        """
        gui_emitter_var.set(GuiEmitter(self.agent_path, self._queue))
        try:
            self.emit(AgentStarted(agent_path=self.agent_path))
            with agent_span(
                self.tracer,
                agent_name=self.config.name,
                agent_path=self.agent_path,
                usage=lambda: self.usage,
                extra_attributes=self.span_attributes,
            ):
                async with asyncio.timeout(self.config.timeout_seconds):
                    self.answer = await self.run()
            self.emit(
                AgentFinished(agent_path=self.agent_path, answer=self.answer, usage=self.usage)
            )
        except Exception as error:
            self.failure = error
            self.emit(
                AgentFailed(
                    agent_path=self.agent_path, error=describe_error(error), usage=self.usage
                )
            )
        finally:
            self._queue.put_nowait(None)


class ReActAgent(AgentRun):
    """The example's loop, written as the application's half of the contract: one coroutine.

    Everything here is ordinary async code. The per-call deadline is an ordinary `asyncio.timeout`, the
    events go out through emit(), and the run's own deadline and span are the base class's business.
    """

    def __init__(
        self,
        *,
        agent_path: str,
        config: AgentConfig,
        tracer: Tracer,
        registry: dict[str, AgentRun],
        bound: TracedBoundLLM[str],
        tool_manager: ToolManager,
        prompt: str,
    ) -> None:
        """Add what this loop needs on top of what every run needs."""
        super().__init__(agent_path=agent_path, config=config, tracer=tracer, registry=registry)
        self.bound = bound
        self.tool_manager = tool_manager
        self.prompt = prompt
        self.turn_number = 0
        self.tool_calls_made = 0
        self.critique_approved = False
        self.conversation: list[Message] = []

    @override
    def span_attributes(self) -> Mapping[str, str | int | float | bool]:
        """Report the turn count on the agent span, which only this loop knows."""
        return {"langchaint.agent.turns": self.turn_number}

    @override
    async def run(self) -> str:
        """Drive the turn loop to a final answer.

        A run with config.self_correction_enabled does not accept a final answer until some critique
        has returned an approval; until then the answer is appended and sent back with an instruction
        to critique, and config.max_turns is what bounds that.

        Each turn is one generate_one call under an ordinary asyncio.timeout carrying
        config.per_call_timeout_seconds. spend_log is passed as the call's abandoned_call_log, so a
        cancellation from any deadline (this one, the run's, the app's) leaves its AbandonedCall on
        the log before the CancelledError unwinds this frame; the loop appends the usage of every
        call that returns or raises, which is the app's half of the accounting.

        Raises:
            GenerationError: a generate call failed after its retries; its usage is appended to
                spend_log before the re-raise.
            TimeoutError: the base class's deadline cut the run off, or one leaked from inside a
                call (the expired() check below tells them apart). A single call outrunning
                config.per_call_timeout_seconds is handled in the loop, not raised.
            RuntimeError: the model kept calling tools for config.max_turns turns.
            DispatchExceptionGroup: a tool function raised; the settled siblings are folded first.
            asyncio.CancelledError: an outer deadline cancelled the run.
        """
        self.conversation.append(UserMessage(content=self.prompt))
        for _ in range(self.config.max_turns):
            self.turn_number += 1
            self.emit(
                TurnStarted(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    usage_so_far=self.usage,
                )
            )
            timeout_scope = asyncio.timeout(self.config.per_call_timeout_seconds)
            try:
                async with timeout_scope:
                    response = await self.bound.generate_one(
                        self.conversation, abandoned_call_log=self.spend_log
                    )
            except TimeoutError:
                if not timeout_scope.expired():
                    raise
                # generate_one appended the AbandonedCall on its way out; the dropped call also
                # appended nothing to the conversation, so the next turn resends it unchanged.
                self.emit(
                    LlmCallAbandoned(
                        agent_path=self.agent_path,
                        turn_number=self.turn_number,
                        usage_so_far=self.usage,
                    )
                )
                continue
            except GenerationError as error:
                self.spend_log.append(error.usage)
                raise
            self.spend_log.append(response.usage)
            self.emit(
                LlmResponse(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    text=response.assistant_message.text,
                    usage=response.usage,
                    usage_so_far=self.usage,
                )
            )
            self.conversation.append(response.assistant_message)
            tool_calls = response.assistant_message.tool_calls
            if not tool_calls:
                if self.config.self_correction_enabled and not self.critique_approved:
                    self.conversation.append(
                        UserMessage(
                            content="Call critique on that draft and revise it before answering."
                        )
                    )
                    continue
                return response.output
            await self._dispatch_all(tool_calls)
        raise RuntimeError(
            f"{self.agent_path} did not finish within {self.config.max_turns} turns"
        )

    async def _dispatch_all(self, tool_calls: Sequence[ToolCall]) -> None:
        """Announce every call, dispatch what the budget affords, then settle each one.

        Every call is announced before any dispatch starts, so a UI shows the whole fan-out at once
        rather than one call appearing per completion.

        Raises:
            DispatchExceptionGroup: one or more tool functions raised. Its completed_outcomes are
                folded and emitted before the re-raise, so a sibling that settled and reported spend
                is accounted for even though the turn does not finish.
            asyncio.CancelledError: an outer deadline cancelled the run mid-dispatch.
        """
        for tool_call in tool_calls:
            self.emit(
                ToolCalled(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args_json=tool_call.args_json,
                    usage_so_far=self.usage,
                )
            )
        remaining = max(0, self.config.max_tool_calls - self.tool_calls_made)
        affordable, refused = list(tool_calls[:remaining]), list(tool_calls[remaining:])
        # Charged where the calls are dispatched rather than where they settle, so a turn that raises
        # partway still spends the budget it used.
        self.tool_calls_made += len(affordable)
        try:
            outcomes = await self.tool_manager.dispatch_many(affordable)
        except DispatchExceptionGroup as group:
            self._settle_outcomes(affordable, group.completed_outcomes)
            raise
        settled = self._settle_outcomes(affordable, outcomes)
        for tool_call in refused:
            message = ToolMessage(
                tool_call_id=tool_call.id,
                content=(
                    f"Tool call budget of {self.config.max_tool_calls} is spent. "
                    "Answer with what you already have."
                ),
                is_error=True,
            )
            settled[tool_call.id] = message
            self.emit(
                ToolResponse(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=content_text(message),
                    is_error=True,
                    reported_usage=ZERO_USAGE,
                    usage_so_far=self.usage,
                )
            )
        # Every call the model made gets a reply in its original order, refused ones included:
        # a provider rejects a turn whose tool calls are not all answered.
        for tool_call in tool_calls:
            self.conversation.append(settled[tool_call.id])

    def _settle_outcomes(
        self, tool_calls: Sequence[ToolCall], outcomes: Sequence[DispatchOutcome]
    ) -> dict[str, ToolMessage]:
        """Fold each settled outcome's reported spend, emit its ToolResponse, and collect its message.

        Outcomes are matched to calls by tool_call_id rather than by position: a
        DispatchExceptionGroup's completed_outcomes covers only the calls that settled, so it is
        shorter than the calls dispatched and no index of it lines up with tool_calls.
        """
        call_of_id = {tool_call.id: tool_call for tool_call in tool_calls}
        settled: dict[str, ToolMessage] = {}
        for outcome in outcomes:
            tool_call = call_of_id[outcome.tool_message.tool_call_id]
            self._note_critique_verdict(tool_call, outcome)
            self._fold_and_emit(tool_call, outcome)
            settled[tool_call.id] = outcome.tool_message
        return settled

    def _note_critique_verdict(self, tool_call: ToolCall, outcome: DispatchOutcome) -> None:
        """Record an approving critique, which is what releases a self-correcting run to answer."""
        if (
            tool_call.name == "critique"
            and isinstance(outcome, DispatchHandled)
            and "approved" in content_text(outcome.tool_message).lower()
        ):
            self.critique_approved = True

    def _fold_and_emit(self, tool_call: ToolCall, outcome: DispatchOutcome) -> None:
        """Fold spend a tool reported through app_data and emit its ToolResponse.

        Only DispatchHandled can carry app_data; the invalid-args and unknown-tool arms are model
        mistakes that billed nothing. delegate reports None, because a sub-run wrote its own
        spend_log and folding a reported total would count the whole subtree twice.
        """
        match outcome:
            case DispatchHandled(app_data=Usage() as reported_usage):
                self.spend_log.append(reported_usage)
            case _:
                reported_usage = ZERO_USAGE
        message = outcome.tool_message
        self.emit(
            ToolResponse(
                agent_path=self.agent_path,
                turn_number=self.turn_number,
                tool_call_id=message.tool_call_id,
                tool_name=tool_call.name,
                content=content_text(message),
                is_error=message.is_error,
                reported_usage=reported_usage,
                usage_so_far=self.usage,
            )
        )


def top_level_path(name: str) -> str:
    """Build the agent path of a top-level node, which a sub-run's path is built under.

    One function rather than a literal at each site: a sub-run's spend is read back as a prefix fold
    under its parent's path, so a path written twice and changed once loses the whole subtree from
    the parent's total without any error.
    """
    return f"root/{name}"


def build_delegate_tool(
    *,
    llm: TracedLLM,
    parent_path: str,
    sub_config: AgentConfig,
    tracer: Tracer,
    capture_message_content: bool,
    registry: dict[str, AgentRun],
) -> PydanticTool[DelegateArgs, None]:
    """Build delegate, whose function drives a whole sub-agent and streams it as it runs.

    The function forwards each sub-run event to the emitter of the run that dispatched it, so the
    specialist's events interleave into the same stream the parent's do, in real time, while delegate
    stays an ordinary langchaint tool returning a value; final() then reads the outcome off the
    exhausted run. A sub-agent cannot be a streaming tool any more directly than this: a Tool returns a
    value and cannot yield, so its events reach the consumer through the run's queue or not at all.

    The sub-run's span nests under this tool's span with no plumbing: _drain creates the run's task
    from the frame delegate is iterating in, and asyncio.create_task copies the context active at
    creation, which is the one the tool span is current in.

    Nothing rides back through app_data, because the sub-run wrote its own spend_log and the parent's
    total is a path prefix fold over the registered runs' logs. The sub-run registers at construction
    (in AgentRun.__init__), so a run cancelled mid-flight is still in the report.
    """

    async def delegate(args: DelegateArgs) -> ToolOutputExplicit[None]:
        """Run the specialist to its answer, forwarding its events; its spend is already on its log."""
        parent_emitter = current_gui_emitter()
        tool_manager = TracedToolManager(
            [search_tool], capture_message_content=capture_message_content, tracer=tracer
        )
        sub_run = ReActAgent(
            agent_path=f"{parent_path}/{sub_config.name}",
            config=sub_config,
            tracer=tracer,
            registry=registry,
            bound=llm.bind(
                system_prompt=sub_config.system_prompt,
                tool_manager=tool_manager,
                automatic_prompt_caching=True,
            ),
            tool_manager=tool_manager,
            prompt=args.question,
        )
        try:
            async for event in sub_run:
                parent_emitter.emit(event)
            answer = await sub_run.final()
        except Exception as error:
            # Every failure, not just the langchaint taxonomy: a defect in a sub-agent's own tool must
            # reach the parent model as a message it can answer around, not as an exception. The spend
            # does not depend on this catch, because the rows were written as the sub-run spent.
            return ToolOutputExplicit(
                content=f"The specialist failed: {describe_error(error)}. Answer without it.",
                app_data=None,
                is_error=True,
            )
        return ToolOutputExplicit(content=answer, app_data=None)

    return PydanticTool(
        name="delegate",
        description="Delegate a focused question to the specialist sub-agent.",
        args_model=DelegateArgs,
        function=delegate,
    )


class App:
    """The graph as one async iterable, built from ReActAgents.

    The queue that interleaves two concurrent agents is in _fan_in and nowhere else, and an app running
    a single top-level agent needs no App at all: an AgentRun is already iterable.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        configs: Mapping[str, AgentConfig],
        tracer: Tracer,
        capture_message_content: bool = False,
    ) -> None:
        """Store the launch-time config of every agent, and wrap the LLM for tracing here.

        Wrapping inside the app rather than accepting a TracedLLM keeps one provider exporting the
        generate spans, the tool spans and the agent spans alike. A caller-wrapped LLM picks the
        provider for the generate spans alone, and with the global provider and none configured those
        are non-recording: the agent and tool spans arrive with holes in them and nothing reports an
        error.

        capture_message_content is passed to every wrapper the app builds and defaults to False here
        only because this example's prompts are fabricated; langchaint itself requires it with no
        default, and an application should pass it explicitly for the same reason.
        """
        self._llm = TracedLLM(llm, capture_message_content=capture_message_content, tracer=tracer)
        self._configs = configs
        self._tracer = tracer
        self._capture_message_content = capture_message_content
        self._runs: dict[str, AgentRun] = {}
        self.answers: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.final_answer: str | None = None

    @property
    def runs(self) -> Mapping[str, AgentRun]:
        """Every run the tree started, sub-agents included, keyed by agent path."""
        return self._runs

    @property
    def total_usage(self) -> Usage:
        """The tree's whole spend, folded from every registered run's spend_log.

        Each row lives on exactly one log, so this counts each exactly once and is readable after
        any cancellation.
        """
        return Usage.sum_of(run.own_usage for run in self._runs.values())

    @property
    def abandoned_calls(self) -> int:
        """Generate calls cancelled in flight anywhere in the tree, by any of the three deadlines.

        Counted as the AbandonedCall rows across every run's spend_log. What a row records is the
        cancellation, not which deadline caused it: every deadline leaves the same row, so counting
        only the innermost one would report zero abandoned calls for a run an outer deadline cut off
        mid-request.
        """
        return sum(
            1
            for run in self._runs.values()
            for row in run.spend_log
            if isinstance(row, AbandonedCall)
        )

    def _build_run(
        self,
        *,
        name: str,
        tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        prompt: str,
    ) -> ReActAgent:
        """Construct one node's agent from its config and register it."""
        config = self._configs[name]
        tool_list = [*tools, critique_tool] if config.self_correction_enabled else list(tools)
        tool_manager = TracedToolManager(
            tool_list, capture_message_content=self._capture_message_content, tracer=self._tracer
        )
        return ReActAgent(
            agent_path=top_level_path(name),
            config=config,
            tracer=self._tracer,
            registry=self._runs,
            bound=self._llm.bind(
                system_prompt=config.system_prompt,
                tool_manager=tool_manager,
                automatic_prompt_caching=True,
            ),
            tool_manager=tool_manager,
            prompt=prompt,
        )

    def _record(self, run: ReActAgent) -> None:
        """Record one finished node's outcome, so the next node can read it.

        Nothing is caught here: the run ended in AgentFailed rather than raising, so a dead node has
        already left its failure on the object.
        """
        if run.failure is not None:
            self.failures[run.agent_path] = describe_error(run.failure)
        elif run.answer is not None:
            self.answers[run.agent_path] = run.answer

    async def __aiter__(self) -> AsyncGenerator[Event]:
        """Run the graph and yield every event of the tree, sub-agents included.

        A sub-agent's events arrive through its parent's stream, forwarded by delegate, so this
        iterates only the top-level runs and still yields all three levels.

        Adding a phase means more straight-line code in this frame: another `async for` over one run,
        or over _fan_in for concurrent runs, never a helper generator wrapping this one. A consumer
        that abandons the stream closes at most the generator it holds, so teardown never rides the
        generator chain: settle() reaches every run through the registry instead.

        Yields:
            Each event in emission order.
        """
        # delegate is built before the run that owns it, so both take the path from one function
        # rather than the tool repeating it as a literal.
        delegate_tool = build_delegate_tool(
            llm=self._llm,
            parent_path=top_level_path("research_climate"),
            sub_config=self._configs["specialist"],
            tracer=self._tracer,
            capture_message_content=self._capture_message_content,
            registry=self._runs,
        )
        climate = self._build_run(
            name="research_climate",
            tools=[search_tool, delegate_tool],
            prompt="Research the climate outlook to 2030.",
        )
        energy = self._build_run(
            name="research_energy",
            tools=[search_tool],
            prompt="Research the energy outlook to 2030.",
        )
        async for event in self._fan_in([climate, energy]):
            yield event
        for run in (climate, energy):
            self._record(run)

        upstream = "\n".join(
            f"{path}: {self.answers.get(path, f'FAILED ({self.failures.get(path)})')}"
            for path in (climate.agent_path, energy.agent_path)
        )
        synthesize = self._build_run(
            name="synthesize",
            tools=[],
            prompt=f"Synthesize these findings, critiquing your draft before answering:\n{upstream}",
        )
        # No pump: one agent is already an iterator, so the app forwards its events by iterating it.
        async for event in synthesize:
            yield event
        self._record(synthesize)
        self.final_answer = self.answers.get(synthesize.agent_path)

    async def _fan_in(self, runs: Sequence[ReActAgent]) -> AsyncGenerator[Event]:
        """Interleave several agents' events into one stream.

        One `async for` consumes one iterator, so concurrent agents need something to pump into. Each
        run's first pull starts its own task, so past that first pull a pump only moves events; it is
        not what advances a loop.

        Yields:
            Events from every run, in the order the pumps enqueue them, which is emission order
            because each pump enqueues without awaiting.
        """
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def pump(run: ReActAgent) -> None:
            async for event in run:
                queue.put_nowait(event)

        async def pump_all() -> None:
            try:
                async with asyncio.TaskGroup() as group:
                    for run in runs:
                        group.create_task(pump(run))
            finally:
                queue.put_nowait(None)

        pump_task = asyncio.create_task(pump_all())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            await pump_task
        finally:
            pump_task.cancel()

    async def settle(self) -> None:
        """Cancel every started run and wait for it to unwind, so the accounting is final when read.

        Each run owns its task, so settling is asking each of them rather than tracking pumps: a
        consumer that abandoned the iteration left those tasks still running, or cancelled but not
        unwound, and generate_one has not yet appended the cancelled calls' AbandonedCall rows.
        """
        for run in list(self._runs.values()):
            await run.settle()
