"""The recommended shape for a streaming multi-agent app on langchaint.

The app hands every run one on_event callback, and a run reports by calling it; nothing here is an
async iterator.

AgentRun splits the agent into two roles. The base class owns final(), the single entry point: it
installs the run's GuiEmitter, opens the agent span and the run deadline around the loop, emits
AgentStarted and the terminal event, and returns the answer or re-raises the failure. The application
subclasses it and writes only run(), a plain coroutine, calling self.on_event(event) as it goes.

on_event runs inside the run's frame, so its time counts against config.timeout_seconds. A consumer
wanting its pace decoupled from the run's deadline passes on_event=queue.put_nowait and drains the
queue at its own pace; the decoupling is the consumer's choice, never the shape's.

Two capabilities ride on contextvars, the same mechanism that nests OTel spans across tasks:

    Each run's final() installs its own GuiEmitter, so a tool function reports progress into the
    on_event of whichever run dispatched it (search does, through events.current_gui_emitter).
    Threading cannot do this without a parameter on every tool, because a tool function holds no
    handle on its run.

    delegate constructs its sub-run with the same on_event as every other run, so a three-level tree
    reports to one consumer in real time while delegate stays an ordinary langchaint tool returning a
    value.

Only the emitter is ambient. The run registry and on_event stay constructor arguments, where a missing
value is a TypeError at the call site instead of a LookupError at the first read inside an agent; the
emitter is ambient because for it there is no call site to thread through.

The state a failure must leave behind lives on this object, never in the coroutine frame doing the
work: a timeout cancels the frame, and anything local to it is gone. The record follows the same
rule: each run appends every TurnRecord to its own turn_log at the moment it happens (langchaint
appends the AbandonedCall a cancellation would otherwise erase, because the log doubles as the
call's abandoned_call_log), and every run registers itself at construction. Any metric a consumer
could want is a post-run fold over the registered runs' ordered logs; a parent's subtree total is
the fold filtered by path prefix. A CancelledError is a BaseException that passes through every
`except Exception` in the tree, so a design that carries a sub-run's records home on a return path
loses them when that path never runs; records already written need no carrying, so nothing can be
dropped.
"""

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    describe_error,
    gui_emitter_var,
)
from opentelemetry.trace import Tracer
from pydantic import BaseModel
from scenario import CritiqueVerdict, DelegateArgs, build_critique_tool, search_tool

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
    Response,
    Tool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    Usage,
    UserMessage,
)
from langchaint.tracing import TracedBoundLLM, TracedLLM, TracedToolManager, agent_span


@dataclass(frozen=True)
class LlmTurn:
    """One settled generate call, held as the Response langchaint returned.

    response carries the assistant message, the call's usage and the raw SDK payload by reference,
    so the record loses nothing of the call; turn_number is the same number max_turns bounds.
    """

    turn_number: int
    response: Response[str]


@dataclass(frozen=True)
class LlmFailure:
    """One generate call that failed after its retries, held as the GenerationError the run re-raised.

    error.usage is what the failed call billed, which is why a failure is a record and not only a raise.
    """

    turn_number: int
    error: GenerationError


@dataclass(frozen=True)
class ToolTurn:
    """One tool call answered within a turn; a turn with several calls appends several of these.

    tool_message is the reply the model reads, budget refusals included. reported_usage is spend
    the tool reported through app_data, ZERO_USAGE when it reported none; a delegate call reports
    none, because its sub-run writes its own turn_log.
    """

    turn_number: int
    tool_name: str
    tool_message: ToolMessage
    reported_usage: Usage


type TurnRecord = LlmTurn | LlmFailure | ToolTurn | AbandonedCall
"""One record in a run's ordered turn_log, the full account of what the run did and spent.

The loop appends the first three; generate_one appends the AbandonedCall inside the frame a
cancellation unwinds, because the log doubles as the call's abandoned_call_log. Any metric a
consumer could want is a post-run fold over these records.
"""


def _spend_of(record: TurnRecord) -> Usage:
    """Return what one record billed, the fold step that turns an ordered turn_log into a total.

    An AbandonedCall contributes only its settled attempts' spend, because the in-flight attempt's
    cost is unobservable.
    """
    match record:
        case LlmTurn():
            return record.response.usage
        case LlmFailure():
            return record.error.usage
        case ToolTurn():
            return record.reported_usage
        case AbandonedCall():
            return record.usage_settled


class AgentRun(ABC):
    """The reporting half of an agent: the emitter install, the deadline, the span, the terminal events.

    A subclass supplies run(), its loop, as a coroutine, and calls on_event as it goes. Awaiting final()
    drives that loop to the answer, and the outcome rides final()'s return and raise; the accounting
    stays on this object, readable whatever the outcome.

    AgentStarted, AgentFinished and AgentFailed are emitted here, because this class is what knows how
    the run ended. A subclass emits only what happens inside its loop.

    turn_log is the run's append-only ordered record: the loop appends one TurnRecord per settled
    generate call, per failed one and per answered tool call, and langchaint appends an
    AbandonedCall where a cancellation cut a generate call off. Any metric a consumer could want
    is a fold over the logs, never a running sum, so there is no second copy of any record to
    drift from.
    """

    def __init__(
        self,
        *,
        agent_path: str,
        config: AgentConfig,
        tracer: Tracer,
        registry: dict[str, "AgentRun"],
        on_event: Callable[[Event], None],
    ) -> None:
        """Fix the run's identity, limits and on_event; a subclass adds whatever its loop needs.

        The run registers itself in registry here, at construction rather than at start, so a run
        cancelled mid-flight is still in every report folded over the registry.

        Raises:
            ValueError: registry already holds agent_path. A registry row is the run object itself,
                its turn_log and counters held by reference, so two runs cannot share one and a
                silent replacement would drop the first run's records from every fold. A spawner that
                reuses an agent name disambiguates the path, as delegate does with its spawn index.
        """
        if agent_path in registry:
            raise ValueError(
                f"agent_path {agent_path!r} is already registered: a registry row is one run held "
                "by reference, so a second run under the same path would replace the first and "
                "drop its turn_log records from every fold. Disambiguate the path, as delegate "
                "does with its spawn index."
            )
        self.agent_path = agent_path
        self.config = config
        self.tracer = tracer
        self.registry = registry
        self.on_event = on_event
        self.turn_log: list[TurnRecord] = []
        registry[agent_path] = self

    @abstractmethod
    async def run(self) -> str:
        """Drive this agent's loop to its final answer, calling on_event as it goes.

        A coroutine, never a generator: the base class awaits it under the run's deadline and span, and
        a `yield` here would make it an async generator that cannot be awaited at all.

        Returns:
            The run's final answer. Raising instead is how a run fails; the base class reports it.
        """

    @property
    def own_usage(self) -> Usage:
        """The run's own spend, folded from turn_log; a sub-run's records live on its own log."""
        return Usage.sum_of(_spend_of(record) for record in self.turn_log)

    @property
    def usage(self) -> Usage:
        """The run's own spend plus every sub-run beneath it, folded from the registry by path prefix.

        The prefix fold is what hands a parent its subtree's total without any run having carried a
        number home, and it counts each record exactly once because each lives on exactly one log.
        """
        return Usage.sum_of(
            run.own_usage
            for path, run in self.registry.items()
            if path == self.agent_path or path.startswith(f"{self.agent_path}/")
        )

    def span_attributes(self) -> Mapping[str, str | int | float | bool]:
        """Extra attributes to set on the agent span when the run ends; override to add.

        Called after the loop leaves, so a subclass reports final counters here rather than tracking a
        span it does not own.
        """
        return {}

    async def final(self) -> str:
        """Drive the run to its answer, reporting through on_event as it goes.

        The run's GuiEmitter is installed for the duration and reset on the way out, so a tool
        function dispatched by this run reports to this run's on_event, and a sub-run started inside
        a tool restores its parent's emitter when it ends.

        Raises:
            TimeoutError: the run outran config.timeout_seconds.
            Exception: whatever run() failed with; AgentFailed is emitted before the re-raise.
            asyncio.CancelledError: an outer deadline cancelled the run. No terminal event is emitted,
                because the run has no outcome to report; the turn logs already hold the records.
        """
        token = gui_emitter_var.set(GuiEmitter(self.agent_path, self.on_event))
        try:
            self.on_event(AgentStarted(agent_path=self.agent_path))
            try:
                with agent_span(
                    self.tracer,
                    agent_name=self.config.name,
                    agent_path=self.agent_path,
                    usage=lambda: self.usage,
                    extra_attributes=self.span_attributes,
                ):
                    async with asyncio.timeout(self.config.timeout_seconds):
                        answer = await self.run()
            except Exception as error:
                self.on_event(
                    AgentFailed(
                        agent_path=self.agent_path, error=describe_error(error), usage=self.usage
                    )
                )
                raise
            self.on_event(
                AgentFinished(agent_path=self.agent_path, answer=answer, usage=self.usage)
            )
            return answer
        finally:
            gui_emitter_var.reset(token)


class ReActAgent(AgentRun):
    """The example's loop, written as the application's half of the contract: one coroutine.

    Everything here is ordinary async code. The per-call deadline is an ordinary `asyncio.timeout`, the
    events go out through on_event, and the run's own deadline and span are the base class's business.
    """

    def __init__(  # noqa: PLR0913 (the five arguments every run needs plus this loop's bound, tool_manager and prompt)
        self,
        *,
        agent_path: str,
        config: AgentConfig,
        tracer: Tracer,
        registry: dict[str, AgentRun],
        on_event: Callable[[Event], None],
        bound: TracedBoundLLM[str],
        tool_manager: ToolManager,
        prompt: str,
    ) -> None:
        """Add what this loop needs on top of what every run needs."""
        super().__init__(
            agent_path=agent_path,
            config=config,
            tracer=tracer,
            registry=registry,
            on_event=on_event,
        )
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
        config.per_call_timeout_seconds. turn_log is passed as the call's abandoned_call_log, so a
        cancellation from any deadline (this one, the run's, the app's) leaves its AbandonedCall on
        the log before the CancelledError unwinds this frame; the loop appends a record for every
        call that returns or raises, which is the app's half of the log.

        Raises:
            GenerationError: a generate call failed after its retries; an LlmFailure holding it is
                appended to turn_log before the re-raise.
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
            self.on_event(
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
                        self.conversation, abandoned_call_log=self.turn_log
                    )
            except TimeoutError:
                if not timeout_scope.expired():
                    raise
                # generate_one appended the AbandonedCall on its way out; the dropped call also
                # appended nothing to the conversation, so the next turn resends it unchanged.
                self.on_event(
                    LlmCallAbandoned(
                        agent_path=self.agent_path,
                        turn_number=self.turn_number,
                        usage_so_far=self.usage,
                    )
                )
                continue
            except GenerationError as error:
                self.turn_log.append(LlmFailure(turn_number=self.turn_number, error=error))
                raise
            self.turn_log.append(LlmTurn(turn_number=self.turn_number, response=response))
            self.on_event(
                LlmResponse(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    text=response.assistant_message.text,
                    usage=response.usage,
                    usage_so_far=self.usage,
                )
            )
            self.conversation.append(response.assistant_message)
            tool_calls = response.tool_calls
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
        usage_so_far = self.usage
        for tool_call in tool_calls:
            self.on_event(
                ToolCalled(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args_json=tool_call.args_json,
                    usage_so_far=usage_so_far,
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
            self._record_and_emit(tool_call, message, reported_usage=ZERO_USAGE)
        # Every call the model made gets a reply in its original order, refused ones included:
        # a provider rejects a turn whose tool calls are not all answered.
        for tool_call in tool_calls:
            self.conversation.append(settled[tool_call.id])

    def _settle_outcomes(
        self, tool_calls: Sequence[ToolCall], outcomes: Sequence[DispatchOutcome]
    ) -> dict[str, ToolMessage]:
        """Record each settled outcome on turn_log, emit its ToolResponse, and collect its message.

        Outcomes are matched to calls by tool_call_id rather than by position: a
        DispatchExceptionGroup's completed_outcomes covers only the calls that settled, so it is
        shorter than the calls dispatched and no index of it lines up with tool_calls.

        app_data is read here, at its type.
        A Usage is spend the tool reported, carried only by DispatchHandled:
        the invalid-args and unknown-tool arms are model mistakes that billed nothing,
        and delegate reports None, because a sub-run wrote its own turn_log
        and folding a reported total would count the whole subtree twice.
        An approving CritiqueVerdict is what releases a self-correcting run to answer.
        """
        call_of_id = {tool_call.id: tool_call for tool_call in tool_calls}
        settled: dict[str, ToolMessage] = {}
        for outcome in outcomes:
            tool_call = call_of_id[outcome.tool_message.tool_call_id]
            match outcome:
                case DispatchHandled(app_data=Usage() as reported_usage):
                    pass
                case DispatchHandled(app_data=CritiqueVerdict(approved=True)):
                    self.critique_approved = True
                    reported_usage = ZERO_USAGE
                case _:
                    reported_usage = ZERO_USAGE
            self._record_and_emit(tool_call, outcome.tool_message, reported_usage=reported_usage)
            settled[tool_call.id] = outcome.tool_message
        return settled

    def _record_and_emit(
        self, tool_call: ToolCall, message: ToolMessage, *, reported_usage: Usage
    ) -> None:
        """Append the settled call's ToolTurn to turn_log and emit its ToolResponse.

        Serves dispatched and budget-refused calls alike, so the record and the event have one home.
        """
        self.turn_log.append(
            ToolTurn(
                turn_number=self.turn_number,
                tool_name=tool_call.name,
                tool_message=message,
                reported_usage=reported_usage,
            )
        )
        self.on_event(
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


def _tool_manager_for(
    config: AgentConfig,
    tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]],
    *,
    capture_message_content: bool,
    tracer: Tracer,
) -> TracedToolManager:
    """Assemble a run's tool manager from its config, the one place a tool list is built.

    self_correction_enabled adds a fresh critique here,
    so the flag cannot separate from the tool its bounce requires:
    a run with it on always has critique in its schema, delegate-spawned runs included.
    """
    tool_list = [*tools, build_critique_tool()] if config.self_correction_enabled else list(tools)
    return TracedToolManager(
        tool_list, capture_message_content=capture_message_content, tracer=tracer
    )


def build_delegate_tool(
    *,
    llm: TracedLLM,
    parent_path: str,
    sub_config: AgentConfig,
    tracer: Tracer,
    capture_message_content: bool,
    registry: dict[str, AgentRun],
    on_event: Callable[[Event], None],
) -> PydanticTool[DelegateArgs, None]:
    """Build delegate, whose function drives a whole sub-agent and reports it as it runs.

    The sub-run is constructed with the same on_event as every other run, so the specialist's events
    reach the same consumer the parent's do, in real time, while delegate stays an ordinary langchaint
    tool returning a value.

    Each call spawns a fresh run at "{parent_path}/{name}#{spawn_index}": an agent's name is not
    unique within a parent, and a registry row is one run held by reference, so identity is per
    spawn and AgentRun.__init__ rejects a duplicate path. The index rides the agent_path, the one
    identity every event carries, so a UI showing two spawns of one name shows two runs.

    The sub-run's span nests under this tool's span with no bookkeeping: final() runs in the frame
    delegate is awaiting in, which is the one the tool span is current in.

    Nothing rides back through app_data, because the sub-run wrote its own turn_log and the parent's
    total is a path prefix fold over the registered runs' logs. The sub-run registers at construction
    (in AgentRun.__init__), so a run cancelled mid-flight is still in the record.
    """
    spawn_counter = itertools.count()

    async def delegate(args: DelegateArgs) -> ToolOutputExplicit[None]:
        """Run the specialist to its answer; its spend is already on its log.

        The tool manager is assembled per spawn, through the same _tool_manager_for every run uses,
        so a self-correcting sub_config hands each spawn its own critique script.
        """
        tool_manager = _tool_manager_for(
            sub_config,
            [search_tool],
            capture_message_content=capture_message_content,
            tracer=tracer,
        )
        sub_run = ReActAgent(
            agent_path=f"{parent_path}/{sub_config.name}#{next(spawn_counter)}",
            config=sub_config,
            tracer=tracer,
            registry=registry,
            on_event=on_event,
            bound=llm.bind(
                system_prompt=sub_config.system_prompt,
                tool_manager=tool_manager,
                automatic_prompt_caching=True,
            ),
            tool_manager=tool_manager,
            prompt=args.question,
        )
        try:
            answer = await sub_run.final()
        except Exception as error:
            # Every failure, not just the langchaint taxonomy: a defect in a sub-agent's own tool must
            # reach the parent model as a message it can answer around, not as an exception. The spend
            # does not depend on this catch, because the records were written as the sub-run spent.
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
    """The graph, built from ReActAgents that all report to one on_event.

    An app running a single top-level agent needs no App at all: constructing an AgentRun and awaiting
    final() is the whole apparatus.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        configs: Mapping[str, AgentConfig],
        tracer: Tracer,
        on_event: Callable[[Event], None],
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
        self._on_event = on_event
        self._runs: dict[str, AgentRun] = {}
        self.answers: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.final_answer: str | None = None

    @property
    def runs(self) -> Mapping[str, AgentRun]:
        """Every run the tree started, sub-agents included, keyed by agent path."""
        return self._runs

    def _build_run(
        self,
        *,
        name: str,
        tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        prompt: str,
    ) -> ReActAgent:
        """Construct one node's agent from its config and register it."""
        config = self._configs[name]
        tool_manager = _tool_manager_for(
            config,
            tools,
            capture_message_content=self._capture_message_content,
            tracer=self._tracer,
        )
        return ReActAgent(
            agent_path=top_level_path(name),
            config=config,
            tracer=self._tracer,
            registry=self._runs,
            on_event=self._on_event,
            bound=self._llm.bind(
                system_prompt=config.system_prompt,
                tool_manager=tool_manager,
                automatic_prompt_caching=True,
            ),
            tool_manager=tool_manager,
            prompt=prompt,
        )

    async def _settle_node(self, run: ReActAgent) -> None:
        """Await one node's outcome and record it, so the next phase can read it.

        A failure is recorded rather than propagated, because a dead node must not cancel its
        concurrent sibling and the next phase synthesizes around the hole. A CancelledError still
        propagates: it is a BaseException, and an outer deadline that raised it is ending the graph.
        """
        try:
            self.answers[run.agent_path] = await run.final()
        except Exception as error:
            self.failures[run.agent_path] = describe_error(error)

    async def run(self) -> None:
        """Run the graph, reporting every event of the tree through on_event.

        A sub-agent's events arrive through the same on_event, because delegate constructs its
        sub-run with it, so all three levels reach one consumer with no forwarding.

        Adding a phase is more straight-line code in this frame: another awaited _settle_node, or a
        TaskGroup of them for concurrent nodes.

        The whole-app deadline is the caller's asyncio.timeout around this call. Its cancellation
        lands in whatever frame is running, and the TaskGroup awaits every child's unwind before
        letting it propagate, so the accounting is final when the caller's except reads it: the
        cancelled calls' AbandonedCall records are on the turn logs, appended by generate_one inside
        the frames the cancellation unwound.

        Raises:
            asyncio.CancelledError: an outer deadline cancelled the graph mid-run.
        """
        # delegate is built before the run that owns it, so both take the path from one function
        # rather than the tool repeating it as a literal; climate_name feeds both calls so the
        # parent path and the run name cannot drift apart.
        climate_name = "research_climate"
        delegate_tool = build_delegate_tool(
            llm=self._llm,
            parent_path=top_level_path(climate_name),
            sub_config=self._configs["specialist"],
            tracer=self._tracer,
            capture_message_content=self._capture_message_content,
            registry=self._runs,
            on_event=self._on_event,
        )
        climate = self._build_run(
            name=climate_name,
            tools=[search_tool, delegate_tool],
            prompt="Research the climate outlook to 2030.",
        )
        energy = self._build_run(
            name="research_energy",
            tools=[search_tool],
            prompt="Research the energy outlook to 2030.",
        )
        async with asyncio.TaskGroup() as group:
            for node in (climate, energy):
                group.create_task(self._settle_node(node))

        upstream = "\n".join(
            f"{path}: {self.answers.get(path, f'FAILED ({self.failures.get(path)})')}"
            for path in (climate.agent_path, energy.agent_path)
        )
        synthesize = self._build_run(
            name="synthesize",
            tools=[],
            prompt=f"Synthesize these findings, critiquing your draft before answering:\n{upstream}",
        )
        await self._settle_node(synthesize)
        self.final_answer = self.answers.get(synthesize.agent_path)
