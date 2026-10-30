"""Run a multi-agent app with accounting and synchronous progress events."""

import asyncio
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, final, override

from config import AgentConfig
from events import (
    AgentCancelled,
    AgentFailed,
    AgentFinished,
    AgentStarted,
    Event,
    GenerationReported,
    GuiEmitter,
    LlmInputTimedOut,
    ToolCalled,
    ToolDispatched,
    TurnStarted,
    describe_error,
    gui_emitter_var,
)
from tools import CritiqueVerdict, DelegateArgs, build_critique_tool, search_tool

from langchaint import (
    LLM,
    ZERO_USAGE,
    BoundLLM,
    DispatchExceptionGroup,
    DispatchManyItemOutcome,
    GenerationError,
    GenerationOutcome,
    Message,
    PydanticTool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolReturnExplicit,
    ToolSequence,
    Usage,
    UserMessage,
    tool,
)


@dataclass(frozen=True)
class GenerationEntry:
    """Record one input's outcome and its billing."""

    turn_number: int
    outcome: GenerationOutcome[str]
    kind: Literal["generation"] = "generation"


@dataclass(frozen=True)
class DispatchEntry:
    """Record one tool call and its settled outcome."""

    turn_number: int
    tool_call: ToolCall
    outcome: DispatchManyItemOutcome
    kind: Literal["dispatch"] = "dispatch"


type TurnLogEntry = GenerationEntry | DispatchEntry
"""One entry in a run's ordered turn_log."""


def _usage_of(entry: TurnLogEntry) -> Usage:
    match entry.kind:
        case "generation":
            return entry.outcome.usage
        case "dispatch":
            outcome = entry.outcome
            if outcome.kind == "handled" and isinstance(outcome.app_data, Usage):
                return outcome.app_data
            return ZERO_USAGE


class RunLimitError(RuntimeError):
    """A configured turn or cost limit prevents another turn."""


def _check_cost_limit(usage: Usage, max_cost_in_usd: float | None) -> None:
    """Require known cost below max_cost_in_usd when configured."""
    if max_cost_in_usd is None:
        return
    cost_in_usd = usage.cost_in_usd
    if math.isnan(cost_in_usd):
        raise RunLimitError("Cannot enforce max_cost_in_usd because cost_in_usd is NaN")
    if cost_in_usd >= max_cost_in_usd:
        raise RunLimitError(f"Run reached max_cost_in_usd={max_cost_in_usd}")


class AgentRun(ABC):
    """Define this demo app's base class for agent execution.

    The app's agent classes inherit from AgentRun and implement their own workflow in _execute.
    Subclasses append settled outcomes to turn_log for accounting and return a final answer.
    Call the inherited run method to route tool progress and emit lifecycle events:
    AgentStarted, AgentFinished, AgentFailed, and AgentCancelled.
    """

    def __init__(
        self,
        *,
        agent_path: str,
        registry: dict[str, "AgentRun"],
        on_event: Callable[[Event], None],
    ) -> None:
        """Register agent_path for descendant accounting and store on_event for synchronous progress callbacks.

        registry stores runs under unique agent_path values.

        Raises:
            ValueError: registry already contains agent_path.
        """
        if agent_path in registry:
            raise ValueError(f"agent_path {agent_path!r} is already registered")
        self.agent_path: str = agent_path
        self.registry: dict[str, AgentRun] = registry
        self.on_event: Callable[[Event], None] = on_event
        self.turn_log: list[TurnLogEntry] = []
        registry[agent_path] = self

    @property
    def own_usage(self) -> Usage:
        """Sum this run's turn_log Usage."""
        return Usage.sum_of(_usage_of(entry) for entry in self.turn_log)

    @property
    def usage(self) -> Usage:
        """Sum this run's Usage and its descendant runs' Usage."""
        return Usage.sum_of(
            run.own_usage
            for path, run in self.registry.items()
            if path == self.agent_path or path.startswith(f"{self.agent_path}/")
        )

    @final
    async def run(self) -> str:
        """Execute the agent with GuiEmitter and emit a terminal event.

        Raises:
            Exception: _execute or on_event raises.
            asyncio.CancelledError: An outer scope cancels the run.
        """
        token = gui_emitter_var.set(GuiEmitter(self.agent_path, self.on_event))
        try:
            self.on_event(AgentStarted(agent_path=self.agent_path))
            try:
                answer = await self._execute()
            except asyncio.CancelledError as error:
                try:
                    self.on_event(AgentCancelled(agent_path=self.agent_path, usage=self.usage))
                except Exception as callback_error:
                    error.add_note(f"AgentCancelled callback raised {callback_error!r}")
                raise
            except Exception as error:
                self.on_event(
                    AgentFailed(
                        agent_path=self.agent_path,
                        error_text=describe_error(error),
                        usage=self.usage,
                    )
                )
                raise
            self.on_event(
                AgentFinished(agent_path=self.agent_path, answer=answer, usage=self.usage)
            )
            return answer
        finally:
            gui_emitter_var.reset(token)

    @abstractmethod
    async def _execute(self) -> str:
        """Record settled outcomes in turn_log and return the final answer."""


class ReActAgent(AgentRun):
    """Generate and dispatch tools until the model answers within configured limits.

    Raises:
        GenerationError: Generation fails, except with kind timed_out_error.
        RunLimitError: The turn or cost limit prevents another turn.
        RuntimeError: Tool-call IDs repeat or a dispatch outcome has an invalid ID.
        DispatchExceptionGroup: A tool function raises after settled sibling outcomes enter turn_log.
    """

    def __init__(
        self,
        *,
        agent_path: str,
        config: AgentConfig,
        llm: LLM,
        tools: ToolSequence,
        prompt: str,
        registry: dict[str, AgentRun],
        on_event: Callable[[Event], None],
    ) -> None:
        """Bind config and tools, initialize prompt messages, and register agent_path.

        config sets the system prompt, binding options, and execution limits.
        self_correction_enabled adds a fresh scripted critique tool to tools.
        registry stores runs for descendant accounting under unique agent_path values.
        on_event receives synchronous progress callbacks from the run and its tools.

        Raises:
            ValueError: registry already contains agent_path.
            ValueError: A binding option in config is invalid or unsupported by the provider.
        """
        tools = [*tools, build_critique_tool()] if config.self_correction_enabled else tools
        self.bound: BoundLLM[str, ToolManager] = llm.bind(
            system_prompt=config.system_prompt,
            tools=tools,
            max_requests=config.max_requests,
            automatic_cache_breakpoints=config.automatic_cache_breakpoints,
        )
        self.config: AgentConfig = config
        self.turn_number: int = 0
        self.tool_calls_made: int = 0
        self.critique_approved: bool = False
        self.messages: list[Message] = [UserMessage(content=prompt)]
        super().__init__(agent_path=agent_path, registry=registry, on_event=on_event)

    @override
    async def _execute(self) -> str:
        """Generate and dispatch tools until the agent answers."""
        for _ in range(self.config.max_turns):
            if (
                self.tool_calls_made >= self.config.max_tool_calls
                and self.bound.binding.tool_choice != "none"
            ):
                self.bound = self.bound.bind(tool_choice="none")
            usage_so_far = self.usage
            _check_cost_limit(usage_so_far, self.config.max_cost_in_usd)
            self.turn_number += 1
            self.on_event(
                TurnStarted(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    usage_so_far=usage_so_far,
                )
            )
            try:
                generation = await self.bound.generate_one(
                    self.messages,
                    timeout_seconds=self.config.generate_one_timeout_seconds,
                )
            except GenerationError as error:
                self.turn_log.append(GenerationEntry(turn_number=self.turn_number, outcome=error))
                if error.kind != "timed_out_error":
                    raise
                # The timed-out `generate_one` leaves self.messages unchanged for the next turn.
                self.on_event(
                    LlmInputTimedOut(
                        agent_path=self.agent_path,
                        turn_number=self.turn_number,
                        usage_so_far=self.usage,
                    )
                )
                continue
            self.turn_log.append(GenerationEntry(turn_number=self.turn_number, outcome=generation))
            self.on_event(
                GenerationReported(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    generation=generation,
                    usage_so_far=self.usage,
                )
            )
            self.messages.append(generation.assistant_message)
            match generation.kind:
                case "tool_call":
                    await self._dispatch_all(generation.tool_calls)
                case "plain":
                    if self.config.self_correction_enabled and not self.critique_approved:
                        self.messages.append(
                            UserMessage(
                                content="Call critique on that draft and revise it before answering."
                            )
                        )
                        continue
                    return generation.output
        raise RunLimitError(
            f"{self.agent_path} did not finish within {self.config.max_turns} turns"
        )

    async def _dispatch_all(self, tool_calls: Sequence[ToolCall]) -> None:
        """Dispatch the affordable prefix and settle dispatched and declined calls."""
        call_of_id = {tool_call.id: tool_call for tool_call in tool_calls}
        if len(call_of_id) != len(tool_calls):
            raise RuntimeError("ToolCall.id appears twice in one assistant message")
        usage_so_far = self.usage
        for tool_call in tool_calls:
            self.on_event(
                ToolCalled(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    tool_call=tool_call,
                    usage_so_far=usage_so_far,
                )
            )
        remaining = max(0, self.config.max_tool_calls - self.tool_calls_made)
        affordable_ids = {tool_call.id for tool_call in tool_calls[:remaining]}
        # Dispatch consumes the budget even when settlement raises.
        self.tool_calls_made += len(affordable_ids)

        def _decline_over_budget(tool_call: ToolCall) -> ToolMessage | None:
            if tool_call.id in affordable_ids:
                return None
            return ToolMessage.error(
                tool_call,
                f"Tool call budget of {self.config.max_tool_calls} is spent. "
                "Answer with what you already have.",
            )

        try:
            outcomes = await self.bound.tool_manager.dispatch_many(
                tool_calls, precomputed=_decline_over_budget
            )
        except DispatchExceptionGroup as group:
            self._settle_outcomes(call_of_id, group.completed_outcomes)
            raise
        self._settle_outcomes(call_of_id, outcomes)
        self.messages.extend(outcome.tool_message for outcome in outcomes)

    def _settle_outcomes(
        self, call_of_id: Mapping[str, ToolCall], outcomes: Sequence[DispatchManyItemOutcome]
    ) -> None:
        """Record the batch, then emit events with usage captured after each entry."""
        settled_ids: set[str] = set()
        events: list[ToolDispatched] = []
        for outcome in outcomes:
            tool_message = outcome.tool_message
            tool_call_id = tool_message.tool_call_id
            if tool_call_id in settled_ids:
                raise RuntimeError(f"dispatch_many returned tool_call_id {tool_call_id!r} twice")
            settled_ids.add(tool_call_id)
            try:
                tool_call = call_of_id[tool_call_id]
            except KeyError:
                raise RuntimeError(
                    f"dispatch_many returned unknown tool_call_id {tool_call_id!r}"
                ) from None
            entry = DispatchEntry(
                turn_number=self.turn_number,
                tool_call=tool_call,
                outcome=outcome,
            )
            self.turn_log.append(entry)
            if (
                outcome.kind == "handled"
                and isinstance(outcome.app_data, CritiqueVerdict)
                and outcome.app_data.approved
            ):
                self.critique_approved = True
            events.append(
                ToolDispatched(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    tool_call=entry.tool_call,
                    tool_message=outcome.tool_message,
                    reported_usage=_usage_of(entry),
                    usage_so_far=self.usage,
                )
            )
        for event in events:
            self.on_event(event)


def top_level_path(name: str) -> str:
    """Build a top-level agent_path."""
    return f"root/{name}"


def build_delegate_tool(
    *,
    llm: LLM,
    parent_path: str,
    config: AgentConfig,
    registry: dict[str, AgentRun],
    on_event: Callable[[Event], None],
) -> PydanticTool[DelegateArgs, None]:
    """Build a delegate tool that runs and reports a specialist sub-agent.

    Each call appends a spawn index to agent_path.
    The sub-agent shares on_event and records Usage in its own turn_log.
    """
    spawn_counter = itertools.count()

    @tool(description="Delegate a focused question to the specialist sub-agent.")
    async def delegate(args: DelegateArgs) -> str | ToolReturnExplicit[None]:
        """Return the specialist's answer or a tool error for a classified generation failure or run limit.

        Raises:
            DispatchExceptionGroup: A specialist tool function raises.
            GenerationError: The specialist reports an unknown generation failure.
            Exception: A callback or application defect raises.
        """
        sub_run = ReActAgent(
            agent_path=f"{parent_path}/{config.name}#{next(spawn_counter)}",
            config=config,
            registry=registry,
            on_event=on_event,
            llm=llm,
            tools=[search_tool],
            prompt=args.question,
        )
        try:
            return await sub_run.run()
        except (GenerationError, RunLimitError) as error:
            if isinstance(error, GenerationError) and error.kind == "unknown_exception_error":
                raise
            return ToolReturnExplicit(
                content=f"The specialist failed: {describe_error(error)}. Answer without it.",
                is_error=True,
            )

    return delegate


class App:
    """Run a graph of AgentRun values through one on_event callback."""

    def __init__(
        self,
        *,
        llm: LLM,
        configs: Mapping[str, AgentConfig],
        on_event: Callable[[Event], None],
    ) -> None:
        """Share llm and on_event across runs; configs must include every node name and specialist."""
        self._llm = llm
        self._configs = configs
        self._on_event = on_event
        self._runs: dict[str, AgentRun] = {}
        self.answers: dict[str, str] = {}
        self.failures: dict[str, str] = {}

    @property
    def final_answer(self) -> str | None:
        """Return the synthesis answer when it completed successfully."""
        return self.answers.get(top_level_path("synthesize"))

    @property
    def runs(self) -> Mapping[str, AgentRun]:
        """Return all runs keyed by agent_path."""
        return self._runs

    def _build_run(
        self,
        *,
        name: str,
        tools: ToolSequence,
        prompt: str,
    ) -> ReActAgent:
        config = self._configs[name]
        return ReActAgent(
            agent_path=top_level_path(name),
            config=config,
            registry=self._runs,
            on_event=self._on_event,
            llm=self._llm,
            tools=tools,
            prompt=prompt,
        )

    async def _settle_node(self, run: AgentRun) -> None:
        """Await one node and record its answer or failure.

        Raises:
            DispatchExceptionGroup: A tool function raises.
            GenerationError: The run reports an unknown generation failure.
            Exception: A callback or application defect raises.
            asyncio.CancelledError: An outer scope cancels the graph.
        """
        try:
            self.answers[run.agent_path] = await run.run()
        except (GenerationError, RunLimitError) as error:
            if isinstance(error, GenerationError) and error.kind == "unknown_exception_error":
                raise
            self.failures[run.agent_path] = describe_error(error)

    async def run(self) -> None:
        """Run each graph node and report its events through on_event.

        Raises:
            KeyError: A required configuration is missing.
            ValueError: An `agent_path` is already registered.
            ValueError: `max_requests` is boolean or below one.
            ValueError: `automatic_cache_breakpoints` is unsupported.
            asyncio.CancelledError: An outer deadline cancels the graph after researcher tasks settle.
            DispatchExceptionGroup: A synthesize tool function raises.
            GenerationError: Synthesis reports an unknown generation failure.
            ExceptionGroup: A concurrent researcher raises an unhandled failure.
            Exception: A synthesis callback or application defect raises.
        """
        delegate_tool = build_delegate_tool(
            llm=self._llm,
            parent_path=top_level_path("research_climate"),
            config=self._configs["specialist"],
            registry=self._runs,
            on_event=self._on_event,
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
        async with asyncio.TaskGroup() as group:
            for node in (climate, energy):
                _ = group.create_task(self._settle_node(node))

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
