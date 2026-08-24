"""Run a multi-agent app that reports progress through on_event.

AgentRun.final installs GuiEmitter, opens the agent span, emits terminal events, and drives run.
on_event executes synchronously inside the run.
TimedOutError lets the loop record a timed-out call and continue.

Each run registers at construction and appends settled TurnRecord values to turn_log.
Usage is derived from the registered runs' turn_log values.
"""

import asyncio
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import override

from config import AgentConfig
from events import (
    AgentCancelled,
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
    DispatchExceptionGroup,
    DispatchManyOutcome,
    GenerationError,
    Message,
    PydanticTool,
    Response,
    TimedOutError,
    Tool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    Usage,
    UserMessage,
    tool,
)
from langchaint.tracing import TracedBoundLLM, TracedLLM, agent_span


@dataclass(frozen=True)
class LlmTurn:
    """Record one successful generate call."""

    turn_number: int
    response: Response[str]


@dataclass(frozen=True)
class LlmFailure:
    """Record one failed or timed-out generate call and its billing."""

    turn_number: int
    error: GenerationError


@dataclass(frozen=True)
class ToolTurn:
    """Record one settled tool call and its reported Usage."""

    turn_number: int
    tool_name: str
    tool_message: ToolMessage
    reported_usage: Usage


type TurnRecord = LlmTurn | LlmFailure | ToolTurn
"""One entry in a run's ordered turn_log."""


def _spend_of(record: TurnRecord) -> Usage:
    """Return one record's reported Usage."""
    match record:
        case LlmTurn():
            return record.response.usage
        case LlmFailure():
            return record.error.usage
        case ToolTurn():
            return record.reported_usage


def _check_cost_limit(usage: Usage, max_cost_in_usd: float | None) -> None:
    """Raise when usage reaches max_cost_in_usd or has an unknown cost.

    Raises:
        RuntimeError: max_cost_in_usd is set and usage cannot support another turn.
    """
    if max_cost_in_usd is None:
        return
    cost_in_usd = usage.cost_in_usd
    if math.isnan(cost_in_usd):
        raise RuntimeError("Cannot enforce max_cost_in_usd because cost_in_usd is NaN")
    if cost_in_usd >= max_cost_in_usd:
        raise RuntimeError(f"Run reached max_cost_in_usd={max_cost_in_usd}")


class AgentRun(ABC):
    """Report one agent's execution through on_event.

    Subclasses implement run and append settled calls to turn_log.
    final installs GuiEmitter, opens the span, and emits terminal events.
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
        """Store and register the run.

        Raises:
            ValueError: registry already contains agent_path.
        """
        if agent_path in registry:
            raise ValueError(
                f"agent_path {agent_path!r} is already registered: a registry row is one run held "
                "by reference, so a second run under the same path would replace the first and "
                "drop its turn_log records from every fold. Disambiguate the path, as delegate "
                "does with its spawn index."
            )
        self.agent_path: str = agent_path
        self.config: AgentConfig = config
        self.tracer: Tracer = tracer
        self.registry: dict[str, AgentRun] = registry
        self.on_event: Callable[[Event], None] = on_event
        self.turn_log: list[TurnRecord] = []
        registry[agent_path] = self

    @abstractmethod
    async def run(self) -> str:
        """Run the agent and report progress through on_event.

        Returns:
            The final answer.
        """

    @property
    def own_usage(self) -> Usage:
        """Sum this run's turn_log Usage."""
        return Usage.sum_of(_spend_of(record) for record in self.turn_log)

    @property
    def usage(self) -> Usage:
        """Sum this run's Usage and its descendant runs' Usage."""
        return Usage.sum_of(
            run.own_usage
            for path, run in self.registry.items()
            if path == self.agent_path or path.startswith(f"{self.agent_path}/")
        )

    def span_attributes(self) -> Mapping[str, str | int | float | bool]:
        """Return agent span attributes after run finishes."""
        return {}

    async def final(self) -> str:
        """Run the agent with its GuiEmitter and span.

        Raises:
            Exception: whatever run() failed with; AgentFailed is emitted before the re-raise.
            asyncio.CancelledError: An outer scope cancelled the run.
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
                    answer = await self.run()
            except asyncio.CancelledError as error:
                try:
                    self.on_event(AgentCancelled(agent_path=self.agent_path, usage=self.usage))
                except Exception as callback_error:
                    error.add_note(f"AgentCancelled callback raised {callback_error!r}")
                raise
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
    """Implement the example's generate and tool loop."""

    def __init__(
        self,
        *,
        agent_path: str,
        config: AgentConfig,
        tracer: Tracer,
        registry: dict[str, AgentRun],
        on_event: Callable[[Event], None],
        bound: TracedBoundLLM[str, ToolManager],
        prompt: str,
    ) -> None:
        """Store the loop state."""
        super().__init__(
            agent_path=agent_path,
            config=config,
            tracer=tracer,
            registry=registry,
            on_event=on_event,
        )
        self.bound: TracedBoundLLM[str, ToolManager] = bound
        self.prompt: str = prompt
        self.turn_number: int = 0
        self.tool_calls_made: int = 0
        self.critique_approved: bool = False
        self.messages: list[Message] = []

    @override
    def span_attributes(self) -> Mapping[str, str | int | float | bool]:
        """Return the final turn count for the agent span."""
        return {"langchaint.agent.turns": self.turn_number}

    @override
    async def run(self) -> str:
        """Run generate and tool turns until the agent answers.

        config.self_correction_enabled requires critique approval.
        Each generate_one call uses config.generate_one_timeout_seconds.
        Each settled outcome is appended to turn_log.

        Raises:
            GenerationError: a generate call fails after its retries, except TimedOutError.
            RuntimeError: `max_turns` elapsed, or configured cost cannot permit another turn.
            DispatchExceptionGroup: a tool function raised; the settled siblings are folded first.
            asyncio.CancelledError: an outer deadline cancelled the run.
        """
        self.messages.append(UserMessage(content=self.prompt))
        for _ in range(self.config.max_turns):
            if (
                self.tool_calls_made >= self.config.max_tool_calls
                and self.bound.binding.tool_choice != "none"
            ):
                self.bound = self.bound.rebind(tool_choice="none")
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
                response = await self.bound.generate_one(
                    self.messages,
                    timeout_seconds=self.config.generate_one_timeout_seconds,
                )
            except GenerationError as error:
                self.turn_log.append(LlmFailure(turn_number=self.turn_number, error=error))
                if not isinstance(error, TimedOutError):
                    raise
                # The timed-out call leaves self.messages unchanged for the next turn.
                self.on_event(
                    LlmCallAbandoned(
                        agent_path=self.agent_path,
                        turn_number=self.turn_number,
                        usage_so_far=self.usage,
                    )
                )
                continue
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
            self.messages.append(response.assistant_message)
            tool_calls = response.tool_calls
            if not tool_calls:
                if self.config.self_correction_enabled and not self.critique_approved:
                    self.messages.append(
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
        """Announce, dispatch, and settle each tool call.

        Calls above config.max_tool_calls are declined through precomputed.

        Raises:
            DispatchExceptionGroup: one or more tool functions raise after completed_outcomes settle.
            RuntimeError: ToolCall.id repeats.
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
        _validate_tool_call_ids(tool_calls)
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
            self._settle_outcomes(tool_calls, group.completed_outcomes)
            raise
        self._settle_outcomes(tool_calls, outcomes)
        # Preserve call order and answer declined calls.
        for outcome in outcomes:
            self.messages.append(outcome.tool_message)

    def _settle_outcomes(
        self, tool_calls: Sequence[ToolCall], outcomes: Sequence[DispatchManyOutcome]
    ) -> None:
        """Record each outcome and emit ToolResponse.

        tool_call_id matches partial outcomes to their calls.
        An outcome with `kind == "handled"` may carry reported Usage or CritiqueVerdict through app_data.

        Raises:
            RuntimeError: An outcome repeats or names an unknown tool_call_id.
        """
        call_of_id = {tool_call.id: tool_call for tool_call in tool_calls}
        settled_ids: set[str] = set()
        for outcome in outcomes:
            tool_call_id = outcome.tool_message.tool_call_id
            if tool_call_id in settled_ids:
                raise RuntimeError(f"dispatch_many returned tool_call_id {tool_call_id!r} twice")
            settled_ids.add(tool_call_id)
            try:
                tool_call = call_of_id[tool_call_id]
            except KeyError:
                raise RuntimeError(
                    f"dispatch_many returned unknown tool_call_id {tool_call_id!r}"
                ) from None
            reported_usage = ZERO_USAGE
            if outcome.kind == "handled":
                app_data = outcome.app_data
                if isinstance(app_data, Usage):
                    reported_usage = app_data
                elif isinstance(app_data, CritiqueVerdict) and app_data.approved:
                    self.critique_approved = True
            self.turn_log.append(
                ToolTurn(
                    turn_number=self.turn_number,
                    tool_name=tool_call.name,
                    tool_message=outcome.tool_message,
                    reported_usage=reported_usage,
                )
            )
            self.on_event(
                ToolResponse(
                    agent_path=self.agent_path,
                    turn_number=self.turn_number,
                    tool_call_id=outcome.tool_message.tool_call_id,
                    tool_name=tool_call.name,
                    content=content_text(outcome.tool_message),
                    is_error=outcome.tool_message.is_error,
                    reported_usage=reported_usage,
                    usage_so_far=self.usage,
                )
            )


def top_level_path(name: str) -> str:
    """Build a top-level agent_path."""
    return f"root/{name}"


def _tools_for(
    config: AgentConfig,
    tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]],
) -> Sequence[Tool[BaseModel | Mapping[str, object] | None]]:
    """Add a fresh critique tool when self_correction_enabled."""
    return [*tools, build_critique_tool()] if config.self_correction_enabled else tools


def build_delegate_tool(
    *,
    llm: TracedLLM,
    parent_path: str,
    sub_config: AgentConfig,
    tracer: Tracer,
    registry: dict[str, AgentRun],
    on_event: Callable[[Event], None],
) -> PydanticTool[DelegateArgs, None]:
    """Build a delegate tool that runs and reports a specialist sub-agent.

    Each call appends a spawn index to agent_path.
    The sub-agent shares on_event and records Usage in its own turn_log.
    """
    spawn_counter = itertools.count()

    @tool(description="Delegate a focused question to the specialist sub-agent.")
    async def delegate(args: DelegateArgs) -> ToolOutputExplicit[None]:
        """Run the specialist and return its answer.

        Raises:
            DispatchExceptionGroup: A specialist tool function raised.
        """
        sub_run = ReActAgent(
            agent_path=f"{parent_path}/{sub_config.name}#{next(spawn_counter)}",
            config=sub_config,
            tracer=tracer,
            registry=registry,
            on_event=on_event,
            bound=llm.bind(
                system_prompt=sub_config.system_prompt,
                tools=_tools_for(sub_config, [search_tool]),
                max_attempts=sub_config.max_attempts,
                automatic_cache_breakpoints=sub_config.automatic_cache_breakpoints,
            ),
            prompt=args.question,
        )
        try:
            answer = await sub_run.final()
        except DispatchExceptionGroup:
            raise
        except Exception as error:
            # Return sub-agent failures to the parent model.
            return ToolOutputExplicit(
                content=f"The specialist failed: {describe_error(error)}. Answer without it.",
                app_data=None,
                is_error=True,
            )
        return ToolOutputExplicit(content=answer, app_data=None)

    return delegate


def _validate_tool_call_ids(tool_calls: Sequence[ToolCall]) -> None:
    """Reject duplicate IDs because partial outcomes correlate through tool_call_id.

    Raises:
        RuntimeError: tool_calls contains a duplicate ToolCall.id.
    """
    seen_ids: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.id in seen_ids:
            raise RuntimeError(f"ToolCall.id {tool_call.id!r} appears twice in one turn")
        seen_ids.add(tool_call.id)


class App:
    """Run a graph of ReActAgent values through one on_event callback."""

    def __init__(
        self,
        *,
        llm: LLM,
        configs: Mapping[str, AgentConfig],
        tracer: Tracer,
        on_event: Callable[[Event], None],
        capture_message_content: bool,
    ) -> None:
        """Store agent config and wrap llm for tracing.

        capture_message_content configures each traced LLM and ToolManager.
        """
        self._llm = TracedLLM(llm, capture_message_content=capture_message_content, tracer=tracer)
        self._configs = configs
        self._tracer = tracer
        self._on_event = on_event
        self._runs: dict[str, AgentRun] = {}
        self.answers: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.final_answer: str | None = None

    @property
    def runs(self) -> Mapping[str, AgentRun]:
        """Return all runs keyed by agent_path."""
        return self._runs

    def _build_run(
        self,
        *,
        name: str,
        tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]],
        prompt: str,
    ) -> ReActAgent:
        """Build and register one graph node."""
        config = self._configs[name]
        return ReActAgent(
            agent_path=top_level_path(name),
            config=config,
            tracer=self._tracer,
            registry=self._runs,
            on_event=self._on_event,
            bound=self._llm.bind(
                system_prompt=config.system_prompt,
                tools=_tools_for(config, tools),
                max_attempts=config.max_attempts,
                automatic_cache_breakpoints=config.automatic_cache_breakpoints,
            ),
            prompt=prompt,
        )

    async def _settle_node(self, run: AgentRun) -> None:
        """Await one node and record its answer or failure.

        Raises:
            DispatchExceptionGroup: A tool function raised.
            asyncio.CancelledError: An outer scope cancelled the graph.
        """
        try:
            self.answers[run.agent_path] = await run.final()
        except DispatchExceptionGroup:
            raise
        except Exception as error:
            self.failures[run.agent_path] = describe_error(error)

    async def run(self) -> None:
        """Run each graph node and report its events through on_event.

        TaskGroup waits for researcher cancellation before propagating it.

        Raises:
            asyncio.CancelledError: an outer deadline cancelled the graph mid-run.
            DispatchExceptionGroup: A synthesize tool function raised.
            ExceptionGroup: A concurrent researcher tool function raised.
        """
        # Use climate_name for the run and the delegate's parent_path.
        climate_name = "research_climate"
        delegate_tool = build_delegate_tool(
            llm=self._llm,
            parent_path=top_level_path(climate_name),
            sub_config=self._configs["specialist"],
            tracer=self._tracer,
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
        self.final_answer = self.answers.get(synthesize.agent_path)
