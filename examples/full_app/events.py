"""Define UI events and route tool progress to the current run.

agent_path identifies the emitting run and includes a spawn index for tool-spawned runs.
usage_so_far and terminal usage include sub-agent spend.
GenerationReported.generation.usage covers one input.
"""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from langchaint import Generation, ToolCall, ToolMessage, Usage


def describe_error(error: BaseException) -> str:
    """Render an exception with its type and optional message."""
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


@dataclass(frozen=True)
class AgentStarted:
    """Report that a run began."""

    agent_path: str
    kind: Literal["agent_started"] = "agent_started"


@dataclass(frozen=True)
class TurnStarted:
    """Report the start of a numbered turn."""

    agent_path: str
    turn_number: int
    usage_so_far: Usage
    kind: Literal["turn_started"] = "turn_started"


@dataclass(frozen=True)
class GenerationReported:
    """Report one generation and captured cumulative usage."""

    agent_path: str
    turn_number: int
    generation: Generation[str]
    usage_so_far: Usage
    kind: Literal["generation_reported"] = "generation_reported"


@dataclass(frozen=True)
class ToolCalled:
    """Report a tool call before dispatch."""

    agent_path: str
    turn_number: int
    tool_call: ToolCall
    usage_so_far: Usage
    kind: Literal["tool_called"] = "tool_called"


@dataclass(frozen=True)
class ToolDispatched:
    """Report one dispatched tool call.

    reported_usage uses Usage in the handled outcome's app_data, or ZERO_USAGE, and enters usage_so_far.
    """

    agent_path: str
    turn_number: int
    tool_call: ToolCall
    tool_message: ToolMessage
    reported_usage: Usage
    usage_so_far: Usage
    kind: Literal["tool_dispatched"] = "tool_dispatched"


@dataclass(frozen=True)
class ToolProgress:
    """Report tool progress through the current run's GuiEmitter."""

    agent_path: str
    tool_name: str
    message: str
    kind: Literal["tool_progress"] = "tool_progress"


@dataclass(frozen=True)
class LlmInputTimedOut:
    """Report an input that exceeded config.generate_one_timeout_seconds.

    usage_so_far includes billing reported before the timeout.
    The next turn reuses the unchanged messages.
    """

    agent_path: str
    turn_number: int
    usage_so_far: Usage
    kind: Literal["llm_input_timed_out"] = "llm_input_timed_out"


@dataclass(frozen=True)
class AgentFinished:
    """Report a run's answer and total usage."""

    agent_path: str
    answer: str
    usage: Usage
    kind: Literal["agent_finished"] = "agent_finished"


@dataclass(frozen=True)
class AgentFailed:
    """Report a run's failure and settled usage."""

    agent_path: str
    error_text: str
    usage: Usage
    kind: Literal["agent_failed"] = "agent_failed"


@dataclass(frozen=True)
class AgentCancelled:
    """Report a run's cancellation and settled usage."""

    agent_path: str
    usage: Usage
    kind: Literal["agent_cancelled"] = "agent_cancelled"


type Event = (
    AgentStarted
    | TurnStarted
    | GenerationReported
    | ToolCalled
    | ToolDispatched
    | ToolProgress
    | LlmInputTimedOut
    | AgentFinished
    | AgentFailed
    | AgentCancelled
)


@dataclass(frozen=True)
class GuiEmitter:
    """Route progress to the current run's on_event callback."""

    agent_path: str
    on_event: Callable[[Event], None]

    def emit_tool_progress(self, *, tool_name: str, message: str) -> None:
        """Emit ToolProgress with this run's agent_path."""
        self.on_event(
            ToolProgress(agent_path=self.agent_path, tool_name=tool_name, message=message)
        )


gui_emitter_var: ContextVar[GuiEmitter] = ContextVar("full_app_gui_emitter")
"""Store the current run's GuiEmitter."""


def current_gui_emitter() -> GuiEmitter:
    """Return the current run's GuiEmitter.

    Raises:
        LookupError: No `GuiEmitter` is active.
    """
    try:
        return gui_emitter_var.get()
    except LookupError:
        raise LookupError("No GuiEmitter is active.") from None
