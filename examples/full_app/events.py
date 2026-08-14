"""Define UI events and route tool progress to the current run.

agent_path identifies the emitting run and includes a spawn index for tool-spawned runs.
usage_so_far and terminal usage include sub-agent spend.
LlmResponse.usage covers one generate_one call.
"""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

from langchaint import TextPart, ToolMessage, Usage


def describe_error(error: BaseException) -> str:
    """Render an exception with its type and optional message."""
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def content_text(message: ToolMessage) -> str:
    """Render a tool message's content as the plain text a UI shows."""
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(part.text for part in content if isinstance(part, TextPart))


@dataclass(frozen=True)
class AgentStarted:
    """Report that a run began."""

    agent_path: str


@dataclass(frozen=True)
class TurnStarted:
    """Report the start of a numbered turn."""

    agent_path: str
    turn_number: int
    usage_so_far: Usage


@dataclass(frozen=True)
class LlmResponse:
    """Report one generate response and updated usage.

    text is empty for a response containing only tool calls.
    usage covers this call, and usage_so_far covers the run.
    """

    agent_path: str
    turn_number: int
    text: str
    usage: Usage
    usage_so_far: Usage


@dataclass(frozen=True)
class ToolCalled:
    """Report a tool request before dispatch.

    args_json preserves the model's unvalidated argument text.
    tool_call_id links the request to its ToolResponse.
    """

    agent_path: str
    turn_number: int
    tool_call_id: str
    tool_name: str
    args_json: str
    usage_so_far: Usage


@dataclass(frozen=True)
class ToolResponse:
    """Report one settled tool request.

    tool_call_id links this event to ToolCalled.
    content is the model-facing result.
    is_error marks failed, declined, and misrouted calls.
    reported_usage is the Usage from DispatchHandled.app_data, or ZERO_USAGE otherwise, and is included in usage_so_far.
    """

    agent_path: str
    turn_number: int
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool
    reported_usage: Usage
    usage_so_far: Usage


@dataclass(frozen=True)
class ToolProgress:
    """Report tool progress through the current run's GuiEmitter."""

    agent_path: str
    tool_name: str
    message: str


@dataclass(frozen=True)
class LlmCallAbandoned:
    """Report a call that exceeded config.generate_one_timeout_seconds.

    usage_so_far includes billing reported before the timeout.
    The next turn reuses the unchanged messages.
    """

    agent_path: str
    turn_number: int
    usage_so_far: Usage


@dataclass(frozen=True)
class AgentFinished:
    """Report a run's answer and total usage."""

    agent_path: str
    answer: str
    usage: Usage


@dataclass(frozen=True)
class AgentFailed:
    """Report a run's failure and settled usage."""

    agent_path: str
    error: str
    usage: Usage


@dataclass(frozen=True)
class AgentCancelled:
    """Report a run's cancellation and settled usage."""

    agent_path: str
    usage: Usage


type Event = (
    AgentStarted
    | TurnStarted
    | LlmResponse
    | ToolCalled
    | ToolResponse
    | ToolProgress
    | LlmCallAbandoned
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
        LookupError: no run is current.
    """
    try:
        return gui_emitter_var.get()
    except LookupError:
        raise LookupError(
            "no GuiEmitter in context: emit is only meaningful inside a run's loop, where "
            "AgentRun.final installs one for that run."
        ) from None
