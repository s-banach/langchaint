"""The UI event vocabulary, and the ambient emitter that lets a tool function join it.

One frozen dataclass per event, matched by type; agent_path names the run that emitted it
("root", "root/research_climate", "root/research_climate/specialist#0"),
which is what lets a UI nest a sub-agent's events under the tool call that started it.
A tool-spawned run carries a spawn index after "#", because an agent's name is not unique within a
parent and agent_path is the one identity an event carries: without it, two spawns of one name would
interleave indistinguishably in a UI.

Every event a run emits after its first generate carries usage_so_far, that run's running total,
so a UI redraws one agent's token counts from the event alone with no lookup into the orchestrator.
usage_so_far includes spend reported by tools the run called, sub-agent runs included,
so a parent's total is always the whole subtree beneath it. ToolProgress is the exception: a tool
function does not know its run's total, so the event carries none and a UI leaves the counter alone.

GuiEmitter and current_gui_emitter live here, in the lowest shared module, so a tool function reaches
its run's on_event without importing the run machinery, which itself imports the tools.

describe_error and content_text live here rather than in the run machinery, because they render the
strings two of these events carry: AgentFailed.error and ToolResponse.content.
"""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

from langchaint import TextPart, ToolMessage, Usage


def describe_error(error: BaseException) -> str:
    """Render an exception for a UI, keeping the type when the message is empty.

    TimeoutError carries no message, so str() alone would show a failure with no stated cause.
    """
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
    """A run began; agent_path's last segment is the run's own name plus any spawn index."""

    agent_path: str


@dataclass(frozen=True)
class TurnStarted:
    """A run is about to send its next generate call.

    turn_number counts from 1 and is the same number max_turns bounds,
    so a UI showing "turn 3/10" reads both from here.
    """

    agent_path: str
    turn_number: int
    usage_so_far: Usage


@dataclass(frozen=True)
class LlmResponse:
    """One generate call returned, which is where a UI refreshes the run's token counts.

    text is the assistant text, empty on a pure tool-call turn.
    usage is that one call's billing; usage_so_far is the run's total with this call already folded in,
    which is the number a per-agent token counter redraws to.
    """

    agent_path: str
    turn_number: int
    text: str
    usage: Usage
    usage_so_far: Usage


@dataclass(frozen=True)
class ToolCalled:
    """The model requested a tool, emitted before dispatch so a UI shows the call while it runs.

    args_json is the model's raw argument text, unvalidated: it is what the model actually sent,
    including the malformed text behind a DispatchInvalidToolArgs.
    tool_call_id ties this event to the ToolResponse that settles it.
    """

    agent_path: str
    turn_number: int
    tool_call_id: str
    tool_name: str
    args_json: str
    usage_so_far: Usage


@dataclass(frozen=True)
class ToolResponse:
    """One tool dispatch settled, which is where a UI marks the call done and shows its result.

    tool_call_id matches the ToolCalled that opened it.
    content is what the model will read; is_error marks a failed, declined, or misrouted call.
    reported_usage is spend the tool itself reported through app_data, ZERO_USAGE when it reported none;
    it is already folded into usage_so_far. A sub-agent call reports none, because a sub-run records its
    own spend as it goes, so its cost shows up as a jump in usage_so_far and not here.
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
    """A tool function reported progress mid-call, through the run's ambient GuiEmitter.

    Emitted between a ToolCalled and its ToolResponse, so a UI shows a long call moving rather than a
    spinner. agent_path is the run whose dispatch is executing the tool, stamped by
    GuiEmitter.emit_tool_progress, so the same tool function reports to whichever run called it.
    """

    agent_path: str
    tool_name: str
    message: str


@dataclass(frozen=True)
class LlmCallAbandoned:
    """One generate call outran config.per_call_timeout_seconds and was dropped; the run continues.

    The request may have completed and billed server-side, so its spend is unobservable: the
    AbandonedCall langchaint appended to the run's turn_log records the drop, and usage_so_far
    gains only what the call's settled attempts had already reported.
    turn_number is the turn whose call was dropped; the next turn resends from the same conversation,
    which the dropped call left untouched.
    """

    agent_path: str
    turn_number: int
    usage_so_far: Usage


@dataclass(frozen=True)
class AgentFinished:
    """A run produced its answer; usage is that run's whole spend, sub-runs included."""

    agent_path: str
    answer: str
    usage: Usage


@dataclass(frozen=True)
class AgentFailed:
    """A run ended in an error; usage is what it had spent when the error landed.

    A timeout lands here too, which is the case that makes usage-on-failure worth carrying at all.
    """

    agent_path: str
    error: str
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
)


@dataclass(frozen=True)
class GuiEmitter:
    """The identity and on_event of the run that is executing right now, held without a reference to the run.

    One emitter per run, installed in a contextvar by that run's final() for its duration, so code
    reached from anywhere inside the run's loop sends to the right on_event by asking rather than by
    being handed one. A tool function is the case that needs it: the function holds no handle on the
    run that dispatched it, and no amount of threading gets one in without a parameter on every tool.
    """

    agent_path: str
    on_event: Callable[[Event], None]

    def emit_tool_progress(self, *, tool_name: str, message: str) -> None:
        """Emit a ToolProgress stamped with this run's agent_path; on_event runs in this frame.

        The stamp lives here so a tool function cannot misfile its progress under another run's path.
        """
        self.on_event(
            ToolProgress(agent_path=self.agent_path, tool_name=tool_name, message=message)
        )


gui_emitter_var: ContextVar[GuiEmitter] = ContextVar("full_app_gui_emitter")
"""Set by each run's final() and reset on its way out, so nesting restores the parent's emitter.

asyncio copies the context at task creation, so a task created inside the run (a tool dispatch)
inherits the run's emitter, for the same reason OTel spans nest across tasks; a sub-run started
inside such a task installs its own without disturbing the parent's.
"""


def current_gui_emitter() -> GuiEmitter:
    """Return the emitter of the run this call is running inside.

    Raises:
        LookupError: no run is current, which means this code is not reached from inside a run's loop.
            Emitting from outside a run has no on_event to go to, so it is an error rather than a
            silent drop, and the default LookupError names only a variable, so it is replaced with one
            naming what to install and where.
    """
    try:
        return gui_emitter_var.get()
    except LookupError:
        raise LookupError(
            "no GuiEmitter in context: emit is only meaningful inside a run's loop, where "
            "AgentRun.final installs one for that run."
        ) from None
