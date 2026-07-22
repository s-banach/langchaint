"""The UI event vocabulary, and the ambient emitter that lets a tool function join it.

One frozen dataclass per event, matched by type; agent_path names the run that emitted it
("root", "root/research_climate", "root/research_climate/specialist"),
which is what lets a UI nest a sub-agent's events under the tool call that started it.

Every event a run emits after its first generate carries usage_so_far, that run's running total,
so a UI redraws one agent's token counts from the event alone with no lookup into the orchestrator.
usage_so_far includes spend reported by tools the run called, sub-agent runs included,
so a parent's total is always the whole subtree beneath it. ToolProgress is the exception: a tool
function does not know its run's total, so the event carries none and a UI leaves the counter alone.

GuiEmitter and current_gui_emitter live here, in the lowest shared module, so a tool function reaches
its run's stream without importing the run machinery, which itself imports the tools.

describe_error and content_text live here rather than in the run machinery, because they render the
strings two of these events carry: AgentFailed.error and ToolResponse.content.
"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass

from langchaint import ToolMessage, Usage


def describe_error(error: BaseException) -> str:
    """Render an exception for a UI, keeping the type when the message is empty.

    TimeoutError carries no message, so str() alone would show a failure with no stated cause.
    """
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def content_text(message: ToolMessage) -> str:
    """Render a tool message's content as the plain text a UI shows."""
    content = message.content
    return (
        content
        if isinstance(content, str)
        else " ".join(getattr(part, "text", "") for part in content)
    )


@dataclass(frozen=True)
class AgentStarted:
    """A run began; agent_path's last segment is the run's own name."""

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
    spinner. agent_path is the run whose dispatch is executing the tool, read off the emitter rather
    than passed in, so the same tool function reports to whichever run called it.
    """

    agent_path: str
    tool_name: str
    message: str


@dataclass(frozen=True)
class LlmCallAbandoned:
    """One generate call outran config.per_call_timeout_seconds and was dropped; the run continues.

    The request may have completed and billed server-side, so its spend is unobservable: the
    AbandonedCall langchaint appended to the run's spend_log records the drop, and usage_so_far
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


class GuiEmitter:
    """The stream of the run that is executing right now, held without a reference to the run.

    One emitter per run, installed in a contextvar by that run's own task, so code reached from
    anywhere inside the run's loop sends to the right stream by asking rather than by being handed one.
    A tool function is the case that needs it: the function holds no handle on the run that dispatched
    it, and no amount of threading gets one in without a parameter on every tool.
    """

    def __init__(self, agent_path: str, queue: asyncio.Queue[Event | None]) -> None:
        """Bind this emitter to one run's stream."""
        self.agent_path = agent_path
        self._queue = queue

    def emit(self, event: Event) -> None:
        """Push one event onto the bound run's queue; never suspends and never raises."""
        self._queue.put_nowait(event)


gui_emitter_var: ContextVar[GuiEmitter] = ContextVar("full_app_gui_emitter")
"""Set by each run's own task, so everything the loop reaches inherits that run's emitter.

asyncio copies the context at task creation, which is what scopes the value: a sub-run's task sets its
own emitter without disturbing the parent's, for the same reason OTel spans nest across tasks.
"""


def current_gui_emitter() -> GuiEmitter:
    """Return the emitter of the run this call is running inside.

    Raises:
        LookupError: no run is current, which means this code is not reached from inside a run's loop.
            Emitting from outside a run has no stream to go to, so it is an error rather than a silent
            drop, and the default LookupError names only a variable, so it is replaced with one naming
            what to install and where.
    """
    try:
        return gui_emitter_var.get()
    except LookupError:
        raise LookupError(
            "no GuiEmitter in context: emit is only meaningful inside a run's loop, where "
            "AgentRun._driven installs one for that run."
        ) from None
