"""The console renderer standing in for a UI.

One line per event, indented by the emitting run's depth in the agent tree. Everything it draws comes off
the event itself, which is the property the event vocabulary exists to have: a UI never looks back into
the orchestrator to redraw a token counter.
"""

from events import (
    AgentFailed,
    AgentFinished,
    AgentStarted,
    Event,
    LlmCallAbandoned,
    LlmResponse,
    ToolCalled,
    ToolProgress,
    ToolResponse,
    TurnStarted,
)


def render(event: Event) -> str:
    """Draw one event as a single line, indented by its depth in the agent tree."""
    indent = "  " * (event.agent_path.count("/"))
    return f"{indent}{_render_body(event)}"


def _render_body(event: Event) -> str:
    match event:
        case AgentStarted(agent_path=path):
            return f"* {path} started"
        case AgentFinished(agent_path=path, usage=usage):
            return f"* {path} done, spent ${usage.cost_in_usd:.4f}"
        case AgentFailed(agent_path=path, error=error, usage=usage):
            return f"! {path} FAILED ({error[:40]}), spent ${usage.cost_in_usd:.4f}"
        case ToolProgress(tool_name=name, message=message):
            return f"  .. {name}: {message}"
        case _:
            return _render_activity(event)


def _render_activity(event: Event) -> str:
    match event:
        case TurnStarted(turn_number=turn, usage_so_far=usage):
            return f"  turn {turn} begins ({usage.input_tokens_total}in/{usage.output_tokens}out ${usage.cost_in_usd:.4f})"
        case LlmResponse(turn_number=turn, text=text, usage_so_far=usage):
            shown = text or "(tool calls only)"
            return f"  llm t{turn}: {shown[:40]!r} -> {usage.input_tokens_total}in/{usage.output_tokens}out ${usage.cost_in_usd:.4f}"
        case LlmCallAbandoned(turn_number=turn, usage_so_far=usage):
            return (
                f"  llm t{turn}: ABANDONED past the per-call deadline -> ${usage.cost_in_usd:.4f}"
            )
        case ToolCalled(tool_name=name, args_json=args):
            return f"  -> {name}({args})"
        case ToolResponse(
            tool_name=name,
            content=content,
            is_error=is_error,
            reported_usage=reported,
            usage_so_far=usage,
        ):
            mark = "ERR" if is_error else "ok"
            fee = f" +${reported.cost_in_usd:.4f}" if reported.cost_in_usd else ""
            return f"  <- {name} {mark}: {content[:38]!r}{fee} -> ${usage.cost_in_usd:.4f}"
        case _:
            return f"? {event}"
