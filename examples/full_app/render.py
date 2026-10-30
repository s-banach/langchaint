"""Render each event as one line indented by agent_path depth."""

from events import Event


def render(event: Event) -> str:
    """Render one event indented by agent_path depth."""
    match event.kind:
        case "agent_started":
            body = f"* {event.agent_path} started"
        case "agent_finished":
            body = f"* {event.agent_path} done, spent ${event.usage.cost_in_usd:.4f}"
        case "agent_failed":
            body = f"! {event.agent_path} FAILED ({event.error_text[:40]}), spent ${event.usage.cost_in_usd:.4f}"
        case "agent_cancelled":
            body = f"! {event.agent_path} CANCELLED, settled spend ${event.usage.cost_in_usd:.4f}"
        case "tool_progress":
            body = f"  .. {event.tool_name}: {event.message}"
        case "turn_started":
            usage = event.usage_so_far
            body = f"  turn {event.turn_number} begins ({usage.input_tokens_total}in/{usage.output_tokens}out ${usage.cost_in_usd:.4f})"
        case "generation_reported":
            usage = event.usage_so_far
            text = "\n".join(
                text
                for text in (event.generation.commentary, event.generation.output)
                if text is not None
            )
            shown = text or "(no text)"
            body = f"  llm t{event.turn_number}: {shown[:40]!r} -> {usage.input_tokens_total}in/{usage.output_tokens}out ${usage.cost_in_usd:.4f}"
        case "llm_input_timed_out":
            body = f"  llm t{event.turn_number}: TIMED OUT past the per-input deadline -> ${event.usage_so_far.cost_in_usd:.4f}"
        case "tool_called":
            body = f"  -> {event.tool_call.name}({event.tool_call.args_json})"
        case "tool_dispatched":
            mark = "ERR" if event.tool_message.is_error else "ok"
            reported_usage = event.reported_usage
            fee = f" +${reported_usage.cost_in_usd:.4f}" if reported_usage.cost_in_usd else ""
            content = event.tool_message.content
            text = (
                content
                if isinstance(content, str)
                else " ".join(part.text for part in content if part.kind == "text")
            )
            body = f"  <- {event.tool_call.name} {mark}: {text[:38]!r}{fee} -> ${event.usage_so_far.cost_in_usd:.4f}"
    indent = "  " * event.agent_path.count("/")
    return f"{indent}{body}"
