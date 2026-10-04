"""Print OTel spans for a tool loop.

Besides OTel SDK configuration, tracing changes only the backend construction.
The tool loop is the same code an untraced application runs.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel

from langchaint import Message, UserMessage, tool
from langchaint.openai import OpenAI
from langchaint.tracing import OtelObserver


class WeatherArgs(BaseModel):
    """Select the city for a weather report."""

    city: str


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> str:
    """Return a fixed weather report."""
    return f"It is 18C and clear in {args.city}."


async def print_tool_loop_spans(prompt: str, max_turns: int = 10) -> str:
    """Run a tool loop and print one span per input and per tool dispatch.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
        DispatchExceptionGroup: A tool function raises.
        RuntimeError: The model exceeded `max_turns`.
    """
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    openai = OpenAI(
        observer=OtelObserver(capture_message_content=False, tracer_provider=tracer_provider)
    )
    bound = openai.model("gpt-5.6-terra").bind(
        system_prompt="Use tools when needed.",
        tools=[get_weather],
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        generation = await bound.generate_one(messages)
        match generation.kind:
            case "with_tool_calls":
                messages.append(generation.assistant_message)
                outcomes = await bound.tool_manager.dispatch_many(generation.tool_calls)
                messages.extend(outcome.tool_message for outcome in outcomes)
            case "without_tool_calls":
                return generation.output
    raise RuntimeError(f"model did not finish within {max_turns} turns")
