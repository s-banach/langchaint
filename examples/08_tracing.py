"""Export traced generation and tool-dispatch spans to standard output."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel

from langchaint import Message, UserMessage, tool
from langchaint.openai import OpenAI
from langchaint.tracing import TracedLLM


class WeatherArgs(BaseModel):
    """Select the city for a weather report."""

    city: str


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> str:
    """Return a fixed weather report."""
    return f"It is 18C and clear in {args.city}."


async def traced_tool_loop(prompt: str, max_turns: int = 10) -> str:
    """Trace generation and dispatch until the model returns text.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: A `generate_one` call failed.
        DispatchExceptionGroup: A tool function raised.
        RuntimeError: The model exceeded `max_turns`.
    """
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = tracer_provider.get_tracer("langchaint.example")

    openai = OpenAI()
    traced = TracedLLM(
        openai.model("gpt-5.6-terra"),
        capture_message_content=False,
        tracer=tracer,
    )
    bound = traced.bind(
        system_prompt="Use tools when needed.",
        tools=[get_weather],
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    for _ in range(max_turns):
        response = await bound.generate_one(messages)
        messages.append(response.assistant_message)
        if not response.tool_calls:
            return response.output
        outcomes = await bound.tool_manager.dispatch_many(response.tool_calls)
        messages.extend(outcome.tool_message for outcome in outcomes)
    raise RuntimeError(f"model did not finish within {max_turns} turns")
