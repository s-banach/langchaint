"""Telemetry: wrap an LLM in TracedLLM and every generate call becomes an OTel span.

Wrapping is unconditional; enabling, disabling, or routing tracing is OTel SDK configuration, never application code.
An app that never configures a TracerProvider gets non-recording no-op spans,
so the wrapper is free when tracing is off.
TracedLLM mirrors bind and rebind, so a rebound object stays traced.
The default mapper, gen_ai_attributes, emits GenAI-convention attributes (token counts, cache counters, finish reason)
plus the two langchaint scalars the convention has no counterpart for, cost and attempts;
no mapper receives the conversation, so gen_ai_attributes cannot put a prompt on a span,
while a custom mapper reads whatever it reaches on the result, raw included.

capture_message_content decides separately whether the spans carry the conversation itself.
It is required and has no default, because recording prompts is a privacy choice langchaint never makes for you.
False below: the spans carry metrics and no message content.
The tracing module docstring lists every attribute each span kind emits under either value.

Install: opentelemetry-api, which the tracing subpackage imports.
Exporting spans additionally needs opentelemetry-sdk (shown below); a production app wires the SDK it already runs.

The LangChain call-for-call map (LangSmith, callbacks, LANGCHAIN_TRACING_V2) lives in MIGRATING_FROM_LANGCHAIN.md.
"""

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import BaseModel

from langchaint import PydanticTool, ToolCall
from langchaint.openai import openai_model
from langchaint.tracing import TracedLLM, TracedToolManager


def configure_otel() -> None:
    """Configure the OTel SDK once at process start.

    ConsoleSpanExporter prints spans to stdout; swap in an OTLPSpanExporter to send them to a collector.
    This is the only tracing-related code that is not langchaint: langchaint emits spans, the SDK routes them.
    """
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)


async def traced_generate() -> None:
    """Wrap the LLM once, then use it exactly like an untraced one.

    TracedLLM.bind returns a TracedBoundLLM whose generate_one opens a CLIENT span around the call,
    sets cost and token attributes from the Response, and records one event per failed retry attempt.
    A GenerationError sets error status on the span and re-raises;
    the paid result is never discarded by a telemetry bug.
    """
    traced = TracedLLM(openai_model("gpt-5.6-terra"), capture_message_content=False)
    assistant = traced.bind(
        system_prompt="Answer in one sentence.", automatic_prompt_caching=False
    )
    response = await assistant.generate_one("What is the capital of Japan?")
    print(response.output)
    # Streaming is traced too: assistant.stream_one(...) returns a TracedStreamHandle that records
    # gen_ai.response.time_to_first_chunk and closes its span on a failing or abandoned stream.


class CityArgs(BaseModel):
    """Arguments of the lookup tool."""

    city: str


async def lookup(args: CityArgs) -> str:
    """Return a fixed population figure, standing in for a real lookup."""
    return f"{args.city}: 13,960,000"


async def traced_tool_dispatch() -> None:
    """Dispatch one tool call through a TracedToolManager, which opens one execute_tool span per call.

    TracedToolManager is a ToolManager subclass, so it passes to bind's tool_manager parameter as one object
    and the application's own tool loop calls dispatch on it unchanged.
    It takes its own capture_message_content, inheriting nothing from TracedLLM,
    because the application constructs it rather than a TracedLLM handing it down.

    The span's status and error.type come from the outcome, and the two values meaning the tool function
    never ran are invalid_tool_args and unknown_tool.
    The call below dispatches a name the manager does not hold, so its span carries error.type unknown_tool
    and ERROR status: a model calling a tool that is not there is designed control flow here
    (the model reads the correction and retries), so ERROR describes that operation's outcome,
    not the health of the system.
    """
    tool_manager = TracedToolManager(
        [
            PydanticTool(
                name="lookup",
                description="Look up a city's population",
                args_model=CityArgs,
                function=lookup,
            )
        ],
        capture_message_content=False,
    )
    handled = await tool_manager.dispatch(
        ToolCall(id="call1", name="lookup", args_json='{"city": "Tokyo"}')
    )
    print(handled.tool_message.content)
    unknown = await tool_manager.dispatch(ToolCall(id="call2", name="missing", args_json="{}"))
    print(unknown.tool_message.content)


async def main() -> None:
    """Configure the SDK, then run one traced generate and one traced dispatch; the spans print to stdout."""
    configure_otel()
    await traced_generate()
    await traced_tool_dispatch()


if __name__ == "__main__":
    asyncio.run(main())
