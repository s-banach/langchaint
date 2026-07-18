"""Telemetry: wrap an LLM in TracedLLM and every generate call becomes an OTel span.

Wrapping is unconditional; enabling, disabling, or routing tracing is OTel SDK configuration, never application code.
An app that never configures a TracerProvider gets non-recording no-op spans,
so the wrapper is free when tracing is off.
TracedLLM mirrors bind and rebind, so a rebound object stays traced.
The default "gen_ai" mapper emits GenAI-convention attributes plus langchaint scalars (cost, attempts, the cache
partition); the mapper never receives the conversation, so no built-in mapper can leak a prompt.

Install: opentelemetry-api, which the tracing subpackage imports.
Exporting spans additionally needs opentelemetry-sdk (shown below); a production app wires the SDK it already runs.

The LangChain call-for-call map (LangSmith, callbacks, LANGCHAIN_TRACING_V2) lives in MIGRATING_FROM_LANGCHAIN.md.
"""

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from langchaint.openai import openai_model
from langchaint.tracing import TracedLLM


def configure_otel() -> None:
    """Configure the OTel SDK once at process start.

    ConsoleSpanExporter prints spans to stdout; swap in an OTLPSpanExporter to send them to a collector.
    This is the only tracing-related code that is not langchaint: langchaint emits spans, the SDK routes them.
    """
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


async def traced_generate() -> None:
    """Wrap the LLM once, then use it exactly like an untraced one.

    TracedLLM.bind returns a TracedBoundLLM whose generate_one opens a CLIENT span around the call,
    sets cost and token attributes from the Response, and records one event per failed retry attempt.
    A GenerationError sets error status on the span and re-raises;
    the paid result is never discarded by a telemetry bug.
    """
    traced = TracedLLM(openai_model("gpt-5.6-terra"))
    assistant = traced.bind(system_prompt="Answer in one sentence.", automatic_prompt_caching=False)
    response = await assistant.generate_one("What is the capital of Japan?")
    print(response.output)
    # Streaming is traced too: traced_bound.stream_one(...) returns a TracedStreamHandle that records
    # langchaint.time_to_first_token_seconds and closes its span on a failing or abandoned stream.


async def main() -> None:
    """Configure the SDK, then run one traced generate; watch the span print to stdout."""
    configure_otel()
    await traced_generate()


if __name__ == "__main__":
    asyncio.run(main())
