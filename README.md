# langchaint

langchaint is a provider-neutral Python client for LLM applications.
It provides fully typed, asynchronous APIs for generation, streaming, embeddings, tools, retries, and billing.
The application owns the agent loop.

Alpha: the API may change without notice.

## Why langchaint

- **Consistent API.** Configure a model with a provider factory, then call `generate_one()`, `generate_many()`, or `stream_one()` through the `LLM` protocol.
- **Useful type annotations.** For example, `openai.llm(model, response_format=Answer)` types `generation.output` as `Answer`.
- **Outcome variants with autocomplete.** Match on `.kind` with editor autocomplete and no class imports.
- **Coordinated retries.** Share concurrency limits, request-start pacing, and provider-directed pauses across models using one rate-limit quota.
- **Complete billing.** `Generation` and `GenerationError` values retain provider-reported usage from every recorded request, including billed retries.
- **Streaming.** `stream_one()` returns an async context manager and async iterator. `final()` returns the typed generation with its usage.
- **Agent loops in Python.** Provider-neutral messages, typed tools with argument validation, concurrent dispatch, and explicit outcome variants support async control flow.

## Install

langchaint requires Python 3.13 or newer.

Install the extra for each backend you use:

```bash
pip install "langchaint[openai]"
```

| Backend | Class | Install |
| --- | --- | --- |
| Anthropic | `Anthropic` | `langchaint[anthropic]` |
| Anthropic on Amazon Bedrock | `AnthropicBedrock` | `langchaint[anthropic-bedrock]` |
| Cohere embeddings on Amazon Bedrock | `CohereBedrock` | `langchaint[cohere-bedrock]` |
| DeepSeek | `DeepSeek` | `langchaint[deepseek]` |
| Gemini | `Gemini` | `langchaint[gemini]` |
| OpenAI | `OpenAI` | `langchaint[openai]` |
| OpenAI embeddings | `OpenAI` | `langchaint[openai-embedding]` |
| OpenAI on Amazon Bedrock | `OpenAIBedrock` | `langchaint[openai-bedrock]` |

Install `langchaint[tracing]` for OpenTelemetry tracing.

## Generate a typed output

```python
import asyncio

from pydantic import BaseModel

from langchaint.openai import OpenAI


class Answer(BaseModel):
    answer: str
    confidence: float


async def main() -> None:
    assistant = OpenAI().llm(
        "gpt-5.6-terra",
        system_prompt="Answer clearly and concisely.",
        response_format=Answer,
    )
    generation = await assistant.generate_one("Why is the sky blue?")

    print(generation.output.answer)
    print(generation.usage.cost_in_usd)


asyncio.run(main())
```

The Pydantic model validates the provider response.
`assistant.rebind(system_prompt="Explain in detail.")` creates another configured model and preserves omitted settings.
`Anthropic.llm` and `AnthropicBedrock.llm` require `max_completion_tokens`; OpenAI permits omission.

Adapter authors can configure their own `Adapter` with the SDK-free `langchaint.llm_from_adapter(adapter, ...)` factory.
It returns `AdapterLLM`, which provides generation and typed `rebind()` methods.
The adapter validates provider-specific constraints, including required token limits.

`langchaint.span_parsing.llm_from_otel(span, llm=assistant)` reconstructs captured request settings and preserves the concrete model class.
Configure the input model with the required output type and executable tools; reconstruction checks them against captured output and tool metadata.

Provider-neutral application functions accept `LLM[OutputT, ToolManagerT]` without requiring inheritance.
The protocol exposes typed `rebind`, `tool_manager`, and generation methods with read-only batch results.
`rebind` preserves omitted settings and infers changed output and tool types.
Its common `max_completion_tokens` parameter accepts integers because Anthropic requires a limit; concrete OpenAI, DeepSeek, and Gemini models also accept `None` to clear it.
Generation methods return `Generation[OutputT]`; match `generation.kind` before reading `.output` through this interface.
Concrete provider models retain precise plain-generation results when configured without tools.

`generate_many()` returns generations in input order, or raises `GenerationError` after cancelling and awaiting unfinished inputs.
`generate_many_records()` returns one normalized record per input, including generation failures.
Pass `resume_path` to persist records and resume interrupted batches.

## Coordinate retries across a rate-limit quota

Create one `SharedBackoff` for each rate-limit quota, and pass it to every backend that sends requests against that quota. Models from one `OpenAI` share its `SharedBackoff`:

```python
from langchaint import SharedBackoff

openai = OpenAI(
    shared_backoff=SharedBackoff(max_concurrent_requests=8, max_request_starts_per_second=50.0),
)

fast_model = openai.llm("gpt-5.6-luna")
strong_model = openai.llm("gpt-5.6-sol")
```

A rate-limit response pauses request starts across the shared quota.
After a transient failure local to one request, langchaint waits and retries that request.

## Stream with an explicit lifetime

```python
text_assistant = OpenAI().llm("gpt-5.6-terra")

async with text_assistant.stream_one("Explain photosynthesis.") as stream:
    async for item in stream:
        # Answer text chunks are str; ReasoningDelta, ToolCallDelta, and ToolCall are the other items
        if isinstance(item, str):
            print(item, end="", flush=True)

    generation = await stream.final()
```

`final()` consumes the remaining stream and returns the assembled generation.

## Build agent loops

The application controls turn limits, state, approvals, model changes, and persistence.

```python
messages: list[Message] = [UserMessage(content=prompt)]

for _ in range(max_turns):
    generation = await bound.generate_one(messages)

    match generation.kind:
        case "tool_call":
            messages.append(generation.assistant_message)
            outcomes = await bound.tool_manager.dispatch_many(generation.tool_calls)
            messages.extend(outcome.tool_message for outcome in outcomes)
        case "plain":
            return generation.output

raise RuntimeError("model did not finish within max_turns")
```

`ToolManager.dispatch_many()` runs tool calls concurrently and preserves their order.

See [`examples/02_tool_loop.py`](examples/02_tool_loop.py) for a complete typed tool loop.

## Account for every request

`generation.usage.cost_in_usd` includes every billed retry recorded for the input.
`GenerationError.usage` preserves the recorded cost of a failed input.

## More examples

See [`examples/README.md`](examples/README.md) for complete examples.

## License

[MIT License](LICENSE)
