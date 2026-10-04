# langchaint

langchaint is an opinionated, provider-neutral Python client for LLM applications.
It provides fully typed, asynchronous APIs for generation, streaming, embeddings, tools, retries, and billing.
The application owns the agent loop.

Alpha: the API may change without notice.

## Why langchaint

- **Consistent API.** Bind request fields once with `LLM.bind()`, then call `generate_one()`, `generate_many()`, or `stream_one()` on the resulting `BoundLLM`.
- **Output types determined by binding.** Binding `response_format=Answer` gives `generate_one()` the return type `GenerationWithoutToolCalls[Answer]`. Binding `tools` adds `GenerationWithToolCalls` to the return type of any binding.
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
    assistant = (
        OpenAI()
        .model("gpt-5.6-terra")
        .bind(
            system_prompt="Answer clearly and concisely.",
            response_format=Answer,
        )
    )
    generation = await assistant.generate_one("Why is the sky blue?")

    print(generation.output.answer)
    print(generation.usage.cost_in_usd)


asyncio.run(main())
```

The Pydantic model validates the provider response.

`generate_many()` returns one outcome per input in input order.
A terminal failure becomes that input's `GenerationError`, so sibling outcomes remain available.

## Coordinate retries across a rate-limit quota

Create one `SharedBackoff` for each rate-limit quota, and pass it to every backend that sends requests against that quota. Models from one `OpenAI` share its `SharedBackoff`:

```python
from langchaint import SharedBackoff

openai = OpenAI(
    shared_backoff=SharedBackoff(max_concurrent_requests=8, max_request_starts_per_second=50.0),
)

fast_model = openai.model("gpt-5.6-luna")
strong_model = openai.model("gpt-5.6-sol")
```

A rate-limit response pauses request starts across the shared quota.
After a transient failure local to one request, langchaint waits and retries that request.

## Stream with an explicit lifetime

```python
text_assistant = OpenAI().model("gpt-5.6-terra").bind()

async with text_assistant.stream_one("Explain photosynthesis.") as stream:
    async for item in stream:
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
        case "with_tool_calls":
            messages.append(generation.assistant_message)
            outcomes = await bound.tool_manager.dispatch_many(generation.tool_calls)
            messages.extend(outcome.tool_message for outcome in outcomes)
        case "without_tool_calls":
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
