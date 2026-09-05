# langchaint

langchaint is an opinionated, provider-neutral Python client for LLM applications.
It provides fully typed, asynchronous APIs for generation, streaming, embeddings, tools, retries, and billing.
The application owns the agent loop.

Alpha: the API may change without notice.

## Why langchaint

- **Consistent API.** Bind request fields once with `LLM.bind()`, then call `generate_one()`, `generate_many()`, or `stream_one()` on the resulting `BoundLLM`.
- **Output types determined by binding.** Binding `response_format=Answer` gives `generate_one()` the return type `Response[Answer]`. Binding `tools` as well adds `ToolCallTurn[Answer]` to that return type.
- **Result variants with autocomplete.** Match on `.kind` with editor autocomplete and no class imports.
- **Coordinated retries.** Share concurrency limits, request-start pacing, and provider-directed pauses across models using one rate-limit quota.
- **Complete billing.** Successful results and `GenerationError` values retain provider-reported usage from every recorded attempt, including billed retries.
- **Streaming.** `stream_one()` returns an async context manager and async iterator. `final()` returns the typed result with its usage.
- **Agent loops in Python.** Provider-neutral messages, typed tools with argument validation, concurrent dispatch, and explicit result variants support async control flow.

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

## Generate a typed response

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
    response = await assistant.generate_one("Why is the sky blue?")

    print(response.output.answer)
    print(response.usage.cost_in_usd)


asyncio.run(main())
```

The Pydantic model validates the provider response.

`generate_many()` returns one result per input in input order.
A terminal failure becomes that input's `GenerationError`, so sibling results remain available.

## Coordinate retries across a rate-limit quota

Create one `OpenAI` for each rate-limit quota:

```python
openai = OpenAI(
    max_concurrent_requests=8,
    max_request_starts_per_second=50.0,
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

    response = await stream.final()
```

`final()` consumes the remaining stream and returns the assembled result.

## Build agent loops

The application controls turn limits, state, approvals, model changes, and persistence.

```python
messages: list[Message] = [UserMessage(content=prompt)]

for _ in range(max_turns):
    result = await bound.generate_one(messages)

    match result.kind:
        case "tool_call_turn":
            messages.append(result.assistant_message)
            outcomes = await bound.tool_manager.dispatch_many(result.tool_calls)
            messages.extend(outcome.tool_message for outcome in outcomes)
        case "response":
            return result.output

raise RuntimeError("model did not finish within max_turns")
```

`ToolManager.dispatch_many()` runs tool calls concurrently and preserves their order.

See [`examples/02_tool_loop.py`](examples/02_tool_loop.py) for a complete typed tool loop.

## Account for the complete call

`response.usage.cost_in_usd` includes every billed retry recorded for the call.
`GenerationError.usage` preserves the recorded cost of failed calls.

## More examples

See [`examples/README.md`](examples/README.md) for complete examples.

## License

langchaint uses the [MIT License](LICENSE).
