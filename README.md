# langchaint

langchaint is a provider-neutral Python client for LLM applications, with typed, asynchronous APIs for generation, streaming, embeddings, and tool dispatch.

Alpha: the API may change without notice.

## Why langchaint

- **Types follow configuration:** typed output and tool-call variants, narrowed by `.kind`.
- **Agent loops in ordinary Python:** your code controls state, approvals, model changes, recovery, stopping, and persistence.
- **Shared rate limiting:** concurrency limits, request pacing, and rate-limit pauses across your application.
- **Usage accounting:** available provider-reported usage across every recorded request, including retries and failures.

## Install

Requires Python 3.13+.
Install each backend's extra:

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

`response_format=Answer` validates the response with Pydantic and types `generation.output` as `Answer`.
`bind(response_format=...)` returns a new model with the corresponding output type.

`generate_many()` returns generations in input order, or raises `GenerationError` after cancelling and awaiting unfinished inputs.
`generate_many_records()` returns one normalized record per input, including failures; pass `resume_path` to persist records and resume interrupted batches.

To configure an adapter directly, use `llm_from_adapter(adapter, ...)` from `langchaint`.

## Build agent loops

Configuring tools adds a tool-call variant to the generation's discriminated union.
Match `.kind` to narrow its type, and set turn limits in your loop:

```python
llm = OpenAI().llm("gpt-5.6-terra", tools=tools)
messages: list[Message] = [UserMessage(content=prompt)]

for _ in range(max_turns):
    generation = await llm.generate_one(messages)

    match generation.kind:
        case "tool_call":
            messages.append(generation.assistant_message)
            outcomes = await llm.tool_manager.dispatch_many(generation.tool_calls)
            messages.extend(outcome.tool_message for outcome in outcomes)
        case "plain":
            return generation.output

raise RuntimeError("model did not finish within max_turns")
```

`ToolManager.dispatch_many()` validates arguments and runs calls concurrently, preserving their order.
Invalid arguments produce an error `ToolMessage` you can return to the model.

Complete typed loop: [`examples/02_tool_loop.py`](examples/02_tool_loop.py).

## Coordinate rate limiting

`SharedBackoff` coordinates request concurrency, pacing, and rate-limit pauses across batches and separate tasks:

```python
from langchaint import SharedBackoff
from langchaint.openai import OpenAI

shared_backoff = SharedBackoff(max_concurrent_requests=8, max_request_starts_per_second=50.0)
openai = OpenAI(shared_backoff=shared_backoff)
assistant = openai.llm("gpt-5.6-terra")

generations = await assistant.generate_many(
    ["Explain photosynthesis.", "Explain cellular respiration."],
)
```

Rate limits pause further request starts against the quota.
For transient failures local to one request, langchaint waits and retries that request.

## Track costs

`generation.usage.cost_in_usd` includes every recorded billed retry; `GenerationError.usage` retains a failed input's recorded cost.

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

## More examples

Complete examples: [`examples/README.md`](examples/README.md).

## License

[MIT License](LICENSE)
