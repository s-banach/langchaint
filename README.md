# langchaint

langchaint is an opinionated, provider-neutral Python client for LLM applications.
It provides fully typed, asynchronous APIs for generation, streaming, embeddings, tools, retries, and billing.
The application owns the agent loop.

Alpha: the API is unstable and may change without notice.

## Why langchaint

- **One consistent API.** Set `system_prompt`, `tools`, `inference_params`, and `response_format` once with `LLM.bind()`. Use the resulting `BoundLLM` with `generate_one()`, `generate_many()`, `generate_many_records()`, or `stream_one()`.
- **Types that follow your configuration.** Passing `response_format=Answer` makes `response.output` an `Answer`. Passing `tools` also determines the `tool_manager` type and whether `generate_one()` can return `ToolCallTurn`.
- **Easy-to-discover result variants.** Every `GenerateResult`, `DispatchOutcome`, and `DispatchManyOutcome` variant has a literal `.kind` value. Every `StreamItem` variant except `str` does too. Editors can autocomplete the values, so a match statement needs no class imports.
- **Names that explain themselves.** Public names state their meaning, units, and scope through names such as `timeout_seconds`, `max_attempts`, `cost_in_usd`, and `input_tokens_cache_read`.
- **Retries that cooperate.** One `SharedBackoff` coordinates concurrency, request-start pacing, and provider-directed pauses across models that share a rate-limit quota.
- **Complete billing.** Successful results and `GenerationError` values retain provider-reported usage from every recorded attempt, including billed retries.
- **Streaming with clear ownership.** `stream_one()` returns an async context manager and async iterator. `final()` returns the typed result with its usage and attempt history.
- **Agent loops in plain Python.** Provider-neutral messages, typed tools, concurrent dispatch, and explicit result variants support ordinary async control flow.

langchaint ships `py.typed`.
Static type information preserves structured output types through generation, batching, and streaming.

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

Install `langchaint[tracing]` for OTel tracing.

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

The Pydantic model determines the output type and validates the provider response.
The response retains the complete assistant turn, raw SDK response, stop reason, per-attempt history, and billing.

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

Both `LLM` values share one `SharedBackoff`.
A rate-limit response pauses request starts across the shared quota.
A transient failure local to one request waits and retries that request.
Each binding keeps its own `max_attempts` limit.

Unlike independent retry loops, `SharedBackoff` can slow the whole quota as soon as one request receives a rate-limit response.

## Stream with an explicit lifetime

```python
text_assistant = OpenAI().model("gpt-5.6-terra").bind()

async with text_assistant.stream_one("Explain photosynthesis.") as stream:
    async for item in stream:
        if isinstance(item, str):
            print(item, end="", flush=True)

    response = await stream.final()
```

The context manager owns the provider stream and the shared request permit.
Iteration exposes text, reasoning, and tool-call `StreamItem` values.
`final()` drains unread `StreamItem` values and returns the assembled `Response` or `ToolCallTurn`.

## Build agent loops with ordinary async Python

langchaint provides the messages, typed tools, argument validation, concurrent dispatch, and result variants needed for an agent loop.
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

`ToolManager.dispatch_many()` runs parallel tool calls concurrently and preserves their order.

See [`examples/02_tool_loop.py`](examples/02_tool_loop.py) for a complete typed tool loop.

## Account for the complete call

`response.usage.cost_in_usd` includes every billed retry recorded for the call.
`response.usage_successful_attempt.cost_in_usd` reports the kept answer.
`GenerationError.usage` preserves the recorded cost of failed calls.

`Usage` separates uncached input, cache reads, cache writes, output, reasoning output, and provider-executed tool charges.
Each `Billing` records the service tier and token rates applied to one attempt.
When a used category has no known rate, its cost is `NaN` instead of zero.

## More examples

[`examples/README.md`](examples/README.md) covers structured output, batches, streaming, tools, tracing, pricing, failures, prompt caching, reasoning, embeddings, and application structure.

## License

langchaint uses the [MIT License](LICENSE).
