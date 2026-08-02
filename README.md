# langchaint

Provider-neutral async LLM client over the official anthropic and openai SDKs.
Alpha: the API is unstable and may change without notice.

## The point

langchaint is the layer between an application's own agent loop and the provider SDKs.
Across providers it gives one message tree, one error taxonomy, one priced `Usage`, and one `SharedBackoff` that owns pacing.
The application keeps the agent loop and states every billing-relevant choice itself, starting with prompt caching; langchaint defaults none of them.

## Install

Requires Python >= 3.13.
The hard dependencies are `pydantic` and `jsonschema`; the provider SDKs are optional dependencies the application pins directly, and langchaint declares no extras.

    pip install langchaint openai        # or anthropic, or both

`import langchaint` needs neither SDK; importing `langchaint.openai` without the openai package raises a `ModuleNotFoundError` naming the package to install.

## Example

```python
import asyncio

from pydantic import BaseModel

from langchaint.openai import openai_model


class Sentiment(BaseModel):
    label: str
    confidence: float


async def main() -> None:
    llm = openai_model("gpt-5.6-terra")
    classifier = llm.bind(
        system_prompt="Classify the sentiment of the user's message.",
        response_format=Sentiment,
        automatic_prompt_caching=False,
    )
    response = await classifier.generate_one("This is the best day I have had in months.")
    print(response.output.label, response.usage.cost_in_usd)


asyncio.run(main())
```

`bind(response_format=Sentiment)` returns a `BoundLLM[Sentiment]`, so `response.output` is a validated `Sentiment` instance; without `response_format`, `output` is the assistant text.
A bare `str` generation_input is shorthand for `[UserMessage(content=generation_input)]`.
`examples/` holds one snippet file per subject and `MIGRATING_FROM_LANGCHAIN.md`, the LangChain call-for-call map.

## What it has

**Generation only via binding.**
`LLM.bind(...)` freezes everything that determines the cacheable prompt prefix into a `BoundLLM[OutputT]`, and changing parameters is `rebind(...)`.
`BoundLLM` has `generate_one`, `generate_many`, and `stream_one`.

**A constructor per backend returning a ready `LLM`.**
`openai_model(...)`, `anthropic_model(...)`, `anthropic_bedrock_model(...)`, and `openai_bedrock_model(...)`.

**One accounting contract for success and failure.**
Success is a `Response[OutputT]` and a terminal failure is a `GenerationError`.
Both carry `usage`, the paid total across every attempt.

**Priced usage.**
`Usage` partitions input tokens by cache outcome and carries one cost per priced category; `cost_in_usd` is their sum.

**One `SharedBackoff` owning pacing.**
One instance is one backpressure domain for the account it guards.
Its `admitted()` block gates every request start; a rate limit pauses the whole domain.
`max_attempts` on the `LLM` bounds retrying.

**User-stated prompt caching.**
`automatic_prompt_caching` is a required keyword of `bind` with no default, because caching changes billing.
`cache_breakpoint=True` on a content part places a prompt-cache boundary at exactly that part.

**Streaming as a handle.**
`stream_one` returns a `StreamHandle`: an async context manager that iterates `str | ReasoningDelta | ToolCall` items, with `await handle.final()` returning the assembled `Response`.

**Tools under one protocol.**
`PydanticTool`, `JSONSchemaTool` (for tools discovered at run time, such as MCP tools), and `CaptureTool` share the `Tool` protocol.
Use `@tool(description=...)` for an async function with one `BaseModel` parameter.
The parameter annotation supplies `args_model`.
`name` defaults to `function.__name__`; pass `name` to override it.
Construct `PydanticTool(...)` directly when its fields come from runtime data.
One `ToolManager` holds a mix, and an application adds its own form by implementing `Tool`.

**Reasoning preserved across turns.**
Every provider reasoning element is re-sent verbatim on later requests, so tool-use continuations satisfy each provider's replay rules without application code.

**OTel tracing as a wrapper.**
`langchaint.tracing` wraps `LLM`, `BoundLLM`, `StreamHandle`, and `ToolManager` in Traced counterparts; `capture_message_content` is a required keyword with no default, because recording prompts is a privacy choice.

## What it does not have

- No agent class and no agent loop: the loop is ~15 lines of application code, shown in `examples/02_tool_loop.py`, and a tool returns data, never a control-flow signal.
- No client-side guessing at provider rules.
- No document or PDF part: convert before sending, rasterizing pages to `ImagePart` or extracting the text layer to `TextPart`.

No Chat Completions adapter is built yet; OpenAI support is the Responses API.

## Layout

    src/langchaint/           the neutral core; imports no SDK
    src/langchaint/anthropic/ the anthropic backend
    src/langchaint/openai/    the openai backend
    src/langchaint/tracing/   the OTel tracing subpackage
    examples/                 one snippet file per subject and MIGRATING_FROM_LANGCHAIN.md

## Verification

Run `scripts/CI.sh`.
The tests are offline and need no API keys.
