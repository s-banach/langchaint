# langchaint

Provider-neutral async LLM client over the official anthropic and openai SDKs.
Alpha: the API is unstable and may change without notice.

## The point

langchaint is the layer between an application's own agent loop and the provider SDKs.
Across providers it gives one message tree, one error taxonomy, one priced `Usage`, and one `RateLimiter` that owns retrying and pacing.
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
A bare `str` argument is shorthand for a conversation of one `UserMessage` holding that text.
`examples/` holds runnable files and `MIGRATING_FROM_LANGCHAIN.md`, the LangChain call-for-call map.

## What it has

**Generation only via binding.**
`LLM.bind(...)` freezes everything that determines the cacheable prompt prefix into a `BoundLLM[OutputT]`, and changing parameters is `rebind(...)`.
`BoundLLM` has `generate_one`, `generate_many`, and `stream_one`.

**A constructor per backend returning a ready `LLM`.**
`openai_model(...)`, `anthropic_model(...)`, `anthropic_bedrock_model(...)`, and `openai_bedrock_model(...)`; models outside a catalog are built directly from the re-exported adapter.

**One accounting contract for success and failure.**
Success is a `Response[OutputT]` and a terminal failure is a `GenerationError`, but both carry `usage`, the paid total across every attempt, and `to_row` flattens either to one row shape, so a mixed batch is one table.

**Priced usage.**
`Usage` partitions input tokens by cache outcome, counts reasoning output separately, and carries `cost_in_usd`, computed against a `PricingTable`; the raw SDK usage rides beside it.

**One `RateLimiter` owning retrying and pacing.**
One instance shared by several `LLM`s is one budget for the account they hit, gating every request start; a rate-limit error pauses admission for everyone sharing it.
The SDK clients are configured to never retry beneath langchaint, so attempt counts stay true.

**User-stated prompt caching.**
`automatic_prompt_caching` is a required keyword of `bind` with no default, because caching changes billing.
`cache_breakpoint=True` on a content part places a prompt-cache boundary at exactly that part; the wire mechanics are in the adapter module docstrings.

**Streaming as a handle.**
`stream_one` returns a `StreamHandle`: an async context manager that iterates `str | ToolCall` items, with `await handle.final()` returning the assembled `Response`.

**Tools under one protocol.**
`PydanticTool`, `JSONSchemaTool` (for tools discovered at run time, such as MCP tools), and `CaptureTool` (the structured exit for a `tool_choice="required"` loop) share the `Tool` protocol, so one `ToolManager` holds a mix and an application adds its own form by implementing `Tool`.
`ToolManager.dispatch` returns an outcome union; every arm carries the `tool_message` to append, and bad arguments or an unknown tool name become an outcome the model can correct rather than a raise.

**Reasoning preserved across turns.**
Every provider reasoning element is re-sent verbatim on later requests, so tool-use continuations satisfy each provider's replay rules without application code.

**OTel tracing as a wrapper.**
`langchaint.tracing` wraps `LLM`, `BoundLLM`, `StreamHandle`, and `ToolManager` in Traced counterparts; `capture_message_content` is a required keyword with no default, because recording prompts is a privacy choice.

## What it does not have

Deliberate absences, each with its reason recorded in `CLAUDE.md` or a module docstring:

- No agent class and no agent loop: the loop is ~15 lines of application code, shown in `examples/02_tool_loop.py`, and a tool returns data, never a control-flow signal.
- No per-call parameter overrides: changing parameters is `rebind`.
- No default for `automatic_prompt_caching`: every `bind` states it.
- No `requests_per_minute`: throughput under the `max_in_flight` bound follows request duration, while a client-side rate number goes stale with the account tier.
- No Converse adapter for Bedrock: the adapters take the SDKs' bundled Bedrock clients.
- No provider-parameter passthrough dict: an unmapped provider parameter is reached by subclassing the concrete adapter.
- No hand-written wire types and no client-side guessing at provider rules: stream assembly and structured-output parsing are the SDK's, and invalid inputs are sent so the provider's own error surfaces.
- No delta, usage, or stop items in a stream: a stream yields `str | ToolCall`, and `usage` and `stop_reason` live on `final()`'s `Response`.

A Chat Completions adapter, and with it third-party compatible servers such as vLLM and Ollama, is not built yet; OpenAI support is the Responses API.

## Layout

    src/langchaint/           the neutral core; imports no SDK
    src/langchaint/anthropic/ the anthropic backend
    src/langchaint/openai/    the openai backend
    src/langchaint/tracing/   the OTel tracing subpackage
    examples/                 runnable examples and MIGRATING_FROM_LANGCHAIN.md
    CLAUDE.md                 design tenets, naming rules, and the per-module map

Module docstrings are the spec of record for mechanics; `CLAUDE.md` holds the design rules.

## Verification

Run `scripts/CI.sh`; everything it runs must pass with zero errors.
The tests are offline and need no API keys.
Provider behavior claims are verified by introspection against anthropic 0.116.0 and openai 2.45.0.
