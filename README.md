# langchaint

Provider-neutral async LLM client over the official provider SDKs, verified against anthropic 0.116.0 and openai 2.45.0.
Alpha: the API is unstable and may change without notice.

## The point

langchaint is the layer between an application's own agent loop and the provider SDKs.
It gives one message tree, one error taxonomy, one `Usage` shape with a priced `cost_in_usd` on every result, and one `RateLimiter` that owns retrying and pacing, the same across providers.
The application keeps the loop: langchaint ships no agent class, and every billing-relevant choice (prompt caching above all) is stated by the application rather than defaulted by the library.

## Install

Requires Python >= 3.13.
The hard dependencies are `pydantic` and `jsonschema`; the provider SDKs are optional dependencies the application pins directly, and langchaint declares no extras.

    pip install langchaint openai        # or anthropic, or both

`import langchaint` needs neither SDK.
Each backend subpackage imports its SDK at module top, so importing `langchaint.openai` without the openai package raises a `ModuleNotFoundError` naming the package to install.

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

`bind(response_format=Sentiment)` returns `BoundLLM[Sentiment]` by overload, so `response.output` is a validated `Sentiment` instance; without `response_format` it returns `BoundLLM[str]` and `output` is the assistant text.
A bare `str` argument is shorthand for a conversation of one `UserMessage` holding that text.
`examples/` holds runnable files from basics through the streaming tool loop, prompt caching, and a budgeted `tool_choice="required"` loop, plus `MIGRATING_FROM_LANGCHAIN.md`, the LangChain call-for-call map.

## What it has

**Generation only via binding.**
`LLM` has no generate methods; `LLM.bind(...)` freezes everything that determines the cacheable prompt prefix (`system_prompt`, `tool_manager`, `inference_params`, `tool_choice`, `parallel_tool_calls`, `automatic_prompt_caching`, and `response_format`, which fixes the output type) into a `BoundLLM[OutputT]`.
Changing parameters is `rebind(...)`, which returns a new `BoundLLM` and carries the same `response_format` overloads.
`BoundLLM` has `generate_one`, `generate_many` (an order-aligned `list[Response[OutputT] | GenerationError]`), and `stream_one`.

**A constructor per backend returning a ready `LLM`.**
`openai_model("gpt-5.6-terra")`, `anthropic_model("claude-sonnet-5")`, and `anthropic_bedrock_model(model, aws_region=...)` each take a `Literal` model name (so typos fail the type check), look up public prices, and wrap the adapter in an `LLM`; `client`, `pricing`, and `rate_limiter` default sensibly and are overridable.
`anthropic_bedrock_model` takes the Bedrock wire model id and sends it verbatim, routing each id to its Bedrock API and pricing through `ANTHROPIC_BEDROCK`; OpenAI on Bedrock is the SDK's bundled `AsyncBedrockOpenAI` passed as `client`.
Models outside a catalog are built directly from the adapter: `LLM(AnthropicMessagesProvider(...))`.

**One result shape for success and failure.**
Success is a frozen `Response[OutputT]`; a terminal failure is a `GenerationError` leaf: `RetriesExhaustedError`, `RefusalError`, or `MaxCompletionTokensExceededError` (the structured path never returns silently wrong data).
Both carry `attempt_records`, one `AttemptRecord` per request sent, and `usage`, the paid total across every attempt, so the default a caller bills on is the money actually spent; `to_row` flattens either to one row shape, so a mixed batch is one table.
`Response.raw` is the SDK's own response object held by reference, and `usage_raw` is the raw SDK usage beside every `Usage`.

**Priced usage.**
`Usage` partitions input tokens into `input_tokens_cache_read`, `input_tokens_cache_write`, and `input_tokens_cache_none` (the partition both providers' counters map onto), counts `output_tokens` and `output_tokens_reasoning`, and carries `cost_in_usd`, computed inside the adapter against a `PricingTable`.
`Usage.sum_of` folds usages and costs together; each backend's `cost_breakdown(usage_raw, pricing)` reports the per-category split through the same `price` call that produced the stored `cost_in_usd`, so the two cannot drift.

**One `RateLimiter` owning retrying and pacing.**
It holds `max_attempts`, `backoff_base_seconds`, `backoff_max_seconds`, and `max_in_flight`; it is stateful and shareable, so one instance passed to several `LLM`s is one shared budget for the account they hit, gating every request start (first attempts, retries, batch items, stream openings).
A rate-limit error pauses admission for everyone sharing it, honoring a server-stated retry-after up to 60 seconds, then admits one probe at a time until a probe succeeds; other transient errors pause nobody.
Adapters store a `with_options(max_retries=0)` copy of the SDK client, so the SDK never retries beneath the package and attempt counts stay true.

**User-stated prompt caching.**
`automatic_prompt_caching` is a required keyword of `bind` with no default, because caching changes billing.
`cache_breakpoint=True` on a `TextPart` or `ImagePart` places a prompt-cache boundary at exactly that part, honored under either binding value; `system_prompt` also binds as a sequence of `TextPart`s, so a boundary can sit inside the frozen prefix.
The anthropic adapter takes `cache_ttl` (`"5m"` default, `"1h"` at 2x write cost); `generate_many(conversations, warm_cache=True)` runs the first conversation to completion before admitting the rest, so a batch sharing a prefix pays one cache write instead of one per in-flight item.
The wire mechanics, including the anthropic 4-marker budget, are in the two adapter module docstrings.

**Streaming as a handle.**
`stream_one` returns a `StreamHandle`: an async iterator of `StreamItem = str | ToolCall`, an idempotent `await handle.final()` returning the assembled `Response`, and an async context manager so abandoning a stream closes the connection.
Text chunks are the SDK's own strings passed through without a wrapper, and each `ToolCall` is yielded once, complete, when its block closes.

**Three tool forms under one protocol.**
`PydanticTool` (an async function taking one validated `args_model` instance), `JSONSchemaTool` (a raw JSON `args_schema`, for tools discovered at run time such as MCP tools, with arguments validated against it by `jsonschema` before the function runs), and `CaptureTool` (no function; `capture` returns the validated `args_model` instance as `DispatchCaptured.captured`, the structured exit for a `tool_choice="required"` loop) share the `Tool` protocol, so one `ToolManager` holds a mix and an application adds its own form by implementing `Tool`.
`ToolManager.dispatch` returns a `DispatchOutcome` union (`DispatchHandled`, `DispatchInvalidToolArgs`, `DispatchUnknownTool`); every arm carries the `tool_message` to append, and bad argument JSON or an off-list tool name becomes an outcome the model can correct rather than a raise.
A tool function returns model-facing content, or `ToolOutputExplicit` adding `is_error` and `app_data`, a typed channel the model never sees.
`dispatch_many` runs one turn's calls concurrently, returns outcomes in call order, and raises tool-function defects as a `DispatchExceptionGroup` only after every sibling settles.

**Reasoning preserved across turns.**
Each provider reasoning element becomes one `ReasoningTrace` in `AssistantMessage.turn`, re-sent verbatim on later requests, so tool-use continuations satisfy each provider's replay rules without application code.

**OTel tracing as a wrapper.**
`langchaint.tracing` (imports only opentelemetry-api) has `TracedLLM`, `TracedBoundLLM`, `TracedStreamHandle`, and `TracedToolManager`; attribute mappers own span attribute names and never receive the conversation, so no built-in mapper can leak a prompt.

## What it does not have

Each absence is deliberate; the reasons are recorded in `CLAUDE.md` and the module docstrings.

- No agent class or library-owned tool loop: the loop is ~15 lines of application code over `generate_one` (or `stream_one`) and `dispatch`, shown in `examples/02_tool_loop.py`, and a tool returns data, never a control-flow signal, so stop, route, and escalate stay decisions the application makes between turns.
- No per-call parameter overrides: changing parameters is `rebind`.
- No default for `automatic_prompt_caching`: every `bind` states it.
- No `requests_per_minute`: the `max_in_flight` bound self-adjusts throughput along request duration, while a client-side rate number goes stale with the account tier.
- No Chat Completions adapter and no third-party chat-completions-compatible servers (vLLM, Ollama): OpenAI support is the Responses API only.
- No Converse adapter for Bedrock: Bedrock is served through the SDKs' bundled clients.
- No provider-parameter passthrough dict: `InferenceParams` is `max_completion_tokens`, `reasoning_effort`, and `temperature` (no `top_p`, no `seed`), and an unmapped provider parameter is reached by subclassing the adapter.
- No hand-written wire types and no client-side guessing at provider rules: stream assembly and structured-output parsing are the SDK's, SDK objects ride by reference instead of being copied into same-shaped package objects, and invalid inputs are sent so the provider's own error surfaces.
- No tool-call delta stream items and no usage or stop stream items: a stream yields `str | ToolCall`, and `usage` and `stop_reason` live on `final()`'s `Response`.
- No extras and no bundled SDKs: the application pins `anthropic`, `openai`, or `opentelemetry-api` itself, and only the subpackage imports require them.

## Layout

    src/langchaint/           the neutral core (imports no SDK): llm.py, messages.py, tools.py, rate_limiter.py, exceptions.py, response.py, streaming.py, usage.py, pricing.py, inference_params.py, provider.py, checked_copy.py
    src/langchaint/anthropic/ the anthropic backend: anthropic_model, anthropic_bedrock_model, pricing tables, AnthropicMessagesProvider
    src/langchaint/openai/    the openai backend: openai_model, OPENAI_PRICING, OpenAIResponsesProvider
    src/langchaint/tracing/   the OTel tracing subpackage
    examples/                 runnable examples and MIGRATING_FROM_LANGCHAIN.md
    CLAUDE.md                 design tenets, naming rules, and the per-module map

Module docstrings are the spec of record for mechanics; `CLAUDE.md` holds the design rules and the reasons behind each deliberate absence.

## Verification

Run `scripts/CI.sh`; it runs `pyrefly check`, `ruff check`, and `pytest` through `uv run`, so the tools resolve from the locked dev dependency group, and all must pass with zero errors.
The tests are offline: they feed constructed SDK objects into the adapter helpers and drive `BoundLLM`/`StreamHandle` with stub providers, so they need no API keys.
