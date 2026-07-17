# langchaint

Provider-neutral async LLM client over the official provider SDKs (anthropic >= 0.116.0, openai >= 2.45.0).
Prototype: nothing is stable, everything can be rewritten.

## Design

**Generation only via binding.**
`LLM` has no generate methods.
`LLM.bind(...) -> BoundLLM[OutputT]` freezes everything that determines the cacheable prompt prefix: `system_prompt`, `tool_manager`, `inference_params`, `tool_choice`, `parallel_tool_calls`, `automatic_prompt_caching`.
`response_format` is frozen too, as the field that fixes the output type.
`automatic_prompt_caching` has no default: caching changes billing, so every `bind` states it.
A `TextPart` or `ImagePart` with `cache_breakpoint=True` places a prompt-cache boundary at exactly that part, under either `automatic_prompt_caching` value, so binding `False` and marking parts is the fully user-specified caching configuration.
`generate_many(conversations, warm_cache=True)` runs the first conversation to completion before admitting the rest, so a batch sharing a cached prefix pays one cache write instead of one per in-flight item; it is opt-in because it costs one item of serial latency.
`system_prompt` also binds as a sequence of `TextPart`s, so a boundary can sit inside the frozen prefix (stable instructions marked, semi-stable context after).
There are no per-call parameter overrides; changing parameters is `rebind(...)`, which returns a new `BoundLLM` with the SDK keyword arguments converted again.
`bind(response_format=Model)` returns `BoundLLM[Model]`; without `response_format` it returns `BoundLLM[str]`, selected by overload.
`rebind` carries the same overload, so `rebind(response_format=Model)` switches the output type to `BoundLLM[Model]`, `rebind(response_format=None)` switches it back to `BoundLLM[str]`, and leaving it out keeps the current type.
Every generate and stream method takes a conversation of `Message`s; a bare `str` is accepted as shorthand for a conversation of one `UserMessage` holding that text.

**The catalog is a constructor function per backend returning a ready LLM (anthropic adds a second for Bedrock).**
`from langchaint.openai import openai_model` then `openai_model("gpt-5.6-terra")`, and `from langchaint.anthropic import anthropic_model` then `anthropic_model("claude-sonnet-5")`, take the provider's own model identifier (a `Literal`, so typos fail the type check), look up the public prices, construct the adapter, and wrap it in an `LLM`.
`client=None` constructs the native first-party SDK client from environment credentials.
Anthropic Bedrock is a second sibling constructor, `anthropic_bedrock_model(model, aws_region=...)`: it names the same catalog model and reads the model's Bedrock surface (which of two SDK client classes) and wire model id from a table, so the application names neither the client class nor the Bedrock id.
Constructing an adapter directly covers models outside the catalog.

**The SDKs are optional dependencies, one extra per backend.**
The neutral core (`LLM`, the message tree, the error taxonomy) imports no SDK, so `import langchaint` needs neither package.
Each backend lives in its own subpackage that imports its SDK at module top: install `langchaint[openai]` or `langchaint[anthropic]` (or `langchaint[all]`) for the ones you use, and importing `langchaint.openai` without the openai package raises a `ModuleNotFoundError` naming the extra to install.
The import path is the boundary: only code that reaches for a backend requires its SDK, and a type checker following `langchaint.openai` never resolves anthropic's types.
OTel tracing follows the same pattern: `langchaint.tracing` imports only opentelemetry-api, install `langchaint[otel]`, and it stays off `import langchaint`.

**Adapters wrap SDK clients and delegate to the SDK.**
`AnthropicMessagesProvider(client=AsyncAnthropic(...))` and `OpenAIResponsesProvider(client=AsyncOpenAI(...))` call `messages.create/parse/stream` and `responses.create/parse/stream`; stream assembly and structured-output parsing are the SDK's, not hand-written.
Adapters store a `with_options(max_retries=0)` copy of the client, so the SDK never retries beneath the package's retry loop.
OpenAI support is the Responses API only: every supported OpenAI model speaks it, so there is no Chat Completions adapter.
The adapter always sends `store=False` because conversation state is the caller's conversation argument.
Anthropic Bedrock is two distinct SDK surfaces, the legacy `InvokeModel` client `AsyncAnthropicBedrock` and the Messages-API client `AsyncAnthropicBedrockMantle`, and the catalog models split across them, so `anthropic_bedrock_model` reads the client class and wire model id per model from `ANTHROPIC_BEDROCK`.
OpenAI on Bedrock is the bundled `AsyncBedrockOpenAI` passed as `client`; there is no Converse adapter.
Requests travel as typed frozen dataclasses whose optional fields are `X | Omit`, passed to the SDK as explicit keywords: no `**kwargs`, no hand-written wire TypedDicts, and the SDK overloads resolve without casts.

**Anthropic prompt caching is placed by the adapter.**
A bind-time `cache_control` marker goes on the system block (or the last tool when there is no system prompt), and a per-request marker goes on the last block of the last message, so the breakpoint follows a growing conversation.
Placement is manual because `messages.parse` lacks the top-level `cache_control` parameter; manual placement keeps create/parse/stream uniform.
Every marker's TTL is the adapter's `cache_ttl` (`"5m"` default, `"1h"` for entries that must survive longer gaps at 2x write cost), a keyword of `anthropic_model` and `anthropic_bedrock_model`.

**Usage counters partition, cost is the adapter's job.**
`Usage` stores `input_tokens_cache_read`, `input_tokens_cache_write`, `input_tokens_cache_none`, and `output_tokens`; the three input counters are a disjoint partition and `input_tokens_total` is derived.
`input_tokens_total_provider_reported` holds what the provider itself reported: openai's `input_tokens` includes cached and cache-write tokens, so the adapter fills it and a validator cross-checks the partition; anthropic reports no all-inclusive total, so it stays None.
The counters are validated non-negative, which is what guards the openai path: there the partition is derived by subtraction and sums to the reported total by construction.
`cost_in_usd` is computed inside the adapter from raw provider counts against a `PricingTable`, because providers split counters the normalized `Usage` collapses: anthropic bills 5-minute and 1-hour cache writes at different rates, and an adapter that sees 1-hour writes without `cache_write_1h_usd_per_million_tokens` raises `AbortBatchError` rather than misbill.

**Success is a `Response`, failure is a `GenerationError`.**
A generate that succeeds returns a frozen `Response[OutputT]` with every field present; one that ends terminally raises (or, in a batch, returns) a `GenerationError`.
Its three leaves are the terminal per-item outcomes: `RetriesExhaustedError` (transient budget spent), `RefusalError` (the model refused on the structured path), and `ExceededMaxCompletionTokensError` (the structured response hit the token cap before its JSON parsed).
On the base, `stop_reason` is `StopReason | None`: `"refusal"` or `"max_tokens"` on those two leaves, `None` on `RetriesExhaustedError`, whose attempts never reached a completed turn.
Both hold `attempt_records`, one `AttemptRecord` per request sent: raw `time.monotonic()` start and end readings bracketing the send only (slot waits and backoff sleeps excluded, so rate limiting is distinguishable from slow requests), the attempt's `TransientError` (None on the attempt that succeeded or a rejected 200), and the attempt's `usage`/`cost_in_usd` (None for a transport failure that billed nothing).
On a `Response` every record but the last failed; on a `GenerationError` the records describe the terminal outcome.
`attempts` is derived from the records on both.
Both also carry `model`, `provider_name`, and `elapsed_seconds` (first request to completion, waits included, so stored rather than derived), so the module-level `to_row(result)` flattens either to the same scalar keys and a mixed list of successes and failures is one table.
A `GenerationError` derives `usage`/`cost_in_usd` from its records, so a refusal or truncation reports its real cost while a retry-exhausted item whose attempts billed nothing reports zero; `error_text` carries the failure reason.
On a `Response`, `usage`, `cost_in_usd` (the successful attempt's own), `stop_reason`, `assistant_message`, and `raw` are always present.
`raw` is the SDK's own response model held by reference; it is never dumped to a dict, because dumping deep-copies every response body per request.

**One RateLimiter owns retrying and pacing.**
`RateLimiter` holds `max_attempts`, `backoff_base_seconds`, `backoff_max_seconds`, and `max_in_flight`; it is stateful and shareable, so one instance passed to several `LLM`s is one shared budget for the account they hit.
Its slot gates every request start on every path (first attempts, retries, batch items, stream openings); backoff sleeps outside the slot.
There is deliberately no `requests_per_minute`: an in-flight bound self-adjusts throughput along request duration, while a client-side rate number models one dimension of the provider's multi-dimensional limit and goes stale with the account tier.
A rate-limit error (`Provider.classify` returning `"rate_limit"`, or any error naming a server-stated retry-after) pauses admission for everyone sharing the limiter: for the server-stated wait when one was sent (parsed from the `retry-after-ms` / `retry-after` headers onto `TransientError.retry_after_seconds`, capped at 60 seconds), else for the failing task's backoff delay.
After the pause, admission stays limited to one probe request at a time until the probe succeeds; other transient errors (timeouts, 5xx) pause nobody, because they say nothing about the account's quota.

**Retry stays in the package, classification in the adapter.**
Adapters raise `TransientError`, `AbortBatchError`, or a `GenerationError` leaf directly where they know the answer: rate limits and overload (transient), a bad request (abort), and an empty structured parse, which splits three ways (a refusal is `RefusalError`, a token-cap truncation is `ExceededMaxCompletionTokensError`, anything else is `TransientError`).
The retry loop retries only `TransientError`; it honors the rest without asking.
Unrecognized exceptions go through `Provider.classify`, which returns `"rate_limit"`, `"transient"`, or `"abort"`, and anything it does not recognize is abort.
`generate_one` raises `RetriesExhaustedError` for transient exhaustion, `RefusalError` or `ExceededMaxCompletionTokensError` on the structured path, and `AbortBatchError` for an abort classification; the first three share the `GenerationError` base a caller can catch at once.
`generate_many` returns `list[Response[OutputT] | GenerationError]`, so a terminal per-item failure is a row in its slot rather than a raise, while an `AbortBatchError` still cancels the siblings and raises; results stay order-aligned.

**Streaming is a handle.**
`stream_one` returns a `StreamHandle`: an async iterator of `StreamItem = str | ToolCall`, an idempotent `await handle.final()` returning the assembled `Response`, and an async context manager so abandoning a stream closes the connection.
Text chunks are the provider SDK's own strings passed through without a wrapper class or copy, and each `ToolCall` is yielded once, complete, when its block closes.
There are deliberately no tool-call delta items (a consumer cannot act on partial argument JSON, and both SDKs accumulate the arguments and hand over the finished call) and no usage or stop items (usage, `cost_in_usd`, and `stop_reason` live on `final()`'s `Response`, and a bare-`str` stop reason could not share a union with bare-`str` text chunks).
Connection failures before the first yielded item retry under the `RateLimiter`; after the first yielded item nothing retries, because replaying chunks the caller already consumed would duplicate output.
An open stream holds one `RateLimiter` slot from opening until it closes or exhausts, so long-lived streams count against `max_in_flight`.

**Tools are explicit, dispatch is owned.**
A `Tool` is an async function taking one pydantic model and returning str or a sequence of content parts (text and images the model then sees), plus an explicit name and description; the args model is the schema source, so there is no signature introspection and no docstring scraping.
Tool content is model-facing, so it is exactly `MessageContent` (`str | Sequence[Part]`, the model-facing message body, aliased in `messages.py`): a function with a typed result serializes it to that form itself.
A function may instead return a `ToolOutputExplicit` wrapping that content plus `is_error` and `app_data` (the app-facing channel, which does carry a typed `BaseModel` or `Mapping` the model never sees).
`Tool.validate_and_run` validates raw call JSON against `args_model` and runs the function; it lives on `Tool` because there the args type parameter is concrete, so the validated arguments reach the function fully typed.
`Tool.dispatch` returns `DispatchHandled[AppDataT] | DispatchInvalidToolArgs`, so on the handled outcome a caller that dispatched a known tool reads `app_data` back at its concrete type with no `isinstance`.
`ToolManager` serializes the tools provider-neutrally, resolves the called name, and returns each outcome as a `DispatchOutcome`, a three-arm union over a heterogeneous tool set (where `app_data` erases to `BaseModel | Mapping[str, object] | None`): `DispatchHandled` carries the model-facing `ToolMessage` the caller appends to the conversation, paired with the function's `app_data` (data the model never sees); `DispatchInvalidToolArgs` carries a default `is_error` `ToolMessage` plus the pydantic `ValidationError` as a required field, so a caller that authors its own reply reads `validation_error.errors()` with no narrowing crutch; `DispatchUnknownTool` carries a default `is_error` `ToolMessage` naming the held tools plus the off-list `called_name`.
Every arm carries `tool_message`, so a caller that only appends the reply reads `result.tool_message` with no `match`.
An off-list tool name becomes a `DispatchUnknownTool` outcome rather than a raise, just like argument JSON the model got wrong becomes a `DispatchInvalidToolArgs` outcome: both are model data the model can correct (a provider can emit a name outside the sent schemas, and a rebind can strand an earlier turn's `tool_call`).
`tool_choice` is the neutral vocabulary `"auto" | "required" | SpecificTool | "none"` (anthropic's `"any"` maps from `"required"`); `parallel_tool_calls: bool` maps to anthropic `disable_parallel_tool_use` and openai `parallel_tool_calls`.

**The ReAct loop is a recipe, not vendored code.**
There is no agent class.
The loop is ~15 lines and is correct because the hard parts (retries, pacing, classification, validation) live below `generate_one` and `dispatch`, so a hand-copied loop cannot be subtly wrong.
The non-streaming loop runs over `generate_one`; the streaming loop runs over `stream_one`, printing text chunks as they arrive and dispatching the completed `ToolCall`s between turns, reading `dispatch(call).tool_message` for the ToolMessage it appends and matching `DispatchHandled` for the function's model-invisible `app_data` or `DispatchInvalidToolArgs` for the `validation_error`.
Owning the loop is what lets a caller enforce a budget mid-run, stream tokens, or swap the binding on a tool result, none of which a fixed agent surface exposes.

## Layout

    CLAUDE.md                design, naming, and review rules
    README.md                this file
    src/langchaint/
        messages.py          message and part models, StopReason
        usage.py             Usage counters
        inference_params.py  InferenceParams
        exceptions.py        error vocabulary: TransientError,
                             AbortBatchError, AttemptRecord,
                             GenerationError and its leaves
        response.py          Response[OutputT], to_row
        rate_limiter.py      RateLimiter
        tools.py             Tool, ToolSchema, ToolManager
        provider.py          Binding, ToolChoice, PricingTable,
                             StreamItem, ProviderResult,
                             ProviderStream, BoundProvider, Provider
        streaming.py         StreamHandle
        llm.py               LLM, BoundLLM, the retry loop
        anthropic/           the anthropic backend (needs the anthropic
                             SDK): __init__ has anthropic_model,
                             ANTHROPIC_PRICING, AnthropicModelName;
                             messages_provider.py has the adapter
                             AnthropicMessagesProvider
        openai/              the openai backend (needs the openai SDK):
                             __init__ has openai_model, OPENAI_PRICING,
                             OpenAIModelName; responses_provider.py has
                             the adapter OpenAIResponsesProvider
        tracing/             OTel span tracing (needs opentelemetry-api,
                             install langchaint[otel]): __init__ has
                             TracedLLM, TracedBoundLLM, TracedStreamHandle

## Verification

Run `scripts/CI.sh`; it runs `pyrefly check`, `ruff check`, and `pytest` through `uv run`, so the tools resolve from the locked dev dependency group, not a hand-activated `.venv`.
All must pass with zero errors.
The tests are offline: they feed constructed SDK objects into the adapter helpers and drive `BoundLLM`/`StreamHandle` with stub providers, so they need no API keys.
