# Migrating from LangChain

langchaint is a thin, provider-neutral client over the official anthropic and openai SDKs.
It has no chains, no runnables, no middleware stack, and no agent class.
Most LangChain abstractions map to one plain call or to code you write in a loop you own.
This guide gives the call-for-call map, then explains what replaces the middleware layer.

## API map

| LangChain | langchaint |
| --- | --- |
| `ChatOpenAI(...)`, `init_chat_model(...)` | `openai_model("gpt-5.6-terra")` (or `anthropic_model("claude-sonnet-5")`), returns an `LLM` |
| `model.invoke(messages)` | `llm.bind(...).generate_one(conversation)`, returns a `Response` |
| `model.ainvoke(...)` | `generate_one` is already async; there is no sync API |
| `model.bind_tools([...])` | `llm.bind(tool_manager=ToolManager([PydanticTool(...)]))` |
| `model.with_structured_output(Model)` | `llm.bind(response_format=Model)`, read `response.output` (a parsed `Model`) |
| `model.batch([...])`, `model.abatch([...])` | `bound.generate_many([...])`, returns `list[Response \| GenerationError]` |
| `model.stream(...)`, `model.astream(...)` | `bound.stream_one(...)`, iterate `str \| ToolCall`, `await handle.final()` for the `Response` |
| `astream_events(...)` to catch tool calls | the same `stream_one` iterator yields each completed `ToolCall` |
| `create_react_agent(...)`, `AgentExecutor` | own the loop over `generate_one` and `ToolManager.dispatch` (see `02_tool_loop.py`) |
| a tool returning `Command(goto=/update=)` | not supported by design; a tool returns data, the app routes between turns |
| `RunnableRetry`, per-model `max_retries` | `RateLimiter(max_attempts=...)`, one instance shared across `LLM`s |
| `InMemoryRateLimiter`, rate-limit middleware | `RateLimiter(max_in_flight=...)`, one shared account budget |
| `.with_fallbacks([...])` | app-level `try`/`except` over two bindings (see below) |
| `set_llm_cache(...)` client-side cache | provider prompt caching via `automatic_prompt_caching`, required on `bind` (no client cache) |
| callbacks, LangSmith tracing | `langchaint.tracing.TracedLLM` over any OTel exporter (see `04_tracing.py`) |
| `temperature=`, `top_p=`, `seed=` on the model | not exposed; `InferenceParams` carries only `max_completion_tokens` and `reasoning_effort` |
| `SystemMessage` in the message list | `system_prompt=` on `bind`, frozen into the binding |
| `HumanMessage` / `AIMessage` / `ToolMessage` | `UserMessage` / `AssistantMessage` / `ToolMessage` |

## The middleware layer: own the loop instead

LangChain's agent middleware hooks (`before_model`, `after_model`, `modify_model_request`, `wrap_tool_call`, and the rest) exist because the framework owns the loop and lets you splice code into it.
langchaint ships no loop, so there is nothing to splice into: each hook is plain code at the matching point in the loop you write.

| Middleware hook | Where it goes in your loop |
| --- | --- |
| `before_model` | a statement before `await bound.generate_one(conversation)` |
| `after_model` | a statement after it, inspecting the `Response` (`stop_reason`, `tool_calls`, `usage`) |
| `modify_model_request` | `bound = bound.rebind(...)` before the next turn; the binding is the request shape |
| `wrap_tool_call`, tool error handling | `dispatch` already returns an is_error tool message for bad names and bad arguments; wrap your own code around the `dispatch` call |
| human-in-the-loop / interrupts | check `call.name` (or a tool's `app_data`) between turns and decide; a declined call is an is_error `ToolMessage` you append (see `run_agent`'s `approve` gate) |
| summarization / message trimming | edit the `conversation` list you hold before the next turn |
| structured-output middleware | `bind(response_format=Model)`; a refusal or truncation raises a `GenerationError` leaf you catch |
| usage / cost tracking | bill on `response.usage` (the paid total across retries, carrying `cost_in_usd`); `response.usage_successful_attempt` is the single kept answer's own usage, equal to `usage` unless a billed 200 was retried. Or `to_row(result)` for a table. |

The gain from owning the loop is that a budget check, an approval gate, or a binding swap is ordinary control flow with the full conversation in scope, not a callback fighting an engine that holds the state.

## From LangGraph

`StateGraph` nodes and conditional edges are plain control flow in the loop you own, and the middleware table above covers the `create_agent` hooks. Three mappings are specific to LangGraph apps.

| LangGraph | langchaint |
| --- | --- |
| summing `AIMessage.usage_metadata` across turns to bill a run | `response.usage` is the paid total across retries and carries `cost_in_usd`; `Usage.sum_of` folds a run, and a cancelled call's settled spend lands in `abandoned_call_log` |
| a per-call deadline as `awrap_model_call` middleware | `asyncio.timeout` around the call it bounds, in the loop you own (see `examples/full_app`) |
| `get_stream_writer()` and `astream(subgraphs=True)` stream a nested sub-agent progress tree to one consumer with no reference passing | no counterpart: the application owns the event stream, and `examples/full_app` (`events.py`, `harness.py`) is the runnable pattern |

## Retries and rate limiting: one RateLimiter, not per-call config

There is no retry setting on a generate call and no rate-limit middleware.
One `RateLimiter` owns retrying (`max_attempts`, `backoff_base_seconds`, `backoff_max_seconds`) and pacing (`max_in_flight`, default 8).
It is stateful and shareable: pass one instance to every `LLM` hitting the same account and they share one budget, so a rate-limit error pauses admission for all of them until a request succeeds again.
The runnable setup, one limiter across an openai and an anthropic model, is `shared_rate_limiter` in `05_rate_limiting_and_errors.py`.

There is deliberately no `requests_per_minute`: an in-flight bound self-adjusts throughput along request duration, while a client-side rate number models one dimension of the provider's limit and goes stale with the account tier.

## Fallbacks: a try/except over two bindings

There is no `.with_fallbacks`. A fallback is app code, because the app decides what counts as worth failing over.
The runnable version, a `try`/`except (GenerationError, AbortBatchError)` over two bindings, is `generate_with_fallback` in `05_rate_limiting_and_errors.py`.

## Errors: success is a Response, failure is a GenerationError

`generate_one` returns a `Response` on success and raises on a terminal outcome.
`GenerationError` is the base of the three terminal per-item leaves: `RetriesExhaustedError` (transient budget spent), `RefusalError` (the model refused on the structured path), and `MaxCompletionTokensExceededError` (the structured response hit the token cap).
Catch `GenerationError` to handle all three at once.
`AbortBatchError` is separate: it means the request is misconfigured (a bad request, an unmapped cache rate), so retrying cannot help, and in a batch it cancels the siblings.
In a batch, `generate_many` returns each terminal per-item failure as a `GenerationError` in its slot instead of raising, so the batch finishes and `to_row` renders successes and failures to the same table.
The runnable catch, one `try`/`except GenerationError` around a structured `generate_one`, is `catch_generation_error` in `05_rate_limiting_and_errors.py`.
