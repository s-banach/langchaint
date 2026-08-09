# Migrating from LangChain

langchaint is a thin, provider-neutral client over the official anthropic and openai SDKs.
It has no chains, no runnables, no middleware stack, and no agent class.

## API map

| LangChain | langchaint |
| --- | --- |
| `ChatOpenAI(...)`, `init_chat_model(...)` | `OpenAIAccount`, then `openai.model("gpt-5.6-terra")` |
| `model.invoke(messages)` | `llm.bind(...).generate_one(messages)`, returns a `Response` |
| `model.ainvoke(...)` | `generate_one` is already async; there is no sync API |
| `model.bind_tools([...])` | decorate `get_weather` with `@tool(description=...)`, then pass `ToolManager([get_weather])` |
| `model.with_structured_output(Model)` | `llm.bind(response_format=Model)`, read `response.output` (a parsed `Model`) |
| `model.batch([...])`, `model.abatch([...])` | `bound.generate_many([...])`, returns `list[Response \| GenerationError]` |
| `model.stream(...)`, `model.astream(...)` | `bound.stream_one(...)` gives a handle: iterate `str \| ReasoningDelta \| ToolCallDelta \| ToolCall`, then `await handle.final()` for the `Response` |
| `astream_events(...)` to catch tool calls | the same `stream_one` iterator yields each completed `ToolCall` |
| `create_react_agent(...)`, `AgentExecutor` | own the loop over `generate_one` and `ToolManager.dispatch` (see `02_tool_loop.py`) |
| a tool returning `Command(goto=/update=)` | a tool returns data; the app routes between turns |
| `RunnableRetry`, per-model `max_retries` | `max_attempts` on `bind` |
| `InMemoryRateLimiter`, rate-limit middleware | account `max_request_starts_per_second` and `max_concurrent_requests` |
| `.with_fallbacks([...])` | app-level `try`/`except` over two bindings (see below) |
| `set_llm_cache(...)` client-side cache | provider prompt caching via `automatic_prompt_caching`, required on `bind` (no client cache) |
| callbacks, LangSmith tracing | `langchaint.tracing.TracedLLM` over any OTel exporter (see `08_tracing.py`) |
| `temperature=` on the model | `InferenceParams(temperature=...)` on `bind` |
| `top_p=`, `seed=` on the model | subclass the adapter to send a provider parameter `InferenceParams` does not carry |
| `SystemMessage` in the message list | `system_prompt=` on `bind` |
| `HumanMessage` / `AIMessage` / `ToolMessage` | `UserMessage` / `AssistantMessage` / `ToolMessage` |

## The middleware layer: own the loop instead

langchaint ships no loop.
Each hook is plain code at the matching point in the loop you write.

| Middleware hook | Where it goes in your loop |
| --- | --- |
| `before_model` | a statement before `await bound.generate_one(messages)` |
| `after_model` | a statement after it, inspecting the `Response` (`stop_reason`, `tool_calls`, `usage`) |
| `modify_model_request` | `bound = bound.rebind(...)` before the next turn |
| `wrap_tool_call`, tool error handling | wrap your own code around the `dispatch` call; `dispatch` answers a bad name or bad arguments with an is_error `ToolMessage` |
| running one turn's tool calls concurrently | `ToolManager.dispatch_many(tool_calls)` (see `02_tool_loop.py`) |
| human-in-the-loop / interrupts | `dispatch_many(tool_calls, precomputed=...)`; the `ToolMessage` you return replaces the tool run (see `approve_or_deny` in `02_tool_loop.py`) |
| summarization / message trimming | edit the `messages` list you hold before the next turn |
| structured-output middleware | `bind(response_format=Model)`; a refusal or a truncation raises a `GenerationError` |
| usage / cost tracking | bill on `response.usage.cost_in_usd`, the paid total across every attempt. `response.usage_successful_attempt` is the kept answer's own usage, smaller wherever a failed attempt billed. Or `to_tables(results)` for a calls table and an attempts table |

## From LangGraph

`StateGraph` nodes and conditional edges are plain control flow in the loop you own.
The middleware table above covers the `create_agent` hooks.

| LangGraph | langchaint |
| --- | --- |
| summing `AIMessage.usage_metadata` across turns to bill a run | `Usage.sum_of(response.usage for response in responses)`, then `.cost_in_usd`. `TimedOutError.usage` accounts for a call its deadline cut off |
| a per-call deadline as `awrap_model_call` middleware | `timeout_seconds` on the call it bounds (see `04_failures_and_deadlines.py`) |
| `get_stream_writer()` and `astream(subgraphs=True)` stream a nested sub-agent progress tree to one consumer with no reference passing | no counterpart: the application owns the event stream (see `events.py` and `task_stream.py` in `examples/full_app`) |

## Retries and rate limiting: one SharedBackoff

There is no retry setting on a generate call and no rate-limit middleware.
`max_attempts` on `bind` bounds retrying for that `BoundLLM`.
One account constructs one `SharedBackoff` for every `LLM` it creates.
`max_concurrent_requests` bounds requests inside `admitted()` blocks.
A provider rate limit pauses every request using that account.

`max_request_starts_per_second` limits request starts while demand is queued.
Set it with `max_concurrent_requests` when constructing an account.

```python
from langchaint.openai import OpenAIAccount

async with OpenAIAccount(
    max_concurrent_requests=16,
    max_request_starts_per_second=5,
) as openai:
    bound = openai.model("gpt-5.6-terra").bind(
        max_attempts=5,
        automatic_prompt_caching=False,
    )
```

## Fallbacks: a try/except over two bindings

There is no `.with_fallbacks`. A fallback is app code.
Over `generate_one` it is a `try`/`except GenerationError` around a second binding.
Over `generate_many` the failure is already a value in the returned list, so the fallback is an `isinstance` branch.
`04_failures_and_deadlines.py` shows that form.

## Errors: success is a Response, failure is a GenerationError

`generate_one` returns a `Response` on success and raises a `GenerationError` on a terminal failure.
Catch `GenerationError` to handle them all at once.
`langchaint.exceptions` names each subclass and the condition for it.
`generate_many` returns each failure in place of that item's `Response` instead of raising.
Reading a batch's failures back is in `04_failures_and_deadlines.py`.
