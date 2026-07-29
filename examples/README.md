# Examples

Short, runnable examples of langchaint.
Each file is a set of small async functions with a `__main__` guard; they read top to bottom and use real API calls, so running one needs the matching SDK installed and the provider's API key in the environment.
The `openai` package and `OPENAI_API_KEY` cover the openai examples.
`05_rate_limiting_and_errors.py`, `06_prompt_caching.py`, `09_batch_failures.py`, and `10_deadlines.py` build anthropic models, so they need the `anthropic` package and `ANTHROPIC_API_KEY`.
`07_json_schema_tool_validation.py` needs no API key: it dispatches constructed `ToolCall`s with no provider involved.
`full_app/` needs none either: its adapter is scripted and offline.
Where a tool's specifics do not matter, the code uses a minimal tool (a canned weather lookup, a canned search) rather than a realistic one.

| File | Shows |
| --- | --- |
| [`01_basics.py`](01_basics.py) | construct a model, `bind`, `generate_one`, structured output via `response_format`, `rebind`, and `generate_many` + `to_tables` |
| [`02_tool_loop.py`](02_tool_loop.py) | the ReAct loop over `generate_one` and `ToolManager.dispatch`, the three dispatch outcomes, `app_data`, and an approval gate as an optional argument to the same loop |
| [`03_streaming.py`](03_streaming.py) | `stream_one`, the `str \| ReasoningDelta \| ToolCall` iterator, `final()` for usage and cost, and the streaming tool loop |
| [`04_tracing.py`](04_tracing.py) | OTel telemetry with `TracedLLM` and a span exporter |
| [`05_rate_limiting_and_errors.py`](05_rate_limiting_and_errors.py) | one shared `RateLimiter` across an openai and an anthropic model, catching a `GenerationError`, and a try/except fallback |
| [`06_prompt_caching.py`](06_prompt_caching.py) | `cache_breakpoint` marks in the frozen prefix, the anthropic 4-marker budget and `cache_ttl`, openai's implicit/explicit modes, and the marks each provider rejects |
| [`07_json_schema_tool_validation.py`](07_json_schema_tool_validation.py) | `JSONSchemaTool` argument validation: `dispatch` validates the arguments against `args_schema`, landing schema violations in the same `DispatchInvalidToolArgs` house message as the `PydanticTool` path |
| [`08_required_choice_and_limits.py`](08_required_choice_and_limits.py) | the budgeted `tool_choice="required"` loop: a structured exit captured through a `CaptureTool`, `SpecificToolChoice` forcing that exit when `max_turns` is spent, a tool budget fed by `Usage` reported as `app_data`, and a whole sub-agent loop wrapped as one tool |
| [`09_batch_failures.py`](09_batch_failures.py) | a batch whose middle item the adapter will not send: every slot settles, `to_tables` renders all three, `Usage.sum_of` totals the spend over successes and failures, and the failed slots map back to the conversations to resubmit |
| [`10_deadlines.py`](10_deadlines.py) | `timeout_seconds` on one call, on every item of a batch, chained across calls against one monotonic budget, and over a whole `stream_one` block, plus what an `asyncio.timeout` of your own costs you |
| [`MIGRATING_FROM_LANGCHAIN.md`](MIGRATING_FROM_LANGCHAIN.md) | the call-for-call API map and what replaces the middleware layer |
| [`full_app/`](full_app/README.md) | the reference architecture for a streaming multi-agent app: `AgentRun`, sub-agents as tools, a per-call `timeout_seconds` inside a whole-app `asyncio.timeout`, and accounting that survives every failure |

Start with `MIGRATING_FROM_LANGCHAIN.md` for the mental model, then `01_basics.py`; the centerpiece is `02_tool_loop.py`, because the loop LangChain's agent classes hide is the one langchaint expects you to write yourself.
`full_app/` is where those pieces compose into a whole application.
