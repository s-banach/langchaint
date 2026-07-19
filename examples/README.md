# Examples

Short, runnable examples of langchaint.
Each file is a set of small async functions with a `__main__` guard; they read top to bottom and use real API calls, so running one needs the matching SDK installed and the provider's API key in the environment.
The `openai` package and `OPENAI_API_KEY` cover the openai examples.
`05_rate_limiting_and_errors.py` and `06_prompt_caching.py` also build anthropic models, so they additionally need the `anthropic` package and `ANTHROPIC_API_KEY`.
`07_json_schema_tool_validation.py` is the one exception to needing an API key: it dispatches constructed `ToolCall`s with no provider involved.
Where a tool's specifics do not matter, the code uses a minimal tool (a canned weather lookup, a canned search) rather than a realistic one.

| File | Shows |
| --- | --- |
| [`01_basics.py`](01_basics.py) | construct a model, `bind`, `generate_one`, structured output via `response_format`, `rebind`, and `generate_many` + `to_row` |
| [`02_tool_loop.py`](02_tool_loop.py) | the ReAct loop over `generate_one` and `ToolManager.dispatch`, the three dispatch outcomes, `app_data`, and an approval gate as an optional argument to the same loop |
| [`03_streaming.py`](03_streaming.py) | `stream_one`, the `str \| ToolCall` iterator, `final()` for usage and cost, and the streaming tool loop |
| [`04_tracing.py`](04_tracing.py) | OTel telemetry with `TracedLLM` and a span exporter |
| [`05_rate_limiting_and_errors.py`](05_rate_limiting_and_errors.py) | one shared `RateLimiter` across an openai and an anthropic model, catching a `GenerationError`, and a try/except fallback |
| [`06_prompt_caching.py`](06_prompt_caching.py) | `cache_breakpoint` marks in the frozen prefix, the anthropic 4-marker budget and `cache_ttl`, openai's implicit/explicit modes, and the marks each provider rejects |
| [`07_json_schema_tool_validation.py`](07_json_schema_tool_validation.py) | `JSONSchemaTool` argument validation: `dispatch` validates the arguments against `args_schema`, landing schema violations in the same `DispatchInvalidToolArgs` house message as the `PydanticTool` path |
| [`08_required_choice_and_limits.py`](08_required_choice_and_limits.py) | the budgeted `tool_choice="required"` loop: a structured exit captured through a `CaptureTool`, `SpecificToolChoice` forcing that exit when `max_turns` is spent, a tool budget fed by `Usage` reported as `app_data`, and a whole sub-agent loop wrapped as one tool |
| [`MIGRATING_FROM_LANGCHAIN.md`](MIGRATING_FROM_LANGCHAIN.md) | the call-for-call API map and what replaces the middleware layer |

Start with `MIGRATING_FROM_LANGCHAIN.md` for the mental model, then `01_basics.py`; the centerpiece is `02_tool_loop.py`, because the loop LangChain's agent classes hide is the one langchaint expects you to write yourself.
