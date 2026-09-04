# Progress events from a multi-agent app

This example implements a ReAct loop and a three-level agent graph.
langchaint supplies generation, tools, accounting, deadlines, and tracing.
The application supplies the loop and progress events.

`task_stream.py` contains the application.
Its synchronous `on_event: Callable[[Event], None]` callback reports progress.
The callback is not `BoundLLM.stream_one`.

## Provider construction

`run_live_task_stream.py` uses one `OpenAI` throughout the application lifetime.
It uses `gpt-5.6-terra` for every run.
Model responses still determine the loop path.

## Graph

`research_climate` and `research_energy` run concurrently.
`synthesize` starts after both settle.
`research_climate` can call `delegate`.
Each `delegate` call creates a `specialist` run.

Each run registers under `agent_path` during construction.
A delegated path includes its creation index.
For example, the first specialist uses `root/research_climate/specialist#0`.

`AgentRun.own_usage` folds one run's `turn_log`.
`AgentRun.usage` includes descendant runs selected by `agent_path`.

## Required configuration

`AgentConfig.automatic_cache_breakpoints` has no default and passes unchanged to each binding.
`App.capture_message_content` has no default and passes unchanged to each tracing wrapper.

`AgentConfig.generate_one_timeout_seconds` becomes `generate_one(timeout_seconds=...)`.
That deadline includes admission, retries, and provider work.
`max_turns` bounds repeated `TimedOutErrorRecord` outcomes.

`AgentConfig.max_cost_in_usd` is optional.
The loop checks `cost_in_usd` before each turn and stops at the configured value.
The loop raises on a NaN `cost_in_usd` instead of continuing with unknown cost.

## Progress events

Every event carries `agent_path`.
The remaining accounting fields are exact:

| event | accounting fields |
| --- | --- |
| `AgentStarted` | none |
| `TurnStarted` | `usage_so_far` |
| `LlmResponse` | `usage`, `usage_so_far` |
| `ToolCalled` | `usage_so_far` |
| `ToolResponse` | `reported_usage`, `usage_so_far` |
| `ToolProgress` | none |
| `LlmCallAbandoned` | `usage_so_far` |
| `AgentFinished` | `usage` |
| `AgentFailed` | `usage` |
| `AgentCancelled` | `usage` |

`usage` on `LlmResponse` covers that `generate_one` call.
`usage` on terminal events covers that run and its descendants.
`usage_so_far` covers the emitting run and its descendants.
`reported_usage` covers spend returned through `ToolOutputExplicit.app_data`.

`ToolProgress` carries no accounting field because it cannot read the current `AgentRun.usage`.

## Deadlines and cancellation

`generate_one_timeout_seconds` belongs to one `generate_one` call.
A `TimedOutErrorRecord` carries settled attempt accounting.
The loop appends `LlmFailure` before continuing.
`LlmCallAbandoned` reports that continuation.

The whole-app deadline surrounds `App.run()`.
Its cancellation propagates through the task tree.
Each active run emits `AgentCancelled` before propagation.
The event carries accounting already stored in `turn_log`.

## Tool dispatch

`max_tool_calls` counts tool-call positions.
The first affordable positions dispatch.
Later positions receive `DispatchPrecomputed` outcomes.
After the count is spent, the next generation uses `tool_choice="none"` while retaining the bound tool definitions.
Repeated `ToolCall.id` values raise before dispatch.
That validation keeps partial-outcome correlation unambiguous.

`dispatch_many` returns complete outcomes in tool-call order.
`DispatchExceptionGroup.completed_outcomes` contains only settled calls.
Those partial outcomes correlate through validated `tool_call_id` values.
The loop records settled `app_data` before re-raising the defect.

`DispatchExceptionGroup` always propagates from `delegate` and `_settle_node`.
Other sub-agent failures become parent-readable tool errors, so the parent can finish without the failed answer.

## Tracing

`TracedLLM` and `TracedToolManager` create generation and tool spans.
A delegated run's generation and tool spans become children of its `delegate` tool span.

`capture_message_content` controls message content on generated spans.
OpenTelemetry configuration controls recording and export.
