# Progress events from a multi-agent app

This example implements a ReAct loop and a three-level agent graph.
langchaint supplies generation, tools, accounting, deadlines, and tracing.
The application supplies the loop and progress events.

`task_stream.py` contains the application.
Its callback is `on_event: Callable[[Event], None]`.
The callback reports progress synchronously.
It is not `BoundLLM.stream_one`.

## Run offline

`harness.py` provides a scripted adapter.
It sends no network requests.

```console
uv run python examples/full_app/run_task_stream.py
```

`run_task_stream.py` executes the committed `SCENARIOS` table.

| scenario | behavior |
| --- | --- |
| `happy` | Every run completes. |
| `subagent_error` | A provider failure becomes a parent-readable tool error. |
| `call_timeout` | One call times out, then the run continues. |
| `app_timeout` | The whole-app deadline cancels active runs. |
| `tool_budget` | `max_tool_calls` declines later calls. |
| `unapproved_answer` | `synthesize` requests critique before accepting an answer. |

## Run live

`run_live_task_stream.py` uses one `OpenAI` throughout the application lifetime.
It uses `gpt-5.6-terra` for every run.

```console
OPENAI_API_KEY=... uv run python examples/full_app/run_live_task_stream.py
```

The live run uses canned application tools.
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

`AgentConfig.automatic_cache_breakpoints` has no default.
Each binding receives that exact value.
`App.capture_message_content` has no default.
Each tracing wrapper receives that exact value.

`AgentConfig.generate_one_timeout_seconds` becomes `generate_one(timeout_seconds=...)`.
That deadline includes admission, retries, and provider work.
`max_turns` bounds repeated `TimedOutError` outcomes.

`AgentConfig.max_cost_in_usd` is optional.
The loop checks it before each turn.
The loop stops when `cost_in_usd` reaches the configured value.
A NaN `cost_in_usd` cannot satisfy a configured limit.
The loop raises instead of continuing with unknown cost.

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

`ToolProgress` cannot read the current `AgentRun.usage`.
It therefore carries no accounting field.

## Deadlines and cancellation

`generate_one_timeout_seconds` belongs to one `generate_one` call.
A `TimedOutError` carries settled attempt accounting.
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
Repeated `ToolCall.id` values raise before dispatch.
That validation keeps partial-outcome correlation unambiguous.

`dispatch_many` returns complete outcomes in tool-call order.
`DispatchExceptionGroup.completed_outcomes` contains only settled calls.
Those partial outcomes correlate through validated `tool_call_id` values.
The loop records settled `app_data` before re-raising the defect.

`DispatchExceptionGroup` always propagates from `delegate` and `_settle_node`.
Other sub-agent failures become parent-readable tool errors.
The parent can then finish without the failed sub-agent answer.

## Tracing

`AgentRun.final` opens one `agent_span` around the loop.
Its `generate_one` spans become children of that span.
A delegated run starts inside the `delegate` tool span.
Its `agent_span` therefore becomes that tool span's child.

`capture_message_content` controls message content on generated spans.
OpenTelemetry configuration controls recording and export.

## Tests

`test_task_stream_claims.py` runs offline.
It verifies accounting, cancellation, deadlines, event routing, and span nesting.
It also verifies NaN cost rejection and tool-defect propagation.
