# Progress events from a multi-agent app

This example implements a ReAct loop with concurrent research, delegation, and synthesis.
`research_climate` and `research_energy` run concurrently.
`synthesize` starts after both settle.
`research_climate` can delegate questions to fresh `specialist` runs.

Run the live example with:

```sh
uv run examples/full_app/run_live_task_stream.py
```

The entrypoint uses one `OpenAI` with `gpt-5.6-terra` and prints progress through `App(on_event=...)`.
The local tools return fixed search results and scripted critique verdicts.

| File | Responsibility |
| --- | --- |
| `task_stream.py` | Agent execution, graph scheduling, and accounting |
| `config.py` | Agent prompts and limits |
| `tools.py` | Search, critique, and delegation arguments |
| `events.py` | Progress events and tool progress routing |
| `render.py` | Event display |

## Architecture

langchaint handles generation, request retries, and tool dispatch; `App` schedules runs, and `ReActAgent` owns the message history and turn loop.
`AgentRun` is a base class defined by this demo app for registration, accounting, progress routing, and lifecycle events.
The app's agent classes, including `ReActAgent`, inherit from `AgentRun`, implement their own workflow in `_execute()`, and record settled outcomes in `turn_log`.
Callers use the inherited `run()` method, which routes tool progress and emits `AgentStarted`, `AgentFinished`, `AgentFailed`, and `AgentCancelled` lifecycle events.
All runs share one synchronous `on_event` callback for application progress; `BoundLLM.stream_one` provides model token streaming.
`GuiEmitter` uses a context variable that follows async tasks to route tool progress to the current run.
Each delegated run installs its own emitter, so concurrent tools report the correct `agent_path`.

Each run keeps settled outcomes in its own `turn_log` and registers under a unique `agent_path`; repeated delegates receive spawn indexes.
Parent usage includes registered descendants, so the delegate tool supplies no app_data to avoid counting specialist usage twice.
Dispatch records enter the log before `ToolDispatched` callbacks run, preserving completed accounting if one of those callbacks fails.
Each dispatch event retains the usage snapshot captured after its corresponding entry.

An input timeout records its accounting and lets the turn loop continue; the outer application deadline cancels active runs.
`App` records classified generation failures and configured limits as failures; `delegate` returns them as tool errors so its parent can continue.
Tool defects, callback defects, and unknown generation failures propagate, with recorded accounting available to the caller.

## Tracing

The entrypoint attaches `OtelObserver(capture_message_content=False)`.
The observer creates generation and tool spans; a delegated run's spans inherit the delegate tool span as their parent.
Configure an OTel SDK tracer provider to record and export spans, as shown in [`08_tracing.py`](../08_tracing.py).
