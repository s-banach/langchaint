# Streaming a multi-agent app

The reference architecture for a multi-agent application on langchaint: a ReAct loop with sub-agents, a
graph, and two layers of timeout streams its progress to a UI and keeps its token accounting through
every failure.
langchaint ships no agent loop, so the loop is application code.
`task_stream.py` is the shape to copy for it, and the rest of these files are the app around it.

Unlike the numbered examples, this one runs offline: `harness.py` is a scripted adapter, so nothing
reaches a network and no API key is needed.

```
uv run python examples/full_app/run_task_stream.py
```

## The app

`research_climate` and `research_energy` run concurrently; `synthesize` starts once both settle.
`research_climate` delegates to a `specialist` sub-agent, so the run tree is three levels deep.
Each `delegate` call spawns a fresh run at `{parent}/{name}#{spawn_index}`: an agent's name is not
unique within a parent, and a registry row is one run held by reference, so identity is per spawn.
`synthesize` self-corrects: it drafts, calls `critique`, is told to revise, drafts again, is approved, answers.
`search` reports a flat per-call fee through `app_data`, so a run's total covers tool spend as well as token
spend, and reports its progress mid-call through the ambient `GuiEmitter`. A sub-agent reports no fee that
way: it writes its own `turn_log` as it goes.

Each agent is constructed from an `AgentConfig` (`config.py`) fixing its `max_turns`, `max_tool_calls`,
`timeout_seconds`, and `self_correction_enabled`.

Two deadlines apply, and they differ in what a run can do afterwards:

- `timeout_seconds` bounds one provider request, handed to `generate_one` as its own `timeout_seconds`. langchaint owns that scope, so a request that runs out comes back as a `TimedOutError` while the loop is still running: the loop records it and takes its next turn with the same `messages`, which the dropped call left untouched.
- The whole-app deadline is the consumer's, an `asyncio.timeout` around `await app.run()`. It cancels, so nothing continues past it.

A run whose every call hangs is bounded by `max_turns`.

`LlmCallAbandoned` is the event that reports a dropped request, so a UI shows the gap rather than a turn
that silently produced nothing.

A dropped request leaves a hole no client can close: it may have completed and billed server-side
after the deadline landed, and that remainder is unobservable. What is observable is on the
`TimedOutError` the loop appends to `turn_log`.

The whole-app deadline is where the accounting stops. A call in flight when it lands dies with the
frame holding its account, so what survives is every turn that had already settled; the whole tree
runs in one task tree under `App.run`, so those are all written before the `TimeoutError` reaches the
consumer.

## The scenarios

`run_task_stream.py` plays the app through six scenarios, each exercising one failure layer:

| scenario | exercises |
|---|---|
| `happy` | every agent completes; the specialist's spend lands inside its parent's subtree |
| `subagent_error` | the specialist raises mid-run; the parent reads an `is_error` tool message and still answers |
| `call_timeout` | one request outruns `timeout_seconds`; the run records it and answers on the next turn |
| `app_timeout` | the whole-app deadline cuts both researchers off mid-request; the settled accounting survives |
| `tool_budget` | `max_tool_calls=1` declines the remaining calls and the model finishes with what it has |
| `unapproved_answer` | `synthesize` answers uncritiqued and the self-correction bounce sends it back |

`test_task_stream_claims.py` asserts the claims the docstrings make, the two above all: a call its own
deadline cut off is on `turn_log` and its run carries on, and a tool function's progress reaches the
`on_event` of whichever run dispatched it.

## The shape

Every run is constructed with one `on_event: Callable[[Event], None]` and reports by calling it;
nothing in the example is an async iterator.
`task_stream.py`'s module docstring is the spec of the mechanics:
the `AgentRun` split between `final()` and `run()`,
the `on_event=queue.put_nowait` decoupling a consumer can choose,
and the two capabilities that ride on contextvars.

`on_event` returns `None` and runs in the run's frame, so it must be quick and synchronous.
A consumer that blocks in it slows the run down, and it has no way to await.
One that is slow, or must await per event, passes `on_event=queue.put_nowait` and drains the queue in its own task.
That queue is unbounded, and a run finishes with events still in it.

## The event vocabulary

`events.py` holds one frozen dataclass per event, matched by type. Every event a run emits after its
first generate carries `usage_so_far`, that run's running total including its sub-agents, so a UI redraws
one agent's token counter from the event alone. `ToolCalled` is emitted before dispatch with the model's
raw `args_json`, and `ToolResponse` settles it by `tool_call_id`. `ToolProgress` carries no usage,
because a tool function does not know its run's total. `TurnStarted` carries the `turn_number` that
`max_turns` bounds.

## Why this shape

**One task tree, so the app deadline's `except` reads every settled turn.** A callback adds no task and no
queue per run, so the whole graph runs under `App.run`'s frame and its `TaskGroup`. The cancellation
unwinds every child before it propagates, which is what lets the consumer read the totals
right in its `except` with no settling step: cancelling a task it does not await would not be the same
as that task having unwound.

**The state that matters must not live in a coroutine frame.** A timeout cancels the frame; anything
local to it is gone. The `messages`, the counters and the turn records live on the `AgentRun` object;
the answer and the failure ride `final()`'s return and raise, which the caller is awaiting.

**Records are written where things happen, and no total is a running sum.** A whole-app
cancellation raises `CancelledError`, a `BaseException` that passes through every `except Exception`,
so a design that carries a sub-run's records home on a return path loses them when that path never runs.
Each `TurnRecord` goes to the run's own `turn_log` at the moment it happens, and every run registers itself in the registry at construction, so a run cancelled mid-flight is still in the record.
Any metric a consumer could possibly want is a post-run fold over the registered runs' ordered logs; a parent's subtree total is the fold filtered by path prefix.
Nothing has to be carried, so nothing can be dropped, and nothing is stored twice, so there is no second copy to drift.

**Ask langchaint for the deadline instead of wrapping the call in one.** A cancelled call carries
nothing back: it returns no value and raises nothing the loop can keep, because the scope that
converts the cancellation sits above the frame holding the call's account. Passing
`timeout_seconds` puts that scope inside the frame instead, so a request that runs out arrives as a
`TimedOutError` carrying its `usage`, and one `except GenerationError` records the returned
`Response`, an ordinary failure, and a drop the same way.
`DispatchExceptionGroup.completed_outcomes` keeps settled siblings' `app_data` when another tool
raised, and it is worthless unhandled: an app that awaits `dispatch_many` with no handler loses its
settled siblings' reported spend when a tool raises, because the money is on the exception and nobody
reads it. The loop catches the group, folds `completed_outcomes` matched by `tool_call_id` (the group
covers only the calls that settled, so no index lines up), and re-raises.

**A sub-agent failure should be data, not an exception.** `delegate` catches and returns an `is_error`
tool message. The parent model reads it and adapts, and in `subagent_error` the parent still produces a
correct final answer. The catch decides the `messages` only: the spend was recorded as it happened, so
deleting the `except` would cost an answer, never a record.

**OTel tracing needs a span the app owns.** langchaint spans one generate call and one tool dispatch and
has no concept of a run, so a multi-turn agent's generate spans would arrive as siblings with nothing
tying them together. `agent_span` (in `langchaint.tracing`) opens the `invoke_agent` span around the
whole loop, in `final()` rather than in the loop, so a caller cannot forget it.
Sub-agent nesting needs no plumbing, because the sub-run's `final()` runs in the frame `delegate` is
awaiting in, which is the frame the tool span is current in; `test_task_stream_claims.py` asserts that
rather than assumes it.

**A contextvar buys one thing threading cannot: reach without a reference.** A tool function holds no
handle on the run that dispatched it,
so a progress report documented as sendable from inside a tool is unusable without one.
What it costs is where the missing value shows up: a missing constructor argument
is a `TypeError` at the call site, a missing contextvar is a `LookupError` at the first read, at runtime,
on whichever path ran first. `current_gui_emitter` therefore raises with a message naming what to
install and where, code that exists only because the mechanism is ambient. Two rules keep it correct:
install the value before anything reads it and reset it on the way out (`final()` does both, and a
dispatch task created inside the run inherits the value because `create_task` copies the creating
context), and read an ambient value at the boundary, then hold it.
