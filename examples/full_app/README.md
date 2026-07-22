# Streaming a multi-agent app

The reference architecture for a multi-agent application on langchaint: a ReAct loop with sub-agents, a
graph, and three layers of timeout streams its progress to a UI and keeps its token accounting through
every failure. langchaint deliberately ships no
agent loop, so the loop is application code; `task_stream.py` is the shape to copy for it, and the rest
of these files are the app around it.

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
`timeout_seconds`, `per_call_timeout_seconds`, and `self_correction_enabled`.

Three deadlines nest, and each catches a failure the others cannot:

- `per_call_timeout_seconds` bounds one provider request, an ordinary `asyncio.timeout` around `generate_one`. The request is dropped and the loop goes on to its next turn with the same conversation, since an abandoned call appended nothing to it.
- `timeout_seconds` bounds a whole run, so an agent making fast progress toward nothing still stops.
- The whole-app deadline is applied by the consumer, an `asyncio.timeout` around `await app.run()`.

Only the innermost one leaves a run able to continue, and that is what makes it a different limit rather
than a shorter spelling of the next one out. A run whose every call hangs is bounded by `max_turns`.

`LlmCallAbandoned` is the event that reports a dropped request, so a UI shows the gap rather than a turn
that silently produced nothing.

A cancellation leaves a hole no client can close: the cancelled call may have completed and billed
server-side, so its cost is unobservable. Each run's `turn_log` is passed to `generate_one` as its
`abandoned_call_log`, so whichever deadline cancels a call, langchaint appends an `AbandonedCall`
record before the unwind; counting only the innermost deadline's drops would report zero abandoned
calls for a tree the app deadline cut off mid-request.

The totals are final the moment the deadline's `except` runs: the whole tree runs in one task tree
under `App.run`, so the cancellation has unwound every frame, `AbandonedCall` records included, before
the `TimeoutError` reaches the consumer.

## The scenarios

`run_task_stream.py` plays the app through seven scenarios, each exercising one failure layer:

| scenario | exercises |
|---|---|
| `happy` | every agent completes; the specialist's spend lands inside its parent's subtree |
| `subagent_error` | the specialist raises mid-run; the parent reads an `is_error` tool message and still answers |
| `call_timeout` | one request outruns `per_call_timeout_seconds`; the run drops it and answers on the next turn |
| `agent_timeout` | no single call trips the per-call limit, but cumulative time crosses `timeout_seconds` |
| `app_timeout` | the whole-app deadline cuts both researchers off mid-request; the partial accounting survives |
| `tool_budget` | `max_tool_calls=1` refuses the remaining calls and the model finishes with what it has |
| `unapproved_answer` | `synthesize` answers uncritiqued and the self-correction bounce sends it back |

`test_task_stream_claims.py` asserts the claims the docstrings make, the two above all: the accounting
is final the moment the app deadline's `except` runs, and a tool function's progress reaches the
`on_event` of whichever run dispatched it.

## The shape

Every run is constructed with one `on_event: Callable[[Event], None]` and reports by calling it;
nothing in the example is an async iterator.
`task_stream.py`'s module docstring is the spec of the mechanics:
the `AgentRun` split between `final()` and `run()`,
the `on_event=queue.put_nowait` decoupling a consumer can choose,
and the two capabilities that ride on contextvars.

## The event vocabulary

`events.py` holds one frozen dataclass per event, matched by type. Every event a run emits after its
first generate carries `usage_so_far`, that run's running total including its sub-agents, so a UI redraws
one agent's token counter from the event alone. `ToolCalled` is emitted before dispatch with the model's
raw `args_json`, and `ToolResponse` settles it by `tool_call_id`. `ToolProgress` carries no usage,
because a tool function does not know its run's total. `TurnStarted` carries the `turn_number` that
`max_turns` bounds.

## Why this shape

**One task tree, so a deadline's `except` reads final accounting.** A callback adds no task and no
queue per run, so the whole graph runs under `App.run`'s frame and its `TaskGroup`. A cancellation from
any deadline unwinds every child before it propagates, which is what lets the consumer read the totals
right in its `except` with no settling step: cancelling a task it does not await would not be the same
as that task having unwound.

**The state that matters must not live in a coroutine frame.** A timeout cancels the frame; anything
local to it is gone. The conversation, the counters and the turn records live on the `AgentRun` object;
the answer and the failure ride `final()`'s return and raise, which the caller is awaiting.

**Records are written where things happen, and no total is a running sum.** A whole-app
cancellation raises `CancelledError`, a `BaseException` that passes through every `except Exception`,
so a design that carries a sub-run's records home on a return path loses them when that path never runs.
Each `TurnRecord` goes to the run's own `turn_log` at the moment it happens, and every run registers itself in the registry at construction, so a run cancelled mid-flight is still in the record.
Any metric a consumer could possibly want is a post-run fold over the registered runs' ordered logs; a parent's subtree total is the fold filtered by path prefix.
Nothing has to be carried, so nothing can be dropped, and nothing is stored twice, so there is no second copy to drift.

**Each record rides the value that carries it, except the one that has no value.** A returned
`Response` and a raised `GenerationError` carry their own `usage`; appending their records is the
loop's job. A cancelled call returns nothing and raises nothing the loop can keep, so `generate_one`
appends its `AbandonedCall` record to the `turn_log` passed as `abandoned_call_log`, inside the frame the
cancellation unwinds, which no wrapper outside the call could do.
`DispatchExceptionGroup.completed_outcomes` keeps settled siblings' `app_data` when another tool
raised, and it is worthless unhandled: an app that awaits `dispatch_many` with no handler loses its
settled siblings' reported spend when a tool raises, because the money is on the exception and nobody
reads it. The loop catches the group, folds `completed_outcomes` matched by `tool_call_id` (the group
covers only the calls that settled, so no index lines up), and re-raises.

**A sub-agent failure should be data, not an exception.** `delegate` catches and returns an `is_error`
tool message. The parent model reads it and adapts, and in `subagent_error` the parent still produces a
correct final answer. The catch decides the conversation only: the spend was recorded as it happened, so
deleting the `except` would cost an answer, never a record.

**OTel tracing needs a span the app owns.** langchaint spans one generate call and one tool dispatch and
has no concept of a run, so a multi-turn agent's generate spans would arrive as siblings with nothing
tying them together. `agent_span` (in `langchaint.tracing`) opens the `invoke_agent` span around the
whole loop, in the object that owns the deadline and for the same reason: a caller cannot forget it.
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

## Rejected shapes

**The agent as an async generator**: the shape an application reaches for first, yielding events from
inside the loop. An async generator runs in whichever context its consumer resumes it from, so a run
deadline or a span entered around a `yield` is open while the consumer runs: the deadline measures the
consumer's pace and cancels into the consumer's frame, and the span leaks into whatever the consumer
opens next. Every failure mode is silent and needs a slow or abandoned consumer to appear. Here the
application never writes a generator: `run()` emits instead of yielding, and `final()` awaits it as a
coroutine, so a `yield` in `run()` fails loudly at `await self.run()`.

**The run as an async iterator**: each run owns a queue and a task driving its loop, and a consumer
`async for`s the events. It buys the decoupling this shape leaves to the consumer, and the price is
spread over everything else: a task and queue per run, a run that reports itself finished with events
still queued, and a settle step on every abandon path, because cancelling the run's task is not the
same as it having unwound and the accounting is not final until it has. The callback keeps the tree in
one task tree, and a consumer that wants the decoupling back passes `on_event=queue.put_nowait`.

**Everything ambient**: the run registry and `on_event` through contextvars too, not just the emitter.
It removes constructor arguments that never vary per run, at the price of turning a missing value
into a runtime `LookupError`; the emitter is the one value that earns that price, because tools cannot
be threaded a reference.

## What would change the recommendation

The shape assumes two things this app happens to be true of, and an application they are false of should
choose differently.

**That `on_event` is quick and synchronous.** It returns `None` and runs in the run's frame, so its
time counts against `timeout_seconds` and it has no way to await. That is correct for an `on_event`
that renders a line or appends to a list. A consumer that is slow, or must await per event (feeding a
rate-limited socket), fronts it with a queue: `on_event=queue.put_nowait` is constant-time, and the
drain loop is the consumer's own task, paced by the consumer. That choice re-creates the iterator
shape's costs for that consumer alone: the queue is unbounded, and a run finishes with events still
queued.

**That `timeout_seconds` may include the consumer's time.** With `on_event` in the run's frame, the
deadline bounds work plus reporting, and a consumer doing real work in `on_event` gets backpressure by
construction: the run cannot outrun it. If the deadline must be a budget on provider work alone,
passing `on_event=queue.put_nowait` takes the consumer's time back out of it.
