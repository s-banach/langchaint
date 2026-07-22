# Streaming a multi-agent app without callbacks

The reference architecture for a multi-agent application on langchaint: a ReAct loop with sub-agents, a
graph, and three layers of timeout streams its progress to a UI and keeps its token accounting through
every failure, without the application handing langchaint a callback. langchaint deliberately ships no
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
`synthesize` self-corrects: it drafts, calls `critique`, is told to revise, drafts again, is approved, answers.
`search` reports a flat per-call fee through `app_data`, so a run's total covers tool spend as well as token
spend, and reports its progress mid-call through the ambient `GuiEmitter`. A sub-agent reports no fee that
way: it writes its own `spend_log` as it goes.

Each agent is constructed from an `AgentConfig` (`config.py`) fixing its `max_turns`, `max_tool_calls`,
`timeout_seconds`, `per_call_timeout_seconds`, and `self_correction_enabled`.

Three deadlines nest, and each catches a failure the others cannot:

- `per_call_timeout_seconds` bounds one provider request, an ordinary `asyncio.timeout` around `generate_one`. The request is dropped and the loop goes on to its next turn with the same conversation, since an abandoned call appended nothing to it.
- `timeout_seconds` bounds a whole run, so an agent making fast progress toward nothing still stops.
- The whole-app deadline is applied by the consumer around the event iteration.

Only the innermost one leaves a run able to continue, and that is what makes it a different limit rather
than a shorter spelling of the next one out. A run whose every call hangs is bounded by `max_turns`.

`LlmCallAbandoned` is the event that reports a dropped request, so a UI shows the gap rather than a turn
that silently produced nothing.

A cancellation leaves a hole no client can close: the cancelled call may have completed and billed
server-side, so its cost is unobservable. Each run's `spend_log` is passed to `generate_one` as its
`abandoned_call_log`, so whichever deadline cancels a call, langchaint appends an `AbandonedCall` row
before the unwind; counting only the innermost deadline's drops would report zero abandoned calls for
a tree the app deadline cut off mid-request.

Cancelling a run is not the same as it having unwound, so a consumer that abandons the iteration awaits
`app.settle()` before reading the totals. Without it the accounting is read one step before the deadline,
with the cancelled calls' `AbandonedCall` rows still pending.

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

`test_task_stream_claims.py` asserts the claims the docstrings make, the two above all: a `run()`
written as a generator is rejected at construction, and a consumer slower than the run's deadline does
not kill the run.

## The shape

`AgentRun` (`task_stream.py`) splits an agent into two roles. The base class owns the streaming: the
queue, the iterator, `final()`, the run deadline, and the `invoke_agent` span. The application
subclasses it and writes only `run()`, a plain coroutine, calling `self.emit(event)` as it goes. An
agent is directly consumable (`async for` it, or `await final()`), a slow consumer never counts against
the run's deadline, and the one async generator in the design is `_drain`, which holds nothing across
its `yield` and contains no application code.

Two capabilities ride on contextvars, the same mechanism that nests OTel spans across tasks:

- Each run's task installs its own `GuiEmitter` (`events.py`), so a tool function reports progress into
  the stream of whichever run dispatched it, with no emitter parameter on any tool. `search` does this,
  and its progress lands in the specialist's stream when the specialist calls it and in a researcher's
  when a researcher does.
- `delegate` forwards its sub-run's events to that same emitter, so the specialist's stream interleaves
  into the top-level consumer's in real time while `delegate` stays an ordinary langchaint tool. A
  streaming sub-agent cannot be a tool any more directly than this: a `Tool` returns a value and cannot
  yield.

Only the emitter is ambient. The run registry stays a constructor argument, where a missing value is a
`TypeError` at the call site instead of a `LookupError` at the first read inside an agent; the
emitter is ambient because a tool function has no call site to thread it through.

## The event vocabulary

`events.py` holds one frozen dataclass per event, matched by type. Every event a run emits after its
first generate carries `usage_so_far`, that run's running total including its sub-agents, so a UI redraws
one agent's token counter from the event alone. `ToolCalled` is emitted before dispatch with the model's
raw `args_json`, and `ToolResponse` settles it by `tool_call_id`. `ToolProgress` carries no usage,
because a tool function does not know its run's total. `TurnStarted` carries the `turn_number` that
`max_turns` bounds.

## Why this shape

**No callback is needed anywhere.** The consumer controls pacing, so a slow UI never blocks inside a
callback the orchestrator invoked, and an exception surfaces at the consumer's `await` rather than
inside a callback with nowhere to raise.

**The state that matters must not live in a coroutine frame.** A timeout cancels the frame; anything
local to it is gone. The conversation, the counters and the failure live on the `AgentRun` object the
caller holds.

**Spend rows are written where the spend happens, and no total is a running sum.** A whole-app
cancellation raises `CancelledError`, a `BaseException` that passes through every `except Exception`,
so a design that carries a sub-run's spend home on a return path loses it when that path never runs.
Each row goes to the run's own `spend_log` at the moment it happens, every run registers itself in the
registry at construction (so a run cancelled mid-flight is still in the report), and a parent's total
is a fold over the registered runs' logs by path prefix. Nothing has to be carried, so nothing can be
dropped, and totals fold from the rows on demand, so there is no second copy to drift.

**Each accounting row rides the value that carries it, except the one that has no value.** A returned
`Response` and a raised `GenerationError` carry their own `usage`; appending those rows is the loop's
job. A cancelled call returns nothing and raises nothing the loop can keep, so `generate_one` appends
its `AbandonedCall` row to the `spend_log` passed as `abandoned_call_log`, inside the frame the
cancellation unwinds, which no wrapper outside the call could do.
`DispatchExceptionGroup.completed_outcomes` keeps settled siblings' `app_data` when another tool
raised, and it is worthless unhandled: an app that awaits `dispatch_many` with no handler loses its
settled siblings' reported spend when a tool raises, because the money is on the exception and nobody
reads it. The loop catches the group, folds `completed_outcomes` matched by `tool_call_id` (the group
covers only the calls that settled, so no index lines up), and re-raises.

**A sub-agent failure should be data, not an exception.** `delegate` catches and returns an `is_error`
tool message. The parent model reads it and adapts, and in `subagent_error` the parent still produces a
correct final answer. The catch decides the conversation only: the spend was recorded as it happened, so
deleting the `except` would cost an answer, never a row.

**OTel tracing needs a span the app owns.** langchaint spans one generate call and one tool dispatch and
has no concept of a run, so a multi-turn agent's generate spans would arrive as siblings with nothing
tying them together. `agent_span` (in `langchaint.tracing`) opens the `invoke_agent` span around the
whole loop, in the object that owns the deadline and for the same reason: a caller cannot forget it.
Sub-agent nesting needs no plumbing, because OTel context is a contextvar and `create_task` copies the
creating context, which `test_task_stream_claims.py` asserts rather than assumes.

**A contextvar buys one thing threading cannot: reach without a reference.** A tool function holds no
handle on the run that dispatched it, so an `emit()` documented as callable from inside a tool is
unusable without one. What it costs is where the missing value shows up: a missing constructor argument
is a `TypeError` at the call site, a missing contextvar is a `LookupError` at the first read, at runtime,
on whichever path ran first. `current_gui_emitter` therefore raises with a message naming what to
install and where, code that exists only because the mechanism is ambient. Two rules keep it correct:
install the value in the run's own task before anything reads it (a value set after `create_task` is
invisible inside that task), and read an ambient value at the boundary, then hold it.

## Rejected shapes

**The agent as an async generator**: the shape an application reaches for first, yielding events from
inside the loop. An async generator runs in whichever context its consumer resumes it from, so a run
deadline or a span entered around a `yield` is open while the consumer runs: the deadline measures the
consumer's pace and cancels into the consumer's frame, and the span leaks into whatever the consumer
opens next. Every failure mode is silent and needs a slow or abandoned consumer to appear. It also makes
the run's deadline unavoidably a deadline on the consumer. `task_stream.py` makes the hazard unwritable
instead: a `yield` in `run()` is rejected at construction.

**One app-owned bus**: every run pushes to one shared channel the app drains. Fewer moving parts, but an
agent is not consumable on its own: its events reach a UI only through the bus the app owns, so an
application wanting one agent's stream builds the whole apparatus. `task_stream.py` keeps the same
coroutine-shaped loop and gives each run its own queue instead.

**Everything ambient**: the run registry through a contextvar too, not just the emitter.
It removes a constructor argument that never varies per run, at the price of turning a missing value
into a runtime `LookupError`; the emitter is the one value that earns that price, because tools cannot
be threaded a reference.

## What would change the recommendation

The shape assumes two things this app happens to be true of, and an application they are false of should
choose differently.

**That `timeout_seconds` is a budget on work, not an end-to-end promise.** `task_stream.py` excludes
consumer latency from the run's deadline. That is correct for a budget on provider spend and wall-clock
work. If the deadline encodes "this user sees an answer within 30 seconds", then consumer time is part
of what was promised, and a pull-driven shape measures the thing that matters while this one measures a
proxy.

**That the consumer is a reader, not the work.** A run here reports itself finished when its loop stops,
not when the consumer has drained it, and its queue is unbounded (bounded in practice by `max_turns` and
`max_tool_calls`; it becomes a memory question only when events carry large payloads). Where the
consumer is itself the work (writing rows, feeding a rate-limited socket), a run that "completed" with
events still queued has not completed anything, and a bounded queue is what the application wants; that
puts consumer pace back inside the deadline, which the pull-driven shape has by construction.
