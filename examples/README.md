# Examples

Short snippets of langchaint, one file per subject.
Each numbered file is one function to read and copy into an application, so none of them has a `__main__` guard.
That function builds everything it needs.
A comment appears only where a fact is not readable off the code, at the line it governs.

| File | Shows |
| --- | --- |
| [`01_basics.py`](01_basics.py) | `basics`: `OpenAIAccount`, `bind`, `generate_one`, structured output, `rebind`, and `generate_many` |
| [`02_tool_loop.py`](02_tool_loop.py) | `run_agent`: the tool loop over `generate_one` and `dispatch_many`, an approval gate as `precomputed`, and a tool's `app_data` totalled as each call settles |
| [`03_streaming.py`](03_streaming.py) | `stream_answer`: `stream_one` as a context manager, the `str \| ReasoningDelta \| ToolCallDelta \| ToolCall` iterator, and `final()` for usage and cost |
| [`04_failures_and_deadlines.py`](04_failures_and_deadlines.py) | `run_batch_and_handle_what_failed`: account request limits, deadlines, and provider fallback |
| [`05_prompt_caching.py`](05_prompt_caching.py) | `cache_a_long_prefix`: `cache_breakpoint` marks in the bound prefix and on a tool result, anthropic's `cache_ttl`, and the cache token counters with their costs |
| [`06_required_choice.py`](06_required_choice.py) | `run_required_choice_agent`: a `tool_choice="required"` loop exiting through a `CaptureTool`, forced by `SpecificToolChoice` on the last turn |
| [`07_pricing.py`](07_pricing.py) | `price_at_negotiated_rates`: uncataloged model pricing and both billing scopes |
| [`08_tracing.py`](08_tracing.py) | `traced_tool_turn`: `TracedLLM` and `TracedToolManager` over one generate, dispatch, generate turn |
| [`MIGRATING_FROM_LANGCHAIN.md`](MIGRATING_FROM_LANGCHAIN.md) | the call-for-call API map and what replaces the middleware layer |
| [`full_app/`](full_app/README.md) | the reference architecture for a streaming multi-agent app: `AgentRun`, sub-agents as tools, a per-call `timeout_seconds` inside a whole-app `asyncio.timeout`, and accounting that survives every failure. It runs, offline, on a scripted adapter |

Start with `MIGRATING_FROM_LANGCHAIN.md` for the mental model, then `01_basics.py`.
The centerpiece is `02_tool_loop.py`: langchaint expects you to write the tool loop yourself.
`full_app/` is where those pieces compose into a whole application.
