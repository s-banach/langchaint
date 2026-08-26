# Examples

Each numbered file has one subject and one callable entry point.
Most examples construct backend classes directly to show request code.
`04_failures_and_deadlines.py` coordinates requests across `Anthropic` and `OpenAI`.

| File | Shows |
| --- | --- |
| [`01_basics.py`](01_basics.py) | `OpenAI` construction, bindings, `Response.call`, structured output, rebinding, batches, and normalized result JSON |
| [`02_tool_loop.py`](02_tool_loop.py) | `Response \| ToolCallTurn` and a complete structured tool loop |
| [`03_streaming.py`](03_streaming.py) | Every `StreamItem` variant and `StreamHandle.final()` |
| [`04_failures_and_deadlines.py`](04_failures_and_deadlines.py) | Request pacing, deadlines, batch failures, and provider fallback |
| [`05_prompt_caching.py`](05_prompt_caching.py) | `cache_breakpoint` and `generate_many(warm_cache=True)` |
| [`06_required_choice.py`](06_required_choice.py) | `CaptureTool`, `AllowedToolsChoice`, `tool_choice="required"`, and `SpecificToolChoice` |
| [`07_pricing.py`](07_pricing.py) | Uncataloged model pricing and both usage scopes |
| [`08_tracing.py`](08_tracing.py) | A console exporter and a complete traced tool loop |
| [`09_embeddings.py`](09_embeddings.py) | Retrieval tasks, normalized matrices, and `OpenAI` generation |
| [`10_tool_forms_and_approval.py`](10_tool_forms_and_approval.py) | Tool forms, approval, `app_data`, and `DispatchExceptionGroup.completed_outcomes` |
| [`11_provider_executed_tools.py`](11_provider_executed_tools.py) | OpenAI web search, `RawPart`, and `Usage.provider_executed_tool_cost_in_usd` |
| [`12_messages.py`](12_messages.py) | Multimodal content and message JSON serialization |
| [`MIGRATING_FROM_LANGCHAIN.md`](MIGRATING_FROM_LANGCHAIN.md) | LangChain migration by API concept |
| [`full_app/`](full_app/README.md) | Progress-event application architecture |

Read `MIGRATING_FROM_LANGCHAIN.md`, `01_basics.py`, then `02_tool_loop.py`.
