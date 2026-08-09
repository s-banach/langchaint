# Provider tools: what enabling one would look like

A sketch, not a decision. It exists to make TODO.md's first item concrete.

## What a provider tool is

A provider tool runs on the provider's own servers, inside the turn.
You enable it with an entry in the request's `tools` array.
The entry names a type and carries no function schema, and the application never dispatches it.
The provider runs the tool and answers in the assistant turn, in a content block where the provider models one and beside the turn where it does not.

Verified against the installed SDKs (anthropic 0.120.2, openai 2.51.0, google-genai 2.16.0):

| provider | request entry | what comes back in the turn |
| --- | --- | --- |
| anthropic | `{"type": "web_search_20250305", "name": "web_search"}` | `server_tool_use` block, then `web_search_tool_result` block |
| openai Responses | `{"type": "web_search"}` | output item of type `web_search_call` |
| gemini | `Tool(code_execution=ToolCodeExecution())` | `executable_code` and `code_execution_result` parts |

`ServerToolUseBlock.name` enumerates anthropic's set: `web_search`, `web_fetch`, `code_execution`, `bash_code_execution`, `text_editor_code_execution`, `tool_search_tool_regex`, `tool_search_tool_bm25`.
`google.genai.types.Tool` enumerates gemini's: `google_search`, `code_execution`, `url_context`, `file_search`, `computer_use`, and others.

Not every gemini provider tool answers in a content part.
`Tool(google_search=...)` grounds the response and puts its sources on `Candidate.grounding_metadata`, which no assistant turn holds.
That result remains in `Response.raw`.

## The binding side

```python
from langchaint import ProviderTool
from langchaint.anthropic import anthropic_model

llm = anthropic_model(model="claude-opus-4-6", automatic_prompt_caching=False)

bound = llm.bind(
    system_prompt="Answer with citations.",
    provider_tools=(
        ProviderTool(entry={"type": "web_search_20250305", "name": "web_search", "max_uses": 5}),
    ),
)
```

`ProviderTool.entry` is an opaque wire mapping, like `ReasoningPart.raw`.
langchaint models nothing inside it, so a new provider tool needs no langchaint release.
The adapter appends the entry to the same wire `tools` array it builds from `Binding.tool_schemas`.
A binding parameter rather than an `extra_body` key, because every adapter refuses `tools` in `extra_body` as a wire key it populates.

`provider_tools` is separate from `tool_manager` because the two have nothing in common.
A `Tool` carries a name, a schema, and a Python function langchaint calls.
A `ProviderTool` carries a wire entry and nothing langchaint calls.

## The response side

Adapters store provider tool blocks as `RawPart` values.
`RawPart.raw` preserves each SDK block for replay.

Anthropic's existing fallback branch handles every remaining content block:

```python
# anthropic/messages_adapter.py
else:
    turn.append(RawPart(raw=block.model_dump(mode="python", exclude_none=True)))
```

Anthropic's stream path needs no separate change.
`get_final_message()` returns the SDK's accumulated `Message`, and `_assistant_message_from` runs on it.
The per-item stream yields nothing for these blocks; the assembled turn carries them.

## What the round trip then looks like

```python
messages: list[Message] = [UserMessage(content="What shipped in Python 3.14?")]
response = await bound.generate_one(messages)

messages.append(response.assistant_message)  # carries the search blocks verbatim
messages.append(UserMessage(content="Which of those are C API changes?"))
second = await bound.generate_one(messages)
```

The first `assistant_message` holds each search block as a `RawPart`.

Cross-provider replay returns `InvalidRequest` or lets the provider reject the `RawPart`.

## What the sketch does not settle

**Cost.** `usage.server_tool_use.web_search_requests` is a priced counter anthropic reports per response.
langchaint's `Usage` has no field for it, so `cost_in_usd` under-reports a turn that searched.
The counter is reachable on the raw SDK usage. Whether a per-tool fee is one of the priced categories that partition a request's cost is the design question to answer.

**openai and gemini blocks.** openai's `web_search_call` and gemini's `executable_code` become `RawPart` values.
Gemini's case overlaps the existing `thought_signature` branch, so the part-level ordering there needs care.

## Conformance

One invariant covers anthropic, openai Responses, and gemini:
a constructed response holding a provider tool block round-trips through `_assistant_message_from` and back to the wire unchanged.
Use `test_raw_part_round_trips_verbatim_in_position` from `conformance.py`.

`ChatCompletionAssistantMessageParam` has no `annotations` field, so openai's citations are output-only and there is nothing to round-trip them into.
