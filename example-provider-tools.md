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
That one needs a different home than the turn element below.

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

`ProviderTool.entry` is an opaque wire mapping, like `ReasoningTrace.raw`.
langchaint models nothing inside it, so a new provider tool needs no langchaint release.
The adapter appends the entry to the same wire `tools` array it builds from `Binding.tool_schemas`.
A binding parameter rather than an `extra_body` key, because every adapter refuses `tools` in `extra_body` as a wire key it populates.

`provider_tools` is separate from `tool_manager` because the two have nothing in common.
A `Tool` carries a name, a schema, and a Python function langchaint calls.
A `ProviderTool` carries a wire entry and nothing langchaint calls.

## The response side

A new variant of the `TurnElement` union, opaque for the same reason `ReasoningTrace` is:

```python
class ProviderToolTrace(CheckedCopyModel):
    """One provider-executed tool call or its result, round-tripped verbatim.

    raw is the producing SDK block's model_dump(exclude_none=True).
    The consuming adapter re-feeds it unchanged, because a provider that ran the tool
    requires its own blocks back to continue the turn.
    Read what the tool did from Response.raw, which holds the SDK response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: Mapping[str, object]
    kind: Literal["provider_tool_trace"] = "provider_tool_trace"
```

`_assistant_message_from` gains one branch in each adapter:

```python
# anthropic/messages_adapter.py
elif block.type in ("server_tool_use", "web_search_tool_result", "code_execution_tool_result"):
    turn.append(ProviderToolTrace(raw=block.model_dump(mode="python", exclude_none=True)))
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

Without the new variant, the first `assistant_message` holds only the text, and anthropic 400s the second request.

Cross-provider replay fails exactly as it does for `ReasoningTrace`:
a trace one provider produced is a malformed request at another.
The rule is already written in `ReasoningTrace`'s docstring and would extend to this class.

## What the sketch does not settle

**Cost.** `usage.server_tool_use.web_search_requests` is a priced counter anthropic reports per response.
langchaint's `Usage` has no field for it, so `cost_in_usd` under-reports a turn that searched.
The counter is reachable on the raw SDK usage. Whether a per-tool fee is one of the priced categories that partition a request's cost is the design question to answer.

**Reading the result.** `Response.raw` already holds the SDK response, so a caller who wants the search results reads them there.
If that is the answer, say it in the `ProviderToolTrace` docstring and add nothing.

**openai and gemini blocks.** openai's `web_search_call` item and gemini's `executable_code` part need the same variant.
Gemini's case overlaps the existing `thought_signature` branch, so the part-level ordering there needs care.

## Conformance

One invariant covers anthropic, openai Responses, and gemini:
a constructed response holding a provider tool block round-trips through `_assistant_message_from` and back to the wire unchanged.
Same shape as the reasoning-trace round trip already in `conformance.py`.

`ChatCompletionAssistantMessageParam` has no `annotations` field, so openai's citations are output-only and there is nothing to round-trip them into.
