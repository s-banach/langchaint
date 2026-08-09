# Provider-executed tools

Provider-executed tools run during generation on the provider's service.
`ToolManager` tools run inside the application.
This execution ownership produces separate binding arguments.

The user query remains in `GenerationInput`.
Each `provider_executed_tools` entry carries provider request configuration.
No provider-executed result enters `ToolManager.dispatch()`.

Verified against anthropic 0.121.0, openai 2.53.0, and google-genai 2.17.0.

| Provider API | Request entry | Result evidence |
| --- | --- | --- |
| Anthropic Messages | `{"type": "web_search_20260318", "name": "web_search"}` | `server_tool_use` and result blocks |
| OpenAI Responses | `{"type": "web_search"}` | `web_search_call` output items |
| Gemini GenerateContent | `{"google_search": {}}` | paired `Part.tool_call` and `Part.tool_response` |

OpenAI Chat Completions uses `web_search_options` directly.
langchaint rejects that field because responses omit exact invocation counts.

## Direct SDK forms

Anthropic accepts provider-executed entries beside application tool schemas.

```python
await anthropic_client.messages.create(
    model=model,
    max_tokens=1024,
    messages=[{"role": "user", "content": question}],
    tools=[
        {
            "type": "web_search_20260318",
            "name": "web_search",
            "max_uses": 5,
        }
    ],
)
```

OpenAI Responses uses the same request arrangement.

```python
await openai_client.responses.create(
    model=model,
    input=question,
    tools=[
        {
            "type": "web_search",
            "search_context_size": "medium",
        }
    ],
)
```

Gemini places `types.Tool` values inside `GenerateContentConfig.tools`.

```python
await gemini_client.aio.models.generate_content(
    model=model,
    contents=question,
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        tool_config=types.ToolConfig(
            include_server_side_tool_invocations=True,
        ),
    ),
)
```

Gemini mappings are normalized through `types.Tool`.
The invocation flag returns billing evidence and replayable provider blocks.

Chat Completions accepts its separate request field directly.

```python
await openai_client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": question}],
    web_search_options={"search_context_size": "medium"},
)
```

## Binding API

`provider_executed_tools` accepts provider-shaped mappings directly.

```python
from anthropic.types import WebSearchTool20260318Param

from langchaint.anthropic import AnthropicAccount

web_search: WebSearchTool20260318Param = {
    "type": "web_search_20260318",
    "name": "web_search",
    "max_uses": 5,
}

async with AnthropicAccount() as anthropic:
    bound = anthropic.model("claude-opus-4-8").bind(
        system_prompt="Answer with citations.",
        provider_executed_tools=(web_search,),
        automatic_prompt_caching=False,
    )
    response = await bound.generate_one("What changed in Python 3.14?")
```

A strict type checker checks provider fields through the SDK annotation.
langchaint imports no SDK type into its neutral core.

Omitting `provider_executed_tools` during `rebind` preserves its value.
Passing `provider_executed_tools=()` removes every entry.
Changing it preserves `max_attempts` and `BoundLLM` output types.

## Supported configurations

Anthropic supports the reviewed web-search, web-fetch, and tool-search `type` values.
Qualifying web-search and web-fetch `type` values can accompany reviewed code execution.
Standalone code execution lacks exact response billing evidence.
Bedrock clients reject every nonempty `provider_executed_tools` value.

OpenAI Responses supports reviewed web-search aliases and `file_search`.
Other `ToolParam` types require unavailable billing evidence or application execution.
Non-OpenAI providers reject every nonempty `provider_executed_tools` value.

Gemini supports `google_search`, `google_maps`, `code_execution`, `url_context`, and `file_search`.
Gemini image search and other `types.Tool` fields are rejected.
Provider-executed tools require a Gemini 3 model identifier.
Gemini 2.5 has no catalog entries.

## Adapter behavior

Generated application schemas precede `provider_executed_tools` entries.
Anthropic and OpenAI preserve accepted mappings by reference.
Gemini normalizes each mapping and inspects every populated field.

Mixed Gemini requests use `VALIDATED` function calling.
Provider-only Gemini requests omit function-calling configuration.
Every Gemini provider request enables server invocation blocks.

Anthropic applies automatic cache control after combining both collections.
The adapter copies any mapping receiving automatic `cache_control`.
Caller mappings remain unchanged.

## Response behavior

Provider-executed response blocks become `RawPart` values.
`RawPart.raw` carries provider data for same-provider replay.
Metadata-only evidence remains reachable through `Response.raw`.
Nothing dispatches through `ToolManager`.

## Billing

`Usage.provider_executed_tool_cost_in_usd` carries one summed cost category.
`Usage.cost_in_usd` includes that category.
Provider-specific evidence remains on raw provider responses.

Provider-tool prices default to `None`.
Charged bindings require configured provider-tool prices.
Configured charged rates must be finite and nonnegative.
Catalog rates estimate post-quota list prices.

Anthropic prices `web_search_requests` at `web_search_usd_per_invocation`.
Reviewed web fetch, tool search, and exempt code execution add zero.

OpenAI prices only `web_search_call` items with `action.type == "search"`.
Each `file_search_call` costs `file_search_usd_per_invocation` once.
Vector-store storage costs remain application-owned.

Gemini Search bills unique nonempty queries.
Gemini Maps bills every returned nonempty query entry.
Paired server blocks distinguish Search from Maps.

Supported complete responses produce finite provider-executed costs.
NaN remains a fallback for unexpected or incomplete billing evidence.
Uncataloged served tiers also produce NaN for charged outcomes.
