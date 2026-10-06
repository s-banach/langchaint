# Migrating from LangChain

langchaint provides generation, embeddings, tools, retries, and OTel tracing.
It provides no chains, middleware stack, or agent loop.

## Basic construction

Construct `OpenAI`, select a model, bind configuration, then generate.

```python
from langchaint import UserMessage
from langchaint.openai import OpenAI

openai = OpenAI()
llm = openai.llm("gpt-5.6-terra")
bound = llm.bind()
generation = await bound.generate_one([UserMessage(content="Hello")])
print(generation.output)
```

Every model from `openai` uses `openai.client` and one `SharedBackoff`.

## API map

| LangChain | langchaint |
| --- | --- |
| `ChatOpenAI(...)` | `openai = OpenAI()` |
| `init_chat_model(...)` | `llm = openai.llm("gpt-5.6-terra")` |
| `model.invoke(messages)` | `await bound.generate_one(messages)` |
| `model.ainvoke(messages)` | `await bound.generate_one(messages)` |
| `model.batch(inputs)` | `await bound.generate_many(inputs)` |
| `model.stream(messages)` | `async with bound.stream_one(messages) as stream:` |
| `model.bind_tools(tools)` | `llm.bind(tools=tools)` |
| `model.with_structured_output(Model)` | `llm.bind(response_format=Model)` |
| `create_react_agent(...)` | application tool loop |
| `RunnableRetry` | `max_requests` on `bind` |
| `InMemoryRateLimiter` | `SharedBackoff(max_concurrent_requests=..., max_request_starts_per_second=...)` |
| `.with_fallbacks(...)` | application `try` and `except` |
| `set_llm_cache(...)` | provider prompt caching |
| callbacks and LangSmith | `OpenAI(observer=OtelObserver(...))` |
| `temperature=` | `temperature=` |
| unmatched provider fields | `extra_body={...}` |
| `SystemMessage` | `system_prompt=` on `bind` |
| `HumanMessage` | `UserMessage` |
| `AIMessage` | `AssistantMessage` |
| `ToolMessage` | `ToolMessage` |

langchaint provides only asynchronous generation methods.

## Backend classes

Each backend subpackage exports a class named for its provider.
These constructions use cataloged model identifiers.

```python
from langchaint.anthropic import Anthropic, AnthropicBedrock
from langchaint.cohere import CohereBedrock
from langchaint.deepseek import DeepSeek
from langchaint.gemini import Gemini
from langchaint.openai import OpenAI, OpenAIBedrock

openai = OpenAI()
openai_llm = openai.llm("gpt-5.6-terra")

anthropic = Anthropic()
anthropic_llm = anthropic.llm("claude-sonnet-5")

gemini = Gemini()
gemini_llm = gemini.llm("gemini-3.6-flash")

deepseek = DeepSeek()
deepseek_llm = deepseek.llm("deepseek-v4-flash")

anthropic_bedrock = AnthropicBedrock(aws_region="us-east-1")
anthropic_bedrock_llm = anthropic_bedrock.llm("anthropic.claude-sonnet-5")

openai_bedrock = OpenAIBedrock(aws_region="us-east-1")

cohere_bedrock = CohereBedrock(aws_region="us-east-1")
cohere_embeddings = cohere_bedrock.embedding_model(
    "cohere.embed-v4:0",
    dimension=1024,
)
```

`DeepSeek()` reads `DEEPSEEK_API_KEY` when `client` is absent.
Uncataloged models require explicit pricing.
`OpenAI.llm` also requires `supports_prompt_cache_options` for uncataloged models.
`OpenAIBedrock.llm` always requires both values.
`Anthropic` and `AnthropicBedrock` models require `max_completion_tokens` in `bind`, because the Messages API requires `max_tokens`.

## Bind again

`LLM.bind` freezes request configuration.
`BoundLLM.bind` returns another binding with selected fields replaced.

```python
concise = llm.bind(
    system_prompt="Answer in one sentence.",
    temperature=0.2,
    max_requests=3,
)
creative = concise.bind(
    system_prompt="Write a vivid paragraph.",
    temperature=0.8,
)
```

`BoundLLM.bind` preserves every omitted field.

Use `AllowedToolsChoice` to change `tool_choice` without changing `tools`:

```python
from langchaint import AllowedToolsChoice

bound = llm.bind(tools=[search, final_answer])
search_only = bound.bind(
    tool_choice=AllowedToolsChoice(mode="required", tool_names=(search.name,))
)
```

`tool_names` must be nonempty and name entries supplied through `tools`.
`mode="auto"` permits text or a named tool call, while `mode="required"` requires a named tool call.
`OpenAIResponsesAdapter`, `OpenAIChatCompletionsAdapter`, and `GeminiGenerateContentAdapter` support `AllowedToolsChoice`.
`AnthropicMessagesAdapter` raises `TypeError` during binding because it does not support `AllowedToolsChoice`.

See [`06_required_choice.py`](06_required_choice.py) for `AllowedToolsChoice`, `tool_choice="required"`, and `SpecificToolChoice` with unchanged `tools`.

## Generation types

| Binding | `generate_one` return type |
| --- | --- |
| text, without tools | `PlainGeneration[str]` |
| text, with `ToolManager` | `PlainGeneration[str] \| ToolCallGeneration[str]` |
| structured, without tools | `PlainGeneration[Model]` |
| structured, with `ToolManager` | `PlainGeneration[Model] \| ToolCallGeneration[Model \| None]` |

`Generation[str]` and `Generation[Model, Model | None]` name the unions `generate_one` returns.
`GenerationOutcome` adds `GenerationError` for batch outcomes.
A binding with `ToolManager` returns `ToolCallGeneration` whenever the kept assistant message has tool calls.

```python
from pydantic import BaseModel


class Answer(BaseModel):
    text: str


generation = await llm.bind(
    response_format=Answer,
    tools=tools,
).generate_one("Answer the question")

match generation.kind:
    case "tool_call":
        print(generation.tool_calls)
    case "plain":
        print(generation.output.text)
```

A text binding's `output` is the assistant message's text, which is `""` for an assistant message without text.
A structured `ToolCallGeneration.output` is `None` when the assistant message has no valid `Model`.
Append `assistant_message`, never `output`, when continuing a conversation.

See [`02_tool_loop.py`](02_tool_loop.py) for the basic tool loop.
See [`10_tool_forms_and_approval.py`](10_tool_forms_and_approval.py) for advanced tool forms.

## `provider_executed_tools`

`provider_executed_tools` accepts provider-shaped tool definitions.
The provider executes these tools.
Do not pass their calls to `ToolManager`.

`provider_executed_tools` response items appear as `RawPart` values.
`Usage.provider_executed_tool_cost_in_usd` contributes to `Usage.cost_in_usd`.
An unusable required rate raises during `bind`.

See [`11_provider_executed_tools.py`](11_provider_executed_tools.py) for a complete request.

## Embeddings

`EmbeddingModel.embed` returns normalized `float32` rows.
Row order matches input order.

```python
from langchaint.openai import OpenAI

openai = OpenAI()
embedding_model = openai.embedding_model(
    "text-embedding-3-small",
    dimension=512,
)
documents = await embedding_model.embed(
    ["Oslo is in Norway.", "Tokyo is in Japan."],
    task="retrieval_document",
)
query = await embedding_model.embed(
    ["Which city is in Norway?"],
    task="retrieval_query",
)
print(documents.shape, query.shape)
```

`OpenAI.llm()` returns `LLM`.
`OpenAI.embedding_model()` returns `EmbeddingModel`.
Both use `openai.client` and one `SharedBackoff`.
See [`09_embeddings.py`](09_embeddings.py) for both embedding tasks.

## Prompt caching

`LLM.bind()` uses `Adapter.automatic_cache_breakpoints_default` when the argument is `None`.
Pass `automatic_cache_breakpoints` to override `Adapter.automatic_cache_breakpoints_default`.

Explicit `cache_breakpoint=True` values remain active under either `automatic_cache_breakpoints` value.

```python
from langchaint import TextPart

bound = llm.bind(
    system_prompt=[
        TextPart(
            text="Stable instructions and reference material.",
            cache_breakpoint=True,
        ),
        TextPart(text="Request-specific context."),
    ],
    automatic_cache_breakpoints=False,
)
```

Provider minimum-token requirements still apply.
Inspect `Usage.input_tokens_cache_read` and `Usage.input_tokens_cache_write`.

Use `warm_cache=True` for batches sharing a reusable prefix.
The first item completes before remaining items start.

```python
outcomes = await bound.generate_many(
    ["First question", "Second question", "Third question"],
    warm_cache=True,
)
```

The first outcome may be a `GenerationError`.
Remaining items still start afterward.
See [`05_prompt_caching.py`](05_prompt_caching.py) for measured cache counters.

## Unmatched provider fields

`LLM.bind` accepts `max_completion_tokens`, `reasoning_level`, and `temperature` directly.
Use `extra_body` for other provider wire fields.

```python
bound = llm.bind(
    extra_body={"top_p": 0.9},
)
```

OpenAI SDK 2.53.0 accepts `top_p` on Responses requests.
An adapter rejects `extra_body` keys that it already populates.

## Retries, batches, and errors

`max_requests` counts requests, including the first request.
Set `max_requests=1` to disable retries.

```python
from langchaint import SharedBackoff

openai = OpenAI(
    shared_backoff=SharedBackoff(max_concurrent_requests=16, max_request_starts_per_second=5),
)
bound = openai.llm("gpt-5.6-terra").bind(
    max_requests=5,
)
```

`generate_one` raises `GenerationError` for terminal generation failures.
`generate_many` returns each `GenerationError` at its input index.

```python
from langchaint import GenerationError

try:
    generation = await primary.generate_one(messages, timeout_seconds=30)
except GenerationError:
    generation = await fallback.generate_one(messages, timeout_seconds=30)

outcomes = await primary.generate_many(inputs)
for index, outcome in enumerate(outcomes):
    if isinstance(outcome, GenerationError):
        outcomes[index] = await fallback.generate_one(inputs[index])
```

`GenerationError.usage` includes paid usage across settled requests.
`max_working_seconds_per_input` excludes admission waits.
Use `timeout_seconds` for a `generate_one` wall-clock deadline.

See [`04_failures_and_deadlines.py`](04_failures_and_deadlines.py) for failure handling.

## Middleware becomes application code

| LangChain hook | Application location |
| --- | --- |
| `before_model` | before `await bound.generate_one(messages)` |
| `after_model` | after receiving `PlainGeneration` or `ToolCallGeneration` |
| `modify_model_request` | `bound = bound.bind(...)` |
| `wrap_tool_call` | around `dispatch` or `dispatch_many` |
| tool error handling | inspect `DispatchOutcome`, or catch `DispatchExceptionGroup` |
| concurrent tool calls | `ToolManager.dispatch_many(tool_calls)` |
| human approval | `dispatch_many(..., precomputed=...)` |
| message trimming | edit `messages` before the next call |
| structured output | `bind(response_format=Model, ...)` |
| usage tracking | read `generation.usage` |

The application owns routing between `generate_one` calls.
A tool returns data instead of a control-flow instruction.

See [`03_streaming.py`](03_streaming.py) for provider response streaming.
See [`08_tracing.py`](08_tracing.py) for OTel tracing.
See [`full_app`](full_app) for an application event stream.
