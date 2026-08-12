# Migrating from LangChain

langchaint provides generation, embeddings, tools, retries, and OTel tracing.
It provides no chains, middleware stack, or agent loop.

## Basic construction

Construct `OpenAI`, select a model, bind configuration, then generate.

```python
from langchaint import UserMessage
from langchaint.openai import OpenAI

openai = OpenAI()
llm = openai.model("gpt-5.6-terra")
bound = llm.bind(automatic_prompt_caching=False)
response = await bound.generate_one([UserMessage(content="Hello")])
print(response.output)
```

Every model from `openai` uses `openai.client` and one `SharedBackoff`.

## API map

| LangChain | langchaint |
| --- | --- |
| `ChatOpenAI(...)` | `openai = OpenAI()` |
| `init_chat_model(...)` | `llm = openai.model("gpt-5.6-terra")` |
| `model.invoke(messages)` | `await bound.generate_one(messages)` |
| `model.ainvoke(messages)` | `await bound.generate_one(messages)` |
| `model.batch(inputs)` | `await bound.generate_many(inputs)` |
| `model.stream(messages)` | `async with bound.stream_one(messages) as stream:` |
| `model.bind_tools(tools)` | `llm.bind(tools=tools, automatic_prompt_caching=False)` |
| `model.with_structured_output(Model)` | `llm.bind(response_format=Model, automatic_prompt_caching=False)` |
| `create_react_agent(...)` | application tool loop |
| `RunnableRetry` | `max_attempts` on `bind` |
| `InMemoryRateLimiter` | `max_concurrent_requests` and `max_request_starts_per_second` |
| `.with_fallbacks(...)` | application `try` and `except` |
| `set_llm_cache(...)` | provider prompt caching |
| callbacks and LangSmith | `langchaint.tracing` with OTel |
| `temperature=` | `InferenceParams(temperature=...)` |
| unmatched provider fields | `extra_body={...}` |
| `SystemMessage` | `system_prompt=` on `bind` |
| `HumanMessage` | `UserMessage` |
| `AIMessage` | `AssistantMessage` |
| `ToolMessage` | `ToolMessage` |

Generation methods are asynchronous.
There is no synchronous generation API.

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
openai_llm = openai.model("gpt-5.6-terra")

anthropic = Anthropic()
anthropic_llm = anthropic.model("claude-sonnet-5")

gemini = Gemini()
gemini_llm = gemini.model("gemini-3.6-flash")

deepseek = DeepSeek()
deepseek_llm = deepseek.model("deepseek-v4-flash")

anthropic_bedrock = AnthropicBedrock(aws_region="us-east-1")
anthropic_bedrock_llm = anthropic_bedrock.model("anthropic.claude-sonnet-5")

openai_bedrock = OpenAIBedrock(aws_region="us-east-1")

cohere_bedrock = CohereBedrock(aws_region="us-east-1")
cohere_embeddings = cohere_bedrock.embedding_model(
    "cohere.embed-v4:0",
    dimension=1024,
)
```

`DeepSeek()` reads `DEEPSEEK_API_KEY` when `client` is absent.
Uncataloged models require explicit pricing.
`OpenAI.model` also requires `supports_prompt_cache_options` for uncataloged models.
`OpenAIBedrock.model` always requires both values.

## Bind and rebind

`LLM.bind` freezes request configuration.
`BoundLLM.rebind` returns another binding with selected fields replaced.

```python
from langchaint import InferenceParams

concise = llm.bind(
    system_prompt="Answer in one sentence.",
    inference_params=InferenceParams(temperature=0.2),
    max_attempts=3,
    automatic_prompt_caching=False,
)
creative = concise.rebind(
    system_prompt="Write a vivid paragraph.",
    inference_params=InferenceParams(temperature=0.8),
)
```

`rebind` replaces the complete `InferenceParams` value.

## Result types

| Binding | `generate_one` success type |
| --- | --- |
| text, without tools | `Response[str]` |
| text, with `ToolManager` | `Response[str]` |
| structured, without tools | `Response[Model]` |
| structured, with `ToolManager` | `Response[Model] \| ToolCallTurn[Model]` |

`GenerateResult[Model]` names the success union.
`CallResult[Model]` adds `GenerationError` for batch results.
Text bindings expose tool calls through `Response.tool_calls`.

```python
from pydantic import BaseModel

from langchaint import Response, ToolCallTurn


class Answer(BaseModel):
    text: str


result = await llm.bind(
    response_format=Answer,
    tools=tools,
    automatic_prompt_caching=False,
).generate_one("Answer the question")

match result:
    case ToolCallTurn():
        print(result.tool_calls)
    case Response():
        print(result.output.text)
```

`Response.output` is never `None`.
`ToolCallTurn.output` may be `None`.
Append `assistant_message` when continuing a conversation.

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

`OpenAI.model()` returns `LLM`.
`OpenAI.embedding_model()` returns `EmbeddingModel`.
Both use `openai.client` and one `SharedBackoff`.
See [`09_embeddings.py`](09_embeddings.py) for both embedding tasks.

## Prompt caching

`automatic_prompt_caching` is required on every `bind`.
The value has billing consequences.
langchaint never selects it for the caller.

Explicit `cache_breakpoint=True` values remain active under either setting.

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
    automatic_prompt_caching=False,
)
```

Provider minimum-token requirements still apply.
Inspect `Usage.input_tokens_cache_read` and `Usage.input_tokens_cache_write`.

Use `warm_cache=True` for batches sharing a reusable prefix.
The first item completes before remaining items start.

```python
results = await bound.generate_many(
    ["First question", "Second question", "Third question"],
    warm_cache=True,
)
```

The first result may be a `GenerationError`.
Remaining items still start afterward.
See [`05_prompt_caching.py`](05_prompt_caching.py) for measured cache counters.

## Unmatched provider fields

`InferenceParams` contains `max_completion_tokens`, `reasoning_effort`, and `temperature`.
Use `extra_body` for other provider wire fields.

```python
bound = llm.bind(
    extra_body={"top_p": 0.9},
    automatic_prompt_caching=False,
)
```

OpenAI SDK 2.53.0 accepts `top_p` on Responses requests.
An adapter rejects `extra_body` keys that it already populates.

## Retries, batches, and errors

`max_attempts` counts requests, including the first request.
Set `max_attempts=1` to disable retries.

```python
openai = OpenAI(
    max_concurrent_requests=16,
    max_request_starts_per_second=5,
)
bound = openai.model("gpt-5.6-terra").bind(
    max_attempts=5,
    automatic_prompt_caching=False,
)
```

`generate_one` raises `GenerationError` for terminal generation failures.
`generate_many` returns each `GenerationError` at its input index.

```python
from langchaint import GenerationError

try:
    response = await primary.generate_one(messages, timeout_seconds=30)
except GenerationError:
    response = await fallback.generate_one(messages, timeout_seconds=30)

results = await primary.generate_many(inputs)
for index, result in enumerate(results):
    if isinstance(result, GenerationError):
        results[index] = await fallback.generate_one(inputs[index])
```

`GenerationError.usage` includes paid usage across settled attempts.
`max_working_seconds_per_item` excludes admission waits.
Use `timeout_seconds` for a `generate_one` wall-clock deadline.

See [`04_failures_and_deadlines.py`](04_failures_and_deadlines.py) for failure handling.

## Middleware becomes application code

| LangChain hook | Application location |
| --- | --- |
| `before_model` | before `await bound.generate_one(messages)` |
| `after_model` | after receiving `Response` or `ToolCallTurn` |
| `modify_model_request` | `bound = bound.rebind(...)` |
| `wrap_tool_call` | around `dispatch` or `dispatch_many` |
| tool error handling | inspect `DispatchOutcome`, or catch `DispatchExceptionGroup` |
| concurrent tool calls | `ToolManager.dispatch_many(tool_calls)` |
| human approval | `dispatch_many(..., precomputed=...)` |
| message trimming | edit `messages` before the next call |
| structured output | `bind(response_format=Model, ...)` |
| usage tracking | read `result.usage` |

The application owns routing between turns.
A tool returns data instead of a control-flow instruction.

See [`03_streaming.py`](03_streaming.py) for provider response streaming.
See [`08_tracing.py`](08_tracing.py) for OTel tracing.
See [`full_app`](full_app) for an application event stream.
