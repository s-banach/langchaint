# Migrating from LangChain

langchaint provides generation, embeddings, tools, retries, and OTel tracing.
The application owns the code between model calls: prompt construction, conversation history, tool loops, middleware, and fallbacks.
Start by replacing a model call, then move the surrounding orchestration into application functions.

## Construct and call a model

Construct a backend, configure a model, then generate.
Pass system instructions through `system_prompt` and conversation messages through `generate_one`.

```python
from langchaint import UserMessage
from langchaint.openai import OpenAI

openai = OpenAI()
llm = openai.llm("gpt-5.6-terra", system_prompt="Answer briefly.")
generation = await llm.generate_one([UserMessage(content="Hello")])
print(generation.output)
```

Reuse `openai` when creating models that share its client and rate-limit quota.
Its generation and embedding models use one `SharedBackoff` to coordinate requests.
All generation calls are asynchronous, including single calls, batches, and stream consumption.

## API map

| LangChain | langchaint |
| --- | --- |
| `ChatOpenAI(...)` | `openai = OpenAI()` |
| `init_chat_model(...)` | `llm = openai.llm("gpt-5.6-terra")` |
| `model.invoke(messages)` | `await llm.generate_one(messages)` |
| `model.ainvoke(messages)` | `await llm.generate_one(messages)` |
| `model.batch(inputs)` | `await llm.generate_many(inputs)` |
| `model.stream(messages)` | `async with llm.stream_one(messages) as stream:` |
| `model.bind_tools(tools)` | `llm.rebind(tools=tools)` |
| `model.with_structured_output(Model)` | `llm.rebind(response_format=Model)` |
| `create_react_agent(...)` | application tool loop |
| `RunnableRetry` | `max_requests` on `llm` |
| `InMemoryRateLimiter` | `SharedBackoff(max_concurrent_requests=..., max_request_starts_per_second=...)` |
| `.with_fallbacks(...)` | application `try` and `except` |
| callbacks and LangSmith | `OpenAI(observer=OtelObserver(...))` |
| unmatched provider fields | `extra_body={...}` |
| `SystemMessage` | `system_prompt=` on `llm` |
| `HumanMessage` | `UserMessage` |
| `AIMessage` | `AssistantMessage` |
| `ToolMessage` | `ToolMessage` |

## Messages and output

`generate_one` accepts a string as shorthand for one `UserMessage`, or a sequence of `UserMessage`, `AssistantMessage`, and `ToolMessage` values.
Keep the conversation history in the application.
Append `generation.assistant_message` when continuing a conversation:

```python
from langchaint import Message, UserMessage

messages: list[Message] = [UserMessage(content="Name a city in Norway.")]
generation = await llm.generate_one(messages)
print(generation.output)
messages.append(generation.assistant_message)
messages.append(UserMessage(content="What is its population?"))
generation = await llm.generate_one(messages)
```

For text generation, `generation.output` contains the final text part of `generation.assistant_message`.
`generation.commentary` contains preceding text.
`generation.assistant_message` preserves the ordered content, including reasoning and tool calls, for replay.
Reconstructing an assistant message from `output` loses that content.

See [`12_messages.py`](12_messages.py) for multimodal messages and message serialization.

## Structured output and tools

Use `rebind` to return a new configured model while preserving omitted settings.
Pass a pydantic model through `response_format` to validate structured output:

```python
from pydantic import BaseModel


class Answer(BaseModel):
    text: str


structured = llm.rebind(response_format=Answer)
generation = await structured.generate_one("Name a city in Norway.")
print(generation.output.text)
```

Without application tools, `generate_one` returns `PlainGeneration`, whose `output` is text or the validated model.
With application tools, it can also return `ToolCallGeneration`, which has `tool_calls` and no `output`.
Check `generation.kind` before reading the result.

With the application's tool definitions stored in `tools`, one step of an application tool loop is:

```python
bound = llm.rebind(tools=tools)
generation = await bound.generate_one(messages)
match generation.kind:
    case "tool_call":
        messages.append(generation.assistant_message)
        outcomes = await bound.tool_manager.dispatch_many(generation.tool_calls)
        messages.extend(outcome.tool_message for outcome in outcomes)
    case "plain":
        print(generation.output)
```

After dispatch, call the model again with the updated messages.
The application decides when to stop, request approval, or route to another model.
Tools return data; the application uses that data to choose its next action.

See [`02_tool_loop.py`](02_tool_loop.py) for a complete loop with a turn limit and structured output.
See [`06_required_choice.py`](06_required_choice.py) for tool selection and [`10_tool_forms_and_approval.py`](10_tool_forms_and_approval.py) for tool definitions and approval.
Provider-executed tools use `provider_executed_tools` and run at the provider; they are separate from application tool dispatch.
See [`11_provider_executed_tools.py`](11_provider_executed_tools.py) for an example.

## Streaming

Consume a stream inside `async with`, then call `final()` to obtain the assembled generation:

```python
async with llm.stream_one(messages) as stream:
    async for item in stream:
        if isinstance(item, str):
            print(item, end="", flush=True)
    generation = await stream.final()
```

The stream yields text, reasoning deltas, tool-call deltas, and completed tool calls.
Use the assembled generation for validated output, usage, and conversation replay.
langchaint can retry transient failures while opening a stream.
A request failure after the stream opens raises `GenerationError` without retrying.

See [`03_streaming.py`](03_streaming.py) for handling every stream item.

## Embeddings

Create an embedding model from the backend and await `embed` for both documents and queries.
Select the purpose with `task`:

```python
embedding_model = openai.embedding_model("text-embedding-3-small", dimension=512)
documents = await embedding_model.embed(
    ["Oslo is in Norway.", "Tokyo is in Japan."],
    task="retrieval_document",
)
query = await embedding_model.embed(
    ["Which city is in Norway?"],
    task="retrieval_query",
)
print(documents.output.shape, query.output.shape)
```

`output` is a normalized `float32` matrix with one row per input, in input order.
`usage` reports billed input tokens and their estimated cost.
See [`09_embeddings.py`](09_embeddings.py) for retrieval using these matrices.

## Retries, batches, and fallbacks

Set `max_requests` to limit requests per input, including the first request.
Use `max_requests=1` to disable retries.
Use `SharedBackoff` to coordinate concurrency and request-start rate for models sharing a rate-limit quota:

```python
from langchaint import SharedBackoff

primary = OpenAI(
    shared_backoff=SharedBackoff(max_concurrent_requests=16, max_request_starts_per_second=5),
).llm("gpt-5.6-terra", max_requests=5)
```

`generate_one` raises `GenerationError` when generation fails.
With `fallback` set to another configured model, choose when to call it in application code:

```python
from langchaint import GenerationError

try:
    generation = await primary.generate_one(messages, timeout_seconds=30)
except GenerationError:
    generation = await fallback.generate_one(messages, timeout_seconds=30)
```

`GenerationError.usage` includes billing across the input's settled requests.
Use it when accounting for failed calls as well as successful generations.
`timeout_seconds` bounds the entire `generate_one` call, including admission and retry waits.

Choose the batch method according to how the application handles failure.
`generate_many` returns generations in input order; a generation failure cancels and awaits unfinished inputs before raising `GenerationError`.
`generate_many_records` returns an outcome record at each input index, including generation failures, so the application can handle each failure separately:

```python
inputs = ["Define entropy.", "Define enthalpy."]
outcomes = await primary.generate_many_records(inputs)
for index, outcome in enumerate(outcomes):
    if outcome.kind == "plain":
        print(outcome.output)
    else:
        print(f"input {index} failed: {outcome.error_text}")
```

Here `primary` has no application tools, so each record is a plain generation or a generation failure.
For batch deadlines, `max_working_seconds_per_input` excludes admission waits.
See [`04_failures_and_deadlines.py`](04_failures_and_deadlines.py) for batch fallback and deadline handling.

## Middleware and application orchestration

Move middleware behavior to the application function that owns each call or tool dispatch.

| LangChain hook | Application location |
| --- | --- |
| `before_model` | before `await bound.generate_one(messages)` |
| `after_model` | after receiving `PlainGeneration` or `ToolCallGeneration` |
| `modify_model_request` | `bound = bound.rebind(...)` |
| `wrap_tool_call` | around `dispatch` or `dispatch_many` |
| tool error handling | inspect `DispatchOutcome`, or catch `DispatchExceptionGroup` |
| concurrent tool calls | `ToolManager.dispatch_many(tool_calls)` |
| human approval | `dispatch_many(..., precomputed=...)` |
| message trimming | edit `messages` before the next call |
| usage tracking | read `generation.usage` |

Keep prompt templates, routing, and response caching in application code.
For provider prompt caching, see [`05_prompt_caching.py`](05_prompt_caching.py).
For OTel tracing, attach an observer to the backend as shown in [`08_tracing.py`](08_tracing.py).
See [`full_app`](full_app) for an application that combines a tool loop with progress events.
