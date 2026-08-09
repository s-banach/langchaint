# langchaint

Provider-neutral async LLM and embedding clients over official SDKs.
Alpha: the API is unstable and may change without notice.

## The point

langchaint sits between application workflows and provider SDKs.
Generation uses one message tree, error taxonomy, priced `Usage`, and `SharedBackoff`.
Embeddings return normalized NumPy matrices directly.
Applications keep agent loops, caching choices, and vector persistence.

## Install

langchaint requires Python 3.13 or newer.
Its core dependencies are `pydantic`, `jsonschema`, and NumPy.
Applications pin every provider SDK directly.
langchaint declares no dependency extras.

Install OpenAI generation support:

    pip install langchaint openai

Install OpenAI embedding support:

    pip install langchaint openai tiktoken

Install Cohere embedding support through Amazon Bedrock:

    pip install langchaint boto3

Top-level `import langchaint` requires no provider SDK.
Backend imports report missing SDK dependencies through `ModuleNotFoundError`.

## Generation example

```python
import asyncio

from pydantic import BaseModel

from langchaint.openai import OpenAIAccount


class Sentiment(BaseModel):
    label: str
    confidence: float


async def main() -> None:
    async with OpenAIAccount() as account:
        classifier = account.model("gpt-5.6-terra").bind(
            system_prompt="Classify the sentiment of the user's message.",
            response_format=Sentiment,
            automatic_prompt_caching=False,
        )
        response = await classifier.generate_one("This is the best day in months.")
        print(response.output.label, response.usage.cost_in_usd)


asyncio.run(main())
```

`bind(response_format=Sentiment)` returns `BoundLLM[Sentiment]`.
Its `response.output` is a validated `Sentiment` instance.
Without `response_format`, `response.output` is assistant text.
A bare generation `str` becomes one `UserMessage`.

## Embedding example

```python
import asyncio

from langchaint.openai import OpenAIAccount


async def main() -> None:
    async with OpenAIAccount() as account:
        embedding_model = account.embedding_model(
            "text-embedding-3-small",
            dimension=1024,
        )
        documents = await embedding_model.embed(
            ["The Moon orbits Earth.", "Mars has two small moons."],
            task="retrieval_document",
        )
        query = await embedding_model.embed(
            ["Which object circles Earth?"],
            task="retrieval_query",
        )
        print(documents.shape, query.shape)


asyncio.run(main())
```

OpenAI accepts `task` for interface consistency.
It sends no corresponding OpenAI request field.

Use this construction for Cohere through Amazon Bedrock:

```python
from langchaint.cohere import CohereBedrockAccount

async with CohereBedrockAccount(aws_region="us-east-1") as account:
    embedding_model = account.embedding_model("cohere.embed-v4:0", dimension=1024)
```

Each result has shape `(len(inputs), dimension)` and dtype `np.float32`.
Every row has L2 norm one.
Each result owns writable, C-contiguous storage.
langchaint performs no vector persistence.

## What it has

**Generation only through binding.**
`LLM.bind()` returns `BoundLLM[OutputT]` with frozen inference configuration.
`BoundLLM` provides `generate_one`, `generate_many`, and `stream_one`.
Changing generation parameters uses `rebind()`.

**Embeddings without binding.**
`EmbeddingModel.embed()` takes texts and one required `task`.
It returns `Float2D` directly.
Provider adapters maximize ordered batches under documented request limits.

**One `Account` per SDK client configuration.**
Accounts share SDK clients and one `SharedBackoff` across request clients.
Each account closes resources it created.

**One accounting contract for generation.**
Generation success and `GenerationError` both carry paid usage across attempts.

**Priced generation usage.**
`Usage` partitions input tokens by cache outcome.
It carries one cost per priced category.

**One `SharedBackoff` owning pacing.**
Its `admitted()` block gates every request start.
Rate limits pause the complete account request domain.

**User-stated prompt caching.**
`automatic_prompt_caching` is required because caching changes billing.
`cache_breakpoint=True` places a prompt-cache boundary on that content part.

**Streaming through a handle.**
`stream_one` returns a `StreamHandle` async context manager.
Its `final()` method returns the assembled result.

**Tools under one protocol.**
`PydanticTool`, `JSONSchemaTool`, and `CaptureTool` implement `Tool`.
Applications may implement additional `Tool` forms.

**Reasoning preserved across turns.**
langchaint re-emits provider reasoning elements verbatim on later requests.

**OTel tracing as a wrapper.**
`langchaint.tracing` wraps generation clients and `ToolManager`.
`capture_message_content` is required because prompt recording affects privacy.

## What it does not have

- No agent class or agent loop.
- No vector storage or retrieval index.
- No client-side guessing at undocumented provider rules.
- No document or PDF content part.

Convert documents before generation requests.
Use `ImagePart` for rasterized pages or `TextPart` for extracted text.

## Layout

    src/langchaint/           the provider-neutral core
    src/langchaint/anthropic/ the Anthropic backend
    src/langchaint/cohere/    the Cohere Bedrock embedding backend
    src/langchaint/deepseek/  the DeepSeek backend using the OpenAI SDK
    src/langchaint/gemini/    the Gemini backend
    src/langchaint/openai/    the OpenAI backend
    src/langchaint/tracing/   the OTel tracing subpackage
    examples/                 focused examples and migration guidance

## Verification

Run `scripts/CI.sh`.
The tests are offline and require no API keys.
