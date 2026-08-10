# langchaint

Provider-neutral generation and embeddings over official SDKs.
Alpha: the API is unstable and may change without notice.

## Purpose

langchaint provides `BoundLLM` and `EmbeddingModel` as provider-neutral async interfaces.  
langchaint provides no agent class or agent loop.

## Install and authenticate

langchaint requires Python 3.13 or newer.
Applications install and pin each provider SDK directly.
langchaint declares no dependency extras.
Top-level `import langchaint` requires no provider SDK.
Backend imports report missing dependencies through `ModuleNotFoundError`.

| Provider | Class | Creates | Depends | Default credentials |
| --- | --- | --- | --- | --- |
| Anthropic | `Anthropic` | `LLM` | `anthropic` | `ANTHROPIC_API_KEY` |
| Amazon Bedrock | `AnthropicBedrock` | `LLM` | `anthropic[bedrock]` | AWS credential provider chain |
| Amazon Bedrock | `CohereBedrock` | `EmbeddingModel` | `boto3` | AWS credential provider chain |
| DeepSeek | `DeepSeek` | `LLM` | `openai` | `DEEPSEEK_API_KEY` |
| Gemini | `Gemini` | `LLM` | `google-genai` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| OpenAI | `OpenAI` | `LLM`, `EmbeddingModel` | `openai` (and `tiktoken` for embeddings) | `OPENAI_API_KEY` |
| Amazon Bedrock | `OpenAIBedrock` | `LLM` | `openai[bedrock]` | AWS credential provider chain |

Every listed class accepts `client=` for SDK client configuration.
The Amazon Bedrock classes use the AWS credential provider chain.
This includes environment credentials, profiles, SSO, containers, and instance roles.
Pass `aws_region=` to select a Bedrock region explicitly.

## Generation quickstart

```python
import asyncio

from langchaint.openai import OpenAI


async def main() -> None:
    openai = OpenAI()
    bound_llm = openai.model("gpt-5.6-terra").bind(
        system_prompt="Answer clearly and concisely.",
        automatic_prompt_caching=False,
    )
    response = await bound_llm.generate_one("Why is the sky blue?")
    print(response.output, response.usage.cost_in_usd)


asyncio.run(main())
```

`response.output` is assistant text.
Pass a Pydantic model to `LLM.bind(response_format=...)` for validated structured output.

## Embedding quickstart

```python
import asyncio

from langchaint.openai import OpenAI


async def main() -> None:
    openai = OpenAI()
    embedding_model = openai.embedding_model(
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

`EmbeddingModel.embed()` requires `task` for every adapter.
The OpenAI adapter sends no corresponding request field.
`EmbeddingModel.embed()` returns `Float2D`, a normalized two-dimensional NumPy `float32` array.
Each input produces one row.

Use `CohereBedrock` for Cohere embeddings through Amazon Bedrock.

## Share one rate-limit quota

Create one `OpenAI` per rate-limit quota.

```python
openai = OpenAI(
    max_concurrent_requests=8,
    max_request_starts_per_second=50.0,
)
terra = openai.model("gpt-5.6-terra")
sol = openai.model("gpt-5.6-sol")
```

Both `terra` and `sol` use `openai.client` and one `SharedBackoff`.
`max_concurrent_requests` applies across both `LLM` values.
`max_request_starts_per_second` applies across both `LLM` values.
An `EmbeddingModel` from `openai.embedding_model()` uses the same client and `SharedBackoff`.

Pass an SDK client to close it directly.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()
openai = OpenAI(client=client)
terra = openai.model("gpt-5.6-terra")

# Use terra.

await client.close()
```

## Binding and results

Call `LLM.bind()` before generating.
`BoundLLM` provides `generate_one`, `generate_many`, and `stream_one`.
`BoundLLM.rebind()` replaces selected binding fields.
Pass `tools=[tool]` to `LLM.bind()`, then dispatch through `BoundLLM.tool_manager`.
`LLM.bind(max_attempts=...)` limits requests for one `GenerationInput`, including the first.
An embedding batch contains inputs sent together during each attempt.
`embedding_model(max_attempts=...)` limits requests for one embedding batch, including the first.
`automatic_prompt_caching` is required because it changes billing.
`cache_breakpoint=True` ends the reusable prefix at that `ContentPart`.
`GenerateResult` and `GenerationError` include paid `Usage` across attempts.

## More examples

[`examples/README.md`](examples/README.md) indexes focused examples and migration guidance.
The examples cover structured output, batches, streaming, tools, tracing, pricing, and failures.
They also cover prompt caching, reasoning, embeddings, and complete application structure.

## Development

Run `scripts/CI.sh` before committing.
The tests are offline and require no API keys.

## License

langchaint uses the [MIT License](LICENSE).
