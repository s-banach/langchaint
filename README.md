# langchaint

langchaint provides typed generation and embeddings with raw provider responses and per-attempt billing.
Alpha: the API is unstable and may change without notice.

## Purpose

langchaint provides provider-neutral async `BoundLLM` and `EmbeddingModel` interfaces.
langchaint provides no agent class or agent loop.
langchaint uses pyrefly's strict `all` preset.
Applications should use a strict type checker because langchaint leaves argument-type checks to static checking.

## Install and authenticate

langchaint requires Python 3.13 or newer.
Install one or more backend extras, such as `pip install "langchaint[openai]"`.
Each extra declares langchaint's tested dependency lower bounds.
Applications may add tighter dependency pins.
Top-level `import langchaint` requires no provider SDK or `numpy`.
Backend imports report missing dependencies through `ModuleNotFoundError`.

| Backend | Class | Creates | Install | Default credentials |
| --- | --- | --- | --- | --- |
| Anthropic | `Anthropic` | `LLM` | `langchaint[anthropic]` | `ANTHROPIC_API_KEY` |
| Anthropic on Amazon Bedrock | `AnthropicBedrock` | `LLM` | `langchaint[anthropic-bedrock]` | AWS credential provider chain |
| Cohere on Amazon Bedrock | `CohereBedrock` | `EmbeddingModel` | `langchaint[cohere-bedrock]` | AWS credential provider chain |
| DeepSeek | `DeepSeek` | `LLM` | `langchaint[deepseek]` | `DEEPSEEK_API_KEY` |
| Gemini | `Gemini` | `LLM` | `langchaint[gemini]` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| OpenAI | `OpenAI` | `LLM` | `langchaint[openai]` | `OPENAI_API_KEY` |
| OpenAI embeddings | `OpenAI` | `EmbeddingModel` | `langchaint[openai-embedding]` | `OPENAI_API_KEY` |
| OpenAI on Amazon Bedrock | `OpenAIBedrock` | `LLM` | `langchaint[openai-bedrock]` | AWS credential provider chain |

Install `langchaint[tracing]` to use the OTel tracing subpackage.

Every listed class accepts `client=`.
Amazon Bedrock classes use the AWS credential provider chain.
It includes environment credentials, profiles, SSO, containers, and instance roles.
Pass `aws_region=` to select a Bedrock region explicitly.

## Generation quickstart

```python
import asyncio

from langchaint.openai import OpenAI


async def main() -> None:
    openai = OpenAI()
    bound_llm = openai.model("gpt-5.6-terra").bind(system_prompt="Answer clearly and concisely.")
    response = await bound_llm.generate_one("Why is the sky blue?")
    print(response.output, response.usage.cost_in_usd)


asyncio.run(main())
```

`response.output` is assistant text.
`regional_processing=False` uses the standard `1.0` token-price multiplier.
Use `regional_processing=True` for an endpoint with regional processing.
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
The OpenAI adapter sends no `task` request field.
`EmbeddingModel.embed()` returns normalized `Float2D` values with `numpy.float32` elements.
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

`terra` and `sol` share `openai.client` and one `SharedBackoff`.
`max_concurrent_requests` and `max_request_starts_per_second` apply across both `LLM` values.
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

Call `LLM.bind()` before `generate_one`, `generate_many`, or `stream_one`.
Use `BoundLLM.rebind()` to replace selected binding fields.
Pass `tools=[tool]` to `LLM.bind()` and dispatch through `BoundLLM.tool_manager`.
`LLM.bind(max_attempts=...)` limits requests for one `GenerationInput`, including the first.
`embedding_model(max_attempts=...)` limits requests for one batch of inputs sent together, including the first.
`LLM.bind()` uses `Adapter.automatic_cache_breakpoints_default` unless `automatic_cache_breakpoints` overrides it.
`cache_breakpoint=True` requests an explicit breakpoint at that `ContentPart`.
`GenerateResult` and `GenerationError` include paid `Usage` across attempts.

## More examples

[`examples/README.md`](examples/README.md) indexes migration guidance and examples.
The examples cover structured output, batches, streaming, tools, tracing, pricing, failures, prompt caching, reasoning, embeddings, and application structure.

## Development

Run `scripts/CI.sh` before committing.
The tests are offline and require no API keys.

## License

langchaint uses the [MIT License](LICENSE).
