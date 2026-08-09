"""OpenAI embedding requests over the official asynchronous SDK.

OpenAI 2.53.0 returns indexed float embeddings in `CreateEmbeddingResponse`.
OpenAI accepts at most 2048 inputs and 300,000 tokens per request.
This adapter counts tokens with `cl100k_base` through tiktoken 0.13.0.
"""

from collections.abc import Sequence
from functools import partial
from typing import override

import tiktoken
from openai import AsyncOpenAI, omit
from openai.types import Embedding as OpenAIEmbedding

from langchaint.adapter import ErrorClassification, _require_provider_name
from langchaint.cancellation import to_thread_cancellation_safe
from langchaint.embedding import (
    EmbeddingTask,
    Float2D,
    _EmbeddingAdapter,
    _validated_embeddings,
)
from langchaint.exceptions import EmbeddingOutputError
from langchaint.openai.shared import (
    OPENAI_FAILURE_TYPES,
    PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS,
    classify_openai,
)

_MAX_INPUTS_PER_REQUEST = 2048
_MAX_TOKENS_PER_REQUEST = 300_000
_TOKEN_ENCODING = "cl100k_base"


def _partition_inputs_sync(inputs: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Partition inputs under OpenAI's request count and token limits.

    Raises:
        ValueError: One input is empty.
    """
    encoding = tiktoken.get_encoding(_TOKEN_ENCODING)
    batches: list[tuple[str, ...]] = []
    current_batch: list[str] = []
    current_tokens = 0
    for input_text in inputs:
        if input_text == "":
            raise ValueError("OpenAI embedding inputs must not be empty strings")
        input_tokens = len(encoding.encode(input_text, disallowed_special=()))
        batch_is_full = len(current_batch) == _MAX_INPUTS_PER_REQUEST
        token_limit_would_be_exceeded = (
            bool(current_batch) and current_tokens + input_tokens > _MAX_TOKENS_PER_REQUEST
        )
        if batch_is_full or token_limit_would_be_exceeded:
            batches.append(tuple(current_batch))
            current_batch = []
            current_tokens = 0
        current_batch.append(input_text)
        current_tokens += input_tokens
    if current_batch:
        batches.append(tuple(current_batch))
    return tuple(batches)


class _OpenAIEmbeddingAdapter(_EmbeddingAdapter):
    """Map provider-neutral embedding operations to OpenAI embeddings requests."""

    failure_types = OPENAI_FAILURE_TYPES

    def __init__(self, *, client: AsyncOpenAI, model: str, dimension: int) -> None:
        """Store one client, cataloged model, and validated output dimension.

        Raises:
            ValueError: The client reaches another known provider.
        """
        _require_provider_name(
            client,
            provider_name="openai",
            provider_name_by_client_class=PROVIDER_NAME_BY_OPENAI_CLIENT_CLASS,
        )
        self.client = client
        self.model = model
        self.dimension = dimension

    @override
    async def partition_inputs(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> tuple[tuple[str, ...], ...]:
        """Partition copied inputs without blocking the event loop.

        Raises:
            ValueError: One input is empty.
            asyncio.CancelledError: Cancellation waits for token counting to finish.
        """
        del task
        return await to_thread_cancellation_safe(partial(_partition_inputs_sync, inputs))

    @override
    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        """Return normalized embeddings for one prepared request batch.

        Raises:
            openai.OpenAIError: The SDK request fails.
            EmbeddingOutputError: Response indexes or vectors violate output invariants.
        """
        del task
        response = await self.client.embeddings.create(
            input=tuple(inputs),
            model=self.model,
            dimensions=(omit if self.model == "text-embedding-ada-002" else self.dimension),
            encoding_format="float",
        )
        response_data = response.data
        if not isinstance(response_data, list):
            raise EmbeddingOutputError("OpenAI returned invalid embedding data")
        vectors_by_index: dict[int, Sequence[float]] = {}
        for embedding in response_data:
            if not isinstance(embedding, OpenAIEmbedding):
                raise EmbeddingOutputError("OpenAI returned invalid embedding data")
            index = embedding.index
            if (
                type(index) is not int
                or index < 0
                or index >= len(inputs)
                or index in vectors_by_index
            ):
                raise EmbeddingOutputError("OpenAI returned invalid embedding indexes")
            vectors_by_index[index] = embedding.embedding
        if set(vectors_by_index) != set(range(len(inputs))):
            raise EmbeddingOutputError("OpenAI returned invalid embedding indexes")
        ordered_vectors = [vectors_by_index[index] for index in range(len(inputs))]
        return _validated_embeddings(
            ordered_vectors,
            expected_rows=len(inputs),
            dimension=self.dimension,
        )

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Delegate failure classification to `classify_openai`."""
        return classify_openai(error)
