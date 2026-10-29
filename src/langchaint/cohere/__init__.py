"""Construct cataloged Cohere `EmbeddingModel` values for Bedrock.

Importing this subpackage requires `boto3`.
Every cataloged model identifier is sent verbatim.
`CohereBedrock` creates its default client before the first request admission.
Each `EmbeddingModel` shares that client and one `SharedBackoff`.
AWS v4 documentation shows list and float-keyed response forms.
Cohere v4 and v3 `invoke_model` responses carry the billed input token count.
Its header is `x-amzn-bedrock-input-token-count`, observed on 2026-10-13.
botocore 1.43.112 does not model that header.
The adapter reads it from `ResponseMetadata.HTTPHeaders`, which holds lowercased header names.
Cataloged models receive `COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS`.
An identifier with a `COHERE_BEDROCK_CROSS_REGION_MULTIPLIER` prefix resolves through the unprefixed identifier.
`CohereBedrock` multiplies that rate by the prefix multiplier unless `apply_cross_region_multiplier=False`.

Request source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html.
Model source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-v4.html.
Token price source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json.
Recheck https://aws.amazon.com/bedrock/pricing/ before relying on these prices.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Literal, overload, override

from pydantic import ConfigDict, TypeAdapter, ValidationError

try:
    import boto3
except ModuleNotFoundError as exc:
    if exc.name != "boto3":
        raise
    raise ModuleNotFoundError(
        "langchaint's cohere backend requires boto3; install langchaint[cohere-bedrock]."
    ) from exc

from botocore.config import Config
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError

from langchaint.adapter import (
    RequestFailure,
    retry_after_seconds_from_headers,
)
from langchaint.cohere._generated_pricing import (
    _COHERE_EMBED_V3_MODELS,
    _COHERE_EMBED_V4_MODELS,
    COHERE_BEDROCK_CROSS_REGION_MULTIPLIER,
    COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS,
    CohereEmbedV3ModelId,
    CohereEmbedV4ModelId,
    _CohereBedrockEmbeddingModelId,
)
from langchaint.common.bedrock import BedrockCatalogResolution, resolve_bedrock_catalog_id
from langchaint.common.exceptions import EmbeddingOutputError
from langchaint.concurrency.cancellation import (
    await_task_cancellation_safe,
    to_thread_cancellation_safe,
)
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.embedding import (
    EmbeddingModel,
    EmbeddingTask,
    _EmbeddingAdapter,
    _EmbeddingBatchResponse,
    _validated_embeddings,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from langchaint.common.sequence_not_str import SequenceNotStr
    from langchaint.embedding_requests import EmbeddingObserver


type CohereEmbedV4Dimension = Literal[256, 512, 1024, 1536]
"""Output dimensions accepted by every cataloged Cohere Embed v4 model."""

type _CohereInputType = Literal[
    "search_document",
    "search_query",
    "classification",
    "clustering",
]

COHERE_BEDROCK_EMBEDDING_MODELS: frozenset[_CohereBedrockEmbeddingModelId] = (
    _COHERE_EMBED_V4_MODELS | _COHERE_EMBED_V3_MODELS
)
"""Cataloged unprefixed Cohere embedding identifiers.

`CohereBedrock` also accepts each under a `COHERE_BEDROCK_CROSS_REGION_MULTIPLIER` prefix.
"""
_COHERE_EMBED_V4_DIMENSIONS: frozenset[CohereEmbedV4Dimension] = frozenset({256, 512, 1024, 1536})
_COHERE_INPUT_TYPE_BY_TASK: dict[EmbeddingTask, _CohereInputType] = {
    "retrieval_document": "search_document",
    "retrieval_query": "search_query",
    "classification": "classification",
    "clustering": "clustering",
}
_COHERE_MAX_INPUTS = 96
_COHERE_V4_BODY_TARGET_BYTES = 20_000_000
_INVOKE_MODEL_MAX_BODY_BYTES = 25_000_000
_JSON_CONTENT_TYPE = "application/json"
_INPUT_TOKEN_COUNT_HEADER = "x-amzn-bedrock-input-token-count"
_BEDROCK_RESPONSE = TypeAdapter(dict[str, object])
type _CohereEmbeddings = list[list[float]] | dict[Literal["float"], list[list[float]]]
_FLOAT_EMBEDDINGS: TypeAdapter[_CohereEmbeddings] = TypeAdapter(
    _CohereEmbeddings,
    config=ConfigDict(strict=True),
)
_DIMENSION_UNSET = object()


def _cohere_bedrock_request_failure(error: Exception) -> RequestFailure:
    """Return the `RequestFailure` of one exception a Bedrock request raised.

    `botocore==1.43.67` maps Bedrock errors to `ClientError`.
    Its service model marks `ThrottlingException` as throttling, which is transient and pauses the quota.
    It marks `ModelNotReadyException` as transient without throttling.
    Statuses 408, 429, and 500 or above are transient without a pause.
    A botocore connection failure is transient without a pause.
    Every other exception is `unknown_exception`, including a `ClientError` without response metadata.
    """
    if isinstance(error, BotocoreConnectionError | HTTPClientError):
        return RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None)
    unknown_exception = RequestFailure(
        kind="unknown_exception", pauses_quota=False, retry_after_seconds=None
    )
    if not isinstance(error, ClientError):
        return unknown_exception
    metadata = error.response.get("ResponseMetadata")
    if metadata is None:
        return unknown_exception
    status_code = metadata["HTTPStatusCode"]
    error_code = error.response.get("Error", {}).get("Code")
    retry_after_seconds = retry_after_seconds_from_headers(metadata.get("HTTPHeaders", {}))
    if error_code == "ThrottlingException":
        return RequestFailure(
            kind="transient", pauses_quota=True, retry_after_seconds=retry_after_seconds
        )
    if status_code in (408, 429) or status_code >= 500:
        return RequestFailure(
            kind="transient", pauses_quota=False, retry_after_seconds=retry_after_seconds
        )
    return unknown_exception


def _require_one_sdk_request(client: BedrockRuntimeClient) -> None:
    """Require that a passed Bedrock client sends one request per SDK call.

    Raises:
        ValueError: The client retry configuration is missing or differs.
    """
    retries = vars(client.meta.config).get("retries")
    if not isinstance(retries, dict) or retries.get("total_max_attempts") != 1:
        raise ValueError("client.meta.config.retries must set total_max_attempts to 1")


class _CohereBedrockClientCache:
    """Share one Bedrock Runtime client across embedding models."""

    def __init__(
        self,
        *,
        aws_region: str | None,
        client: BedrockRuntimeClient | None,
    ) -> None:
        """Store client creation inputs without creating a client."""
        self._aws_region = aws_region
        self._client = client
        self._creation_task: asyncio.Task[BedrockRuntimeClient] | None = None

    async def _create(self) -> BedrockRuntimeClient:
        """Create the default client.

        Raises:
            Exception: `boto3.client` failed.
        """
        client = await to_thread_cancellation_safe(
            lambda: boto3.client(
                "bedrock-runtime",
                region_name=self._aws_region,
                config=Config(retries={"total_max_attempts": 1}),
            )
        )
        self._client = client
        return client

    async def prepare(self) -> None:
        """Create the default client once before request admission.

        A cancelled caller waits for client creation to settle.

        Raises:
            asyncio.CancelledError: This caller was cancelled after creation settled.
            Exception: `boto3.client` failed.
        """
        if self._client is not None:
            return
        if self._creation_task is None:
            self._creation_task = asyncio.create_task(self._create())
        creation_task = self._creation_task
        try:
            await await_task_cancellation_safe(creation_task)
        finally:
            if self._creation_task is creation_task:
                self._creation_task = None

    def client(self) -> BedrockRuntimeClient:
        """Return the prepared client.

        Raises:
            RuntimeError: `prepare()` has not created the client.
        """
        if self._client is None:
            raise RuntimeError("prepare() must create the client before a request")
        return self._client


class _CohereBedrockEmbeddingAdapter(_EmbeddingAdapter):
    """Map provider-neutral embedding requests to Cohere on Bedrock."""

    def __init__(
        self,
        *,
        client_cache: _CohereBedrockClientCache,
        model: str,
        is_embed_v4: bool,
        dimension: int,
    ) -> None:
        """Store one model's validated request configuration.

        `is_embed_v4` states whether `model` resolves to a Cohere Embed v4 catalog identifier.
        """
        self._client_cache = client_cache
        self.provider_name = "aws.bedrock"
        self.model = model
        self.is_embed_v4 = is_embed_v4
        self.dimension = dimension

    def _request_body(
        self,
        inputs: SequenceNotStr[str],
        *,
        task: EmbeddingTask,
    ) -> bytes:
        payload: dict[str, object] = {
            "texts": list(inputs),
            "input_type": _COHERE_INPUT_TYPE_BY_TASK[task],
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
        if self.is_embed_v4:
            payload["output_dimension"] = self.dimension
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()

    def _partition_inputs(
        self, inputs: tuple[str, ...], task: EmbeddingTask
    ) -> tuple[tuple[str, ...], ...]:
        """Partition inputs under Cohere and InvokeModel limits.

        Raises:
            ValueError: One input exceeds InvokeModel's body limit.
        """
        batches: list[tuple[str, ...]] = []
        current: tuple[str, ...] = ()
        target_bytes = (
            _COHERE_V4_BODY_TARGET_BYTES if self.is_embed_v4 else _INVOKE_MODEL_MAX_BODY_BYTES
        )
        for input_text in inputs:
            candidate = (*current, input_text)
            body_size_bytes = len(self._request_body(candidate, task=task))
            if current and (len(candidate) > _COHERE_MAX_INPUTS or body_size_bytes > target_bytes):
                batches.append(current)
                current = (input_text,)
                body_size_bytes = len(self._request_body(current, task=task))
            else:
                current = candidate
            if len(current) == 1 and body_size_bytes > _INVOKE_MODEL_MAX_BODY_BYTES:
                raise ValueError("A Cohere input exceeds InvokeModel's body limit")
        if current:
            batches.append(current)
        return tuple(batches)

    @override
    async def prepare(self) -> None:
        """Create the default Bedrock client before request admission.

        Raises:
            asyncio.CancelledError: This caller was cancelled after creation settled.
            Exception: Client creation failed.
        """
        await self._client_cache.prepare()

    @override
    async def partition_inputs(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> tuple[tuple[str, ...], ...]:
        """Partition inputs by count and serialized request bytes.

        Raises:
            ValueError: One input exceeds InvokeModel's body limit.
            asyncio.CancelledError: Cancellation follows completed partitioning.
        """
        return await to_thread_cancellation_safe(lambda: self._partition_inputs(inputs, task))

    def _invoke(
        self,
        client: BedrockRuntimeClient,
        inputs: tuple[str, ...],
        task: EmbeddingTask,
    ) -> _EmbeddingBatchResponse:
        """Invoke Bedrock and validate its response vectors and billed token count.

        Raises:
            EmbeddingOutputError: Cohere returned an invalid response, or one without a token count.
            Exception: The SDK request, read, or body closure failed.
        """
        response = client.invoke_model(
            body=self._request_body(inputs, task=task),
            modelId=self.model,
            accept=_JSON_CONTENT_TYPE,
            contentType=_JSON_CONTENT_TYPE,
        )
        response_body = response["body"]
        try:
            body_bytes = response_body.read()
        finally:
            response_body.close()
        token_count_text = response["ResponseMetadata"]["HTTPHeaders"].get(
            _INPUT_TOKEN_COUNT_HEADER
        )
        if token_count_text is None:
            raise EmbeddingOutputError(
                f"Bedrock returned no {_INPUT_TOKEN_COUNT_HEADER} header", input_tokens_total=None
            )
        input_tokens_total = int(token_count_text)
        try:
            payload = _BEDROCK_RESPONSE.validate_json(body_bytes)
            embeddings_response = _FLOAT_EMBEDDINGS.validate_python(payload.get("embeddings"))
        except ValidationError as error:
            raise EmbeddingOutputError(
                "Cohere returned an invalid embedding response",
                input_tokens_total=input_tokens_total,
            ) from error
        embeddings = (
            embeddings_response["float"]
            if isinstance(embeddings_response, dict)
            else embeddings_response
        )
        vectors = _validated_embeddings(
            embeddings,
            expected_rows=len(inputs),
            dimension=self.dimension,
            input_tokens_total=input_tokens_total,
        )
        return _EmbeddingBatchResponse(
            vectors=vectors, input_tokens_total=input_tokens_total, model_served=None
        )

    @override
    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> _EmbeddingBatchResponse:
        """Invoke Bedrock and close its response body outside the event loop.

        Raises:
            EmbeddingOutputError: Cohere returned invalid vectors or no token count.
            asyncio.CancelledError: Cancellation follows completed synchronous work.
            Exception: The SDK request, read, or body closure failed.
        """
        client = self._client_cache.client()
        return await to_thread_cancellation_safe(lambda: self._invoke(client, inputs, task))

    @override
    def request_failure(self, error: Exception) -> RequestFailure:
        """Delegate to `_cohere_bedrock_request_failure`."""
        return _cohere_bedrock_request_failure(error)


class CohereBedrock:
    """Create `EmbeddingModel` values for Cohere models on Bedrock."""

    def __init__(
        self,
        *,
        aws_region: str | None = None,
        client: BedrockRuntimeClient | None = None,
        apply_cross_region_multiplier: bool = True,
        shared_backoff: SharedBackoff | None = None,
        observer: EmbeddingObserver | None = None,
    ) -> None:
        """Build `CohereBedrock` without creating a client.

        `aws_region` selects the region for a client created by `CohereBedrock`.
        A passed `client` must disable SDK retries.
        `apply_cross_region_multiplier=False` leaves a prefixed identifier's catalog rate unmultiplied.
        `shared_backoff` admits every request of the created `EmbeddingModel` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every request of the created `EmbeddingModel` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` accompanies `aws_region`.
                Also raised when client retries remain enabled.
        """
        if client is not None and aws_region is not None:
            raise ValueError("Pass at most one of client= or aws_region=")
        if client is not None:
            _require_one_sdk_request(client)
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.aws_region: str | None = aws_region
        self.apply_cross_region_multiplier: bool = apply_cross_region_multiplier
        self._client_cache = _CohereBedrockClientCache(
            aws_region=aws_region,
            client=client,
        )

    @overload
    def embedding_model(
        self,
        model: CohereEmbedV3ModelId,
        *,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: CohereEmbedV4ModelId | str,
        *,
        dimension: CohereEmbedV4Dimension = 1536,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    def embedding_model(
        self,
        model: str,
        *,
        dimension: CohereEmbedV4Dimension | object = _DIMENSION_UNSET,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel:
        """Build an `EmbeddingModel` for one cataloged Bedrock model.

        `model` is sent verbatim.
        An exact catalog identifier selects the model and its rate.
        Otherwise a `COHERE_BEDROCK_CROSS_REGION_MULTIPLIER` prefix is stripped before the catalog lookups.
        The type checker accepts `dimension` for a prefixed v3 identifier, which is typed as `str`.
        V4 models default to 1536 dimensions.
        V3 models always return 1024 dimensions.
        `usd_per_million_tokens` prices input tokens and replaces the catalog rate.
        A NaN prefix multiplier makes the catalog rate NaN.

        Raises:
            ValueError: `model`, `dimension`, or `max_requests` is invalid.
        """
        resolution = resolve_bedrock_catalog_id(
            model,
            catalog_ids=COHERE_BEDROCK_EMBEDDING_MODELS,
            cross_region_multipliers=COHERE_BEDROCK_CROSS_REGION_MULTIPLIER,
        )
        is_embed_v4 = resolution.catalog_id in _COHERE_EMBED_V4_MODELS
        if is_embed_v4:
            dimension = 1536 if dimension is _DIMENSION_UNSET else dimension
            if type(dimension) is not int or dimension not in _COHERE_EMBED_V4_DIMENSIONS:
                raise ValueError(f"dimension is invalid for Cohere Embed v4: {dimension!r}")
        elif resolution.catalog_id in _COHERE_EMBED_V3_MODELS:
            if dimension is not _DIMENSION_UNSET:
                raise ValueError("Cohere Embed v3 accepts no dimension argument")
            dimension = 1024
        else:
            raise ValueError(f"model {model!r} is not in COHERE_BEDROCK_EMBEDDING_MODELS")
        adapter = _CohereBedrockEmbeddingAdapter(
            client_cache=self._client_cache,
            model=model,
            is_embed_v4=is_embed_v4,
            dimension=dimension,
        )
        return EmbeddingModel(
            adapter=adapter,
            shared_backoff=self._shared_backoff,
            max_requests=max_requests,
            usd_per_million_tokens=(
                self._catalog_usd_per_million_tokens(resolution)
                if usd_per_million_tokens is None
                else usd_per_million_tokens
            ),
            observer=self._observer,
        )

    def _catalog_usd_per_million_tokens(self, resolution: BedrockCatalogResolution) -> float:
        """Return the catalog input-token rate for `resolution`.

        `apply_cross_region_multiplier` multiplies the rate by the prefix multiplier of `resolution`.
        """
        catalog_rate = COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS[resolution.catalog_id]
        if resolution.cross_region_multiplier is None or not self.apply_cross_region_multiplier:
            return catalog_rate
        return catalog_rate * resolution.cross_region_multiplier


__all__ = [
    "COHERE_BEDROCK_CROSS_REGION_MULTIPLIER",
    "COHERE_BEDROCK_EMBEDDING_MODELS",
    "COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS",
    "CohereBedrock",
    "CohereEmbedV3ModelId",
    "CohereEmbedV4Dimension",
    "CohereEmbedV4ModelId",
]
