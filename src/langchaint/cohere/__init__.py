"""The Cohere backend provides Bedrock embedding accounts and model catalogs.

Importing this subpackage requires `boto3`.
Every cataloged model identifier is sent verbatim.
The account creates its default client before the first request admission.
The account shares one client and `SharedBackoff` across its embedding models.
AWS v4 documentation shows list and float-keyed response forms.

Request source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html.
Model source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-v4.html.
Pricing source: https://aws.amazon.com/bedrock/pricing/.
Recheck that page before relying on its prices.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, ClassVar, Literal, overload, override

from pydantic import ConfigDict, TypeAdapter, ValidationError

try:
    import boto3
except ModuleNotFoundError as exc:
    if exc.name != "boto3":
        raise
    raise ModuleNotFoundError(
        "langchaint's cohere backend requires the boto3 package; install boto3."
    ) from exc

from botocore.config import Config
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError

from langchaint.account_base import AccountBase
from langchaint.account_state import AccountClosedError, AccountState
from langchaint.adapter import (
    ErrorClassification,
    retry_after_seconds_from_headers,
)
from langchaint.cancellation import await_task_cancellation_safe, to_thread_cancellation_safe
from langchaint.embedding import (
    EmbeddingModel,
    EmbeddingTask,
    Float2D,
    _EmbeddingAdapter,
    _validated_embeddings,
)
from langchaint.exceptions import EmbeddingOutputError
from langchaint.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


type CohereEmbedV4Dimension = Literal[256, 512, 1024, 1536]
"""Output dimensions accepted by every cataloged Cohere Embed v4 model."""

type CohereEmbedV4ModelName = Literal[
    "cohere.embed-v4:0",
    "global.cohere.embed-v4:0",
    "us.cohere.embed-v4:0",
    "eu.cohere.embed-v4:0",
]
"""Cataloged Cohere Embed v4 identifiers for Amazon Bedrock."""

type CohereEmbedV3ModelName = Literal[
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
]
"""Cataloged Cohere Embed v3 identifiers for Amazon Bedrock."""

type _CohereBedrockEmbeddingModelName = CohereEmbedV4ModelName | CohereEmbedV3ModelName
type _CohereInputType = Literal[
    "search_document",
    "search_query",
    "classification",
    "clustering",
]

_COHERE_EMBED_V4_MODELS: frozenset[CohereEmbedV4ModelName] = frozenset({
    "cohere.embed-v4:0",
    "global.cohere.embed-v4:0",
    "us.cohere.embed-v4:0",
    "eu.cohere.embed-v4:0",
})
_COHERE_EMBED_V3_MODELS: frozenset[CohereEmbedV3ModelName] = frozenset({
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
})
COHERE_BEDROCK_EMBEDDING_MODELS: frozenset[_CohereBedrockEmbeddingModelName] = (
    _COHERE_EMBED_V4_MODELS | _COHERE_EMBED_V3_MODELS
)
"""Cohere embedding identifiers accepted by `CohereBedrockAccount`."""
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
_BEDROCK_RESPONSE = TypeAdapter(dict[str, object])
type _CohereEmbeddings = list[list[float]] | dict[Literal["float"], list[list[float]]]
_FLOAT_EMBEDDINGS: TypeAdapter[_CohereEmbeddings] = TypeAdapter(
    _CohereEmbeddings,
    config=ConfigDict(strict=True),
)
_DIMENSION_UNSET = object()


def _parse_cohere_bedrock(failure: Exception) -> Verdict:
    """Map one Bedrock `ClientError` to its retry verdict.

    `botocore==1.43.67` maps Bedrock error responses to `ClientError`.
    Its service model marks `ThrottlingException` as throttling.
    It marks `ModelNotReadyException` as transient without throttling.
    Statuses 408, 500, and 503 also identify transient errors.
    """
    if not isinstance(failure, ClientError):
        return DoNotRetry()
    metadata = failure.response.get("ResponseMetadata")
    if metadata is None:
        return DoNotRetry()
    status_code = metadata["HTTPStatusCode"]
    error_code = failure.response.get("Error", {}).get("Code")
    retry_after = retry_after_seconds_from_headers(metadata.get("HTTPHeaders", {}))
    if error_code == "ThrottlingException":
        return PauseAll(retry_after=retry_after)
    if status_code in (408, 429) or status_code >= 500:
        return RetryThisOne(retry_after=retry_after)
    return DoNotRetry()


def _classify_cohere_bedrock(error: Exception) -> ErrorClassification:
    """Classify errors outside `_parse_cohere_bedrock` terminal verdicts."""
    if isinstance(error, BotocoreConnectionError | HTTPClientError):
        return "transient"
    return "unknown_exception"


def _require_one_sdk_attempt(client: BedrockRuntimeClient) -> None:
    """Require one request attempt from a passed Bedrock client.

    Raises:
        ValueError: The client retry configuration is missing or differs.
    """
    retries = vars(client.meta.config).get("retries")
    if not isinstance(retries, dict) or retries.get("total_max_attempts") != 1:
        raise ValueError("client.meta.config.retries must set total_max_attempts to 1")


class _CohereBedrockClientManager:
    """Create, lease, and close one Bedrock Runtime client."""

    def __init__(
        self,
        *,
        account_state: AccountState,
        aws_region: str | None,
        client: BedrockRuntimeClient | None,
    ) -> None:
        """Store client creation inputs without creating a client."""
        self._account_state = account_state
        self._aws_region = aws_region
        self._client = client
        self._owns_client = client is None
        self._lock = asyncio.Lock()
        self._creation_task: asyncio.Task[None] | None = None
        self._active_leases = 0
        self._leases_drained = asyncio.Event()
        self._leases_drained.set()

    async def _create_and_install(self) -> None:
        """Create one client and install it while the account remains open.

        Raises:
            AccountClosedError: Account closure started during creation.
            Exception: `boto3.client` or client closure failed.
        """
        client = await to_thread_cancellation_safe(
            lambda: boto3.client(
                "bedrock-runtime",
                region_name=self._aws_region,
                config=Config(retries={"total_max_attempts": 1}),
            )
        )
        close_client = False
        async with self._lock:
            try:
                self._account_state.ensure_open()
            except AccountClosedError:
                close_client = True
            if not close_client:
                self._client = client
            self._creation_task = None
        if close_client:
            await to_thread_cancellation_safe(client.close)
            raise AccountClosedError("Account is closed")

    async def prepare(self) -> None:
        """Create the default client once before request admission.

        A cancelled caller waits for creation and leaves the client installed.

        Raises:
            AccountClosedError: Account closure started.
            asyncio.CancelledError: This caller was cancelled after creation settled.
            Exception: `boto3.client` or client closure failed.
        """
        self._account_state.ensure_open()
        async with self._lock:
            self._account_state.ensure_open()
            if self._client is not None:
                return
            if self._creation_task is None:
                self._creation_task = asyncio.create_task(self._create_and_install())
            creation_task = self._creation_task
        await await_task_cancellation_safe(creation_task)

    @asynccontextmanager
    async def request_lease(self) -> AsyncGenerator[BedrockRuntimeClient]:
        """Lease the prepared client through one complete SDK attempt.

        Yields:
            The prepared client.

        Raises:
            AccountClosedError: Account closure started.
            RuntimeError: `prepare()` has not installed the client.
        """
        async with self._lock:
            self._account_state.ensure_open()
            client = self._client
            if client is None:
                raise RuntimeError("prepare() must install the client before a request lease")
            self._active_leases += 1
            self._leases_drained.clear()
        try:
            yield client
        finally:
            async with self._lock:
                self._active_leases -= 1
                if self._active_leases == 0:
                    self._leases_drained.set()

    async def aclose(self) -> None:
        """Reject leases and close an internally created client.

        Raises:
            Exception: Client creation or client closure failed.
        """
        async with self._lock:
            creation_task = self._creation_task
        if creation_task is not None:
            try:
                await creation_task
            except AccountClosedError:
                pass
        await self._leases_drained.wait()
        async with self._lock:
            client = self._client if self._owns_client else None
            self._client = None
        if client is not None:
            await to_thread_cancellation_safe(client.close)


class _CohereBedrockEmbeddingAdapter(_EmbeddingAdapter):
    """Map provider-neutral embedding requests to Cohere on Bedrock."""

    failure_types: ClassVar[tuple[type[Exception], ...]] = (ClientError,)

    def __init__(
        self,
        *,
        client_manager: _CohereBedrockClientManager,
        model: _CohereBedrockEmbeddingModelName,
        dimension: int,
        supports_dimension: bool,
    ) -> None:
        """Store one model's validated request configuration."""
        self._client_manager = client_manager
        self.model = model
        self.dimension = dimension
        self._supports_dimension = supports_dimension

    def _request_body(
        self,
        inputs: Sequence[str],
        *,
        task: EmbeddingTask,
    ) -> bytes:
        payload: dict[str, object] = {
            "texts": list(inputs),
            "input_type": _COHERE_INPUT_TYPE_BY_TASK[task],
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
        if self._supports_dimension:
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
        current: list[str] = []
        target_bytes = (
            _COHERE_V4_BODY_TARGET_BYTES
            if self._supports_dimension
            else _INVOKE_MODEL_MAX_BODY_BYTES
        )
        for input_text in inputs:
            current.append(input_text)
            candidate_bytes = len(self._request_body(current, task=task))
            if len(current) == 1:
                if candidate_bytes > _INVOKE_MODEL_MAX_BODY_BYTES:
                    raise ValueError("A Cohere input exceeds InvokeModel's body limit")
                continue
            if len(current) > _COHERE_MAX_INPUTS or candidate_bytes > target_bytes:
                next_input = current.pop()
                batches.append(tuple(current))
                current = [next_input]
                singleton_bytes = len(self._request_body(current, task=task))
                if singleton_bytes > _INVOKE_MODEL_MAX_BODY_BYTES:
                    raise ValueError("A Cohere input exceeds InvokeModel's body limit")
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    @override
    async def prepare(self) -> None:
        """Create the default Bedrock client before request admission.

        Raises:
            AccountClosedError: Account closure started.
            asyncio.CancelledError: This caller was cancelled after creation settled.
            Exception: Client creation or client closure failed.
        """
        await self._client_manager.prepare()

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
    ) -> Float2D:
        """Invoke Bedrock and validate its response vectors.

        Raises:
            EmbeddingOutputError: Cohere returned an invalid response.
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
        try:
            payload = _BEDROCK_RESPONSE.validate_json(body_bytes)
            embeddings_response = _FLOAT_EMBEDDINGS.validate_python(payload.get("embeddings"))
        except ValidationError as error:
            raise EmbeddingOutputError("Cohere returned an invalid embedding response") from error
        embeddings = (
            embeddings_response["float"]
            if isinstance(embeddings_response, dict)
            else embeddings_response
        )
        return _validated_embeddings(
            embeddings,
            expected_rows=len(inputs),
            dimension=self.dimension,
        )

    @override
    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        """Invoke Bedrock and close its response body outside the event loop.

        Raises:
            AccountClosedError: Account closure started.
            EmbeddingOutputError: Cohere returned invalid vectors.
            asyncio.CancelledError: Cancellation follows completed synchronous work.
            Exception: The SDK request, read, or body closure failed.
        """
        async with self._client_manager.request_lease() as client:
            return await to_thread_cancellation_safe(lambda: self._invoke(client, inputs, task))

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify an exception outside `failure_types`."""
        return _classify_cohere_bedrock(error)


class CohereBedrockAccount(AccountBase):
    """Share one Bedrock Runtime client and `SharedBackoff`."""

    def __init__(  # noqa: PLR0913 (every request policy reaches SharedBackoff)
        self,
        *,
        aws_region: str | None = None,
        client: BedrockRuntimeClient | None = None,
        max_concurrent_requests: int | None = 8,
        max_request_starts_per_second: float = 50.0,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
    ) -> None:
        """Build a Cohere Bedrock account without creating a client.

        `aws_region` selects the region for an account-created client.
        A passed `client` remains caller-owned.
        A passed `client` must disable SDK retries.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the minimum adaptive wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            ValueError: `client` accompanies `aws_region`.
                Also raised when client retries remain enabled.
                Also raised when a `SharedBackoff` setting is invalid.
        """
        if client is not None and aws_region is not None:
            raise ValueError("Pass at most one of client= or aws_region=")
        if client is not None:
            _require_one_sdk_attempt(client)
        super().__init__(
            parse=_parse_cohere_bedrock,
            failure_types=_CohereBedrockEmbeddingAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.aws_region = aws_region
        self._client_manager = _CohereBedrockClientManager(
            account_state=self._state,
            aws_region=aws_region,
            client=client,
        )
        self._register_owned_close(self._client_manager.aclose)

    @overload
    def embedding_model(
        self,
        model: CohereEmbedV4ModelName,
        *,
        dimension: CohereEmbedV4Dimension = 1536,
        max_attempts: int = 3,
    ) -> EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: CohereEmbedV3ModelName,
        *,
        max_attempts: int = 3,
    ) -> EmbeddingModel: ...

    def embedding_model(
        self,
        model: _CohereBedrockEmbeddingModelName,
        *,
        dimension: CohereEmbedV4Dimension | object = _DIMENSION_UNSET,
        max_attempts: int = 3,
    ) -> EmbeddingModel:
        """Build an `EmbeddingModel` for one cataloged Bedrock model.

        V4 models default to 1536 dimensions.
        V3 models always return 1024 dimensions.

        Raises:
            RuntimeError: This account is closed.
            ValueError: `model`, `dimension`, or `max_attempts` is invalid.
        """
        self._state.ensure_open()
        if model in _COHERE_EMBED_V4_MODELS:
            selected_dimension = 1536 if dimension is _DIMENSION_UNSET else dimension
            if (
                type(selected_dimension) is not int
                or selected_dimension not in _COHERE_EMBED_V4_DIMENSIONS
            ):
                raise ValueError(
                    f"dimension is invalid for Cohere Embed v4: {selected_dimension!r}"
                )
            supports_dimension = True
        elif model in _COHERE_EMBED_V3_MODELS:
            if dimension is not _DIMENSION_UNSET:
                raise ValueError("Cohere Embed v3 accepts no dimension argument")
            selected_dimension = 1024
            supports_dimension = False
        else:
            raise ValueError(f"model {model!r} is not in COHERE_BEDROCK_EMBEDDING_MODELS")
        adapter = _CohereBedrockEmbeddingAdapter(
            client_manager=self._client_manager,
            model=model,
            dimension=selected_dimension,
            supports_dimension=supports_dimension,
        )
        return self._embedding_model(adapter, max_attempts=max_attempts)


__all__ = [
    "COHERE_BEDROCK_EMBEDDING_MODELS",
    "CohereBedrockAccount",
    "CohereEmbedV3ModelName",
    "CohereEmbedV4Dimension",
    "CohereEmbedV4ModelName",
]
