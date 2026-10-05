"""Provider-neutral embedding execution and output validation.

`EmbeddingModel` copies inputs before partitioning them into provider requests.
`EmbeddingModel` retries each embedding batch independently.
All requests use its `SharedBackoff`.
The adapter owns provider batching and response decoding.
The returned matrix preserves input order.
The returned matrix owns writable `float32` storage.
Every returned row has L2 norm one.
"""

import asyncio
from collections.abc import Sequence
from functools import partial
from typing import Literal, Protocol

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    raise ModuleNotFoundError(
        "langchaint embeddings require the numpy package; install numpy."
    ) from exc

from langchaint.common.exceptions import EmbeddingOutputError
from langchaint.common.request_failure import RequestFailure, _request_failure_of
from langchaint.common.sequence_not_str import SequenceNotStr
from langchaint.concurrency.run_many import max_pending_for_requests, run_many
from langchaint.concurrency.shared_backoff import (
    PrivateBackoff,
    SharedBackoff,
)

type EmbeddingTask = Literal[
    "retrieval_document",
    "retrieval_query",
    "classification",
    "clustering",
]
"""The purpose a provider may use while creating embeddings."""

type Float2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
"""A two-axis `np.ndarray` containing `np.float32` values."""

type _FloatMatrixValues = Sequence[Sequence[float]] | Float2D


class _EmbeddingAdapter(Protocol):
    """The provider-specific operations `EmbeddingModel` executes."""

    model: str
    dimension: int

    def request_failure(self, error: Exception) -> RequestFailure:
        """Return what one failed request means, as `Adapter.request_failure` does, without raising."""
        ...

    async def prepare(self) -> None:
        """Complete preparation without work."""
        return None  # noqa: RET501 (keeps pyrefly from treating this method as abstract)

    async def partition_inputs(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> tuple[tuple[str, ...], ...]:
        """Partition copied inputs into ordered provider request batches.

        Raises:
            ValueError: An input cannot form a provider request.
            asyncio.CancelledError: Partitioning was cancelled after synchronous work settled.
        """
        ...

    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        """Return validated vectors for one request batch.

        Raises:
            Exception: The provider request or response validation failed.
        """
        ...


def _validated_embeddings(
    values: _FloatMatrixValues,
    *,
    expected_rows: int,
    dimension: int,
    copy: bool = True,
) -> Float2D:
    """Convert provider values into a normalized owned matrix.

    Set `copy=False` only for owned C-contiguous `float32` matrices.

    Raises:
        EmbeddingOutputError: Values violate the public output invariants.
    """
    try:
        vectors = np.array(values, dtype=np.float32, order="C", copy=copy)
    except (OverflowError, TypeError, ValueError) as error:
        raise EmbeddingOutputError(
            "Embedding response is not a rectangular float matrix"
        ) from error
    if vectors.ndim != 2:
        raise EmbeddingOutputError("Embedding response must have two axes")
    if vectors.shape != (expected_rows, dimension):
        raise EmbeddingOutputError(
            f"Embedding response shape must be ({expected_rows}, {dimension})"
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingOutputError("Embedding response contains a non-finite value")
    norms = np.linalg.vector_norm(vectors.astype(np.float64), axis=1)
    if np.equal(norms, 0.0).any():
        raise EmbeddingOutputError("Embedding response contains a zero-norm row")
    vectors /= norms[:, np.newaxis]
    normalized_norms = np.linalg.vector_norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(normalized_norms, 1.0, rtol=1e-6, atol=1e-6):
        raise EmbeddingOutputError("Embedding response normalization failed")
    return vectors


class EmbeddingModel:
    """Create normalized text embeddings through one provider model."""

    def __init__(
        self,
        *,
        adapter: _EmbeddingAdapter,
        shared_backoff: SharedBackoff,
        max_requests: int,
    ) -> None:
        """Store one adapter, `SharedBackoff`, and `max_requests`.

        Args:
            adapter: The provider-specific embedding operations.
            shared_backoff: The request admission state.
            max_requests: The maximum number of requests sent for one batch; i.e., max_requests = max_retries + 1.
                `embed` splits its inputs into batches.

        Raises:
            ValueError: `max_requests` is boolean or below one.
        """
        if isinstance(max_requests, bool) or max_requests < 1:
            raise ValueError(f"max_requests must be a positive int, got {max_requests!r}")
        self.model: str = adapter.model
        self.dimension: int = adapter.dimension
        self.max_requests: int = max_requests
        self._adapter = adapter
        self._shared_backoff = shared_backoff

    async def _embed_batch_with_retries(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        """Run one request batch through its retry budget.

        A failed request retries when its `RequestFailure` is transient.

        Raises:
            asyncio.CancelledError: The caller cancelled this operation.
            Exception: A provider request failed terminally or used the last of `max_requests`.
        """
        private_backoff = PrivateBackoff(self._shared_backoff)
        request_index = 0
        while True:
            request_index += 1
            request_failure: RequestFailure | None = None
            try:
                async with self._shared_backoff.admitted() as admission:
                    try:
                        return await self._adapter.embed_batch(inputs, task=task)
                    except Exception as error:
                        request_failure = admission.record(
                            _request_failure_of(error, self._adapter.request_failure)
                        )
                        raise
            except Exception:
                # `request_failure` is `None` when the adapter's `request_failure` raised.
                if (
                    request_failure is None
                    or request_failure.kind != "transient"
                    or request_index == self.max_requests
                ):
                    raise
                if not request_failure.pauses_quota:
                    await asyncio.sleep(
                        private_backoff.next_wait(request_failure.retry_after_seconds)
                    )

    async def embed(
        self,
        inputs: SequenceNotStr[str],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        """Return one normalized row per input in input order.

        Empty input returns a writable `(0, dimension)` matrix without provider work.

        Args:
            inputs: The input texts.
            task: The embedding purpose to send to the provider.

        Raises:
            TypeError: `inputs` is a bare `str`.
            ValueError: An input cannot form a provider request.
            EmbeddingOutputError: A response contains invalid vectors.
            asyncio.CancelledError: The caller cancelled this operation.
            Exception: A provider request failed terminally or used the last of `max_requests`.
        """
        if isinstance(inputs, str):
            raise TypeError("inputs is a bare str; wrap one input in a list")
        input_snapshot = tuple(inputs)
        if not input_snapshot:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches = await self._adapter.partition_inputs(input_snapshot, task=task)
        await self._adapter.prepare()

        run_batches = tuple(
            partial(self._embed_batch_with_retries, batch, task=task) for batch in batches
        )
        matrices = await run_many(
            run_batches,
            max_pending=max_pending_for_requests(self._shared_backoff.max_concurrent_requests),
        )
        combined = np.concatenate(matrices, axis=0, dtype=np.float32)
        return _validated_embeddings(
            combined,
            expected_rows=len(input_snapshot),
            dimension=self.dimension,
            copy=False,
        )
