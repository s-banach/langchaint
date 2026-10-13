"""Provider-neutral embedding execution, accounting, and output validation.

`EmbeddingModel` copies inputs before partitioning them into provider requests.
`EmbeddingModel` retries each embedding batch independently.
All requests use its `SharedBackoff`.
The adapter owns provider batching, response decoding, and reading the reported token count.
The returned matrix preserves input order.
The returned matrix owns writable `float32` storage.
Every returned row has L2 norm one.
"""

import asyncio
import contextlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from operator import attrgetter
from typing import Literal, NamedTuple, Protocol

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    raise ModuleNotFoundError(
        "langchaint embeddings require the numpy package; install numpy."
    ) from exc

from langchaint.billing.pricing import category_cost_in_usd
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

type EmbeddingErrorKind = Literal[
    "auth_error",
    "rejected_error",
    "provider_failed_terminally_error",
    "unknown_exception_error",
    "retries_exhausted_error",
    "invalid_output_error",
]
"""How a request batch of `EmbeddingModel.embed` failed.

- `retries_exhausted_error`: a transient failure used the last of `max_requests`.
- `invalid_output_error`: a response had unusable vectors or reported no token count.
- Every other kind appends `_error` to the `RequestFailure` kind of a failure that is not transient.
"""


@dataclass(frozen=True, kw_only=True)
class EmbeddingUsage:
    """Provider-reported input tokens and their estimated cost.

    Embedding models price input tokens only.
    `cost_in_usd` is NaN when `input_tokens_total` is nonzero and the model has no rate.
    """

    input_tokens_total: int
    cost_in_usd: float


@dataclass(frozen=True, kw_only=True)
class EmbeddingRequestRecord:
    """One provider request that `EmbeddingModel.embed` sent.

    `input_count` is the number of inputs in the request.
    `started_after_seconds` runs from the start of `embed` until the request was sent, after admission.
    `usage` is `None` when the provider reported no token count, as when the request raised before a response arrived.
    `error_text` is `None` when the request returned usable vectors.
    """

    input_count: int
    started_after_seconds: float
    elapsed_seconds: float
    usage: EmbeddingUsage | None
    error_text: str | None


def _usage_of(request_history: Iterable[EmbeddingRequestRecord]) -> EmbeddingUsage:
    """Sum the usage of every request that reported one; no such request sums to zero."""
    usages = [record.usage for record in request_history if record.usage is not None]
    return EmbeddingUsage(
        input_tokens_total=sum(usage.input_tokens_total for usage in usages),
        cost_in_usd=sum(usage.cost_in_usd for usage in usages),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class Embeddings:
    """What `EmbeddingModel.embed` returns when every request batch succeeds.

    `output` has one normalized row per input, in input order.
    `request_history` holds every request sent, ordered by `started_after_seconds`.
    It includes failed requests that a retry replaced.
    """

    output: Float2D
    request_history: tuple[EmbeddingRequestRecord, ...]

    @property
    def usage(self) -> EmbeddingUsage:
        """Sum the usage of every request that reported one."""
        return _usage_of(self.request_history)


class EmbeddingError(Exception):
    """`EmbeddingModel.embed` failed after it started sending requests.

    `__cause__` is the exception of the request that ended the failed batch.
    When several batches fail, the batch holding the earliest inputs decides `kind` and `__cause__`.
    `request_history` holds every request sent, ordered by `started_after_seconds`.
    `usage` therefore keeps the billing of each request that returned.
    `str()` names `kind` before the exception text of that request.
    """

    kind: EmbeddingErrorKind
    request_history: tuple[EmbeddingRequestRecord, ...]

    def __init__(
        self,
        error_text: str,
        *,
        kind: EmbeddingErrorKind,
        request_history: tuple[EmbeddingRequestRecord, ...],
    ) -> None:
        """Store the failure kind and the requests sent."""
        super().__init__(error_text)
        self.kind = kind
        self.request_history = request_history

    @property
    def usage(self) -> EmbeddingUsage:
        """Sum the usage of every request that reported one."""
        return _usage_of(self.request_history)


class _EmbeddingBatchResponse(NamedTuple):
    """One request's validated vectors and the input token count the provider reported."""

    vectors: Float2D
    input_tokens_total: int


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
    ) -> _EmbeddingBatchResponse:
        """Return validated vectors and the reported token count for one request batch.

        Raises:
            EmbeddingOutputError: The response has unusable vectors or reports no token count.
            Exception: The provider request failed.
        """
        ...


def _validated_embeddings(
    values: Sequence[Sequence[float]],
    *,
    expected_rows: int,
    dimension: int,
    input_tokens_total: int | None,
) -> Float2D:
    """Convert provider values into a normalized owned matrix.

    Args:
        values: The provider vectors.
        expected_rows: The number of inputs the vectors embed.
        dimension: The required row length.
        input_tokens_total: The token count the provider reported for `values`, stored on a raised error.

    Raises:
        EmbeddingOutputError: Values violate the public output invariants.
    """
    try:
        vectors = np.array(values, dtype=np.float32, order="C")
    except (OverflowError, TypeError, ValueError) as error:
        raise EmbeddingOutputError(
            "Embedding response is not a rectangular float matrix",
            input_tokens_total=input_tokens_total,
        ) from error
    if vectors.ndim != 2:
        raise EmbeddingOutputError(
            "Embedding response must have two axes", input_tokens_total=input_tokens_total
        )
    if vectors.shape != (expected_rows, dimension):
        raise EmbeddingOutputError(
            f"Embedding response shape must be ({expected_rows}, {dimension})",
            input_tokens_total=input_tokens_total,
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingOutputError(
            "Embedding response contains a non-finite value",
            input_tokens_total=input_tokens_total,
        )
    norms = np.linalg.vector_norm(vectors.astype(np.float64), axis=1)
    if np.equal(norms, 0.0).any():
        raise EmbeddingOutputError(
            "Embedding response contains a zero-norm row", input_tokens_total=input_tokens_total
        )
    vectors /= norms[:, np.newaxis]
    normalized_norms = np.linalg.vector_norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(normalized_norms, 1.0, rtol=1e-6, atol=1e-6):
        raise EmbeddingOutputError(
            "Embedding response normalization failed", input_tokens_total=input_tokens_total
        )
    return vectors


class _RequestFailed(NamedTuple):
    """One request's exception and its `RequestFailure` after `Admission.record`."""

    error: Exception
    request_failure: RequestFailure


class _BatchFailure(NamedTuple):
    """How a request batch failed and the exception of its last request."""

    kind: EmbeddingErrorKind
    error: Exception


class _BatchResult(NamedTuple):
    """The requests one batch sent and how the batch ended.

    `ending` is `None` when a sibling batch failed before this batch sent its next request.
    """

    records: tuple[EmbeddingRequestRecord, ...]
    ending: Float2D | _BatchFailure | None


class EmbeddingModel:
    """Create normalized text embeddings through one provider model."""

    def __init__(
        self,
        *,
        adapter: _EmbeddingAdapter,
        shared_backoff: SharedBackoff,
        max_requests: int,
        usd_per_million_tokens: float,
    ) -> None:
        """Store one adapter, `SharedBackoff`, `max_requests`, and input-token rate.

        Args:
            adapter: The provider-specific embedding operations.
            shared_backoff: The request admission state.
            max_requests: The maximum number of requests sent for one batch; i.e., max_requests = max_retries + 1.
                `embed` splits its inputs into batches.
            usd_per_million_tokens: The input-token rate that prices `EmbeddingUsage`, or NaN when unknown.

        Raises:
            ValueError: `max_requests` is boolean or below one.
        """
        if isinstance(max_requests, bool) or max_requests < 1:
            raise ValueError(f"max_requests must be a positive int, got {max_requests!r}")
        self.model: str = adapter.model
        self.dimension: int = adapter.dimension
        self.max_requests: int = max_requests
        self.usd_per_million_tokens: float = usd_per_million_tokens
        self._adapter = adapter
        self._shared_backoff = shared_backoff

    def _request_record(
        self,
        *,
        input_count: int,
        embed_started_at: float,
        request_started_at: float,
        input_tokens_total: int | None,
        error_text: str | None,
    ) -> EmbeddingRequestRecord:
        """Record one request that settled now, pricing its reported token count."""
        return EmbeddingRequestRecord(
            input_count=input_count,
            started_after_seconds=request_started_at - embed_started_at,
            elapsed_seconds=time.monotonic() - request_started_at,
            usage=(
                None
                if input_tokens_total is None
                else EmbeddingUsage(
                    input_tokens_total=input_tokens_total,
                    cost_in_usd=category_cost_in_usd(
                        input_tokens_total, usd_per_million_tokens=self.usd_per_million_tokens
                    ),
                )
            ),
            error_text=error_text,
        )

    async def _send_request(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
        embed_started_at: float,
        stop: asyncio.Event,
    ) -> tuple[EmbeddingRequestRecord, Float2D | EmbeddingOutputError | _RequestFailed] | None:
        """Send one request after admission, or return `None` when `stop` is set before or at admission.

        An `EmbeddingOutputError` follows a response, so it bypasses the adapter's `request_failure`.

        Raises:
            asyncio.CancelledError: The caller cancelled this operation.
            Exception: The adapter's `request_failure` raised, which is an adapter defect.
        """
        if stop.is_set():
            return None
        async with self._shared_backoff.admitted() as admission:
            if stop.is_set():
                return None
            request_started_at = time.monotonic()
            try:
                response = await self._adapter.embed_batch(inputs, task=task)
            except EmbeddingOutputError as error:
                request_record = self._request_record(
                    input_count=len(inputs),
                    embed_started_at=embed_started_at,
                    request_started_at=request_started_at,
                    input_tokens_total=error.input_tokens_total,
                    error_text=str(error),
                )
                return request_record, error
            except Exception as error:  # noqa: BLE001 (every failure of a sent request becomes its record)
                request_failure = admission.record(
                    _request_failure_of(error, self._adapter.request_failure)
                )
                request_record = self._request_record(
                    input_count=len(inputs),
                    embed_started_at=embed_started_at,
                    request_started_at=request_started_at,
                    input_tokens_total=None,
                    error_text=str(error),
                )
                return request_record, _RequestFailed(error=error, request_failure=request_failure)
        request_record = self._request_record(
            input_count=len(inputs),
            embed_started_at=embed_started_at,
            request_started_at=request_started_at,
            input_tokens_total=response.input_tokens_total,
            error_text=None,
        )
        return request_record, response.vectors

    def _error_kind(
        self, request_failed: _RequestFailed, *, request_number: int
    ) -> EmbeddingErrorKind | None:
        """Return how the batch fails, or `None` when the failed request is sent again."""
        match request_failed.request_failure.kind:
            case "transient":
                return None if request_number < self.max_requests else "retries_exhausted_error"
            case "auth":
                return "auth_error"
            case "rejected":
                return "rejected_error"
            case "provider_failed_terminally":
                return "provider_failed_terminally_error"
            case "unknown_exception":
                return "unknown_exception_error"

    async def _embed_batch_with_retries(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
        embed_started_at: float,
        stop: asyncio.Event,
    ) -> _BatchResult:
        """Run one request batch through its retry budget.

        A failed request is sent again when its `RequestFailure` is transient and `max_requests` permits.
        The batch sends no request once `stop` is set, and sets `stop` when it fails.
        Setting `stop` also ends the wait before a retry.

        Raises:
            asyncio.CancelledError: The caller cancelled this operation.
            Exception: The adapter's `request_failure` raised, which is an adapter defect.
        """
        private_backoff = PrivateBackoff(self._shared_backoff)
        records: list[EmbeddingRequestRecord] = []
        request_number = 0
        while True:
            request_number += 1
            sent = await self._send_request(
                inputs, task=task, embed_started_at=embed_started_at, stop=stop
            )
            if sent is None:
                return _BatchResult(records=tuple(records), ending=None)
            request_record, ending = sent
            records.append(request_record)
            if isinstance(ending, EmbeddingOutputError):
                stop.set()
                return _BatchResult(
                    records=tuple(records),
                    ending=_BatchFailure(kind="invalid_output_error", error=ending),
                )
            if not isinstance(ending, _RequestFailed):
                return _BatchResult(records=tuple(records), ending=ending)
            kind = self._error_kind(ending, request_number=request_number)
            if kind is not None:
                stop.set()
                return _BatchResult(
                    records=tuple(records), ending=_BatchFailure(kind=kind, error=ending.error)
                )
            if not ending.request_failure.pauses_quota:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        stop.wait(),
                        private_backoff.next_wait_seconds(
                            ending.request_failure.retry_after_seconds
                        ),
                    )

    async def embed(
        self,
        inputs: SequenceNotStr[str],
        *,
        task: EmbeddingTask,
    ) -> Embeddings:
        """Return one normalized row per input in input order, with the requests that produced them.

        Empty input returns a writable `(0, dimension)` matrix without provider work.
        Once a request batch fails, no batch sends another request.
        Requests already sent finish, so `EmbeddingError.usage` includes their billing.

        Args:
            inputs: The input texts.
            task: The embedding purpose to send to the provider.

        Raises:
            TypeError: `inputs` is a bare `str`.
            ValueError: An input cannot form a provider request.
            EmbeddingError: A request batch failed.
            asyncio.CancelledError: The caller cancelled this operation.
            Exception: Adapter preparation failed, such as creating the Bedrock client.
                Also raised when the adapter's `request_failure` raises.
        """
        if isinstance(inputs, str):
            raise TypeError("inputs is a bare str; wrap one input in a list")
        embed_started_at = time.monotonic()
        input_snapshot = tuple(inputs)
        if not input_snapshot:
            return Embeddings(
                output=np.empty((0, self.dimension), dtype=np.float32), request_history=()
            )
        batches = await self._adapter.partition_inputs(input_snapshot, task=task)
        await self._adapter.prepare()

        stop = asyncio.Event()
        run_batches = tuple(
            partial(
                self._embed_batch_with_retries,
                batch,
                task=task,
                embed_started_at=embed_started_at,
                stop=stop,
            )
            for batch in batches
        )
        batch_results = await run_many(
            run_batches,
            max_pending=max_pending_for_requests(self._shared_backoff.max_concurrent_requests),
        )
        request_history = tuple(
            sorted(
                chain.from_iterable(batch_result.records for batch_result in batch_results),
                key=attrgetter("started_after_seconds"),
            )
        )
        endings = [batch_result.ending for batch_result in batch_results]
        failure = next((ending for ending in endings if isinstance(ending, _BatchFailure)), None)
        if failure is not None:
            raise EmbeddingError(
                f"Embedding failed with {failure.kind}: {failure.error}",
                kind=failure.kind,
                request_history=request_history,
            ) from failure.error
        output = np.concatenate(
            [ending for ending in endings if isinstance(ending, np.ndarray)], axis=0
        )
        return Embeddings(output=output, request_history=request_history)
