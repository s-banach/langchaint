"""Cover provider-neutral embedding execution and output invariants."""

import asyncio
import time
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from langchaint import EmbeddingModel, EmbeddingOutputError, Float2D
from langchaint.adapter import RequestFailure
from langchaint.common.sequence_not_str import SequenceNotStr
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.embedding import EmbeddingTask, _validated_embeddings
from tests.helpers import run_with_timeout


class _ProviderError(Exception):
    """Identify a transient provider failure."""


class _TransportError(Exception):
    """Identify a transient transport failure."""


def _place_provider_and_transport_errors(error: Exception) -> RequestFailure:
    """Place `_ProviderError` and `_TransportError` as transient, and every other exception as unknown_exception."""
    if isinstance(error, (_ProviderError, _TransportError)):
        return RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None)
    return RequestFailure(kind="unknown_exception", pauses_quota=False, retry_after_seconds=None)


class _StubEmbeddingAdapter:
    """Script partitioning and provider requests for neutral tests."""

    model = "stub-embedding"
    dimension = 2

    def __init__(
        self,
        requests: Sequence[Float2D | Exception],
        *,
        request_failure: Callable[
            [Exception], RequestFailure
        ] = _place_provider_and_transport_errors,
    ) -> None:
        self._requests = list(requests)
        self._request_failure = request_failure
        self.prepare_calls = 0
        self.partition_calls = 0
        self.embed_calls = 0
        self.partition_tasks: list[EmbeddingTask] = []
        self.embed_inputs: list[tuple[str, ...]] = []
        self.embed_tasks: list[EmbeddingTask] = []

    def request_failure(self, error: Exception) -> RequestFailure:
        return self._request_failure(error)

    async def prepare(self) -> None:
        self.prepare_calls += 1

    async def partition_inputs(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> tuple[tuple[str, ...], ...]:
        self.partition_calls += 1
        self.partition_tasks.append(task)
        return (tuple(inputs),)

    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        self.embed_calls += 1
        self.embed_inputs.append(tuple(inputs))
        self.embed_tasks.append(task)
        outcome = self._requests.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _PartitioningEmbeddingAdapter:
    """Run one request batch per input under explicit completion controls."""

    model = "partitioning-embedding"
    dimension = 2

    def __init__(self, inputs: SequenceNotStr[str], *, fail_once: set[str] | None = None) -> None:
        """Create one start and release event per input."""
        self.started = {input_text: asyncio.Event() for input_text in inputs}
        self.release = {input_text: asyncio.Event() for input_text in inputs}
        self.fail_once = set(fail_once or ())
        self.requests: Counter[str] = Counter()

    def request_failure(self, error: Exception) -> RequestFailure:
        """Retry every `_ProviderError` and `_TransportError`."""
        return _place_provider_and_transport_errors(error)

    async def prepare(self) -> None:
        """Complete preparation without work."""

    async def partition_inputs(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> tuple[tuple[str, ...], ...]:
        """Return one ordered request batch per input."""
        del task
        return tuple((input_text,) for input_text in inputs)

    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> Float2D:
        """Wait for release, then fail once or return the input's row."""
        del task
        input_text = inputs[0]
        self.requests[input_text] += 1
        self.started[input_text].set()
        await self.release[input_text].wait()
        if input_text in self.fail_once and self.requests[input_text] == 1:
            raise _ProviderError(input_text)
        row = [1.0, 0.0] if input_text == "first" else [0.0, 1.0]
        return np.array([row], dtype=np.float32)


def _shared_backoff(*, max_wait_seconds: float = 0.001) -> SharedBackoff:
    return SharedBackoff(
        max_concurrent_requests=2,
        max_request_starts_per_second=100_000.0,
        min_wait_ceiling_seconds=0.001,
        max_wait_seconds=max_wait_seconds,
    )


def _model(
    adapter: _StubEmbeddingAdapter,
    *,
    max_requests: int = 3,
    max_wait_seconds: float = 0.001,
) -> EmbeddingModel:
    return EmbeddingModel(
        adapter=adapter,
        shared_backoff=_shared_backoff(max_wait_seconds=max_wait_seconds),
        max_requests=max_requests,
    )


def test_embed_returns_normalized_owned_float32_rows() -> None:
    """Output preserves row order and owns writable normalized storage."""
    adapter = _StubEmbeddingAdapter([np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)])

    vectors = run_with_timeout(_model(adapter).embed(["first", "second"], task="clustering"))

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    assert vectors.flags.c_contiguous
    assert vectors.flags.owndata
    assert vectors.flags.writeable
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)
    assert adapter.partition_calls == 1
    assert adapter.prepare_calls == 1
    assert adapter.embed_calls == 1


def test_empty_input_returns_without_adapter_work() -> None:
    """Empty input returns the required empty matrix immediately."""
    adapter = _StubEmbeddingAdapter([])

    vectors = run_with_timeout(_model(adapter).embed([], task="retrieval_document"))

    assert vectors.shape == (0, 2)
    assert vectors.dtype == np.float32
    assert vectors.flags.owndata
    assert vectors.flags.writeable
    assert adapter.partition_calls == 0
    assert adapter.prepare_calls == 0
    assert adapter.embed_calls == 0


@pytest.mark.parametrize("max_requests", [True, False, 0, -1])
def test_embedding_model_rejects_invalid_max_requests(max_requests: int) -> None:
    """Invalid retry budgets fail during model construction."""
    with pytest.raises(ValueError, match="max_requests"):
        _ = _model(_StubEmbeddingAdapter([]), max_requests=max_requests)


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, 2.0]],
        [[1.0], [2.0]],
        [[1.0], [1.0, 2.0]],
        [[0.0, 0.0], [1.0, 2.0]],
        [[float("nan"), 1.0], [1.0, 2.0]],
        [[float("inf"), 1.0], [1.0, 2.0]],
    ],
)
def test_output_validation_rejects_invalid_matrices(
    values: Sequence[Sequence[float]],
) -> None:
    """Invalid provider matrices raise one output exception type."""
    with pytest.raises(EmbeddingOutputError):
        _ = _validated_embeddings(values, expected_rows=2, dimension=2)


def test_a_transient_failure_retries_and_returns_vectors() -> None:
    """A failure the adapter places as transient retries."""
    adapter = _StubEmbeddingAdapter([
        _ProviderError("provider"),
        np.array([[1.0, 0.0]], dtype=np.float32),
    ])

    vectors = run_with_timeout(_model(adapter).embed(["one"], task="classification"))

    np.testing.assert_array_equal(vectors, [[1.0, 0.0]])
    assert adapter.embed_calls == 2


def test_a_retry_after_sets_the_minimum_private_wait() -> None:
    """A transient failure's retry_after_seconds reaches `PrivateBackoff.next_wait_seconds` as the wait's minimum.

    The private ceiling starts at 0.001 seconds, and request starts are 0.00001 seconds apart.
    So only retry_after_seconds can make the retry wait 0.02 seconds.
    """
    adapter = _StubEmbeddingAdapter(
        [_ProviderError("slow"), np.array([[1.0, 0.0]], np.float32)],
        request_failure=lambda _error: RequestFailure(
            kind="transient", pauses_quota=False, retry_after_seconds=0.02
        ),
    )
    model = _model(adapter, max_wait_seconds=1.0)
    started_at = time.monotonic()
    _ = run_with_timeout(model.embed(["one"], task="classification"))
    assert time.monotonic() - started_at >= 0.02
    assert adapter.embed_calls == 2


def test_an_unknown_exception_fails_the_batch_without_a_retry() -> None:
    """A failure the adapter places as unknown_exception propagates unchanged after one request."""
    failure = KeyError("unplaced")
    adapter = _StubEmbeddingAdapter([failure, np.array([[1.0, 0.0]], dtype=np.float32)])
    model = _model(adapter)
    with pytest.raises(KeyError) as caught:
        _ = run_with_timeout(model.embed(["one"], task="classification"))
    assert caught.value is failure
    assert adapter.embed_calls == 1


def test_terminal_provider_failure_propagates_unchanged() -> None:
    """A terminal provider failure receives no replacement exception."""
    failure = _ProviderError("provider text")
    adapter = _StubEmbeddingAdapter(
        [failure],
        request_failure=lambda _error: RequestFailure(
            kind="rejected", pauses_quota=False, retry_after_seconds=None
        ),
    )

    async def scenario() -> None:
        with pytest.raises(_ProviderError, match="provider text") as caught:
            _ = await _model(adapter).embed(
                ["one"],
                task="classification",
            )
        assert caught.value is failure

    run_with_timeout(scenario())


def test_exhausted_transport_failure_propagates_unchanged() -> None:
    """Retry exhaustion preserves the final transport exception."""
    first = _TransportError("first")
    final = _TransportError("final")
    adapter = _StubEmbeddingAdapter([first, final])

    async def scenario() -> None:
        with pytest.raises(_TransportError, match="final") as caught:
            _ = await _model(adapter, max_requests=2).embed(
                ["one"],
                task="classification",
            )
        assert caught.value is final

    run_with_timeout(scenario())


def test_request_batches_run_concurrently_and_preserve_input_order() -> None:
    """Concurrent completion order cannot change returned row order."""

    async def scenario() -> None:
        adapter = _PartitioningEmbeddingAdapter(["first", "second"])
        model = EmbeddingModel(
            adapter=adapter,
            shared_backoff=_shared_backoff(),
            max_requests=1,
        )
        embed_task = asyncio.create_task(
            model.embed(["first", "second"], task="retrieval_document")
        )
        await asyncio.gather(adapter.started["first"].wait(), adapter.started["second"].wait())
        adapter.release["second"].set()
        await asyncio.sleep(0)
        assert not embed_task.done()
        adapter.release["first"].set()
        vectors = await embed_task
        np.testing.assert_array_equal(vectors, [[1.0, 0.0], [0.0, 1.0]])

    run_with_timeout(scenario())


def test_retrying_one_request_batch_does_not_repeat_its_sibling() -> None:
    """Only the failed request batch sends another request."""

    async def scenario() -> None:
        adapter = _PartitioningEmbeddingAdapter(
            ["first", "second"],
            fail_once={"first"},
        )
        adapter.release["first"].set()
        adapter.release["second"].set()
        model = EmbeddingModel(
            adapter=adapter,
            shared_backoff=_shared_backoff(),
            max_requests=2,
        )
        vectors = await model.embed(["first", "second"], task="retrieval_document")
        np.testing.assert_array_equal(vectors, [[1.0, 0.0], [0.0, 1.0]])
        assert adapter.requests == {"first": 2, "second": 1}

    run_with_timeout(scenario())
