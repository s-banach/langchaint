"""Cover provider-neutral embedding execution and output invariants."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest

from langchaint import EmbeddingModel, EmbeddingOutputError, Float2D
from langchaint.adapter import ErrorClassification
from langchaint.embedding import EmbeddingTask, _validated_embeddings
from langchaint.sequence_not_str import SequenceNotStr
from langchaint.shared_backoff import DoNotRetry, RetryThisOne, SharedBackoff, Verdict


class _ProviderError(Exception):
    """Identify a parsed provider failure."""


class _TransportError(Exception):
    """Identify a transient transport failure."""


def _retry_provider_failure(_failure: Exception) -> Verdict:
    return RetryThisOne(retry_after=None)


def _reject_provider_failure(_failure: Exception) -> Verdict:
    return DoNotRetry()


class _StubEmbeddingAdapter:
    """Script partitioning and provider attempts for neutral tests."""

    model = "stub-embedding"
    dimension = 2
    failure_types: ClassVar[tuple[type[Exception], ...]] = (_ProviderError,)

    def __init__(self, attempts: Sequence[Float2D | Exception]) -> None:
        self._attempts = list(attempts)
        self.prepare_calls = 0
        self.partition_calls = 0
        self.embed_calls = 0
        self.partition_tasks: list[EmbeddingTask] = []
        self.embed_inputs: list[tuple[str, ...]] = []
        self.embed_tasks: list[EmbeddingTask] = []

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
        outcome = self._attempts.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def classify(self, error: Exception) -> ErrorClassification:
        if isinstance(error, _TransportError):
            return "transient"
        return "unknown_exception"


class _PartitioningEmbeddingAdapter:
    """Run one request batch per input under explicit completion controls."""

    model = "partitioning-embedding"
    dimension = 2
    failure_types: ClassVar[tuple[type[Exception], ...]] = (_ProviderError,)

    def __init__(self, inputs: SequenceNotStr[str], *, fail_once: set[str] | None = None) -> None:
        """Create one start and release event per input."""
        self.started = {input_text: asyncio.Event() for input_text in inputs}
        self.release = {input_text: asyncio.Event() for input_text in inputs}
        self.fail_once = set(fail_once or ())
        self.attempts: Counter[str] = Counter()

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
        self.attempts[input_text] += 1
        self.started[input_text].set()
        await self.release[input_text].wait()
        if input_text in self.fail_once and self.attempts[input_text] == 1:
            raise _ProviderError(input_text)
        row = [1.0, 0.0] if input_text == "first" else [0.0, 1.0]
        return np.array([row], dtype=np.float32)

    def classify(self, error: Exception) -> ErrorClassification:
        """Classify every unparsed exception as unknown."""
        del error
        return "unknown_exception"


def _shared_backoff(*, retry_provider_failures: bool = True) -> SharedBackoff:
    return SharedBackoff(
        parse=_retry_provider_failure if retry_provider_failures else _reject_provider_failure,
        failure_types=(_ProviderError,),
        max_concurrent_requests=2,
        max_request_starts_per_second=100_000.0,
        minimum_wait_ceiling_seconds=0.001,
        longest_wait_seconds=0.001,
        wait_multiplier=2.0,
        quiet_seconds_per_decay_step=60.0,
    )


def _model(
    adapter: _StubEmbeddingAdapter,
    *,
    max_attempts: int = 3,
    retry_provider_failures: bool = True,
) -> EmbeddingModel:
    return EmbeddingModel(
        adapter=adapter,
        shared_backoff=_shared_backoff(retry_provider_failures=retry_provider_failures),
        max_attempts=max_attempts,
    )


def test_embed_returns_normalized_owned_float32_rows() -> None:
    """Output preserves row order and owns writable normalized storage."""
    adapter = _StubEmbeddingAdapter([np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)])

    vectors = asyncio.run(_model(adapter).embed(["first", "second"], task="clustering"))

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

    vectors = asyncio.run(_model(adapter).embed([], task="retrieval_document"))

    assert vectors.shape == (0, 2)
    assert vectors.dtype == np.float32
    assert vectors.flags.owndata
    assert vectors.flags.writeable
    assert adapter.partition_calls == 0
    assert adapter.prepare_calls == 0
    assert adapter.embed_calls == 0


@pytest.mark.parametrize("max_attempts", [True, False, 0, -1])
def test_embedding_model_rejects_invalid_max_attempts(max_attempts: int) -> None:
    """Invalid retry budgets fail during model construction."""
    with pytest.raises(ValueError, match="max_attempts"):
        _ = _model(_StubEmbeddingAdapter([]), max_attempts=max_attempts)


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


@pytest.mark.parametrize(
    "failure",
    [_ProviderError("provider"), _TransportError("transport")],
)
def test_transient_failures_retry_and_return_vectors(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """Parsed and transport failures retry only their request batch."""
    adapter = _StubEmbeddingAdapter([
        failure,
        np.array([[1.0, 0.0]], dtype=np.float32),
    ])
    waits: list[float] = []

    async def record_sleep(wait_seconds: float) -> None:
        waits.append(wait_seconds)

    monkeypatch.setattr("langchaint.embedding.asyncio.sleep", record_sleep)

    vectors = asyncio.run(_model(adapter).embed(["one"], task="classification"))

    np.testing.assert_array_equal(vectors, [[1.0, 0.0]])
    assert adapter.embed_calls == 2
    assert len(waits) == 1


def test_terminal_provider_failure_propagates_unchanged() -> None:
    """A terminal provider failure receives no replacement exception."""
    failure = _ProviderError("provider text")
    adapter = _StubEmbeddingAdapter([failure])

    async def scenario() -> None:
        with pytest.raises(_ProviderError, match="provider text") as caught:
            _ = await _model(adapter, retry_provider_failures=False).embed(
                ["one"],
                task="classification",
            )
        assert caught.value is failure

    asyncio.run(scenario())


def test_exhausted_transport_failure_propagates_unchanged() -> None:
    """Retry exhaustion preserves the final transport exception."""
    first = _TransportError("first")
    final = _TransportError("final")
    adapter = _StubEmbeddingAdapter([first, final])

    async def scenario() -> None:
        with pytest.raises(_TransportError, match="final") as caught:
            _ = await _model(adapter, max_attempts=2).embed(
                ["one"],
                task="classification",
            )
        assert caught.value is final

    asyncio.run(scenario())


def test_request_batches_run_concurrently_and_preserve_input_order() -> None:
    """Concurrent completion order cannot change returned row order."""

    async def scenario() -> None:
        adapter = _PartitioningEmbeddingAdapter(["first", "second"])
        model = EmbeddingModel(
            adapter=adapter,
            shared_backoff=_shared_backoff(),
            max_attempts=1,
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

    asyncio.run(scenario())


def test_retrying_one_request_batch_does_not_repeat_its_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the failed request batch consumes another attempt."""

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
            max_attempts=2,
        )

        async def skip_sleep(_wait_seconds: float) -> None:
            """Skip the private retry wait."""

        monkeypatch.setattr("langchaint.embedding.asyncio.sleep", skip_sleep)
        vectors = await model.embed(["first", "second"], task="retrieval_document")
        np.testing.assert_array_equal(vectors, [[1.0, 0.0], [0.0, 1.0]])
        assert adapter.attempts == {"first": 2, "second": 1}

    asyncio.run(scenario())
