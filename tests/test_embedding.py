"""Cover provider-neutral embedding execution, accounting, and output invariants."""

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from langchaint import (
    EmbeddingError,
    EmbeddingModel,
    EmbeddingRequestErrorRecord,
    EmbeddingUsage,
    Float2D,
)
from langchaint.adapter import RequestFailure
from langchaint.common.exceptions import EmbeddingOutputError
from langchaint.common.sequence_not_str import SequenceNotStr
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.embedding import EmbeddingTask, _EmbeddingBatchResponse, _validated_embeddings
from langchaint.embedding_requests import EmbeddingObserver
from tests.helpers import RaisingObserver, run_with_timeout

_TOKENS_PER_REQUEST = 7
_USD_PER_MILLION_TOKENS = 2.0
_COST_PER_REQUEST = _TOKENS_PER_REQUEST * _USD_PER_MILLION_TOKENS / 1_000_000


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
    """Script partitioning and provider requests for neutral tests.

    Each scripted matrix returns with `_TOKENS_PER_REQUEST` reported tokens.
    """

    provider_name = "stub-provider"
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
        del task
        self.partition_calls += 1
        return (tuple(inputs),)

    async def embed_batch(
        self,
        inputs: tuple[str, ...],
        *,
        task: EmbeddingTask,
    ) -> _EmbeddingBatchResponse:
        del inputs, task
        self.embed_calls += 1
        outcome = self._requests.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _EmbeddingBatchResponse(
            vectors=outcome, input_tokens_total=_TOKENS_PER_REQUEST, model_served="stub-served"
        )


class _PartitioningEmbeddingAdapter:
    """Run one request batch per input under explicit completion controls."""

    provider_name = "stub-provider"
    model = "partitioning-embedding"
    dimension = 2

    def __init__(
        self,
        inputs: SequenceNotStr[str],
        *,
        fail_once: frozenset[str] = frozenset(),
        fail_terminally: frozenset[str] = frozenset(),
        retry_after_seconds: float | None = None,
    ) -> None:
        """Create one start and release event per input."""
        self.started = {input_text: asyncio.Event() for input_text in inputs}
        self.release = {input_text: asyncio.Event() for input_text in inputs}
        self.fail_once = fail_once
        self.fail_terminally = fail_terminally
        self.retry_after_seconds = retry_after_seconds
        self.requests: Counter[str] = Counter()

    def request_failure(self, error: Exception) -> RequestFailure:
        """Retry every `_ProviderError` and `_TransportError` after `retry_after_seconds`."""
        request_failure = _place_provider_and_transport_errors(error)
        return RequestFailure(
            kind=request_failure.kind,
            pauses_quota=False,
            retry_after_seconds=self.retry_after_seconds,
        )

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
    ) -> _EmbeddingBatchResponse:
        """Wait for release, then fail as scripted or return the input's row."""
        del task
        input_text = inputs[0]
        self.requests[input_text] += 1
        self.started[input_text].set()
        await self.release[input_text].wait()
        if input_text in self.fail_terminally:
            raise KeyError(input_text)
        if input_text in self.fail_once and self.requests[input_text] == 1:
            raise _ProviderError(input_text)
        row = [1.0, 0.0] if input_text == "first" else [0.0, 1.0]
        return _EmbeddingBatchResponse(
            vectors=np.array([row], dtype=np.float32),
            input_tokens_total=_TOKENS_PER_REQUEST,
            model_served=None,
        )


def _shared_backoff(*, max_wait_seconds: float = 0.001) -> SharedBackoff:
    return SharedBackoff(
        max_concurrent_requests=2,
        max_request_starts_per_second=100_000.0,
        min_wait_ceiling_seconds=0.001,
        max_wait_seconds=max_wait_seconds,
    )


def _model(
    adapter: _StubEmbeddingAdapter | _PartitioningEmbeddingAdapter,
    *,
    max_requests: int = 3,
    max_wait_seconds: float = 0.001,
    observer: EmbeddingObserver | None = None,
) -> EmbeddingModel:
    return EmbeddingModel(
        adapter=adapter,
        shared_backoff=_shared_backoff(max_wait_seconds=max_wait_seconds),
        max_requests=max_requests,
        usd_per_million_tokens=_USD_PER_MILLION_TOKENS,
        observer=observer,
    )


def _embedding_error(model: EmbeddingModel, inputs: list[str]) -> EmbeddingError:
    """Return the `EmbeddingError` that embedding `inputs` raises."""
    with pytest.raises(EmbeddingError) as caught:
        _ = run_with_timeout(model.embed(inputs, task="classification"))
    return caught.value


def test_embed_returns_owned_float32_rows_and_priced_usage() -> None:
    """Output preserves row order and owns writable storage, and its request is priced."""
    adapter = _StubEmbeddingAdapter([np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)])

    embeddings = run_with_timeout(_model(adapter).embed(["first", "second"], task="clustering"))

    vectors = embeddings.output
    assert vectors.dtype == np.float32
    assert vectors.flags.c_contiguous
    assert vectors.flags.owndata
    assert vectors.flags.writeable
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)
    [request_record] = embeddings.request_history
    assert request_record.input_count == 2
    assert request_record.model_served == "stub-served"
    assert request_record.error is None
    assert embeddings.usage == EmbeddingUsage(
        input_tokens_total=_TOKENS_PER_REQUEST, cost_in_usd=_COST_PER_REQUEST
    )
    assert (adapter.partition_calls, adapter.prepare_calls, adapter.embed_calls) == (1, 1, 1)


def test_empty_input_returns_without_adapter_work() -> None:
    """Empty input returns the required empty matrix and zero usage immediately."""
    adapter = _StubEmbeddingAdapter([])

    embeddings = run_with_timeout(_model(adapter).embed([], task="retrieval_document"))

    assert embeddings.output.shape == (0, 2)
    assert embeddings.output.dtype == np.float32
    assert embeddings.output.flags.owndata
    assert embeddings.output.flags.writeable
    assert embeddings.request_history == ()
    assert embeddings.usage == EmbeddingUsage(input_tokens_total=0, cost_in_usd=0.0)
    assert (adapter.partition_calls, adapter.prepare_calls, adapter.embed_calls) == (0, 0, 0)


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
    """Invalid provider matrices raise one output exception type that keeps the reported token count."""
    with pytest.raises(EmbeddingOutputError) as caught:
        _ = _validated_embeddings(values, expected_rows=2, dimension=2, input_tokens_total=5)
    assert caught.value.input_tokens_total == 5


def test_a_transient_failure_retries_and_records_both_requests() -> None:
    """A failure the adapter places as transient retries, and the failed request stays in the history."""
    adapter = _StubEmbeddingAdapter([
        _ProviderError("provider"),
        np.array([[1.0, 0.0]], dtype=np.float32),
    ])

    embeddings = run_with_timeout(_model(adapter).embed(["one"], task="classification"))

    np.testing.assert_array_equal(embeddings.output, [[1.0, 0.0]])
    failed, succeeded = embeddings.request_history
    assert (failed.error, failed.usage) == (
        EmbeddingRequestErrorRecord(kind="transient", error_text="provider"),
        None,
    )
    assert failed.model_served is None
    assert succeeded.error is None
    assert embeddings.usage.input_tokens_total == _TOKENS_PER_REQUEST


@pytest.mark.parametrize("raise_at_start", [True, False], ids=["start", "operation"])
def test_an_observer_that_raises_never_changes_what_embed_returns(
    *, raise_at_start: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """langchaint logs every observer failure, and `embed` retries and returns as it does unobserved."""
    adapter = _StubEmbeddingAdapter([
        _ProviderError("provider"),
        np.array([[1.0, 0.0]], dtype=np.float32),
    ])
    model = _model(adapter, observer=RaisingObserver(raise_at_start=raise_at_start))

    with caplog.at_level(logging.WARNING, logger="langchaint.common.observed_operation"):
        embeddings = run_with_timeout(model.embed(["one"], task="classification"))

    np.testing.assert_array_equal(embeddings.output, [[1.0, 0.0]])
    assert [record.error for record in embeddings.request_history] == [
        EmbeddingRequestErrorRecord(kind="transient", error_text="provider"),
        None,
    ]
    assert "the observer raised" in caplog.text


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


@pytest.mark.parametrize(
    ("request_failure", "error_kind"),
    [
        (RequestFailure(kind="auth", pauses_quota=False, retry_after_seconds=None), "auth_error"),
        (
            RequestFailure(kind="rejected", pauses_quota=False, retry_after_seconds=None),
            "rejected_error",
        ),
        (
            RequestFailure(
                kind="provider_failed_terminally", pauses_quota=False, retry_after_seconds=None
            ),
            "provider_failed_terminally_error",
        ),
        (
            RequestFailure(kind="unknown_exception", pauses_quota=False, retry_after_seconds=None),
            "unknown_exception_error",
        ),
    ],
)
def test_a_terminal_failure_raises_its_kind_without_a_retry(
    request_failure: RequestFailure, error_kind: str
) -> None:
    """A failure that is not transient ends `embed` after one request, with the provider exception as the cause."""
    failure = _ProviderError("provider text")
    adapter = _StubEmbeddingAdapter([failure], request_failure=lambda _error: request_failure)

    error = _embedding_error(_model(adapter), ["one"])

    assert error.kind == error_kind
    assert error.__cause__ is failure
    assert str(error) == f"Embedding failed with {error_kind}: provider text"
    [request_record] = error.request_history
    assert request_record.error == EmbeddingRequestErrorRecord(
        kind=request_failure.kind, error_text="provider text"
    )
    assert adapter.embed_calls == 1


def test_exhausted_retries_keep_the_final_exception() -> None:
    """Retry exhaustion raises `retries_exhausted_error` caused by the final exception."""
    final = _TransportError("final")
    adapter = _StubEmbeddingAdapter([_TransportError("first"), final])

    error = _embedding_error(_model(adapter, max_requests=2), ["one"])

    assert error.kind == "retries_exhausted_error"
    assert error.__cause__ is final
    assert len(error.request_history) == 2


def test_invalid_output_keeps_the_billed_token_count() -> None:
    """A response with unusable vectors ends `embed` without a retry, and its reported tokens stay billed."""
    failure = EmbeddingOutputError("bad vectors", input_tokens_total=_TOKENS_PER_REQUEST)
    adapter = _StubEmbeddingAdapter([failure, np.array([[1.0, 0.0]], dtype=np.float32)])

    error = _embedding_error(_model(adapter), ["one"])

    assert error.kind == "invalid_output_error"
    assert error.__cause__ is failure
    assert error.usage == EmbeddingUsage(
        input_tokens_total=_TOKENS_PER_REQUEST, cost_in_usd=_COST_PER_REQUEST
    )
    assert adapter.embed_calls == 1


def test_a_failed_batch_lets_sent_requests_finish_and_starts_no_more() -> None:
    """After one batch fails, a sibling already sent keeps its billing, and an unsent batch never starts.

    The `SharedBackoff` admits two requests at once, so the third batch waits for a permit.
    """

    async def scenario() -> EmbeddingError:
        adapter = _PartitioningEmbeddingAdapter(
            ["first", "second", "third"], fail_terminally=frozenset({"first"})
        )
        embed_task = asyncio.create_task(
            _model(adapter, max_requests=1).embed(
                ["first", "second", "third"], task="retrieval_document"
            )
        )
        await asyncio.gather(adapter.started["first"].wait(), adapter.started["second"].wait())
        adapter.release["first"].set()
        await asyncio.sleep(0.01)
        assert not embed_task.done()
        adapter.release["second"].set()
        with pytest.raises(EmbeddingError) as caught:
            await embed_task
        assert adapter.requests == {"first": 1, "second": 1}
        return caught.value

    error = run_with_timeout(scenario())

    assert error.kind == "unknown_exception_error"
    assert [record.input_count for record in error.request_history] == [1, 1]
    assert error.usage.input_tokens_total == _TOKENS_PER_REQUEST


def test_a_failed_batch_ends_a_sibling_retry_wait() -> None:
    """A batch waiting to retry stops waiting when a sibling fails, so `embed` raises without the full wait.

    The 60-second retry wait exceeds the test timeout, so the test fails if the wait runs out.
    """

    async def scenario() -> EmbeddingError:
        adapter = _PartitioningEmbeddingAdapter(
            ["first", "second"],
            fail_once=frozenset({"second"}),
            fail_terminally=frozenset({"first"}),
            retry_after_seconds=60.0,
        )
        embed_task = asyncio.create_task(
            _model(adapter, max_wait_seconds=60.0).embed(
                ["first", "second"], task="retrieval_document"
            )
        )
        await asyncio.gather(adapter.started["first"].wait(), adapter.started["second"].wait())
        adapter.release["second"].set()
        await asyncio.sleep(0.01)
        adapter.release["first"].set()
        with pytest.raises(EmbeddingError) as caught:
            await embed_task
        assert adapter.requests == {"first": 1, "second": 1}
        return caught.value

    error = run_with_timeout(scenario())

    assert error.kind == "unknown_exception_error"


def test_request_batches_run_concurrently_and_preserve_input_order() -> None:
    """Concurrent completion order cannot change returned row order."""

    async def scenario() -> None:
        adapter = _PartitioningEmbeddingAdapter(["first", "second"])
        embed_task = asyncio.create_task(
            _model(adapter, max_requests=1).embed(["first", "second"], task="retrieval_document")
        )
        await asyncio.gather(adapter.started["first"].wait(), adapter.started["second"].wait())
        adapter.release["second"].set()
        await asyncio.sleep(0)
        assert not embed_task.done()
        adapter.release["first"].set()
        embeddings = await embed_task
        np.testing.assert_array_equal(embeddings.output, [[1.0, 0.0], [0.0, 1.0]])
        assert embeddings.usage.input_tokens_total == 2 * _TOKENS_PER_REQUEST

    run_with_timeout(scenario())


def test_retrying_one_request_batch_does_not_repeat_its_sibling() -> None:
    """Only the failed request batch sends another request."""

    async def scenario() -> None:
        adapter = _PartitioningEmbeddingAdapter(
            ["first", "second"], fail_once=frozenset({"first"})
        )
        adapter.release["first"].set()
        adapter.release["second"].set()
        embeddings = await _model(adapter, max_requests=2).embed(
            ["first", "second"], task="retrieval_document"
        )
        np.testing.assert_array_equal(embeddings.output, [[1.0, 0.0], [0.0, 1.0]])
        assert adapter.requests == {"first": 2, "second": 1}
        assert len(embeddings.request_history) == 3

    run_with_timeout(scenario())
