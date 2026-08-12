"""OpenAI embedding catalog, batching, request, response, and cancellation tests."""

import asyncio
import json
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Literal, assert_type

import httpx2
import numpy as np
import pytest
import tiktoken
from openai import AsyncOpenAI

from langchaint import EmbeddingModel
from langchaint.embedding import EmbeddingTask
from langchaint.exceptions import EmbeddingOutputError
from langchaint.openai import (
    OPENAI_EMBEDDING_MODELS,
    OpenAI,
)
from langchaint.openai.embedding_adapter import (
    _OpenAIEmbeddingAdapter,
    _partition_inputs_sync,
)
from langchaint.shared_backoff import PrivateBackoff


def _client(handler: Callable[[httpx2.Request], httpx2.Response]) -> AsyncOpenAI:
    """Build an offline client whose requests reach `handler`."""
    return AsyncOpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _response(
    vectors: Sequence[Sequence[float]],
    *,
    indexes: Sequence[int] | None = None,
) -> httpx2.Response:
    """Build one successful SDK response from vectors and optional indexes."""
    if indexes is None:
        indexes = range(len(vectors))
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": vector,
                    "index": index,
                }
                for vector, index in zip(vectors, indexes, strict=True)
            ],
            "model": "offline",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


def _pin_embedding_model_overloads(openai: OpenAI) -> None:
    """Pin each overload's result type without running this function."""
    assert_type(openai.embedding_model("text-embedding-3-small"), EmbeddingModel)
    assert_type(openai.embedding_model("text-embedding-3-large"), EmbeddingModel)
    assert_type(openai.embedding_model("text-embedding-ada-002"), EmbeddingModel)


def test_embedding_catalog_has_the_documented_models() -> None:
    """Expose the three cataloged OpenAI embedding identifiers."""
    assert {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    } == OPENAI_EMBEDDING_MODELS


@pytest.mark.parametrize(
    ("model", "dimension"),
    [
        ("text-embedding-3-small", 1),
        ("text-embedding-3-small", 1536),
        ("text-embedding-3-large", 1),
        ("text-embedding-3-large", 3072),
    ],
)
def test_third_generation_models_accept_dimension_boundaries(
    model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    dimension: int,
) -> None:
    """Accept each third-generation model's inclusive dimension boundaries."""
    client = _client(lambda _request: _response([[1.0]]))
    openai = OpenAI(client=client)
    embedding_model = openai.embedding_model(model, dimension=dimension)
    assert embedding_model.dimension == dimension
    asyncio.run(client.close())


@pytest.mark.parametrize(
    ("model", "dimension"),
    [
        ("text-embedding-3-small", 0),
        ("text-embedding-3-small", 1537),
        ("text-embedding-3-large", 0),
        ("text-embedding-3-large", 3073),
        ("text-embedding-3-small", True),
        ("text-embedding-3-large", False),
    ],
)
def test_third_generation_models_reject_invalid_dimensions(
    model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    dimension: int,
) -> None:
    """Reject out-of-range dimensions and boolean dimensions."""
    client = _client(lambda _request: _response([[1.0]]))
    openai = OpenAI(client=client)
    with pytest.raises(ValueError, match="dimension"):
        _ = openai.embedding_model(model, dimension=dimension)
    asyncio.run(client.close())


def test_embedding_model_defaults_and_ada_dimension() -> None:
    """Resolve each model's documented default dimension during construction."""
    client = _client(lambda _request: _response([[1.0]]))
    openai = OpenAI(client=client)
    assert openai.embedding_model("text-embedding-3-small").dimension == 1536
    assert openai.embedding_model("text-embedding-3-large").dimension == 3072
    assert openai.embedding_model("text-embedding-ada-002").dimension == 1536
    asyncio.run(client.close())


def test_embedding_model_performs_no_tokenizer_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct an embedding model without loading `cl100k_base`."""
    client = _client(lambda _request: _response([[1.0]]))
    openai = OpenAI(client=client)

    def reject_loading(_name: str) -> tiktoken.Encoding:
        raise AssertionError("embedding_model loaded a tokenizer")

    monkeypatch.setattr(tiktoken, "get_encoding", reject_loading)
    model = openai.embedding_model("text-embedding-3-small")
    assert model.dimension == 1536
    asyncio.run(client.close())


def test_embedding_model_names_missing_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name tiktoken when the optional tokenizer dependency is unavailable."""
    client = _client(lambda _request: _response([[1.0]]))
    openai = OpenAI(client=client)
    monkeypatch.delitem(sys.modules, "langchaint.openai.embedding_adapter")
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ModuleNotFoundError, match="require the tiktoken package"):
        _ = openai.embedding_model("text-embedding-3-small")
    asyncio.run(client.close())


@pytest.mark.parametrize(
    "task",
    ["retrieval_document", "retrieval_query", "classification", "clustering"],
)
def test_request_maps_inputs_model_dimension_and_encoding(task: EmbeddingTask) -> None:
    """Send OpenAI fields without sending provider-neutral `task`."""
    request_bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request_bodies.append(json.loads(request.content))
        return _response([[3.0, 4.0], [0.0, 2.0]], indexes=[1, 0])

    client = _client(handler)

    async def scenario() -> None:
        adapter = _OpenAIEmbeddingAdapter(
            client=client,
            model="text-embedding-3-small",
            dimension=2,
        )
        vectors = await adapter.embed_batch(("first", "second"), task=task)
        np.testing.assert_allclose(vectors, [[0.0, 1.0], [0.6, 0.8]])
        await client.close()

    asyncio.run(scenario())
    assert request_bodies == [
        {
            "input": ["first", "second"],
            "model": "text-embedding-3-small",
            "dimensions": 2,
            "encoding_format": "float",
        }
    ]


def test_ada_request_omits_dimensions() -> None:
    """Omit `dimensions` for `text-embedding-ada-002`."""
    request_bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request_bodies.append(json.loads(request.content))
        return _response([[1.0] * 1536])

    client = _client(handler)

    async def scenario() -> None:
        model = OpenAI(client=client).embedding_model("text-embedding-ada-002")
        _ = await model.embed(["text"], task="classification")
        await client.close()

    asyncio.run(scenario())
    assert "dimensions" not in request_bodies[0]
    assert request_bodies[0]["encoding_format"] == "float"


def test_response_indexes_restore_order_and_output_storage_invariants() -> None:
    """Restore indexed rows into writable C-contiguous owned float32 storage."""
    client = _client(lambda _request: _response([[3.0, 4.0], [0.0, 2.0]], indexes=[1, 0]))

    async def scenario() -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        adapter = _OpenAIEmbeddingAdapter(
            client=client,
            model="text-embedding-3-small",
            dimension=2,
        )
        vectors = await adapter.embed_batch(("first", "second"), task="clustering")
        await client.close()
        return vectors

    vectors = asyncio.run(scenario())
    np.testing.assert_allclose(vectors, [[0.0, 1.0], [0.6, 0.8]])
    assert vectors.dtype == np.float32
    assert vectors.flags.c_contiguous
    assert vectors.flags.writeable
    assert vectors.flags.owndata


@pytest.mark.parametrize(
    "indexes",
    [
        [0],
        [0, 0],
        [0, 2],
        [-1, 1],
    ],
)
def test_response_rejects_invalid_indexes(indexes: list[int]) -> None:
    """Reject missing, repeated, and out-of-range response indexes."""
    vectors = [[1.0], [2.0]][: len(indexes)]
    client = _client(lambda _request: _response(vectors, indexes=indexes))

    async def scenario() -> None:
        adapter = _OpenAIEmbeddingAdapter(
            client=client,
            model="text-embedding-3-small",
            dimension=1,
        )
        with pytest.raises(EmbeddingOutputError, match="indexes"):
            _ = await adapter.embed_batch(("first", "second"), task="clustering")
        await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "data",
    [None, [None], [{"object": "embedding", "embedding": [1.0], "index": None}]],
)
def test_response_rejects_invalid_data_shapes(data: object) -> None:
    """Malformed successful response data raises `EmbeddingOutputError`."""

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": "offline",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = _client(handler)

    async def scenario() -> None:
        adapter = _OpenAIEmbeddingAdapter(
            client=client,
            model="text-embedding-3-small",
            dimension=1,
        )
        with pytest.raises(EmbeddingOutputError):
            _ = await adapter.embed_batch(("text",), task="clustering")
        await client.close()

    asyncio.run(scenario())


def test_partition_splits_at_input_count_limit() -> None:
    """Fill 2048 inputs before placing the remainder in another batch."""
    batches = _partition_inputs_sync(["x"] * 2049)
    assert tuple(map(len, batches)) == (2048, 1)


@pytest.mark.parametrize("model", list(OPENAI_EMBEDDING_MODELS))
def test_partition_uses_cl100k_base_for_each_model(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count each cataloged model with `cl100k_base`."""
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding_names: list[str] = []

    def recording_encoding(name: str) -> tiktoken.Encoding:
        encoding_names.append(name)
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", recording_encoding)
    client = _client(lambda _request: _response([[1.0]]))

    async def scenario() -> None:
        adapter = _OpenAIEmbeddingAdapter(client=client, model=model, dimension=1)
        batches = await adapter.partition_inputs(("text",), task="classification")
        assert batches == (("text",),)
        await client.close()

    asyncio.run(scenario())
    assert encoding_names == ["cl100k_base"]


class _CountEncoding:
    """Return the decimal input as its token count."""

    def encode(self, text: str, *, disallowed_special: tuple[()] = ()) -> range:
        """Return a sized range representing `text` tokens."""
        del disallowed_special
        return range(int(text))


def test_partition_splits_before_request_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill 300,000 tokens before placing the remainder in another batch."""

    def count_encoding(_name: str) -> _CountEncoding:
        return _CountEncoding()

    monkeypatch.setattr(tiktoken, "get_encoding", count_encoding)
    batches = _partition_inputs_sync(["299999", "1", "2"])
    assert batches == (("299999", "1"), ("2",))


def test_partition_keeps_oversized_input_for_provider_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one oversized input intact for OpenAI's per-input validation."""

    def count_encoding(_name: str) -> _CountEncoding:
        return _CountEncoding()

    monkeypatch.setattr(tiktoken, "get_encoding", count_encoding)
    batches = _partition_inputs_sync(["300001", "1"])
    assert batches == (("300001",), ("1",))


def test_empty_input_string_fails_before_sdk_request() -> None:
    """Reject an empty string before request admission and SDK execution."""
    request_count = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        return _response([[1.0]])

    client = _client(handler)

    async def scenario() -> None:
        model = OpenAI(client=client).embedding_model(
            "text-embedding-3-small",
            dimension=1,
        )
        with pytest.raises(ValueError, match="empty strings"):
            _ = await model.embed(["valid", ""], task="retrieval_document")
        await client.close()

    asyncio.run(scenario())
    assert request_count == 0


@pytest.mark.parametrize("failure_type", [httpx2.ConnectError, httpx2.ReadTimeout])
def test_transport_failure_retries_the_failed_batch(
    failure_type: type[httpx2.ConnectError] | type[httpx2.ReadTimeout],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry an OpenAI transport failure through the private batch backoff."""
    request_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            raise failure_type("offline transport failure", request=request)
        return _response([[1.0]])

    def zero_wait(_backoff: PrivateBackoff, _retry_after: float | None) -> float:
        return 0.0

    monkeypatch.setattr(PrivateBackoff, "next_wait", zero_wait)
    client = _client(handler)

    async def scenario() -> None:
        model = OpenAI(client=client).embedding_model(
            "text-embedding-3-small",
            dimension=1,
            max_attempts=2,
        )
        vectors = await model.embed(["text"], task="retrieval_query")
        assert vectors.tolist() == [[1.0]]
        await client.close()

    asyncio.run(scenario())
    assert request_count == 2


class _BlockingEncoding:
    """Block token counting until a test permits completion."""

    def __init__(self, started: threading.Event, permit_completion: threading.Event) -> None:
        """Store the synchronization events."""
        self._started = started
        self._permit_completion = permit_completion

    def encode(self, _text: str, *, disallowed_special: tuple[()] = ()) -> range:
        """Wait for permission before returning one token."""
        del disallowed_special
        self._started.set()
        if not self._permit_completion.wait(timeout=5.0):
            raise AssertionError("test did not permit token counting completion")
        return range(1)


def test_partition_cancellation_waits_for_token_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settle synchronous token counting before propagating cancellation."""
    started = threading.Event()
    permit_completion = threading.Event()
    encoding = _BlockingEncoding(started, permit_completion)

    def blocking_encoding(_name: str) -> _BlockingEncoding:
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", blocking_encoding)
    client = _client(lambda _request: _response([[1.0]]))

    async def scenario() -> None:
        model = OpenAI(client=client).embedding_model(
            "text-embedding-3-small",
            dimension=1,
        )
        embedding_task = asyncio.create_task(model.embed(["text"], task="retrieval_document"))
        assert await asyncio.to_thread(started.wait, 1.0)
        _ = embedding_task.cancel()
        await asyncio.sleep(0)
        assert not embedding_task.done()
        permit_completion.set()
        with pytest.raises(asyncio.CancelledError):
            _ = await embedding_task
        await client.close()

    asyncio.run(scenario())
