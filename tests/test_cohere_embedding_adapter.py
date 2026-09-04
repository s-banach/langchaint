"""Tests for Cohere embeddings through Amazon Bedrock."""

from __future__ import annotations

import asyncio
import json
import threading
from functools import wraps
from io import BytesIO
from typing import TYPE_CHECKING

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import Stubber

import langchaint.cohere as cohere_backend
from langchaint.cohere import COHERE_BEDROCK_EMBEDDING_MODELS, CohereBedrock
from langchaint.exceptions import EmbeddingOutputError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from langchaint.cohere import CohereEmbedV4Dimension, CohereEmbedV4ModelName


_JSON_CONTENT_TYPE = "application/json"


def _run_async_test[**Parameters](
    function: Callable[Parameters, Coroutine[object, object, None]],
) -> Callable[Parameters, None]:
    """Run one asynchronous test through `asyncio.run`."""

    @wraps(function)
    def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> None:
        # Separate wrappers per pytest signature duplicate event-loop orchestration.
        asyncio.run(function(*args, **kwargs))

    return wrapped


def _bedrock_client(*, total_max_attempts: int = 1) -> BedrockRuntimeClient:
    """Build an offline Bedrock Runtime client."""
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"total_max_attempts": total_max_attempts}),
    )


def _request_body(payload: dict[str, object]) -> bytes:
    """Serialize one expected Cohere request."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _response(vectors: list[list[float]]) -> dict[str, object]:
    """Build one constructed `invoke_model` response."""
    body_bytes = json.dumps({"embeddings": vectors}).encode()
    return {
        "body": StreamingBody(BytesIO(body_bytes), len(body_bytes)),
        "contentType": _JSON_CONTENT_TYPE,
    }


def _expected_parameters(body: bytes, model: str) -> dict[str, object]:
    """Build expected `invoke_model` parameters."""
    return {
        "body": body,
        "modelId": model,
        "accept": _JSON_CONTENT_TYPE,
        "contentType": _JSON_CONTENT_TYPE,
    }


@pytest.mark.parametrize(
    ("model", "dimension"),
    [
        ("cohere.embed-v4:0", 256),
        ("global.cohere.embed-v4:0", 512),
        ("us.cohere.embed-v4:0", 1024),
        ("eu.cohere.embed-v4:0", 1536),
    ],
)
@_run_async_test
async def test_v4_routes_model_and_dimension_verbatim(
    model: CohereEmbedV4ModelName,
    dimension: CohereEmbedV4Dimension,
) -> None:
    """Every v4 route sends its model and dimension verbatim."""
    client = _bedrock_client()
    stubber = Stubber(client)
    body = _request_body({
        "texts": ["document"],
        "input_type": "search_document",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": dimension,
    })
    vector = [1.0, *([0.0] * (dimension - 1))]
    stubber.add_response(
        "invoke_model",
        _response([vector]),
        _expected_parameters(body, model),
    )
    stubber.activate()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        embeddings = await cohere_bedrock.embedding_model(
            model,
            dimension=dimension,
            max_attempts=1,
        ).embed(["document"], task="retrieval_document")
        assert embeddings.shape == (1, dimension)
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
        client.close()


@pytest.mark.parametrize(
    ("task", "input_type"),
    [
        ("retrieval_document", "search_document"),
        ("retrieval_query", "search_query"),
        ("classification", "classification"),
        ("clustering", "clustering"),
    ],
)
@_run_async_test
async def test_task_mapping(
    task: cohere_backend.EmbeddingTask,
    input_type: str,
) -> None:
    """Each neutral task maps to Cohere's corresponding `input_type`."""
    client = _bedrock_client()
    stubber = Stubber(client)
    body = _request_body({
        "texts": ["text"],
        "input_type": input_type,
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": 256,
    })
    stubber.add_response(
        "invoke_model",
        _response([[1.0, *([0.0] * 255)]]),
        _expected_parameters(body, "cohere.embed-v4:0"),
    )
    stubber.activate()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
        await model.embed(["text"], task=task)
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
        client.close()


@_run_async_test
async def test_v3_omits_dimension() -> None:
    """V3 requests omit `output_dimension`."""
    client = _bedrock_client()
    stubber = Stubber(client)
    body = _request_body({
        "texts": ["query"],
        "input_type": "search_query",
        "embedding_types": ["float"],
        "truncate": "NONE",
    })
    stubber.add_response(
        "invoke_model",
        _response([[1.0, *([0.0] * 1023)]]),
        _expected_parameters(body, "cohere.embed-multilingual-v3"),
    )
    stubber.activate()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        model = cohere_bedrock.embedding_model("cohere.embed-multilingual-v3")
        embeddings = await model.embed(["query"], task="retrieval_query")
        assert embeddings.shape == (1, 1024)
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
        client.close()


@_run_async_test
async def test_v4_accepts_float_keyed_embeddings() -> None:
    """Accept the float-keyed v4 response form."""
    client = _bedrock_client()
    stubber = Stubber(client)
    body = _request_body({
        "texts": ["text"],
        "input_type": "classification",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": 256,
    })
    vector = [1.0, *([0.0] * 255)]
    body_bytes = json.dumps({"embeddings": {"float": [vector]}}).encode()
    response = {
        "body": StreamingBody(BytesIO(body_bytes), len(body_bytes)),
        "contentType": _JSON_CONTENT_TYPE,
    }
    stubber.add_response(
        "invoke_model",
        response,
        _expected_parameters(body, "cohere.embed-v4:0"),
    )
    stubber.activate()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        embedding_model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
        vectors = await embedding_model.embed(["text"], task="classification")
        assert vectors.shape == (1, 256)
    finally:
        stubber.deactivate()
        client.close()


def test_embedding_model_default_dimensions() -> None:
    """Check model-specific default dimensions."""
    client = _bedrock_client()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        assert cohere_bedrock.embedding_model("cohere.embed-v4:0").dimension == 1536
        assert cohere_bedrock.embedding_model("cohere.embed-english-v3").dimension == 1024
    finally:
        client.close()


@pytest.mark.parametrize(
    ("error_code", "expected_kind"),
    [
        ("ThrottlingException", "pause_all"),
        ("ModelNotReadyException", "retry_this_one"),
    ],
)
def test_429_classification_uses_the_service_model(
    error_code: str,
    expected_kind: str,
) -> None:
    """Only the service model's throttling error pauses every request."""
    failure = ClientError(
        {
            "Error": {"Code": error_code, "Message": "retry"},
            "ResponseMetadata": {
                "HTTPStatusCode": 429,
                "HTTPHeaders": {},
                "RequestId": "test",
                "HostId": "test",
                "RetryAttempts": 0,
            },
        },
        "InvokeModel",
    )
    assert cohere_backend._parse_cohere_bedrock(failure).kind == expected_kind


@_run_async_test
async def test_v4_local_body_target_splits_before_the_next_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V4 batches remain beneath the local body target."""
    monkeypatch.setattr(cohere_backend, "_COHERE_V4_BODY_TARGET_BYTES", 500)
    monkeypatch.setattr(cohere_backend, "_INVOKE_MODEL_MAX_BODY_BYTES", 2000)
    client = _bedrock_client()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        embedding_model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
        texts = ("x" * 300, "y" * 300)
        batches = await embedding_model._adapter.partition_inputs(
            texts,
            task="retrieval_document",
        )
        assert batches == ((texts[0],), (texts[1],))
    finally:
        client.close()


@_run_async_test
async def test_v3_exact_body_limit_splits_before_the_next_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V3 batches remain beneath the exact `InvokeModel` body limit."""
    monkeypatch.setattr(cohere_backend, "_INVOKE_MODEL_MAX_BODY_BYTES", 1000)
    client = _bedrock_client()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        embedding_model = cohere_bedrock.embedding_model("cohere.embed-english-v3")
        texts = ("x" * 500, "y" * 500)
        batches = await embedding_model._adapter.partition_inputs(
            texts,
            task="retrieval_document",
        )
        assert batches == ((texts[0],), (texts[1],))
    finally:
        client.close()


@_run_async_test
async def test_batching_uses_ninety_six_inputs() -> None:
    """A ninety-seventh input starts the following request."""
    client = _bedrock_client()
    stubber = Stubber(client)
    inputs = [f"text-{index}" for index in range(97)]
    vector = [1.0, *([0.0] * 255)]
    first_body = _request_body({
        "texts": inputs[:96],
        "input_type": "clustering",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": 256,
    })
    second_body = _request_body({
        "texts": inputs[96:],
        "input_type": "clustering",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": 256,
    })
    stubber.add_response(
        "invoke_model",
        _response([vector] * 96),
        _expected_parameters(first_body, "cohere.embed-v4:0"),
    )
    stubber.add_response(
        "invoke_model",
        _response([vector]),
        _expected_parameters(second_body, "cohere.embed-v4:0"),
    )
    stubber.activate()
    try:
        cohere_bedrock = CohereBedrock(client=client, max_concurrent_requests=1)
        model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
        embeddings = await model.embed(inputs, task="clustering")
        assert embeddings.shape == (97, 256)
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
        client.close()


@_run_async_test
async def test_oversized_singleton_fails_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized singleton raises before client preparation."""
    monkeypatch.setattr(cohere_backend, "_INVOKE_MODEL_MAX_BODY_BYTES", 100)
    creation_calls = 0

    def create_client(
        service_name: str,
        *,
        region_name: str | None,
        config: Config,
    ) -> object:
        nonlocal creation_calls
        assert service_name == "bedrock-runtime"
        assert region_name == "us-east-1"
        assert isinstance(config, Config)
        creation_calls += 1
        return object()

    monkeypatch.setattr(cohere_backend.boto3, "client", create_client)
    cohere_bedrock = CohereBedrock(aws_region="us-east-1")
    model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
    with pytest.raises(ValueError, match="body limit"):
        await model.embed(["x" * 100], task="retrieval_document")
    assert creation_calls == 0


@_run_async_test
async def test_invalid_response_hides_vectors() -> None:
    """Malformed response errors contain no generated vector values."""
    client = _bedrock_client()
    stubber = Stubber(client)
    body = _request_body({
        "texts": ["text"],
        "input_type": "classification",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": 256,
    })
    response_bytes = b'{"embeddings":[[8675309.0]]}'
    response = {
        "body": StreamingBody(BytesIO(response_bytes), len(response_bytes)),
        "contentType": _JSON_CONTENT_TYPE,
    }
    stubber.add_response(
        "invoke_model",
        response,
        _expected_parameters(body, "cohere.embed-v4:0"),
    )
    stubber.activate()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
        with pytest.raises(EmbeddingOutputError) as raised:
            await model.embed(["text"], task="classification")
        assert "8675309" not in str(raised.value)
    finally:
        stubber.deactivate()
        client.close()


class _FakeBody:
    """A controllable synchronous response body."""

    def __init__(self, body_bytes: bytes) -> None:
        """Store response bytes and start open."""
        self._body_bytes = body_bytes
        self.closed = False

    def read(self) -> bytes:
        """Return stored response bytes."""
        return self._body_bytes

    def close(self) -> None:
        """Record body closure."""
        self.closed = True


class _FakeClient:
    """A controllable Bedrock Runtime client."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        invoke_started: threading.Event | None = None,
        invoke_release: threading.Event | None = None,
        failures: list[Exception] | None = None,
    ) -> None:
        """Store request controls."""
        self.dimension = dimension
        self.invoke_started = invoke_started
        self.invoke_release = invoke_release
        self.failures = list(failures or ())
        self.invoke_thread_id: int | None = None
        self.requests: list[bytes] = []
        self.bodies: list[_FakeBody] = []
        self.closed = False

    def invoke_model(
        self,
        *,
        body: bytes,
        modelId: str,  # noqa: N803 (matches boto3's required keyword)
        accept: str,
        contentType: str,  # noqa: N803 (matches boto3's required keyword)
    ) -> dict[str, object]:
        """Record one request and return one valid vector."""
        assert modelId in COHERE_BEDROCK_EMBEDDING_MODELS
        assert accept == _JSON_CONTENT_TYPE
        assert contentType == _JSON_CONTENT_TYPE
        self.invoke_thread_id = threading.get_ident()
        self.requests.append(body)
        if self.failures:
            raise self.failures.pop(0)
        if self.invoke_started is not None:
            self.invoke_started.set()
        if self.invoke_release is not None:
            _ = self.invoke_release.wait()
        body_bytes = json.dumps({"embeddings": [[1.0, *([0.0] * (self.dimension - 1))]]}).encode()
        response_body = _FakeBody(body_bytes)
        self.bodies.append(response_body)
        return {"body": response_body}

    def close(self) -> None:
        """Record client closure."""
        self.closed = True


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: _FakeClient,
    *,
    creation_started: threading.Event | None = None,
    creation_release: threading.Event | None = None,
) -> list[Config]:
    """Install one fake lazy-client factory."""
    configs: list[Config] = []

    def create_client(
        service_name: str,
        *,
        region_name: str | None,
        config: Config,
    ) -> _FakeClient:
        assert service_name == "bedrock-runtime"
        assert region_name == "us-east-1"
        configs.append(config)
        if creation_started is not None:
            creation_started.set()
        if creation_release is not None:
            _ = creation_release.wait()
        return fake_client

    monkeypatch.setattr(cohere_backend.boto3, "client", create_client)
    return configs


async def _wait_for_event(event: threading.Event) -> None:
    """Wait for a thread event without blocking the event loop."""
    completed = await asyncio.to_thread(event.wait, 1.0)
    assert completed


@_run_async_test
async def test_lazy_client_creation_precedes_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy creation completes before the first admission."""
    creation_started = threading.Event()
    fake_client = _FakeClient()
    configs = _install_fake_client(
        monkeypatch,
        fake_client,
        creation_started=creation_started,
    )
    cohere_bedrock = CohereBedrock(aws_region="us-east-1")
    cohere_bedrock._shared_backoff._pause_until = float("inf")
    model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
    embed_task = asyncio.create_task(model.embed(["text"], task="retrieval_query"))
    try:
        await _wait_for_event(creation_started)
        assert configs
        assert not embed_task.done()
        assert vars(configs[0])["retries"] == {"total_max_attempts": 1}
    finally:
        _ = embed_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await embed_task
    assert not fake_client.closed


@_run_async_test
async def test_concurrent_preparation_creates_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first requests share one serialized client creation."""
    creation_started = threading.Event()
    creation_release = threading.Event()
    fake_client = _FakeClient()
    configs = _install_fake_client(
        monkeypatch,
        fake_client,
        creation_started=creation_started,
        creation_release=creation_release,
    )
    cohere_bedrock = CohereBedrock(aws_region="us-east-1")
    model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
    first = asyncio.create_task(model.embed(["first"], task="classification"))
    second = asyncio.create_task(model.embed(["second"], task="classification"))
    await _wait_for_event(creation_started)
    await asyncio.sleep(0)
    assert len(configs) == 1
    creation_release.set()
    first_vectors, second_vectors = await asyncio.gather(first, second)
    assert first_vectors.shape == (1, 256)
    assert second_vectors.shape == (1, 256)
    assert len(configs) == 1
    assert len(fake_client.requests) == 2
    assert not fake_client.closed


@_run_async_test
async def test_synchronous_attempt_runs_outside_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete SDK attempt runs outside the event-loop thread."""
    fake_client = _FakeClient()
    _ = _install_fake_client(monkeypatch, fake_client)
    cohere_bedrock = CohereBedrock(aws_region="us-east-1")
    model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
    event_loop_thread_id = threading.get_ident()
    await model.embed(["text"], task="classification")
    assert fake_client.invoke_thread_id != event_loop_thread_id
    assert fake_client.bodies[0].closed


@_run_async_test
async def test_transient_client_error_retries_only_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient `ClientError` retries its request."""
    failure = ClientError(
        {
            "Error": {"Code": "InternalServerException", "Message": "retry"},
            "ResponseMetadata": {
                "HTTPStatusCode": 500,
                "HTTPHeaders": {},
                "RequestId": "test",
                "HostId": "test",
                "RetryAttempts": 0,
            },
        },
        "InvokeModel",
    )
    fake_client = _FakeClient(failures=[failure])
    _ = _install_fake_client(monkeypatch, fake_client)
    cohere_bedrock = CohereBedrock(
        aws_region="us-east-1",
        minimum_wait_ceiling_seconds=0.000_001,
        longest_wait_seconds=0.000_002,
        quiet_seconds_per_decay_step=0.000_001,
    )
    model = cohere_bedrock.embedding_model(
        "cohere.embed-v4:0",
        dimension=256,
        max_attempts=2,
    )
    embeddings = await model.embed(["text"], task="classification")
    assert embeddings.shape == (1, 256)
    assert len(fake_client.requests) == 2


@_run_async_test
async def test_cancellation_waits_for_synchronous_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation waits for invocation and response closure."""
    invoke_started = threading.Event()
    invoke_release = threading.Event()
    fake_client = _FakeClient(
        invoke_started=invoke_started,
        invoke_release=invoke_release,
    )
    _ = _install_fake_client(monkeypatch, fake_client)
    cohere_bedrock = CohereBedrock(aws_region="us-east-1")
    model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
    embed_task = asyncio.create_task(model.embed(["text"], task="classification"))
    await _wait_for_event(invoke_started)
    _ = embed_task.cancel()
    await asyncio.sleep(0)
    assert not embed_task.done()
    invoke_release.set()
    with pytest.raises(asyncio.CancelledError):
        await embed_task
    assert fake_client.bodies[0].closed


@_run_async_test
async def test_cancelled_creation_retains_completed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelled creation remains available for the following request."""
    creation_started = threading.Event()
    creation_release = threading.Event()
    fake_client = _FakeClient()
    _ = _install_fake_client(
        monkeypatch,
        fake_client,
        creation_started=creation_started,
        creation_release=creation_release,
    )
    cohere_bedrock = CohereBedrock(aws_region="us-east-1")
    model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
    first_task = asyncio.create_task(model.embed(["first"], task="classification"))
    await _wait_for_event(creation_started)
    _ = first_task.cancel()
    await asyncio.sleep(0)
    assert not first_task.done()
    creation_release.set()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    embeddings = await model.embed(["second"], task="classification")
    assert embeddings.shape == (1, 256)
    assert len(fake_client.requests) == 1


@_run_async_test
async def test_cancelled_failed_creation_retries_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later prepare retries client creation hidden by caller cancellation."""
    creation_calls = 0
    installed_client = _bedrock_client()
    prepare_task: asyncio.Task[None] | None = None

    async def create(
        client_cache: cohere_backend._CohereBedrockClientCache,
    ) -> BedrockRuntimeClient:
        nonlocal creation_calls
        creation_calls += 1
        if creation_calls == 1:
            assert prepare_task is not None
            _ = asyncio.get_running_loop().call_soon(prepare_task.cancel)
            raise RuntimeError("client creation failed")
        client_cache._client = installed_client
        return installed_client

    monkeypatch.setattr(cohere_backend._CohereBedrockClientCache, "_create", create)
    client_cache = cohere_backend._CohereBedrockClientCache(
        aws_region="us-east-1",
        client=None,
    )
    try:
        prepare_task = asyncio.create_task(client_cache.prepare())
        with pytest.raises(asyncio.CancelledError):
            await prepare_task

        await client_cache.prepare()
        assert creation_calls == 2
    finally:
        installed_client.close()


def test_passed_client_retry_configuration() -> None:
    """Passed clients must configure one total SDK attempt."""
    retrying_client = _bedrock_client(total_max_attempts=2)
    one_attempt_client = _bedrock_client()
    try:
        with pytest.raises(ValueError, match="total_max_attempts"):
            _ = CohereBedrock(client=retrying_client)
        _ = CohereBedrock(client=one_attempt_client)
        with pytest.raises(ValueError, match="at most one"):
            _ = CohereBedrock(client=one_attempt_client, aws_region="us-east-1")
    finally:
        retrying_client.close()
        one_attempt_client.close()
