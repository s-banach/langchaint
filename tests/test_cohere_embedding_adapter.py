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
from langchaint import EmbeddingError, SharedBackoff
from langchaint.adapter import RequestFailure
from langchaint.cohere import (
    COHERE_BEDROCK_CROSS_REGION_MULTIPLIER,
    COHERE_BEDROCK_EMBEDDING_MODELS,
    COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS,
    CohereBedrock,
)
from langchaint.tracing import OtelObserver
from tests.helpers import in_memory_tracer_provider, run_with_timeout

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from langchaint.cohere import CohereEmbedV4Dimension, CohereEmbedV4ModelId


_JSON_CONTENT_TYPE = "application/json"
_INPUT_TOKENS_TOTAL = 2
_RESPONSE_METADATA: dict[str, object] = {
    "HTTPHeaders": {"x-amzn-bedrock-input-token-count": str(_INPUT_TOKENS_TOTAL)}
}


def _run_async_test[**Parameters](
    function: Callable[Parameters, Coroutine[object, object, None]],
) -> Callable[Parameters, None]:
    @wraps(function)
    def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> None:
        run_with_timeout(function(*args, **kwargs))

    return wrapped


def _bedrock_client(*, total_max_attempts: int = 1) -> BedrockRuntimeClient:
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"total_max_attempts": total_max_attempts}),
    )


def _request_body(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _response(
    vectors: list[list[float]], *, response_metadata: dict[str, object] = _RESPONSE_METADATA
) -> dict[str, object]:
    body_bytes = json.dumps({"embeddings": vectors}).encode()
    return {
        "body": StreamingBody(BytesIO(body_bytes), len(body_bytes)),
        "contentType": _JSON_CONTENT_TYPE,
        "ResponseMetadata": response_metadata,
    }


def _expected_parameters(body: bytes, model: str) -> dict[str, object]:
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
    model: CohereEmbedV4ModelId,
    dimension: CohereEmbedV4Dimension,
) -> None:
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
            max_requests=1,
        ).embed(["document"], task="retrieval_document")
        assert embeddings.output.shape == (1, dimension)
        assert embeddings.usage.input_tokens_total == _INPUT_TOKENS_TOTAL
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
        assert embeddings.output.shape == (1, 1024)
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
        client.close()


@_run_async_test
async def test_v4_accepts_float_keyed_embeddings() -> None:
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
        "ResponseMetadata": _RESPONSE_METADATA,
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
        embeddings = await embedding_model.embed(["text"], task="classification")
        assert embeddings.output.shape == (1, 256)
    finally:
        stubber.deactivate()
        client.close()


def test_embedding_model_default_dimensions() -> None:
    client = _bedrock_client()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        assert cohere_bedrock.embedding_model("cohere.embed-v4:0").dimension == 1536
        assert cohere_bedrock.embedding_model("cohere.embed-english-v3").dimension == 1024
    finally:
        client.close()


@pytest.mark.parametrize(
    ("error_code", "pauses_quota"),
    [
        ("ThrottlingException", True),
        ("ModelNotReadyException", False),
    ],
)
def test_a_429_pauses_the_quota_only_for_the_service_model_throttling_error(
    error_code: str,
    *,
    pauses_quota: bool,
) -> None:
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
    assert cohere_backend._cohere_bedrock_request_failure(failure) == RequestFailure(
        kind="transient", pauses_quota=pauses_quota, retry_after_seconds=None
    )


@pytest.mark.parametrize(
    ("model", "body_target_bytes", "body_limit_bytes", "input_length"),
    [
        pytest.param("cohere.embed-v4:0", 500, 2000, 300, id="v4-local-target"),
        pytest.param("cohere.embed-english-v3", 500, 1000, 500, id="v3-body-limit"),
    ],
)
@_run_async_test
async def test_request_body_size_splits_before_the_next_input(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    body_target_bytes: int,
    body_limit_bytes: int,
    input_length: int,
) -> None:
    monkeypatch.setattr(cohere_backend, "_COHERE_V4_BODY_TARGET_BYTES", body_target_bytes)
    monkeypatch.setattr(cohere_backend, "_INVOKE_MODEL_MAX_BODY_BYTES", body_limit_bytes)
    client = _bedrock_client()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        embedding_model = cohere_bedrock.embedding_model(model)
        texts = ("x" * input_length, "y" * input_length)
        batches = await embedding_model._adapter.partition_inputs(
            texts,
            task="retrieval_document",
        )
        assert batches == ((texts[0],), (texts[1],))
    finally:
        client.close()


@_run_async_test
async def test_batching_uses_ninety_six_inputs() -> None:
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
        cohere_bedrock = CohereBedrock(
            client=client, shared_backoff=SharedBackoff(max_concurrent_requests=1)
        )
        model = cohere_bedrock.embedding_model("cohere.embed-v4:0", dimension=256)
        embeddings = await model.embed(inputs, task="clustering")
        assert embeddings.output.shape == (97, 256)
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
        client.close()


@pytest.mark.parametrize("inputs", [("x" * 200,), ("small", "x" * 200)])
@_run_async_test
async def test_oversized_singleton_fails_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
    inputs: tuple[str, ...],
) -> None:
    monkeypatch.setattr(cohere_backend, "_COHERE_V4_BODY_TARGET_BYTES", 150)
    monkeypatch.setattr(cohere_backend, "_INVOKE_MODEL_MAX_BODY_BYTES", 200)
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
        await model.embed(inputs, task="retrieval_document")
    assert creation_calls == 0


@_run_async_test
async def test_invalid_response_hides_vectors() -> None:
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
        "ResponseMetadata": _RESPONSE_METADATA,
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
        with pytest.raises(EmbeddingError) as raised:
            await model.embed(["text"], task="classification")
        assert raised.value.kind == "invalid_output_error"
        assert raised.value.usage.input_tokens_total == _INPUT_TOKENS_TOTAL
        assert "8675309" not in str(raised.value)
        assert "8675309" not in str(raised.value.__cause__)
    finally:
        stubber.deactivate()
        client.close()


@_run_async_test
async def test_a_response_without_a_token_count_is_invalid_output() -> None:
    client = _bedrock_client()
    stubber = Stubber(client)
    body = _request_body({
        "texts": ["text"],
        "input_type": "classification",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "output_dimension": 256,
    })
    stubber.add_response(
        "invoke_model",
        _response([[1.0, *([0.0] * 255)]], response_metadata={"HTTPHeaders": {}}),
        _expected_parameters(body, "cohere.embed-v4:0"),
    )
    stubber.activate()
    try:
        model = CohereBedrock(client=client).embedding_model("cohere.embed-v4:0", dimension=256)
        with pytest.raises(EmbeddingError) as raised:
            await model.embed(["text"], task="classification")
        assert raised.value.kind == "invalid_output_error"
        [request_record] = raised.value.request_history
        assert request_record.usage is None
        assert request_record.error is not None
        assert request_record.error.kind == "invalid_output"
    finally:
        stubber.deactivate()
        client.close()


_V4_RATE = COHERE_BEDROCK_EMBEDDING_USD_PER_MILLION_TOKENS["cohere.embed-v4:0"]


@pytest.mark.parametrize(
    ("model", "apply_cross_region_multiplier", "usd_per_million_tokens", "expected"),
    [
        ("cohere.embed-v4:0", True, None, _V4_RATE),
        (
            "eu.cohere.embed-v4:0",
            True,
            None,
            _V4_RATE * COHERE_BEDROCK_CROSS_REGION_MULTIPLIER["eu"],
        ),
        ("eu.cohere.embed-v4:0", False, None, _V4_RATE),
        ("eu.cohere.embed-v4:0", True, 1.5, 1.5),
    ],
)
def test_embedding_model_rate_resolves_through_the_unprefixed_catalog_rate(
    model: str,
    *,
    apply_cross_region_multiplier: bool,
    usd_per_million_tokens: float | None,
    expected: float,
) -> None:
    """The vendored LiteLLM snapshot lists no `eu.cohere.embed-v4:0` entry, so the eu multiplier is NaN.

    That makes the multiplied rate differ from the unmultiplied rate.
    """
    client = _bedrock_client()
    try:
        embedding_model = CohereBedrock(
            client=client, apply_cross_region_multiplier=apply_cross_region_multiplier
        ).embedding_model(model, usd_per_million_tokens=usd_per_million_tokens)
        assert embedding_model.usd_per_million_tokens == pytest.approx(expected, nan_ok=True)
    finally:
        client.close()


def test_prefixed_v3_identifier_resolves_to_v3() -> None:
    client = _bedrock_client()
    try:
        cohere_bedrock = CohereBedrock(client=client)
        assert cohere_bedrock.embedding_model("us.cohere.embed-english-v3").dimension == 1024
        with pytest.raises(ValueError, match="accepts no dimension"):
            _ = cohere_bedrock.embedding_model("us.cohere.embed-english-v3", dimension=256)
    finally:
        client.close()


def test_uncataloged_identifier_is_rejected() -> None:
    client = _bedrock_client()
    try:
        with pytest.raises(ValueError, match="not in COHERE_BEDROCK_EMBEDDING_MODELS"):
            _ = CohereBedrock(client=client).embedding_model("xx.cohere.embed-v4:0")
    finally:
        client.close()


class _FakeBody:
    def __init__(self, body_bytes: bytes) -> None:
        self._body_bytes = body_bytes
        self.closed = False

    def read(self) -> bytes:
        return self._body_bytes

    def close(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(
        self,
        *,
        dimension: int = 256,
        invoke_started: threading.Event | None = None,
        invoke_release: threading.Event | None = None,
        failures: list[Exception] | None = None,
    ) -> None:
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
        return {"body": response_body, "ResponseMetadata": _RESPONSE_METADATA}

    def close(self) -> None:
        self.closed = True


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: _FakeClient,
    *,
    creation_started: threading.Event | None = None,
    creation_release: threading.Event | None = None,
) -> list[Config]:
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
    completed = await asyncio.to_thread(event.wait, 1.0)
    assert completed


@_run_async_test
async def test_lazy_client_creation_precedes_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    first_embeddings, second_embeddings = await asyncio.gather(first, second)
    assert first_embeddings.output.shape == (1, 256)
    assert second_embeddings.output.shape == (1, 256)
    assert len(configs) == 1
    assert len(fake_client.requests) == 2
    assert not fake_client.closed


@_run_async_test
async def test_synchronous_request_runs_outside_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        shared_backoff=SharedBackoff(
            min_wait_ceiling_seconds=0.000_001,
            max_wait_seconds=0.000_002,
            quiet_seconds_per_decay_step=0.000_001,
        ),
    )
    model = cohere_bedrock.embedding_model(
        "cohere.embed-v4:0",
        dimension=256,
        max_requests=2,
    )
    embeddings = await model.embed(["text"], task="classification")
    assert embeddings.output.shape == (1, 256)
    assert len(fake_client.requests) == 2


@_run_async_test
async def test_cancellation_waits_for_synchronous_request(
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
    assert embeddings.output.shape == (1, 256)
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
    retrying_client = _bedrock_client(total_max_attempts=2)
    one_request_client = _bedrock_client()
    try:
        with pytest.raises(ValueError, match="total_max_attempts"):
            _ = CohereBedrock(client=retrying_client)
        _ = CohereBedrock(client=one_request_client)
        with pytest.raises(ValueError, match="at most one"):
            _ = CohereBedrock(client=one_request_client, aws_region="us-east-1")
    finally:
        retrying_client.close()
        one_request_client.close()


@_run_async_test
async def test_the_cohere_bedrock_observer_traces_embedding_requests() -> None:
    tracer_provider, exporter = in_memory_tracer_provider()
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
        cohere_bedrock = CohereBedrock(
            client=client,
            observer=OtelObserver(capture_message_content=False, tracer_provider=tracer_provider),
        )
        model = cohere_bedrock.embedding_model("cohere.embed-multilingual-v3")
        _ = await model.embed(["query"], task="retrieval_query")
    finally:
        stubber.deactivate()
        client.close()
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["gen_ai.provider.name"] == "aws.bedrock"
    assert "gen_ai.response.model" not in span.attributes
