"""Pin SDK facts used by langchaint arithmetic and request mapping.

These tests inspect declared fields, method signatures, and Bedrock service models.
"""

import inspect
import subprocess
import sys
import typing

import anthropic
import boto3
import botocore.session
import httpx2
import openai
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from anthropic.types import (
    ImageBlockParam,
    RawContentBlockDeltaEvent,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.cache_creation import CacheCreation
from anthropic.types.output_tokens_details import OutputTokensDetails as AnthropicOutputDetails
from botocore.exceptions import ClientError
from google.genai import _api_client
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from openai.resources.embeddings import AsyncEmbeddings
from openai.types import CreateEmbeddingResponse
from openai.types import Embedding as OpenAIEmbedding
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseStreamEvent,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from langchaint.anthropic import messages_adapter
from langchaint.gemini import generate_content_adapter
from langchaint.openai import responses_adapter as openai_responses
from langchaint.openai import shared as openai_shared

_ANTHROPIC_LISTED_STATUSES = (
    messages_adapter._PAUSE_STATUSES
    | messages_adapter._RETRY_THIS_ONE_STATUSES
    | messages_adapter._DO_NOT_RETRY_STATUSES
)
"""Every status parse_anthropic's three tables list, whatever verdict each gives it."""

_OPENAI_LISTED_STATUSES = (
    openai_shared._PAUSE_STATUSES
    | openai_shared._RETRY_THIS_ONE_STATUSES
    | openai_shared._DO_NOT_RETRY_STATUSES
)
"""Every status parse_openai's three tables list, whatever verdict each gives it."""

type _SupportedClient = (
    AsyncAnthropic
    | AsyncAnthropicBedrock
    | AsyncAnthropicBedrockMantle
    | AsyncOpenAI
    | AsyncAzureOpenAI
    | AsyncBedrockOpenAI
)
"""Every client class the anthropic and openai adapters accept.

The gemini adapter's genai.Client is absent: google-genai has no _make_status_error and gives no
status its own exception class, so its tables are checked against the SDK's retryable set instead.
"""


def _field_annotation(model: type[BaseModel], name: str) -> object:
    """Return one pydantic field's declared annotation.

    Raises:
        AssertionError: the model has no field of that name.
    """
    assert name in model.model_fields, f"{model.__name__} lost the field {name}"
    return model.model_fields[name].annotation


def _type_literals_of_union(annotation: object) -> set[str]:
    """Collect each variant's type Literal value.

    Raises:
        AssertionError: a variant has no `type` field.
    """
    variants = typing.get_args(annotation)
    if typing.get_origin(annotation) is typing.Annotated:
        variants = typing.get_args(variants[0])
    return {typing.get_args(_field_annotation(variant, "type"))[0] for variant in variants}


def _is_required(model: type[BaseModel], name: str) -> bool:
    """Whether one pydantic field must be present for the model to validate.

    Raises:
        AssertionError: the model has no field of that name.
    """
    assert name in model.model_fields, f"{model.__name__} lost the field {name}"
    return model.model_fields[name].is_required()


def test_anthropic_usage_fields_match_the_billing_parser() -> None:
    """Distinguish optional fallback counters from required counters and TTL details."""
    for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        assert _field_annotation(AnthropicUsage, name) == (int | None), name
        assert not _is_required(AnthropicUsage, name), name
    for name in ("input_tokens", "output_tokens"):
        assert _field_annotation(AnthropicUsage, name) is int, name
        assert _is_required(AnthropicUsage, name), name
    for name in ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"):
        assert _field_annotation(CacheCreation, name) is int, name
        assert _is_required(CacheCreation, name), name
    assert _field_annotation(AnthropicOutputDetails, "thinking_tokens") is int
    assert _is_required(AnthropicOutputDetails, "thinking_tokens")


def test_openai_usage_fields_match_the_billing_parser() -> None:
    """Require the counters and detail objects accessed directly during response billing."""
    for name in ("input_tokens", "output_tokens"):
        assert _field_annotation(ResponseUsage, name) is int, name
        assert _is_required(ResponseUsage, name), name
    for name in ("cached_tokens", "cache_write_tokens"):
        assert _field_annotation(InputTokensDetails, name) is int, name
        assert _is_required(InputTokensDetails, name), name
    assert _field_annotation(OutputTokensDetails, "reasoning_tokens") is int
    assert _is_required(OutputTokensDetails, "reasoning_tokens")
    for name in ("input_tokens_details", "output_tokens_details"):
        assert _is_required(ResponseUsage, name), name
    assert _field_annotation(ResponseUsage, "input_tokens_details") is InputTokensDetails
    assert _field_annotation(ResponseUsage, "output_tokens_details") is OutputTokensDetails


def test_openai_embeddings_create_accepts_the_adapter_parameters() -> None:
    """Pin the keyword parameters sent by `_OpenAIEmbeddingAdapter`."""
    signature = inspect.signature(AsyncEmbeddings.create)
    for name in ("input", "model", "dimensions", "encoding_format"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    hints = typing.get_type_hints(AsyncEmbeddings.create)
    assert hints["return"] is CreateEmbeddingResponse


def test_openai_embedding_response_keeps_the_fields_the_adapter_reads() -> None:
    """Pin required response fields read by `_OpenAIEmbeddingAdapter`."""
    data_annotation = _field_annotation(CreateEmbeddingResponse, "data")
    assert typing.get_origin(data_annotation) is list
    assert typing.get_args(data_annotation) == (OpenAIEmbedding,)
    embedding_annotation = _field_annotation(OpenAIEmbedding, "embedding")
    assert typing.get_origin(embedding_annotation) is list
    assert typing.get_args(embedding_annotation) == (float,)
    assert _field_annotation(OpenAIEmbedding, "index") is int
    for name in ("embedding", "index"):
        assert _is_required(OpenAIEmbedding, name)


def test_bedrock_invoke_model_keeps_the_adapter_request_shape() -> None:
    """Pin request fields and the exact `body` byte limit."""
    service = botocore.session.get_session().get_service_model("bedrock-runtime")
    operation = service.operation_model("InvokeModel")
    input_shape = operation.input_shape
    assert input_shape is not None
    assert {"body", "modelId", "accept", "contentType"} <= set(input_shape.members)
    body = input_shape.members["body"]
    assert body.type_name == "blob"
    assert body.metadata["max"] == 25_000_000
    assert operation.has_streaming_output


def test_bedrock_invoke_model_error_statuses_match_retry_parsing() -> None:
    """Pin the retryable statuses parsed by the Cohere adapter."""
    service = botocore.session.get_session().get_service_model("bedrock-runtime")
    operation = service.operation_model("InvokeModel")
    status_by_error = {
        shape.name: shape.metadata["error"]["httpStatusCode"] for shape in operation.error_shapes
    }
    assert {
        "ThrottlingException": 429,
        "ModelNotReadyException": 429,
        "ModelTimeoutException": 408,
        "InternalServerException": 500,
        "ServiceUnavailableException": 503,
    }.items() <= status_by_error.items()
    model_not_ready = next(
        shape for shape in operation.error_shapes if shape.name == "ModelNotReadyException"
    )
    assert model_not_ready.metadata["retryable"] == {"throttling": False}


def test_bedrock_modeled_exceptions_remain_client_errors() -> None:
    """Pin the exception type caught by the Cohere adapter."""
    service = botocore.session.get_session().get_service_model("bedrock-runtime")
    operation = service.operation_model("InvokeModel")
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="offline",
        aws_secret_access_key="offline",
    )
    try:
        for shape in operation.error_shapes:
            error_class = getattr(client.exceptions, shape.name)
            assert issubclass(error_class, ClientError)
    finally:
        client.close()


def test_anthropic_stop_reasons_are_the_set_the_adapter_maps() -> None:
    """A value the provider adds fails here. _normalized_stop_reason would map it to "other" unnoticed."""
    annotation = _field_annotation(AnthropicMessage, "stop_reason")
    assert typing.get_args(typing.get_args(annotation)[0]) == (
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    )


def test_openai_response_statuses_are_the_set_the_adapter_maps() -> None:
    """Response.status values the adapter branches on: completed, failed, and incomplete."""
    annotation = _field_annotation(OpenAIResponse, "status")
    assert typing.get_args(typing.get_args(annotation)[0]) == (
        "completed",
        "failed",
        "in_progress",
        "cancelled",
        "queued",
        "incomplete",
    )


def test_openai_incomplete_reasons_have_dispositions() -> None:
    """Fail when the installed SDK adds an incomplete reason without an adapter disposition."""
    annotation = _field_annotation(IncompleteDetails, "reason")
    reasons = set(typing.get_args(typing.get_args(annotation)[0]))
    assert reasons <= openai_responses._STOP_REASON_BY_INCOMPLETE_REASON.keys()


def test_every_openai_error_code_has_a_disposition() -> None:
    """_DISPOSITION_BY_ERROR_CODE covers each ResponseError.code value."""
    annotation = _field_annotation(ResponseError, "code")
    codes = set(typing.get_args(annotation))
    assert codes <= set(openai_shared._DISPOSITION_BY_ERROR_CODE)


def test_anthropic_reported_service_tier_values_match_the_sdk() -> None:
    """The SDK response tiers match `_AnthropicReportedServiceTier`."""
    annotation = _field_annotation(AnthropicUsage, "service_tier")
    reported = typing.get_args(typing.get_args(annotation)[0])
    assert reported == typing.get_args(messages_adapter._AnthropicReportedServiceTier.__value__)
    assert reported == ("standard", "priority", "batch")


def _anthropic_response_with(status_code: int, headers: dict[str, str]) -> httpx2.Response:
    """Build the httpx2.Response read by anthropic error constructors."""
    return httpx2.Response(
        status_code=status_code,
        headers=headers,
        request=httpx2.Request("POST", "https://example.invalid/v1"),
    )


def _openai_response_with(status_code: int, headers: dict[str, str]) -> httpx2.Response:
    """Build the httpx2.Response read by openai error constructors."""
    return httpx2.Response(
        status_code=status_code,
        headers=headers,
        request=httpx2.Request("POST", "https://example.invalid/v1"),
    )


def test_anthropic_status_error_reads_the_error_type_parse_branches_on() -> None:
    """APIStatusError.type is the body's error.type, and None where the body is not that shape.

    `parse_anthropic()` checks `failure.type` before status.
    This lets an unlisted `overloaded_error` pause the rate-limit quota.
    """
    client = anthropic.Anthropic(api_key="k")
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "boom"}}
    error = client._make_status_error(
        "boom", body=body, response=_anthropic_response_with(529, {})
    )
    assert isinstance(error, anthropic.APIStatusError)
    assert error.type == "overloaded_error"
    bodyless = client._make_status_error(
        "boom", body=None, response=_anthropic_response_with(529, {})
    )
    assert isinstance(bodyless, anthropic.APIStatusError)
    assert bodyless.type is None


def test_openai_status_error_reads_the_code_parse_branches_on() -> None:
    """APIStatusError.code is the body's code, and None where the body is not a dict.

    `parse_openai()` checks `failure.code` within status 429.
    This keeps spend-limit responses terminal without pausing the rate-limit quota.
    """
    client = openai.OpenAI(api_key="k")
    body = {"code": "insufficient_quota", "type": "insufficient_quota", "message": "boom"}
    error = client._make_status_error("boom", body=body, response=_openai_response_with(429, {}))
    assert isinstance(error, openai.APIStatusError)
    assert error.code == "insufficient_quota"
    bodyless = client._make_status_error(
        "boom", body=None, response=_openai_response_with(429, {})
    )
    assert isinstance(bodyless, openai.APIStatusError)
    assert bodyless.code is None


def test_openai_maps_only_the_statuses_it_lists_to_a_subclass() -> None:
    """OpenAI uses APIStatusError for unlisted status codes."""
    client = openai.OpenAI(api_key="k")
    listed = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        409: openai.ConflictError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        529: openai.InternalServerError,
    }
    for status_code, error_class in listed.items():
        error = client._make_status_error(
            "boom", body=None, response=_openai_response_with(status_code, {})
        )
        assert type(error) is error_class, status_code
    unlisted = client._make_status_error(
        "boom", body=None, response=_openai_response_with(413, {})
    )
    assert type(unlisted) is openai.APIStatusError


def test_anthropic_maps_only_the_statuses_it_lists_to_a_subclass() -> None:
    """Anthropic uses APIStatusError for unlisted status codes."""
    client = anthropic.Anthropic(api_key="k")
    listed = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        403: anthropic.PermissionDeniedError,
        404: anthropic.NotFoundError,
        409: anthropic.ConflictError,
        413: anthropic.RequestTooLargeError,
        422: anthropic.UnprocessableEntityError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
        529: anthropic.OverloadedError,
    }
    for status_code, error_class in listed.items():
        error = client._make_status_error(
            "boom", body=None, response=_anthropic_response_with(status_code, {})
        )
        assert type(error) is error_class, status_code
    unlisted = client._make_status_error(
        "boom", body=None, response=_anthropic_response_with(451, {})
    )
    assert type(unlisted) is anthropic.APIStatusError


def _statuses_with_a_dedicated_class(client: _SupportedClient) -> set[int]:
    """Return statuses with client-specific error classes."""
    named: set[int] = set()
    for status_code in range(400, 600):
        if isinstance(
            client, (AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle)
        ):
            error = client._make_status_error(
                "boom", body=None, response=_anthropic_response_with(status_code, {})
            )
        else:
            error = client._make_status_error(
                "boom", body=None, response=_openai_response_with(status_code, {})
            )
        if getattr(type(error), "status_code", None) is not None:
            named.add(status_code)
    return named


def test_every_status_a_supported_client_names_is_in_one_of_the_verdict_tables() -> None:
    """Verdict tables cover each client-specific status class."""
    clients_and_listed_statuses = (
        (AsyncAnthropic(api_key="k"), _ANTHROPIC_LISTED_STATUSES),
        (
            AsyncAnthropicBedrock(aws_region="us-east-1", aws_access_key="k", aws_secret_key="s"),
            _ANTHROPIC_LISTED_STATUSES,
        ),
        (AsyncAnthropicBedrockMantle(aws_region="us-east-1"), _ANTHROPIC_LISTED_STATUSES),
        (AsyncOpenAI(api_key="k"), _OPENAI_LISTED_STATUSES),
        (
            AsyncAzureOpenAI(
                api_key="k", api_version="2024-10-01", azure_endpoint="https://example.invalid"
            ),
            _OPENAI_LISTED_STATUSES,
        ),
        (
            AsyncBedrockOpenAI(
                aws_region="us-east-1", aws_access_key_id="k", aws_secret_access_key="s"
            ),
            _OPENAI_LISTED_STATUSES,
        ),
    )
    for client, listed_statuses in clients_and_listed_statuses:
        assert _statuses_with_a_dedicated_class(client) <= listed_statuses, type(client).__name__


def test_the_gemini_sdk_retryable_statuses_are_all_retried_or_paused() -> None:
    """Gemini retry tables cover the SDK retryable statuses."""
    assert set(_api_client._RETRY_HTTP_STATUS_CODES) <= (
        generate_content_adapter._PAUSE_STATUSES
        | generate_content_adapter._RETRY_THIS_ONE_STATUSES
    )


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Python 3.13 does not emit this deprecation warning."
)
def test_google_genai_deprecation_filter_remains_necessary() -> None:
    """Fail when the exact google-genai warning exemption can be removed."""
    warning_text = "'_UnionGenericAlias' is deprecated and slated for removal in Python 3.17"
    source = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        '    warnings.simplefilter("always")\n'
        "    from google import genai\n"
        "assert len(caught) == 1, caught\n"
        "warning = caught[0]\n"
        f"assert str(warning.message) == {warning_text!r}, warning\n"
        "assert warning.category is DeprecationWarning, warning\n"
        'assert warning.filename.replace("\\\\", "/").endswith("/google/genai/types.py"), warning\n'
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "google-genai changed its import warning; remove or update its filterwarnings exemption\n"
        f"{completed.stderr}"
    )


def test_anthropic_content_block_deltas_carry_the_two_kinds_of_text_the_stream_yields() -> None:
    """The stream branches on these two delta types. A rename drops that text from the stream unnoticed."""
    type_values = _type_literals_of_union(_field_annotation(RawContentBlockDeltaEvent, "delta"))
    assert "text_delta" in type_values
    assert "thinking_delta" in type_values


def test_openai_stream_events_carry_the_delta_and_done_types_the_stream_branches_on() -> None:
    """ResponseStreamEvent includes each handled reasoning event type."""
    type_values = _type_literals_of_union(ResponseStreamEvent)
    assert "response.output_text.delta" in type_values
    assert "response.reasoning_summary_text.delta" in type_values
    assert "response.reasoning_text.delta" in type_values
    assert "response.reasoning_summary_text.done" in type_values
    assert "response.reasoning_text.done" in type_values


def test_the_cache_breakpoint_keys_both_adapters_write_still_exist() -> None:
    """Each SDK parameter type contains its cache-breakpoint key."""
    anthropic_params = (
        TextBlockParam,
        ImageBlockParam,
        ToolResultBlockParam,
        ToolUseBlockParam,
        ToolParam,
    )
    for anthropic_param in anthropic_params:
        keys = anthropic_param.__required_keys__ | anthropic_param.__optional_keys__
        assert "cache_control" in keys, anthropic_param.__name__
    openai_params = (
        ResponseInputTextParam,
        ResponseInputImageParam,
        ResponseInputTextContentParam,
        ResponseInputImageContentParam,
    )
    for openai_param in openai_params:
        keys = openai_param.__required_keys__ | openai_param.__optional_keys__
        assert "prompt_cache_breakpoint" in keys, openai_param.__name__
