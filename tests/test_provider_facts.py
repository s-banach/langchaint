"""Check dependency facts used by provider parsing and request limits."""

import typing

import anthropic
import boto3
import botocore.session
import httpx2
import openai
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from botocore.exceptions import ClientError
from google.genai import _api_client
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_error import ResponseError
from pydantic import BaseModel

from langchaint.anthropic import messages_adapter
from langchaint.cohere import _INVOKE_MODEL_MAX_BODY_BYTES
from langchaint.gemini import generate_content_adapter
from langchaint.openai import responses_adapter as openai_responses
from langchaint.openai import shared as openai_shared
from langchaint.shared_backoff import DoNotRetry, PauseAll

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


def test_bedrock_body_limit_matches_the_partition_limit() -> None:
    """Compare the partition limit with the SDK service model's body limit."""
    service = botocore.session.get_session().get_service_model("bedrock-runtime")
    input_shape = service.operation_model("InvokeModel").input_shape
    assert input_shape is not None
    assert input_shape.members["body"].metadata["max"] == _INVOKE_MODEL_MAX_BODY_BYTES


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
    assert set(typing.get_args(typing.get_args(annotation)[0])) == {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    }


def test_openai_response_statuses_are_the_set_the_adapter_maps() -> None:
    """Response.status values the adapter branches on: completed, failed, and incomplete."""
    annotation = _field_annotation(OpenAIResponse, "status")
    assert set(typing.get_args(typing.get_args(annotation)[0])) == {
        "completed",
        "failed",
        "in_progress",
        "cancelled",
        "queued",
        "incomplete",
    }


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
    assert set(reported) == set(
        typing.get_args(messages_adapter._AnthropicReportedServiceTier.__value__)
    )


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
        "boom", body=body, response=_anthropic_response_with(418, {})
    )
    assert isinstance(error, anthropic.APIStatusError)
    assert messages_adapter.parse_anthropic(error) == PauseAll(retry_after=None)
    bodyless = client._make_status_error(
        "boom", body=None, response=_anthropic_response_with(418, {})
    )
    assert isinstance(bodyless, anthropic.APIStatusError)
    assert messages_adapter.parse_anthropic(bodyless) == DoNotRetry()


def test_openai_status_error_reads_the_code_parse_branches_on() -> None:
    """APIStatusError.code is the body's code, and None where the body is not a dict.

    `parse_openai()` checks `failure.code` within status 429.
    This keeps spend-limit responses terminal without pausing the rate-limit quota.
    """
    client = openai.OpenAI(api_key="k")
    body = {"code": "credit_balance_exhausted", "type": "insufficient_quota", "message": "boom"}
    error = client._make_status_error("boom", body=body, response=_openai_response_with(429, {}))
    assert isinstance(error, openai.APIStatusError)
    assert openai_shared.parse_openai(error) == DoNotRetry()
    bodyless = client._make_status_error(
        "boom", body=None, response=_openai_response_with(429, {})
    )
    assert isinstance(bodyless, openai.APIStatusError)
    assert openai_shared.parse_openai(bodyless) == PauseAll(retry_after=None)


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
