"""Check dependency facts used by provider parsing and request limits."""

import typing

import anthropic
import boto3
import botocore.session
import httpx2
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from botocore.exceptions import ClientError
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_error import ResponseError
from pydantic import BaseModel

from langchaint.anthropic import messages_adapter
from langchaint.cohere import _INVOKE_MODEL_MAX_BODY_BYTES
from langchaint.openai import responses_adapter as openai_responses
from langchaint.openai import shared as openai_shared


def _field_annotation(model: type[BaseModel], name: str) -> object:
    """Return one pydantic field's declared annotation.

    Raises:
        AssertionError: the model has no field of that name.
    """
    assert name in model.model_fields, f"{model.__name__} lost the field {name}"
    return model.model_fields[name].annotation


def test_bedrock_body_limit_matches_the_partition_limit() -> None:
    service = botocore.session.get_session().get_service_model("bedrock-runtime")
    input_shape = service.operation_model("InvokeModel").input_shape
    assert input_shape is not None
    assert input_shape.members["body"].metadata["max"] == _INVOKE_MODEL_MAX_BODY_BYTES


def test_bedrock_invoke_model_error_statuses_match_retry_parsing() -> None:
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


def test_openai_incomplete_reasons_have_dispositions() -> None:
    """Fail when the installed SDK adds an incomplete reason without an adapter disposition."""
    annotation = _field_annotation(IncompleteDetails, "reason")
    reasons = set(typing.get_args(typing.get_args(annotation)[0]))
    assert reasons <= openai_responses._STOP_REASON_BY_INCOMPLETE_REASON.keys()


def test_every_openai_error_code_has_a_disposition() -> None:
    annotation = _field_annotation(ResponseError, "code")
    codes = set(typing.get_args(annotation))
    assert codes <= set(openai_shared._DISPOSITION_BY_ERROR_CODE)


def test_anthropic_reported_service_tier_values_match_the_sdk() -> None:
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


def test_anthropic_status_error_reads_the_error_type_request_failure_branches_on() -> None:
    """APIStatusError.type is the body's error.type, and None where the body is not that shape.

    `anthropic_request_failure()` uses `error.type` to pause the rate-limit quota.
    An `overloaded_error` pauses even when the SDK declines to retry its status.
    """
    client = anthropic.AsyncAnthropic(api_key="k")
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "boom"}}
    error = client._make_status_error(
        "boom", body=body, response=_anthropic_response_with(418, {})
    )
    assert isinstance(error, anthropic.APIStatusError)
    assert messages_adapter.anthropic_request_failure(error, client=client).pauses_quota
    bodyless = client._make_status_error(
        "boom", body=None, response=_anthropic_response_with(418, {})
    )
    assert isinstance(bodyless, anthropic.APIStatusError)
    assert not messages_adapter.anthropic_request_failure(bodyless, client=client).pauses_quota
