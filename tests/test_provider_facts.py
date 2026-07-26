"""Pins the SDK facts langchaint's arithmetic and classification are written against.

Both SDKs' response models are pydantic models configured extra="allow", so a renamed or withdrawn
field arrives as an extra rather than as an error: the adapters keep reading the name they were
written for, get None or a default, and every other test in this suite keeps passing on the stale
literal. Nothing else here can fail on that drift, which is what these tests are for. They capture
no defect present at the version they were written against (anthropic 0.120.0, openai 2.45.0).

The literal-set assertions compare against written-out sets rather than against a subset check, so a
value the provider adds also fails.
"""

import typing

import anthropic
import httpx
import openai
import pytest
from anthropic.types import (
    ImageBlockParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.cache_creation import CacheCreation
from anthropic.types.output_tokens_details import OutputTokensDetails as AnthropicOutputDetails
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from langchaint.adapter import classification_from_response
from langchaint.anthropic.messages_adapter import (
    _RATE_LIMIT_STATUSES as _ANTHROPIC_RATE_LIMIT_STATUSES,
)
from langchaint.anthropic.messages_adapter import AnthropicPricedServiceTier
from langchaint.openai.responses_adapter import (
    _RATE_LIMIT_STATUSES as _OPENAI_RATE_LIMIT_STATUSES,
)


def _field_annotation(model: type[BaseModel], name: str) -> object:
    """Return one pydantic field's declared annotation.

    Raises:
        AssertionError: the model has no field of that name.
    """
    assert name in model.model_fields, f"{model.__name__} lost the field {name}"
    return model.model_fields[name].annotation


def _is_required(model: type[BaseModel], name: str) -> bool:
    """Whether one pydantic field must be present for the model to validate.

    Raises:
        AssertionError: the model has no field of that name.
    """
    assert name in model.model_fields, f"{model.__name__} lost the field {name}"
    return model.model_fields[name].is_required()


def test_anthropic_cache_counters_stay_optional_ints() -> None:
    """The anthropic cache counters are Optional[int], which is what the `or 0` in the adapter reads.

    _normalized_usage writes `usage.cache_read_input_tokens or 0`. Were these to become required
    ints, the `or 0` would still be correct but pointless.
    """
    for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        assert _field_annotation(AnthropicUsage, name) == (int | None), name
        assert not _is_required(AnthropicUsage, name), name


def test_anthropic_unguarded_counters_stay_required_ints() -> None:
    """input_tokens and output_tokens are required ints, which the adapter reads unguarded."""
    for name in ("input_tokens", "output_tokens"):
        assert _field_annotation(AnthropicUsage, name) is int, name
        assert _is_required(AnthropicUsage, name), name


def test_anthropic_cache_creation_keeps_both_ttl_counters() -> None:
    """The two-TTL split the adapter prices at different rates is still two required int fields.

    The adapter prices ephemeral_1h_input_tokens and ephemeral_5m_input_tokens separately when
    cache_creation is present.
    """
    for name in ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"):
        assert _field_annotation(CacheCreation, name) is int, name
        assert _is_required(CacheCreation, name), name


def test_anthropic_reasoning_counter_is_thinking_tokens() -> None:
    """The reasoning counter the adapter reports as output_tokens_reasoning is still named thinking_tokens."""
    assert _field_annotation(AnthropicOutputDetails, "thinking_tokens") is int
    assert _is_required(AnthropicOutputDetails, "thinking_tokens")


def test_openai_usage_counters_stay_required_ints() -> None:
    """The openai counters are required ints, which is why the adapter subtracts them unguarded.

    _normalized_usage computes the uncached count as input_tokens minus cached_tokens minus
    cache_write_tokens with no None handling. An optional field here would raise a TypeError on
    every response that omitted it.
    """
    assert _field_annotation(ResponseUsage, "input_tokens") is int
    assert _is_required(ResponseUsage, "input_tokens")
    for name in ("cached_tokens", "cache_write_tokens"):
        assert _field_annotation(InputTokensDetails, name) is int, name
        assert _is_required(InputTokensDetails, name), name
    assert _field_annotation(OutputTokensDetails, "reasoning_tokens") is int
    assert _is_required(OutputTokensDetails, "reasoning_tokens")


def test_openai_usage_details_objects_are_not_optional() -> None:
    """input_tokens_details and output_tokens_details are non-nullable, so the adapter reaches into them directly.

    Requiredness alone would not do: an annotation widened to `| None` with no default stays
    required, and _normalized_usage's details.cached_tokens read would raise AttributeError on
    every response.
    """
    for name in ("input_tokens_details", "output_tokens_details"):
        assert _is_required(ResponseUsage, name), name
    assert _field_annotation(ResponseUsage, "input_tokens_details") is InputTokensDetails
    assert _field_annotation(ResponseUsage, "output_tokens_details") is OutputTokensDetails


def test_anthropic_stop_reasons_are_the_set_the_adapter_maps() -> None:
    """A member the provider adds fails here; _normalized_stop_reason would map it to "other" unnoticed."""
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
    """Response.status members the adapter branches on: completed, failed, and incomplete."""
    annotation = _field_annotation(OpenAIResponse, "status")
    assert typing.get_args(typing.get_args(annotation)[0]) == (
        "completed",
        "failed",
        "in_progress",
        "cancelled",
        "queued",
        "incomplete",
    )


def test_openai_incomplete_reasons_are_the_set_the_adapter_maps() -> None:
    """max_output_tokens is MaxCompletionTokensExceeded and content_filter is not; a third member would fall through."""
    annotation = _field_annotation(IncompleteDetails, "reason")
    assert typing.get_args(typing.get_args(annotation)[0]) == (
        "max_output_tokens",
        "content_filter",
    )


def test_anthropic_service_tier_members_are_the_pricing_mapping_keys() -> None:
    """The tier words a response can report are exactly AnthropicPricedServiceTier.

    An adapter's pricing mapping is keyed by that alias, so a member the SDK adds and the alias
    lacks would price NaN with no table a caller could supply for it.
    """
    annotation = _field_annotation(AnthropicUsage, "service_tier")
    reported = typing.get_args(typing.get_args(annotation)[0])
    assert reported == typing.get_args(AnthropicPricedServiceTier.__value__)
    assert reported == ("standard", "priority", "batch")


_RETRY_GRID_STATUSES = (400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 529)
"""Statuses driven through both SDKs' retry predicate."""


def _response_with(status_code: int, headers: dict[str, str]) -> httpx.Response:
    """Build the httpx.Response the SDK retry predicates and error constructors read."""
    return httpx.Response(
        status_code=status_code,
        headers=headers,
        request=httpx.Request("POST", "https://example.invalid/v1"),
    )


@pytest.mark.parametrize("should_retry_header", [None, "true", "false"])
@pytest.mark.parametrize("status_code", _RETRY_GRID_STATUSES)
def test_langchaint_retries_exactly_what_both_sdks_retry(
    status_code: int, should_retry_header: str | None
) -> None:
    """classification_from_response reproduces both SDKs' _should_retry over the status grid.

    langchaint constructs its clients with max_retries=0 and owns the retrying, so this predicate is
    reimplemented rather than called.
    Each adapter's own _RATE_LIMIT_STATUSES is compared against its own SDK, so the two differ here
    the way they differ in the adapters (anthropic counts 529 a rate limit, openai does not).
    A rate-limit status carrying x-should-retry: false is the one deliberate departure: the SDKs obey
    the directive and stop, langchaint classifies rate_limit and retries, because the directive
    speaks for the one request while the pause a rate limit triggers protects the shared account.
    """
    headers = {} if should_retry_header is None else {"x-should-retry": should_retry_header}
    response = _response_with(status_code, headers)
    anthropic_retries = anthropic.Anthropic(api_key="k")._should_retry(response)
    openai_retries = openai.OpenAI(api_key="k")._should_retry(response)
    assert anthropic_retries == openai_retries, "the two SDKs' retry policies diverged"
    for sdk_retries, rate_limit_statuses in (
        (anthropic_retries, _ANTHROPIC_RATE_LIMIT_STATUSES),
        (openai_retries, _OPENAI_RATE_LIMIT_STATUSES),
    ):
        classification = classification_from_response(
            status_code=status_code,
            headers=response.headers,
            rate_limit_statuses=rate_limit_statuses,
        )
        langchaint_retries = classification in ("transient", "rate_limit")
        if status_code in rate_limit_statuses and should_retry_header == "false":
            assert classification == "rate_limit", rate_limit_statuses
            assert langchaint_retries, rate_limit_statuses
            assert not sdk_retries, rate_limit_statuses
        else:
            assert langchaint_retries == sdk_retries, rate_limit_statuses


def test_both_sdk_clients_retry_twice_by_default() -> None:
    """Both SDKs retry internally unless told otherwise, which is why every client is built with max_retries=0.

    A default of 0 would make langchaint's max_attempts count true without the parameter; a default
    above 0 that langchaint failed to override would make each langchaint attempt several requests,
    so the AttemptRecord count and the pacing would both understate what the account was charged.
    """
    assert anthropic.DEFAULT_MAX_RETRIES == 2
    assert openai.DEFAULT_MAX_RETRIES == 2


def test_openai_maps_only_the_statuses_it_lists_to_a_subclass() -> None:
    """An unlisted status is the bare APIStatusError, which is why classify branches on the status.

    413 is the named case: openai raises the bare APIStatusError for it, so a class list would drop
    it, and would drop whatever status the provider adds next the same way.
    529 is listed to pin the other half of the anthropic comparison: openai has no class of its own
    for it and reaches InternalServerError through the 5xx arm.
    """
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
            "boom", body=None, response=_response_with(status_code, {})
        )
        assert type(error) is error_class, status_code
    unlisted = client._make_status_error("boom", body=None, response=_response_with(413, {}))
    assert type(unlisted) is openai.APIStatusError


def test_anthropic_maps_only_the_statuses_it_lists_to_a_subclass() -> None:
    """The anthropic half of the same claim: an unlisted status is the bare APIStatusError.

    The two lists differ, which is itself the reason neither adapter classifies by exception class:
    anthropic gives 413 and 529 their own classes while openai maps 413 to the bare APIStatusError
    and 529 to InternalServerError alongside every other 5xx, so one shared class list would be wrong
    for both.
    """
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
            "boom", body=None, response=_response_with(status_code, {})
        )
        assert type(error) is error_class, status_code
    unlisted = client._make_status_error("boom", body=None, response=_response_with(451, {}))
    assert type(unlisted) is anthropic.APIStatusError


def test_the_schema_builders_both_adapters_call_are_still_where_they_were() -> None:
    """Both adapters build the structured request's schema themselves, with these two SDK calls.

    Each adapter imports its own at module top, so an SDK that moves either breaks importing that
    backend subpackage. Naming both here says which symbol moved, and openai's is the likelier to
    move: it lives in openai.lib._parsing._responses, a private module.
    Each adapter's own test file pins that its binding sends what these calls return.
    """
    assert "transform_schema" in anthropic.__all__
    assert callable(anthropic.transform_schema)
    assert callable(type_to_text_format_param)


def test_the_cache_breakpoint_keys_both_adapters_write_still_exist() -> None:
    """The cache-breakpoint key each adapter writes, on every param type it writes it on.

    `pyrefly check` rejects a write to a key the SDK's TypedDict lacks, so it is the gate a rename
    trips first; this names the two keys and the nine param types in one place, and its failure
    message names the class that dropped one.
    """
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
