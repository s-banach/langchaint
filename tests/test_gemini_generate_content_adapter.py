"""Test Gemini generateContent with constructed SDK objects.

Tests cover Usage, stop reasons, reasoning, tool calls, streams, and errors.
"""

import asyncio
import json
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import override

import httpx
import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, TypeAdapter

from langchaint import (
    AllowedToolsChoice,
    AssistantMessage,
    AudioPart,
    Billing,
    ContentPart,
    ImagePart,
    ImageUrlPart,
    Message,
    RawPart,
    ReasoningDelta,
    ReasoningPart,
    SpecificToolChoice,
    StreamItem,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.adapter import (
    REASONING_PART_SEPARATOR,
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    EmptyTurn,
    ErrorClassification,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    NoOutput,
    ProviderBilling,
    Refusal,
    RequestParams,
    SchemaViolation,
    ToolChoice,
    UnfinishedTurn,
)
from langchaint.conformance import AdapterConformance
from langchaint.exceptions import TransientError
from langchaint.gemini import (
    GeminiGenerateContentAdapter,
    GeminiPricingTable,
    GeminiRates,
    assembled_response,
)
from langchaint.gemini.generate_content_adapter import (
    _billing_from_response as _provider_billing_from_response,
)
from langchaint.gemini.generate_content_adapter import (
    _billing_from_usage as _provider_billing_from_usage,
)
from langchaint.gemini.generate_content_adapter import (
    _GeminiRequestParams,
    _GeminiStream,
)
from langchaint.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict
from langchaint.tools import ToolSchema


def _billing_from_usage(
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
    pricing: Mapping[str, GeminiPricingTable],
    *,
    provider_executed_tool_cost_in_usd: float,
) -> Billing:
    return _provider_billing_from_usage(
        usage_metadata,
        pricing,
        provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
    ).billing


def _billing_from_response(
    response: types.GenerateContentResponse,
    pricing: Mapping[str, GeminiPricingTable],
    *,
    configured_fields: frozenset[str] = frozenset(),
    billing_complete: bool = True,
) -> Billing:
    return _provider_billing_from_response(
        response,
        pricing,
        configured_fields=configured_fields,
        billing_complete=billing_complete,
    ).billing


_ON_DEMAND_RATES = GeminiRates(
    input_cache_none_usd_per_million_tokens=1.0,
    cache_read_usd_per_million_tokens=0.1,
    output_usd_per_million_tokens=10.0,
)

_PRICING: dict[str, GeminiPricingTable] = {
    "ON_DEMAND": GeminiPricingTable(
        rates=_ON_DEMAND_RATES,
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    )
}
"""The on-demand tier alone, so a response reporting another traffic_type prices NaN."""

_LONG_PROMPT_TABLE = GeminiPricingTable(
    rates=_ON_DEMAND_RATES,
    google_search_usd_per_query=0.014,
    google_maps_usd_per_query=0.014,
    long_prompt_threshold_tokens=200,
    long_prompt_rates=GeminiRates(
        input_cache_none_usd_per_million_tokens=2.0,
        cache_read_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=20.0,
    ),
)
"""Every long rate is twice its base rate, so a test tells the two tiers apart by one factor."""


def _adapter() -> GeminiGenerateContentAdapter:
    """Build the adapter under test on an offline client."""
    return GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=False),
        model="gemini-3.5-flash",
        pricing=_PRICING,
        provider_name="gcp.gemini",
    )


def test_config_fingerprint_data_contains_only_stored_request_configuration() -> None:
    """Fingerprint data includes constructor request settings and excludes billing settings."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=False),
        model="gemini-3.5-flash",
        pricing=_PRICING,
        provider_name="gcp.gemini",
        service_tier="priority",
    )
    assert adapter.config_fingerprint_data() == {"service_tier": "priority"}


def _binding(
    *,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    tool_schemas: tuple[ToolSchema, ...] = (),
    provider_executed_tools: tuple[Mapping[str, object], ...] = (),
    tool_choice: ToolChoice = "auto",
    parallel_tool_calls: bool = True,
    max_completion_tokens: int | None = None,
    reasoning_level: str | None = None,
    temperature: float | None = None,
    automatic_cache_breakpoints: bool = False,
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    """Build a Binding with every field stated, defaults naming the plainest choice."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        provider_executed_tools=provider_executed_tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        max_completion_tokens=max_completion_tokens,
        reasoning_level=reasoning_level,
        temperature=temperature,
        automatic_cache_breakpoints=automatic_cache_breakpoints,
        extra_body=extra_body,
    )


def _bound_config(binding: Binding) -> types.GenerateContentConfig:
    """Bind for text and read the config the binding produced."""
    bound = _adapter().bind_text(binding)
    request = bound.build_request([UserMessage(content="hi")])
    assert isinstance(request, _GeminiRequestParams)
    return request.config


def _built_request(
    messages: Sequence[Message], binding: Binding | None = None
) -> _GeminiRequestParams:
    """Build a request that must be valid."""
    request = _adapter().bind_text(binding or _binding()).build_request(messages)
    assert isinstance(request, _GeminiRequestParams)
    return request


def _invalid_request(messages: Sequence[Message]) -> InvalidRequest:
    """Build a request that must be invalid."""
    request = _adapter().bind_text(_binding()).build_request(messages)
    assert isinstance(request, InvalidRequest)
    return request


def _usage_metadata(
    *,
    prompt_token_count: int = 100,
    cached_content_token_count: int | None = 40,
    tool_use_prompt_token_count: int | None = 20,
    candidates_token_count: int | None = 50,
    thoughts_token_count: int | None = 10,
    traffic_type: types.TrafficType | None = None,
) -> types.GenerateContentResponseUsageMetadata:
    """Build usage metadata. The defaults exercise every counter the partition reads."""
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_token_count,
        cached_content_token_count=cached_content_token_count,
        tool_use_prompt_token_count=tool_use_prompt_token_count,
        candidates_token_count=candidates_token_count,
        thoughts_token_count=thoughts_token_count,
        traffic_type=traffic_type,
    )


def _response(
    parts: Sequence[types.Part] | None,
    *,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
    usage_metadata: types.GenerateContentResponseUsageMetadata | None = None,
    grounding_metadata: types.GroundingMetadata | None = None,
    block_reason: types.BlockedReason | None = None,
) -> types.GenerateContentResponse:
    """Build a response. parts None with finish_reason None builds one without candidates."""
    candidates: list[types.Candidate] | None = None
    if parts is not None or finish_reason is not None:
        candidates = [
            types.Candidate(
                content=(
                    types.Content(role="model", parts=list(parts)) if parts is not None else None
                ),
                finish_reason=finish_reason,
                grounding_metadata=grounding_metadata,
            )
        ]
    return types.GenerateContentResponse(
        candidates=candidates,
        prompt_feedback=(
            types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
            if block_reason is not None
            else None
        ),
        usage_metadata=usage_metadata,
        model_version="gemini-3.5-flash",
        response_id="resp-1",
    )


def _provider_call_parts(
    *,
    tool_call_id: str,
    tool_type: types.ToolType,
    queries: object,
) -> list[types.Part]:
    """Build one matched server-side tool call and response pair."""
    return [
        types.Part(
            tool_call=types.ToolCall(
                id=tool_call_id,
                tool_type=tool_type,
                args={"queries": queries},
            )
        ),
        types.Part(
            tool_response=types.ToolResponse(
                id=tool_call_id,
                tool_type=tool_type,
                response={},
            )
        ),
    ]


def _gemini_stream(chunks: Sequence[types.GenerateContentResponse]) -> _GeminiStream:
    """Wrap constructed chunks in the adapter stream."""

    async def chunk_iterator() -> AsyncIterator[types.GenerateContentResponse]:
        for chunk in chunks:
            yield chunk

    return _GeminiStream(chunks=chunk_iterator(), pricing=_PRICING)


def _drained(stream: AdapterStream) -> list[StreamItem]:
    """Drain a stream to its item list."""

    async def drain() -> list[StreamItem]:
        return [item async for item in stream.items()]

    return asyncio.run(drain())


def _api_error(
    code: int,
    *,
    headers: Mapping[str, str] | None = None,
    retry_delay: str | None = None,
) -> errors.APIError:
    """Build an APIError as raise_for_response builds one, its body the {"error": ...} envelope.

    RetryInfo rows exercise errors with and without responses.
    """
    details: list[dict[str, object]] = []
    if retry_delay is not None:
        details.append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": retry_delay,
        })
    body = {
        "error": {"code": code, "message": "provider text", "status": "STATUS", "details": details}
    }
    response = httpx.Response(code, headers=dict(headers)) if headers is not None else None
    return errors.APIError(code, body, response)


class _Answer(BaseModel):
    """The structured-output model under test."""

    value: int


# --- bind ---


def test_every_request_suppresses_sdk_retries() -> None:
    """Both bindings send retry_options attempts=1, so max_attempts counts every request."""
    text_config = _bound_config(_binding())
    structured = _adapter().bind_structured(_binding(), _Answer)
    request = structured.build_request([UserMessage(content="hi")])
    assert isinstance(request, _GeminiRequestParams)
    for config in (text_config, request.config):
        assert isinstance(config.http_options, types.HttpOptions)
        assert config.http_options.retry_options == types.HttpRetryOptions(attempts=1)


def test_system_prompt_forms() -> None:
    """A str binds as-is. A parts tuple binds as one Content of text parts. None binds none."""
    assert _bound_config(_binding()).system_instruction is None
    assert _bound_config(_binding(system_prompt="be terse")).system_instruction == "be terse"
    parts_config = _bound_config(_binding(system_prompt=(TextPart(text="a"), TextPart(text="b"))))
    assert parts_config.system_instruction == types.Content(
        parts=[types.Part(text="a"), types.Part(text="b")]
    )


def test_marked_system_part_raises_at_bind() -> None:
    """cache_breakpoint has no Gemini wire form, so a marked system part is a bind defect."""
    with pytest.raises(ValueError, match="cache_breakpoint"):
        _ = _adapter().bind_text(
            _binding(system_prompt=(TextPart(text="a", cache_breakpoint=True),))
        )


def test_empty_system_parts_tuple_raises_at_bind() -> None:
    """An empty parts tuple can only come from a directly constructed Binding."""
    with pytest.raises(ValueError, match="empty tuple"):
        _ = _adapter().bind_text(_binding(system_prompt=()))


def test_parallel_tool_calls_false_raises_at_bind() -> None:
    """No wire form disables parallel function calls."""
    with pytest.raises(ValueError, match="parallel_tool_calls"):
        _ = _adapter().bind_text(_binding(parallel_tool_calls=False))


def test_automatic_cache_breakpoints_values_build_identical_requests() -> None:
    """Implicit caching has no wire form, so the parameter changes nothing."""
    enabled = _built_request(
        [UserMessage(content="hi")], _binding(automatic_cache_breakpoints=True)
    )
    disabled = _built_request(
        [UserMessage(content="hi")], _binding(automatic_cache_breakpoints=False)
    )
    assert enabled.config == disabled.config
    assert enabled.contents == disabled.contents


def _echo_schema() -> ToolSchema:
    return ToolSchema(
        name="echo",
        description="Echo the city back.",
        args_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )


def test_tool_schemas_become_function_declarations() -> None:
    """One Tool holds every declaration, its schema passed as parameters_json_schema."""
    config = _bound_config(_binding(tool_schemas=(_echo_schema(),)))
    assert config.tools == [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="echo",
                    description="Echo the city back.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            ]
        )
    ]


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)),
        ("required", types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)),
        ("none", types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)),
        (
            SpecificToolChoice(tool_name="echo"),
            types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=["echo"]
            ),
        ),
        (
            AllowedToolsChoice(mode="auto", tool_names=("echo",)),
            types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.VALIDATED,
                allowed_function_names=["echo"],
            ),
        ),
        (
            AllowedToolsChoice(mode="required", tool_names=("echo",)),
            types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=["echo"],
            ),
        ),
    ],
)
def test_tool_choice_mapping(
    tool_choice: ToolChoice, expected: types.FunctionCallingConfig
) -> None:
    """Neutral "required" is mode ANY. A specific choice is ANY narrowed to the one name."""
    config = _bound_config(_binding(tool_schemas=(_echo_schema(),), tool_choice=tool_choice))
    assert config.tool_config == types.ToolConfig(function_calling_config=expected)


def test_allowed_tools_choice_keeps_complete_gemini_function_declarations() -> None:
    """AllowedToolsChoice changes ToolConfig without removing function_declarations."""
    lookup_schema = ToolSchema(
        name="lookup",
        description="Look up the city.",
        args_schema={"type": "object", "properties": {}},
    )
    config = _bound_config(
        _binding(
            tool_schemas=(_echo_schema(), lookup_schema),
            tool_choice=AllowedToolsChoice(mode="auto", tool_names=("lookup",)),
        )
    )
    assert config.tools is not None
    assert isinstance(config.tools[0], types.Tool)
    declarations = config.tools[0].function_declarations
    assert declarations is not None
    assert [declaration.name for declaration in declarations] == ["echo", "lookup"]
    assert config.tool_config == types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.VALIDATED,
            allowed_function_names=["lookup"],
        )
    )


def test_no_tools_binds_no_tool_config() -> None:
    """Without tool_schemas, neither tools nor tool_config is sent."""
    config = _bound_config(_binding())
    assert config.tools is None
    assert config.tool_config is None


def test_provider_executed_tools_reach_gemini_tools() -> None:
    """Gemini validates provider-executed mappings through GenerateContentConfig."""
    config = _bound_config(_binding(provider_executed_tools=({"google_search": {}},)))
    assert config.tools is not None
    assert isinstance(config.tools[0], types.Tool)
    assert config.tools[0].google_search is not None
    assert config.tool_config == types.ToolConfig(include_server_side_tool_invocations=True)


def test_provider_executed_tools_follow_function_tools_in_gemini() -> None:
    """Gemini places function declarations before provider-executed tools."""
    config = _bound_config(
        _binding(
            tool_schemas=(_echo_schema(),),
            provider_executed_tools=({"google_search": {}},),
        )
    )
    assert config.tools is not None
    assert isinstance(config.tools[0], types.Tool)
    assert config.tools[0].function_declarations is not None
    assert isinstance(config.tools[1], types.Tool)
    assert config.tools[1].google_search is not None
    assert config.tool_config == types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.VALIDATED
        ),
        include_server_side_tool_invocations=True,
    )


@pytest.mark.parametrize(
    ("provider_tool", "field_name"),
    [
        ({"code_execution": {}}, "code_execution"),
        ({"file_search": {"file_search_store_names": ["fileSearchStores/one"]}}, "file_search"),
        ({"google_maps": {}}, "google_maps"),
        ({"google_search": {}}, "google_search"),
        ({"url_context": {}}, "url_context"),
    ],
)
def test_every_supported_gemini_provider_field_binds(
    provider_tool: Mapping[str, object], field_name: str
) -> None:
    """Each reviewed Gemini provider field survives `types.Tool` normalization."""
    config = _bound_config(_binding(provider_executed_tools=(provider_tool,)))
    assert config.tools is not None
    assert getattr(config.tools[0], field_name) is not None


def test_one_gemini_mapping_can_populate_search_and_maps() -> None:
    """Validation inspects every populated `types.Tool` field."""
    config = _bound_config(
        _binding(provider_executed_tools=({"google_search": {}, "google_maps": {}},))
    )
    assert config.tools is not None
    assert isinstance(config.tools[0], types.Tool)
    assert config.tools[0].google_search is not None
    assert config.tools[0].google_maps is not None


@pytest.mark.parametrize(
    "provider_tool",
    [
        {},
        {"computer_use": {}},
        {"enterprise_web_search": {}},
        {"exa_ai_search": {}},
        {"function_declarations": []},
        {"google_search_retrieval": {}},
        {"mcp_servers": []},
        {"parallel_ai_search": {}},
        {"retrieval": {}},
    ],
)
def test_every_unsupported_gemini_provider_field_is_rejected(
    provider_tool: Mapping[str, object],
) -> None:
    """Every installed `types.Tool` field outside the reviewed set is rejected."""
    with pytest.raises(ValueError, match=r"unsupported|validation"):
        _ = _bound_config(_binding(provider_executed_tools=(provider_tool,)))


def test_gemini_image_search_is_rejected() -> None:
    """Google Search image results have separate unimplemented billing."""
    with pytest.raises(ValueError, match="image search"):
        _ = _bound_config(
            _binding(
                provider_executed_tools=(
                    {"google_search": {"search_types": {"image_search": {}}}},
                )
            )
        )


def test_gemini_provider_tools_reject_non_gemini_3_models() -> None:
    """Deprecated Gemini 2.5 models have no provider-executed tool path."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=False),
        model="gemini-2.5-flash",
        pricing=_PRICING,
        provider_name="gcp.gemini",
    )
    with pytest.raises(ValueError, match="Gemini 3"):
        _ = adapter.bind_text(_binding(provider_executed_tools=({"google_search": {}},)))


def test_gemini_vertex_rejects_provider_tools() -> None:
    """Vertex bindings reject provider-executed tools before requests."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=True),
        model="gemini-3.5-flash",
        pricing=_PRICING,
        provider_name="gcp.vertex_ai",
    )
    with pytest.raises(ValueError, match="Gemini Developer API"):
        _ = adapter.bind_text(_binding(provider_executed_tools=({"google_search": {}},)))


@pytest.mark.parametrize("provider_field", ["google_search", "google_maps"])
@pytest.mark.parametrize("rate", [None, True, math.nan, math.inf, -0.01])
def test_configured_gemini_charged_tool_requires_usable_pricing(
    provider_field: str, rate: float | None
) -> None:
    """Every supplied pricing table must price each configured charged tool."""
    pricing = {
        "ON_DEMAND": GeminiPricingTable(
            rates=_ON_DEMAND_RATES,
            google_search_usd_per_query=(rate if provider_field == "google_search" else 0.014),
            google_maps_usd_per_query=(rate if provider_field == "google_maps" else 0.014),
        )
    }
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=False),
        model="gemini-3.5-flash",
        pricing=pricing,
        provider_name="gcp.gemini",
    )
    with pytest.raises(ValueError, match=r"unavailable|finite and nonnegative"):
        _ = adapter.bind_text(_binding(provider_executed_tools=({provider_field: {}},)))


def test_provider_executed_tools_reject_non_auto_tool_choice() -> None:
    """Gemini ToolConfig cannot select provider-executed tools."""
    with pytest.raises(ValueError, match="tool_choice='auto'"):
        _ = _bound_config(
            _binding(
                provider_executed_tools=({"google_search": {}},),
                tool_choice="required",
            )
        )


def test_provider_executed_tools_reject_allowed_tools_choice() -> None:
    """allowed_function_names cannot restrict Gemini provider-executed tools."""
    with pytest.raises(ValueError, match="tool_choice='auto'"):
        _ = _bound_config(
            _binding(
                tool_schemas=(_echo_schema(),),
                provider_executed_tools=({"google_search": {}},),
                tool_choice=AllowedToolsChoice(mode="auto", tool_names=("echo",)),
            )
        )


def test_binding_maps_to_generation_fields() -> None:
    """The binding fields land as temperature and max_output_tokens. None omits either."""
    config = _bound_config(_binding(max_completion_tokens=64, temperature=0.5))
    assert config.temperature == 0.5
    assert config.max_output_tokens == 64
    defaulted = _bound_config(_binding())
    assert defaulted.temperature is None
    assert defaulted.max_output_tokens is None


def test_reasoning_level_maps_to_thinking_level() -> None:
    """The exact provider value reaches thinking_level with include_thoughts True."""
    config = _bound_config(_binding(reasoning_level="HIGH"))
    assert config.thinking_config == types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True
    )


def test_reasoning_level_rejects_sdk_normalization() -> None:
    """A value the Gemini SDK would change does not reach the request."""
    with pytest.raises(ValueError, match="normalizes it to 'HIGH'"):
        _ = _bound_config(_binding(reasoning_level="high"))


def test_reasoning_level_outside_the_sdk_enum_passes_through() -> None:
    """The exact provider value "XHIGH" reaches the wire unchanged."""
    with pytest.warns(UserWarning, match="XHIGH"):
        config = _bound_config(_binding(reasoning_level="XHIGH"))
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is not None
    assert config.thinking_config.thinking_level.value == "XHIGH"
    assert config.thinking_config.include_thoughts is True


def test_structured_bind_sends_the_response_schema() -> None:
    """The structured binding sends response_json_schema with the JSON mime type. text sends neither."""
    structured = _adapter().bind_structured(_binding(), _Answer)
    request = structured.build_request([UserMessage(content="hi")])
    assert isinstance(request, _GeminiRequestParams)
    assert request.config.response_mime_type == "application/json"
    assert request.config.response_json_schema == TypeAdapter(_Answer).json_schema()
    text_config = _bound_config(_binding())
    assert text_config.response_mime_type is None
    assert text_config.response_json_schema is None


def test_service_tier_lands_on_the_config() -> None:
    """The adapter's service_tier is sent on every request. None sends nothing."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=False),
        model="gemini-3.5-flash",
        pricing=_PRICING,
        provider_name="gcp.gemini",
        service_tier="flex",
    )
    request = adapter.bind_text(_binding()).build_request([UserMessage(content="hi")])
    assert isinstance(request, _GeminiRequestParams)
    assert request.config.service_tier == types.ServiceTier.FLEX
    assert _bound_config(_binding()).service_tier is None


@pytest.mark.parametrize(
    "extra_body",
    [
        {"contents": []},
        {"systemInstruction": {"parts": []}},
        {"systeminstruction": {"parts": []}},
        {"SYSTEM_INSTRUCTION": {"parts": []}},
        {"tool_config": {}},
        {"serviceTier": "flex"},
        {"generationConfig": {"temperature": 0.1}},
        {"generation_config": {"maxOutputTokens": 5}},
        {"generationconfig": {"temperature": 9.9}},
        {"generationConfig": {"response_json_schema": {}}},
        {"generationConfig": {"MAXOUTPUTTOKENS": 5}},
    ],
)
def test_extra_body_keys_the_adapter_populates_are_refused(
    extra_body: Mapping[str, object],
) -> None:
    """Binding rejects extra_body keys that duplicate adapter fields."""
    with pytest.raises(ValueError, match="collide"):
        _ = _adapter().bind_text(_binding(extra_body=extra_body))


@pytest.mark.parametrize("not_an_object", ["junk", None])
def test_extra_body_generation_config_must_be_an_object(not_an_object: object) -> None:
    """A non-object generationConfig, None included, would replace the adapter's own wholesale."""
    with pytest.raises(ValueError, match="object"):
        _ = _adapter().bind_text(_binding(extra_body={"generationConfig": not_an_object}))


def test_extra_body_generation_config_rejects_a_non_string_key() -> None:
    """Invalid `generationConfig` keys fail before collision detection."""
    generation_config: dict[object, object] = {1: "value", "temperature": 0.1}
    with pytest.raises(ValueError, match="only string keys"):
        _ = _adapter().bind_text(_binding(extra_body={"generationConfig": generation_config}))


def test_extra_body_passes_unpopulated_keys_through() -> None:
    """Unmapped wire fields stay reachable, delivered as per-request HttpOptions.extra_body."""
    extra_body = {"cachedContent": "caches/abc", "generationConfig": {"topK": 5}}
    config = _bound_config(_binding(extra_body=extra_body))
    assert isinstance(config.http_options, types.HttpOptions)
    assert config.http_options.extra_body == extra_body


# --- build_request ---


def test_user_message_forms() -> None:
    """UserMessage maps ImagePart and AudioPart to Part.inline_data.

    ImageUrlPart maps to Part.file_data.
    """
    request = _built_request([
        UserMessage(content="hi"),
        UserMessage(
            content=(
                TextPart(text="look:"),
                ImagePart(data=b"\x89PNG", media_type="image/png"),
                ImageUrlPart(url="gs://bucket/image.png", media_type="image/png"),
                AudioPart(data=b"WAV", media_type="audio/wav"),
                ImageUrlPart(url="https://example.com/image.png"),
            )
        ),
    ])
    assert request.contents[0] == types.Content(role="user", parts=[types.Part(text="hi")])
    assert request.contents[1] == types.Content(
        role="user",
        parts=[
            types.Part(text="look:"),
            types.Part(inline_data=types.Blob(data=b"\x89PNG", mime_type="image/png")),
            types.Part(
                file_data=types.FileData(
                    file_uri="gs://bucket/image.png",
                    mime_type="image/png",
                )
            ),
            types.Part(inline_data=types.Blob(data=b"WAV", mime_type="audio/wav")),
            types.Part(file_data=types.FileData(file_uri="https://example.com/image.png")),
        ],
    )


def test_tool_results_group_and_recover_names() -> None:
    """Consecutive tool results share one user Content, each naming the call it answers."""
    turn = AssistantMessage(
        turn=(
            ToolCall(id="call-a", name="f", args_json='{"x": 1}'),
            ToolCall(id="g", name="g", args_json="{}"),
        )
    )
    request = _built_request([
        UserMessage(content="go"),
        turn,
        ToolMessage(tool_call_id="call-a", content="ra"),
        ToolMessage(tool_call_id="g", content="rb", is_error=True),
        UserMessage(content="next"),
    ])
    assert [content.role for content in request.contents] == ["user", "model", "user", "user"]
    model_parts = request.contents[1].parts
    assert model_parts is not None
    assert model_parts[0].function_call == types.FunctionCall(id="call-a", name="f", args={"x": 1})
    assert model_parts[1].function_call == types.FunctionCall(id=None, name="g", args={})
    tool_parts = request.contents[2].parts
    assert tool_parts is not None
    assert tool_parts[0].function_response == types.FunctionResponse(
        id="call-a", name="f", response={"output": "ra"}
    )
    assert tool_parts[1].function_response == types.FunctionResponse(
        id=None, name="g", response={"error": "rb"}
    )


def test_tool_message_maps_image_part_image_url_part_and_audio_part() -> None:
    """ToolMessage maps ContentPart values to FunctionResponsePart fields."""
    request = _built_request([
        AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{}"),)),
        ToolMessage(
            tool_call_id="c",
            content=(
                TextPart(text="a"),
                ImagePart(data=b"IMG", media_type="image/png"),
                ImageUrlPart(url="gs://bucket/image.png", media_type="image/png"),
                AudioPart(data=b"WAV", media_type="audio/wav"),
                TextPart(text="b"),
            ),
        ),
    ])
    tool_parts = request.contents[1].parts
    assert tool_parts is not None
    function_response = tool_parts[0].function_response
    assert function_response is not None
    assert function_response.response == {"output": "ab"}
    assert function_response.parts == [
        types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(data=b"IMG", mime_type="image/png")
        ),
        types.FunctionResponsePart(
            file_data=types.FunctionResponseFileData(
                file_uri="gs://bucket/image.png",
                mime_type="image/png",
            )
        ),
        types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(data=b"WAV", mime_type="audio/wav")
        ),
    ]


def test_unmatched_tool_call_id_is_invalid() -> None:
    """The wire requires the function name, recoverable only from the call the id answers."""
    invalid = _invalid_request([ToolMessage(tool_call_id="ghost", content="r")])
    assert "ghost" in invalid.reason


def test_unparseable_args_json_is_invalid() -> None:
    """The wire field holds the parsed arguments object, so text that is not JSON has nowhere to go."""
    invalid = _invalid_request([
        AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{not json"),))
    ])
    assert "args_json" in invalid.reason


def test_non_object_args_json_is_invalid() -> None:
    """JSON that is not an object has no FunctionCall.args form."""
    invalid = _invalid_request([
        AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="[1]"),))
    ])
    assert "JSON object" in invalid.reason


@pytest.mark.parametrize(
    ("part", "message_class"),
    [
        (TextPart(text="a", cache_breakpoint=True), UserMessage),
        (TextPart(text="a", cache_breakpoint=True), ToolMessage),
        (
            ImageUrlPart(
                url="gs://bucket/image.png",
                cache_breakpoint=True,
            ),
            UserMessage,
        ),
        (
            AudioPart(
                data=b"WAV",
                media_type="audio/wav",
                cache_breakpoint=True,
            ),
            UserMessage,
        ),
    ],
)
def test_cache_breakpoint_on_content_part_is_invalid(
    part: ContentPart, message_class: type[UserMessage] | type[ToolMessage]
) -> None:
    """cache_breakpoint has no Gemini wire form in UserMessage or ToolMessage."""
    if message_class is UserMessage:
        messages: Sequence[Message] = [UserMessage(content=(part,))]
    else:
        messages = [
            AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{}"),)),
            ToolMessage(tool_call_id="c", content=(part,)),
        ]
    invalid = _invalid_request(messages)
    assert "cache_breakpoint" in invalid.reason
    assert type(part).__name__ in invalid.reason
    assert message_class.__name__ in invalid.reason


def test_a_cross_provider_reasoning_part_is_invalid() -> None:
    """A foreign ReasoningPart fails when ReasoningPart.raw cannot restore a Gemini Part."""
    invalid = _invalid_request([
        AssistantMessage(
            turn=(
                ReasoningPart(
                    raw={"type": "thinking", "thinking": "x", "signature": "s"}, text="x"
                ),
            )
        )
    ])
    assert "ReasoningPart" in invalid.reason


def test_empty_assistant_text_is_skipped_on_replay() -> None:
    """An empty TextPart puts nothing on the wire."""
    request = _built_request([AssistantMessage(turn=(TextPart(text=""), TextPart(text="kept")))])
    assert request.contents[0].parts == [types.Part(text="kept")]


# --- the thought-signature pairing ---


def _interpreted_turn(response: types.GenerateContentResponse) -> AssistantMessage:
    """Interpret under the text binding and return the turn."""
    outcome = _adapter().bind_text(_binding()).interpret(response)
    assert isinstance(outcome, AdapterResult)
    return outcome.assistant_message


def test_a_signed_function_call_yields_a_reasoning_part_and_call() -> None:
    """ReasoningPart.raw preserves the signature. ToolCall remains dispatchable."""
    original = types.Part(
        thought_signature=b"\x00\x01sig",
        function_call=types.FunctionCall(name="f", args={"x": 1}),
    )
    turn = _interpreted_turn(_response([original]))
    reasoning_part, tool_call = turn.turn
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.raw == original.model_dump(mode="json", exclude_none=True)
    assert tool_call == ToolCall(id="f", name="f", args_json='{"x": 1}')
    request = _built_request([UserMessage(content="go"), turn])
    model_parts = request.contents[1].parts
    assert model_parts == [original]


def test_signed_answer_text_yields_a_reasoning_part_and_text_part() -> None:
    """Non-thought text carrying a signature stays readable as answer text and replays signed."""
    original = types.Part(text="final answer", thought_signature=b"sig")
    turn = _interpreted_turn(_response([original]))
    reasoning_part, text_part = turn.turn
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.text is None
    assert text_part == TextPart(text="final answer")
    assert turn.text == "final answer"
    request = _built_request([UserMessage(content="go"), turn])
    assert request.contents[1].parts == [original]


def test_a_thought_part_replays_byte_identical() -> None:
    """The signature bytes survive the JSON round trip through ReasoningPart.raw."""
    original = types.Part(thought=True, text="reasoning", thought_signature=b"\x00\xffsig")
    turn = _interpreted_turn(_response([original, types.Part(text="answer")]))
    request = _built_request([UserMessage(content="go"), turn])
    model_parts = request.contents[1].parts
    assert model_parts is not None
    assert model_parts[0] == original
    assert model_parts[0].thought_signature == b"\x00\xffsig"


def test_an_executable_code_part_becomes_a_raw_part_and_replays_as_itself() -> None:
    """An executable_code Part becomes RawPart and returns unchanged.

    The billed raw part remains in the turn for replay.
    """
    original = types.Part(
        executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON)
    )
    turn = _interpreted_turn(_response([original, types.Part(text="answer")]))
    raw_part, text_part = turn.turn
    assert isinstance(raw_part, RawPart)
    assert raw_part.raw == original.model_dump(mode="json", exclude_none=True)
    assert text_part == TextPart(text="answer")
    request = _built_request([UserMessage(content="go"), turn])
    assert request.contents[1].parts == [original, types.Part(text="answer")]


def test_an_empty_text_beside_a_payload_still_becomes_a_raw_part() -> None:
    """A Part with empty text and executable_code becomes RawPart.

    Reading that field as present rather than as non-empty drops the whole part.
    """
    original = types.Part(
        text="",
        executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON),
    )
    turn = _interpreted_turn(_response([original]))
    (raw_part,) = turn.turn
    assert isinstance(raw_part, RawPart)
    assert raw_part.raw == original.model_dump(mode="json", exclude_none=True)
    request = _built_request([UserMessage(content="go"), turn])
    assert request.contents[1].parts == [original]


def test_thought_text_is_reasoning_part_text_and_not_output() -> None:
    """Thought text reaches ReasoningPart.text and stays outside output."""
    turn = _interpreted_turn(
        _response([
            types.Part(thought=True, text="thinking..."),
            types.Part(text="answer"),
        ])
    )
    reasoning_part = turn.turn[0]
    assert isinstance(reasoning_part, ReasoningPart)
    assert reasoning_part.text == "thinking..."
    assert turn.text == "answer"


# --- as_json ---


def test_as_json_holds_the_request_without_transport_config() -> None:
    """The archive cell carries model, contents, config, and extra_body. http_options stays out."""
    request = _built_request(
        [UserMessage(content="hi")],
        _binding(
            temperature=0.5,
            extra_body={"cachedContent": "caches/abc"},
        ),
    )
    body = json.loads(request.as_json())
    assert body["model"] == "gemini-3.5-flash"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["config"]["temperature"] == 0.5
    assert "http_options" not in body["config"]
    assert body["extra_body"] == {"cachedContent": "caches/abc"}


# --- interpret ---


def test_text_binding_reads_stop_reasons() -> None:
    """STOP is end_turn or tool_use by the turn's calls. MAX_TOKENS and SAFETY name themselves."""
    bound = _adapter().bind_text(_binding())
    ended = bound.interpret(_response([types.Part(text="hi")]))
    assert isinstance(ended, AdapterResult)
    assert (ended.output, ended.stop_reason) == ("hi", "end_turn")
    called = bound.interpret(
        _response([types.Part(function_call=types.FunctionCall(name="f", args={}))])
    )
    assert isinstance(called, AdapterResult)
    assert called.stop_reason == "tool_use"
    truncated = bound.interpret(
        _response([types.Part(text="par")], finish_reason=types.FinishReason.MAX_TOKENS)
    )
    assert isinstance(truncated, AdapterResult)
    assert (truncated.output, truncated.stop_reason) == ("par", "max_tokens")
    refused = bound.interpret(_response(None, finish_reason=types.FinishReason.SAFETY))
    assert isinstance(refused, AdapterResult)
    assert (refused.output, refused.stop_reason) == ("", "refusal")
    other = bound.interpret(
        _response([types.Part(text="?")], finish_reason=types.FinishReason.LANGUAGE)
    )
    assert isinstance(other, AdapterResult)
    assert other.stop_reason == "other"


def test_both_bindings_report_a_missing_finish_reason_as_unfinished() -> None:
    """A candidate without a finish_reason is a turn that never closed, its partial turn carried."""
    response = _response([types.Part(text="par")], finish_reason=None)
    text_outcome = _adapter().bind_text(_binding()).interpret(response)
    assert isinstance(text_outcome, UnfinishedTurn)
    assert text_outcome.assistant_message.text == "par"
    structured_outcome = _adapter().bind_structured(_binding(), _Answer).interpret(response)
    assert isinstance(structured_outcome, UnfinishedTurn)


def test_no_candidates_reads_the_block_reason() -> None:
    """A blocked prompt is a Refusal with an empty turn. No candidates and no block is unfinished."""
    bound = _adapter().bind_text(_binding())
    blocked = bound.interpret(
        _response(None, finish_reason=None, block_reason=types.BlockedReason.SAFETY)
    )
    assert isinstance(blocked, Refusal)
    assert blocked.assistant_message.turn == ()
    silent = bound.interpret(_response(None, finish_reason=None))
    assert isinstance(silent, UnfinishedTurn)


def test_structured_binding_outcomes() -> None:
    """The structured matrix: instance, tool-call None, refusal, truncation, violation, empty, unfinished."""
    bound = _adapter().bind_structured(_binding(), _Answer)
    parsed = bound.interpret(_response([types.Part(text='{"value": 3}')]))
    assert isinstance(parsed, AdapterResult)
    assert parsed.output == _Answer(value=3)
    tool_turn = bound.interpret(
        _response([types.Part(function_call=types.FunctionCall(name="f", args={}))])
    )
    assert isinstance(tool_turn, AdapterResult)
    assert tool_turn.output is None
    assert tool_turn.stop_reason == "tool_use"
    refused = bound.interpret(_response(None, finish_reason=types.FinishReason.SAFETY))
    assert isinstance(refused, Refusal)
    truncated = bound.interpret(
        _response([types.Part(text='{"value"')], finish_reason=types.FinishReason.MAX_TOKENS)
    )
    assert isinstance(truncated, MaxCompletionTokensExceeded)
    violated = bound.interpret(_response([types.Part(text='{"value": "not int"}')]))
    assert isinstance(violated, SchemaViolation)
    assert "value" in violated.validation_error_json
    empty = bound.interpret(_response([]))
    assert isinstance(empty, EmptyTurn)
    unfinished = bound.interpret(
        _response([types.Part(text="?")], finish_reason=types.FinishReason.LANGUAGE)
    )
    assert isinstance(unfinished, UnfinishedTurn)
    assert "LANGUAGE" in unfinished.reason


def test_structured_output_may_inherit_no_output() -> None:
    """AdapterResult distinguishes successful output from NoOutput."""

    class ReportAlsoNoOutput(BaseModel, NoOutput):
        assistant_message: AssistantMessage = AssistantMessage(turn=())
        value: int

    bound = _adapter().bind_structured(_binding(), ReportAlsoNoOutput)
    outcome = bound.interpret(_response([types.Part(text='{"value": 3}')]))
    assert isinstance(outcome, AdapterResult)
    assert outcome.output == ReportAlsoNoOutput(value=3)


def test_a_structured_turn_ignores_thought_text_when_validating() -> None:
    """Only non-thought text is the candidate instance."""
    bound = _adapter().bind_structured(_binding(), _Answer)
    outcome = bound.interpret(
        _response([
            types.Part(thought=True, text="let me think"),
            types.Part(text='{"value": 7}'),
        ])
    )
    assert isinstance(outcome, AdapterResult)
    assert outcome.output == _Answer(value=7)


def test_identity_reads_the_response_fields() -> None:
    """model_version and response_id report as-is, absent ones as the empty string."""
    bound = _adapter().bind_text(_binding())
    identity = bound.identity_from_raw(_response([types.Part(text="hi")]), request_id="req-gemini")
    assert (identity.model_served, identity.response_id, identity.request_id) == (
        "gemini-3.5-flash",
        "resp-1",
        "req-gemini",
    )
    bare = bound.identity_from_raw(types.GenerateContentResponse(), request_id=None)
    assert (bare.model_served, bare.response_id) == ("", "")


# --- billing ---


def test_the_usage_partition() -> None:
    """cache_read is the cached counter, cache_none the remainder plus tool-use, output includes thoughts."""
    billing = _billing_from_usage(
        _usage_metadata(), _PRICING, provider_executed_tool_cost_in_usd=0.0
    )
    usage = billing.usage
    assert usage.input_tokens_cache_read == 40
    assert usage.input_tokens_cache_none == 80
    assert usage.input_tokens_cache_write == 0
    assert usage.output_tokens == 60
    assert usage.output_tokens_reasoning == 10
    assert usage.input_tokens_cache_read_cost_in_usd == pytest.approx(40 * 0.1 / 1_000_000)
    assert usage.input_tokens_cache_none_cost_in_usd == pytest.approx(80 * 1.0 / 1_000_000)
    assert usage.input_tokens_cache_write_cost_in_usd == 0.0
    assert usage.output_tokens_cost_in_usd == pytest.approx(60 * 10.0 / 1_000_000)
    assert billing.service_tier == "ON_DEMAND"


def test_search_queries_deduplicate_across_candidates_and_ignore_empty_strings() -> None:
    """Gemini bills unique nonempty Search queries across the complete response."""
    first_candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=_provider_call_parts(
                tool_call_id="search-1",
                tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                queries=["first", "", "second"],
            ),
        ),
        finish_reason=types.FinishReason.STOP,
    )
    second_candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=_provider_call_parts(
                tool_call_id="search-2",
                tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                queries=["second", "third"],
            ),
        ),
        finish_reason=types.FinishReason.STOP,
    )
    response = types.GenerateContentResponse(
        candidates=[first_candidate, second_candidate],
        usage_metadata=_usage_metadata(),
        model_version="gemini-3.5-flash",
        response_id="response-1",
    )
    usage = _billing_from_response(
        response, _PRICING, configured_fields=frozenset({"google_search"})
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(3 * 0.014)


def test_mixed_search_and_maps_calls_use_separate_counting_rules() -> None:
    """Search deduplicates queries while Maps counts every returned query entry."""
    pricing = {
        "ON_DEMAND": GeminiPricingTable(
            rates=_ON_DEMAND_RATES,
            google_search_usd_per_query=0.01,
            google_maps_usd_per_query=0.02,
        )
    }
    parts = [
        *_provider_call_parts(
            tool_call_id="search-1",
            tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
            queries=["same", "same", "different"],
        ),
        *_provider_call_parts(
            tool_call_id="maps-1",
            tool_type=types.ToolType.GOOGLE_MAPS,
            queries=["same", "same", "different"],
        ),
    ]
    usage = _billing_from_response(
        _response(parts, usage_metadata=_usage_metadata()),
        pricing,
        configured_fields=frozenset({"google_search", "google_maps"}),
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(2 * 0.01 + 3 * 0.02)


def test_free_gemini_provider_tools_add_no_separate_fee() -> None:
    """Code execution, URL context, and file search add no separate fee."""
    parts = [
        types.Part(
            executable_code=types.ExecutableCode(
                code="print(1)",
                language=types.Language.PYTHON,
            )
        ),
        *_provider_call_parts(
            tool_call_id="url-1",
            tool_type=types.ToolType.URL_CONTEXT,
            queries=[],
        ),
        *_provider_call_parts(
            tool_call_id="file-1",
            tool_type=types.ToolType.FILE_SEARCH,
            queries=[],
        ),
    ]
    usage = _billing_from_response(
        _response(parts, usage_metadata=_usage_metadata()),
        _PRICING,
        configured_fields=frozenset({"code_execution", "url_context", "file_search"}),
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == 0.0


def test_gemini_charged_tool_at_an_unpriced_tier_produces_nan() -> None:
    """A charged query uses `_UNPRICED` for an unknown served tier."""
    response = _response(
        _provider_call_parts(
            tool_call_id="search-1",
            tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
            queries=["query"],
        ),
        usage_metadata=_usage_metadata(traffic_type=types.TrafficType.PROVISIONED_THROUGHPUT),
    )
    usage = _billing_from_response(
        response,
        _PRICING,
        configured_fields=frozenset({"google_search"}),
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


@pytest.mark.parametrize(
    ("tool_type", "queries"),
    [
        (types.ToolType.GOOGLE_SEARCH_WEB, ["valid", 1]),
        (types.ToolType.GOOGLE_SEARCH_WEB, "valid"),
        (types.ToolType.GOOGLE_MAPS, [""]),
        (types.ToolType.GOOGLE_MAPS, ["valid", 1]),
    ],
)
def test_malformed_gemini_query_evidence_produces_nan(
    tool_type: types.ToolType, queries: object
) -> None:
    """Unexpected query shapes cannot produce an exact provider-executed cost."""
    field = "google_search" if tool_type == types.ToolType.GOOGLE_SEARCH_WEB else "google_maps"
    usage = _billing_from_response(
        _response(
            _provider_call_parts(tool_call_id="call-1", tool_type=tool_type, queries=queries),
            usage_metadata=_usage_metadata(),
        ),
        _PRICING,
        configured_fields=frozenset({field}),
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


def test_incomplete_gemini_tool_pair_produces_nan() -> None:
    """A server call without its matching response is incomplete billing evidence."""
    part = types.Part(
        tool_call=types.ToolCall(
            id="search-1",
            tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
            args={"queries": ["query"]},
        )
    )
    usage = _billing_from_response(
        _response([part], usage_metadata=_usage_metadata()),
        _PRICING,
        configured_fields=frozenset({"google_search"}),
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


def test_mismatched_gemini_tool_pair_produces_nan() -> None:
    """Server call and response identifiers and tool types must both match."""
    parts = _provider_call_parts(
        tool_call_id="search-1",
        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
        queries=["query"],
    )
    assert parts[1].tool_response is not None
    parts[1].tool_response.id = "search-2"
    usage = _billing_from_response(
        _response(parts, usage_metadata=_usage_metadata()),
        _PRICING,
        configured_fields=frozenset({"google_search"}),
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


def test_truncated_gemini_search_billing_produces_nan() -> None:
    """A partial response cannot prove the final Search query count."""
    usage = _billing_from_response(
        _response([], usage_metadata=_usage_metadata()),
        _PRICING,
        configured_fields=frozenset({"google_search"}),
        billing_complete=False,
    ).usage
    assert math.isnan(usage.provider_executed_tool_cost_in_usd)


def test_the_long_prompt_threshold_reprices_every_category() -> None:
    """Above the threshold the long rates price. At or below it the base rates do."""
    short = _LONG_PROMPT_TABLE.price(
        service_tier="ON_DEMAND",
        usage_raw=None,
        prompt_token_count=200,
        input_tokens_cache_read=100,
        input_tokens_cache_none=100,
        output_tokens=10,
        output_tokens_reasoning=0,
        provider_executed_tool_cost_in_usd=0.0,
    ).billing
    assert short.input_cache_none_usd_per_million_tokens == 1.0
    long = _LONG_PROMPT_TABLE.price(
        service_tier="ON_DEMAND",
        usage_raw=None,
        prompt_token_count=201,
        input_tokens_cache_read=100,
        input_tokens_cache_none=101,
        output_tokens=10,
        output_tokens_reasoning=0,
        provider_executed_tool_cost_in_usd=0.0,
    ).billing
    assert long.input_cache_none_usd_per_million_tokens == 2.0
    assert long.cache_read_usd_per_million_tokens == 0.2
    assert long.output_usd_per_million_tokens == 20.0


def test_tool_execution_input_does_not_cross_the_long_prompt_threshold() -> None:
    """The threshold reads prompt_token_count, which excludes the tool-execution input priced beside it."""
    billing = _billing_from_usage(
        _usage_metadata(prompt_token_count=200, tool_use_prompt_token_count=50),
        {"ON_DEMAND": _LONG_PROMPT_TABLE},
        provider_executed_tool_cost_in_usd=0.0,
    )
    assert billing.usage.input_tokens_cache_none == 210
    assert billing.input_cache_none_usd_per_million_tokens == 1.0


def test_the_long_prompt_fields_are_required_together() -> None:
    """A threshold without rates prices nothing, and rates without a threshold never apply."""
    with pytest.raises(ValueError, match="together"):
        _ = GeminiPricingTable(
            rates=_ON_DEMAND_RATES,
            google_search_usd_per_query=0.014,
            google_maps_usd_per_query=0.014,
            long_prompt_threshold_tokens=200,
        )
    with pytest.raises(ValueError, match="together"):
        _ = GeminiPricingTable(
            rates=_ON_DEMAND_RATES,
            google_search_usd_per_query=0.014,
            google_maps_usd_per_query=0.014,
            long_prompt_rates=_ON_DEMAND_RATES,
        )


def test_traffic_type_selects_the_table() -> None:
    """A reported tier prices at its own table. UNSPECIFIED and None price at ON_DEMAND."""
    flex_rates = GeminiRates(
        input_cache_none_usd_per_million_tokens=0.5,
        cache_read_usd_per_million_tokens=0.05,
        output_usd_per_million_tokens=5.0,
    )
    pricing = {
        **_PRICING,
        "ON_DEMAND_FLEX": GeminiPricingTable(
            rates=flex_rates,
            google_search_usd_per_query=0.014,
            google_maps_usd_per_query=0.014,
        ),
    }
    flexed = _billing_from_usage(
        _usage_metadata(traffic_type=types.TrafficType.ON_DEMAND_FLEX),
        pricing,
        provider_executed_tool_cost_in_usd=0.0,
    )
    assert flexed.service_tier == "ON_DEMAND_FLEX"
    assert flexed.input_cache_none_usd_per_million_tokens == 0.5
    unspecified = _billing_from_usage(
        _usage_metadata(traffic_type=types.TrafficType.TRAFFIC_TYPE_UNSPECIFIED),
        pricing,
        provider_executed_tool_cost_in_usd=0.0,
    )
    assert unspecified.service_tier == "ON_DEMAND"
    assert unspecified.input_cache_none_usd_per_million_tokens == 1.0


# --- streaming ---


def test_items_translate_parts_with_reasoning_separators() -> None:
    """Thought text streams as deltas. Each part boundary emits a separator. Answer text streams bare."""
    items = _drained(
        _gemini_stream([
            _response([types.Part(thought=True, text="think a")], finish_reason=None),
            _response(
                [types.Part(thought=True, text=" more", thought_signature=b"s1")],
                finish_reason=None,
            ),
            _response([types.Part(thought=True, text="part two")], finish_reason=None),
            _response(
                [
                    types.Part(text="answer"),
                    types.Part(function_call=types.FunctionCall(id="c1", name="f", args={"x": 1})),
                ],
                finish_reason=types.FinishReason.STOP,
            ),
        ])
    )
    assert items == [
        ReasoningDelta(text="think a"),
        ReasoningDelta(text=" more"),
        ReasoningDelta(text=REASONING_PART_SEPARATOR),
        ReasoningDelta(text="part two"),
        "answer",
        ToolCall(id="c1", name="f", args_json='{"x": 1}'),
    ]


def test_two_thought_parts_in_one_chunk_are_separated() -> None:
    """Parts arriving as separate entries of one chunk's list are distinct parts."""
    items = _drained(
        _gemini_stream([
            _response(
                [types.Part(thought=True, text="one"), types.Part(thought=True, text="two")],
                finish_reason=types.FinishReason.STOP,
            )
        ])
    )
    assert items == [
        ReasoningDelta(text="one"),
        ReasoningDelta(text=REASONING_PART_SEPARATOR),
        ReasoningDelta(text="two"),
    ]


def test_assembly_merges_text_slices_and_signatures_end_parts() -> None:
    """final() reads the same turn a whole response would carry."""
    chunks = [
        _response([types.Part(thought=True, text="think a")], finish_reason=None),
        _response(
            [types.Part(thought=True, text=" more", thought_signature=b"s1")], finish_reason=None
        ),
        _response([types.Part(text="ans")], finish_reason=None),
        _response(
            [types.Part(text="wer")],
            finish_reason=types.FinishReason.STOP,
            usage_metadata=_usage_metadata(),
            grounding_metadata=types.GroundingMetadata(web_search_queries=["weather"]),
        ),
    ]
    assembled = assembled_response(chunks)
    assert assembled == _response(
        [
            types.Part(thought=True, text="think a more", thought_signature=b"s1"),
            types.Part(text="answer"),
        ],
        usage_metadata=_usage_metadata(),
        grounding_metadata=types.GroundingMetadata(web_search_queries=["weather"]),
    )


def test_a_mid_stream_error_propagates_from_items() -> None:
    """The SDK iterator's APIError reaches the retry loop unchanged."""

    async def failing_iterator() -> AsyncIterator[types.GenerateContentResponse]:
        yield _response([types.Part(text="he")], finish_reason=None)
        raise errors.APIError(503, {"error": {"code": 503, "message": "overloaded"}})

    stream = _GeminiStream(chunks=failing_iterator(), pricing=_PRICING)
    with pytest.raises(errors.APIError):
        _ = _drained(stream)


def test_a_blocked_prompt_stream_ends_cleanly_and_interprets_as_refusal() -> None:
    """A block_reason is a terminal event: no protocol error, and the final response is a Refusal."""
    stream = _gemini_stream([
        _response(None, finish_reason=None, block_reason=types.BlockedReason.SAFETY)
    ])
    assert _drained(stream) == []

    async def final() -> types.GenerateContentResponse:
        return await stream.final()

    outcome = _adapter().bind_text(_binding()).interpret(asyncio.run(final()))
    assert isinstance(outcome, Refusal)


def test_billing_reported_follows_usage_arrival() -> None:
    """Return Billing only after usage_metadata arrives."""

    async def scenario() -> tuple[ProviderBilling | None, ProviderBilling | None]:
        stream = _gemini_stream([
            _response([types.Part(text="he")], finish_reason=None),
            _response(
                [types.Part(text="y")],
                finish_reason=types.FinishReason.STOP,
                usage_metadata=_usage_metadata(),
            ),
        ])
        items = stream.items()
        _ = await anext(items)
        before = stream.billing_reported()
        _ = [item async for item in items]
        return before, stream.billing_reported()

    before, after = asyncio.run(scenario())
    assert before is None
    assert after is not None
    assert after.billing.usage.output_tokens == 60
    assert after.billing.service_tier == "ON_DEMAND"


def test_cutoff_gemini_provider_tool_billing_is_nan() -> None:
    """A charged query cannot report zero before terminal usage arrives."""

    async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
        responses: tuple[types.GenerateContentResponse, ...] = ()
        for response in responses:
            yield response

    async def scenario() -> ProviderBilling | None:
        stream = _GeminiStream(
            chunks=chunks(),
            pricing=_PRICING,
            provider_tool_fields=frozenset({"google_search"}),
            first_chunk=_response(
                [
                    *_provider_call_parts(
                        tool_call_id="search-1",
                        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                        queries=["query"],
                    ),
                    types.Part(text="partial"),
                ],
                finish_reason=None,
            ),
        )
        items = stream.items()
        assert await anext(items) == "partial"
        billing = stream.billing_reported()
        await stream.close()
        return billing

    billing = asyncio.run(scenario())
    assert billing is not None
    assert math.isnan(billing.billing.usage.provider_executed_tool_cost_in_usd)


def test_stream_billing_collects_every_candidate_provider_query() -> None:
    """Stream billing collects Search queries from every candidate."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=_provider_call_parts(
                        tool_call_id="search-1",
                        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                        queries=["first"],
                    ),
                ),
                finish_reason=types.FinishReason.STOP,
            ),
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=_provider_call_parts(
                        tool_call_id="search-2",
                        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                        queries=["second"],
                    ),
                ),
                finish_reason=types.FinishReason.STOP,
            ),
        ],
        usage_metadata=_usage_metadata(),
    )

    async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
        yield response

    async def scenario() -> ProviderBilling | None:
        stream = _GeminiStream(
            chunks=chunks(),
            pricing=_PRICING,
            provider_tool_fields=frozenset({"google_search"}),
        )
        _ = [item async for item in stream.items()]
        return stream.billing_reported()

    billing = asyncio.run(scenario())
    assert billing is not None
    assert billing.billing.usage.provider_executed_tool_cost_in_usd == pytest.approx(0.028)


def test_close_closes_the_sdk_iterator() -> None:
    """close() calls the async generator's aclose, so the connection is released."""
    closed = False

    async def chunk_iterator() -> AsyncIterator[types.GenerateContentResponse]:
        nonlocal closed
        try:
            yield _response([types.Part(text="he")], finish_reason=None)
        finally:
            closed = True

    async def scenario() -> None:
        stream = _GeminiStream(chunks=chunk_iterator(), pricing=_PRICING)
        items = stream.items()
        _ = await anext(items)
        await stream.close()

    asyncio.run(scenario())
    assert closed


def test_open_stream_performs_the_connection_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_stream pulls and preserves the first chunk."""
    adapter = _adapter()
    bound = adapter.bind_text(_binding())
    request = bound.build_request([UserMessage(content="hi")])
    assert isinstance(request, _GeminiRequestParams)

    async def failing_sdk_stream(
        *, model: str, contents: object, config: object
    ) -> AsyncIterator[types.GenerateContentResponse]:
        assert (model, contents, config) == (request.model, request.contents, request.config)

        async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
            raise httpx.ConnectError("no route")
            yield _response([])  # Unreachable. The yield makes chunks an async generator.

        return chunks()

    async def working_sdk_stream(
        *, model: str, contents: object, config: object
    ) -> AsyncIterator[types.GenerateContentResponse]:
        assert (model, contents, config) == (request.model, request.contents, request.config)

        async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
            yield _response([types.Part(text="he")], finish_reason=None)
            yield _response([types.Part(text="y")])

        return chunks()

    async def scenario() -> list[StreamItem]:
        monkeypatch.setattr(
            adapter.client.aio.models, "generate_content_stream", failing_sdk_stream
        )
        with pytest.raises(httpx.ConnectError):
            _ = await bound.open_stream(request)
        monkeypatch.setattr(
            adapter.client.aio.models, "generate_content_stream", working_sdk_stream
        )
        stream = await bound.open_stream(request)
        return [item async for item in stream.items()]

    assert asyncio.run(scenario()) == ["he", "y"]


# --- conformance ---


def _executable_code_part() -> types.Part:
    """One code-execution part without another TurnPart variant."""
    return types.Part(
        executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON)
    )


def _reasoning_turn_response(
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
) -> types.GenerateContentResponse:
    """Build reasoning, executable-code, and text parts."""
    return _response(
        [
            types.Part(thought=True, text="check first", thought_signature=b"\x00\x01sig"),
            _executable_code_part(),
            types.Part(text="hello"),
        ],
        usage_metadata=usage_metadata,
    )


class TestGeminiGenerateContentConformance(AdapterConformance):
    """The neutral invariants, over the Gemini generateContent adapter's own SDK objects."""

    @override
    def make_adapter(self) -> Adapter:
        """Build the adapter these invariants run against, priced for on-demand alone."""
        return _adapter()

    @override
    def response_with_cache_writes(self) -> BaseModel:
        """Return a turn whose usage fills every counter Gemini bills. Writes are always zero."""
        return _reasoning_turn_response(_usage_metadata())

    @override
    def response_without_usage(self) -> BaseModel:
        """Return a turn carrying no usage_metadata at all."""
        return _reasoning_turn_response(None)

    @override
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return a turn served at PROVISIONED_THROUGHPUT, which _PRICING holds no table for."""
        return _reasoning_turn_response(
            _usage_metadata(traffic_type=types.TrafficType.PROVISIONED_THROUGHPUT)
        )

    @override
    def response_with_impossible_counters(self) -> BaseModel:
        """Return a turn whose cached counter exceeds the prompt total it is a share of."""
        return _reasoning_turn_response(
            _usage_metadata(prompt_token_count=100, cached_content_token_count=200)
        )

    @override
    def response_with_reasoning(self) -> BaseModel:
        """Return the reasoning turn with signature bytes as the payload."""
        return _reasoning_turn_response(_usage_metadata())

    @override
    def response_with_raw_part(self) -> BaseModel | None:
        """Return the turn whose middle part carries executable_code."""
        return _reasoning_turn_response(_usage_metadata())

    @override
    def assistant_wire_parts(self, request: RequestParams) -> Sequence[object]:
        """Read the parts of the model Content this request ends with, as their JSON dumps."""
        assert isinstance(request, _GeminiRequestParams)
        parts = request.contents[-1].parts
        assert parts is not None
        return [part.model_dump(mode="json", exclude_none=True) for part in parts]

    @override
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return one turn as assembled_response builds it and as a whole response.

        Assembly preserves terminal executable_code and merged text.
        """
        chunks = [
            _response([types.Part(text="Hel")], finish_reason=None),
            _response([types.Part(text="lo")], finish_reason=None),
            _response(
                [_executable_code_part()],
                finish_reason=types.FinishReason.STOP,
                usage_metadata=_usage_metadata(),
                grounding_metadata=types.GroundingMetadata(web_search_queries=["weather"]),
            ),
        ]
        whole = _response(
            [types.Part(text="Hello"), _executable_code_part()],
            usage_metadata=_usage_metadata(),
            grounding_metadata=types.GroundingMetadata(web_search_queries=["weather"]),
        )
        return assembled_response(chunks), whole

    @override
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream ending with neither a finish_reason nor a block_reason."""
        return _gemini_stream([_response([types.Part(text="he")], finish_reason=None)])

    @override
    def sdk_errors_and_classifications(self) -> Mapping[Exception, ErrorClassification]:
        """Return Gemini error classification cases."""
        return {
            httpx.ConnectError("no route"): "transient",
            httpx.ReadTimeout("slow"): "transient",
            _api_error(400): "invalid_request",
            _api_error(403): "invalid_request",
            _api_error(404): "invalid_request",
            _api_error(429): "invalid_request",
            _api_error(500): "unknown_exception",
            _api_error(503): "unknown_exception",
            ValueError("boom"): "unknown_exception",
        }

    @override
    def sdk_errors_and_verdicts(self) -> Mapping[Exception, Verdict]:
        """Return Gemini error verdict cases."""
        return {
            _api_error(429, headers={"retry-after": "7"}): PauseAll(retry_after=7.0),
            _api_error(429, retry_delay="32s"): PauseAll(retry_after=32.0),
            _api_error(503): PauseAll(retry_after=None),
            _api_error(408): RetryThisOne(retry_after=None),
            _api_error(500): RetryThisOne(retry_after=None),
            _api_error(502): RetryThisOne(retry_after=None),
            _api_error(504): RetryThisOne(retry_after=None),
            _api_error(400): DoNotRetry(),
            _api_error(403): DoNotRetry(),
            _api_error(404): DoNotRetry(),
            _api_error(418): DoNotRetry(),
            _api_error(599): RetryThisOne(retry_after=None),
            TransientError("throttled body", retry_after_seconds=3.0, is_rate_limit=True): (
                PauseAll(retry_after=3.0)
            ),
            TransientError("failed body"): RetryThisOne(retry_after=None),
        }
