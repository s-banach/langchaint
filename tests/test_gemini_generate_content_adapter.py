"""Test Gemini generateContent with constructed SDK objects.

Tests cover Usage, stop reasons, reasoning, tool calls, streams, and errors.
"""

import json
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import NamedTuple, override

import httpx
import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, JsonValue, TypeAdapter

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
    TurnPart,
    UserMessage,
)
from langchaint.adapter import (
    REASONING_PART_SEPARATOR,
    Adapter,
    AdapterStream,
    Binding,
    BoundAdapter,
    ErrorClassification,
    InvalidRequest,
    ProviderBilling,
    RequestParams,
    ToolChoice,
)
from langchaint.common.exceptions import TransientError
from langchaint.concurrency.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict
from langchaint.conformance import AdapterConformance
from langchaint.gemini import (
    GeminiGenerateContentAdapter,
    GeminiPricingTable,
    GeminiRates,
    GeminiServiceTier,
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
from langchaint.tools import ToolSchema
from tests.helpers import run_with_timeout


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

_OFFLINE_CLIENT = genai.Client(api_key="offline", vertexai=False)
"""The client every adapter here shares unless a test needs another, since each construction takes milliseconds."""


def _adapter(
    *,
    client: genai.Client = _OFFLINE_CLIENT,
    model: str = "gemini-3.5-flash",
    pricing: Mapping[str, GeminiPricingTable] = _PRICING,
    provider_name: str = "gcp.gemini",
    service_tier: GeminiServiceTier | None = None,
) -> GeminiGenerateContentAdapter:
    """Build the adapter under test, offline unless client says otherwise."""
    return GeminiGenerateContentAdapter(
        client=client,
        model=model,
        pricing=pricing,
        provider_name=provider_name,
        service_tier=service_tier,
    )


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


def _bound_config(
    binding: Binding, adapter: GeminiGenerateContentAdapter | None = None
) -> types.GenerateContentConfig:
    """Bind for text and read the config the binding produced."""
    bound = (adapter or _adapter()).bind_text(binding)
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
    tool_response_id: str | None = None,
) -> list[types.Part]:
    """Build one server-side tool call and its response, matched unless tool_response_id differs."""
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
                id=tool_response_id or tool_call_id,
                tool_type=tool_type,
                response={},
            )
        ),
    ]


def _search_candidates_response(
    queries_per_candidate: Sequence[Sequence[str]],
) -> types.GenerateContentResponse:
    """Build one finished candidate per query list, each holding one Search call and response."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=_provider_call_parts(
                        tool_call_id=f"search-{index}",
                        tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                        queries=list(queries),
                    ),
                ),
                finish_reason=types.FinishReason.STOP,
            )
            for index, queries in enumerate(queries_per_candidate)
        ],
        usage_metadata=_usage_metadata(),
    )


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

    return run_with_timeout(drain())


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


_NO_SDK_RETRIES = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
"""The http_options every binding sends, so max_attempts counts every request."""


def _echo_schema() -> ToolSchema:
    return ToolSchema(
        name="echo",
        description="Echo the city back.",
        args_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )


_ECHO_DECLARATION = types.FunctionDeclaration(
    name="echo",
    description="Echo the city back.",
    parameters_json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)

_SUPPORTED_PROVIDER_TOOLS: tuple[Mapping[str, object], ...] = (
    {"code_execution": {}},
    {"file_search": {"file_search_store_names": ["fileSearchStores/one"]}},
    {"google_maps": {}},
    {"google_search": {}},
    {"url_context": {}},
    {"google_search": {}, "google_maps": {}},
)
"""Each reviewed Gemini provider field, and one mapping populating two of them."""


class _ConfigCase(NamedTuple):
    case_id: str
    binding: Binding
    config: types.GenerateContentConfig


_CONFIG_CASES = [
    _ConfigCase(
        "unset_fields_send_nothing",
        _binding(),
        types.GenerateContentConfig(http_options=_NO_SDK_RETRIES),
    ),
    # Implicit caching has no wire form, so the parameter changes nothing.
    _ConfigCase(
        "automatic_cache_breakpoints_has_no_wire_form",
        _binding(automatic_cache_breakpoints=True),
        types.GenerateContentConfig(http_options=_NO_SDK_RETRIES),
    ),
    _ConfigCase(
        "system_prompt_text",
        _binding(system_prompt="be terse"),
        types.GenerateContentConfig(system_instruction="be terse", http_options=_NO_SDK_RETRIES),
    ),
    _ConfigCase(
        "system_prompt_parts",
        _binding(system_prompt=(TextPart(text="a"), TextPart(text="b"))),
        types.GenerateContentConfig(
            system_instruction=types.Content(parts=[types.Part(text="a"), types.Part(text="b")]),
            http_options=_NO_SDK_RETRIES,
        ),
    ),
    _ConfigCase(
        "max_completion_tokens_and_temperature",
        _binding(max_completion_tokens=64, temperature=0.5),
        types.GenerateContentConfig(
            temperature=0.5, max_output_tokens=64, http_options=_NO_SDK_RETRIES
        ),
    ),
    _ConfigCase(
        "reasoning_level",
        _binding(reasoning_level="HIGH"),
        types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True
            ),
            http_options=_NO_SDK_RETRIES,
        ),
    ),
    # Unmapped wire fields stay reachable as per-request HttpOptions.extra_body.
    _ConfigCase(
        "extra_body",
        _binding(extra_body={"cachedContent": "caches/abc", "generationConfig": {"topK": 5}}),
        types.GenerateContentConfig(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
                extra_body={"cachedContent": "caches/abc", "generationConfig": {"topK": 5}},
            )
        ),
    ),
    # Function declarations precede provider-executed tools, and mode VALIDATED keeps both usable.
    _ConfigCase(
        "function_declarations_precede_provider_tools",
        _binding(tool_schemas=(_echo_schema(),), provider_executed_tools=({"google_search": {}},)),
        types.GenerateContentConfig(
            tools=[
                types.Tool(function_declarations=[_ECHO_DECLARATION]),
                types.Tool(google_search=types.GoogleSearch()),
            ],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.VALIDATED
                ),
                include_server_side_tool_invocations=True,
            ),
            http_options=_NO_SDK_RETRIES,
        ),
    ),
    *(
        _ConfigCase(
            f"provider_tool_{'_'.join(provider_tool)}",
            _binding(provider_executed_tools=(provider_tool,)),
            types.GenerateContentConfig(
                tools=[types.Tool.model_validate(provider_tool)],
                tool_config=types.ToolConfig(include_server_side_tool_invocations=True),
                http_options=_NO_SDK_RETRIES,
            ),
        )
        for provider_tool in _SUPPORTED_PROVIDER_TOOLS
    ),
]


@pytest.mark.parametrize("case", _CONFIG_CASES, ids=[case.case_id for case in _CONFIG_CASES])
def test_binding_fields_build_the_config(case: _ConfigCase) -> None:
    """Each binding field lands on its GenerateContentConfig field, and an unset field sends nothing."""
    assert _bound_config(case.binding) == case.config


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
            AllowedToolsChoice(mode="auto", tool_names=("lookup",)),
            types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.VALIDATED,
                allowed_function_names=["lookup"],
            ),
        ),
        (
            AllowedToolsChoice(mode="required", tool_names=("lookup",)),
            types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=["lookup"],
            ),
        ),
    ],
)
def test_tool_choice_mapping(
    tool_choice: ToolChoice, expected: types.FunctionCallingConfig
) -> None:
    """Neutral "required" is mode ANY. A specific choice is ANY narrowed to the one name.

    One Tool holds every declaration whatever the choice, its schema passed as parameters_json_schema.
    """
    lookup_schema = ToolSchema(
        name="lookup",
        description="Look up the city.",
        args_schema={"type": "object", "properties": {}},
    )
    config = _bound_config(
        _binding(tool_schemas=(_echo_schema(), lookup_schema), tool_choice=tool_choice)
    )
    assert config.tools == [
        types.Tool(
            function_declarations=[
                _ECHO_DECLARATION,
                types.FunctionDeclaration(
                    name="lookup",
                    description="Look up the city.",
                    parameters_json_schema={"type": "object", "properties": {}},
                ),
            ]
        )
    ]
    assert config.tool_config == types.ToolConfig(function_calling_config=expected)


def _pricing_with_query_rate(
    provider_field: str, rate: float | None
) -> dict[str, GeminiPricingTable]:
    """Price provider_field's queries at rate and every other charged tool at a usable rate."""
    return {
        "ON_DEMAND": GeminiPricingTable(
            rates=_ON_DEMAND_RATES,
            google_search_usd_per_query=(rate if provider_field == "google_search" else 0.014),
            google_maps_usd_per_query=(rate if provider_field == "google_maps" else 0.014),
        )
    }


_UNSUPPORTED_PROVIDER_TOOLS: tuple[Mapping[str, object], ...] = (
    {},
    {"computer_use": {}},
    {"enterprise_web_search": {}},
    {"exa_ai_search": {}},
    {"function_declarations": []},
    {"google_search_retrieval": {}},
    {"mcp_servers": []},
    {"parallel_ai_search": {}},
    {"retrieval": {}},
)
"""Every installed `types.Tool` field outside the reviewed set, and a mapping populating none."""

_COLLIDING_EXTRA_BODIES: tuple[Mapping[str, object], ...] = (
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
)
"""extra_body keys the adapter populates, in the spellings and casings the wire accepts."""

_NON_STRING_KEY_GENERATION_CONFIG: dict[object, object] = {1: "value", "temperature": 0.1}


class _BindDefect(NamedTuple):
    case_id: str
    binding: Binding
    match: str
    adapter: GeminiGenerateContentAdapter = _adapter()


_BIND_DEFECTS = [
    # cache_breakpoint has no Gemini wire form.
    _BindDefect(
        "system_prompt_cache_breakpoint",
        _binding(system_prompt=(TextPart(text="a", cache_breakpoint=True),)),
        "cache_breakpoint",
    ),
    # An empty parts tuple can only come from a directly constructed Binding.
    _BindDefect("empty_system_prompt_parts", _binding(system_prompt=()), "empty tuple"),
    # No wire form disables parallel function calls.
    _BindDefect(
        "parallel_tool_calls_false", _binding(parallel_tool_calls=False), "parallel_tool_calls"
    ),
    # A value the Gemini SDK would change does not reach the request.
    _BindDefect(
        "reasoning_level_the_sdk_normalizes",
        _binding(reasoning_level="high"),
        "normalizes it to 'HIGH'",
    ),
    # Gemini ToolConfig cannot select or restrict provider-executed tools.
    _BindDefect(
        "provider_tools_with_required_choice",
        _binding(provider_executed_tools=({"google_search": {}},), tool_choice="required"),
        "tool_choice='auto'",
    ),
    _BindDefect(
        "provider_tools_with_allowed_tools_choice",
        _binding(
            tool_schemas=(_echo_schema(),),
            provider_executed_tools=({"google_search": {}},),
            tool_choice=AllowedToolsChoice(mode="auto", tool_names=("echo",)),
        ),
        "tool_choice='auto'",
    ),
    # Google Search image results have separate unimplemented billing.
    _BindDefect(
        "google_search_image_search",
        _binding(
            provider_executed_tools=({"google_search": {"search_types": {"image_search": {}}}},)
        ),
        "image search",
    ),
    # Every installed `types.Tool` field outside the reviewed set is rejected.
    *(
        _BindDefect(
            f"unsupported_provider_tool_{'_'.join(provider_tool) or 'empty'}",
            _binding(provider_executed_tools=(provider_tool,)),
            "unsupported|validation",
        )
        for provider_tool in _UNSUPPORTED_PROVIDER_TOOLS
    ),
    # Deprecated Gemini 2.5 models have no provider-executed tool path.
    _BindDefect(
        "provider_tools_on_gemini_2_5",
        _binding(provider_executed_tools=({"google_search": {}},)),
        "Gemini 3",
        _adapter(model="gemini-2.5-flash"),
    ),
    _BindDefect(
        "provider_tools_on_vertex_ai",
        _binding(provider_executed_tools=({"google_search": {}},)),
        "Gemini Developer API",
        _adapter(
            client=genai.Client(api_key="offline", vertexai=True), provider_name="gcp.vertex_ai"
        ),
    ),
    # Every supplied pricing table must price each configured charged tool.
    *(
        _BindDefect(
            f"{provider_field}_rate_{rate}",
            _binding(provider_executed_tools=({provider_field: {}},)),
            "unavailable|finite and nonnegative",
            _adapter(pricing=_pricing_with_query_rate(provider_field, rate)),
        )
        for provider_field in ("google_search", "google_maps")
        for rate in (None, True, float("nan"), float("inf"), -0.01)
    ),
    # Binding rejects extra_body keys that duplicate adapter fields.
    *(
        _BindDefect(
            f"colliding_extra_body_{extra_body!r}",
            _binding(extra_body=extra_body),
            "collide",
        )
        for extra_body in _COLLIDING_EXTRA_BODIES
    ),
    # A non-object generationConfig, None included, would replace the adapter's own wholesale.
    _BindDefect(
        "generation_config_not_an_object",
        _binding(extra_body={"generationConfig": "junk"}),
        "object",
    ),
    _BindDefect(
        "generation_config_none", _binding(extra_body={"generationConfig": None}), "object"
    ),
    # Invalid `generationConfig` keys fail before collision detection.
    _BindDefect(
        "generation_config_non_string_key",
        _binding(extra_body={"generationConfig": _NON_STRING_KEY_GENERATION_CONFIG}),
        "only string keys",
    ),
]


@pytest.mark.parametrize("defect", _BIND_DEFECTS, ids=[defect.case_id for defect in _BIND_DEFECTS])
def test_an_unsendable_binding_raises_at_bind(defect: _BindDefect) -> None:
    """A binding without a Gemini wire form fails before any request is built."""
    with pytest.raises(ValueError, match=defect.match):
        _ = defect.adapter.bind_text(defect.binding)


def test_reasoning_level_outside_the_sdk_enum_passes_through() -> None:
    """The exact provider value "XHIGH" reaches the wire unchanged."""
    with pytest.warns(UserWarning, match="XHIGH"):
        config = _bound_config(_binding(reasoning_level="XHIGH"))
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is not None
    assert config.thinking_config.thinking_level.value == "XHIGH"
    assert config.thinking_config.include_thoughts is True


def test_structured_bind_sends_the_response_schema() -> None:
    """The structured binding sends response_json_schema with the JSON mime type."""
    request = (
        _adapter().bind_structured(_binding(), _Answer).build_request([UserMessage(content="hi")])
    )
    assert isinstance(request, _GeminiRequestParams)
    assert request.config == types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=TypeAdapter(_Answer).json_schema(),
        http_options=_NO_SDK_RETRIES,
    )


def test_service_tier_is_sent_and_fingerprinted() -> None:
    """The adapter's service_tier is sent on every request and is the only fingerprinted setting.

    Pricing is billing configuration, so the fingerprint excludes it.
    """
    adapter = _adapter(service_tier="flex")
    assert adapter.config_fingerprint_data() == {"service_tier": "flex"}
    assert _bound_config(_binding(), adapter) == types.GenerateContentConfig(
        service_tier=types.ServiceTier.FLEX, http_options=_NO_SDK_RETRIES
    )


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


def _marked_part_messages(
    part: ContentPart, message_class: type[UserMessage] | type[ToolMessage]
) -> tuple[Message, ...]:
    """Carry one marked ContentPart in a message of message_class."""
    if message_class is UserMessage:
        return (UserMessage(content=(part,)),)
    return (
        AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{}"),)),
        ToolMessage(tool_call_id="c", content=(part,)),
    )


class _InvalidMessages(NamedTuple):
    case_id: str
    messages: tuple[Message, ...]
    reason_fragments: tuple[str, ...]


_INVALID_MESSAGES = [
    # The wire requires the function name, recoverable only from the call the id answers.
    _InvalidMessages(
        "tool_message_without_its_call",
        (ToolMessage(tool_call_id="ghost", content="r"),),
        ("ghost",),
    ),
    # The wire field holds the parsed arguments object, so text that is not JSON has nowhere to go.
    _InvalidMessages(
        "tool_call_args_not_json",
        (AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{not json"),)),),
        ("args_json",),
    ),
    _InvalidMessages(
        "tool_call_args_not_an_object",
        (AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="[1]"),)),),
        ("JSON object",),
    ),
    # A foreign ReasoningPart.raw cannot restore a Gemini Part.
    _InvalidMessages(
        "foreign_reasoning_part",
        (
            AssistantMessage(
                turn=(
                    ReasoningPart(
                        raw={"type": "thinking", "thinking": "x", "signature": "s"}, text="x"
                    ),
                )
            ),
        ),
        ("ReasoningPart",),
    ),
    # cache_breakpoint has no Gemini wire form in UserMessage or ToolMessage.
    *(
        _InvalidMessages(
            f"{type(part).__name__}_cache_breakpoint_in_{message_class.__name__}",
            _marked_part_messages(part, message_class),
            ("cache_breakpoint", type(part).__name__, message_class.__name__),
        )
        for part, message_class in (
            (TextPart(text="a", cache_breakpoint=True), UserMessage),
            (TextPart(text="a", cache_breakpoint=True), ToolMessage),
            (ImageUrlPart(url="gs://bucket/image.png", cache_breakpoint=True), UserMessage),
            (AudioPart(data=b"WAV", media_type="audio/wav", cache_breakpoint=True), UserMessage),
        )
    ),
]


@pytest.mark.parametrize(
    "case", _INVALID_MESSAGES, ids=[case.case_id for case in _INVALID_MESSAGES]
)
def test_unsendable_messages_build_an_invalid_request(case: _InvalidMessages) -> None:
    """Messages without a Gemini wire form become InvalidRequest naming what cannot be sent."""
    invalid = _adapter().bind_text(_binding()).build_request(case.messages)
    assert isinstance(invalid, InvalidRequest)
    for fragment in case.reason_fragments:
        assert fragment in invalid.reason


def test_empty_assistant_text_is_skipped_on_replay() -> None:
    """An empty TextPart puts nothing on the wire."""
    request = _built_request([AssistantMessage(turn=(TextPart(text=""), TextPart(text="kept")))])
    assert request.contents[0].parts == [types.Part(text="kept")]


# --- the thought-signature pairing ---


def _interpreted_turn(response: types.GenerateContentResponse) -> AssistantMessage:
    """Interpret under the text binding and return the turn."""
    outcome = _adapter().bind_text(_binding()).interpret(response)
    assert outcome.kind == "adapter_result"
    return outcome.assistant_message


def _dump(part: types.Part) -> dict[str, JsonValue]:
    """Dump a Part as the adapter stores it in ReasoningPart.raw and RawPart.raw."""
    return part.model_dump(mode="json", exclude_none=True)


_SIGNED_CALL = types.Part(
    thought_signature=b"\x00\x01sig", function_call=types.FunctionCall(name="f", args={"x": 1})
)
# The signature is not valid UTF-8, so only a byte-preserving encoding in ReasoningPart.raw replays it.
_SIGNED_ANSWER = types.Part(text="final answer", thought_signature=b"\x00\xffsig")
_THOUGHT = types.Part(thought=True, text="thinking...")
_EMPTY_TEXT_BESIDE_CODE = types.Part(
    text="",
    executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON),
)


class _ReadTurn(NamedTuple):
    case_id: str
    wire_parts: tuple[types.Part, ...]
    turn: tuple[TurnPart, ...]


_READ_TURNS = [
    # ReasoningPart.raw preserves the signature bytes. ToolCall remains dispatchable.
    _ReadTurn(
        "signed_function_call",
        (_SIGNED_CALL,),
        (ReasoningPart(raw=_dump(_SIGNED_CALL)), ToolCall(id="f", name="f", args_json='{"x": 1}')),
    ),
    # Non-thought text carrying a signature stays readable as answer text.
    _ReadTurn(
        "signed_answer_text",
        (_SIGNED_ANSWER,),
        (ReasoningPart(raw=_dump(_SIGNED_ANSWER)), TextPart(text="final answer")),
    ),
    # Thought text reaches ReasoningPart.text and stays outside the answer text.
    _ReadTurn(
        "thought_text",
        (_THOUGHT, types.Part(text="answer")),
        (ReasoningPart(raw=_dump(_THOUGHT), text="thinking..."), TextPart(text="answer")),
    ),
    # Reading empty text as present rather than as non-empty would drop the whole part.
    _ReadTurn(
        "empty_text_beside_code",
        (_EMPTY_TEXT_BESIDE_CODE,),
        (RawPart(raw=_dump(_EMPTY_TEXT_BESIDE_CODE)),),
    ),
]


@pytest.mark.parametrize("case", _READ_TURNS, ids=[case.case_id for case in _READ_TURNS])
def test_candidate_parts_read_into_the_turn_and_replay_as_themselves(case: _ReadTurn) -> None:
    """Each wire part becomes its turn parts, and the turn replays the original wire parts."""
    turn = _interpreted_turn(_response(case.wire_parts))
    assert turn.turn == case.turn
    request = _built_request([UserMessage(content="go"), turn])
    assert request.contents[1].parts == list(case.wire_parts)


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
    assert ended.kind == "adapter_result"
    assert (ended.output, ended.stop_reason) == ("hi", "end_turn")
    called = bound.interpret(
        _response([types.Part(function_call=types.FunctionCall(name="f", args={}))])
    )
    assert called.kind == "adapter_result"
    assert called.stop_reason == "tool_use"
    truncated = bound.interpret(
        _response([types.Part(text="par")], finish_reason=types.FinishReason.MAX_TOKENS)
    )
    assert truncated.kind == "adapter_result"
    assert (truncated.output, truncated.stop_reason) == ("par", "max_tokens")
    refused = bound.interpret(_response(None, finish_reason=types.FinishReason.SAFETY))
    assert refused.kind == "adapter_result"
    assert (refused.output, refused.stop_reason) == ("", "refusal")
    other = bound.interpret(
        _response([types.Part(text="?")], finish_reason=types.FinishReason.LANGUAGE)
    )
    assert other.kind == "adapter_result"
    assert other.stop_reason == "other"


def test_both_bindings_report_a_missing_finish_reason_as_unfinished() -> None:
    """A candidate without a finish_reason is a turn that never closed, its partial turn carried."""
    response = _response([types.Part(text="par")], finish_reason=None)
    text_outcome = _adapter().bind_text(_binding()).interpret(response)
    assert text_outcome.kind == "unfinished_turn"
    assert text_outcome.assistant_message.text == "par"
    structured_outcome = _adapter().bind_structured(_binding(), _Answer).interpret(response)
    assert structured_outcome.kind == "unfinished_turn"


def test_no_candidates_reads_the_block_reason() -> None:
    """A blocked prompt is a Refusal with an empty turn. No candidates and no block is unfinished."""
    bound = _adapter().bind_text(_binding())
    blocked = bound.interpret(
        _response(None, finish_reason=None, block_reason=types.BlockedReason.SAFETY)
    )
    assert blocked.kind == "refusal"
    assert blocked.assistant_message.turn == ()
    silent = bound.interpret(_response(None, finish_reason=None))
    assert silent.kind == "unfinished_turn"


def test_structured_binding_outcomes() -> None:
    """The structured matrix: instance, tool-call None, refusal, truncation, violation, empty, unfinished."""
    bound = _adapter().bind_structured(_binding(), _Answer)
    parsed = bound.interpret(_response([types.Part(text='{"value": 3}')]))
    assert parsed.kind == "adapter_result"
    assert parsed.output == _Answer(value=3)
    tool_turn = bound.interpret(
        _response([types.Part(function_call=types.FunctionCall(name="f", args={}))])
    )
    assert tool_turn.kind == "adapter_result"
    assert tool_turn.output is None
    assert tool_turn.stop_reason == "tool_use"
    refused = bound.interpret(_response(None, finish_reason=types.FinishReason.SAFETY))
    assert refused.kind == "refusal"
    truncated = bound.interpret(
        _response([types.Part(text='{"value"')], finish_reason=types.FinishReason.MAX_TOKENS)
    )
    assert truncated.kind == "max_completion_tokens_exceeded"
    violated = bound.interpret(_response([types.Part(text='{"value": "not int"}')]))
    assert violated.kind == "schema_violation"
    assert "value" in violated.validation_error_json
    empty = bound.interpret(_response([]))
    assert empty.kind == "empty_turn"
    unfinished = bound.interpret(
        _response([types.Part(text="?")], finish_reason=types.FinishReason.LANGUAGE)
    )
    assert unfinished.kind == "unfinished_turn"
    assert "LANGUAGE" in unfinished.reason


def test_a_structured_turn_ignores_thought_text_when_validating() -> None:
    """Only non-thought text is the candidate instance."""
    bound = _adapter().bind_structured(_binding(), _Answer)
    outcome = bound.interpret(
        _response([
            types.Part(thought=True, text="let me think"),
            types.Part(text='{"value": 7}'),
        ])
    )
    assert outcome.kind == "adapter_result"
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


def _search_call(queries: object) -> list[types.Part]:
    """Build one matched Google Search call and response pair."""
    return _provider_call_parts(
        tool_call_id="search-1", tool_type=types.ToolType.GOOGLE_SEARCH_WEB, queries=queries
    )


class _ToolCost(NamedTuple):
    case_id: str
    response: types.GenerateContentResponse
    configured_fields: frozenset[str]
    cost_in_usd: float
    pricing: Mapping[str, GeminiPricingTable] = _PRICING
    billing_complete: bool = True


_TOOL_COSTS = [
    # Gemini bills unique nonempty Search queries across the complete response.
    _ToolCost(
        "search_counts_unique_nonempty_queries",
        _search_candidates_response([["first", "", "second"], ["second", "third"]]),
        frozenset({"google_search"}),
        3 * 0.014,
    ),
    # Search deduplicates queries while Maps counts every returned query entry.
    _ToolCost(
        "search_deduplicates_and_maps_counts_every_query",
        _response(
            [
                *_search_call(["same", "same", "different"]),
                *_provider_call_parts(
                    tool_call_id="maps-1",
                    tool_type=types.ToolType.GOOGLE_MAPS,
                    queries=["same", "same", "different"],
                ),
            ],
            usage_metadata=_usage_metadata(),
        ),
        frozenset({"google_search", "google_maps"}),
        2 * 0.01 + 3 * 0.02,
        {
            "ON_DEMAND": GeminiPricingTable(
                rates=_ON_DEMAND_RATES,
                google_search_usd_per_query=0.01,
                google_maps_usd_per_query=0.02,
            )
        },
    ),
    # Code execution, URL context, and file search add no separate fee.
    _ToolCost(
        "free_provider_tools",
        _response(
            [
                types.Part(
                    executable_code=types.ExecutableCode(
                        code="print(1)", language=types.Language.PYTHON
                    )
                ),
                *_provider_call_parts(
                    tool_call_id="url-1", tool_type=types.ToolType.URL_CONTEXT, queries=[]
                ),
                *_provider_call_parts(
                    tool_call_id="file-1", tool_type=types.ToolType.FILE_SEARCH, queries=[]
                ),
            ],
            usage_metadata=_usage_metadata(),
        ),
        frozenset({"code_execution", "url_context", "file_search"}),
        0.0,
    ),
    # A charged query at a served tier no table prices costs NaN.
    _ToolCost(
        "unpriced_tier",
        _response(
            _search_call(["query"]),
            usage_metadata=_usage_metadata(traffic_type=types.TrafficType.PROVISIONED_THROUGHPUT),
        ),
        frozenset({"google_search"}),
        float("nan"),
    ),
    # Unexpected query shapes cannot produce an exact provider-executed cost.
    *(
        _ToolCost(
            f"unexpected_{field}_queries_{queries!r}",
            _response(
                _provider_call_parts(tool_call_id="call-1", tool_type=tool_type, queries=queries),
                usage_metadata=_usage_metadata(),
            ),
            frozenset({field}),
            float("nan"),
        )
        for tool_type, field, queries in (
            (types.ToolType.GOOGLE_SEARCH_WEB, "google_search", ["valid", 1]),
            (types.ToolType.GOOGLE_SEARCH_WEB, "google_search", "valid"),
            (types.ToolType.GOOGLE_MAPS, "google_maps", [""]),
            (types.ToolType.GOOGLE_MAPS, "google_maps", ["valid", 1]),
        )
    ),
    # A server call without its matching response is incomplete billing evidence.
    _ToolCost(
        "call_without_response",
        _response(_search_call(["query"])[:1], usage_metadata=_usage_metadata()),
        frozenset({"google_search"}),
        float("nan"),
    ),
    # Server call and response identifiers must match.
    _ToolCost(
        "mismatched_call_and_response_ids",
        _response(
            _provider_call_parts(
                tool_call_id="search-1",
                tool_type=types.ToolType.GOOGLE_SEARCH_WEB,
                queries=["query"],
                tool_response_id="search-2",
            ),
            usage_metadata=_usage_metadata(),
        ),
        frozenset({"google_search"}),
        float("nan"),
    ),
    # A partial response cannot prove the final Search query count.
    _ToolCost(
        "partial_response",
        _response([], usage_metadata=_usage_metadata()),
        frozenset({"google_search"}),
        float("nan"),
        billing_complete=False,
    ),
]


@pytest.mark.parametrize("case", _TOOL_COSTS, ids=[case.case_id for case in _TOOL_COSTS])
def test_provider_executed_tool_cost(case: _ToolCost) -> None:
    """Search and Maps queries cost their per-query rate, and evidence that cannot prove a count costs NaN."""
    usage = _billing_from_response(
        case.response,
        case.pricing,
        configured_fields=case.configured_fields,
        billing_complete=case.billing_complete,
    ).usage
    assert usage.provider_executed_tool_cost_in_usd == pytest.approx(case.cost_in_usd, nan_ok=True)


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


def test_pricing_table_multiplied_scales_both_rate_sets_and_keeps_tool_prices() -> None:
    """`multiplied` scales base and long-prompt token rates and preserves the threshold and tool prices."""
    doubled_on_demand_rates = GeminiRates(
        input_cache_none_usd_per_million_tokens=2.0,
        cache_read_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=20.0,
    )
    assert _LONG_PROMPT_TABLE.multiplied(2.0) == GeminiPricingTable(
        rates=doubled_on_demand_rates,
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
        long_prompt_threshold_tokens=200,
        long_prompt_rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=4.0,
            cache_read_usd_per_million_tokens=0.4,
            output_usd_per_million_tokens=40.0,
        ),
    )
    assert _PRICING["ON_DEMAND"].multiplied(2.0) == GeminiPricingTable(
        rates=doubled_on_demand_rates,
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
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
    """Thought text streams as deltas. Each part boundary emits a separator. Answer text streams bare.

    A signature ends a part across chunks, and a following entry of one chunk's list ends it within one.
    """
    items = _drained(
        _gemini_stream([
            _response([types.Part(thought=True, text="think a")], finish_reason=None),
            _response(
                [types.Part(thought=True, text=" more", thought_signature=b"s1")],
                finish_reason=None,
            ),
            _response(
                [
                    types.Part(thought=True, text="part two"),
                    types.Part(thought=True, text="three"),
                ],
                finish_reason=None,
            ),
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
        ReasoningDelta(text=REASONING_PART_SEPARATOR),
        ReasoningDelta(text="three"),
        "answer",
        ToolCall(id="c1", name="f", args_json='{"x": 1}'),
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

    outcome = _adapter().bind_text(_binding()).interpret(run_with_timeout(final()))
    assert outcome.kind == "refusal"


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

    before, after = run_with_timeout(scenario())
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
                [*_search_call(["query"]), types.Part(text="partial")], finish_reason=None
            ),
        )
        items = stream.items()
        assert await anext(items) == "partial"
        billing = stream.billing_reported()
        await stream.close()
        return billing

    billing = run_with_timeout(scenario())
    assert billing is not None
    assert math.isnan(billing.billing.usage.provider_executed_tool_cost_in_usd)


def test_stream_billing_collects_every_candidate_provider_query() -> None:
    """Stream billing collects Search queries from every candidate."""

    async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
        yield _search_candidates_response([["first"], ["second"]])

    async def scenario() -> ProviderBilling | None:
        stream = _GeminiStream(
            chunks=chunks(),
            pricing=_PRICING,
            provider_tool_fields=frozenset({"google_search"}),
        )
        _ = [item async for item in stream.items()]
        return stream.billing_reported()

    billing = run_with_timeout(scenario())
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

    run_with_timeout(scenario())
    assert closed


def _bound_over(http_client: httpx.AsyncClient) -> BoundAdapter[str]:
    """Bind for text over a client whose HTTP requests go to http_client."""
    client = genai.Client(
        api_key="offline",
        vertexai=False,
        http_options=types.HttpOptions(httpx_async_client=http_client),
    )
    return _adapter(client=client).bind_text(_binding())


_TWO_CHUNK_SSE_BODY = (
    b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "he"}]}}]}\r\n\r\n'
    b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "y"}]},'
    b' "finishReason": "STOP"}]}\r\n\r\n'
)
"""A streamGenerateContent server-sent-event body of two chunks, the second finishing the turn."""


def test_open_stream_sends_the_request_and_pulls_the_first_chunk() -> None:
    """open_stream sends the built request to its model and pulls the first chunk.

    A connection failure therefore raises from open_stream, and the stream still yields the pulled chunk first.
    """
    request = _built_request([UserMessage(content="hi")], _binding(temperature=0.5))
    sent: list[tuple[str, object]] = []

    def refuse(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=http_request)

    def serve(http_request: httpx.Request) -> httpx.Response:
        sent.append((http_request.url.path, json.loads(http_request.content)))
        return httpx.Response(
            200, content=_TWO_CHUNK_SSE_BODY, headers={"content-type": "text/event-stream"}
        )

    async def open_refused() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as http_client:
            with pytest.raises(httpx.ConnectError):
                _ = await _bound_over(http_client).open_stream(request)

    async def served_items() -> list[StreamItem]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http_client:
            stream = await _bound_over(http_client).open_stream(request)
            return [item async for item in stream.items()]

    run_with_timeout(open_refused())
    assert run_with_timeout(served_items()) == ["he", "y"]
    assert sent == [
        (
            "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            {
                "contents": [{"parts": [{"text": "hi"}], "role": "user"}],
                "generationConfig": {"temperature": 0.5},
            },
        )
    ]


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
    def response_with_text(self, text: str) -> BaseModel:
        return _response([types.Part(text=text)])

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
            _api_error(401): "auth",
            _api_error(403): "auth",
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
