"""Gemini generateContent adapter tests over constructed SDK objects.

These pin behavior the type checker cannot: the usage partition and its long-prompt repricing,
finish-reason mapping, the thought-signature pairing and its replay, tool-call id synthesis and
FunctionResponse name recovery, the adapter-owned stream assembly, and the parse and classify tables.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import override

import httpx
import pytest
from google import genai

# The types suppression: the SDK publishes this exact import in its own docs.
from google.genai import errors, types  # pyrefly: ignore[implicit-reexport]
from pydantic import BaseModel, TypeAdapter

from langchaint import (
    AssistantMessage,
    Billing,
    ImagePart,
    InferenceParams,
    Message,
    ReasoningDelta,
    ReasoningTrace,
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
    _billing_from_usage,
    _GeminiRequestParams,
    _GeminiStream,
)
from langchaint.shared_backoff import DoNotRetry, PauseAll, RetryThisOne, Verdict
from langchaint.tools import ToolSchema

_ON_DEMAND_RATES = GeminiRates(
    input_cache_none_usd_per_million_tokens=1.0,
    cache_read_usd_per_million_tokens=0.1,
    output_usd_per_million_tokens=10.0,
)

_PRICING: dict[str, GeminiPricingTable] = {"ON_DEMAND": GeminiPricingTable(rates=_ON_DEMAND_RATES)}
"""The on-demand tier alone, so a response reporting another traffic_type prices NaN."""

_LONG_PROMPT_TABLE = GeminiPricingTable(
    rates=_ON_DEMAND_RATES,
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
        model="gemini-2.5-flash",
        pricing=_PRICING,
        provider_name="gcp.gemini",
    )


def _binding(
    *,
    system_prompt: str | tuple[TextPart, ...] | None = None,
    tool_schemas: tuple[ToolSchema, ...] = (),
    tool_choice: ToolChoice = "auto",
    parallel_tool_calls: bool = True,
    inference_params: InferenceParams | None = None,
    automatic_prompt_caching: bool = False,
    extra_body: Mapping[str, object] | None = None,
) -> Binding:
    """Build a Binding with every field stated, defaults naming the plainest choice."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        inference_params=inference_params if inference_params is not None else InferenceParams(),
        automatic_prompt_caching=automatic_prompt_caching,
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
    """Build usage metadata; the defaults exercise every counter the partition reads."""
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
    block_reason: types.BlockedReason | None = None,
) -> types.GenerateContentResponse:
    """Build a response; parts None with finish_reason None builds one without candidates."""
    candidates: list[types.Candidate] | None = None
    if parts is not None or finish_reason is not None:
        candidates = [
            types.Candidate(
                content=(
                    types.Content(role="model", parts=list(parts)) if parts is not None else None
                ),
                finish_reason=finish_reason,
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
        model_version="gemini-2.5-flash",
        response_id="resp-1",
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

    return asyncio.run(drain())


def _api_error(
    code: int,
    *,
    headers: Mapping[str, str] | None = None,
    retry_delay: str | None = None,
) -> errors.APIError:
    """Build an APIError as raise_for_response builds one, its body the {"error": ...} envelope.

    A response is attached only when headers are stated, so the RetryInfo rows also exercise the
    response-less raise path a mid-stream error chunk takes.
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
    """A str binds as-is; a parts tuple binds as one Content of text parts; None binds none."""
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


def test_automatic_prompt_caching_values_build_identical_requests() -> None:
    """Implicit caching has no wire form, so the flag changes nothing about the request."""
    caching = _built_request([UserMessage(content="hi")], _binding(automatic_prompt_caching=True))
    not_caching = _built_request(
        [UserMessage(content="hi")], _binding(automatic_prompt_caching=False)
    )
    assert caching.config == not_caching.config
    assert caching.contents == not_caching.contents


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
    ],
)
def test_tool_choice_mapping(
    tool_choice: ToolChoice, expected: types.FunctionCallingConfig
) -> None:
    """Neutral "required" is mode ANY; a specific choice is ANY narrowed to the one name."""
    config = _bound_config(_binding(tool_schemas=(_echo_schema(),), tool_choice=tool_choice))
    assert config.tool_config == types.ToolConfig(function_calling_config=expected)


def test_no_tools_binds_no_tool_config() -> None:
    """Without tool_schemas, neither tools nor tool_config is sent."""
    config = _bound_config(_binding())
    assert config.tools is None
    assert config.tool_config is None


def test_inference_params_map_to_generation_fields() -> None:
    """The InferenceParams fields land as temperature and max_output_tokens; None omits either."""
    config = _bound_config(
        _binding(inference_params=InferenceParams(max_completion_tokens=64, temperature=0.5))
    )
    assert config.temperature == 0.5
    assert config.max_output_tokens == 64
    defaulted = _bound_config(_binding())
    assert defaulted.temperature is None
    assert defaulted.max_output_tokens is None


def test_reasoning_effort_none_disables_thinking() -> None:
    """Effort "none" is thinking_budget=0, Gemini's disable form."""
    config = _bound_config(_binding(inference_params=InferenceParams(reasoning_effort="none")))
    assert config.thinking_config == types.ThinkingConfig(thinking_budget=0)


def test_reasoning_effort_maps_to_thinking_level() -> None:
    """A known tier upper-cases onto thinking_level, with include_thoughts True."""
    config = _bound_config(_binding(inference_params=InferenceParams(reasoning_effort="high")))
    assert config.thinking_config == types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True
    )


def test_reasoning_effort_outside_the_sdk_enum_passes_through() -> None:
    """Effort "xhigh" reaches the wire as "XHIGH": the SDK constructs the synthetic member with a warning."""
    with pytest.warns(UserWarning, match="XHIGH"):
        config = _bound_config(
            _binding(inference_params=InferenceParams(reasoning_effort="xhigh"))
        )
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is not None
    assert config.thinking_config.thinking_level.value == "XHIGH"
    assert config.thinking_config.include_thoughts is True


def test_structured_bind_sends_the_response_schema() -> None:
    """The structured binding sends response_json_schema with the JSON mime type; text sends neither."""
    structured = _adapter().bind_structured(_binding(), _Answer)
    request = structured.build_request([UserMessage(content="hi")])
    assert isinstance(request, _GeminiRequestParams)
    assert request.config.response_mime_type == "application/json"
    assert request.config.response_json_schema == TypeAdapter(_Answer).json_schema()
    text_config = _bound_config(_binding())
    assert text_config.response_mime_type is None
    assert text_config.response_json_schema is None


def test_service_tier_lands_on_the_config() -> None:
    """The adapter's service_tier is sent on every request; None sends nothing."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=False),
        model="gemini-2.5-flash",
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
    """A key the adapter populates raises at bind in any spelling the SDK merge would match.

    The SDK matches extra_body keys to wire keys ignoring case and underscores, so the rows cover
    the documented spellings and the normalized ones that would otherwise slip past an exact check.
    """
    with pytest.raises(ValueError, match="collide"):
        _ = _adapter().bind_text(_binding(extra_body=extra_body))


@pytest.mark.parametrize("not_an_object", ["junk", None])
def test_extra_body_generation_config_must_be_an_object(not_an_object: object) -> None:
    """A non-object generationConfig, None included, would replace the adapter's own wholesale."""
    with pytest.raises(ValueError, match="object"):
        _ = _adapter().bind_text(_binding(extra_body={"generationConfig": not_an_object}))


def test_extra_body_passes_unpopulated_keys_through() -> None:
    """Unmapped wire fields stay reachable, delivered as per-request HttpOptions.extra_body."""
    extra_body = {"cachedContent": "caches/abc", "generationConfig": {"topK": 5}}
    config = _bound_config(_binding(extra_body=extra_body))
    assert isinstance(config.http_options, types.HttpOptions)
    assert config.http_options.extra_body == extra_body


# --- build_request ---


def test_user_message_forms() -> None:
    """A str is one text part; a parts tuple maps text to text and images to inline_data blobs."""
    request = _built_request([
        UserMessage(content="hi"),
        UserMessage(
            content=(TextPart(text="look:"), ImagePart(data=b"\x89PNG", media_type="image/png"))
        ),
    ])
    assert request.contents[0] == types.Content(role="user", parts=[types.Part(text="hi")])
    assert request.contents[1] == types.Content(
        role="user",
        parts=[
            types.Part(text="look:"),
            types.Part(inline_data=types.Blob(data=b"\x89PNG", mime_type="image/png")),
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


def test_tool_result_parts_split_text_and_media() -> None:
    """TextPart texts concatenate into the response dict; each ImagePart becomes an inline blob."""
    request = _built_request([
        AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{}"),)),
        ToolMessage(
            tool_call_id="c",
            content=(
                TextPart(text="a"),
                ImagePart(data=b"IMG", media_type="image/png"),
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
        )
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


def test_marked_message_parts_are_invalid() -> None:
    """cache_breakpoint has no Gemini wire form, on user parts and tool-result parts alike."""
    marked_user = _invalid_request([
        UserMessage(content=(TextPart(text="a", cache_breakpoint=True),))
    ])
    assert "cache_breakpoint" in marked_user.reason
    marked_tool = _invalid_request([
        AssistantMessage(turn=(ToolCall(id="c", name="f", args_json="{}"),)),
        ToolMessage(tool_call_id="c", content=(TextPart(text="a", cache_breakpoint=True),)),
    ])
    assert "cache_breakpoint" in marked_tool.reason


def test_a_cross_provider_trace_is_invalid() -> None:
    """A trace another provider produced does not restore to a Part, so the item fails on its own."""
    invalid = _invalid_request([
        AssistantMessage(
            turn=(
                ReasoningTrace(
                    raw={"type": "thinking", "thinking": "x", "signature": "s"}, text="x"
                ),
            )
        )
    ])
    assert "reasoning trace" in invalid.reason


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


def test_a_signed_function_call_yields_a_trace_and_the_call_and_replays_as_one_part() -> None:
    """The trace preserves the signature, the ToolCall reaches dispatch, and replay reunites them."""
    original = types.Part(
        thought_signature=b"\x00\x01sig",
        function_call=types.FunctionCall(name="f", args={"x": 1}),
    )
    turn = _interpreted_turn(_response([original]))
    trace, tool_call = turn.turn
    assert isinstance(trace, ReasoningTrace)
    assert trace.raw == original.model_dump(mode="json", exclude_none=True)
    assert tool_call == ToolCall(id="f", name="f", args_json='{"x": 1}')
    request = _built_request([UserMessage(content="go"), turn])
    model_parts = request.contents[1].parts
    assert model_parts == [original]


def test_signed_answer_text_yields_a_trace_and_the_text_and_replays_as_one_part() -> None:
    """Non-thought text carrying a signature stays readable as answer text and replays signed."""
    original = types.Part(text="final answer", thought_signature=b"sig")
    turn = _interpreted_turn(_response([original]))
    trace, text_part = turn.turn
    assert isinstance(trace, ReasoningTrace)
    assert trace.text is None
    assert text_part == TextPart(text="final answer")
    assert turn.text == "final answer"
    request = _built_request([UserMessage(content="go"), turn])
    assert request.contents[1].parts == [original]


def test_a_thought_part_replays_byte_identical() -> None:
    """The signature bytes survive the JSON round trip through ReasoningTrace.raw."""
    original = types.Part(thought=True, text="reasoning", thought_signature=b"\x00\xffsig")
    turn = _interpreted_turn(_response([original, types.Part(text="answer")]))
    request = _built_request([UserMessage(content="go"), turn])
    model_parts = request.contents[1].parts
    assert model_parts is not None
    assert model_parts[0] == original
    assert model_parts[0].thought_signature == b"\x00\xffsig"


def test_thought_text_is_trace_text_and_not_output() -> None:
    """Thought text reaches trace.text and never the answer."""
    turn = _interpreted_turn(
        _response([
            types.Part(thought=True, text="thinking..."),
            types.Part(text="answer"),
        ])
    )
    trace = turn.turn[0]
    assert isinstance(trace, ReasoningTrace)
    assert trace.text == "thinking..."
    assert turn.text == "answer"


# --- as_json ---


def test_as_json_holds_the_request_without_transport_config() -> None:
    """The archive cell carries model, contents, config, and extra_body; http_options stays out."""
    request = _built_request(
        [UserMessage(content="hi")],
        _binding(
            inference_params=InferenceParams(temperature=0.5),
            extra_body={"cachedContent": "caches/abc"},
        ),
    )
    body = json.loads(request.as_json())
    assert body["model"] == "gemini-2.5-flash"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["config"]["temperature"] == 0.5
    assert "http_options" not in body["config"]
    assert body["extra_body"] == {"cachedContent": "caches/abc"}


# --- interpret ---


def test_text_binding_reads_stop_reasons() -> None:
    """STOP is end_turn or tool_use by the turn's calls; MAX_TOKENS and SAFETY name themselves."""
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
    """A blocked prompt is a Refusal with an empty turn; no candidates and no block is unfinished."""
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
    identity = bound.identity_from_raw(_response([types.Part(text="hi")]))
    assert (identity.model_served, identity.response_id, identity.request_id) == (
        "gemini-2.5-flash",
        "resp-1",
        None,
    )
    bare = bound.identity_from_raw(types.GenerateContentResponse())
    assert (bare.model_served, bare.response_id) == ("", "")


# --- billing ---


def test_the_usage_partition() -> None:
    """cache_read is the cached counter, cache_none the remainder plus tool-use, output includes thoughts."""
    billing = _billing_from_usage(_usage_metadata(), _PRICING)
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


def test_the_long_prompt_threshold_reprices_every_category() -> None:
    """Above the threshold the long rates price; at or below it the base rates do."""
    short = _LONG_PROMPT_TABLE.price(
        service_tier="ON_DEMAND",
        usage_raw=None,
        prompt_token_count=200,
        input_tokens_cache_read=100,
        input_tokens_cache_none=100,
        output_tokens=10,
        output_tokens_reasoning=0,
    )
    assert short.input_cache_none_usd_per_million_tokens == 1.0
    long = _LONG_PROMPT_TABLE.price(
        service_tier="ON_DEMAND",
        usage_raw=None,
        prompt_token_count=201,
        input_tokens_cache_read=100,
        input_tokens_cache_none=101,
        output_tokens=10,
        output_tokens_reasoning=0,
    )
    assert long.input_cache_none_usd_per_million_tokens == 2.0
    assert long.cache_read_usd_per_million_tokens == 0.2
    assert long.output_usd_per_million_tokens == 20.0


def test_tool_execution_input_does_not_cross_the_long_prompt_threshold() -> None:
    """The threshold reads prompt_token_count, which excludes the tool-execution input priced beside it."""
    billing = _billing_from_usage(
        _usage_metadata(prompt_token_count=200, tool_use_prompt_token_count=50),
        {"ON_DEMAND": _LONG_PROMPT_TABLE},
    )
    assert billing.usage.input_tokens_cache_none == 210
    assert billing.input_cache_none_usd_per_million_tokens == 1.0


def test_the_long_prompt_fields_are_required_together() -> None:
    """A threshold without rates prices nothing, and rates without a threshold never apply."""
    with pytest.raises(ValueError, match="together"):
        _ = GeminiPricingTable(rates=_ON_DEMAND_RATES, long_prompt_threshold_tokens=200)
    with pytest.raises(ValueError, match="together"):
        _ = GeminiPricingTable(rates=_ON_DEMAND_RATES, long_prompt_rates=_ON_DEMAND_RATES)


def test_traffic_type_selects_the_table() -> None:
    """A reported tier prices at its own table; UNSPECIFIED and None price at ON_DEMAND."""
    flex_rates = GeminiRates(
        input_cache_none_usd_per_million_tokens=0.5,
        cache_read_usd_per_million_tokens=0.05,
        output_usd_per_million_tokens=5.0,
    )
    pricing = {**_PRICING, "ON_DEMAND_FLEX": GeminiPricingTable(rates=flex_rates)}
    flexed = _billing_from_usage(
        _usage_metadata(traffic_type=types.TrafficType.ON_DEMAND_FLEX), pricing
    )
    assert flexed.service_tier == "ON_DEMAND_FLEX"
    assert flexed.input_cache_none_usd_per_million_tokens == 0.5
    unspecified = _billing_from_usage(
        _usage_metadata(traffic_type=types.TrafficType.TRAFFIC_TYPE_UNSPECIFIED), pricing
    )
    assert unspecified.service_tier == "ON_DEMAND"
    assert unspecified.input_cache_none_usd_per_million_tokens == 1.0


# --- streaming ---


def test_items_translate_parts_with_reasoning_separators() -> None:
    """Thought text streams as deltas, a separator at each part boundary; answer text streams bare."""
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
        ),
    ]
    assembled = assembled_response(chunks)
    assert assembled == _response(
        [
            types.Part(thought=True, text="think a more", thought_signature=b"s1"),
            types.Part(text="answer"),
        ],
        usage_metadata=_usage_metadata(),
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
    """None before usage_metadata arrives; the last-seen usage's Billing after."""

    async def scenario() -> tuple[Billing | None, Billing | None]:
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
    expected = _billing_from_usage(_usage_metadata(), _PRICING)
    # Whole-Billing equality would fail on the NaN cache-write rate both sides carry.
    assert after is not None
    assert after.usage == expected.usage
    assert after.service_tier == expected.service_tier


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
    """A first-pull failure raises from open_stream, and a pulled first chunk still streams first.

    The SDK's generate_content_stream returns an unstarted async generator whose first pull does
    the connection I/O, so open_stream must pull the first chunk for BoundAdapter.open_stream's
    connection-I/O contract to hold.
    """
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
            yield _response([])  # unreachable; the yield makes chunks an async generator

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


def _reasoning_turn_response(
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
) -> types.GenerateContentResponse:
    """One thought part carrying signature bytes, then one text part.

    The SDK models forbid unknown keys, so no fixture can carry a key the installed SDK does not
    name; the signature bytes are the payload an adapter that rebuilt parts from its own model
    would drop.
    """
    return _response(
        [
            types.Part(thought=True, text="check first", thought_signature=b"\x00\x01sig"),
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
        """Return a turn whose usage fills every counter Gemini bills; writes are always zero."""
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
        """Return the reasoning turn; its docstring names the signature bytes as the payload."""
        return _reasoning_turn_response(_usage_metadata())

    @override
    def assistant_wire_elements(self, request: RequestParams) -> Sequence[object]:
        """Read the parts of the model Content this request ends with, as their JSON dumps."""
        assert isinstance(request, _GeminiRequestParams)
        parts = request.contents[-1].parts
        assert parts is not None
        return [part.model_dump(mode="json", exclude_none=True) for part in parts]

    @override
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return one turn as assembled_response builds it and as a whole response."""
        chunks = [
            _response([types.Part(text="Hel")], finish_reason=None),
            _response(
                [types.Part(text="lo")],
                finish_reason=types.FinishReason.STOP,
                usage_metadata=_usage_metadata(),
            ),
        ]
        whole = _response([types.Part(text="Hello")], usage_metadata=_usage_metadata())
        return assembled_response(chunks), whole

    @override
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream ending with neither a finish_reason nor a block_reason."""
        return _gemini_stream([_response([types.Part(text="he")], finish_reason=None)])

    @override
    def sdk_errors_and_classifications(self) -> Mapping[Exception, ErrorClassification]:
        """Return the adapter's whole exception table.

        The transport rows are the failures the SDK's own retry predicate names retryable.
        A status row states the name a DoNotRetry failure takes, never whether it is retried:
        every 4xx is this request's rejection, and any other code is one langchaint has no
        account of. ValueError stands in for an exception the adapter cannot place.
        """
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
        """Return the parse rows: every listed status, both defaults, both retry-after sources.

        The statuses come from the troubleshooting page parse_gemini's docstring cites, plus the
        408 and 502 rows it sources from the SDK's own retryable set; 418 and 599 exercise the
        two fallthrough defaults.
        """
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
