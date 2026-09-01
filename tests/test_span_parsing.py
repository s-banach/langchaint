"""Test parsing OTel chat spans and reconstructing BoundLLM."""

import base64
import json
import re

import pytest
from pydantic import ValidationError

from langchaint import (
    LLM,
    AssistantMessage,
    AudioPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.tools import ToolSchema
from langchaint.tracing import (
    ExtractedOutputMessage,
    ParsedChatSpan,
    reconstruct_bound_llm,
)
from tests.test_bound_llm import _FakeAdapter


def _required_span() -> dict[str, JsonValue]:
    return {
        "gen_ai.provider.name": "fake",
        "gen_ai.request.model": "fake-model",
    }


def _span_with(attributes: dict[str, JsonValue]) -> dict[str, JsonValue]:
    raw_span = _required_span()
    raw_span.update(attributes)
    return raw_span


def test_parses_supported_span_values_and_preserves_unapplied_request_parameters() -> None:
    """Parse reconstructable values and retain other request attributes as JSON values."""
    raw_span = _span_with({
        "gen_ai.request.max_tokens": 200,
        "gen_ai.request.temperature": 0.25,
        "gen_ai.request.reasoning.level": "high",
        "gen_ai.request.stop_sequences": ["done"],
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.stream": True,
        "gen_ai.output.type": "text",
        "gen_ai.response.model": "served-model",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.system_instructions": json.dumps([
            {"type": "text", "content": "Follow the rules."}
        ]),
        "gen_ai.tool.definitions": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up one value.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "gen_ai.input.messages": json.dumps([
            {"role": "user", "parts": [{"type": "text", "content": "question"}]},
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "checking"},
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call-1",
                        "is_error": True,
                        "response": "lookup failed",
                    }
                ],
            },
        ]),
        "gen_ai.output.messages": json.dumps([
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "result"},
                    {
                        "type": "tool_call",
                        "id": "call-2",
                        "name": "lookup",
                        "arguments": {"key": "output"},
                    },
                ],
                "finish_reason": "tool_call",
            }
        ]),
    })

    parsed_span = ParsedChatSpan.model_validate(raw_span)

    assert parsed_span.provider_name == "fake"
    assert parsed_span.model == "fake-model"
    assert parsed_span.output_type == "text"
    assert parsed_span.max_completion_tokens == 200
    assert parsed_span.reasoning_level == "high"
    assert parsed_span.temperature == 0.25
    assert parsed_span.system_prompt == (TextPart(text="Follow the rules."),)
    assert parsed_span.tool_schemas == (
        ToolSchema(
            name="lookup",
            description="Look up one value.",
            args_schema={"type": "object", "properties": {}},
        ),
    )
    assert parsed_span.unapplied_request_parameters == {
        "gen_ai.request.stop_sequences": ["done"],
        "gen_ai.request.stream": True,
        "gen_ai.request.top_p": 0.9,
    }
    assert parsed_span.generation_input == (
        UserMessage(content=(TextPart(text="question"),)),
        AssistantMessage(
            turn=(
                TextPart(text="checking"),
                ToolCall(id="call-1", name="lookup", args_json='{"key":"value"}'),
            )
        ),
        ToolMessage(tool_call_id="call-1", content="lookup failed", is_error=True),
    )
    assert parsed_span.output_messages == (
        ExtractedOutputMessage(
            assistant_message=AssistantMessage(
                turn=(
                    TextPart(text="result"),
                    ToolCall(id="call-2", name="lookup", args_json='{"key":"output"}'),
                )
            ),
            finish_reason="tool_call",
        ),
    )


def test_parses_tool_response_content_parts_from_json_lists() -> None:
    """Convert a JSON list response into ToolMessage content parts."""
    parsed_span = ParsedChatSpan.model_validate(
        _span_with({
            "gen_ai.input.messages": [
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": "call-1",
                            "is_error": False,
                            "response": [
                                {"type": "text", "content": "result"},
                                {
                                    "type": "image_url",
                                    "url": "https://example.com/image.png",
                                },
                            ],
                        }
                    ],
                }
            ]
        })
    )

    assert parsed_span.generation_input == (
        ToolMessage(
            tool_call_id="call-1",
            content=(
                TextPart(text="result"),
                ImageUrlPart(url="https://example.com/image.png"),
            ),
            is_error=False,
        ),
    )


def test_decodes_foreign_inline_media_and_image_uris() -> None:
    """Decode standard and URL-safe base64 plus both supported image URI forms."""
    parsed_span = ParsedChatSpan.model_validate(
        _span_with({
            "gen_ai.input.messages": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "blob",
                            "modality": "image",
                            "mime_type": "image/png",
                            "content": base64.b64encode(b"image bytes").decode(),
                        },
                        {
                            "type": "blob",
                            "modality": "audio",
                            "mime_type": "audio/wav",
                            "content": base64.urlsafe_b64encode(b"audio bytes").decode(),
                        },
                        {
                            "type": "uri",
                            "modality": "image",
                            "mime_type": "image/webp",
                            "uri": "gs://bucket/image.webp",
                        },
                        {"type": "image_url", "url": "https://example.com/image.png"},
                    ],
                }
            ]
        })
    )

    assert parsed_span.generation_input == (
        UserMessage(
            content=(
                ImagePart(data=b"image bytes", media_type="image/png"),
                AudioPart(data=b"audio bytes", media_type="audio/wav"),
                ImageUrlPart(url="gs://bucket/image.webp", media_type="image/webp"),
                ImageUrlPart(url="https://example.com/image.png"),
            )
        ),
    )


@pytest.mark.parametrize(
    ("attributes", "error_text"),
    [
        (
            {
                "gen_ai.tool.definitions": [
                    {"type": "function", "name": "lookup", "parameters": {}}
                ]
            },
            "description",
        ),
        (
            {
                "gen_ai.input.messages": [
                    {
                        "role": "user",
                        "parts": [{"type": "file", "modality": "document", "file_id": "file-1"}],
                    }
                ]
            },
            "file",
        ),
        (
            {
                "gen_ai.input.messages": [
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "id": "call-1",
                                "response": "result",
                            }
                        ],
                    }
                ]
            },
            "is_error",
        ),
        (
            {
                "gen_ai.input.messages": [
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "id": "call-1",
                                "is_error": False,
                                "response": "first",
                            },
                            {
                                "type": "tool_call_response",
                                "id": "call-2",
                                "is_error": False,
                                "response": "second",
                            },
                        ],
                    }
                ]
            },
            "too_long",
        ),
        (
            {
                "gen_ai.input.messages": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "content": "hello",
                                "provider_field": "value",
                            }
                        ],
                    }
                ]
            },
            "provider_field",
        ),
        (
            {
                "gen_ai.output.messages": [
                    {
                        "role": "assistant",
                        "parts": [{"type": "reasoning", "content": "thinking"}],
                        "finish_reason": "stop",
                    }
                ]
            },
            "reasoning",
        ),
        ({"gen_ai.input.messages": "not JSON"}, "Invalid JSON"),
    ],
)
def test_rejects_values_outside_the_reconstructable_subset(
    attributes: dict[str, JsonValue], error_text: str
) -> None:
    """Reject structured values that langchaint messages cannot represent."""
    with pytest.raises(ValidationError, match=re.escape(error_text)):
        _ = ParsedChatSpan.model_validate(_span_with(attributes))


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_temperature(temperature: float) -> None:
    """Reject a non-finite applied request parameter."""
    with pytest.raises(ValidationError, match="finite"):
        _ = ParsedChatSpan.model_validate(_span_with({"gen_ai.request.temperature": temperature}))


def test_validates_unrelated_telemetry_before_discarding_it() -> None:
    """Reject a non-JSON unrelated attribute before attribute partitioning."""
    with pytest.raises(ValidationError, match="finite"):
        _ = ParsedChatSpan.model_validate(
            _span_with({"gen_ai.response.vendor_measurement": float("nan")})
        )


@pytest.mark.parametrize("missing_attribute", ["gen_ai.provider.name", "gen_ai.request.model"])
def test_requires_provider_name_and_model(missing_attribute: str) -> None:
    """Require the identities used to select and validate the supplied LLM."""
    raw_span = _required_span()
    del raw_span[missing_attribute]

    with pytest.raises(ValidationError, match=re.escape(missing_attribute)):
        _ = ParsedChatSpan.model_validate(raw_span)


def test_reconstructs_provider_neutral_binding_fields() -> None:
    """Bind recorded provider-neutral fields and retain reconstruction defaults."""
    parsed_span = ParsedChatSpan.model_validate(
        _span_with({
            "gen_ai.system_instructions": [{"type": "text", "content": "Follow the rules."}],
            "gen_ai.request.max_tokens": 200,
            "gen_ai.request.reasoning.level": "high",
            "gen_ai.request.temperature": 0.25,
        })
    )

    bound_llm = reconstruct_bound_llm(parsed_span, llm=LLM(_FakeAdapter()))

    assert bound_llm.binding.system_prompt == (TextPart(text="Follow the rules."),)
    assert bound_llm.binding.max_completion_tokens == 200
    assert bound_llm.binding.reasoning_level == "high"
    assert bound_llm.binding.temperature == 0.25
    assert bound_llm.binding.tool_schemas == ()
    assert bound_llm.binding.extra_body is None
    assert bound_llm.response_format is None
    assert bound_llm.tool_manager is None
    assert bound_llm.max_attempts == 3


def test_reconstruction_leaves_unapplied_span_values_unbound() -> None:
    """Keep tool schemas, JSON output, and other request attributes off the binding."""
    parsed_span = ParsedChatSpan.model_validate(
        _span_with({
            "gen_ai.output.type": "json",
            "gen_ai.system_instructions": [],
            "gen_ai.tool.definitions": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up one value.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "gen_ai.request.stream": True,
            "gen_ai.request.vendor_option": {"mode": "fast"},
        })
    )

    bound_llm = reconstruct_bound_llm(parsed_span, llm=LLM(_FakeAdapter()))

    assert parsed_span.output_type == "json"
    assert parsed_span.tool_schemas
    assert parsed_span.unapplied_request_parameters == {
        "gen_ai.request.stream": True,
        "gen_ai.request.vendor_option": {"mode": "fast"},
    }
    assert bound_llm.binding.system_prompt is None
    assert bound_llm.binding.tool_schemas == ()
    assert bound_llm.binding.extra_body is None
    assert bound_llm.response_format is None


@pytest.mark.parametrize(
    ("adapter_attribute", "different_value", "error_text"),
    [
        ("provider_name", "other-provider", "provider_name"),
        ("model", "other-model", "model"),
    ],
)
def test_reconstruction_rejects_a_different_llm_identity(
    adapter_attribute: str, different_value: str, error_text: str
) -> None:
    """Reject an LLM whose provider name or model differs from the parsed span."""
    adapter = _FakeAdapter()
    if adapter_attribute == "provider_name":
        adapter.provider_name = different_value
    else:
        adapter.model = different_value

    with pytest.raises(ValueError, match=error_text):
        _ = reconstruct_bound_llm(
            ParsedChatSpan.model_validate(_required_span()),
            llm=LLM(adapter),
        )
