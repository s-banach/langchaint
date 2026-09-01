"""Test private extraction of langchaint request values from OTel spans."""

import base64
import json
import re

import pytest

from langchaint import (
    AssistantMessage,
    AudioPart,
    ImagePart,
    ImageUrlPart,
    TextPart,
    ToolCall,
    UserMessage,
)
from langchaint.tools import ToolSchema
from langchaint.tracing._span_parsing import ExtractedOutputMessage, extract_span_parameters


def test_extracts_supported_span_values() -> None:
    """Extract langchaint values and preserve standard functional request attributes."""
    extraction = extract_span_parameters({
        "gen_ai.request.max_tokens": 200,
        "gen_ai.request.temperature": 0.25,
        "gen_ai.request.reasoning.level": "high",
        "gen_ai.request.stop_sequences": ("done",),
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.model": "captured-model",
        "gen_ai.request.stream": True,
        "gen_ai.provider.name": "captured-provider",
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

    assert extraction.binding_parameters == {
        "max_completion_tokens": 200,
        "reasoning_level": "high",
        "temperature": 0.25,
        "system_prompt": (TextPart(text="Follow the rules."),),
        "tool_schemas": (
            ToolSchema(
                name="lookup",
                description="Look up one value.",
                args_schema={"type": "object", "properties": {}},
            ),
        ),
    }
    assert extraction.request_parameters == {
        "gen_ai.request.max_tokens": 200,
        "gen_ai.request.model": "captured-model",
        "gen_ai.request.temperature": 0.25,
        "gen_ai.request.reasoning.level": "high",
        "gen_ai.request.stop_sequences": ("done",),
        "gen_ai.request.top_p": 0.9,
    }
    assert extraction.generation_input == (
        UserMessage(content=(TextPart(text="question"),)),
        AssistantMessage(
            turn=(
                TextPart(text="checking"),
                ToolCall(id="call-1", name="lookup", args_json='{"key":"value"}'),
            )
        ),
    )
    assert extraction.output_messages == (
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
    empty_extraction = extract_span_parameters({"gen_ai.system_instructions": []})
    assert empty_extraction.binding_parameters == {}
    assert empty_extraction.output_messages is None


def test_decodes_foreign_inline_media_and_image_uris() -> None:
    """Decode standard and URL-safe base64 plus both supported image URI forms."""
    extraction = extract_span_parameters({
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

    assert extraction.generation_input == (
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
        ({"gen_ai.request.vendor_option": 0.9}, "gen_ai.request.vendor_option"),
        (
            {
                "gen_ai.request.temperature": 10**400,
            },
            "gen_ai.request.temperature",
        ),
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
                "gen_ai.tool.definitions": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "description": "Look up one value.",
                        "parameters": {"properties": {"values": (1, 2)}},
                    }
                ]
            },
            "JSON runtime types",
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
            "tool",
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
    attributes: dict[str, object], error_text: str
) -> None:
    """Raise for unsupported request attributes and values the result cannot preserve."""
    with pytest.raises(ValueError, match=re.escape(error_text)):
        _ = extract_span_parameters(attributes)
