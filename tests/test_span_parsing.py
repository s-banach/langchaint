"""Test OTel-native chat span parsing and explicit langchaint conversion."""

import json
import pathlib

import pytest
from pydantic import ValidationError

import langchaint.tracing
from langchaint import (
    LLM,
    AssistantMessage,
    ImagePart,
    JsonValue,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.tools import ToolSchema
from langchaint.tracing import (
    ExtractedOutputMessage,
    OtelBlobPart,
    OtelChatSpan,
    OtelCompactionPart,
    OtelFilePart,
    OtelFunctionTool,
    OtelGenericPart,
    OtelGenericSystemInstructionPart,
    OtelGenericTool,
    OtelInputMessage,
    OtelOutputMessage,
    OtelReasoningPart,
    OtelServerToolCallPart,
    OtelServerToolCallResponsePart,
    OtelTextPart,
    OtelToLangchaintConversionError,
    OtelToolCallPart,
    OtelToolCallResponsePart,
    OtelUriPart,
    generation_input_from_otel,
    parse_otel,
    reconstruct_bound_llm,
)
from langchaint.tracing._span_parsing import (
    _output_messages_from_otel,
    _system_prompt_from_otel,
    _tool_schemas_from_otel,
)
from scripts import refresh_semconv_genai
from tests.test_bound_llm import _FakeAdapter

SEMCONV_DIRECTORY = pathlib.Path(__file__).parent / "semconv_genai"
_VALID_AND_INVALID_VALUE_BY_OTEL_TYPE: dict[str, tuple[JsonValue, JsonValue]] = {
    "boolean": (True, "true"),
    "double": (0.5, "0.5"),
    "int": (1, 1.5),
    "string": ("value", 1),
    "string[]": (["value"], [1]),
}


def _chat_span(attributes: dict[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    raw_attributes: dict[str, JsonValue] = {"gen_ai.operation.name": "chat"}
    if attributes is not None:
        raw_attributes.update(attributes)
    return raw_attributes


def _assert_refinements(
    refinements: list[JsonValue],
    provider_values: list[JsonValue],
    base_entries_by_name: dict[str, dict[str, JsonValue]],
) -> None:
    conditions_by_id: dict[str, JsonValue] = {}
    retained_overrides: dict[tuple[str, str], JsonValue] = {}
    for refinement in refinements:
        assert isinstance(refinement, dict)
        refinement_id = refinement["id"]
        condition = refinement["condition"]
        assert isinstance(refinement_id, str)
        assert isinstance(condition, dict)
        assert condition["attribute"] == "gen_ai.provider.name"
        assert condition["equals"] in provider_values
        refinement_attributes = refinement["attributes"]
        assert isinstance(refinement_attributes, list)
        for attribute in refinement_attributes:
            assert isinstance(attribute, dict)
            attribute_name = attribute["name"]
            assert isinstance(attribute_name, str)
            assert attribute["condition"] == condition
            base_attribute = base_entries_by_name.get(attribute_name)
            if base_attribute is not None:
                refinement_metadata = {
                    key: value for key, value in attribute.items() if key != "condition"
                }
                base_metadata = {
                    key: value for key, value in base_attribute.items() if key != "condition"
                }
                assert refinement_metadata != base_metadata
                retained_overrides[(refinement_id, attribute_name)] = attribute["presence_rule"]
        conditions_by_id[refinement_id] = condition["equals"]
    assert conditions_by_id["azure.ai.inference.client"] == "azure.ai.inference"
    assert retained_overrides == {
        ("anthropic.inference.client", "gen_ai.request.top_k"): "recommended",
        ("aws.bedrock.inference.client", "gen_ai.request.top_k"): "recommended",
        (
            "azure.ai.inference.client",
            "server.port",
        ): {"conditionally_required": "If not default (443)."},
        ("openai.inference.client", "gen_ai.request.model"): "required",
    }


def _assert_scalar_attributes_parse(
    entries_by_name: dict[str, dict[str, JsonValue]], field_names_by_alias: dict[str, str]
) -> None:
    for attribute_name, entry in entries_by_name.items():
        value_type = entry["value_type"]
        assert isinstance(value_type, str)
        if value_type == "any":
            continue
        valid_value, invalid_value = _VALID_AND_INVALID_VALUE_BY_OTEL_TYPE[value_type]
        if attribute_name == "gen_ai.operation.name":
            valid_value = "chat"
        parsed = parse_otel(_chat_span({attribute_name: valid_value}))
        actual_value = parsed.model_dump()[field_names_by_alias[attribute_name]]
        if value_type == "string[]":
            assert isinstance(valid_value, list)
            expected_value: object = tuple(valid_value)
        else:
            expected_value = valid_value
        assert actual_value == expected_value
        with pytest.raises(ValidationError):
            _ = parse_otel(_chat_span({attribute_name: invalid_value}))
        allowed_values = entry["allowed_values"]
        if allowed_values is None:
            continue
        assert isinstance(allowed_values, list)
        if attribute_name == "gen_ai.operation.name":
            assert "chat" in allowed_values
            continue
        for allowed_value in allowed_values:
            assert isinstance(allowed_value, str)


def test_manifest_attributes_match_otel_chat_span_aliases() -> None:
    """Parser validation matches scalar types in the committed declaration.

    Structured schema paths match ATTRIBUTE_SCHEMA_FILES.
    """
    manifest: dict[str, JsonValue] = json.loads(
        (SEMCONV_DIRECTORY / "chat-span-attributes.json").read_text()
    )
    attributes = manifest["attributes"]
    refinements = manifest["refinements"]
    assert isinstance(attributes, list)
    assert isinstance(refinements, list)
    declared_entries: list[JsonValue] = [*attributes]
    for refinement in refinements:
        assert isinstance(refinement, dict)
        refinement_attributes = refinement["attributes"]
        assert isinstance(refinement_attributes, list)
        declared_entries.extend(refinement_attributes)
    entries_by_name: dict[str, dict[str, JsonValue]] = {}
    for entry in declared_entries:
        assert isinstance(entry, dict)
        name = entry["name"]
        assert isinstance(name, str)
        selected = {
            key: entry[key]
            for key in ("value_type", "allowed_values", "structured_schema", "template_prefix")
        }
        if name in entries_by_name:
            assert entries_by_name[name] == selected
        else:
            entries_by_name[name] = selected
    prompt_variable = entries_by_name.pop("gen_ai.prompt.variable")
    assert prompt_variable == {
        "value_type": "string",
        "allowed_values": None,
        "structured_schema": None,
        "template_prefix": "gen_ai.prompt.variable.",
    }
    expected_schemas = {
        name: f"model/gen-ai/{file_name}"
        for name, file_name in refresh_semconv_genai.ATTRIBUTE_SCHEMA_FILES.items()
        if name in entries_by_name
    }
    actual_schemas = {
        name: entry["structured_schema"]
        for name, entry in entries_by_name.items()
        if entry["structured_schema"] is not None
    }
    assert actual_schemas == expected_schemas
    assert all(entry["template_prefix"] is None for entry in entries_by_name.values())
    provider_values = entries_by_name["gen_ai.provider.name"]["allowed_values"]
    assert isinstance(provider_values, list)
    base_entries_by_name: dict[str, dict[str, JsonValue]] = {}
    for attribute in attributes:
        assert isinstance(attribute, dict)
        attribute_name = attribute["name"]
        assert isinstance(attribute_name, str)
        base_entries_by_name[attribute_name] = attribute
    _assert_refinements(refinements, provider_values, base_entries_by_name)
    field_names_by_alias: dict[str, str] = {}
    for field_name, field in OtelChatSpan.model_fields.items():
        if field_name not in {"prompt_variables", "unused_attributes"}:
            assert field.alias is not None
            field_names_by_alias[field.alias] = field_name
    assert set(entries_by_name) == set(field_names_by_alias)
    _assert_scalar_attributes_parse(entries_by_name, field_names_by_alias)


def test_present_scalar_attribute_cannot_be_null() -> None:
    """A present scalar alias cannot use the absent-value default."""
    with pytest.raises(ValidationError):
        _ = parse_otel(_chat_span({"gen_ai.request.seed": None}))


def test_only_chat_operation_is_required() -> None:
    """The chat operation selects the parser without requiring producer presence labels."""
    assert parse_otel(_chat_span()) == OtelChatSpan.model_validate({
        "gen_ai.operation.name": "chat"
    })


def test_only_generation_input_conversion_is_public() -> None:
    """Only generation_input_from_otel is a public conversion function in langchaint.tracing."""
    assert "generation_input_from_otel" in langchaint.tracing.__all__
    assert "output_messages_from_otel" not in langchaint.tracing.__all__
    assert "system_prompt_from_otel" not in langchaint.tracing.__all__
    assert "tool_schemas_from_otel" not in langchaint.tracing.__all__


@pytest.mark.parametrize("raw_attributes", [{}, {"gen_ai.operation.name": "embeddings"}])
def test_requires_chat_operation(raw_attributes: dict[str, JsonValue]) -> None:
    """A missing or different operation cannot select the chat parser."""
    with pytest.raises(ValidationError, match=r"gen_ai\.operation\.name"):
        _ = parse_otel(raw_attributes)


@pytest.mark.parametrize(
    ("part", "expected_type"),
    [
        ({"type": "text", "content": "hello"}, OtelTextPart),
        ({"type": "tool_call", "name": "lookup"}, OtelToolCallPart),
        ({"type": "tool_call_response", "response": "ok"}, OtelToolCallResponsePart),
        (
            {
                "type": "server_tool_call",
                "name": "search",
                "server_tool_call": {"type": "web", "query": "weather"},
            },
            OtelServerToolCallPart,
        ),
        (
            {
                "type": "server_tool_call_response",
                "server_tool_call_response": {"type": "web", "result": "sunny"},
            },
            OtelServerToolCallResponsePart,
        ),
        ({"type": "blob", "modality": "image", "content": "aW1hZ2U="}, OtelBlobPart),
        ({"type": "file", "modality": "document", "file_id": "file-1"}, OtelFilePart),
        ({"type": "uri", "modality": "image", "uri": "gs://bucket/image"}, OtelUriPart),
        ({"type": "reasoning", "content": "thinking"}, OtelReasoningPart),
        ({"type": "compaction"}, OtelCompactionPart),
        ({"type": "provider_part", "payload": 42}, OtelGenericPart),
    ],
)
def test_parses_each_message_part_variant(
    part: dict[str, JsonValue], expected_type: type[object]
) -> None:
    """Each structured message-part variant parses into its OTel-native model."""
    parsed = parse_otel(_chat_span({"gen_ai.input.messages": [{"role": "user", "parts": [part]}]}))
    assert parsed.input_messages is not None
    assert isinstance(parsed.input_messages[0].parts[0], expected_type)


def test_known_message_type_can_use_the_generic_schema_variant() -> None:
    """OtelGenericPart accepts a known type when the dedicated model does not match."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [
                {"role": "user", "parts": [{"type": "text", "provider_value": 1}]}
            ]
        })
    )
    assert parsed.input_messages is not None
    part = parsed.input_messages[0].parts[0]
    assert isinstance(part, OtelGenericPart)
    assert part.type == "text"
    assert part.additional_properties == {"provider_value": 1}


def test_known_system_type_can_use_the_generic_schema_variant() -> None:
    """OtelGenericSystemInstructionPart accepts text when OtelTextPart does not match."""
    parsed = parse_otel(
        _chat_span({"gen_ai.system_instructions": [{"type": "text", "provider_value": 1}]})
    )
    assert parsed.system_instructions is not None
    part = parsed.system_instructions[0]
    assert isinstance(part, OtelGenericSystemInstructionPart)
    assert part.type == "text"
    assert part.additional_properties == {"provider_value": 1}


def test_known_tool_type_can_use_the_generic_schema_variant() -> None:
    """OtelGenericTool accepts function when OtelFunctionTool does not match."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.tool.definitions": [{"type": "function", "name": "lookup", "description": 42}]
        })
    )
    assert parsed.tool_definitions is not None
    definition = parsed.tool_definitions[0]
    assert isinstance(definition, OtelGenericTool)
    assert definition.type == "function"
    assert definition.additional_properties == {"description": 42}


def test_decodes_json_strings_and_accepts_decoded_structured_values() -> None:
    """Structured attributes accept JSON text and its decoded JSON value."""
    value: list[JsonValue] = [{"type": "text", "content": "Follow the rules."}]
    from_text = parse_otel(_chat_span({"gen_ai.system_instructions": json.dumps(value)}))
    from_value = parse_otel(_chat_span({"gen_ai.system_instructions": value}))
    assert from_text.system_instructions == from_value.system_instructions
    assert from_text.system_instructions == (
        OtelTextPart(type="text", content="Follow the rules."),
    )


def test_preserves_known_and_generic_additional_properties() -> None:
    """Known and generic part models retain properties outside their declared fields."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "content": "hello", "langchain.x.y.z": 42},
                        {"type": "provider_part", "provider_value": True},
                    ],
                }
            ]
        })
    )
    assert parsed.input_messages is not None
    text_part, generic_part = parsed.input_messages[0].parts
    assert isinstance(text_part, OtelTextPart)
    assert text_part.additional_properties == {"langchain.x.y.z": 42}
    assert isinstance(generic_part, OtelGenericPart)
    assert generic_part.additional_properties == {"provider_value": True}


def test_preserves_a_raw_additional_properties_property() -> None:
    """A raw property named additional_properties remains an additional property."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "content": "hello",
                            "additional_properties": {"vendor": 1},
                        }
                    ],
                }
            ]
        })
    )
    assert parsed.input_messages is not None
    part = parsed.input_messages[0].parts[0]
    assert isinstance(part, OtelTextPart)
    assert part.additional_properties == {"additional_properties": {"vendor": 1}}


def test_partitions_prompt_variables_and_unused_attributes_once() -> None:
    """Typed attributes and prompt variables stay outside unused_attributes."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.provider.name": "openai",
            "gen_ai.prompt.variable.user_name": "Alice",
            "langchain.x.y.z": 42,
        })
    )
    assert parsed.provider_name == "openai"
    assert parsed.prompt_variables == {"user_name": "Alice"}
    assert parsed.unused_attributes == {"langchain.x.y.z": 42}


@pytest.mark.parametrize(
    "attributes",
    [
        {"gen_ai.request.max_tokens": True},
        {"gen_ai.request.temperature": float("nan")},
        {"gen_ai.prompt.variable.user_name": 42},
        {"gen_ai.input.messages": "not JSON"},
    ],
)
def test_rejects_malformed_standard_attributes(attributes: dict[str, JsonValue]) -> None:
    """A malformed standard attribute raises Pydantic validation."""
    with pytest.raises(ValidationError):
        _ = parse_otel(_chat_span(attributes))


@pytest.mark.parametrize("encode_as_json", [False, True])
@pytest.mark.parametrize(
    ("attribute_name", "decoded_attribute_value"),
    [
        ("gen_ai.system_instructions", [1]),
        ("gen_ai.tool.definitions", [1]),
        ("gen_ai.input.messages", [1]),
        ("gen_ai.output.messages", [1]),
        ("gen_ai.input.messages", [{"role": "user", "parts": [1]}]),
        ("gen_ai.output.messages", [{"role": "user", "parts": [1]}]),
    ],
)
def test_rejects_non_object_structured_values(
    attribute_name: str,
    decoded_attribute_value: JsonValue,
    *,
    encode_as_json: bool,
) -> None:
    """parse_otel raises ValidationError for a non-object structured value."""
    attribute_value = (
        json.dumps(decoded_attribute_value) if encode_as_json else decoded_attribute_value
    )
    with pytest.raises(ValidationError, match="an OTel structured value must be an object"):
        _ = parse_otel(_chat_span({attribute_name: attribute_value}))


def test_rejects_a_malformed_function_parameters_schema() -> None:
    """OtelFunctionTool validates parameters as a draft-07 schema."""
    with pytest.raises(ValidationError, match="JSON Schema draft-07"):
        _ = OtelFunctionTool(type="function", name="lookup", parameters={"type": 42})


def test_parses_boolean_draft_07_schema_before_conversion_fails() -> None:
    """A boolean draft-07 schema parses before ToolSchema conversion reports unsupported data."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.tool.definitions": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up one value.",
                    "parameters": True,
                }
            ]
        })
    )
    assert parsed.tool_definitions is not None
    with pytest.raises(OtelToLangchaintConversionError, match="description and parameters"):
        _ = _tool_schemas_from_otel(parsed.tool_definitions)


def test_parsing_succeeds_before_unsupported_conversion_fails() -> None:
    """A valid reasoning part parses before langchaint conversion reports unsupported data."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "assistant",
                    "parts": [{"type": "reasoning", "content": "thinking"}],
                }
            ]
        })
    )
    assert parsed.input_messages is not None
    assert isinstance(parsed.input_messages[0].parts[0], OtelReasoningPart)
    with pytest.raises(OtelToLangchaintConversionError, match="assistant part type"):
        _ = generation_input_from_otel(parsed.input_messages)


def test_tool_response_conversion_wraps_malformed_nested_parts() -> None:
    """A valid arbitrary response value fails through OtelToLangchaintConversionError."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "tool",
                    "parts": [{"type": "tool_call_response", "id": "call-1", "response": [1]}],
                }
            ]
        })
    )
    assert parsed.input_messages is not None
    with pytest.raises(OtelToLangchaintConversionError, match="tool response value"):
        _ = generation_input_from_otel(parsed.input_messages)


def test_converts_representable_input_messages() -> None:
    """Representable user, assistant, and tool messages retain prior reconstruction behavior."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "content": "question"},
                        {
                            "type": "blob",
                            "modality": "image",
                            "mime_type": "image/png",
                            "content": "aW1hZ2U=",
                        },
                    ],
                },
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
                            "response": "failed",
                        }
                    ],
                },
            ]
        })
    )
    assert parsed.input_messages is not None
    assert generation_input_from_otel(parsed.input_messages) == (
        UserMessage(
            content=(
                TextPart(text="question"),
                ImagePart(data=b"image", media_type="image/png"),
            )
        ),
        AssistantMessage(
            turn=(
                TextPart(text="checking"),
                ToolCall(id="call-1", name="lookup", args_json='{"key":"value"}'),
            )
        ),
        ToolMessage(tool_call_id="call-1", content="failed", is_error=True),
    )


def test_converts_system_instructions_tools_and_output_messages() -> None:
    """Explicit conversion functions produce the prior langchaint values."""
    assert _system_prompt_from_otel((OtelTextPart(type="text", content="Be brief."),)) == (
        TextPart(text="Be brief."),
    )
    assert _tool_schemas_from_otel((
        OtelFunctionTool(
            type="function",
            name="lookup",
            description="Look up one value.",
            parameters={"type": "object"},
        ),
    )) == (
        ToolSchema(
            name="lookup",
            description="Look up one value.",
            args_schema={"type": "object"},
        ),
    )
    output = OtelOutputMessage(
        role="assistant",
        parts=(OtelTextPart(type="text", content="done"),),
        finish_reason="stop",
    )
    assert _output_messages_from_otel((output,)) == (
        ExtractedOutputMessage(
            assistant_message=AssistantMessage(turn=(TextPart(text="done"),)),
            finish_reason="stop",
        ),
    )


def test_output_message_without_deprecated_finish_reason_parses_and_converts() -> None:
    """The selected schema makes the deprecated finish_reason field optional."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.output.messages": [
                {"role": "assistant", "parts": [{"type": "text", "content": "done"}]}
            ]
        })
    )
    assert parsed.output_messages is not None
    assert _output_messages_from_otel(parsed.output_messages) == (
        ExtractedOutputMessage(
            assistant_message=AssistantMessage(turn=(TextPart(text="done"),)),
            finish_reason=None,
        ),
    )


def test_reconstructs_provider_neutral_binding_fields() -> None:
    """reconstruct_bound_llm binds the representable request fields from OtelChatSpan."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.system_instructions": [{"type": "text", "content": "Be brief."}],
            "gen_ai.request.max_tokens": 200,
            "gen_ai.request.reasoning.level": "high",
            "gen_ai.request.temperature": 0.25,
        })
    )
    bound_llm = reconstruct_bound_llm(parsed, llm=LLM(_FakeAdapter()))
    assert bound_llm.binding.system_prompt == (TextPart(text="Be brief."),)
    assert bound_llm.binding.max_completion_tokens == 200
    assert bound_llm.binding.reasoning_level == "high"
    assert bound_llm.binding.temperature == 0.25


def test_reconstruction_rejects_a_different_llm_identity() -> None:
    """reconstruct_bound_llm rejects a provider or model mismatch."""
    parsed = parse_otel(
        _chat_span({"gen_ai.provider.name": "other", "gen_ai.request.model": "fake-model"})
    )
    with pytest.raises(ValueError, match="provider_name"):
        _ = reconstruct_bound_llm(parsed, llm=LLM(_FakeAdapter()))


def test_public_models_validate_direct_construction() -> None:
    """Public structured models use the same strict validation outside parse_otel."""
    message = OtelInputMessage(role="user", parts=(OtelTextPart(type="text", content="hi"),))
    assert generation_input_from_otel((message,)) == (UserMessage(content=(TextPart(text="hi"),)),)
