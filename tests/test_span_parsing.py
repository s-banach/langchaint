"""Test OTel-native chat span parsing and explicit langchaint conversion."""

import json
import math
import pathlib
from typing import assert_type

import pytest
from pydantic import BaseModel, ValidationError

from langchaint import (
    LLM,
    ZERO_USAGE,
    AssistantMessage,
    BoundLLM,
    DispatchHandled,
    ImagePart,
    JsonValue,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.span_parsing import (
    _OTEL_CHAT_SPAN_FIXED_ALIASES,
    _STRUCTURED_ATTRIBUTE_NAMES,
    ExtractedOutputMessage,
    OtelBlobPart,
    OtelChatSpan,
    OtelCompactionPart,
    OtelFilePart,
    OtelFunctionTool,
    OtelGenericPart,
    OtelGenericSystemInstructionPart,
    OtelGenericTool,
    OtelReasoningPart,
    OtelServerToolCallPart,
    OtelServerToolCallResponsePart,
    OtelTextPart,
    OtelToLangchaintConversionError,
    OtelToolCallPart,
    OtelToolCallResponsePart,
    OtelUriPart,
    generation_input_from_otel,
    output_messages_from_otel,
    parse_otel,
    reconstruct_bound_llm,
    response_record_from_otel,
    system_prompt_from_otel,
    tool_schemas_from_otel,
)
from langchaint.tools import JSONSchemaTool, ToolManager, ToolSchema
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


def _successful_chat_span(
    attributes: dict[str, JsonValue] | None = None,
) -> OtelChatSpan:
    raw_attributes: dict[str, JsonValue] = {
        "gen_ai.provider.name": "fake",
        "gen_ai.request.model": "fake-model",
        "gen_ai.response.finish_reasons": ["stop"],
        "gen_ai.output.messages": [
            {"role": "assistant", "parts": [{"type": "text", "content": "done"}]}
        ],
    }
    if attributes is not None:
        raw_attributes.update(attributes)
    return parse_otel(_chat_span(raw_attributes))


async def _return_tool_arguments(arguments: dict[str, object]) -> str:
    """Return the arguments as stable JSON text."""
    return json.dumps(arguments, sort_keys=True)


def _json_schema_tool(
    *,
    name: str = "lookup",
    description: str = "Look up one value.",
    args_schema: dict[str, object] | None = None,
) -> JSONSchemaTool[None]:
    return JSONSchemaTool(
        name=name,
        description=description,
        args_schema={"type": "object"} if args_schema is None else args_schema,
        function=_return_tool_arguments,
    )


def _captured_tool_definition(
    *,
    name: str = "lookup",
    description: str = "Look up one value.",
    parameters: JsonValue = None,
) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object"} if parameters is None else parameters,
    }


class _StructuredResponse(BaseModel):
    """Provide a concrete structured response type for binding tests."""

    answer: str


class _CountingTool:
    def __init__(self) -> None:
        self.name = "lookup"
        self.schema_calls = 0

    def schema(self) -> ToolSchema:
        self.schema_calls += 1
        return ToolSchema(
            name=self.name,
            description="Look up one value.",
            args_schema={"type": "object"},
        )

    async def dispatch(self, call: ToolCall) -> DispatchHandled[None]:
        return DispatchHandled(tool_message=ToolMessage(tool_call_id=call.id, content="done"))


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


def test_missing_tool_type_defaults_to_function() -> None:
    """A missing tool type selects the function schema variant."""
    tool_definition = _captured_tool_definition()
    del tool_definition["type"]
    parsed = parse_otel(_chat_span({"gen_ai.tool.definitions": [tool_definition]}))
    assert parsed.tool_definitions is not None
    definition = parsed.tool_definitions[0]
    assert isinstance(definition, OtelFunctionTool)
    assert definition.type == "function"


def test_decodes_json_strings_and_accepts_decoded_structured_values() -> None:
    """Structured attributes accept JSON text and its decoded JSON value."""
    value: list[JsonValue] = [{"type": "text", "content": "Follow the rules."}]
    from_text = parse_otel(_chat_span({"gen_ai.system_instructions": json.dumps(value)}))
    from_value = parse_otel(_chat_span({"gen_ai.system_instructions": value}))
    assert from_text.system_instructions == from_value.system_instructions
    assert from_text.system_instructions == (
        OtelTextPart(type="text", content="Follow the rules."),
    )


def test_decodes_each_schema_declared_structured_attribute() -> None:
    """The generated semantic-convention catalog selects JSON decoding."""
    assert {
        "gen_ai.input.messages",
        "gen_ai.memory.records",
        "gen_ai.output.messages",
        "gen_ai.retrieval.documents",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.tool.definitions",
    } == _STRUCTURED_ATTRIBUTE_NAMES
    encoded_value = json.dumps([])
    parsed = parse_otel(_chat_span({name: encoded_value for name in _STRUCTURED_ATTRIBUTE_NAMES}))
    parsed_by_alias = parsed.model_dump(by_alias=True)
    decoded_chat_attributes = _STRUCTURED_ATTRIBUTE_NAMES & _OTEL_CHAT_SPAN_FIXED_ALIASES
    for name in decoded_chat_attributes:
        assert parsed_by_alias[name] == ()
    for name in _STRUCTURED_ATTRIBUTE_NAMES - decoded_chat_attributes:
        assert parsed.unused_attributes[name] == []


def test_preserves_json_text_for_a_scalar_attribute() -> None:
    """A scalar semantic-convention attribute remains text when its value is valid JSON."""
    encoded_value = '{"provider": "custom"}'
    parsed = parse_otel(_chat_span({"gen_ai.provider.name": encoded_value}))
    assert parsed.provider_name == encoded_value


@pytest.mark.parametrize("encoded_value", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_json_constants_in_structured_attributes(encoded_value: str) -> None:
    """Reject non-JSON constants in structured semantic-convention attributes."""
    with pytest.raises(ValidationError, match="is not valid JSON"):
        _ = parse_otel(_chat_span({"gen_ai.input.messages": encoded_value}))


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
        _ = tool_schemas_from_otel(parsed)


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
        _ = generation_input_from_otel(parsed)


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
        _ = generation_input_from_otel(parsed)


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
    assert generation_input_from_otel(parsed) == (
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


@pytest.mark.parametrize("empty_system_instructions", [False, True])
def test_leading_system_message_supplies_the_binding_system_prompt(
    *, empty_system_instructions: bool
) -> None:
    """A leading system message supplies the binding and stays outside GenerationInput."""
    attributes: dict[str, JsonValue] = {
        "gen_ai.provider.name": "fake",
        "gen_ai.request.model": "fake-model",
        "gen_ai.input.messages": [
            {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
            {"role": "user", "parts": [{"type": "text", "content": "Question"}]},
        ],
    }
    if empty_system_instructions:
        attributes["gen_ai.system_instructions"] = list[JsonValue]()
    parsed = parse_otel(_chat_span(attributes))
    bound_llm = reconstruct_bound_llm(parsed, llm=LLM(_FakeAdapter()))
    assert bound_llm.binding.system_prompt == (TextPart(text="Be brief."),)
    assert generation_input_from_otel(parsed) == (
        UserMessage(content=(TextPart(text="Question"),)),
    )


def test_rejects_combined_system_instructions_and_system_message() -> None:
    """A nonempty system-instructions attribute conflicts with a system message."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.system_instructions": [{"type": "text", "content": "First"}],
            "gen_ai.input.messages": [
                {"role": "system", "parts": [{"type": "text", "content": "Second"}]}
            ],
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match="both provide the system prompt"):
        _ = generation_input_from_otel(parsed)
    with pytest.raises(OtelToLangchaintConversionError, match="both provide the system prompt"):
        _ = reconstruct_bound_llm(parsed, llm=LLM(_FakeAdapter()))


@pytest.mark.parametrize(
    ("input_messages", "error_match"),
    [
        (
            [
                {"role": "user", "parts": [{"type": "text", "content": "Question"}]},
                {"role": "system", "parts": [{"type": "text", "content": "Rules"}]},
            ],
            "after the first message",
        ),
        (
            [
                {"role": "system", "parts": [{"type": "text", "content": "First"}]},
                {"role": "system", "parts": [{"type": "text", "content": "Second"}]},
            ],
            "multiple role='system' messages",
        ),
    ],
)
def test_rejects_system_message_position_and_cardinality(
    input_messages: list[JsonValue], error_match: str
) -> None:
    """System messages require one entry at the start of input messages."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.input.messages": input_messages,
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match=error_match):
        _ = generation_input_from_otel(parsed)
    with pytest.raises(OtelToLangchaintConversionError, match=error_match):
        _ = reconstruct_bound_llm(parsed, llm=LLM(_FakeAdapter()))


@pytest.mark.parametrize(
    ("system_message", "error_match"),
    [
        ({"role": "system", "parts": []}, "system message without parts"),
        (
            {
                "role": "system",
                "parts": [
                    {
                        "type": "blob",
                        "modality": "image",
                        "mime_type": "image/png",
                        "content": "aW1hZ2U=",
                    }
                ],
            },
            "system instruction type",
        ),
        (
            {
                "role": "system",
                "name": "named",
                "parts": [{"type": "text", "content": "Rules"}],
            },
            "message name",
        ),
        (
            {
                "role": "system",
                "parts": [{"type": "text", "content": "Rules", "provider_value": 1}],
            },
            "additional properties",
        ),
    ],
)
def test_rejects_unrepresentable_leading_system_message(
    system_message: dict[str, JsonValue], error_match: str
) -> None:
    """A leading system message must convert losslessly to the binding system prompt."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.input.messages": [system_message],
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match=error_match):
        _ = generation_input_from_otel(parsed)


def test_converts_system_instructions_tools_and_output_messages() -> None:
    """Explicit conversion functions produce the prior langchaint values."""
    parsed = parse_otel(
        _chat_span({
            "gen_ai.system_instructions": [{"type": "text", "content": "Be brief."}],
            "gen_ai.tool.definitions": [_captured_tool_definition()],
            "gen_ai.output.messages": [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": "done"}],
                    "finish_reason": "stop",
                }
            ],
        })
    )
    assert system_prompt_from_otel(parsed) == (TextPart(text="Be brief."),)
    assert tool_schemas_from_otel(parsed) == (
        ToolSchema(
            name="lookup",
            description="Look up one value.",
            args_schema={"type": "object"},
        ),
    )
    assert output_messages_from_otel(parsed) == (
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
    assert output_messages_from_otel(parsed) == (
        ExtractedOutputMessage(
            assistant_message=AssistantMessage(turn=(TextPart(text="done"),)),
            finish_reason=None,
        ),
    )


def test_absent_optional_values_return_none() -> None:
    """Conversion functions return `None` when the span has no corresponding value."""
    parsed = parse_otel(_chat_span())
    assert system_prompt_from_otel(parsed) is None
    assert output_messages_from_otel(parsed) is None
    assert tool_schemas_from_otel(parsed) is None


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
    span = OtelChatSpan.model_validate({
        "gen_ai.operation.name": "chat",
        "gen_ai.input.messages": [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}],
    })
    assert generation_input_from_otel(span) == (UserMessage(content=(TextPart(text="hi"),)),)


def test_reconstructs_text_response_record_and_synthetic_fields() -> None:
    """response_record_from_otel preserves supported output and identity values."""
    span = _successful_chat_span({
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "before"},
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                    {"type": "text", "content": "after"},
                ],
            }
        ],
        "gen_ai.response.model": "served-model",
        "gen_ai.response.id": "response-1",
        "gen_ai.usage.input_tokens": 100,
        "openai.response.service_tier": "priority",
    })
    record = response_record_from_otel(span)
    assert record.output == "beforeafter"
    assert record.stop_reason == "end_turn"
    assert record.call.model == "fake-model"
    assert record.call.provider_name == "fake"
    assert record.call.elapsed_seconds == 0.0
    assert record.assistant_message == AssistantMessage(
        turn=(
            TextPart(text="before"),
            ToolCall(id="call-1", name="lookup", args_json='{"key":"value"}'),
            TextPart(text="after"),
        )
    )
    attempt = record.attempt_records[0]
    assert attempt.started_after_seconds == 0.0
    assert attempt.elapsed_seconds == 0.0
    assert attempt.seconds_to_first_item is None
    assert attempt.error is None
    assert attempt.model_served == "served-model"
    assert attempt.response_id == "response-1"
    assert attempt.request_id is None
    assert attempt.billing is not None
    assert attempt.billing.usage == ZERO_USAGE
    assert attempt.billing.service_tier == "unknown"
    assert math.isnan(attempt.billing.input_cache_none_usd_per_million_tokens)
    assert math.isnan(attempt.billing.cache_read_usd_per_million_tokens)
    assert math.isnan(attempt.billing.cache_write_usd_per_million_tokens)
    assert math.isnan(attempt.billing.output_usd_per_million_tokens)


@pytest.mark.parametrize("output", [42, [1, "two"], {"answer": True}])
def test_reconstructs_each_json_value_shape(output: JsonValue) -> None:
    """Declared JSON output accepts JSON scalars, arrays, and objects."""
    span = _successful_chat_span({
        "gen_ai.output.type": "json",
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": json.dumps(output)}],
            }
        ],
    })
    assert response_record_from_otel(span).output == output


def test_rejects_invalid_declared_json_output() -> None:
    """Declared JSON output rejects text that is not a JSON value."""
    span = _successful_chat_span({
        "gen_ai.output.type": "json",
        "gen_ai.output.messages": [
            {"role": "assistant", "parts": [{"type": "text", "content": "not json"}]}
        ],
    })
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.type"):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize("output_type", ["image", "speech", "provider_defined"])
def test_response_record_rejects_unsupported_output_type(output_type: str) -> None:
    """ResponseRecord[JsonValue] rejects each unsupported output type."""
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.type"):
        _ = response_record_from_otel(_successful_chat_span({"gen_ai.output.type": output_type}))


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "tool_call", "name": "lookup", "arguments": {}}],
        [{"type": "reasoning", "content": "thinking"}],
        [{"type": "provider_part", "value": 1}],
    ],
)
def test_response_record_rejects_unsupported_assistant_parts(parts: list[JsonValue]) -> None:
    """Output conversion rejects assistant parts without a lossless representation."""
    span = _successful_chat_span({
        "gen_ai.output.messages": [{"role": "assistant", "parts": parts}]
    })
    with pytest.raises(OtelToLangchaintConversionError):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "parts": [{"type": "text", "content": "done"}]},
        {
            "role": "assistant",
            "name": "named",
            "parts": [{"type": "text", "content": "done"}],
        },
        {
            "role": "assistant",
            "provider_value": 1,
            "parts": [{"type": "text", "content": "done"}],
        },
    ],
)
def test_response_record_rejects_unsupported_output_message_metadata(
    message: dict[str, JsonValue],
) -> None:
    """Output conversion rejects unsupported role, name, and additional properties."""
    span = _successful_chat_span({"gen_ai.output.messages": [message]})
    with pytest.raises(OtelToLangchaintConversionError):
        _ = response_record_from_otel(span)


def test_response_record_rejects_output_part_additional_properties() -> None:
    """Output conversion rejects additional properties on a supported part."""
    span = _successful_chat_span({
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "done", "provider_value": 1}],
            }
        ]
    })
    with pytest.raises(OtelToLangchaintConversionError, match="additional properties"):
        _ = response_record_from_otel(span)


def test_response_record_requires_output_messages() -> None:
    """Output conversion rejects an absent output_messages attribute."""
    span = _successful_chat_span().model_copy(update={"output_messages": None})
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.messages"):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize(
    "output_messages",
    [
        [],
        [
            {"role": "assistant", "parts": [{"type": "text", "content": "one"}]},
            {"role": "assistant", "parts": [{"type": "text", "content": "two"}]},
        ],
    ],
)
def test_response_record_requires_exactly_one_output_message(
    output_messages: list[JsonValue],
) -> None:
    """Output conversion rejects empty and multiple output message sequences."""
    span = _successful_chat_span({"gen_ai.output.messages": output_messages})
    with pytest.raises(OtelToLangchaintConversionError, match="exactly one"):
        _ = response_record_from_otel(span)


def test_response_record_rejects_error_type() -> None:
    """error.type marks a span as failed before output conversion."""
    with pytest.raises(OtelToLangchaintConversionError, match=r"error\.type"):
        _ = response_record_from_otel(_successful_chat_span({"error.type": "ProviderError"}))


def test_response_record_rejects_error_finish_reason() -> None:
    """The selected error finish reason marks a span as failed."""
    span = _successful_chat_span({"gen_ai.response.finish_reasons": ["error"]})
    with pytest.raises(OtelToLangchaintConversionError, match="finish_reason"):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [
        ("stop", "end_turn"),
        ("tool_call", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("context_window_exceeded", "context_window_exceeded"),
        ("other", "other"),
        ("provider_defined", "other"),
    ],
)
def test_response_record_maps_selected_finish_reason(finish_reason: str, stop_reason: str) -> None:
    """Finish-reason conversion maps known values and uses other for unknown values."""
    span = _successful_chat_span({"gen_ai.response.finish_reasons": [finish_reason]})
    assert response_record_from_otel(span).stop_reason == stop_reason


def test_response_record_falls_back_to_message_finish_reason() -> None:
    """The deprecated message finish_reason supplies the value when the span attribute is absent."""
    span = _successful_chat_span({
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "done"}],
                "finish_reason": "length",
            }
        ]
    }).model_copy(update={"response_finish_reasons": None})
    assert response_record_from_otel(span).stop_reason == "max_tokens"


def test_response_record_requires_matching_finish_reason_locations() -> None:
    """Span-level and message-level finish reasons must agree when both are present."""
    span = _successful_chat_span({
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "done"}],
                "finish_reason": "length",
            }
        ]
    })
    with pytest.raises(OtelToLangchaintConversionError, match="differs"):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize("finish_reasons", [[], ["stop", "length"]])
def test_response_record_requires_one_span_finish_reason(
    finish_reasons: list[str],
) -> None:
    """A present span finish-reason sequence must contain one value."""
    span = _successful_chat_span().model_copy(
        update={"response_finish_reasons": tuple(finish_reasons)}
    )
    with pytest.raises(OtelToLangchaintConversionError, match="exactly one"):
        _ = response_record_from_otel(span)


def test_response_record_requires_a_finish_reason() -> None:
    """Output conversion rejects spans with neither finish-reason location."""
    span = _successful_chat_span().model_copy(update={"response_finish_reasons": None})
    with pytest.raises(OtelToLangchaintConversionError, match="contain no value"):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize("field_name", ["provider_name", "request_model"])
@pytest.mark.parametrize("field_value", [None, ""])
def test_response_record_requires_call_identity(field_name: str, field_value: str | None) -> None:
    """CallRecord identity requires a provider name and requested model."""
    span = _successful_chat_span().model_copy(update={field_name: field_value})
    with pytest.raises(OtelToLangchaintConversionError, match="is required"):
        _ = response_record_from_otel(span)


@pytest.mark.parametrize("conversion_case", ["reasoning", "cardinality", "invalid_json"])
def test_response_record_errors_exclude_generated_content(conversion_case: str) -> None:
    """Conversion error text excludes generated content for each content-bearing failure."""
    secret = "generated-secret-value"
    if conversion_case == "reasoning":
        span = _successful_chat_span({
            "gen_ai.output.messages": [
                {
                    "role": "assistant",
                    "parts": [{"type": "reasoning", "content": secret}],
                }
            ]
        })
    elif conversion_case == "cardinality":
        span = _successful_chat_span({
            "gen_ai.output.messages": [
                {"role": "assistant", "parts": [{"type": "text", "content": secret}]},
                {"role": "assistant", "parts": [{"type": "text", "content": "done"}]},
            ]
        })
    else:
        span = _successful_chat_span({
            "gen_ai.output.type": "json",
            "gen_ai.output.messages": [
                {"role": "assistant", "parts": [{"type": "text", "content": secret}]}
            ],
        })
    with pytest.raises(OtelToLangchaintConversionError) as rejected:
        _ = response_record_from_otel(span)
    assert secret not in str(rejected.value)


def test_reconstruction_preserves_tool_manager_and_checks_captured_schemas() -> None:
    """Tool reconstruction preserves ToolManager identity and reads each caller schema once."""
    tool = _CountingTool()
    tool_manager = ToolManager([tool])
    span = _successful_chat_span({"gen_ai.tool.definitions": [_captured_tool_definition()]})
    bound_llm = reconstruct_bound_llm(span, llm=LLM(_FakeAdapter()), tools=tool_manager)
    assert bound_llm.tool_manager is tool_manager
    assert bound_llm.binding.tool_schemas == (
        ToolSchema(
            name="lookup",
            description="Look up one value.",
            args_schema={"type": "object"},
        ),
    )
    assert tool.schema_calls == 1


def test_reconstruction_converts_tool_sequence_once() -> None:
    """A caller tool sequence constructs one ToolManager before binding."""
    span = _successful_chat_span({"gen_ai.tool.definitions": [_captured_tool_definition()]})
    bound_llm = reconstruct_bound_llm(
        span,
        llm=LLM(_FakeAdapter()),
        tools=[_json_schema_tool()],
    )
    assert isinstance(bound_llm.tool_manager, ToolManager)
    assert bound_llm.binding.tool_schemas == (
        ToolSchema(
            name="lookup",
            description="Look up one value.",
            args_schema={"type": "object"},
        ),
    )


@pytest.mark.parametrize(
    "tools",
    [
        [_json_schema_tool(name="different")],
        [_json_schema_tool(description="Different description.")],
        [_json_schema_tool(args_schema={"type": "object", "required": ["value"]})],
        [_json_schema_tool(name="second"), _json_schema_tool(name="lookup")],
    ],
)
def test_reconstruction_rejects_tool_schema_mismatch(
    tools: list[JSONSchemaTool[None]],
) -> None:
    """Captured tool schemas must equal caller schemas in captured order."""
    if len(tools) == 1:
        span = _successful_chat_span({"gen_ai.tool.definitions": [_captured_tool_definition()]})
    else:
        span = _successful_chat_span({
            "gen_ai.tool.definitions": [
                _captured_tool_definition(),
                _captured_tool_definition(name="second"),
            ]
        })
    with pytest.raises(OtelToLangchaintConversionError, match="differs"):
        _ = reconstruct_bound_llm(span, llm=LLM(_FakeAdapter()), tools=tools)


def test_reconstruction_rejects_duplicate_captured_tool_names() -> None:
    """Captured tool definitions reject duplicate names."""
    span = _successful_chat_span({
        "gen_ai.tool.definitions": [
            _captured_tool_definition(),
            _captured_tool_definition(),
        ]
    })
    with pytest.raises(OtelToLangchaintConversionError, match="duplicate"):
        _ = reconstruct_bound_llm(span, llm=LLM(_FakeAdapter()), tools=None)


def test_reconstruction_rejects_duplicate_caller_tool_names() -> None:
    """ToolManager rejects duplicate caller tool names before binding."""
    span = _successful_chat_span()
    with pytest.raises(ValueError, match="duplicate tool name"):
        _ = reconstruct_bound_llm(
            span,
            llm=LLM(_FakeAdapter()),
            tools=[_json_schema_tool(), _json_schema_tool()],
        )


@pytest.mark.parametrize(
    ("definitions", "tools", "accepted"),
    [
        (None, None, True),
        (None, [_json_schema_tool()], True),
        ([], None, True),
        ([], [_json_schema_tool()], False),
        ([_captured_tool_definition()], None, False),
    ],
)
def test_reconstruction_applies_tool_definition_presence_contract(
    definitions: list[JsonValue] | None,
    tools: list[JSONSchemaTool[None]] | None,
    *,
    accepted: bool,
) -> None:
    """Absent definitions skip comparison while present definitions require exact schemas."""
    span = _successful_chat_span()
    if definitions:
        span = _successful_chat_span({"gen_ai.tool.definitions": [_captured_tool_definition()]})
    elif definitions == []:
        span = span.model_copy(update={"tool_definitions": ()})
    if accepted:
        _ = reconstruct_bound_llm(span, llm=LLM(_FakeAdapter()), tools=tools)
    else:
        with pytest.raises(OtelToLangchaintConversionError, match="differs"):
            _ = reconstruct_bound_llm(span, llm=LLM(_FakeAdapter()), tools=tools)


def test_reconstruction_binds_structured_response_model() -> None:
    """JSON output binds the caller-supplied structured response model."""
    span = _successful_chat_span({"gen_ai.output.type": "json"})
    bound_llm = reconstruct_bound_llm(
        span,
        llm=LLM(_FakeAdapter()),
        response_format=_StructuredResponse,
    )
    assert_type(bound_llm, BoundLLM[_StructuredResponse, None])
    assert bound_llm.response_format is _StructuredResponse


@pytest.mark.parametrize(
    ("output_type", "response_format"),
    [("json", None), ("text", _StructuredResponse), (None, _StructuredResponse)],
)
def test_reconstruction_rejects_response_format_mismatch(
    output_type: str | None,
    response_format: type[BaseModel] | None,
) -> None:
    """output_type and response_format must select the same output form."""
    span = _successful_chat_span()
    if output_type is not None:
        span = _successful_chat_span({"gen_ai.output.type": output_type})
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.type"):
        _ = reconstruct_bound_llm(
            span,
            llm=LLM(_FakeAdapter()),
            response_format=response_format,
        )


@pytest.mark.parametrize("output_type", ["image", "speech", "provider_defined"])
def test_reconstruction_rejects_unsupported_output_type(output_type: str) -> None:
    """Binding reconstruction rejects unsupported output types."""
    span = _successful_chat_span({"gen_ai.output.type": output_type})
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.type"):
        _ = reconstruct_bound_llm(span, llm=LLM(_FakeAdapter()))
