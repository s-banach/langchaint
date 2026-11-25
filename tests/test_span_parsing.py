"""Test OTel-native chat span parsing and explicit langchaint conversion."""

import json
import pathlib
from typing import assert_type

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint import (
    LLM,
    AssistantMessage,
    JsonValue,
    TextPart,
    ToolMessage,
    UserMessage,
    llm_from_adapter,
)
from langchaint.span_parsing import (
    OtelChatSpan,
    OtelExecuteToolSpan,
    OtelFunctionTool,
    OtelGenericObject,
    OtelReasoningPart,
    OtelTextPart,
    OtelToLangchaintConversionError,
    generation_input_from_otel,
    llm_from_otel,
    output_messages_from_otel,
    parse_otel_chat,
    parse_otel_execute_tool,
    system_prompt_from_otel,
    tool_schemas_from_otel,
)
from langchaint.tools import JSONSchemaTool, ToolSchema
from scripts import refresh_semconv_genai
from tests.fake_adapter import FakeAdapter

SEMCONV_DIRECTORY = pathlib.Path(__file__).parent / "semconv_genai"
_VALID_AND_INVALID_VALUE_BY_OTEL_TYPE: dict[str, tuple[JsonValue, JsonValue]] = {
    "boolean": (True, "true"),
    "double": (0.5, "0.5"),
    "int": (1, 1.5),
    "string": ("value", 1),
    "string[]": (["value"], [1]),
}


def _chat_span(attributes: dict[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    span_attributes: dict[str, JsonValue] = {"gen_ai.operation.name": "chat"}
    if attributes is not None:
        span_attributes.update(attributes)
    return span_attributes


def _generation_chat_span(
    attributes: dict[str, JsonValue] | None = None,
) -> OtelChatSpan:
    span_attributes: dict[str, JsonValue] = {
        "gen_ai.provider.name": "fake",
        "gen_ai.request.model": "fake-model",
    }
    if attributes is not None:
        span_attributes.update(attributes)
    return parse_otel_chat(_chat_span(span_attributes))


async def _return_tool_arguments(arguments: dict[str, object]) -> str:
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
    answer: str


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
        parsed = parse_otel_chat(_chat_span({attribute_name: valid_value}))
        actual_value = parsed.model_dump()[field_names_by_alias[attribute_name]]
        if value_type == "string[]":
            assert isinstance(valid_value, list)
            expected_value: object = tuple(valid_value)
        else:
            expected_value = valid_value
        assert actual_value == expected_value
        with pytest.raises(ValidationError):
            _ = parse_otel_chat(_chat_span({attribute_name: invalid_value}))


def test_manifest_attributes_match_otel_chat_span_aliases() -> None:
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
    field_names_by_alias: dict[str, str] = {}
    for field_name, field in OtelChatSpan.model_fields.items():
        if field_name != "unused_attributes":
            assert field.alias is not None
            field_names_by_alias[field.alias] = field_name
    assert set(entries_by_name) == set(field_names_by_alias)
    _assert_scalar_attributes_parse(entries_by_name, field_names_by_alias)


def test_decodes_json_strings_and_accepts_decoded_structured_values() -> None:
    value: list[JsonValue] = [{"type": "text", "content": "Follow the rules."}]
    from_text = parse_otel_chat(_chat_span({"gen_ai.system_instructions": json.dumps(value)}))
    from_value = parse_otel_chat(_chat_span({"gen_ai.system_instructions": value}))
    assert from_text.system_instructions == from_value.system_instructions
    assert from_text.system_instructions == (
        OtelTextPart(type="text", content="Follow the rules."),
    )


def test_decodes_structured_unused_attributes_and_preserves_unrecognized_text() -> None:
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.tool.call.arguments": '{"query":"x"}',
            "custom.payload": '{"query":"x"}',
        })
    )
    assert parsed.unused_attributes == {
        "gen_ai.tool.call.arguments": {"query": "x"},
        "custom.payload": '{"query":"x"}',
    }


def test_rejects_non_json_constants_in_structured_attributes() -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        _ = parse_otel_chat(_chat_span({"gen_ai.input.messages": "NaN"}))


def test_preserves_known_and_generic_additional_properties() -> None:
    parsed = parse_otel_chat(
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
    assert isinstance(generic_part, OtelGenericObject)
    assert generic_part.additional_properties == {"provider_value": True}


def test_preserves_a_raw_additional_properties_property() -> None:
    parsed = parse_otel_chat(
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


def test_parses_content_references_and_preserves_unused_attributes() -> None:
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.provider.name": "openai",
            "gen_ai.input.messages_ref": "s3://example/input.json",
            "gen_ai.output.messages_ref": "s3://example/output.json",
            "gen_ai.system_instructions_ref": "s3://example/system.json",
            "gen_ai.tool.definitions_ref": "s3://example/tools.json",
            "gen_ai.prompt.variable.user_name": "Alice",
        })
    )
    assert parsed.provider_name == "openai"
    assert parsed.input_messages_ref == "s3://example/input.json"
    assert parsed.output_messages_ref == "s3://example/output.json"
    assert parsed.system_instructions_ref == "s3://example/system.json"
    assert parsed.tool_definitions_ref == "s3://example/tools.json"
    assert parsed.prompt_variables == {}
    assert parsed.unused_attributes == {"gen_ai.prompt.variable.user_name": "Alice"}


def test_rejects_malformed_structured_attribute_json() -> None:
    with pytest.raises(ValidationError):
        _ = parse_otel_chat(_chat_span({"gen_ai.input.messages": "not JSON"}))


def test_rejects_non_object_structured_values() -> None:
    with pytest.raises(ValidationError, match="an OTel structured value must be an object"):
        _ = parse_otel_chat(
            _chat_span({"gen_ai.input.messages": [{"role": "user", "parts": [1]}]})
        )


def test_rejects_a_malformed_function_parameters_schema() -> None:
    with pytest.raises(ValidationError, match="JSON Schema draft-07"):
        _ = OtelFunctionTool(type="function", name="lookup", parameters={"type": 42})


def test_parses_boolean_draft_07_schema_before_conversion_fails() -> None:
    parsed = parse_otel_chat(
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
    with pytest.raises(OtelToLangchaintConversionError, match="not a schema object"):
        _ = tool_schemas_from_otel(parsed)


def test_parsing_succeeds_before_unsupported_conversion_fails() -> None:
    parsed = parse_otel_chat(
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
    parsed = parse_otel_chat(
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


def test_converts_a_string_tool_response_to_string_content() -> None:
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "tool",
                    "parts": [
                        {"type": "tool_call_response", "id": "call-1", "response": "failed"}
                    ],
                }
            ]
        })
    )
    assert generation_input_from_otel(parsed) == (
        ToolMessage(tool_call_id="call-1", content="failed"),
    )


def test_non_boolean_is_error_fails_tool_message_conversion() -> None:
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.input.messages": [
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": "call-1",
                            "is_error": "yes",
                            "response": "failed",
                        }
                    ],
                }
            ]
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match="non-boolean is_error"):
        _ = generation_input_from_otel(parsed)


@pytest.mark.parametrize("empty_system_instructions", [False, True])
def test_leading_system_message_supplies_the_binding_system_prompt(
    *, empty_system_instructions: bool
) -> None:
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
    parsed = parse_otel_chat(_chat_span(attributes))
    llm = llm_from_otel(parsed, llm=llm_from_adapter(FakeAdapter()))
    assert llm.binding.system_prompt == (TextPart(text="Be brief."),)
    assert generation_input_from_otel(parsed) == (
        UserMessage(content=(TextPart(text="Question"),)),
    )


def test_rejects_combined_system_instructions_and_system_message() -> None:
    parsed = parse_otel_chat(
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
        _ = llm_from_otel(parsed, llm=llm_from_adapter(FakeAdapter()))


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
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.input.messages": input_messages,
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match=error_match):
        _ = generation_input_from_otel(parsed)
    with pytest.raises(OtelToLangchaintConversionError, match=error_match):
        _ = llm_from_otel(parsed, llm=llm_from_adapter(FakeAdapter()))


@pytest.mark.parametrize(
    ("system_message", "error_match"),
    [
        ({"role": "system", "parts": []}, "role='system' message without parts"),
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
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.input.messages": [system_message],
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match=error_match):
        _ = generation_input_from_otel(parsed)


def test_converts_every_output_message_without_span_identity_or_finish_reason() -> None:
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.output.messages": [
                {"role": "assistant", "parts": [{"type": "text", "content": "one"}]},
                {"role": "assistant", "parts": [{"type": "text", "content": "two"}]},
            ]
        })
    )
    assert output_messages_from_otel(parsed) == (
        AssistantMessage(content=(TextPart(text="one"),)),
        AssistantMessage(content=(TextPart(text="two"),)),
    )


def test_function_definition_without_description_converts() -> None:
    parsed = parse_otel_chat(
        _chat_span({
            "gen_ai.tool.definitions": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
            ]
        })
    )
    assert tool_schemas_from_otel(parsed) == (
        ToolSchema(name="lookup", args_schema={"type": "object"}),
    )


def test_absent_optional_values_return_none() -> None:
    parsed = parse_otel_chat(_chat_span())
    assert system_prompt_from_otel(parsed) is None
    assert output_messages_from_otel(parsed) is None
    assert tool_schemas_from_otel(parsed) is None


def test_external_input_messages_prevent_input_and_system_prompt_conversion() -> None:
    span = _generation_chat_span({"gen_ai.input.messages_ref": "s3://example/input"})
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.input\.messages_ref"):
        _ = generation_input_from_otel(span)
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.input\.messages_ref"):
        _ = system_prompt_from_otel(span)


def test_external_system_instructions_block_only_conversions_that_need_them() -> None:
    external = "s3://example/system"
    span = _generation_chat_span({
        "gen_ai.system_instructions_ref": external,
        "gen_ai.input.messages": [{"role": "user", "parts": [{"type": "text", "content": "Hi"}]}],
    })
    assert generation_input_from_otel(span) == (UserMessage(content=(TextPart(text="Hi"),)),)
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.system_instructions_ref"):
        _ = llm_from_otel(span, llm=llm_from_adapter(FakeAdapter()))
    leading_system_message = _generation_chat_span({
        "gen_ai.system_instructions_ref": external,
        "gen_ai.input.messages": [
            {"role": "system", "parts": [{"type": "text", "content": "Hi"}]}
        ],
    })
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.system_instructions_ref"):
        _ = generation_input_from_otel(leading_system_message)


def test_input_conversion_accepts_separate_unrepresentable_system_instructions() -> None:
    span = _generation_chat_span({
        "gen_ai.system_instructions": [{"type": "text", "content": "Rules", "vendor": 1}],
        "gen_ai.input.messages": [{"role": "user", "parts": [{"type": "text", "content": "Hi"}]}],
    })
    assert generation_input_from_otel(span) == (UserMessage(content=(TextPart(text="Hi"),)),)
    with pytest.raises(OtelToLangchaintConversionError, match="additional properties"):
        _ = system_prompt_from_otel(span)


def test_external_outputs_and_tool_definitions_require_inline_content_for_conversion() -> None:
    span = parse_otel_chat(
        _chat_span({
            "gen_ai.provider.name": "fake",
            "gen_ai.request.model": "fake-model",
            "gen_ai.output.messages_ref": "s3://example/output",
            "gen_ai.tool.definitions_ref": "s3://example/tools",
        })
    )
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.messages_ref"):
        _ = output_messages_from_otel(span)
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.tool\.definitions_ref"):
        _ = llm_from_otel(span, llm=llm_from_adapter(FakeAdapter()))


def test_inline_empty_content_takes_precedence_over_references() -> None:
    span = _generation_chat_span({
        "gen_ai.input.messages": [],
        "gen_ai.input.messages_ref": "s3://example/input",
        "gen_ai.output.messages": [],
        "gen_ai.output.messages_ref": "s3://example/output",
        "gen_ai.system_instructions": [],
        "gen_ai.system_instructions_ref": "s3://example/system",
        "gen_ai.tool.definitions": [],
        "gen_ai.tool.definitions_ref": "s3://example/tools",
    })
    assert generation_input_from_otel(span) == ()
    assert output_messages_from_otel(span) == ()
    assert system_prompt_from_otel(span) is None
    assert tool_schemas_from_otel(span) == ()


def test_reconstruction_rejects_a_different_llm_identity() -> None:
    parsed = parse_otel_chat(
        _chat_span({"gen_ai.provider.name": "other", "gen_ai.request.model": "fake-model"})
    )
    with pytest.raises(ValueError, match="provider_name"):
        _ = llm_from_otel(parsed, llm=llm_from_adapter(FakeAdapter()))


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "tool_call", "name": "lookup", "arguments": {}}],
        [{"type": "tool_call", "id": "call1", "name": "lookup", "arguments": '{"query": '}],
        [{"type": "provider_part", "value": 1}],
    ],
)
def test_output_messages_reject_unsupported_assistant_parts(parts: list[JsonValue]) -> None:
    span = _generation_chat_span({
        "gen_ai.output.messages": [{"role": "assistant", "parts": parts}]
    })
    with pytest.raises(OtelToLangchaintConversionError):
        _ = output_messages_from_otel(span)


def test_output_message_conversion_error_excludes_generated_content() -> None:
    secret = "generated-secret-value"
    span = _generation_chat_span({
        "gen_ai.output.messages": [
            {"role": "assistant", "parts": [{"type": "reasoning", "content": secret}]}
        ]
    })
    with pytest.raises(OtelToLangchaintConversionError) as rejected:
        _ = output_messages_from_otel(span)
    assert secret not in str(rejected.value)


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
def test_output_messages_reject_unsupported_metadata(
    message: dict[str, JsonValue],
) -> None:
    span = _generation_chat_span({"gen_ai.output.messages": [message]})
    with pytest.raises(OtelToLangchaintConversionError):
        _ = output_messages_from_otel(span)


def test_output_messages_reject_part_additional_properties() -> None:
    span = _generation_chat_span({
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "done", "provider_value": 1}],
            }
        ]
    })
    with pytest.raises(OtelToLangchaintConversionError, match="additional properties"):
        _ = output_messages_from_otel(span)


@pytest.mark.parametrize(
    "tools",
    [
        [_json_schema_tool(description="Different description.")],
        [_json_schema_tool(name="second"), _json_schema_tool(name="lookup")],
    ],
)
def test_reconstruction_rejects_tool_schema_mismatch(
    tools: list[JSONSchemaTool[None]],
) -> None:
    if len(tools) == 1:
        span = _generation_chat_span({"gen_ai.tool.definitions": [_captured_tool_definition()]})
    else:
        span = _generation_chat_span({
            "gen_ai.tool.definitions": [
                _captured_tool_definition(),
                _captured_tool_definition(name="second"),
            ]
        })
    with pytest.raises(OtelToLangchaintConversionError, match="differs"):
        _ = llm_from_otel(span, llm=llm_from_adapter(FakeAdapter(), tools=tools))


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
    span = _generation_chat_span()
    if definitions is not None:
        span = _generation_chat_span({"gen_ai.tool.definitions": definitions})
    if accepted:
        _ = llm_from_otel(span, llm=llm_from_adapter(FakeAdapter(), tools=tools))
    else:
        with pytest.raises(OtelToLangchaintConversionError, match="differs"):
            _ = llm_from_otel(span, llm=llm_from_adapter(FakeAdapter(), tools=tools))


def test_reconstruction_binds_structured_response_model() -> None:
    span = _generation_chat_span({"gen_ai.output.type": "json"})
    llm = llm_from_otel(
        span,
        llm=llm_from_adapter(FakeAdapter(), response_format=_StructuredResponse),
    )
    assert_type(llm, LLM[_StructuredResponse, None])
    assert llm.response_format is _StructuredResponse


@pytest.mark.parametrize(
    ("output_type", "response_format"),
    [("json", str), ("text", _StructuredResponse), (None, _StructuredResponse)],
)
def test_reconstruction_rejects_response_format_mismatch(
    output_type: str | None,
    response_format: type[BaseModel] | type[str],
) -> None:
    span = _generation_chat_span()
    if output_type is not None:
        span = _generation_chat_span({"gen_ai.output.type": output_type})
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.type"):
        _ = llm_from_otel(
            span,
            llm=llm_from_adapter(FakeAdapter(), response_format=response_format),
        )


def test_reconstruction_rejects_unsupported_output_type() -> None:
    span = _generation_chat_span({"gen_ai.output.type": "image"})
    with pytest.raises(OtelToLangchaintConversionError, match=r"gen_ai\.output\.type"):
        _ = llm_from_otel(span, llm=llm_from_adapter(FakeAdapter()))


def test_manifest_attribute_names_match_otel_execute_tool_span_aliases() -> None:
    names = TypeAdapter(frozenset[str]).validate_json(
        (SEMCONV_DIRECTORY / "execute-tool-span-attribute-names.json").read_text()
    )
    aliases = {
        field.alias
        for field_name, field in OtelExecuteToolSpan.model_fields.items()
        if field_name != "unused_attributes"
    }
    assert aliases == names


@pytest.mark.parametrize(
    "result",
    [
        {"temperature": 20},
        {"type": "tool_call_response", "response": "done", "is_error": "yes"},
    ],
)
def test_execute_tool_result_of_another_shape_parses_as_a_json_value(
    result: dict[str, JsonValue],
) -> None:
    parsed = parse_otel_execute_tool({
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.call.result": json.dumps(result),
    })
    assert parsed.tool_call_result == result


@pytest.mark.parametrize(
    ("attribute_name", "field_name"),
    [
        ("gen_ai.tool.call.arguments", "tool_call_arguments"),
        ("gen_ai.tool.call.result", "tool_call_result"),
    ],
)
def test_execute_tool_null_requires_encoded_json(attribute_name: str, field_name: str) -> None:
    absent = parse_otel_execute_tool({"gen_ai.operation.name": "execute_tool"})
    encoded_null = parse_otel_execute_tool({
        "gen_ai.operation.name": "execute_tool",
        attribute_name: "null",
    })
    assert encoded_null.tool_call_arguments is None
    assert encoded_null.tool_call_result is None
    assert field_name not in absent.model_fields_set
    assert field_name in encoded_null.model_fields_set
    with pytest.raises(ValidationError, match="a present OTel attribute cannot be null"):
        _ = parse_otel_execute_tool({
            "gen_ai.operation.name": "execute_tool",
            attribute_name: None,
        })
