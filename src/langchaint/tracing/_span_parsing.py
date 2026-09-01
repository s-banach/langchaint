"""Parse OTel chat span attributes and convert supported values into langchaint values."""

import json
from dataclasses import dataclass
from typing import Annotated, Literal

import jsonschema
from pydantic import (
    AfterValidator,
    Base64UrlBytes,
    ConfigDict,
    Field,
    FiniteFloat,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from langchaint.checked_copy import CheckedCopyModel
from langchaint.llm import LLM, BoundLLM, GenerationInput
from langchaint.messages import (
    AssistantMessage,
    AudioPart,
    ContentPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
    Message,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    _is_object_dict,
)
from langchaint.tools import ToolSchema

OPERATION_NAME = "gen_ai.operation.name"
PROMPT_VARIABLE_PREFIX = "gen_ai.prompt.variable."
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

type StrictFiniteFloat = Annotated[FiniteFloat, Field(strict=True)]
type StringTuple = Annotated[tuple[str, ...], Field(strict=False)]

RAW_SPAN_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
BASE64_BYTES_ADAPTER: TypeAdapter[Base64UrlBytes] = TypeAdapter(Base64UrlBytes)
_OTEL_RAW_CONTEXT = {"otel_raw": True}


def _string_keyed_object_dict(value: object) -> dict[str, object]:
    if not _is_object_dict(value):
        raise PydanticCustomError(
            "otel_structured_object", "an OTel structured value must be an object"
        )
    result: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise PydanticCustomError(
                "otel_structured_object_key", "an OTel structured object key must be a string"
            )
        result[key] = item
    return result


def _validate_draft_07_schema(value: JsonValue) -> JsonValue:
    if not isinstance(value, (dict, bool)):
        raise PydanticCustomError(
            "json_schema", "parameters must be a JSON Schema draft-07 document"
        )
    try:
        jsonschema.Draft7Validator.check_schema(value)
    except jsonschema.SchemaError as error:
        raise PydanticCustomError(
            "json_schema", "parameters must be a JSON Schema draft-07 document"
        ) from error
    return value


type Draft7Schema = Annotated[JsonValue, AfterValidator(_validate_draft_07_schema)]


class OtelModel(CheckedCopyModel):
    """Pydantic validates an OTel value without constructing a langchaint value."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class OtelStructuredModel(OtelModel):
    """Pydantic validates declared fields and retains permitted additional JSON properties."""

    additional_properties: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _partition_additional_properties(
        cls, input_value: object, validation_info: ValidationInfo
    ) -> object:
        if isinstance(input_value, OtelStructuredModel):
            return input_value
        raw_object = _string_keyed_object_dict(input_value)
        raw_otel_value = (
            validation_info.context is not None and validation_info.context.get("otel_raw") is True
        )
        declared_values: dict[str, object] = {}
        additional_properties: dict[str, object] = {}
        for key, value in raw_object.items():
            if key in cls.model_fields and (key != "additional_properties" or not raw_otel_value):
                declared_values[key] = value
            else:
                additional_properties[key] = value
        if "additional_properties" in declared_values:
            supplied_additional_properties = declared_values.pop("additional_properties")
            validated_supplied = _string_keyed_object_dict(supplied_additional_properties)
            validated_supplied.update(additional_properties)
            additional_properties = validated_supplied
        declared_values["additional_properties"] = additional_properties
        return declared_values


class OtelTextPart(OtelStructuredModel):
    """Pydantic validates the declared text-part fields and retains additional properties."""

    type: Literal["text"]
    content: str


class OtelBlobPart(OtelStructuredModel):
    """Pydantic validates the declared blob-part fields and retains additional properties."""

    type: Literal["blob"]
    modality: str
    content: str
    mime_type: str | None = None


class OtelFilePart(OtelStructuredModel):
    """Pydantic validates the declared file-part fields and retains additional properties."""

    type: Literal["file"]
    modality: str
    file_id: str
    mime_type: str | None = None


class OtelUriPart(OtelStructuredModel):
    """Pydantic validates the declared URI-part fields and retains additional properties."""

    type: Literal["uri"]
    modality: str
    uri: str
    mime_type: str | None = None


class OtelImageUrlPart(OtelStructuredModel):
    """Pydantic validates langchaint's image URL extension and retains additional properties."""

    type: Literal["image_url"]
    url: str
    mime_type: str | None = None


class OtelReasoningPart(OtelStructuredModel):
    """Pydantic validates the declared reasoning-part fields and retains additional properties."""

    type: Literal["reasoning"]
    content: str


class OtelCompactionPart(OtelStructuredModel):
    """Pydantic validates the declared compaction-part fields and retains additional properties."""

    type: Literal["compaction"]
    id: str | None = None
    content: str | None = None


class OtelToolCallPart(OtelStructuredModel):
    """Pydantic validates the declared tool-call fields and retains additional properties."""

    type: Literal["tool_call"]
    name: str
    id: str | None = None
    arguments: JsonValue = None


class OtelToolCallResponsePart(OtelStructuredModel):
    """Pydantic validates the declared tool-response fields and retains additional properties."""

    type: Literal["tool_call_response"]
    response: JsonValue
    id: str | None = None


class OtelGenericServerToolCall(OtelStructuredModel):
    """Pydantic requires a server-tool type and retains its provider-defined properties."""

    type: str


class OtelGenericServerToolCallResponse(OtelStructuredModel):
    """Pydantic requires a server-tool-response type and retains provider-defined properties."""

    type: str


class OtelServerToolCallPart(OtelStructuredModel):
    """Pydantic validates the declared server-tool-call fields and retains additional properties."""

    type: Literal["server_tool_call"]
    name: str
    server_tool_call: OtelGenericServerToolCall
    id: str | None = None


class OtelServerToolCallResponsePart(OtelStructuredModel):
    """Pydantic validates server-tool-response fields and retains additional properties."""

    type: Literal["server_tool_call_response"]
    server_tool_call_response: OtelGenericServerToolCallResponse
    id: str | None = None


class OtelGenericPart(OtelStructuredModel):
    """Pydantic requires a part type and retains its properties for schema fallback."""

    type: str


class OtelGenericSystemInstructionPart(OtelStructuredModel):
    """Pydantic requires an instruction type and retains its properties for schema fallback."""

    type: str


type OtelMessagePart = Annotated[
    OtelTextPart
    | OtelToolCallPart
    | OtelToolCallResponsePart
    | OtelServerToolCallPart
    | OtelServerToolCallResponsePart
    | OtelBlobPart
    | OtelFilePart
    | OtelUriPart
    | OtelImageUrlPart
    | OtelReasoningPart
    | OtelCompactionPart
    | OtelGenericPart,
    Field(union_mode="left_to_right"),
]
type OtelSystemInstructionPart = Annotated[
    OtelTextPart | OtelGenericSystemInstructionPart,
    Field(union_mode="left_to_right"),
]


class OtelInputMessage(OtelStructuredModel):
    """Pydantic validates one input message while accepting provider-defined roles."""

    role: str
    parts: Annotated[tuple[OtelMessagePart, ...], Field(strict=False)]
    name: str | None = None


class OtelOutputMessage(OtelStructuredModel):
    """Pydantic validates one output message while accepting provider-defined values."""

    role: str
    parts: Annotated[tuple[OtelMessagePart, ...], Field(strict=False)]
    finish_reason: str | None = None
    name: str | None = None


class OtelFunctionTool(OtelStructuredModel):
    """Pydantic validates one function definition and its optional JSON Schema."""

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: Draft7Schema | None = None


class OtelGenericTool(OtelStructuredModel):
    """Pydantic requires tool identity fields and retains properties for schema fallback."""

    type: str
    name: str


type OtelToolDefinition = Annotated[
    OtelFunctionTool | OtelGenericTool,
    Field(union_mode="left_to_right"),
]

SYSTEM_INSTRUCTIONS_ADAPTER: TypeAdapter[tuple[OtelSystemInstructionPart, ...]] = TypeAdapter(
    tuple[OtelSystemInstructionPart, ...]
)
TOOL_DEFINITIONS_ADAPTER: TypeAdapter[tuple[OtelToolDefinition, ...]] = TypeAdapter(
    tuple[OtelToolDefinition, ...]
)
INPUT_MESSAGES_ADAPTER: TypeAdapter[tuple[OtelInputMessage, ...]] = TypeAdapter(
    tuple[OtelInputMessage, ...]
)
OUTPUT_MESSAGES_ADAPTER: TypeAdapter[tuple[OtelOutputMessage, ...]] = TypeAdapter(
    tuple[OtelOutputMessage, ...]
)
MESSAGE_PARTS_ADAPTER: TypeAdapter[tuple[OtelMessagePart, ...]] = TypeAdapter(
    tuple[OtelMessagePart, ...]
)


@dataclass(frozen=True, kw_only=True)
class ExtractedOutputMessage:
    assistant_message: AssistantMessage
    finish_reason: str | None


class OtelChatSpan(OtelModel):
    """Pydantic validates chat attributes against the committed OTel convention snapshot."""

    operation_name: Literal["chat"] = Field(alias=OPERATION_NAME)
    error_type: str | None = Field(default=None, alias="error.type")
    conversation_compacted: bool | None = Field(
        default=None, alias="gen_ai.conversation.compacted"
    )
    conversation_id: str | None = Field(default=None, alias="gen_ai.conversation.id")
    input_messages: tuple[OtelInputMessage, ...] | None = Field(default=None, alias=INPUT_MESSAGES)
    output_messages: tuple[OtelOutputMessage, ...] | None = Field(
        default=None, alias=OUTPUT_MESSAGES
    )
    output_type: str | None = Field(default=None, alias="gen_ai.output.type")
    prompt_name: str | None = Field(default=None, alias="gen_ai.prompt.name")
    prompt_variables: dict[str, str] = Field(default_factory=dict)
    prompt_version: str | None = Field(default=None, alias="gen_ai.prompt.version")
    provider_name: str | None = Field(default=None, alias="gen_ai.provider.name")
    request_choice_count: int | None = Field(default=None, alias="gen_ai.request.choice.count")
    request_frequency_penalty: StrictFiniteFloat | None = Field(
        default=None, alias="gen_ai.request.frequency_penalty"
    )
    request_max_tokens: int | None = Field(default=None, alias="gen_ai.request.max_tokens")
    request_model: str | None = Field(default=None, alias="gen_ai.request.model")
    request_presence_penalty: StrictFiniteFloat | None = Field(
        default=None, alias="gen_ai.request.presence_penalty"
    )
    request_previous_response_id: str | None = Field(
        default=None, alias="gen_ai.request.previous_response.id"
    )
    request_reasoning_level: str | None = Field(
        default=None, alias="gen_ai.request.reasoning.level"
    )
    request_seed: int | None = Field(default=None, alias="gen_ai.request.seed")
    request_stop_sequences: StringTuple | None = Field(
        default=None, alias="gen_ai.request.stop_sequences"
    )
    request_stream: bool | None = Field(default=None, alias="gen_ai.request.stream")
    request_temperature: StrictFiniteFloat | None = Field(
        default=None, alias="gen_ai.request.temperature"
    )
    request_top_k: int | None = Field(default=None, alias="gen_ai.request.top_k")
    request_top_p: StrictFiniteFloat | None = Field(default=None, alias="gen_ai.request.top_p")
    response_finish_reasons: StringTuple | None = Field(
        default=None, alias="gen_ai.response.finish_reasons"
    )
    response_id: str | None = Field(default=None, alias="gen_ai.response.id")
    response_model: str | None = Field(default=None, alias="gen_ai.response.model")
    response_time_to_first_chunk: StrictFiniteFloat | None = Field(
        default=None, alias="gen_ai.response.time_to_first_chunk"
    )
    system_instructions: tuple[OtelSystemInstructionPart, ...] | None = Field(
        default=None, alias=SYSTEM_INSTRUCTIONS
    )
    tool_definitions: tuple[OtelToolDefinition, ...] | None = Field(
        default=None, alias=TOOL_DEFINITIONS
    )
    usage_audio_cache_read_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.audio.cache_read.input_tokens"
    )
    usage_audio_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.audio.input_tokens"
    )
    usage_audio_output_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.audio.output_tokens"
    )
    usage_cache_read_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.cache_read.input_tokens"
    )
    usage_cache_write_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.cache_write.input_tokens"
    )
    usage_image_cache_read_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.image.cache_read.input_tokens"
    )
    usage_image_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.image.input_tokens"
    )
    usage_image_output_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.image.output_tokens"
    )
    usage_input_tokens: int | None = Field(default=None, alias="gen_ai.usage.input_tokens")
    usage_output_tokens: int | None = Field(default=None, alias="gen_ai.usage.output_tokens")
    usage_reasoning_output_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.reasoning.output_tokens"
    )
    usage_text_cache_read_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.text.cache_read.input_tokens"
    )
    usage_text_input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.text.input_tokens"
    )
    usage_text_output_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.text.output_tokens"
    )
    server_address: str | None = Field(default=None, alias="server.address")
    server_port: int | None = Field(default=None, alias="server.port")
    aws_bedrock_guardrail_id: str | None = Field(default=None, alias="aws.bedrock.guardrail.id")
    aws_bedrock_knowledge_base_id: str | None = Field(
        default=None, alias="aws.bedrock.knowledge_base.id"
    )
    azure_resource_provider_namespace: str | None = Field(
        default=None, alias="azure.resource_provider.namespace"
    )
    openai_api_type: str | None = Field(default=None, alias="openai.api.type")
    openai_request_service_tier: str | None = Field(
        default=None, alias="openai.request.service_tier"
    )
    openai_response_service_tier: str | None = Field(
        default=None, alias="openai.response.service_tier"
    )
    openai_response_system_fingerprint: str | None = Field(
        default=None, alias="openai.response.system_fingerprint"
    )
    unused_attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("*", mode="before")
    @classmethod
    def _reject_present_null(cls, attribute_value: object) -> object:
        if attribute_value is None:
            raise ValueError("a present OTel attribute cannot be null")
        return attribute_value

    @model_validator(mode="before")
    @classmethod
    def _partition_raw_span(cls, input_value: object) -> object:
        raw_span = RAW_SPAN_ADAPTER.validate_python(input_value)
        parsed_attributes: dict[str, JsonValue] = {
            name: value
            for name, value in raw_span.items()
            if name in _OTEL_CHAT_SPAN_FIXED_ALIASES
        }
        prompt_variables = {
            name.removeprefix(PROMPT_VARIABLE_PREFIX): value
            for name, value in raw_span.items()
            if name.startswith(PROMPT_VARIABLE_PREFIX)
        }
        parsed_attributes["prompt_variables"] = prompt_variables
        parsed_attributes["unused_attributes"] = {
            name: value
            for name, value in raw_span.items()
            if name not in _OTEL_CHAT_SPAN_FIXED_ALIASES
            and not name.startswith(PROMPT_VARIABLE_PREFIX)
        }
        return parsed_attributes

    @field_validator("system_instructions", mode="before")
    @classmethod
    def _parse_system_instructions(
        cls, attribute_value: JsonValue
    ) -> tuple[OtelSystemInstructionPart, ...]:
        return _validate_structured_attribute(SYSTEM_INSTRUCTIONS_ADAPTER, attribute_value)

    @field_validator("tool_definitions", mode="before")
    @classmethod
    def _parse_tool_definitions(cls, attribute_value: JsonValue) -> tuple[OtelToolDefinition, ...]:
        return _validate_structured_attribute(TOOL_DEFINITIONS_ADAPTER, attribute_value)

    @field_validator("input_messages", mode="before")
    @classmethod
    def _parse_input_messages(cls, attribute_value: JsonValue) -> tuple[OtelInputMessage, ...]:
        return _validate_structured_attribute(INPUT_MESSAGES_ADAPTER, attribute_value)

    @field_validator("output_messages", mode="before")
    @classmethod
    def _parse_output_messages(cls, attribute_value: JsonValue) -> tuple[OtelOutputMessage, ...]:
        return _validate_structured_attribute(OUTPUT_MESSAGES_ADAPTER, attribute_value)


_OTEL_CHAT_SPAN_FIXED_ALIASES: frozenset[str] = frozenset(
    field.alias
    for name, field in OtelChatSpan.model_fields.items()
    if name not in {"prompt_variables", "unused_attributes"} and field.alias is not None
)


class OtelToLangchaintConversionError(ValueError):
    """A valid OTel value has no lossless langchaint representation."""


def parse_otel(raw_attributes: dict[str, JsonValue]) -> OtelChatSpan:
    """Parse one deserialized OTel chat span attribute dictionary.

    `parse_otel` does not parse span metadata.
    Decoded JSON strings preserve values without preserving their original formatting.

    Args:
        raw_attributes: The deserialized chat span attributes.

    Raises:
        pydantic.ValidationError: A standard attribute is malformed or the operation is not `chat`.
    """
    return OtelChatSpan.model_validate(raw_attributes, context=_OTEL_RAW_CONTEXT)


def _system_prompt_from_otel(
    system_instructions: tuple[OtelSystemInstructionPart, ...],
) -> tuple[TextPart, ...]:
    """Convert supported OTel system instructions into langchaint text parts."""
    converted: list[TextPart] = []
    for part in system_instructions:
        if not isinstance(part, OtelTextPart):
            raise _unsupported(part, "system instruction type")
        _require_no_additional_properties(part)
        converted.append(TextPart(text=part.content))
    return tuple(converted)


def _tool_schemas_from_otel(
    tool_definitions: tuple[OtelToolDefinition, ...],
) -> tuple[ToolSchema, ...]:
    """Convert supported OTel function definitions into langchaint tool schemas."""
    converted: list[ToolSchema] = []
    for definition in tool_definitions:
        if not isinstance(definition, OtelFunctionTool):
            raise _unsupported(definition, "tool definition type")
        _require_no_additional_properties(definition)
        if (
            definition.description is None
            or definition.parameters is None
            or not isinstance(definition.parameters, dict)
        ):
            raise _unsupported(
                definition, "function definition without description and parameters"
            )
        converted.append(
            ToolSchema(
                name=definition.name,
                description=definition.description,
                args_schema=definition.parameters,
            )
        )
    return tuple(converted)


def generation_input_from_otel(
    input_messages: tuple[OtelInputMessage, ...],
) -> GenerationInput:
    """Convert supported OTel input messages into one langchaint generation input.

    Raises:
        OtelToLangchaintConversionError: An input value has no lossless langchaint representation.
    """
    return tuple(_message_from_otel(message) for message in input_messages)


def _output_messages_from_otel(
    output_messages: tuple[OtelOutputMessage, ...],
) -> tuple[ExtractedOutputMessage, ...]:
    """Convert supported OTel output messages into langchaint assistant messages."""
    converted: list[ExtractedOutputMessage] = []
    for message in output_messages:
        _require_message_metadata(message)
        if message.role != "assistant":
            raise _unsupported(message, "output message role")
        converted.append(
            ExtractedOutputMessage(
                assistant_message=_assistant_message_from_parts(message.parts),
                finish_reason=message.finish_reason,
            )
        )
    return tuple(converted)


def reconstruct_bound_llm(otel_chat_span: OtelChatSpan, *, llm: LLM) -> BoundLLM[str, None]:
    """Bind `system_instructions`, `request_max_tokens`, `request_reasoning_level`, and `request_temperature`.

    Args:
        otel_chat_span: The parsed OTel chat span attributes.
        llm: The provider SDK client state and request-admission configuration.

    Raises:
        ValueError: The provider name or model on `llm` differs from `otel_chat_span`.
        ValueError: A parsed binding field is invalid for the adapter.
        OtelToLangchaintConversionError: The system instructions have no langchaint representation.
    """
    if llm.adapter.provider_name != otel_chat_span.provider_name:
        raise ValueError(
            f"LLM provider_name {llm.adapter.provider_name!r} differs from parsed span "
            f"provider_name {otel_chat_span.provider_name!r}"
        )
    if llm.adapter.model != otel_chat_span.request_model:
        raise ValueError(
            f"LLM model {llm.adapter.model!r} differs from parsed span model "
            f"{otel_chat_span.request_model!r}"
        )
    system_prompt = (
        _system_prompt_from_otel(otel_chat_span.system_instructions)
        if otel_chat_span.system_instructions
        else None
    )
    return llm.bind(
        system_prompt=system_prompt,
        max_completion_tokens=otel_chat_span.request_max_tokens,
        reasoning_level=otel_chat_span.request_reasoning_level,
        temperature=otel_chat_span.request_temperature,
    )


def _validate_structured_attribute[ValueT](
    adapter: TypeAdapter[ValueT], attribute_value: JsonValue
) -> ValueT:
    if isinstance(attribute_value, str):
        return adapter.validate_json(attribute_value, context=_OTEL_RAW_CONTEXT)
    return adapter.validate_python(attribute_value, context=_OTEL_RAW_CONTEXT)


def _message_from_otel(message: OtelInputMessage) -> Message:
    _require_message_metadata(message)
    match message.role:
        case "user":
            return UserMessage(content=_content_parts_from_otel(message.parts))
        case "assistant":
            return _assistant_message_from_parts(message.parts)
        case "tool":
            return _tool_message_from_otel(message)
        case _:
            raise _unsupported(message, "input message role")


def _tool_message_from_otel(message: OtelInputMessage) -> ToolMessage:
    if len(message.parts) != 1 or not isinstance(message.parts[0], OtelToolCallResponsePart):
        raise _unsupported(message, "tool message parts")
    part = message.parts[0]
    if part.id is None:
        raise _unsupported(part, "tool response without id")
    is_error_value = part.additional_properties.get("is_error", False)
    if type(is_error_value) is not bool:
        raise _unsupported(part, "non-boolean is_error")
    remaining_properties = {
        key: value for key, value in part.additional_properties.items() if key != "is_error"
    }
    if remaining_properties:
        raise _unsupported(part, "additional properties")
    if isinstance(part.response, str):
        content: str | tuple[ContentPart, ...] = part.response
    elif isinstance(part.response, list):
        try:
            response_parts = _validate_structured_attribute(MESSAGE_PARTS_ADAPTER, part.response)
        except ValidationError as error:
            raise _unsupported(part, "tool response value") from error
        content = _content_parts_from_otel(response_parts)
    else:
        raise _unsupported(part, "tool response value")
    return ToolMessage(tool_call_id=part.id, content=content, is_error=is_error_value)


def _assistant_message_from_parts(parts: tuple[OtelMessagePart, ...]) -> AssistantMessage:
    converted: list[TextPart | ToolCall] = []
    for part in parts:
        if isinstance(part, OtelTextPart):
            _require_no_additional_properties(part)
            converted.append(TextPart(text=part.content))
        elif isinstance(part, OtelToolCallPart):
            _require_no_additional_properties(part)
            if part.id is None:
                raise _unsupported(part, "tool call without id")
            converted.append(
                ToolCall(
                    id=part.id,
                    name=part.name,
                    args_json=json.dumps(part.arguments, allow_nan=False, separators=(",", ":")),
                )
            )
        else:
            raise _unsupported(part, "assistant part type")
    return AssistantMessage(turn=tuple(converted))


def _content_parts_from_otel(parts: tuple[OtelMessagePart, ...]) -> tuple[ContentPart, ...]:
    return tuple(_content_part_from_otel(part) for part in parts)


def _content_part_from_otel(part: OtelMessagePart) -> ContentPart:
    _require_no_additional_properties(part)
    if isinstance(part, OtelTextPart):
        return TextPart(text=part.content)
    if isinstance(part, OtelBlobPart):
        if part.mime_type is None:
            raise _unsupported(part, "blob without mime_type")
        try:
            data = BASE64_BYTES_ADAPTER.validate_python(part.content)
        except ValidationError as error:
            raise OtelToLangchaintConversionError("blob content is not base64") from error
        if part.modality == "image":
            return ImagePart(data=data, media_type=part.mime_type)
        if part.modality == "audio":
            return AudioPart(data=data, media_type=part.mime_type)
        raise _unsupported(part, "blob modality")
    if isinstance(part, OtelUriPart) and part.modality == "image":
        return ImageUrlPart(url=part.uri, media_type=part.mime_type)
    if isinstance(part, OtelImageUrlPart):
        return ImageUrlPart(url=part.url, media_type=part.mime_type)
    raise _unsupported(part, "user content part type")


def _require_message_metadata(message: OtelInputMessage | OtelOutputMessage) -> None:
    if message.name is not None:
        raise _unsupported(message, "message name")
    _require_no_additional_properties(message)


def _require_no_additional_properties(value: OtelStructuredModel) -> None:
    if value.additional_properties:
        raise _unsupported(value, "additional properties")


def _unsupported(value: OtelModel, description: str) -> OtelToLangchaintConversionError:
    return OtelToLangchaintConversionError(
        f"{type(value).__name__} has no langchaint representation for {description}"
    )
