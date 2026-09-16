"""Parse OTel chat span attributes and convert supported values into langchaint values."""

import json
from importlib.resources import files
from typing import Annotated, Literal, overload

import jsonschema
from pydantic import (
    AfterValidator,
    Base64UrlBytes,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

from langchaint.billing.pricing import Billing
from langchaint.billing.usage import ZERO_USAGE
from langchaint.common.checked_copy import CheckedCopyModel
from langchaint.common.messages import (
    AssistantMessage,
    AudioPart,
    ContentPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
    Message,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    _is_object_dict,
)
from langchaint.generation.call import CallRecord, SettledAttemptRecord
from langchaint.generation.llm import LLM, BoundLLM, GenerationInput
from langchaint.generation.response import ResponseRecord
from langchaint.tools import ToolManager, ToolSchema, ToolSequence

OPERATION_NAME = "gen_ai.operation.name"
PROMPT_VARIABLE_PREFIX = "gen_ai.prompt.variable."
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

type StrictFiniteFloat = Annotated[FiniteFloat, Field(strict=True)]
type StringTuple = Annotated[tuple[str, ...], Field(strict=False)]

_RAW_SPAN_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_BASE64_BYTES_ADAPTER: TypeAdapter[Base64UrlBytes] = TypeAdapter(Base64UrlBytes)
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_STRUCTURED_ATTRIBUTE_NAMES_ADAPTER: TypeAdapter[frozenset[str]] = TypeAdapter(frozenset[str])
_STRUCTURED_ATTRIBUTE_NAMES = _STRUCTURED_ATTRIBUTE_NAMES_ADAPTER.validate_json(
    files("langchaint").joinpath("_semconv_genai_structured_attributes.json").read_text()
)


def _decode_semconv_attribute(name: str, value: JsonValue) -> JsonValue:
    if name not in _STRUCTURED_ATTRIBUTE_NAMES or not isinstance(value, str):
        return value
    return _JSON_VALUE_ADAPTER.validate_json(value)


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
    """Pydantic validates declared fields and retains permitted additional JSON properties.

    Every dictionary input is a raw OTel object, so a raw property named `additional_properties`
    is retained as an additional property.
    Strict validation of `additional_properties` rejects a non-string key.
    """

    additional_properties: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _partition_additional_properties(cls, input_value: object) -> object:
        if isinstance(input_value, OtelStructuredModel):
            return input_value
        if not _is_object_dict(input_value):
            raise PydanticCustomError(
                "otel_structured_object", "an OTel structured value must be an object"
            )
        declared_values: dict[str, object] = {
            name: input_value[name]
            for name in cls.model_fields
            if name != "additional_properties" and name in input_value
        }
        declared_values["additional_properties"] = {
            key: value for key, value in input_value.items() if key not in declared_values
        }
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
    """Pydantic validates the image_url part in spans from earlier langchaint releases.

    Current langchaint releases emit an `ImageUrlPart` as a `uri` part with `modality="image"`.
    """

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
    """Pydantic validates the declared tool-response fields and retains additional properties.

    `is_error` is langchaint's extension that records `ToolMessage.is_error`.
    A part whose `is_error` is not a boolean validates as `OtelGenericObject` instead.
    """

    type: Literal["tool_call_response"]
    response: JsonValue
    id: str | None = None
    is_error: bool = False


class OtelGenericObject(OtelStructuredModel):
    """Pydantic requires a type and retains every other property of an OTel object.

    `OtelGenericObject` is the fallback for a message part, a system instruction part, and the
    provider-defined objects nested in server tool call parts.
    """

    type: str


class OtelServerToolCallPart(OtelStructuredModel):
    """Pydantic validates the declared server-tool-call fields and retains additional properties."""

    type: Literal["server_tool_call"]
    name: str
    server_tool_call: OtelGenericObject
    id: str | None = None


class OtelServerToolCallResponsePart(OtelStructuredModel):
    """Pydantic validates server-tool-response fields and retains additional properties."""

    type: Literal["server_tool_call_response"]
    server_tool_call_response: OtelGenericObject
    id: str | None = None


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
    | OtelGenericObject,
    Field(union_mode="left_to_right"),
]
type OtelSystemInstructionPart = Annotated[
    OtelTextPart | OtelGenericObject,
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

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: Draft7Schema | None = None


class OtelGenericTool(OtelStructuredModel):
    """Pydantic requires tool identity fields and retains properties for schema fallback."""

    type: str
    name: str


# The OTel schema requires `type`, but OpenLLMetry omits it on tool definitions.
# `OtelFunctionTool.type` defaults to `"function"` so those definitions convert.
type OtelToolDefinition = Annotated[
    OtelFunctionTool | OtelGenericTool,
    Field(union_mode="left_to_right"),
]

_MESSAGE_PARTS_ADAPTER: TypeAdapter[tuple[OtelMessagePart, ...]] = TypeAdapter(
    tuple[OtelMessagePart, ...]
)


class OtelChatSpan(OtelModel):
    """Pydantic validates chat attributes against the committed OTel convention snapshot."""

    operation_name: Literal["chat"] = Field(alias=OPERATION_NAME)
    error_type: str | None = Field(default=None, alias="error.type")
    conversation_compacted: bool | None = Field(
        default=None, alias="gen_ai.conversation.compacted"
    )
    conversation_id: str | None = Field(default=None, alias="gen_ai.conversation.id")
    input_messages: tuple[OtelInputMessage, ...] | None = Field(
        default=None, alias=INPUT_MESSAGES, strict=False
    )
    output_messages: tuple[OtelOutputMessage, ...] | None = Field(
        default=None, alias=OUTPUT_MESSAGES, strict=False
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
        default=None, alias=SYSTEM_INSTRUCTIONS, strict=False
    )
    tool_definitions: tuple[OtelToolDefinition, ...] | None = Field(
        default=None, alias=TOOL_DEFINITIONS, strict=False
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

    @model_validator(mode="before")
    @classmethod
    def _partition_raw_span(cls, input_value: object) -> object:
        raw_span = _RAW_SPAN_ADAPTER.validate_python(input_value)
        parsed_attributes: dict[str, JsonValue] = {}
        prompt_variables: dict[str, JsonValue] = {}
        unused_attributes: dict[str, JsonValue] = {}
        for name, value in raw_span.items():
            if name in _OTEL_CHAT_SPAN_FIXED_ALIASES:
                if value is None:
                    raise ValueError(f"a present OTel attribute cannot be null: {name}")
                parsed_attributes[name] = _decode_semconv_attribute(name, value)
            elif name.startswith(PROMPT_VARIABLE_PREFIX):
                prompt_variables[name.removeprefix(PROMPT_VARIABLE_PREFIX)] = value
            else:
                unused_attributes[name] = _decode_semconv_attribute(name, value)
        parsed_attributes["prompt_variables"] = prompt_variables
        parsed_attributes["unused_attributes"] = unused_attributes
        return parsed_attributes


_OTEL_CHAT_SPAN_FIXED_ALIASES: frozenset[str] = frozenset(
    field.alias for field in OtelChatSpan.model_fields.values() if field.alias is not None
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
    return OtelChatSpan.model_validate(raw_attributes)


def _system_prompt_from_parts(
    system_prompt_parts: tuple[OtelSystemInstructionPart, ...] | tuple[OtelMessagePart, ...],
) -> tuple[TextPart, ...]:
    """Convert supported OTel system instructions into langchaint text parts."""
    converted: list[TextPart] = []
    for part in system_prompt_parts:
        if not isinstance(part, OtelTextPart):
            raise _unsupported(part, "system instruction type")
        _require_no_additional_properties(part)
        converted.append(TextPart(text=part.content))
    return tuple(converted)


def system_prompt_from_otel(
    otel_chat_span: OtelChatSpan,
) -> tuple[TextPart, ...] | None:
    """Convert the system prompt from one parsed OTel chat span.

    Args:
        otel_chat_span: The parsed OTel chat span attributes.

    Returns:
        The converted text parts, or `None` when the span has no system prompt.

    Raises:
        OtelToLangchaintConversionError: The system prompt has no lossless langchaint representation.
    """
    system_prompt, _ = _split_system_prompt(otel_chat_span)
    return system_prompt


def _split_system_prompt(
    otel_chat_span: OtelChatSpan,
) -> tuple[tuple[TextPart, ...] | None, tuple[OtelInputMessage, ...]]:
    """Convert the system prompt and return it with the input messages that remain."""
    input_messages = otel_chat_span.input_messages or ()
    system_message_indexes = [
        index for index, message in enumerate(input_messages) if message.role == "system"
    ]
    if system_message_indexes and otel_chat_span.system_instructions:
        raise OtelToLangchaintConversionError(
            f"{SYSTEM_INSTRUCTIONS} and an {INPUT_MESSAGES} role='system' message both provide "
            "the system prompt"
        )
    if len(system_message_indexes) > 1:
        raise _attribute_conversion_error(
            INPUT_MESSAGES, "contains multiple role='system' messages"
        )
    if system_message_indexes and system_message_indexes[0] != 0:
        raise _attribute_conversion_error(
            INPUT_MESSAGES, "contains a role='system' message after the first message"
        )
    if system_message_indexes:
        system_message = input_messages[0]
        _require_message_metadata(system_message)
        if not system_message.parts:
            raise _attribute_conversion_error(
                INPUT_MESSAGES, "contains a role='system' message without parts"
            )
        return _system_prompt_from_parts(system_message.parts), input_messages[1:]
    if otel_chat_span.system_instructions:
        return _system_prompt_from_parts(otel_chat_span.system_instructions), input_messages
    return None, input_messages


def tool_schemas_from_otel(otel_chat_span: OtelChatSpan) -> tuple[ToolSchema, ...] | None:
    """Convert tool definitions from one parsed OTel chat span.

    Args:
        otel_chat_span: The parsed OTel chat span attributes.

    Returns:
        The converted tool schemas, or `None` when the span has no tool definitions.

    Raises:
        OtelToLangchaintConversionError: A tool definition has no lossless langchaint representation.
    """
    tool_definitions = otel_chat_span.tool_definitions
    if tool_definitions is None:
        return None
    converted: list[ToolSchema] = []
    converted_names: set[str] = set()
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
        if definition.name in converted_names:
            raise OtelToLangchaintConversionError(
                f"{TOOL_DEFINITIONS} contains duplicate function definition {definition!r}"
            )
        converted_names.add(definition.name)
        converted.append(
            ToolSchema(
                name=definition.name,
                description=definition.description,
                args_schema=definition.parameters,
            )
        )
    return tuple(converted)


def generation_input_from_otel(
    otel_chat_span: OtelChatSpan,
) -> GenerationInput:
    """Convert one parsed OTel chat span into one langchaint generation input.

    Args:
        otel_chat_span: The parsed OTel chat span attributes.

    Raises:
        OtelToLangchaintConversionError: An input value has no lossless langchaint representation.
    """
    _, input_messages = _split_system_prompt(otel_chat_span)
    return tuple(_message_from_otel(message) for message in input_messages)


def _assistant_message_from_output(message: OtelOutputMessage) -> AssistantMessage:
    _require_message_metadata(message)
    if message.role != "assistant":
        raise _unsupported(message, "output message role")
    return _assistant_message_from_parts(message.parts)


def response_record_from_otel(otel_chat_span: OtelChatSpan) -> ResponseRecord[JsonValue]:
    """Convert one successful parsed OTel chat span into a normalized response record.

    The record contains one synthetic attempt.
    `started_after_seconds`, attempt `elapsed_seconds`, and call `elapsed_seconds` are `0.0`.
    `seconds_to_first_item`, `error`, and `request_id` are `None`.
    `Billing.usage` is `ZERO_USAGE`.
    `Billing.service_tier` is `"unknown"`.
    Every `Billing` rate is NaN.
    Trace usage, cost, retry, attempt, timing, and provider service-tier attributes are ignored.
    Failure detection uses `error.type` and the selected finish reason.
    `OtelChatSpan` does not contain OTel span status.
    A failed span without either failure signal cannot be detected.

    Args:
        otel_chat_span: The parsed OTel chat span attributes.

    Raises:
        OtelToLangchaintConversionError: The span reports failure.
        OtelToLangchaintConversionError: A selected value cannot construct a successful `ResponseRecord` unchanged.
    """
    if otel_chat_span.error_type is not None:
        raise _attribute_conversion_error("error.type", "reports a failed span")
    output_messages = otel_chat_span.output_messages
    if output_messages is None or len(output_messages) != 1:
        raise _attribute_conversion_error(
            OUTPUT_MESSAGES,
            "must contain exactly one output message",
        )
    output_message = output_messages[0]
    assistant_message = _assistant_message_from_output(output_message)
    stop_reason = _stop_reason_from_otel(otel_chat_span, output_message.finish_reason)
    output = _output_from_otel(otel_chat_span.output_type, assistant_message)
    if not otel_chat_span.provider_name:
        raise _attribute_conversion_error("gen_ai.provider.name", "is required")
    if not otel_chat_span.request_model:
        raise _attribute_conversion_error("gen_ai.request.model", "is required")
    billing = Billing(
        usage=ZERO_USAGE,
        service_tier="unknown",
        input_cache_none_usd_per_million_tokens=float("nan"),
        cache_read_usd_per_million_tokens=float("nan"),
        cache_write_usd_per_million_tokens=float("nan"),
        output_usd_per_million_tokens=float("nan"),
    )
    attempt = SettledAttemptRecord(
        started_after_seconds=0.0,
        elapsed_seconds=0.0,
        seconds_to_first_item=None,
        error=None,
        billing=billing,
        assistant_message=assistant_message,
        model_served=otel_chat_span.response_model,
        response_id=otel_chat_span.response_id,
        request_id=None,
    )
    call = CallRecord(
        model=otel_chat_span.request_model,
        provider_name=otel_chat_span.provider_name,
        attempt_records=(attempt,),
        elapsed_seconds=0.0,
    )
    return ResponseRecord[JsonValue](call=call, output=output, stop_reason=stop_reason)


@overload
def reconstruct_bound_llm[ModelT: BaseModel](
    otel_chat_span: OtelChatSpan,
    *,
    llm: LLM,
    tools: ToolManager | ToolSequence,
    response_format: type[ModelT],
) -> BoundLLM[ModelT, ToolManager]: ...


@overload
def reconstruct_bound_llm[ModelT: BaseModel](
    otel_chat_span: OtelChatSpan,
    *,
    llm: LLM,
    tools: None = None,
    response_format: type[ModelT],
) -> BoundLLM[ModelT, None]: ...


@overload
def reconstruct_bound_llm(
    otel_chat_span: OtelChatSpan,
    *,
    llm: LLM,
    tools: ToolManager | ToolSequence,
    response_format: None = None,
) -> BoundLLM[str, ToolManager]: ...


@overload
def reconstruct_bound_llm(
    otel_chat_span: OtelChatSpan,
    *,
    llm: LLM,
    tools: None = None,
    response_format: None = None,
) -> BoundLLM[str, None]: ...


def reconstruct_bound_llm[ModelT: BaseModel](
    otel_chat_span: OtelChatSpan,
    *,
    llm: LLM,
    tools: ToolManager | ToolSequence | None = None,
    response_format: type[ModelT] | None = None,
) -> (
    BoundLLM[ModelT, ToolManager]
    | BoundLLM[ModelT, None]
    | BoundLLM[str, ToolManager]
    | BoundLLM[str, None]
):
    """Bind supported captured request fields and caller-supplied Python objects.

    OTel tool definitions contain schemas without executable Python functions.
    `tools` supplies executable Python functions.
    `response_format` supplies the model class that `output_type="json"` omits.

    Args:
        otel_chat_span: The parsed OTel chat span attributes.
        llm: The provider SDK client state and request-admission configuration.
        tools: Executable Python tools.
            When `gen_ai.tool.definitions` is present, its converted schemas must equal the tool schemas.
        response_format: The structured response model for JSON output, or `None` for text output.

    Raises:
        ValueError: The provider name or model on `llm` differs from `otel_chat_span`.
        ValueError: A parsed binding field is invalid for the adapter.
        ValueError: `tools` contains duplicate names.
        OtelToLangchaintConversionError: Captured configuration differs from caller-supplied objects.
        OtelToLangchaintConversionError: A captured value has no lossless langchaint representation.
        TypeError: The adapter does not support the reconstructed binding.
        pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool model has no JSON schema.
        pydantic.PydanticUserError: `response_format` or a tool model is not fully defined.
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
    system_prompt = system_prompt_from_otel(otel_chat_span)
    _require_matching_response_format(otel_chat_span.output_type, response_format)
    captured_tool_schemas = tool_schemas_from_otel(otel_chat_span)
    bound_llm = llm.bind(
        system_prompt=system_prompt,
        tools=tools,
        response_format=response_format,
        max_completion_tokens=otel_chat_span.request_max_tokens,
        reasoning_level=otel_chat_span.request_reasoning_level,
        temperature=otel_chat_span.request_temperature,
    )
    if (
        captured_tool_schemas is not None
        and captured_tool_schemas != bound_llm.binding.tool_schemas
    ):
        raise OtelToLangchaintConversionError(
            f"{TOOL_DEFINITIONS} {otel_chat_span.tool_definitions!r} converts to "
            f"{captured_tool_schemas!r}, which differs from caller tool schemas "
            f"{bound_llm.binding.tool_schemas!r}"
        )
    return bound_llm


def _stop_reason_from_otel(
    otel_chat_span: OtelChatSpan, message_finish_reason: str | None
) -> StopReason:
    response_finish_reasons = otel_chat_span.response_finish_reasons
    if response_finish_reasons is not None:
        if len(response_finish_reasons) != 1:
            raise _attribute_conversion_error(
                "gen_ai.response.finish_reasons",
                "must contain exactly one value",
            )
        selected_finish_reason = response_finish_reasons[0]
        if message_finish_reason is not None and message_finish_reason != selected_finish_reason:
            raise OtelToLangchaintConversionError(
                f"gen_ai.response.finish_reasons {response_finish_reasons!r} differs from "
                f"{OUTPUT_MESSAGES} finish_reason {message_finish_reason!r}"
            )
    else:
        selected_finish_reason = message_finish_reason
    if selected_finish_reason is None:
        raise OtelToLangchaintConversionError(
            f"gen_ai.response.finish_reasons {response_finish_reasons!r} and "
            f"{OUTPUT_MESSAGES} finish_reason {message_finish_reason!r} contain no value"
        )
    match selected_finish_reason:
        case "error":
            raise OtelToLangchaintConversionError(
                f"finish_reason {selected_finish_reason!r} reports a failed span"
            )
        case "stop":
            return "end_turn"
        case "tool_call":
            return "tool_use"
        case "length":
            return "max_tokens"
        case "content_filter":
            return "refusal"
        case (
            "end_turn"
            | "tool_use"
            | "max_tokens"
            | "refusal"
            | "context_window_exceeded"
            | "other"
        ):
            return selected_finish_reason
        case _:
            return "other"


def _selected_output_type(output_type: str | None) -> str:
    return "text" if output_type is None else output_type


def _output_from_otel(output_type: str | None, assistant_message: AssistantMessage) -> JsonValue:
    selected_output_type = _selected_output_type(output_type)
    if selected_output_type == "text":
        return assistant_message.text
    if selected_output_type == "json":
        try:
            return _JSON_VALUE_ADAPTER.validate_json(assistant_message.text)
        except ValidationError as error:
            raise _attribute_conversion_error(
                "gen_ai.output.type",
                "declares output that is not valid JSON",
            ) from error
    raise _attribute_conversion_error(
        "gen_ai.output.type",
        "has no ResponseRecord[JsonValue] representation",
    )


def _require_matching_response_format(
    output_type: str | None, response_format: type[BaseModel] | None
) -> None:
    selected_output_type = _selected_output_type(output_type)
    if selected_output_type == "json" and response_format is None:
        raise OtelToLangchaintConversionError(
            f"gen_ai.output.type {output_type!r} requires response_format, got {response_format!r}"
        )
    if selected_output_type == "text" and response_format is not None:
        raise OtelToLangchaintConversionError(
            f"gen_ai.output.type {output_type!r} requires response_format=None, got "
            f"{response_format!r}"
        )
    if selected_output_type not in {"json", "text"}:
        raise _attribute_conversion_error("gen_ai.output.type", "has no supported response_format")


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
    _require_no_additional_properties(part)
    if isinstance(part.response, str):
        content: str | tuple[ContentPart, ...] = part.response
    elif isinstance(part.response, list):
        try:
            response_parts = _MESSAGE_PARTS_ADAPTER.validate_python(part.response)
        except ValidationError as error:
            raise _unsupported(part, "tool response value") from error
        content = _content_parts_from_otel(response_parts)
    else:
        raise _unsupported(part, "tool response value")
    return ToolMessage(tool_call_id=part.id, content=content, is_error=part.is_error)


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
            data = _BASE64_BYTES_ADAPTER.validate_python(part.content)
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


def _attribute_conversion_error(
    attribute_name: str, description: str
) -> OtelToLangchaintConversionError:
    return OtelToLangchaintConversionError(f"{attribute_name} {description}")


def _unsupported(value: OtelModel, description: str) -> OtelToLangchaintConversionError:
    return OtelToLangchaintConversionError(
        f"{type(value).__name__} has no langchaint representation for {description}"
    )


__all__ = [
    "OtelBlobPart",
    "OtelChatSpan",
    "OtelCompactionPart",
    "OtelFilePart",
    "OtelFunctionTool",
    "OtelGenericObject",
    "OtelGenericTool",
    "OtelImageUrlPart",
    "OtelInputMessage",
    "OtelMessagePart",
    "OtelOutputMessage",
    "OtelReasoningPart",
    "OtelServerToolCallPart",
    "OtelServerToolCallResponsePart",
    "OtelSystemInstructionPart",
    "OtelTextPart",
    "OtelToLangchaintConversionError",
    "OtelToolCallPart",
    "OtelToolCallResponsePart",
    "OtelToolDefinition",
    "OtelUriPart",
    "generation_input_from_otel",
    "parse_otel",
    "reconstruct_bound_llm",
    "response_record_from_otel",
    "system_prompt_from_otel",
    "tool_schemas_from_otel",
]
