"""Extract langchaint values from OTel span attributes."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import (
    Base64UrlBytes,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
)

from langchaint.checked_copy import CheckedCopyModel
from langchaint.llm import GenerationInput
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
    UserMessage,
)
from langchaint.tools import ToolSchema

MAX_TOKENS = "gen_ai.request.max_tokens"
REASONING_LEVEL = "gen_ai.request.reasoning.level"
TEMPERATURE = "gen_ai.request.temperature"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

EXTRACTED_REQUEST_ATTRIBUTES: frozenset[str] = frozenset({
    "gen_ai.request.choice.count",
    "gen_ai.request.frequency_penalty",
    MAX_TOKENS,
    "gen_ai.request.model",
    "gen_ai.request.presence_penalty",
    "gen_ai.request.previous_response.id",
    REASONING_LEVEL,
    "gen_ai.request.seed",
    "gen_ai.request.stop_sequences",
    "gen_ai.request.stream_cursor",
    TEMPERATURE,
    "gen_ai.request.top_k",
    "gen_ai.request.top_p",
})
IGNORED_REQUEST_ATTRIBUTES: frozenset[str] = frozenset({
    "gen_ai.request.stream",
})


class OtelModel(CheckedCopyModel):
    """Pydantic rejects values outside langchaint's reconstructable OTel subset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class OtelTextPart(OtelModel):
    """Pydantic validates one reconstructable OTel text part."""

    type: Literal["text"]
    content: str


class OtelBlobPart(OtelModel):
    """Pydantic validates and decodes one reconstructable OTel blob part."""

    type: Literal["blob"]
    modality: Literal["image", "audio"]
    mime_type: str
    content: Annotated[Base64UrlBytes, Field(strict=False)]


class OtelUriPart(OtelModel):
    """Pydantic validates one reconstructable OTel image URI part."""

    type: Literal["uri"]
    modality: Literal["image"]
    uri: str
    mime_type: str | None = None


class OtelImageUrlPart(OtelModel):
    """Pydantic validates the image URL extension emitted by langchaint."""

    type: Literal["image_url"]
    url: str
    mime_type: str | None = None


class OtelToolCallPart(OtelModel):
    """Pydantic validates one reconstructable OTel tool call part."""

    type: Literal["tool_call"]
    id: str
    name: str
    arguments: JsonValue = None


type OtelUserPart = Annotated[
    OtelTextPart | OtelBlobPart | OtelUriPart | OtelImageUrlPart,
    Field(discriminator="type"),
]
type OtelAssistantPart = Annotated[
    OtelTextPart | OtelToolCallPart,
    Field(discriminator="type"),
]


class OtelUserMessage(OtelModel):
    """Pydantic validates one OTel user message that UserMessage can represent."""

    role: Literal["user"]
    parts: Annotated[tuple[OtelUserPart, ...], Field(strict=False)]
    name: None = None


class OtelAssistantMessage(OtelModel):
    """Pydantic validates one OTel assistant message that AssistantMessage can represent."""

    role: Literal["assistant"]
    parts: Annotated[tuple[OtelAssistantPart, ...], Field(strict=False)]
    name: None = None


type OtelInputMessage = Annotated[
    OtelUserMessage | OtelAssistantMessage,
    Field(discriminator="role"),
]


class OtelOutputMessage(OtelAssistantMessage):
    """Pydantic validates one OTel output message that AssistantMessage can represent."""

    finish_reason: str


class OtelFunctionTool(OtelModel):
    """Pydantic validates one OTel function definition that ToolSchema can represent."""

    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, JsonValue]


type StrictFiniteFloat = Annotated[FiniteFloat, Field(strict=True)]

MAX_TOKENS_ADAPTER: TypeAdapter[int] = TypeAdapter(StrictInt)
TEMPERATURE_ADAPTER: TypeAdapter[float] = TypeAdapter(StrictFiniteFloat)
REASONING_LEVEL_ADAPTER: TypeAdapter[str] = TypeAdapter(StrictStr)
SYSTEM_INSTRUCTIONS_ADAPTER: TypeAdapter[tuple[OtelTextPart, ...]] = TypeAdapter(
    tuple[OtelTextPart, ...]
)
TOOL_DEFINITIONS_ADAPTER: TypeAdapter[tuple[OtelFunctionTool, ...]] = TypeAdapter(
    tuple[OtelFunctionTool, ...]
)
INPUT_MESSAGES_ADAPTER: TypeAdapter[tuple[OtelInputMessage, ...]] = TypeAdapter(
    tuple[OtelInputMessage, ...]
)
OUTPUT_MESSAGES_ADAPTER: TypeAdapter[tuple[OtelOutputMessage, ...]] = TypeAdapter(
    tuple[OtelOutputMessage, ...]
)


@dataclass(frozen=True, kw_only=True)
class ExtractedOutputMessage:
    assistant_message: AssistantMessage
    finish_reason: str


@dataclass(frozen=True, kw_only=True)
class SpanParameterExtraction:
    binding_parameters: Mapping[str, object]
    request_parameters: Mapping[str, object]
    generation_input: GenerationInput | None
    output_messages: tuple[ExtractedOutputMessage, ...] | None


def extract_span_parameters(attributes: Mapping[str, object]) -> SpanParameterExtraction:
    request_parameters = extract_request_parameters(attributes)
    binding_parameters = extract_request_binding_parameters(request_parameters)

    if SYSTEM_INSTRUCTIONS in attributes:
        system_prompt = parse_system_instructions(attributes[SYSTEM_INSTRUCTIONS])
        if system_prompt:
            binding_parameters["system_prompt"] = system_prompt

    if TOOL_DEFINITIONS in attributes:
        tool_schemas = parse_tool_definitions(attributes[TOOL_DEFINITIONS])
        if tool_schemas:
            binding_parameters["tool_schemas"] = tool_schemas

    generation_input = None
    if INPUT_MESSAGES in attributes:
        generation_input = parse_generation_input(attributes[INPUT_MESSAGES])

    output_messages = None
    if OUTPUT_MESSAGES in attributes:
        output_messages = parse_output_messages(attributes[OUTPUT_MESSAGES])

    return SpanParameterExtraction(
        binding_parameters=binding_parameters,
        request_parameters=request_parameters,
        generation_input=generation_input,
        output_messages=output_messages,
    )


def extract_request_parameters(attributes: Mapping[str, object]) -> dict[str, object]:
    request_parameters: dict[str, object] = {}
    for attribute_name in attributes:
        if not attribute_name.startswith("gen_ai.request."):
            continue
        if attribute_name in EXTRACTED_REQUEST_ATTRIBUTES:
            request_parameters[attribute_name] = attributes[attribute_name]
            continue
        if attribute_name in IGNORED_REQUEST_ATTRIBUTES:
            continue
        raise ValueError(f"{attribute_name} is unsupported by langchaint's OTel request parser")
    return request_parameters


def extract_request_binding_parameters(
    request_parameters: Mapping[str, object],
) -> dict[str, object]:
    binding_parameters: dict[str, object] = {}
    if MAX_TOKENS in request_parameters:
        binding_parameters["max_completion_tokens"] = validate_scalar_attribute(
            MAX_TOKENS_ADAPTER,
            MAX_TOKENS,
            request_parameters[MAX_TOKENS],
        )
    if REASONING_LEVEL in request_parameters:
        binding_parameters["reasoning_level"] = validate_scalar_attribute(
            REASONING_LEVEL_ADAPTER,
            REASONING_LEVEL,
            request_parameters[REASONING_LEVEL],
        )
    if TEMPERATURE in request_parameters:
        binding_parameters["temperature"] = validate_scalar_attribute(
            TEMPERATURE_ADAPTER,
            TEMPERATURE,
            request_parameters[TEMPERATURE],
        )
    return binding_parameters


def parse_system_instructions(attribute_value: object) -> tuple[TextPart, ...]:
    parts = validate_structured_attribute(
        SYSTEM_INSTRUCTIONS_ADAPTER,
        SYSTEM_INSTRUCTIONS,
        attribute_value,
    )
    return tuple(TextPart(text=part.content) for part in parts)


def parse_tool_definitions(attribute_value: object) -> tuple[ToolSchema, ...]:
    definitions = validate_structured_attribute(
        TOOL_DEFINITIONS_ADAPTER,
        TOOL_DEFINITIONS,
        attribute_value,
    )
    return tuple(
        ToolSchema(
            name=definition.name,
            description=definition.description,
            args_schema=definition.parameters,
        )
        for definition in definitions
    )


def parse_generation_input(attribute_value: object) -> GenerationInput:
    messages = validate_structured_attribute(
        INPUT_MESSAGES_ADAPTER,
        INPUT_MESSAGES,
        attribute_value,
    )
    return tuple(message_from_otel(message) for message in messages)


def parse_output_messages(attribute_value: object) -> tuple[ExtractedOutputMessage, ...]:
    messages = validate_structured_attribute(
        OUTPUT_MESSAGES_ADAPTER,
        OUTPUT_MESSAGES,
        attribute_value,
    )
    return tuple(
        ExtractedOutputMessage(
            assistant_message=assistant_message_from_otel(message),
            finish_reason=message.finish_reason,
        )
        for message in messages
    )


def message_from_otel(message: OtelInputMessage) -> Message:
    match message.role:
        case "user":
            return UserMessage(
                content=tuple(content_part_from_otel(part) for part in message.parts)
            )
        case "assistant":
            return assistant_message_from_otel(message)


def assistant_message_from_otel(message: OtelAssistantMessage) -> AssistantMessage:
    return AssistantMessage(turn=tuple(assistant_part_from_otel(part) for part in message.parts))


def content_part_from_otel(part: OtelUserPart) -> ContentPart:
    match part.type:
        case "text":
            return TextPart(text=part.content)
        case "blob":
            if part.modality == "image":
                return ImagePart(data=part.content, media_type=part.mime_type)
            return AudioPart(data=part.content, media_type=part.mime_type)
        case "uri":
            return ImageUrlPart(url=part.uri, media_type=part.mime_type)
        case "image_url":
            return ImageUrlPart(url=part.url, media_type=part.mime_type)


def assistant_part_from_otel(part: OtelAssistantPart) -> TextPart | ToolCall:
    match part.type:
        case "text":
            return TextPart(text=part.content)
        case "tool_call":
            return ToolCall(
                id=part.id,
                name=part.name,
                args_json=json.dumps(part.arguments, allow_nan=False, separators=(",", ":")),
            )


def validate_scalar_attribute[ValueT](
    adapter: TypeAdapter[ValueT],
    attribute_name: str,
    attribute_value: object,
) -> ValueT:
    try:
        return adapter.validate_python(attribute_value)
    except ValidationError as error:
        raise ValueError(
            f"{attribute_name} is outside langchaint's reconstructable OTel subset: {error}"
        ) from error


def validate_structured_attribute[ValueT](
    adapter: TypeAdapter[ValueT],
    attribute_name: str,
    attribute_value: object,
) -> ValueT:
    try:
        if isinstance(attribute_value, str):
            return adapter.validate_json(attribute_value)
        return adapter.validate_python(attribute_value)
    except ValidationError as error:
        raise ValueError(
            f"{attribute_name} is outside langchaint's reconstructable OTel subset: {error}"
        ) from error
