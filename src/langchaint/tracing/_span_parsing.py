"""Extract langchaint values from OTel span attributes."""

import json
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import (
    Base64UrlBytes,
    ConfigDict,
    Field,
    FiniteFloat,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

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
)
from langchaint.tools import ToolSchema

PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
OUTPUT_TYPE = "gen_ai.output.type"
MAX_TOKENS = "gen_ai.request.max_tokens"
REASONING_LEVEL = "gen_ai.request.reasoning.level"
TEMPERATURE = "gen_ai.request.temperature"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

REQUEST_ATTRIBUTE_PREFIX = "gen_ai.request."
PARSED_ATTRIBUTES: frozenset[str] = frozenset({
    PROVIDER_NAME,
    REQUEST_MODEL,
    OUTPUT_TYPE,
    MAX_TOKENS,
    REASONING_LEVEL,
    TEMPERATURE,
    SYSTEM_INSTRUCTIONS,
    TOOL_DEFINITIONS,
    INPUT_MESSAGES,
    OUTPUT_MESSAGES,
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


type OtelToolResponse = str | Annotated[tuple[OtelUserPart, ...], Field(strict=False)]


class OtelToolCallResponsePart(OtelModel):
    """Pydantic validates one OTel tool call response that ToolMessage can represent."""

    type: Literal["tool_call_response"]
    id: str
    is_error: bool = False
    response: OtelToolResponse


class OtelToolMessage(OtelModel):
    """Pydantic validates one OTel tool message that ToolMessage can represent."""

    role: Literal["tool"]
    parts: Annotated[tuple[OtelToolCallResponsePart], Field(strict=False)]
    name: None = None


type OtelInputMessage = Annotated[
    OtelUserMessage | OtelAssistantMessage | OtelToolMessage,
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

RAW_SPAN_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
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


class ParsedChatSpan(OtelModel):
    """Pydantic validates one raw OTel chat span attribute object.

    Other `gen_ai.request.*` attributes remain in `unapplied_request_parameters`.
    Attributes outside the parsed fields and `unapplied_request_parameters` are discarded.
    """

    provider_name: str = Field(alias=PROVIDER_NAME)
    model: str = Field(alias=REQUEST_MODEL)
    output_type: Literal["text", "json"] | None = Field(default=None, alias=OUTPUT_TYPE)
    system_prompt: tuple[TextPart, ...] | None = Field(default=None, alias=SYSTEM_INSTRUCTIONS)
    tool_schemas: tuple[ToolSchema, ...] = Field(default=(), alias=TOOL_DEFINITIONS)
    max_completion_tokens: int | None = Field(default=None, alias=MAX_TOKENS)
    reasoning_level: str | None = Field(default=None, alias=REASONING_LEVEL)
    temperature: StrictFiniteFloat | None = Field(default=None, alias=TEMPERATURE)
    unapplied_request_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    generation_input: GenerationInput | None = Field(default=None, alias=INPUT_MESSAGES)
    output_messages: tuple[ExtractedOutputMessage, ...] | None = Field(
        default=None, alias=OUTPUT_MESSAGES
    )

    @model_validator(mode="before")
    @classmethod
    def _partition_raw_span(cls, input_value: object) -> object:
        raw_span = RAW_SPAN_ADAPTER.validate_python(input_value)
        parsed_attributes: dict[str, JsonValue] = {
            attribute_name: attribute_value
            for attribute_name, attribute_value in raw_span.items()
            if attribute_name in PARSED_ATTRIBUTES
        }
        parsed_attributes["unapplied_request_parameters"] = {
            attribute_name: attribute_value
            for attribute_name, attribute_value in raw_span.items()
            if attribute_name.startswith(REQUEST_ATTRIBUTE_PREFIX)
            and attribute_name not in PARSED_ATTRIBUTES
        }
        return parsed_attributes

    @field_validator("system_prompt", mode="before")
    @classmethod
    def _parse_system_prompt(cls, attribute_value: JsonValue) -> tuple[TextPart, ...]:
        return parse_system_instructions(attribute_value)

    @field_validator("tool_schemas", mode="before")
    @classmethod
    def _parse_tool_schemas(cls, attribute_value: JsonValue) -> tuple[ToolSchema, ...]:
        return parse_tool_definitions(attribute_value)

    @field_validator("generation_input", mode="before")
    @classmethod
    def _parse_generation_input(cls, attribute_value: JsonValue) -> GenerationInput:
        return parse_generation_input(attribute_value)

    @field_validator("output_messages", mode="before")
    @classmethod
    def _parse_output_messages(
        cls, attribute_value: JsonValue
    ) -> tuple[ExtractedOutputMessage, ...]:
        return parse_output_messages(attribute_value)


def reconstruct_bound_llm(
    parsed_span: ParsedChatSpan,
    *,
    llm: LLM,
) -> BoundLLM[str, None]:
    """Bind the provider-neutral request fields recorded by `parsed_span`.

    The returned `BoundLLM` does not apply `tool_schemas`, `output_type`, or `unapplied_request_parameters`.

    Args:
        parsed_span: The validated OTel chat span attribute object.
        llm: The provider SDK client state and request-admission configuration.

    Raises:
        ValueError: The provider name or model on `llm` differs from `parsed_span`.
        ValueError: A parsed binding field is invalid for the adapter.
    """
    if llm.adapter.provider_name != parsed_span.provider_name:
        raise ValueError(
            f"LLM provider_name {llm.adapter.provider_name!r} differs from parsed span "
            f"provider_name {parsed_span.provider_name!r}"
        )
    if llm.adapter.model != parsed_span.model:
        raise ValueError(
            f"LLM model {llm.adapter.model!r} differs from parsed span model {parsed_span.model!r}"
        )
    system_prompt = parsed_span.system_prompt or None
    return llm.bind(
        system_prompt=system_prompt,
        max_completion_tokens=parsed_span.max_completion_tokens,
        reasoning_level=parsed_span.reasoning_level,
        temperature=parsed_span.temperature,
    )


def parse_system_instructions(attribute_value: JsonValue) -> tuple[TextPart, ...]:
    parts = validate_structured_attribute(
        SYSTEM_INSTRUCTIONS_ADAPTER,
        SYSTEM_INSTRUCTIONS,
        attribute_value,
    )
    return tuple(TextPart(text=part.content) for part in parts)


def parse_tool_definitions(attribute_value: JsonValue) -> tuple[ToolSchema, ...]:
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


def parse_generation_input(attribute_value: JsonValue) -> GenerationInput:
    messages = validate_structured_attribute(
        INPUT_MESSAGES_ADAPTER,
        INPUT_MESSAGES,
        attribute_value,
    )
    return tuple(message_from_otel(message) for message in messages)


def parse_output_messages(attribute_value: JsonValue) -> tuple[ExtractedOutputMessage, ...]:
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
        case "tool":
            return tool_message_from_otel(message)


def tool_message_from_otel(message: OtelToolMessage) -> ToolMessage:
    part = message.parts[0]
    content = (
        part.response
        if isinstance(part.response, str)
        else tuple(content_part_from_otel(response_part) for response_part in part.response)
    )
    return ToolMessage(tool_call_id=part.id, content=content, is_error=part.is_error)


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


def validate_structured_attribute[ValueT](
    adapter: TypeAdapter[ValueT],
    attribute_name: str,
    attribute_value: JsonValue,
) -> ValueT:
    try:
        if isinstance(attribute_value, str):
            return adapter.validate_json(attribute_value)
        return adapter.validate_python(attribute_value)
    except ValidationError as error:
        raise ValueError(
            f"{attribute_name} is outside langchaint's reconstructable OTel subset: {error}"
        ) from error
