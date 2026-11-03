"""Provider-neutral messages and their parts.

A part is one ordered value in a message.
`UserMessage.content` and `ToolMessage.content` hold plain text or `ContentPart` values.
`AssistantMessage.content` holds plain text or `AssistantPart` values.
`TextPart` is the only part of both kinds.

Adapters convert each complete `Sequence[Message]` to provider values.
The system prompt remains a binding parameter because providers place it outside messages.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Literal, NamedTuple, TypeIs

from pydantic import BeforeValidator, ConfigDict, Field, FiniteFloat, TypeAdapter

from langchaint.common.checked_copy import CheckedCopyModel


def _is_object_list(value: object) -> TypeIs[list[object]]:
    return type(value) is list


def _is_object_dict(value: object) -> TypeIs[dict[object, object]]:
    return type(value) is dict


def _require_json_runtime_shape(value: object) -> object:
    active_container_ids: set[int] = set()

    def require_value_shape(nested_value: object) -> None:
        if nested_value is None or type(nested_value) in (bool, int, str):
            return
        if type(nested_value) is float:
            if not math.isfinite(nested_value):
                raise ValueError("a JSON number must be finite")
            return
        if _is_object_list(nested_value):
            container_id = id(nested_value)
            if container_id in active_container_ids:
                raise ValueError("a JSON value must not contain a cycle")
            active_container_ids.add(container_id)
            try:
                for item in nested_value:
                    require_value_shape(item)
            finally:
                active_container_ids.remove(container_id)
            return
        if _is_object_dict(nested_value):
            container_id = id(nested_value)
            if container_id in active_container_ids:
                raise ValueError("a JSON value must not contain a cycle")
            active_container_ids.add(container_id)
            try:
                for key, item in nested_value.items():
                    if type(key) is not str:
                        raise ValueError("a JSON object key must be a string")
                    require_value_shape(item)
            finally:
                active_container_ids.remove(container_id)
            return
        raise ValueError("value must use JSON runtime types")

    require_value_shape(value)
    return value


type _JsonValue = None | bool | int | FiniteFloat | str | list[_JsonValue] | dict[str, _JsonValue]
type JsonValue = Annotated[_JsonValue, BeforeValidator(_require_json_runtime_shape)]
"""A JSON value whose floating-point values are finite."""


class TextPart(CheckedCopyModel):
    """Text, followed by a cache breakpoint when `cache_breakpoint=True`.

    OpenAI adapters send assistant text as strings, so they ignore cache breakpoints on assistant text.
    The Gemini adapter ignores every cache breakpoint.
    The Anthropic adapter returns `RejectedMessages` for a cache breakpoint on a `ToolMessage` part except the last.
    It places a tool part's cache breakpoint on the enclosing `tool_result` block.
    `automatic_cache_breakpoints` separately controls automatic cache breakpoints.
    """

    text: str
    cache_breakpoint: bool = False
    kind: Literal["text"] = "text"


def content_parts[PartT](content: str | tuple[PartT, ...]) -> tuple[PartT | TextPart, ...]:
    """Return `content` as parts, with plain text as one `TextPart`."""
    return (TextPart(text=content),) if isinstance(content, str) else content


class ImagePart(CheckedCopyModel):
    """Inline image bytes encoded as URL-safe base64 in JSON.

    `media_type` is an IANA media type such as `"image/png"`.
    `cache_breakpoint` has the same meaning as `TextPart.cache_breakpoint`.
    """

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    data: bytes
    media_type: str
    cache_breakpoint: bool = False
    kind: Literal["image"] = "image"


class ImageUrlPart(CheckedCopyModel):
    """An image URL sent unchanged without fetching it.

    `media_type` is an optional IANA media type.
    Gemini receives `media_type` when present.
    Other adapters receive `url` without `media_type`.
    `cache_breakpoint` has the same meaning as `TextPart.cache_breakpoint`.
    """

    url: str
    media_type: str | None = None
    cache_breakpoint: bool = False
    kind: Literal["image_url"] = "image_url"


class AudioPart(CheckedCopyModel):
    """Inline audio bytes encoded as URL-safe base64 in JSON.

    `media_type` is an IANA media type, such as `"audio/wav"`.
    `cache_breakpoint` has the same meaning as `TextPart.cache_breakpoint`.
    """

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    data: bytes
    media_type: str
    cache_breakpoint: bool = False
    kind: Literal["audio"] = "audio"


type ContentPart = Annotated[
    TextPart | ImagePart | ImageUrlPart | AudioPart, Field(discriminator="kind")
]

type MessageContent = str | tuple[ContentPart, ...]
"""The model-facing content of a user or tool message: plain text or ordered parts."""


class ToolCall(CheckedCopyModel):
    """A model-requested tool call with unvalidated argument JSON.

    Adapters serialize decoded provider arguments to this shared form.
    """

    id: str
    name: str
    args_json: str
    kind: Literal["tool_call"] = "tool_call"


class UserMessage(CheckedCopyModel):
    """Pydantic selects each `ContentPart` variant by `kind` in one user message.

    Raises:
        pydantic.ValidationError: `content` is invalid.
    """

    content: MessageContent
    kind: Literal["user"] = "user"

    @property
    def content_parts(self) -> tuple[ContentPart, ...]:
        """Return `content` as parts, with plain text as one `TextPart`."""
        return content_parts(self.content)


class ReasoningPart(CheckedCopyModel):
    """Provider reasoning preserved for replay.

    The producing adapter stores an SDK dump in `raw`.
    The same adapter replays `raw` unchanged.
    Rebuild assistant messages before switching providers.
    `text` contains readable reasoning for display and does not affect replay.
    `text=None` means the provider returned no readable reasoning.
    """

    raw: dict[str, JsonValue]
    text: str | None = None
    kind: Literal["reasoning"] = "reasoning"


class RawPart(CheckedCopyModel):
    """A provider fragment preserved for replay in its original position.

    `raw` is the producing SDK `model_dump` fragment required for replay.
    The consuming adapter sends `raw` unchanged.
    Another adapter returns `RejectedMessages` or leaves validation to its provider.
    Applications inspect `Generation.raw` for the complete SDK response.
    """

    raw: dict[str, JsonValue]
    kind: Literal["raw"] = "raw"


type AssistantPart = Annotated[
    ReasoningPart | TextPart | ToolCall | RawPart, Field(discriminator="kind")
]
"""One ordered value inside `AssistantMessage.content`.

An image a provider returns arrives as `RawPart`.
"""


class AssistantMessage(CheckedCopyModel):
    """Pydantic selects each `AssistantPart` variant by `kind` in one assistant message.

    `content` accepts plain text or an ordered tuple of parts.

    Raises:
        pydantic.ValidationError: `content` is invalid.
    """

    content: str | tuple[AssistantPart, ...]
    kind: Literal["assistant"] = "assistant"

    @property
    def content_parts(self) -> tuple[AssistantPart, ...]:
        """Return `content` as parts, with plain text as one `TextPart`."""
        return content_parts(self.content)

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the `ToolCall` values from `content` in emission order."""
        return tuple(part for part in self.content_parts if part.kind == "tool_call")


class AssistantText(NamedTuple):
    """Final text and newline-joined commentary selected from complete text parts."""

    final_text: str | None
    commentary: str | None


def assistant_text(assistant_message: AssistantMessage) -> AssistantText:
    """Select final text and commentary from the ordered assistant content.

    Args:
        assistant_message: The completed assistant message.
    """
    texts = tuple(part.text for part in assistant_message.content_parts if part.kind == "text")
    if assistant_message.tool_calls:
        return AssistantText(None, "\n".join(texts) if texts else None)
    return AssistantText(
        texts[-1] if texts else None,
        "\n".join(texts[:-1]) if len(texts) > 1 else None,
    )


class ToolMessage(CheckedCopyModel):
    """Pydantic selects each `ContentPart` variant by `kind` in one tool result.

    `tool_call_id` must match the `ToolCall.id` it answers.
    `is_error=True` tells the model the tool failed.

    Raises:
        pydantic.ValidationError: `content` is invalid.
    """

    tool_call_id: str
    content: MessageContent
    is_error: bool = False
    kind: Literal["tool"] = "tool"

    @property
    def content_parts(self) -> tuple[ContentPart, ...]:
        """Return `content` as parts, with plain text as one `TextPart`."""
        return content_parts(self.content)

    @classmethod
    def error(cls, tool_call: ToolCall, content: MessageContent) -> "ToolMessage":
        """Build a `ToolMessage` with `is_error=True` that answers `tool_call`.

        Args:
            tool_call: The tool call that the message answers.
            content: The model-facing error content.
        """
        return cls(tool_call_id=tool_call.id, content=content, is_error=True)


type Message = Annotated[UserMessage | AssistantMessage | ToolMessage, Field(discriminator="kind")]
"""Pydantic validation selects the message variant from `kind`.

`messages_to_json` and `messages_from_json` restore each message to its exact type.
"""

_MESSAGES_JSON: TypeAdapter[list[Message]] = TypeAdapter(list[Message])


def messages_to_json(messages: Sequence[Message], *, indent: int | None = None) -> str:
    """Serialize messages for `messages_from_json`.

    `ReasoningPart.raw` and `RawPart.raw` remain embedded for replay.

    Args:
        messages: The messages to serialize.
        indent: The JSON indentation width, or `None` for compact JSON.

    Raises:
        pydantic_core.PydanticSerializationError: A raw value cannot be serialized as JSON.
    """
    return _MESSAGES_JSON.dump_json(list(messages), indent=indent).decode()


def messages_from_json(messages_json: str) -> list[Message]:
    """Restore the message list that `messages_to_json` serialized.

    Args:
        messages_json: The serialized message list.

    Raises:
        pydantic.ValidationError: `messages_json` does not hold a serialized message list.
    """
    return _MESSAGES_JSON.validate_json(messages_json)


type StopReason = Literal[
    "stop", "tool_call", "max_completion_tokens", "refusal", "context_window_exceeded", "other"
]
"""Provider stop reasons normalized to one vocabulary.

Adapters map unrecognized provider values to `"other"`.
A caller can shorten the `GenerationInput` or choose a model with a larger context window.
"""
