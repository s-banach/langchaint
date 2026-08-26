"""Provider-neutral messages and content parts.

Adapters convert each complete `Sequence[Message]` to provider values.
The system prompt remains a binding parameter because providers place it outside messages.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Literal, TypeIs

from pydantic import BeforeValidator, ConfigDict, Field, FiniteFloat, TypeAdapter, model_validator

from langchaint.checked_copy import CheckedCopyModel


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
    """Pydantic rejects unknown keys in one text part.

    `cache_breakpoint=True` ends a reusable prompt prefix after this part.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    cache_breakpoint: bool = False
    kind: Literal["text"] = "text"


class ImagePart(CheckedCopyModel):
    """Pydantic rejects unknown keys in one inline image part.

    `media_type` is an IANA media type such as `"image/png"`.
    `cache_breakpoint` has the same meaning as `TextPart.cache_breakpoint`.

    In JSON, `data` is URL-safe base64 text.
    Pydantic's default UTF-8 byte encoding fails on ordinary image bytes.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    data: bytes
    media_type: str
    cache_breakpoint: bool = False
    kind: Literal["image"] = "image"


class ImageUrlPart(CheckedCopyModel):
    """Pydantic rejects unknown keys in one image URL part.

    langchaint sends `url` unchanged.
    langchaint never fetches `url`.
    `media_type` is an optional IANA media type.
    Gemini receives `media_type` when present.
    Other adapters receive `url` without `media_type`.
    `cache_breakpoint` has the same meaning as `TextPart.cache_breakpoint`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    media_type: str | None = None
    cache_breakpoint: bool = False
    kind: Literal["image_url"] = "image_url"


class AudioPart(CheckedCopyModel):
    """Pydantic rejects unknown keys in one inline audio part.

    `media_type` is an IANA media type, such as `"audio/wav"`.
    JSON stores `data` as URL-safe base64 text.
    `cache_breakpoint` has the same meaning as `TextPart.cache_breakpoint`.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    data: bytes
    media_type: str
    cache_breakpoint: bool = False
    kind: Literal["audio"] = "audio"


type ContentPart = Annotated[
    TextPart | ImagePart | ImageUrlPart | AudioPart, Field(discriminator="kind")
]

type MessageContent = str | Sequence[ContentPart]


class ToolCall(CheckedCopyModel):
    """Pydantic rejects unknown keys in one model-requested tool call.

    `args_json` holds argument JSON text before validation.
    Adapters serialize decoded provider arguments to this shared form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    args_json: str
    kind: Literal["tool_call"] = "tool_call"


class UserMessage(CheckedCopyModel):
    """Pydantic selects each `ContentPart` variant by `kind` in one user turn.

    Raises:
        pydantic.ValidationError: `content` is invalid or an unknown key is passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str | tuple[ContentPart, ...]
    kind: Literal["user"] = "user"


class ReasoningPart(CheckedCopyModel):
    """Pydantic rejects unknown keys in one provider reasoning value preserved for replay.

    The producing adapter stores an SDK dump in `raw`.
    The same adapter replays `raw` unchanged.
    Rebuild turns before switching providers.
    `text` contains readable reasoning for display and does not affect replay.
    `text=None` means the provider returned no readable reasoning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: dict[str, JsonValue]
    text: str | None = None
    kind: Literal["reasoning_part"] = "reasoning_part"


class RawPart(CheckedCopyModel):
    """Pydantic rejects unknown keys in one replayable provider fragment.

    `AssistantMessage.turn` preserves `RawPart` response order.
    `raw` is the producing SDK `model_dump` fragment required for replay.
    The consuming adapter sends `raw` unchanged in its original wire position.
    Another adapter returns `InvalidRequest` or leaves validation to its provider.
    Applications inspect `Response.raw` for the complete SDK response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: dict[str, JsonValue]
    kind: Literal["raw_part"] = "raw_part"


type TurnPart = Annotated[
    ReasoningPart | TextPart | ToolCall | RawPart, Field(discriminator="kind")
]
"""One ordered value inside `AssistantMessage.turn`.

An image a provider returns arrives as `RawPart`.
"""


def _text_only_turn(turn: object) -> object:
    if isinstance(turn, str):
        return (TextPart(text=turn),)
    return turn


class AssistantMessage(CheckedCopyModel):
    """Pydantic selects each `TurnPart` variant by `kind` in one assistant turn.

    A bare string becomes one `TextPart`.

    Raises:
        pydantic.ValidationError: `turn` has an invalid value or a `TextPart` sets `cache_breakpoint`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: Annotated[tuple[TurnPart, ...], BeforeValidator(_text_only_turn)]
    kind: Literal["assistant"] = "assistant"

    @model_validator(mode="after")
    def _reject_cache_breakpoint(self) -> "AssistantMessage":
        """Reject a turn whose `TextPart` sets `cache_breakpoint`.

        Raises:
            ValueError: A `TextPart` in `turn` sets `cache_breakpoint`.
        """
        if any(isinstance(part, TextPart) and part.cache_breakpoint for part in self.turn):
            raise ValueError(
                "cache_breakpoint is not supported on assistant turn text. "
                "openai has no breakpoint on assistant replay text. "
                "Mark the following user or tool message instead"
            )
        return self

    @property
    def text(self) -> str:
        """Return the concatenated `TextPart.text` values from `turn`.

        Return an empty string when `turn` contains no `TextPart`.
        """
        return "".join(part.text for part in self.turn if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the `ToolCall` values from `turn` in emission order."""
        return tuple(part for part in self.turn if isinstance(part, ToolCall))


class ToolMessage(CheckedCopyModel):
    """Pydantic selects each `ContentPart` variant by `kind` in one tool result.

    `tool_call_id` must match the `ToolCall.id` it answers.
    `is_error=True` tells the model the tool failed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_call_id: str
    content: str | tuple[ContentPart, ...]
    is_error: bool = False
    kind: Literal["tool"] = "tool"

    @classmethod
    def error(cls, tool_call: ToolCall, content: str | tuple[ContentPart, ...]) -> "ToolMessage":
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
    "end_turn", "tool_use", "max_tokens", "refusal", "context_window_exceeded", "other"
]
"""Provider stop reasons normalized to one vocabulary.

Adapters map unrecognized provider values to `"other"`.
A caller can shorten the `GenerationInput` or choose a model with a larger context window.
"""
