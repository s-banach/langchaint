"""Provider-neutral messages and content parts.

Adapters convert each complete `Sequence[Message]` to provider values.
The system prompt remains a binding parameter because providers place it outside messages.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter, model_validator

from langchaint.checked_copy import CheckedCopyModel


class TextPart(CheckedCopyModel):
    """One text span.

    `cache_breakpoint=True` ends a reusable prompt prefix after this part.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    cache_breakpoint: bool = False
    kind: Literal["text"] = "text"


class ImagePart(CheckedCopyModel):
    """media_type is an IANA media type such as "image/png".

    cache_breakpoint has the same meaning as on TextPart: the reusable prompt prefix ends at this part.

    In JSON, data is URL-safe base64 text.
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
    """langchaint sends url unchanged and never fetches it.

    media_type is an optional IANA media type.
    Gemini receives media_type when present.
    Other adapters receive url without media_type.
    cache_breakpoint has the same meaning as TextPart.cache_breakpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    media_type: str | None = None
    cache_breakpoint: bool = False
    kind: Literal["image_url"] = "image_url"


class AudioPart(CheckedCopyModel):
    """media_type is an IANA media type, such as "audio/wav".

    JSON stores data as URL-safe base64 text.
    cache_breakpoint has the same meaning as TextPart.cache_breakpoint.
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
"""One model-facing text, image, image URL, or audio value."""

type MessageContent = str | Sequence[ContentPart]
"""A model-facing string or sequence of `ContentPart` values."""


class ToolCall(CheckedCopyModel):
    """One tool call requested by the model.

    `args_json` holds argument JSON text before validation.
    Adapters serialize decoded provider arguments to this shared form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    args_json: str
    kind: Literal["tool_call"] = "tool_call"


class UserMessage(CheckedCopyModel):
    """One user turn containing text or ordered `ContentPart` values.

    Raises:
        pydantic.ValidationError: `content` is invalid or an unknown key is passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str | tuple[ContentPart, ...]
    kind: Literal["user"] = "user"


class ReasoningPart(CheckedCopyModel):
    """A provider reasoning value preserved for replay.

    The producing adapter stores an SDK dump in `raw`; the same adapter replays it unchanged.
    Rebuild turns before switching providers.
    `text` contains readable reasoning for display and does not affect replay.
    `text=None` means the provider returned no readable reasoning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: Mapping[str, object]
    text: str | None = None
    kind: Literal["reasoning_part"] = "reasoning_part"


class RawPart(CheckedCopyModel):
    """One replayable provider fragment without another TurnPart variant.

    An adapter emits RawPart for replayable response values lacking another TurnPart variant.
    The turn preserves their response order.
    raw is the producing SDK model_dump fragment required for replay.
    The consuming adapter sends raw unchanged in its original wire position.
    Another adapter returns InvalidRequest or leaves validation to its provider.
    Applications inspect Response.raw for the complete SDK response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: Mapping[str, object]
    kind: Literal["raw_part"] = "raw_part"


type TurnPart = Annotated[
    ReasoningPart | TextPart | ToolCall | RawPart, Field(discriminator="kind")
]
"""One ordered value inside AssistantMessage.turn.

An image a provider returns arrives as RawPart.
"""


def _text_only_turn(turn: object) -> object:
    """Coerce a bare string to a one-TextPart turn, so AssistantMessage(turn="hey") works.

    Runs before validation on every construction path (the constructor and model_validate alike),
    so the stored turn is always the tuple form and readers never branch on a string.
    """
    if isinstance(turn, str):
        return (TextPart(text=turn),)
    return turn


class AssistantMessage(CheckedCopyModel):
    """One assistant turn stored in provider emission order.

    `text` and `tool_calls` are filtered views of `turn`.
    A bare string becomes one `TextPart`.

    Raises:
        pydantic.ValidationError: `turn` has an invalid value or a `TextPart` sets `cache_breakpoint`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: Annotated[tuple[TurnPart, ...], BeforeValidator(_text_only_turn)]
    kind: Literal["assistant"] = "assistant"

    @model_validator(mode="after")
    def _reject_cache_breakpoint(self) -> "AssistantMessage":
        """Reject a turn whose TextPart sets cache_breakpoint; the class docstring states why.

        Raises:
            ValueError: a TextPart in the turn sets cache_breakpoint; pydantic surfaces it as a ValidationError.
        """
        if any(isinstance(part, TextPart) and part.cache_breakpoint for part in self.turn):
            raise ValueError(
                "cache_breakpoint is not supported on assistant turn text: "
                "openai has no breakpoint on assistant replay text; "
                "mark the following user or tool message instead"
            )
        return self

    @property
    def text(self) -> str:
        """The concatenated TextPart texts of the turn; empty when the turn held no text."""
        return "".join(part.text for part in self.turn if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The ToolCall parts of the turn, in emission order."""
        return tuple(part for part in self.turn if isinstance(part, ToolCall))


class ToolMessage(CheckedCopyModel):
    """One tool result sent back to the model.

    tool_call_id must match the id of the ToolCall it answers.
    is_error True tells the model the tool failed; content then holds the error text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_call_id: str
    content: str | tuple[ContentPart, ...]
    is_error: bool = False
    kind: Literal["tool"] = "tool"

    @classmethod
    def error(cls, tool_call: ToolCall, content: str | tuple[ContentPart, ...]) -> "ToolMessage":
        """Build an is_error ToolMessage answering tool_call.

        tool_call_id is bound from tool_call.id, so the message answers that call by construction.
        """
        return cls(tool_call_id=tool_call.id, content=content, is_error=True)


type Message = Annotated[UserMessage | AssistantMessage | ToolMessage, Field(discriminator="kind")]
"""Discriminated on kind: pydantic validation selects the variant from the tag,
never from which variant's fields happen to match,
so messages_to_json and messages_from_json restore each message to its exact type.
"""

_MESSAGES_JSON: TypeAdapter[list[Message]] = TypeAdapter(list[Message])
"""Module-level so TypeAdapter construction compiles the schema once, not on every call."""


def messages_to_json(messages: Sequence[Message], *, indent: int | None = None) -> str:
    """Serialize messages for `messages_from_json`.

    `indent` formats the JSON for human readers.
    `ReasoningPart.raw` and `RawPart.raw` remain embedded for replay.

    Raises:
        pydantic_core.PydanticSerializationError: A raw value cannot be serialized as JSON.
    """
    return _MESSAGES_JSON.dump_json(list(messages), indent=indent).decode()


def messages_from_json(messages_json: str) -> list[Message]:
    """Restore the message list messages_to_json serialized.

    Raises:
        pydantic.ValidationError: messages_json does not hold a serialized message list.
    """
    return _MESSAGES_JSON.validate_json(messages_json)


type StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "refusal", "context_window_exceeded", "other"
]
"""Provider stop reasons normalized to one vocabulary;
adapters map unrecognized provider values to "other" so a new provider value cannot break callers.
context_window_exceeded carries no provider prefix because this vocabulary is langchaint's own;
it earns a value rather than "other" because it names a terminal condition a caller acts on,
by shortening the GenerationInput or moving to a model with a larger window.
"""
