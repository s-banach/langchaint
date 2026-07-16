"""Provider-neutral messages and content parts.

Messages carry no provider knowledge;
provider adapters convert whole conversations to wire shapes because conversion depends on the full sequence,
not on one message at a time.
The system prompt is a generate-method parameter, not a message type,
because providers place it in different request locations.
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    """One text span of a message's content."""

    model_config = ConfigDict(frozen=True)

    text: str


class ImagePart(BaseModel):
    """media_type is an IANA media type such as "image/png"."""

    model_config = ConfigDict(frozen=True)

    data: bytes
    media_type: str


type Part = TextPart | ImagePart

type MessageContent = str | Sequence[Part]
"""A model-facing message body (text and images the model reads).
This is the constructor-facing form a caller or tool provides;
the pydantic message models store it as str | tuple[Part, ...], coercing the sequence to a frozen tuple,
so their fields spell that tuple form out rather than aliasing it.
It is not the possibly-structured generation Response.output,
which can be a parsed BaseModel that is not a Part and never round-trips back into a message body.
"""


class ToolCall(BaseModel):
    """One tool call requested by the model.

    args_json is the raw argument JSON text before validation;
    adapters whose provider delivers decoded arguments serialize them back to JSON
    so every provider yields the same shape.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    args_json: str


class UserMessage(BaseModel):
    """One user turn; content is plain text or a tuple of parts.

    role discriminates the Message union,
    so a persisted conversation re-validates to the same message types by construction instead of by union member order.
    """

    model_config = ConfigDict(frozen=True)

    content: str | tuple[Part, ...]
    role: Literal["user"] = "user"


class AssistantMessage(BaseModel):
    """content excludes ImagePart because assistant turns return text only."""

    model_config = ConfigDict(frozen=True)

    content: str | tuple[TextPart, ...]
    tool_calls: tuple[ToolCall, ...] = ()
    role: Literal["assistant"] = "assistant"

    @property
    def text(self) -> str:
        """The concatenated text of the turn; empty when the turn was only tool calls."""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)


class ToolMessage(BaseModel):
    """One tool result sent back to the model.

    tool_call_id must match the id of the ToolCall it answers.
    is_error True tells the model the tool failed; content then holds the error text.
    """

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    content: str | tuple[Part, ...]
    is_error: bool = False
    role: Literal["tool"] = "tool"


type Message = Annotated[
    UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]
"""Discriminated on role: pydantic validation selects the member from the tag,
never from which member's fields happen to match,
so callers can persist a conversation as JSON and re-validate it with a TypeAdapter.
"""

type StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]
"""Provider stop reasons normalized to one vocabulary;
adapters map unrecognized provider values to "other" so a new provider value cannot break callers.
"""
