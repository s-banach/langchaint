"""Provider-neutral messages and content parts.

Messages carry no provider knowledge;
adapters convert a whole Sequence[Message] to wire shapes because conversion depends on the full sequence,
not on one message at a time.
The system prompt is a generate-method parameter, not a message type,
because providers place it in different request locations.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter, model_validator

from langchaint.checked_copy import CheckedCopyModel


class TextPart(CheckedCopyModel):
    """One text span of a message's content.

    cache_breakpoint True marks the exact end of a reusable prompt prefix:
    everything from the start of the request through this part is the span the provider may cache.
    The adapters map it to anthropic's block-level cache_control and openai's part-level prompt_cache_breakpoint.
    Only the latest marks become cache writes, so a Sequence[Message] that accrues one mark per turn
    keeps working as it grows: each adapter's docstring states the per-request write limit and
    whether the adapter or the API is what applies it.
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
"""One model-facing value inside message content.

Convert a document before sending it, into one form picked by its content.
Extract the text layer to TextPart where words carry the meaning; rasterize the pages to ImagePart where layout does.
The application owns that conversion,
so it picks the resolution, which pages to send, and the text extractor, and can measure what each choice costs.
"""

type MessageContent = str | Sequence[ContentPart]
"""A model-facing message body the model reads.
This is the constructor-facing form a caller or tool provides;
the pydantic message models store it as str | tuple[ContentPart, ...], coercing the sequence to a frozen tuple,
so their fields spell that tuple form out rather than aliasing it.
It is not the possibly-structured generation Response.output,
which can be a parsed BaseModel that is not a ContentPart and never round-trips back into a message body.
"""


class ToolCall(CheckedCopyModel):
    """One tool call requested by the model.

    args_json is the raw argument JSON text before validation;
    adapters whose provider delivers decoded arguments serialize them back to JSON
    so every provider yields the same shape.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    args_json: str
    kind: Literal["tool_call"] = "tool_call"


class UserMessage(CheckedCopyModel):
    """One user turn; content is plain text or a tuple of parts.

    kind discriminates the Message union,
    so a persisted Sequence[Message] re-validates to the same message types by construction
    instead of by variant order.

    content is keyword-only, as on every model here; CheckedCopyModel's module docstring says why a
    positional UserMessage("Hello") is rejected. A Sequence[Message] that is one user turn goes to
    BoundLLM.generate_one as a bare string, which wraps it in a UserMessage.

    Raises:
        pydantic.ValidationError: content is neither str nor a sequence of ContentPart values,
            or a key that is not a field was passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str | tuple[ContentPart, ...]
    kind: Literal["user"] = "user"


class ReasoningPart(CheckedCopyModel):
    """One ReasoningPart the model produced, round-tripped verbatim.

    The core never inspects raw: raw is the producing SDK item's model_dump(exclude_none=True),
    and the consuming adapter re-feeds it
    to the wire unchanged so the provider reads it byte-identical (Anthropic rejects a modified
    thinking block; OpenAI re-reads encrypted_content; Gemini requires thought signatures resent
    exactly as received).
    No provider reads another provider's ReasoningPart.raw.
    Switching providers requires rebuilding turns without foreign ReasoningPart values.
    each adapter's docstring states what its wire does with a foreign one.
    Full reasoning history is the default; editing the Sequence[Message] is the only way to change it.
    Trimming is the application's job.
    Trimming for length removes whole turns.
    A kept turn keeps its ReasoningPart values.
    The anthropic API 400s a tool-use continuation missing the latest assistant turn's thinking.
    Beyond replay correctness, keeping ReasoningPart values matters for quality
    (a reasoning model that cannot see its prior reasoning across a tool loop re-derives or contradicts itself)
    and for prompt caching:
    reasoning sits inside the growing cached prefix, so cache hits need it present and byte-identical every turn.
    The dict field makes this model unhashable; messages are never hashed.

    text is the provider's readable text, assembled from text already inside raw
    and adding nothing raw does not hold;
    raw alone is what the adapter replays, so editing text changes what telemetry and an
    application display and never changes the request.
    None means no readable text came back.
    No adapter stores the empty string, so text-free is the single condition text is None.
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
    """One assistant turn, stored as the ordered part sequence the provider emitted.

    Both providers emit and require the order (Anthropic cannot rearrange thinking blocks;
    OpenAI replays output items in their original order under store=False),
    so the one stored sequence is turn and text/tool_calls are filtered views of it.
    A bare string turn is one TextPart, for hand-written turns such as few-shot examples.

    turn is keyword-only, as on every model here.

    Raises:
        pydantic.ValidationError: a key that is not a field was passed,
            or turn is neither str nor a sequence of TurnPart values,
            or a TextPart in the turn sets cache_breakpoint
            (openai has no breakpoint on assistant replay text,
            so a marked assistant part would be a provider-divergent runtime failure;
            mark the following user or tool message instead).
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
    """Serialize messages as JSON text that messages_from_json restores.

    A release that breaks loading says so in its release notes.

    The output is compact; pass indent (pydantic's dump_json indent) for text a human reads or diffs.
    Each ReasoningPart.raw and each RawPart.raw is embedded as the mapping the adapter replays,
    so a restored conversation builds the same wire request as the original;
    the conformance suite asserts that per adapter.

    Raises:
        pydantic_core.PydanticSerializationError: a ReasoningPart.raw or RawPart.raw value is
            an object JSON cannot represent.
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
