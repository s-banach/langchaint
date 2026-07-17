"""Provider-neutral messages and content parts.

Messages carry no provider knowledge;
provider adapters convert whole conversations to wire shapes because conversion depends on the full sequence,
not on one message at a time.
The system prompt is a generate-method parameter, not a message type,
because providers place it in different request locations.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class TextPart(BaseModel):
    """One text span of a message's content.

    cache_breakpoint True marks the exact end of a reusable prompt prefix:
    everything from the start of the request through this part is the span the provider may cache.
    The adapters map it to anthropic's block-level cache_control and openai's part-level prompt_cache_breakpoint.
    Each provider writes at most its per-request budget of breakpoints (4 on both) and keeps the latest,
    so a conversation that accrues one mark per turn keeps working as it grows.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    cache_breakpoint: bool = False


class ImagePart(BaseModel):
    """media_type is an IANA media type such as "image/png".

    cache_breakpoint has the same meaning as on TextPart: the reusable prompt prefix ends at this part.
    """

    model_config = ConfigDict(frozen=True)

    data: bytes
    media_type: str
    cache_breakpoint: bool = False


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

    Raises:
        pydantic.ValidationError: content is neither a str nor a sequence of Parts.
    """

    model_config = ConfigDict(frozen=True)

    content: str | tuple[Part, ...]
    role: Literal["user"] = "user"

    def __init__(self, content: MessageContent, role: Literal["user"] = "user") -> None:
        """Accept content positionally, so a conversation reads UserMessage("Hello").

        role stays a parameter because pydantic validation (model_validate, TypeAdapter)
        routes through a custom __init__ and passes every field.
        """
        super().__init__(content=content, role=role)


class ReasoningTrace(BaseModel):
    """One reasoning element the model produced, round-tripped verbatim.

    The core never inspects reasoning: reasoning is the producing SDK item's
    model_dump(mode="python", exclude_none=True), and the consuming adapter re-feeds that dict
    to the wire unchanged so the provider reads it byte-identical (Anthropic rejects a modified
    thinking block; OpenAI re-reads encrypted_content).
    Only the producing provider can accept the dict: replaying it through another provider is a
    malformed request that provider rejects, so switching providers means first rebuilding
    concluded assistant turns without their traces.
    The dict field makes this model unhashable, unlike its frozen siblings; messages are never hashed.
    """

    model_config = ConfigDict(frozen=True)

    reasoning: Mapping[str, object]


type TurnElement = ReasoningTrace | TextPart | ToolCall
"""One element of an assistant turn, ordered as the provider emitted them.
The three members have disjoint field sets, so pydantic selects the member by field match
on re-validation of a persisted conversation, like the Part union.
TextPart, not Part: assistant turns still return no images.
"""


def _text_only_turn(turn: object) -> object:
    """Coerce a bare string to a one-TextPart turn, so AssistantMessage("hey") works.

    Runs before validation on every construction path (the constructor and model_validate alike),
    so the stored turn is always the tuple form and readers never branch on a string.
    """
    if isinstance(turn, str):
        return (TextPart(text=turn),)
    return turn


class AssistantMessage(BaseModel):
    """One assistant turn, stored as the ordered element sequence the provider emitted.

    Both providers emit and require the order (Anthropic cannot rearrange thinking blocks;
    OpenAI replays output items in their original order under store=False),
    so the one stored sequence is turn and text/tool_calls are filtered views of it.
    A bare string turn is one TextPart, for hand-written turns such as few-shot examples.

    Raises:
        pydantic.ValidationError: turn is neither a str nor a sequence of TurnElements,
            or a TextPart in the turn sets cache_breakpoint
            (openai has no breakpoint on assistant replay text,
            so a marked assistant part would be a provider-divergent runtime failure;
            mark the following user or tool message instead).
    """

    model_config = ConfigDict(frozen=True)

    turn: Annotated[tuple[TurnElement, ...], BeforeValidator(_text_only_turn)]
    role: Literal["assistant"] = "assistant"

    @model_validator(mode="after")
    def _reject_cache_breakpoint(self) -> "AssistantMessage":
        """Reject a turn whose TextPart sets cache_breakpoint; the class docstring states why.

        Raises:
            ValueError: a TextPart in the turn sets cache_breakpoint; pydantic surfaces it as a ValidationError.
        """
        if any(
            isinstance(element, TextPart) and element.cache_breakpoint for element in self.turn
        ):
            raise ValueError(
                "cache_breakpoint is not supported on assistant turn text: "
                "openai has no breakpoint on assistant replay text; "
                "mark the following user or tool message instead"
            )
        return self

    def __init__(
        self, turn: str | Sequence[TurnElement], role: Literal["assistant"] = "assistant"
    ) -> None:
        """Accept turn positionally, so a conversation reads AssistantMessage("hey").

        role stays a parameter because pydantic validation (model_validate, TypeAdapter)
        routes through a custom __init__ and passes every field.
        """
        super().__init__(turn=turn, role=role)

    @property
    def text(self) -> str:
        """The concatenated TextPart texts of the turn; empty when the turn held no text."""
        return "".join(element.text for element in self.turn if isinstance(element, TextPart))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The ToolCall elements of the turn, in emission order."""
        return tuple(element for element in self.turn if isinstance(element, ToolCall))


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
