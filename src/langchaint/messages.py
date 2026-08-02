"""Provider-neutral messages and content parts.

Messages carry no provider knowledge;
adapters convert a whole Sequence[Message] to wire shapes because conversion depends on the full sequence,
not on one message at a time.
The system prompt is a generate-method parameter, not a message type,
because providers place it in different request locations.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

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


type Part = Annotated[TextPart | ImagePart, Field(discriminator="kind")]
"""One element of a message body.

Send a document by converting it first: rasterize its pages to ImagePart, or extract its text layer to TextPart.
The application owns that conversion,
so it picks the resolution, which pages to send, and the text extractor, and can measure what each choice costs.
"""

type MessageContent = str | Sequence[Part]
"""A model-facing message body (text and images the model reads).
This is the constructor-facing form a caller or tool provides;
the pydantic message models store it as str | tuple[Part, ...], coercing the sequence to a frozen tuple,
so their fields spell that tuple form out rather than aliasing it.
It is not the possibly-structured generation Response.output,
which can be a parsed BaseModel that is not a Part and never round-trips back into a message body.
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
    instead of by union member order.

    content is keyword-only, as on every model here; CheckedCopyModel's module docstring says why a
    positional UserMessage("Hello") is rejected. A Sequence[Message] that is one user turn goes to
    BoundLLM.generate_one as a bare string, which wraps it in a UserMessage.

    Raises:
        pydantic.ValidationError: content is neither a str nor a sequence of Parts,
            or a key that is not a field was passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str | tuple[Part, ...]
    kind: Literal["user"] = "user"


class ReasoningTrace(CheckedCopyModel):
    """One reasoning element the model produced, round-tripped verbatim.

    The core never inspects raw: raw is the producing SDK item's
    model_dump(mode="python", exclude_none=True), and the consuming adapter re-feeds that dict
    to the wire unchanged so the provider reads it byte-identical (Anthropic rejects a modified
    thinking block; OpenAI re-reads encrypted_content).
    Only the producing provider can accept the dict: replaying it through another provider is a
    malformed request that provider rejects, so switching providers means first rebuilding
    concluded assistant turns without their traces.
    Full reasoning history is the default, and editing the Sequence[Message] is the only way to change it:
    trimming is the application's job, done by rebuilding concluded assistant turns without their traces;
    a turn whose tool calls still await results must keep its reasoning,
    and there is no bind-time on/off parameter because it would be redundant with that edit.
    Beyond replay correctness, keeping traces matters for quality
    (a reasoning model that cannot see its prior reasoning across a tool loop re-derives or contradicts itself)
    and for prompt caching:
    reasoning sits inside the growing cached prefix, so cache hits need it present and byte-identical every turn.
    The dict field makes this model unhashable, unlike its frozen siblings; messages are never hashed.

    text is the provider's readable text, assembled from text already inside raw
    and adding nothing raw does not hold;
    raw alone is what the adapter replays, so editing text changes what telemetry and an
    application display and never changes the request.
    None means no readable text came back: an anthropic redacted_thinking block (which carries only
    an opaque string), an anthropic thinking block whose thinking is empty, or an openai response
    holding neither a reasoning summary nor reasoning content.
    No adapter stores the empty string, so text-free is the single condition text is None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: Mapping[str, object]
    text: str | None = None
    kind: Literal["reasoning_trace"] = "reasoning_trace"


type TurnElement = Annotated[ReasoningTrace | TextPart | ToolCall, Field(discriminator="kind")]
"""One element of an assistant turn, ordered as the provider emitted them.
Discriminated on kind, so re-validating a persisted turn selects each element's member from its tag.
TextPart, not Part: assistant turns still return no images.
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
    """One assistant turn, stored as the ordered element sequence the provider emitted.

    Both providers emit and require the order (Anthropic cannot rearrange thinking blocks;
    OpenAI replays output items in their original order under store=False),
    so the one stored sequence is turn and text/tool_calls are filtered views of it.
    A bare string turn is one TextPart, for hand-written turns such as few-shot examples.

    turn is keyword-only, as on every model here.

    Raises:
        pydantic.ValidationError: a key that is not a field was passed,
            or turn is neither a str nor a sequence of TurnElements,
            or a TextPart in the turn sets cache_breakpoint
            (openai has no breakpoint on assistant replay text,
            so a marked assistant part would be a provider-divergent runtime failure;
            mark the following user or tool message instead).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: Annotated[tuple[TurnElement, ...], BeforeValidator(_text_only_turn)]
    kind: Literal["assistant"] = "assistant"

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

    @property
    def text(self) -> str:
        """The concatenated TextPart texts of the turn; empty when the turn held no text."""
        return "".join(element.text for element in self.turn if isinstance(element, TextPart))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The ToolCall elements of the turn, in emission order."""
        return tuple(element for element in self.turn if isinstance(element, ToolCall))


class ToolMessage(CheckedCopyModel):
    """One tool result sent back to the model.

    tool_call_id must match the id of the ToolCall it answers.
    is_error True tells the model the tool failed; content then holds the error text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_call_id: str
    content: str | tuple[Part, ...]
    is_error: bool = False
    kind: Literal["tool"] = "tool"

    @classmethod
    def error(cls, tool_call: ToolCall, content: str | tuple[Part, ...]) -> "ToolMessage":
        """Build an is_error ToolMessage answering tool_call.

        tool_call_id is bound from tool_call.id, so the message answers that call by construction.
        """
        return cls(tool_call_id=tool_call.id, content=content, is_error=True)


type Message = Annotated[UserMessage | AssistantMessage | ToolMessage, Field(discriminator="kind")]
"""Discriminated on kind: pydantic validation selects the member from the tag,
never from which member's fields happen to match,
so callers can persist a Sequence[Message] as JSON and re-validate it with a TypeAdapter.
"""

type StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "refusal", "context_window_exceeded", "other"
]
"""Provider stop reasons normalized to one vocabulary;
adapters map unrecognized provider values to "other" so a new provider value cannot break callers.
context_window_exceeded carries no provider prefix because this vocabulary is langchaint's own;
it earns a member rather than "other" because it names a terminal condition a caller acts on,
by shortening the GenerationInput or moving to a model with a larger window.
"""
