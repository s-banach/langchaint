"""The Message union's role discriminator.

Persist/resume serializes a conversation with a TypeAdapter and re-validates it;
these tests pin that the round trip preserves each message's type via the role tag,
not via which union member's fields happen to match first.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from langchaint import (
    AssistantMessage,
    ImagePart,
    Message,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_CONVERSATION_ADAPTER: TypeAdapter[tuple[Message, ...]] = TypeAdapter(tuple[Message, ...])


def test_conversation_round_trips_through_json_preserving_types() -> None:
    """dump_json then validate_json returns equal messages of the same types."""
    conversation: tuple[Message, ...] = (
        UserMessage(content=(TextPart(text="look"), ImagePart(data=b"png", media_type="image/png"))),
        AssistantMessage(
            content="Checking.",
            tool_calls=(ToolCall(id="c1", name="probe", args_json='{"step": 1}'),),
        ),
        ToolMessage(tool_call_id="c1", content="probe 1: ok"),
        ToolMessage(tool_call_id="c1", content="boom", is_error=True),
        AssistantMessage(content="Done."),
    )
    restored = _CONVERSATION_ADAPTER.validate_json(_CONVERSATION_ADAPTER.dump_json(conversation))
    assert restored == conversation
    assert [type(message) for message in restored] == [type(message) for message in conversation]


def test_validation_selects_the_member_from_the_role_tag() -> None:
    """A payload whose fields alone are ambiguous validates to the type its role names.

    content-only dicts satisfy both UserMessage and AssistantMessage; only the tag decides.
    """
    user = _CONVERSATION_ADAPTER.validate_python([{"role": "user", "content": "hi"}])[0]
    assert type(user) is UserMessage
    assistant = _CONVERSATION_ADAPTER.validate_python([{"role": "assistant", "content": "hi"}])[0]
    assert type(assistant) is AssistantMessage


def test_validation_without_a_role_tag_is_rejected() -> None:
    """A message payload missing role fails validation, proving the discriminator is engaged."""
    with pytest.raises(ValidationError):
        _CONVERSATION_ADAPTER.validate_python([{"content": "hi"}])


def test_tool_message_content_accepts_parts_and_coerces_a_list_to_a_tuple() -> None:
    """A ToolMessage can carry text and image parts; a list of parts coerces to a tuple like UserMessage."""
    parts = [TextPart(text="saw"), ImagePart(data=b"png", media_type="image/png")]
    message = ToolMessage(tool_call_id="c1", content=parts)
    assert message.content == tuple(parts)
    assert isinstance(message.content, tuple)


def test_tool_message_content_still_round_trips_a_bare_string() -> None:
    """A bare string content stays a string, not a one-element tuple."""
    message = ToolMessage(tool_call_id="c1", content="ok")
    assert message.content == "ok"


def test_tool_message_is_frozen() -> None:
    """ToolMessage is immutable; reassigning content raises."""
    message = ToolMessage(tool_call_id="c1", content="ok")
    with pytest.raises(ValidationError):
        message.content = "changed"  # pyrefly: ignore[read-only]
