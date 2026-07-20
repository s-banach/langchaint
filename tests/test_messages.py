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
    ReasoningTrace,
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
            turn=(
                ReasoningTrace(
                    reasoning={"type": "thinking", "thinking": "check first", "signature": "sig"}
                ),
                TextPart(text="Checking."),
                ToolCall(id="c1", name="probe", args_json='{"step": 1}'),
            ),
        ),
        ToolMessage(tool_call_id="c1", content="probe 1: ok"),
        ToolMessage(tool_call_id="c1", content="boom", is_error=True),
        AssistantMessage(turn=(TextPart(text="Done."),)),
    )
    restored = _CONVERSATION_ADAPTER.validate_json(_CONVERSATION_ADAPTER.dump_json(conversation))
    assert restored == conversation
    assert [type(message) for message in restored] == [type(message) for message in conversation]
    restored_assistant = restored[1]
    assert isinstance(restored_assistant, AssistantMessage)
    assert [type(element) for element in restored_assistant.turn] == [
        ReasoningTrace,
        TextPart,
        ToolCall,
    ]


def test_validation_selects_the_member_from_the_role_tag() -> None:
    """The role tag selects the message type, not which member's fields happen to match."""
    user = _CONVERSATION_ADAPTER.validate_python([{"role": "user", "content": "hi"}])[0]
    assert type(user) is UserMessage
    assistant = _CONVERSATION_ADAPTER.validate_python(
        [{"role": "assistant", "turn": [{"text": "hi"}]}]
    )[0]
    assert type(assistant) is AssistantMessage


def test_turn_elements_validate_by_field_match() -> None:
    """TurnElement has no discriminator: matching the most fields selects the type.

    A persisted turn whose dicts re-validate to the wrong member would silently corrupt replay.
    ReasoningTrace and TextPart share a text field, so the reasoning key is what separates them,
    and it separates them by fields matched rather than by failing TextPart validation:
    TextPart ignores extra keys, so it accepts the two-key dict on its own and still loses.
    """
    message = AssistantMessage.model_validate({
        "role": "assistant",
        "turn": [
            {"reasoning": {"type": "reasoning", "id": "rs_1"}, "text": "thought it over"},
            {"reasoning": {"type": "reasoning", "id": "rs_2"}},
            {"text": "hi"},
            {"id": "c1", "name": "probe", "args_json": "{}"},
        ],
    })
    assert [type(element) for element in message.turn] == [
        ReasoningTrace,
        ReasoningTrace,
        TextPart,
        ToolCall,
    ]
    with_text, without_text = message.turn[0], message.turn[1]
    assert isinstance(with_text, ReasoningTrace)
    assert isinstance(without_text, ReasoningTrace)
    assert with_text.text == "thought it over"
    assert without_text.text is None
    accepted_alone = TextPart.model_validate({"reasoning": {}, "text": "thought it over"})
    assert accepted_alone.text == "thought it over"


def test_validation_without_a_role_tag_is_rejected() -> None:
    """A message payload missing role fails validation, proving the discriminator is engaged."""
    with pytest.raises(ValidationError):
        _CONVERSATION_ADAPTER.validate_python([{"content": "hi"}])


def test_positional_construction_and_string_turn_coercion() -> None:
    """UserMessage and AssistantMessage take their one argument positionally.

    A bare string turn coerces to one TextPart on every construction path.
    """
    assert UserMessage("Hello") == UserMessage(content="Hello")
    assistant = AssistantMessage("hey")
    assert assistant.turn == (TextPart(text="hey"),)
    assert assistant == AssistantMessage(turn=(TextPart(text="hey"),))
    assert AssistantMessage.model_validate({"role": "assistant", "turn": "hey"}) == assistant


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


def test_cache_breakpoint_round_trips_and_defaults_false() -> None:
    """A marked part survives the JSON round trip; an unmarked part re-validates with the default."""
    conversation: tuple[Message, ...] = (
        UserMessage(content=(TextPart(text="shared context", cache_breakpoint=True), TextPart(text="question"))),
        ToolMessage(
            tool_call_id="c1",
            content=(ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),),
        ),
    )
    restored = _CONVERSATION_ADAPTER.validate_json(_CONVERSATION_ADAPTER.dump_json(conversation))
    assert restored == conversation
    restored_user = restored[0]
    assert isinstance(restored_user, UserMessage)
    assert isinstance(restored_user.content, tuple)
    assert restored_user.content[0].cache_breakpoint is True
    assert restored_user.content[1].cache_breakpoint is False


def test_assistant_turn_rejects_a_marked_text_part() -> None:
    """A TextPart with cache_breakpoint in an assistant turn fails validation on every construction path."""
    marked = TextPart(text="hey", cache_breakpoint=True)
    with pytest.raises(ValidationError, match="cache_breakpoint"):
        AssistantMessage(turn=(marked,))
    with pytest.raises(ValidationError, match="cache_breakpoint"):
        AssistantMessage.model_validate({
            "role": "assistant",
            "turn": [{"text": "hey", "cache_breakpoint": True}],
        })


def test_assistant_turn_still_accepts_unmarked_text() -> None:
    """The validator rejects only marked parts; the plain turn is untouched."""
    assert AssistantMessage("hey").text == "hey"


def test_model_copy_rejects_a_derived_property_key() -> None:
    """model_copy(update={"tool_calls": ...}) raises instead of silently dropping the key.

    pydantic's unvalidated copy would leave turn unchanged while the property shadows the dead key,
    so an app filtering an assistant turn's tool calls this way would re-send the unfiltered turn.
    """
    message = AssistantMessage(turn=(ToolCall(id="c1", name="probe", args_json="{}"),))
    with pytest.raises(TypeError, match="derived property of AssistantMessage"):
        message.model_copy(update={"tool_calls": ()})


def test_model_copy_rejects_a_key_that_is_not_a_field() -> None:
    """A typo key raises and the message lists the model's fields."""
    with pytest.raises(TypeError, match="not a field of UserMessage"):
        UserMessage("hi").model_copy(update={"contnet": "bye"})


def test_model_copy_with_a_field_key_returns_the_modified_copy() -> None:
    """A field key passes the check and modifies the frozen model's copy as on pydantic's model_copy."""
    message = ToolMessage(tool_call_id="c1", content="ok")
    copy = message.model_copy(update={"is_error": True})
    assert copy.is_error is True
    assert message.is_error is False
