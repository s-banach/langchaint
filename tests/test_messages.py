"""Test kind validation and JSON round trips for Message, ContentPart, and TurnPart."""

import json
from collections.abc import Callable

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from langchaint import (
    AssistantMessage,
    AudioPart,
    ImagePart,
    ImageUrlPart,
    Message,
    RawPart,
    ReasoningPart,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    messages_from_json,
    messages_to_json,
)

_MESSAGES_TYPE_ADAPTER: TypeAdapter[tuple[Message, ...]] = TypeAdapter(tuple[Message, ...])


def test_turn_parts_validate_to_the_variant_their_tag_names() -> None:
    """A persisted turn validates each part from its kind.

    kind preserves each variant and ReasoningPart.raw during replay.
    """
    raw = {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1"}
    message = AssistantMessage.model_validate({
        "kind": "assistant",
        "turn": [
            {"kind": "reasoning_part", "raw": raw, "text": "thought it over"},
            {"kind": "reasoning_part", "raw": {"type": "reasoning", "id": "rs_2"}},
            {"kind": "text", "text": "hi"},
            {"kind": "tool_call", "id": "c1", "name": "probe", "args_json": "{}"},
        ],
    })
    assert [type(part) for part in message.turn] == [
        ReasoningPart,
        ReasoningPart,
        TextPart,
        ToolCall,
    ]
    with_text, without_text = message.turn[0], message.turn[1]
    assert isinstance(with_text, ReasoningPart)
    assert isinstance(without_text, ReasoningPart)
    assert with_text.raw == raw
    assert with_text.text == "thought it over"
    assert without_text.text is None


def test_an_untagged_turn_part_is_rejected_at_its_index() -> None:
    """A TurnPart without kind fails at its index.

    The tag selects the variant before reading fields.
    """
    with pytest.raises(ValidationError) as caught:
        _ = AssistantMessage.model_validate({"kind": "assistant", "turn": [{"text": "hi"}]})
    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        (("turn", 0), "union_tag_not_found")
    ]


def test_a_malformed_tagged_turn_part_reports_one_error_at_its_field() -> None:
    """A tagged TurnPart validates against one variant."""
    with pytest.raises(ValidationError) as caught:
        _ = AssistantMessage.model_validate({
            "kind": "assistant",
            "turn": [{"kind": "text", "text": 1}],
        })
    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        (("turn", 0, "text", "text"), "string_type")
    ]


@pytest.mark.parametrize("kind", ["reasoning_trace", "opaque_element"])
def test_reasoning_trace_and_opaque_element_tags_are_rejected(kind: str) -> None:
    """The reasoning_trace and opaque_element tags fail discriminator validation."""
    with pytest.raises(ValidationError) as caught:
        _ = AssistantMessage.model_validate({
            "kind": "assistant",
            "turn": [{"kind": kind, "raw": {}}],
        })
    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        (("turn", 0), "union_tag_invalid")
    ]


def test_validation_without_a_kind_tag_is_rejected() -> None:
    """A message payload missing kind fails validation, proving the discriminator is engaged."""
    with pytest.raises(ValidationError):
        _ = _MESSAGES_TYPE_ADAPTER.validate_python([{"content": "hi"}])


def test_string_turn_coercion() -> None:
    """A bare string turn coerces to one TextPart on every construction path."""
    assistant = AssistantMessage(turn="hey")
    assert assistant.turn == (TextPart(text="hey"),)
    assert assistant == AssistantMessage(turn=(TextPart(text="hey"),))
    assert AssistantMessage.model_validate({"kind": "assistant", "turn": "hey"}) == assistant


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


def test_tool_message_error_binds_the_call_id_and_sets_is_error() -> None:
    """ToolMessage.error equals the keyword construction with tool_call_id from the call and is_error True."""
    tool_call = ToolCall(id="c1", name="lookup", args_json="{}")
    message = ToolMessage.error(tool_call, "boom")
    assert message == ToolMessage(tool_call_id="c1", content="boom", is_error=True)


def test_tool_message_error_accepts_part_content() -> None:
    """ToolMessage.error takes the same part-tuple content as the keyword construction."""
    tool_call = ToolCall(id="c1", name="lookup", args_json="{}")
    parts = (TextPart(text="saw"), ImagePart(data=b"png", media_type="image/png"))
    message = ToolMessage.error(tool_call, parts)
    assert message.content == parts
    assert message.is_error is True


def test_binary_image_bytes_round_trip_through_json() -> None:
    """A UserMessage and a ToolMessage holding non-UTF-8 image bytes survive the JSON round trip.

    Binary image data has a JSON representation independent of UTF-8.
    """
    image = ImagePart(data=b"\x89PNG\x00\xff", media_type="image/png")
    messages: tuple[Message, ...] = (
        UserMessage(content=(image,)),
        ToolMessage(tool_call_id="c1", content=(TextPart(text="saw"), image)),
    )
    restored = _MESSAGES_TYPE_ADAPTER.validate_json(_MESSAGES_TYPE_ADAPTER.dump_json(messages))
    assert restored == messages


def test_image_bytes_serialize_as_url_safe_base64_text() -> None:
    """The JSON text of data is URL-safe base64, pinned exactly so the persisted form cannot drift."""
    image = ImagePart(data=b"\x89PNG\x00\xff", media_type="image/png")
    assert json.loads(image.model_dump_json())["data"] == "iVBORwD_"


@pytest.mark.parametrize(
    ("validate_json", "payload"),
    [
        (
            ImagePart.model_validate_json,
            '{"kind":"image","data":"!!!not-base64!!!","media_type":"image/png"}',
        ),
        (
            AudioPart.model_validate_json,
            '{"kind":"audio","data":"!!!not-base64!!!","media_type":"audio/wav"}',
        ),
    ],
)
def test_malformed_base64_data_is_rejected_on_json_validation(
    validate_json: Callable[[str], object], payload: str
) -> None:
    """Malformed base64 fails JSON validation."""
    with pytest.raises(ValidationError):
        _ = validate_json(payload)


def test_image_url_part_json_is_pinned_exactly() -> None:
    """ImageUrlPart JSON pins url, media_type, cache_breakpoint, and kind."""
    image_url = ImageUrlPart(url="https://example.com/a.png", media_type="image/png")
    assert image_url.model_dump_json() == (
        '{"url":"https://example.com/a.png","media_type":"image/png",'
        '"cache_breakpoint":false,"kind":"image_url"}'
    )


def test_audio_part_json_is_pinned_exactly() -> None:
    """AudioPart JSON stores data as URL-safe base64 text."""
    audio = AudioPart(data=b"\xfb\xff", media_type="audio/wav")
    assert audio.model_dump_json() == (
        '{"data":"-_8=","media_type":"audio/wav","cache_breakpoint":false,"kind":"audio"}'
    )


def test_cache_breakpoint_round_trips_and_defaults_false() -> None:
    """A marked part survives the JSON round trip; an unmarked part re-validates with the default."""
    messages: tuple[Message, ...] = (
        UserMessage(
            content=(
                TextPart(text="shared context", cache_breakpoint=True),
                TextPart(text="question"),
            )
        ),
        ToolMessage(
            tool_call_id="c1",
            content=(ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),),
        ),
    )
    restored = _MESSAGES_TYPE_ADAPTER.validate_json(_MESSAGES_TYPE_ADAPTER.dump_json(messages))
    assert restored == messages
    restored_user = restored[0]
    assert isinstance(restored_user, UserMessage)
    assert isinstance(restored_user.content, tuple)
    assert restored_user.content[0].cache_breakpoint is True
    assert restored_user.content[1].cache_breakpoint is False


def test_assistant_turn_rejects_a_marked_text_part() -> None:
    """A TextPart with cache_breakpoint in an assistant turn fails validation on every construction path."""
    marked = TextPart(text="hey", cache_breakpoint=True)
    with pytest.raises(ValidationError, match="cache_breakpoint"):
        _ = AssistantMessage(turn=(marked,))
    with pytest.raises(ValidationError, match="cache_breakpoint"):
        _ = AssistantMessage.model_validate({
            "kind": "assistant",
            "turn": [{"kind": "text", "text": "hey", "cache_breakpoint": True}],
        })


def test_assistant_turn_still_accepts_unmarked_text() -> None:
    """The validator rejects only marked parts; the plain turn is untouched."""
    assert AssistantMessage(turn="hey").text == "hey"


_PINNED_MESSAGES: tuple[Message, ...] = (
    UserMessage(content=(TextPart(text="context", cache_breakpoint=True), TextPart(text="q"))),
    AssistantMessage(
        turn=(
            ReasoningPart(raw={"type": "thinking", "thinking": "hm", "signature": "s"}, text="hm"),
            TextPart(text="checking"),
            ToolCall(id="c1", name="probe", args_json='{"depth": 2}'),
            ToolCall(id="c2", name="fetch", args_json="{}"),
        )
    ),
    ToolMessage(
        tool_call_id="c1",
        content=(
            TextPart(text="saw"),
            ImagePart(data=b"\x89PNG\x00\xff", media_type="image/png"),
        ),
    ),
    ToolMessage(tool_call_id="c2", content="fetch failed", is_error=True),
    UserMessage(content="and then?"),
)
"""The conversation _PINNED_MESSAGES_JSON encodes. Changing a part invalidates that literal."""


def _one_of_each_message() -> list[Message]:
    """One conversation holding every Message, ContentPart, and TurnPart variant.

    Append a part here, so _PINNED_MESSAGES stays pinned.
    """
    return [
        *_PINNED_MESSAGES,
        AssistantMessage(
            turn=(
                RawPart(raw={"type": "web_search_call", "id": "ws_1"}),
                TextPart(text="ok"),
            )
        ),
        UserMessage(
            content=(
                ImageUrlPart(url="https://example.com/a.png", media_type="image/png"),
                AudioPart(data=b"wav", media_type="audio/wav"),
            )
        ),
        ToolMessage(
            tool_call_id="media_call",
            content=(
                ImageUrlPart(url="https://example.com/tool.png"),
                AudioPart(data=b"mp3", media_type="audio/mpeg"),
            ),
        ),
    ]


def test_messages_json_round_trip_restores_the_list() -> None:
    """messages_from_json returns the list messages_to_json serialized, every type intact."""
    messages = _one_of_each_message()
    assert messages_from_json(messages_to_json(messages)) == messages


_PINNED_MESSAGES_JSON = r'[{"content":[{"text":"context","cache_breakpoint":true,"kind":"text"},{"text":"q","cache_breakpoint":false,"kind":"text"}],"kind":"user"},{"turn":[{"raw":{"type":"thinking","thinking":"hm","signature":"s"},"text":"hm","kind":"reasoning_part"},{"text":"checking","cache_breakpoint":false,"kind":"text"},{"id":"c1","name":"probe","args_json":"{\"depth\": 2}","kind":"tool_call"},{"id":"c2","name":"fetch","args_json":"{}","kind":"tool_call"}],"kind":"assistant"},{"tool_call_id":"c1","content":[{"text":"saw","cache_breakpoint":false,"kind":"text"},{"data":"iVBORwD_","media_type":"image/png","cache_breakpoint":false,"kind":"image"}],"is_error":false,"kind":"tool"},{"tool_call_id":"c2","content":"fetch failed","is_error":true,"kind":"tool"},{"content":"and then?","kind":"user"}]'
"""One messages_to_json output, pasted rather than computed.

This is the persisted-text format applications hold on disk.
Repasting it is a format break: do so only alongside the promised release-note entry.
"""


def test_a_pinned_serialization_still_loads() -> None:
    """messages_from_json loads _PINNED_MESSAGES_JSON, enforcing messages_to_json's stability promise.

    An added field with a default passes, which the promise permits.
    """
    assert messages_from_json(_PINNED_MESSAGES_JSON) == list(_PINNED_MESSAGES)


def test_messages_to_json_is_compact_and_indent_passes_through() -> None:
    """The default output holds no newlines; indent produces the same messages pretty-printed."""
    messages = _one_of_each_message()
    compact = messages_to_json(messages)
    pretty = messages_to_json(messages, indent=2)
    assert "\n" not in compact
    assert "\n" in pretty
    assert messages_from_json(pretty) == messages_from_json(compact)


def test_messages_from_json_rejects_text_that_is_not_a_message_list() -> None:
    """Text holding no serialized message list raises ValidationError, whether or not it is JSON."""
    with pytest.raises(ValidationError):
        _ = messages_from_json("not json")
    with pytest.raises(ValidationError):
        _ = messages_from_json('[{"content": "hi"}]')


def test_messages_to_json_raises_on_a_raw_value_json_cannot_represent() -> None:
    """A ReasoningPart.raw holding a non-JSON-representable object raises, never silently reshapes.

    A hand-built unserializable ReasoningPart raises without a lossy fallback.
    """
    message = AssistantMessage(turn=(ReasoningPart(raw={"payload": object()}),))
    with pytest.raises(PydanticSerializationError):
        _ = messages_to_json([message])


def test_model_copy_rejects_a_derived_property_key() -> None:
    """model_copy(update={"tool_calls": ...}) raises instead of silently dropping the key.

    pydantic's unvalidated copy would leave turn unchanged while the property shadows the dead key,
    so an app filtering an assistant turn's tool calls this way would re-send the unfiltered turn.
    """
    message = AssistantMessage(turn=(ToolCall(id="c1", name="probe", args_json="{}"),))
    with pytest.raises(TypeError, match="derived property of AssistantMessage"):
        _ = message.model_copy(update={"tool_calls": ()})


def test_model_copy_rejects_a_key_that_is_not_a_field() -> None:
    """A typo key raises and the message lists the model's fields."""
    with pytest.raises(TypeError, match="not a field of UserMessage"):
        _ = UserMessage(content="hi").model_copy(update={"contnet": "bye"})


def test_model_copy_with_a_field_key_returns_the_modified_copy() -> None:
    """A field key passes the check and modifies the frozen model's copy as on pydantic's model_copy."""
    message = ToolMessage(tool_call_id="c1", content="ok")
    copy = message.model_copy(update={"is_error": True})
    assert copy.is_error is True
    assert message.is_error is False
