"""Test custom message validation and JSON round trips."""

import json
import math

import pytest
from pydantic import TypeAdapter, ValidationError

from langchaint import (
    AssistantMessage,
    AudioPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
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


def test_string_turn_coercion() -> None:
    """A bare string turn coerces to one TextPart on every construction path."""
    assistant = AssistantMessage(turn="hey")
    assert assistant.turn == (TextPart(text="hey"),)
    assert assistant == AssistantMessage(turn=(TextPart(text="hey"),))
    assert AssistantMessage.model_validate({"kind": "assistant", "turn": "hey"}) == assistant


def test_tool_message_error_binds_the_call_id_and_sets_is_error() -> None:
    """ToolMessage.error equals the keyword construction with tool_call_id from the call and is_error True."""
    tool_call = ToolCall(id="c1", name="lookup", args_json="{}")
    message = ToolMessage.error(tool_call, "boom")
    assert message == ToolMessage(tool_call_id="c1", content="boom", is_error=True)


def test_image_bytes_serialize_as_url_safe_base64_text() -> None:
    """The JSON text of data is URL-safe base64, pinned exactly so the persisted form cannot drift."""
    image = ImagePart(data=b"\x89PNG\x00\xff", media_type="image/png")
    assert json.loads(image.model_dump_json())["data"] == "iVBORwD_"


def test_audio_bytes_round_trip_as_url_safe_base64() -> None:
    """AudioPart JSON stores data as URL-safe base64 text."""
    audio = AudioPart(data=b"\xfb\xff", media_type="audio/wav")
    audio_json = audio.model_dump_json()
    assert json.loads(audio_json)["data"] == "-_8="
    assert AudioPart.model_validate_json(audio_json).data == b"\xfb\xff"


def test_cache_breakpoint_round_trips() -> None:
    """Preserve marked and unmarked parts through JSON."""
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
    """The default output holds no newlines. indent produces the same messages pretty-printed."""
    messages = _one_of_each_message()
    compact = messages_to_json(messages)
    pretty = messages_to_json(messages, indent=2)
    assert "\n" not in compact
    assert "\n" in pretty
    assert messages_from_json(pretty) == messages_from_json(compact)


def test_reasoning_part_rejects_a_raw_value_json_cannot_represent() -> None:
    """A non-JSON value fails before its `ReasoningPart` enters a message."""
    with pytest.raises(ValidationError):
        _ = TypeAdapter(ReasoningPart).validate_python({"raw": {"payload": object()}})


@pytest.mark.parametrize("value", [(1, 2), {1, 2}])
def test_reasoning_part_rejects_a_non_json_container(value: object) -> None:
    """A tuple or set fails before its `ReasoningPart` enters a message."""
    with pytest.raises(ValidationError):
        _ = TypeAdapter(ReasoningPart).validate_python({"raw": {"payload": value}})


@pytest.mark.parametrize("container_kind", ["list", "dict"])
def test_reasoning_part_rejects_a_cyclic_container(container_kind: str) -> None:
    """A cyclic list or dict fails before its `ReasoningPart` enters a message."""
    if container_kind == "list":
        list_value: list[object] = []
        list_value.append(list_value)
        value: object = list_value
    else:
        dict_value: dict[str, object] = {}
        dict_value["self"] = dict_value
        value = dict_value
    with pytest.raises(ValidationError, match="cycle"):
        _ = TypeAdapter(ReasoningPart).validate_python({"raw": {"payload": value}})


def test_raw_parts_accept_finite_recursive_json() -> None:
    """Finite recursive JSON reconstructs through both provider replay parts."""
    raw: dict[str, JsonValue] = {
        "none": None,
        "flag": True,
        "count": 3,
        "ratio": 1.5,
        "text": "value",
        "nested": [1, {"next": False}],
    }
    for part in (ReasoningPart(raw=raw), RawPart(raw=raw)):
        restored = TypeAdapter(type(part)).validate_json(TypeAdapter(type(part)).dump_json(part))
        assert restored == part


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("part_type", [ReasoningPart, RawPart])
def test_raw_parts_reject_nonfinite_recursive_json(
    value: float, part_type: type[ReasoningPart] | type[RawPart]
) -> None:
    """A non-finite nested float fails before provider replay data enters a message."""
    with pytest.raises(ValidationError):
        _ = TypeAdapter(part_type).validate_python({"raw": {"nested": [value]}})


def test_model_copy_rejects_a_derived_property_key() -> None:
    """model_copy(update={"tool_calls": ...}) raises instead of silently dropping the key.

    pydantic's unvalidated copy would leave turn unchanged.
    The property would shadow the unused key.
    An app could then resend the unfiltered assistant turn.
    """
    message = AssistantMessage(turn=(ToolCall(id="c1", name="probe", args_json="{}"),))
    with pytest.raises(TypeError, match="derived property of AssistantMessage"):
        _ = message.model_copy(update={"tool_calls": ()})


def test_model_copy_rejects_a_key_that_is_not_a_field() -> None:
    """A typo key raises and the message lists the model's fields."""
    with pytest.raises(TypeError, match="not a field of UserMessage"):
        _ = UserMessage(content="hi").model_copy(update={"contnet": "bye"})
