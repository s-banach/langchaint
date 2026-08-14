"""Round-trip multimodal messages through JSON."""

from langchaint import (
    AudioPart,
    ImagePart,
    ImageUrlPart,
    Message,
    TextPart,
    UserMessage,
    messages_from_json,
    messages_to_json,
)


def serialize_messages() -> str:
    """Serialize messages and verify the round trip."""
    messages: list[Message] = [
        UserMessage(
            content=[
                TextPart(text="Compare these inputs."),
                ImagePart(data=b"local image bytes", media_type="image/png"),
                ImageUrlPart(url="https://example.com/image.png", media_type="image/png"),
                AudioPart(data=b"local audio bytes", media_type="audio/wav"),
            ]
        )
    ]
    messages_json = messages_to_json(messages, indent=2)
    restored_messages = messages_from_json(messages_json)
    assert restored_messages == messages
    return messages_json
