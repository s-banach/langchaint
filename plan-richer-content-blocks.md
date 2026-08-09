# Richer content parts: image by URL and audio

This plan covers TODO.md's final media item.
Documents remain outside `ContentPart`.
The existing `ContentPart` docstring specifies text extraction and page rasterization.
Applications choose the resolution, pages, and extraction tools.

langchaint never fetches a URL.
It sends `url` unchanged through an SDK field.
The provider may accept or refuse that value.

Chat Completions response audio remains available through the result's `raw` field.
`AssistantMessage.turn` does not carry response audio.
To resend its bytes, construct `AudioPart` explicitly.
`AudioPart` remains model input only.

## Decision

Add `ImageUrlPart` and `AudioPart` to the public `ContentPart` union.
Every adapter must serialize each supported context or return `InvalidRequest`.
`InvalidRequest` becomes `InvalidRequestError` before any request starts.
Batch processing returns that failure beside the other inputs' outcomes.
Partial adapter support does not justify omitting a provider-neutral input.

Here, sendable means the installed SDK exposes a matching input field.
A sendable value may still receive a provider rejection.
langchaint does not test provider acceptance before sending.

## Verified installed SDK facts

Verified through introspection against anthropic 0.121.0, openai 2.53.0, and google-genai 2.17.0.

| adapter | `ImageUrlPart` in `UserMessage` | `ImageUrlPart` in `ToolMessage` | `AudioPart` in `UserMessage` | `AudioPart` in `ToolMessage` |
| --- | --- | --- | --- | --- |
| anthropic Messages | `ImageBlockParam` with `URLImageSourceParam` | `ToolResultBlockParam.content` includes `ImageBlockParam` | no audio input content type | no audio tool-result content type |
| openai Responses | `ResponseInputImageParam.image_url` | `ResponseInputImageContentParam.image_url` | `ResponseInputContentParam` excludes audio | `ResponseFunctionCallOutputItemListParam` excludes audio |
| openai Chat Completions | `ChatCompletionContentPartImageParam` | `ChatCompletionToolMessageParam.content` is text-only | `ChatCompletionContentPartInputAudioParam` | `ChatCompletionToolMessageParam.content` is text-only |
| gemini | `Part.file_data` with `FileData` | `FunctionResponsePart.file_data` | `Part.inline_data` with `Blob` | `FunctionResponsePart.inline_data` with `FunctionResponseBlob` |

`ResponseInputAudioParam` exists outside `ResponseInputContentParam`.
`ResponseFunctionCallOutputItemListParam` also excludes audio.
Therefore, the Responses adapter has no audio input-content variant.

`ChatCompletionContentPartInputAudioParam.input_audio.data` is base64 text.
Its `format` field accepts only `"wav"` and `"mp3"`.

`FileData.file_uri` describes a Google Cloud Storage URI.
`FunctionResponsePart.file_data` describes a field unsupported by the Gemini API.
Both SDK models serialize a supplied URI.
langchaint sends the value unchanged and leaves rejection to gemini.

`FileData.mime_type` is documented as required, although its model permits `None`.
`ImageUrlPart.media_type=None` therefore remains sendable and may receive a provider rejection.

## The parts

```python
class ImageUrlPart(CheckedCopyModel):
    """langchaint sends url unchanged and never fetches it.

    media_type is an optional IANA media type.
    Gemini receives it when present.
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
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    data: bytes
    media_type: str
    cache_breakpoint: bool = False
    kind: Literal["audio"] = "audio"


type ContentPart = Annotated[
    TextPart | ImagePart | ImageUrlPart | AudioPart,
    Field(discriminator="kind"),
]
```

`ImagePart` carries inline image bytes.
`ImageUrlPart` carries a URL sent for provider retrieval.
Every adapter selects different SDK fields for those values.
Separate variants make each selection explicit and exhaustively typed.
`ImagePart` keeps its existing name.

## Adapter behavior

Each `ContentPart` converter uses `match part.kind` with one case per literal.
Do not use `case _` or an `else` branch.
A supported case constructs its SDK value.
An unsupported case raises `_NotSendableError` with a standard reason.
`build_request` converts that exception into `InvalidRequest`.
Adding a `ContentPart` variant makes every unchanged match non-exhaustive under pyrefly.
That failure requires an explicit case before CI passes.

For Chat Completions, map `audio/wav` to `"wav"`.
Map `audio/mpeg` to `"mp3"`.
Other `AudioPart.media_type` values return `InvalidRequest` for `UserMessage`.
The adapter returns `InvalidRequest` for any image or audio variant in `ChatCompletionToolMessageParam.content`.

Gemini passes `AudioPart.media_type` through unchanged.
Anthropic and Responses return `InvalidRequest` for every `AudioPart`.
Each reason names the adapter, message type, and `ContentPart` variant.

Anthropic, Chat Completions, and gemini already convert `_NotSendableError` into `InvalidRequest`.
Responses currently states that every `Sequence[Message]` is sendable.
Add equivalent `_NotSendableError` conversion to Responses.

## `cache_breakpoint`

`cache_breakpoint` keeps the `TextPart.cache_breakpoint` meaning on both new variants.
Anthropic carries supported marks through its existing block placement rules.
The openai adapters carry supported marks on their content-part fields.
Gemini returns `InvalidRequest` for any marked message part.
Unsupported parts return `InvalidRequest` without dropping the mark.

## Public integration

Export `ImageUrlPart` and `AudioPart` from top-level `langchaint`.
Update `MessageContent` and tool-output documentation to include both variants.
Update each adapter module docstring with its exact mappings.

Tracing records `ImageUrlPart` with `type`, `url`, and optional `mime_type`.
Use `"image_url"` as that trace part's `type`.
Tracing records `AudioPart` as blob metadata and omits `data`.
Both records remain controlled by `capture_message_content`.

Update every exhaustive `ContentPart.kind` match with `"image_url"` and `"audio"`.

## JSON round trip

`ContentPart` JSON gains `"image_url"` and `"audio"` `kind` values.
The updated union continues loading the pinned existing JSON.
Older releases reject new values with Pydantic's `union_tag_invalid` error.
Announce the new `kind` values in the release notes.

Keep the existing pinned JSON literal unchanged.
Add separate exact JSON tests for both new variants.
Pin `AudioPart.data` as URL-safe base64 text.
Test malformed base64 rejection and complete message round trips.

## Conformance and tests

A `UserMessage` containing `ImageUrlPart` builds in every adapter.
A `ToolMessage` containing `ImageUrlPart` builds except in Chat Completions.
Chat Completions returns `InvalidRequest` for that tool message.

A `UserMessage` containing `AudioPart` builds in Chat Completions and gemini.
A `ToolMessage` containing `AudioPart` builds in gemini.
Other adapter and message combinations return `InvalidRequest` naming `AudioPart`.
No `build_request` call raises for an unsupported part.

Add representative values for every `ContentPart` variant to `AdapterConformance`.
Test each value inside `UserMessage` and `ToolMessage`.
The tool sequence includes an earlier matching `ToolCall`.
Compare the case classes against the variants inside `ContentPart.__value__`.
This assertion fails when `ContentPart` grows without a conformance case.

Each standard case requires `build_request` to return `RequestParams` or `InvalidRequest`.
An `InvalidRequest.reason` names the concrete `ContentPart` class and message class.
No standard case may raise.
`AdapterConformance` also requires `UserMessage` with `ImageUrlPart` to build.

Adapter-specific tests enforce the support table.
Supported cases inspect exact SDK input values, proving the part reached the request.
Unsupported cases assert the expected `InvalidRequest.reason`.
Add tracing tests for metadata and byte omission.
Update the `ContentPart.kind` exhaustiveness tests.

Run `scripts/CI.sh` before committing the implementation.
