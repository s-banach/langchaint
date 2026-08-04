# Richer content parts: image by URL, and audio

A plan for TODO.md's last item. Documents are out of scope, and the reason is already written in `Part`'s docstring: the application extracts the text layer or rasterizes the pages, and owns the resolution, the page selection, and the extractor.

langchaint never fetches a URL. A URL part is a URL on the wire, and the provider fetches it or refuses.

## What each adapter can represent

Verified by introspection against anthropic 0.120.2, openai 2.51.0, google-genai 2.16.0.

| adapter | image by URL | audio |
| --- | --- | --- |
| anthropic Messages | `URLImageSourceParam(type="url", url=...)` | nothing: `anthropic.types` holds no audio name |
| openai Responses | `ResponseInputImageParam.image_url` | nothing: `ResponseInputContentParam` is text, image, file |
| openai Chat Completions | `ChatCompletionContentPartImageParam` | `ChatCompletionContentPartInputAudioParam`, `format` is `Literal["wav", "mp3"]` |
| gemini | `FileData(file_uri, mime_type)` | `Blob(data, mime_type)` |

`ResponseInputAudioParam` exists in the openai package but is absent from the Responses input-content union, so the Responses adapter has nothing to send it as.

Image by URL is representable everywhere. Audio is representable in two adapters of four.

## Where the raise happens

Letting the provider refuse the part works wherever a wire field exists to send it in.
Where no field exists, langchaint cannot serialize the part, so nothing reaches the provider to be refused.
That is not a guess about a provider rule; it is the absence of a field in the SDK's own union.

The existing path already covers it.
`_NotSendableError` names an unsendable `Sequence[Message]` inside an adapter, and `build_request` returns `InvalidRequest(reason=...)`.
That becomes an `InvalidRequestError`, which is one item's failure row, so a batch's other items still run.

So:

- An `ImageUrlPart` goes on the wire in all four adapters. Gemini decides whether it fetches an arbitrary `https://` URL; langchaint makes no claim about that.
- An `AudioPart` goes on the wire in the Chat Completions and gemini adapters. In the anthropic and openai Responses adapters it is `InvalidRequest`, with the reason naming the adapter and the part.
- An `AudioPart` whose `media_type` is outside `audio/wav` and `audio/mpeg` is `InvalidRequest` in the Chat Completions adapter, because `format` admits only `wav` and `mp3`. Gemini takes the media type through.

## The parts

```python
class ImageUrlPart(CheckedCopyModel):
    """url is fetched by the provider, never by langchaint.

    media_type is an IANA media type such as "image/png".
    Only gemini carries it on the wire, in FileData.mime_type;
    anthropic's url source and openai's image_url are the URL alone.

    cache_breakpoint has the same meaning as on TextPart.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    media_type: str | None = None
    cache_breakpoint: bool = False
    kind: Literal["image_url"] = "image_url"


class AudioPart(CheckedCopyModel):
    """media_type is an IANA media type such as "audio/wav".

    In JSON, data is URL-safe base64 text, as on ImagePart.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    data: bytes
    media_type: str
    cache_breakpoint: bool = False
    kind: Literal["audio"] = "audio"


type Part = Annotated[TextPart | ImagePart | ImageUrlPart | AudioPart, Field(discriminator="kind")]
```

A separate class rather than a nullable `url` on `ImagePart`:
every adapter branches on which of the two it holds, and the discriminated union is what makes the branch exhaustive.

`ImagePart` keeps its name. It carries inline bytes, which is the common case, and renaming it breaks callers for no gain.

## Round trip

`Part`'s JSON form gains two `kind` values.
A reader on an older langchaint rejects them, because the union has `extra="forbid"` and no matching tag.
That is the pinned `messages_to_json` format changing, so it belongs in the release note.

## Conformance

Two invariants, one per part, in `conformance.py`:

- A message holding an `ImageUrlPart` builds a request in every adapter.
- A message holding an `AudioPart` either builds a request or returns `InvalidRequest` naming the part, and never raises out of `build_request`.

The second is what keeps the two-of-four support honest without making audio look universal.

## What this leaves unanswered

Whether `AudioPart` earns its place at two adapters of four.
The alternative is to write the answer in `Part`'s docstring, next to the document sentence: send audio through the Chat Completions or gemini backend, and there is no anthropic route.
