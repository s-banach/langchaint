"""What an app does after a batch: read every slot, total the spend, and log the failures.

generate_many never raises on an item. Each conversation settles into its own slot as a Response or
as a GenerationError, the siblings run to completion whatever any one item does, and result i belongs
to conversations[i]. So the call itself needs no try/except; the work is the loop over the results,
which to_row renders to one table shape whether a slot succeeded or failed.

The five GenerationError subclasses each name what happened: RetriesExhaustedError (the transient
budget ran out), RefusalError (the model refused on the structured path), MaxCompletionTokensExceededError
(the structured answer hit the token cap), InvalidRequestError (the request was rejected, by the
provider or by the adapter before sending), and UnrecognizedError (the adapter could not name the
provider's error).
The class says what happened, not what to do next, and InvalidRequestError is where that gap is
widest: a rejected conversation, a bad API key, a revoked permission, and an unknown model id all
land there, so after fixing a key the same items are worth resubmitting unchanged. langchaint does
not guess which it was, because a provider states a status and never a cause. So log the class name
and let whoever reads the log decide.

Needs the anthropic package and ANTHROPIC_API_KEY: two of the three items are real calls, and the
third is refused before anything is sent, so it costs nothing.
"""

import asyncio

from langchaint import (
    GenerationError,
    ImagePart,
    Message,
    Response,
    TextPart,
    Usage,
    UserMessage,
    to_row,
)
from langchaint.anthropic import anthropic_model

# One row of the batch carries a scanned page as a TIFF. The adapter sends only image/gif,
# image/jpeg, image/png, and image/webp, so it refuses this conversation and the item fails its own
# row. The refusal happens before the bytes are base64-encoded, which is why the placeholder below
# is enough to trigger it.
_SCANNED_PAGE: list[Message] = [
    UserMessage(
        content=[
            TextPart(text="Summarize the attached page."),
            ImagePart(data=b"<scan bytes>", media_type="image/tiff"),
        ]
    )
]

_CONVERSATIONS: list[str | list[Message]] = [
    "The quarterly report shows revenue up twelve percent on strong subscription growth.",
    _SCANNED_PAGE,
    "The new compiler release cuts build times roughly in half on large projects.",
]


async def run_batch(
    conversations: list[str | list[Message]],
) -> list[Response[str] | GenerationError]:
    """Run the batch, print every slot as a row, and print what the whole batch paid.

    Nothing here catches anything: the conversation carrying the TIFF settles into its own slot as an
    InvalidRequestError while the two real calls complete, so the list is as long as the input and
    to_row fills the same keys for every slot. A success leaves error_text None, a failure leaves
    output None, and the refused item's attempts is 0 because no request went out.
    Usage.sum_of totals the spend over successes and failures alike, since every GenerationError
    carries the usage its attempts billed.
    """
    summarizer = anthropic_model("claude-sonnet-5").bind(
        system_prompt="Summarize in five words.",
        automatic_prompt_caching=False,
    )
    results = await summarizer.generate_many(conversations)
    for index, result in enumerate(results):
        row = to_row(result)
        print(
            f"item {index}: output={row['output']!r} "
            f"error_text={row['error_text']!r} attempts={row['attempts']}"
        )
    total = Usage.sum_of(result.usage for result in results)
    print(f"batch paid {total.cost_in_usd:.6f} USD")
    return results


def log_failures(results: list[Response[str] | GenerationError]) -> None:
    """Log each failed item's class name and error_text.

    to_row has no error-class column: error_text is the message, and the class comes off the object
    with type(result).__name__. That name is the low-cardinality value a log or dashboard groups
    failures by, which is why langchaint.tracing sets it as the span's error.type; error_text is
    prose for a human to read.
    """
    for index, result in enumerate(results):
        if isinstance(result, GenerationError):
            print(f"item {index} failed with {type(result).__name__}: {result.error_text}")


def failed_conversations(
    conversations: list[str | list[Message]],
    results: list[Response[str] | GenerationError],
) -> list[str | list[Message]]:
    """Return the input conversations whose slots failed, ready to resubmit once the cause is fixed.

    Order alignment is what lets the two lists zip: result i belongs to conversations[i], and
    strict=True holds langchaint to that, since a length mismatch would be a defect rather than a
    case to handle.
    What has to be fixed first is not in this list, and it is not always in the items either. A
    RetriesExhaustedError names nothing to change, a MaxCompletionTokensExceededError needs a larger
    max_completion_tokens via rebind, and an InvalidRequestError may need this item's own data
    changed or may need nothing in the item at all.
    """
    return [
        conversation
        for conversation, result in zip(conversations, results, strict=True)
        if isinstance(result, GenerationError)
    ]


async def main() -> None:
    """Run every snippet in this file."""
    results = await run_batch(_CONVERSATIONS)
    log_failures(results)
    to_resubmit = failed_conversations(_CONVERSATIONS, results)
    print(f"{len(to_resubmit)} of {len(results)} items to resubmit")


if __name__ == "__main__":
    asyncio.run(main())
