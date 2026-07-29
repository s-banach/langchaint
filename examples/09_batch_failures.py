"""What an app does after a batch: read every slot, total the spend, and log the failures.

generate_many never raises on an item. Each generation_input settles into its own slot as a Response or
as a GenerationError, the siblings run to completion whatever any one item does, and result i belongs
to generation_inputs[i]. So the call itself needs no try/except; the work is the loop over the results,
which to_tables renders to a calls table and an attempts table whether a slot succeeded or failed.

Each GenerationError subclass names what happened, not what to do next, and InvalidRequestError is where that gap is
widest: a rejected GenerationInput, a bad API key, a revoked permission, and an unknown model id all
land there, so after fixing a key the same items are worth resubmitting unchanged. langchaint does
not guess which it was, because a provider states a status and never a cause. So log the class name
and let whoever reads the log decide.

Needs the anthropic package and ANTHROPIC_API_KEY: two of the three items are real calls, and the
remaining one is never sent, so it costs nothing.
"""

import asyncio

from langchaint import (
    CallResult,
    GenerationError,
    GenerationInput,
    ImagePart,
    Message,
    TextPart,
    Usage,
    UserMessage,
    to_tables,
)
from langchaint.anthropic import anthropic_model

# One row of the batch carries a scanned page as a TIFF. The adapter sends only image/gif,
# image/jpeg, image/png, and image/webp, so build_request returns InvalidRequest and the item
# fails its own row. That happens before the bytes are base64-encoded, which is why the placeholder
# below is enough to trigger it.
_SCANNED_PAGE: list[Message] = [
    UserMessage(
        content=[
            TextPart(text="Summarize the attached page."),
            ImagePart(data=b"<scan bytes>", media_type="image/tiff"),
        ]
    )
]

_GENERATION_INPUTS: list[GenerationInput] = [
    "The quarterly report shows revenue up twelve percent on strong subscription growth.",
    _SCANNED_PAGE,
    "The new compiler release cuts build times roughly in half on large projects.",
]


async def run_batch(
    generation_inputs: list[GenerationInput],
) -> list[CallResult[str]]:
    """Run the batch, print every slot as a row, and print what the whole batch paid.

    Nothing here catches anything: the generation_input carrying the TIFF settles into its own slot as an
    InvalidRequestError while the two real calls complete, so the list is as long as the input and
    to_tables fills the same columns for every slot. A success leaves error_summary None, a failure
    leaves output None, and the unsent item's attempts is 0 because no request went out.
    Usage.sum_of totals the spend over successes and failures alike, since every GenerationError
    carries the usage its attempts billed.
    """
    summarizer = anthropic_model("claude-sonnet-5").bind(
        system_prompt="Summarize in five words.",
        automatic_prompt_caching=False,
    )
    results = await summarizer.generate_many(generation_inputs)
    calls, _ = to_tables(results)
    for row in calls:
        print(
            f"item {row['call_id']}: output={row['output']!r} "
            f"error_summary={row['error_summary']!r} attempts={row['attempts']}"
        )
    total = Usage.sum_of(result.usage for result in results)
    print(f"batch paid {total.cost_in_usd:.6f} USD")
    return results


def log_failures(results: list[CallResult[str]]) -> None:
    """Log each failed item's class name and error_text.

    Neither table has an error-class column: the class comes off the object with
    type(result).__name__. That name is the low-cardinality value a log or dashboard groups failures
    by, which is why langchaint.tracing sets it as the span's error.type; error_text is prose for a
    human to read.
    """
    for index, result in enumerate(results):
        if isinstance(result, GenerationError):
            print(f"item {index} failed with {type(result).__name__}: {result.error_text}")


def failed_generation_inputs(
    generation_inputs: list[GenerationInput],
    results: list[CallResult[str]],
) -> list[GenerationInput]:
    """Return the input generation_inputs whose slots failed, ready to resubmit once the cause is fixed.

    Order alignment is what lets the two lists zip: result i belongs to generation_inputs[i], and
    strict=True holds langchaint to that, since a length mismatch would be a defect rather than a
    case to handle.
    What has to be fixed first is not in this list, and it is not always in the items either. A
    RetriesExhaustedError names nothing to change, a MaxCompletionTokensExceededError needs a larger
    max_completion_tokens via rebind, and an InvalidRequestError may need this item's own data
    changed or may need nothing in the item at all.
    """
    return [
        generation_input
        for generation_input, result in zip(generation_inputs, results, strict=True)
        if isinstance(result, GenerationError)
    ]


async def main() -> None:
    """Run every snippet in this file."""
    results = await run_batch(_GENERATION_INPUTS)
    log_failures(results)
    to_resubmit = failed_generation_inputs(_GENERATION_INPUTS, results)
    print(f"{len(to_resubmit)} of {len(results)} items to resubmit")


if __name__ == "__main__":
    asyncio.run(main())
