"""What goes wrong and what bounds it: per-account rate limiters, terminal errors, and deadlines."""

from langchaint import (
    CallResult,
    GenerationError,
    GenerationInput,
    ImagePart,
    Message,
    RateLimiter,
    TextPart,
    UserMessage,
)
from langchaint.anthropic import anthropic_model
from langchaint.openai import openai_model


async def run_batch_and_handle_what_failed() -> list[CallResult[str]]:
    """Run a batch under a deadline, then send every failed item to a second provider.

    Raises:
        GenerationError: a fallback call failed too.
    """
    # max_attempts counts requests sent including the first.
    # Only transient errors are retried.
    # Construct a RateLimiter explicitly to override the default settings:
    anthropic_limiter = RateLimiter(max_attempts=5, max_in_flight=16)

    summarizer = anthropic_model("claude-sonnet-5", rate_limiter=anthropic_limiter).bind(
        system_prompt="Summarize in one sentence.", automatic_prompt_caching=False
    )
    fallback = openai_model("gpt-5.6-terra").bind(
        system_prompt="Summarize in one sentence.", automatic_prompt_caching=False
    )

    # The following ImagePart uses a media_type anthropic does not accept, to make one item fail.
    # Its request raises InvalidRequestError in the anthropic adapter, before reaching the wire.
    scanned_page: list[Message] = [
        UserMessage(
            content=[
                TextPart(text="Summarize the attached page."),
                ImagePart(data=b"<scan bytes>", media_type="image/tiff"),
            ]
        )
    ]
    documents: list[GenerationInput] = [
        "Revenue rose twelve percent on strong subscription growth.",
        scanned_page,
        "The new compiler release cuts build times roughly in half.",
    ]

    # timeout_seconds is per-item wall time, starting when it is first attempted.
    # generate_many returns errors as results, rather than raising.
    results = await summarizer.generate_many(documents, timeout_seconds=30)

    for index, result in enumerate(results):
        if not isinstance(result, GenerationError):
            continue

        print(f"item {index} failed with {type(result).__name__}: {result.error_text}")
        print(f"item {index} billed {result.usage.cost_in_usd} USD before failing")

        # generate_one raises on error, unlike generate_many.
        results[index] = await fallback.generate_one(documents[index], timeout_seconds=30)
    return results
