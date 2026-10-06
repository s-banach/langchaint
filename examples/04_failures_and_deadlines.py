"""Show request pacing, terminal errors, deadlines, and provider fallback."""

from langchaint import (
    GenerationError,
    GenerationInput,
    ImagePart,
    Message,
    PlainGeneration,
    SharedBackoff,
    TextPart,
    UserMessage,
)
from langchaint.anthropic import Anthropic
from langchaint.openai import OpenAI


async def run_batch_and_handle_what_failed() -> list[PlainGeneration[str] | GenerationError]:
    """Run a batch and send failed items to a second provider.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Fallback generation fails.
    """
    anthropic = Anthropic(
        shared_backoff=SharedBackoff(max_concurrent_requests=16, max_request_starts_per_second=5),
    )
    openai = OpenAI()
    summarizer = anthropic.llm("claude-sonnet-5").bind(
        system_prompt="Summarize in one sentence.",
        max_completion_tokens=256,
        max_requests=5,
    )
    fallback = openai.llm("gpt-5.6-terra").bind(system_prompt="Summarize in one sentence.")

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

    # Admission waits pause each input's clock.
    outcomes = await summarizer.generate_many(documents, max_working_seconds_per_input=30)

    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, GenerationError):
            continue

        print(f"item {index} failed with {type(outcome).__name__}: {outcome.error_text}")
        print(f"item {index} billed {outcome.usage.cost_in_usd} USD before failing")

        # generate_one raises when fallback fails.
        outcomes[index] = await fallback.generate_one(documents[index], timeout_seconds=30)
    return outcomes
