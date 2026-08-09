"""What goes wrong and what bounds it: per-account SharedBackoff domains, terminal errors, and deadlines."""

from langchaint import (
    GenerationError,
    GenerationInput,
    ImagePart,
    Message,
    Response,
    TextPart,
    UserMessage,
)
from langchaint.anthropic import AnthropicAccount
from langchaint.openai import OpenAIAccount


async def run_batch_and_handle_what_failed() -> list[Response[str] | GenerationError]:
    """Run a batch under a deadline, then send every failed item to a second provider.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable during account construction.
        Exception: An owned resource close operation fails.
        GenerationError: a fallback call failed too.
    """
    async with (
        AnthropicAccount(
            max_concurrent_requests=16,
            max_request_starts_per_second=5,
        ) as anthropic,
        OpenAIAccount() as openai,
    ):
        summarizer = anthropic.model("claude-sonnet-5").bind(
            system_prompt="Summarize in one sentence.",
            max_attempts=5,
            automatic_prompt_caching=False,
        )
        fallback = openai.model("gpt-5.6-terra").bind(
            system_prompt="Summarize in one sentence.", automatic_prompt_caching=False
        )

        # This media_type is invalid for anthropic.
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

        # The item clock stops during admission waits.
        results = await summarizer.generate_many(documents, max_working_seconds_per_item=30)

        for index, result in enumerate(results):
            if not isinstance(result, GenerationError):
                continue

            print(f"item {index} failed with {type(result).__name__}: {result.error_text}")
            print(f"item {index} billed {result.usage.cost_in_usd} USD before failing")

            # generate_one raises on error.
            results[index] = await fallback.generate_one(documents[index], timeout_seconds=30)
        return results
