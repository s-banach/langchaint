"""Rate limiting and errors: one shared RateLimiter, and success-or-raise generation.

One RateLimiter owns retrying (max_attempts, backoff_base_seconds, backoff_max_seconds) and pacing (max_in_flight).
It is stateful and shareable: pass one instance to every LLM hitting the same account and they share one budget,
so a rate-limit error pauses admission for all of them until a request succeeds again.
generate_one returns a Response on success and raises a GenerationError leaf on a terminal per-item outcome;
AbortBatchError is separate and means the request is misconfigured, so retrying cannot help.

The LangChain map (RunnableRetry, InMemoryRateLimiter, with_fallbacks) lives in MIGRATING_FROM_LANGCHAIN.md.
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from langchaint import AbortBatchError, BoundLLM, GenerationError, RateLimiter, Response, to_row
from langchaint.anthropic import anthropic_model
from langchaint.openai import openai_model


async def shared_rate_limiter() -> None:
    """Pass one RateLimiter to two models so they share one account budget.

    max_in_flight bounds concurrent requests; it defaults to 8 and self-adjusts throughput along request duration,
    which is why there is deliberately no requests_per_minute parameter that would go stale with the account tier.
    A 429 from either model trips the shared limiter and pauses admission for both until a request succeeds again.
    """
    limiter = RateLimiter(max_attempts=5, max_in_flight=16)
    openai_llm = openai_model("gpt-5.6-terra", rate_limiter=limiter)
    anthropic_llm = anthropic_model("claude-sonnet-5", rate_limiter=limiter)
    for llm in (openai_llm, anthropic_llm):
        assistant = llm.bind(system_prompt="Answer in one word.", automatic_prompt_caching=False)
        response = await assistant.generate_one("Name a primary color.")
        print(response.model, response.output)


class Sentiment(BaseModel):
    """A structured output shape, so a refusal or truncation raises rather than returning bad data."""

    label: Literal["positive", "negative"]


async def catch_generation_error() -> None:
    """Catch the GenerationError base to handle every terminal per-item outcome at once.

    On the structured path a refusal raises RefusalError and a token-cap truncation raises
    ExceededMaxCompletionTokensError; a spent transient budget raises RetriesExhaustedError on any path.
    to_row renders the caught error to the same row shape a Response fills, so a failure logs beside successes.
    """
    classifier = openai_model("gpt-5.6-terra").bind(
        system_prompt="Classify the sentiment as positive or negative.",
        response_format=Sentiment,
        automatic_prompt_caching=False,
    )
    try:
        response = await classifier.generate_one("This is the best day in months.")
        print("ok:", response.output.label)
    except GenerationError as err:
        row = to_row(err)
        print("failed:", type(err).__name__, "|", row["error_text"])


def report_billing(response: Response[str]) -> None:
    """Show the two usage scopes and recover each attempt's raw provider usage.

    response.usage is the paid total across every attempt (the number to bill on), and its cost_in_usd is
    the money the call spent; response.usage_successful_attempt is the single kept answer's own usage.
    The two diverge only when a billed 200 was retried (an empty structured parse retried as transient):
    a call whose only retries were transport, 5xx, or rate-limit failures bills nothing on them, so the two
    are equal. attempt_records holds one record per request sent; usage_raw is the provider's own usage
    object (None for a transport failure), so provider-specific detail stays recoverable after any outcome.
    """
    paid_total = response.usage
    kept_answer = response.usage_successful_attempt
    print(f"paid total: {paid_total.cost_in_usd:.6f} USD across {response.attempts} attempt(s)")
    print(f"kept answer: {kept_answer.cost_in_usd:.6f} USD (equal unless a billed 200 was retried)")
    for index, record in enumerate(response.attempt_records):
        print(f"  attempt {index + 1}: usage_raw={'present' if record.usage_raw is not None else 'none'}")


async def generate_with_fallback(
    primary: BoundLLM[str], secondary: BoundLLM[str], prompt: str
) -> Response[str]:
    """Try the primary binding; on any terminal failure, fall back to the secondary.

    A fallback is app code because the app decides what counts as worth failing over.
    GenerationError covers the per-item terminal leaves;
    AbortBatchError covers a misconfigured request retry cannot fix.
    """
    try:
        return await primary.generate_one(prompt)
    except (GenerationError, AbortBatchError):
        return await secondary.generate_one(prompt)


async def main() -> None:
    """Run every snippet in this file."""
    await shared_rate_limiter()
    await catch_generation_error()
    primary = openai_model("gpt-5.6-terra").bind(
        system_prompt="Answer in one sentence.",
        automatic_prompt_caching=False,
    )
    secondary = anthropic_model("claude-sonnet-5").bind(
        system_prompt="Answer in one sentence.",
        automatic_prompt_caching=False,
    )
    response = await generate_with_fallback(primary, secondary, "What is the capital of France?")
    print(response.output)
    report_billing(response)


if __name__ == "__main__":
    asyncio.run(main())
