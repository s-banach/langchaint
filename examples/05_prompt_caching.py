"""Warm a reusable prompt prefix before running its batch siblings."""

from langchaint import GenerationError, Response, TextPart
from langchaint.anthropic import Anthropic


async def generate_with_a_warm_cache() -> list[Response[str] | GenerationError]:
    """Warm one prefix and print each result's cache usage."""
    anthropic = Anthropic()
    stable_policy = (
        "Support policy: route refunds to a human. "
        "Never request a password or payment-card number. "
    ) * 300
    bound = anthropic.model("claude-sonnet-5", cache_ttl="1h").bind(
        system_prompt=[TextPart(text=stable_policy, cache_breakpoint=True)],
        automatic_cache_breakpoints=False,
    )

    prompts = [
        "Can I return an opened item?",
        "How do I request a refund?",
        "Which details may a support request include?",
    ]
    results = await bound.generate_many(prompts, warm_cache=True)
    for index, result in enumerate(results):
        usage = result.usage
        print(f"item {index} wrote {usage.input_tokens_cache_write} cache tokens")
        print(f"item {index} read {usage.input_tokens_cache_read} cache tokens")
    return results
