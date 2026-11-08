"""Warm a reusable prompt prefix with the first batch input before starting the remaining inputs."""

from langchaint import PlainGeneration, TextPart
from langchaint.anthropic import Anthropic


async def generate_with_a_warm_cache() -> list[PlainGeneration[str]]:
    """Warm one prefix and print each outcome's cache usage."""
    stable_policy = (
        "Support policy: route refunds to a human. "
        "Never request a password or payment-card number. "
    ) * 300
    bound = Anthropic().llm(
        "claude-sonnet-5",
        cache_ttl="1h",
        system_prompt=[TextPart(text=stable_policy, cache_breakpoint=True)],
        max_completion_tokens=1024,
        automatic_cache_breakpoints=False,
    )

    prompts = [
        "Can I return an opened item?",
        "How do I request a refund?",
        "Which details may a support request include?",
    ]
    outcomes = await bound.generate_many(prompts, warm_cache=True)
    for index, outcome in enumerate(outcomes):
        usage = outcome.usage
        print(f"item {index} wrote {usage.input_tokens_cache_write} cache tokens")
        print(f"item {index} read {usage.input_tokens_cache_read} cache tokens")
    return outcomes
