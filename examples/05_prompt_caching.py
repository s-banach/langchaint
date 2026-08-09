"""Prompt caching: mark where the reusable prefix ends, in a binding and in a tool result."""

from langchaint import (
    AssistantMessage,
    Message,
    Response,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.anthropic import AnthropicAccount


async def cache_a_long_prefix() -> Response[str]:
    """Bind a marked prefix, send a marked tool result, and print what the cache read and wrote.

    Raises:
        Exception: An owned resource close operation fails.
        GenerationError: any terminal outcome of the generate call.
    """
    async with AnthropicAccount() as anthropic:
        # cache_ttl "1h" holds a cache entry across longer gaps than the default "5m".
        # Its writes cost more, so "1h" pays off when reuses arrive over five minutes apart.
        llm = anthropic.model("claude-sonnet-5", cache_ttl="1h")

        # Cache the stable instructions, and not the promoted-product line after them.
        # Pad a prefix you expect to cache: a provider caches only above a minimum token count.
        bound = llm.bind(
            system_prompt=[
                TextPart(
                    text="You are a support assistant. Route anything about refunds to a human. ",
                    cache_breakpoint=True,
                ),
                TextPart(text="Today's promoted product is the X-200."),
            ],
            automatic_prompt_caching=False,
        )

        messages: list[Message] = [
            UserMessage(content="Do you handle refunds?"),
            AssistantMessage(turn=[ToolCall(id="call_1", name="lookup_policy", args_json="{}")]),
            # cache_breakpoint on a tool result is honored only on the message's last part.
            ToolMessage(
                tool_call_id="call_1",
                content=[
                    TextPart(text="Retrieved at 09:00."),
                    TextPart(text="Refund policy document.", cache_breakpoint=True),
                ],
            ),
        ]
        response = await bound.generate_one(messages)
        usage = response.usage
        read_cost = usage.input_tokens_cache_read_cost_in_usd
        write_cost = usage.input_tokens_cache_write_cost_in_usd
        print(f"cache read {usage.input_tokens_cache_read} tokens for {read_cost} USD")
        print(f"cache wrote {usage.input_tokens_cache_write} tokens for {write_cost} USD")
        return response
