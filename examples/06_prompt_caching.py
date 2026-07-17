"""Prompt caching: the neutral controls and the provider-specific thorns behind them.

Caching is always user-stated because it changes billing: bind requires automatic_prompt_caching with no default,
and cache_breakpoint on a TextPart/ImagePart marks the exact end of a reusable prompt prefix.
Marks are honored under either automatic_prompt_caching value; bound False with no marks caches nothing.
The same two controls hit different wire mechanics per provider, and those thorns are this file's subject:
anthropic's 4-marker request limit and cache_ttl, openai's implicit/explicit modes and model-version cutoff,
and the two places a mark is rejected because the providers diverge.

Running this file needs langchaint[anthropic] with ANTHROPIC_API_KEY and langchaint[openai] with OPENAI_API_KEY.
"""

import asyncio

from pydantic import ValidationError

from langchaint import (
    AbortBatchError,
    AssistantMessage,
    Message,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.anthropic import anthropic_model
from langchaint.openai import openai_model

# Padded by repetition because each provider caches only a prefix above a minimum token count;
# a realistic system prompt this size caches, a two-line demo prompt does not.
STABLE_INSTRUCTIONS = (
    "You are a support assistant for a fictional product. Answer briefly. "
    "Policy line for padding: route anything involving refunds to a human. "
) * 60


async def marks_inside_the_frozen_prefix() -> None:
    """Place a cache_breakpoint inside the frozen prefix by binding system_prompt as parts.

    The parts form exists for exactly this: stable instructions marked, semi-stable context after,
    so the stable span caches even when the context behind it changes per deployment.
    The wire forms diverge: anthropic renders one system block per part, cache_control on the marked block;
    openai renders one developer-role input message, prompt_cache_breakpoint on its marked parts,
    because openai's plain instructions parameter cannot carry marks.
    automatic_prompt_caching=False here shows that marks alone cache: the first call writes, the second reads.
    """
    stable_part = TextPart(text=STABLE_INSTRUCTIONS, cache_breakpoint=True)
    context_part = TextPart(text="Today's promoted product is the X-200.")
    llm = anthropic_model("claude-sonnet-5")
    assistant = llm.bind(
        system_prompt=[stable_part, context_part],
        automatic_prompt_caching=False,
    )
    for label in ("first call", "second call"):
        response = await assistant.generate_one("Do you handle refunds?")
        usage = response.usage
        print(
            f"{label}: cache_read={usage.input_tokens_cache_read}",
            f"cache_write={usage.input_tokens_cache_write}",
            f"cache_none={usage.input_tokens_cache_none}",
        )


def anthropic_marker_budget() -> None:
    """Anthropic allows at most 4 cache_control markers per request; the binding's markers spend slots first.

    With automatic_prompt_caching=True the binding marks the end of the frozen prefix and the last message
    of every request, and each marked system part is one more marker.
    A binding whose own markers exceed the limit raises ValueError at bind, before any request.
    Whatever the binding leaves is the per-request budget for marked message parts:
    the latest marks up to that budget are written, older marks are left unwritten.
    openai needs no budget arithmetic: every mark is sent, and the server matches on the latest three
    (implicit mode) or four (explicit mode), older breakpoints staying usable read-only.
    Both behaviors let a conversation that accrues one mark per turn keep working as it grows.
    """
    four_marked_parts = [
        TextPart(text=f"Instruction section {index}.", cache_breakpoint=True) for index in range(4)
    ]
    llm = anthropic_model("claude-sonnet-5")
    try:
        llm.bind(system_prompt=four_marked_parts, automatic_prompt_caching=True)
    except ValueError as err:
        print(f"bind rejected the marker budget: {err}")


async def openai_explicit_mode() -> None:
    """On openai, bound True sends nothing, leaving the server's default implicit mode in place.

    In implicit mode (the default for gpt-5.6 and later) the server chooses one implicit cache breakpoint itself.
    Bound False, the adapter sends prompt_cache_options {"mode": "explicit"},
    which disables the implicit breakpoint so the server caches only at explicit breakpoints:
    False with no marks disables caching, and a marked part re-enables it at exactly that boundary.
    prompt_cache_options exists on gpt-5.6 and later, so binding False with an older model may be rejected
    by the API; bound True works on any model because nothing extra is sent.
    There is also no TTL to choose: "30m" is openai's only value, so openai_model has no cache_ttl parameter.
    """
    marked_context = TextPart(text=STABLE_INSTRUCTIONS, cache_breakpoint=True)
    question = TextPart(text="Do you handle refunds?")
    assistant = openai_model("gpt-5.6-terra").bind(automatic_prompt_caching=False)
    for label in ("first call", "second call"):
        response = await assistant.generate_one([UserMessage([marked_context, question])])
        usage = response.usage
        print(
            f"{label}: cache_read={usage.input_tokens_cache_read}",
            f"cache_write={usage.input_tokens_cache_write}",
        )


async def anthropic_one_hour_ttl() -> None:
    """cache_ttl on anthropic_model sets the TTL of every marker the adapter writes, uniformly.

    The default "5m" omits the ttl key (the API default), and its writes bill 1.25x base input;
    "1h" holds entries across longer gaps and its writes bill 2x,
    priced by the PricingTable's cache_write_1h_usd_per_million_tokens.
    A per-part TTL is deliberately not exposed; mixing TTLs is two bindings on two adapters.
    """
    llm = anthropic_model("claude-sonnet-5", cache_ttl="1h")
    assistant = llm.bind(system_prompt=STABLE_INSTRUCTIONS, automatic_prompt_caching=True)
    response = await assistant.generate_one("Do you handle refunds?")
    usage = response.usage
    print(
        f"1h write: cache_write={usage.input_tokens_cache_write}, "
        f"reasoning={usage.output_tokens_reasoning} tokens, {usage.cost_in_usd:.6f} USD"
    )


async def provider_divergent_marks_are_rejected() -> None:
    """Two places a mark is rejected instead of silently degrading.

    AssistantMessage rejects a marked TextPart in its turn at validation:
    openai's assistant replay text param carries no prompt_cache_breakpoint key,
    so the mark would cache on one provider and silently vanish on the other.
    And a marked part in a ToolMessage must be the message's last part:
    anthropic's marker goes on the enclosing tool_result block, whose span ends at the last part,
    so a mark anywhere else raises AbortBatchError instead of moving the cache boundary.
    Both raises happen before any request is sent, so this function costs nothing to run.
    """
    try:
        AssistantMessage([TextPart(text="Reply text.", cache_breakpoint=True)])
    except ValidationError as err:
        print(f"AssistantMessage rejected the marked turn part: {err.errors()[0]['msg']}")
    tool_call = ToolCall(id="call_1", name="lookup_policy", args_json="{}")
    conversation: list[Message] = [
        UserMessage("Do you handle refunds?"),
        AssistantMessage([tool_call]),
        ToolMessage(
            tool_call_id="call_1",
            content=[
                TextPart(text="Refund policy document.", cache_breakpoint=True),
                TextPart(text="Retrieved at 09:00."),
            ],
        ),
    ]
    bound = anthropic_model("claude-sonnet-5").bind(automatic_prompt_caching=False)
    try:
        await bound.generate_one(conversation)
    except AbortBatchError as err:
        print(f"generate_one rejected the non-last marked tool part: {err}")


async def main() -> None:
    """Run every snippet in this file."""
    await marks_inside_the_frozen_prefix()
    anthropic_marker_budget()
    await openai_explicit_mode()
    await anthropic_one_hour_ttl()
    await provider_divergent_marks_are_rejected()


if __name__ == "__main__":
    asyncio.run(main())
