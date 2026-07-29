"""The shapes a deadline takes: one call, one item of a batch, a chain, a stream, and the app's own.

timeout_seconds is elapsed real time over the whole call: the slot wait inside the RateLimiter,
every attempt, and every backoff sleep between them. When it expires, langchaint raises
TimedOutError, a GenerationError like any other, so it carries usage, attempts, and attempt_records,
and to_tables renders it to the same failure row every other error gets. Nothing here catches it on
its own: an app that stops on a timeout stops on a refusal too, and both arrive as GenerationError.

Needs the anthropic package and ANTHROPIC_API_KEY: every call here is real. The deadlines are
generous enough that none of them normally expires.
"""

import asyncio
from time import monotonic

from langchaint import BoundLLM, GenerationError, Message, Response, to_tables
from langchaint.anthropic import anthropic_model

_CHAIN_BUDGET_SECONDS = 60.0
"""The seconds one request handler gets across all of its calls."""


async def one_call(summarizer: BoundLLM[str]) -> str | None:
    """Give one call thirty seconds and treat running out as the failure it is."""
    try:
        response = await summarizer.generate_one(
            "Summarize the release notes.", timeout_seconds=30
        )
    except GenerationError as error:
        print(
            f"{type(error).__name__}: paid {error.usage.cost_in_usd:.6f} USD over {error.attempts} attempts"
        )
        return None
    return response.output


async def one_batch(summarizer: BoundLLM[str], conversations: list[str | list[Message]]) -> None:
    """Give every item its own thirty seconds.

    The deadline starts when that item starts, not when the batch does, and generate_many raises
    nothing: an item that runs out settles into its own slot as a TimedOutError while the rest run
    on. A batch wider than max_in_flight can spend a queued item's whole deadline behind its
    siblings; that item reports attempts 0, which is how its row says it never sent.
    """
    results = await summarizer.generate_many(conversations, timeout_seconds=30)
    calls, _ = to_tables(results)
    for row in calls:
        print(
            f"item {row['call_id']}: error_summary={row['error_summary']!r} attempts={row['attempts']}"
        )


async def one_chain(summarizer: BoundLLM[str], documents: list[str]) -> list[str]:
    """Spend one budget across a sequence of calls by passing each its remaining share.

    langchaint has no budget spanning calls, because it cannot know which calls belong to one
    request: the app that opened the request does. So the app holds the deadline as an instant from
    monotonic() and subtracts, because a clock adjustment mid-request moves what time.time() reads
    and would lengthen or shorten what is left.
    A chain whose budget runs out raises TimedOutError from the call that had nothing left to give,
    and the summaries collected before it go with it. An app that wants to keep them returns them
    from the except clause instead.
    """
    deadline = monotonic() + _CHAIN_BUDGET_SECONDS
    summaries: list[str] = []
    for document in documents:
        response = await summarizer.generate_one(document, timeout_seconds=deadline - monotonic())
        summaries.append(response.output)
    return summaries


async def one_stream(summarizer: BoundLLM[str]) -> Response[str]:
    """Give the whole async with block thirty seconds, item pulls and final() included.

    The clock starts at entry, not at stream_one, so a handle built early and entered late loses
    none of it. An expiry raises TimedOutError out of the block, and it accounts for the call by
    itself: handle.abandoned stays None, because one cut-off call gets one account.
    """
    handle = summarizer.stream_one("Summarize the release notes.", timeout_seconds=30)
    async with handle:
        async for item in handle:
            if isinstance(item, str):
                print(item, end="", flush=True)
        return await handle.final()


async def a_scope_the_app_owns(summarizer: BoundLLM[str]) -> None:
    """Wrap the call in a scope of the app's own, and lose the account of what it spent.

    asyncio.timeout converts the cancellation it sent into a TimeoutError at its own boundary, which
    is above the frame holding the call's ledger, so by the time this except clause runs there is
    nothing left to read: the settled attempts died with the frame. A TimeoutError is not a
    GenerationError either, so it misses whatever handles the other failures.
    Everything this shape does, timeout_seconds does while keeping the account. Reach for a scope of
    your own to bound something wider than one call, and give the calls inside it their own
    timeout_seconds.
    """
    try:
        async with asyncio.timeout(30):
            await summarizer.generate_one("Summarize the release notes.")
    except TimeoutError:
        print("timed out, and this handler cannot say what it cost")


async def main() -> None:
    """Run every shape in this file against one binding."""
    summarizer = anthropic_model("claude-sonnet-5").bind(
        system_prompt="Summarize in five words.",
        automatic_prompt_caching=False,
    )
    await one_call(summarizer)
    await one_batch(summarizer, ["The first release note.", "The second release note."])
    print(await one_chain(summarizer, ["The first document.", "The second document."]))
    await one_stream(summarizer)
    await a_scope_the_app_owns(summarizer)


if __name__ == "__main__":
    asyncio.run(main())
