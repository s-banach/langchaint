"""Batch error postures: every terminal per-item outcome is a row, and what an app does with each.

generate_many never raises on an item: each conversation settles into its own slot, as a Response or
as a GenerationError, and the siblings run to completion whatever any one item does.
So a batch's result list is always order-aligned and complete, and to_row renders successes and
failures to the same keys.
What the app still has to decide is the posture: which failures are worth re-running and which need
the request changed. That is a per-leaf question, which is why the leaves are separate classes.

Needs the anthropic package and ANTHROPIC_API_KEY: two of the three items are real calls, and the
third is rejected before anything is sent, so it costs nothing.
"""

import asyncio

from langchaint import (
    AssistantMessage,
    GenerationError,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    Message,
    RefusalError,
    Response,
    RetriesExhaustedError,
    TextPart,
    ToolCall,
    ToolMessage,
    UnrecognizedError,
    UserMessage,
    to_row,
)
from langchaint.anthropic import anthropic_model

# A ToolMessage whose cache_breakpoint sits on a part other than the last: the anthropic adapter
# refuses to send it, because the marker goes on the enclosing tool_result block, whose span ends at
# the last part, so the wire form would move the cache boundary the message asked for.
_REJECTED_CONVERSATION: list[Message] = [
    UserMessage(content="Do you handle refunds?"),
    AssistantMessage(turn=[ToolCall(id="call_1", name="lookup_policy", args_json="{}")]),
    ToolMessage(
        tool_call_id="call_1",
        content=[
            TextPart(text="Refund policy document.", cache_breakpoint=True),
            TextPart(text="Retrieved at 09:00."),
        ],
    ),
]


def posture(failure: GenerationError) -> str:
    """Name what an app does with one failure, per leaf.

    Three postures cover the five leaves: re-run the same item later, send a different request, or
    look into it. Matching on the leaf class is the whole discrimination; nothing here reads
    error_text, which is a message for a human and not a control-flow value.
    """
    match failure:
        case RetriesExhaustedError():
            return "re-run later: the transient budget ran out"
        case InvalidRequestError():
            return "send a different request: the provider or the adapter rejected this one"
        case RefusalError() | MaxCompletionTokensExceededError():
            return "send a different request: the structured path produced no instance"
        case UnrecognizedError():
            return "look into it: langchaint could not name this error"
        case _:
            return "look into it: an unfamiliar GenerationError leaf"


async def mixed_batch_to_rows() -> list[Response[str] | GenerationError]:
    """Run a batch whose middle item is rejected, and render every slot to a row.

    The rejection reaches its own slot as an InvalidRequestError while the two real calls complete,
    so the returned list is three long and to_row gives all three the same keys:
    a failure row nulls output and carries error_text.
    """
    summarizer = anthropic_model("claude-sonnet-5").bind(
        system_prompt="Summarize in five words.",
        automatic_prompt_caching=False,
    )
    conversations: list[str | list[Message]] = [
        "The quarterly report shows revenue up twelve percent on strong subscription growth.",
        _REJECTED_CONVERSATION,
        "The new compiler release cuts build times roughly in half on large projects.",
    ]
    results = await summarizer.generate_many(conversations)
    for result in results:
        row = to_row(result)
        print(f"{row['output'] or '(no output)'} | {row['error_text'] or 'ok'}")
    return results


def indexes_worth_rerunning(results: list[Response[str] | GenerationError]) -> list[int]:
    """Split the failures by posture and return the item indexes worth re-running unchanged.

    Only RetriesExhaustedError qualifies: the request itself was acceptable and the budget ran out,
    so the same conversation may succeed on a later pass. Every other leaf would fail the same way.
    """
    retryable: list[int] = []
    for index, result in enumerate(results):
        if not isinstance(result, GenerationError):
            continue
        print(f"item {index}: {type(result).__name__} -> {posture(result)}")
        if isinstance(result, RetriesExhaustedError):
            retryable.append(index)
    return retryable


async def main() -> None:
    """Run every snippet in this file."""
    results = await mixed_batch_to_rows()
    retryable = indexes_worth_rerunning(results)
    print("worth re-running unchanged:", retryable)


if __name__ == "__main__":
    asyncio.run(main())
