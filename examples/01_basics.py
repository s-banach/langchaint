"""Construct a model, bind it, and generate: text, structured output, rebind, and a batch.

The whole surface is: pick a model function, freeze the prompt prefix with bind, then call a generate method.
There are no generate methods on the un-bound LLM and no per-call parameter overrides; changing a parameter is rebind.

The LangChain call-for-call map lives in MIGRATING_FROM_LANGCHAIN.md; this file shows the langchaint calls themselves.
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from langchaint import InferenceParams, to_row
from langchaint.openai import openai_model


async def plain_text() -> None:
    """Generate one text turn.

    openai_model reads OPENAI_API_KEY from the environment and returns an LLM.
    Swapping providers is one import: anthropic_model("claude-sonnet-5") returns an LLM with the same surface.
    bind() with no response_format returns BoundLLM[str], so response.output is the assistant text.
    A bare str is shorthand for a conversation of one user turn holding that text.
    """
    llm = openai_model("gpt-5.6-terra")
    assistant = llm.bind(system_prompt="You are a terse assistant.")
    response = await assistant.generate_one("Name three primary colors.")
    print(response.output)
    print(f"{response.cost_in_usd:.6f} USD, {response.usage.output_tokens} output tokens")


class Sentiment(BaseModel):
    """The structured output shape; the model's JSON is parsed into this."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def structured_output() -> None:
    """Fix the output type to a model with bind(response_format=Model).

    The overload makes this a BoundLLM[Sentiment], so response.output is a Sentiment instance, already validated.
    A refusal or a truncation on this structured path raises a GenerationError leaf rather than returning bad data;
    see 05_rate_limiting_and_errors.py for catching those.
    """
    llm = openai_model("gpt-5.6-terra")
    classifier = llm.bind(
        system_prompt="Classify the sentiment of the user's message.",
        response_format=Sentiment,
    )
    response = await classifier.generate_one("This is the best day I have had in months.")
    sentiment = response.output
    print(sentiment.label, sentiment.confidence)


async def rebind_to_change_a_parameter() -> None:
    """Change a bound parameter with rebind, which returns a new BoundLLM; the original is unchanged.

    rebind carries the same overloads as bind, so rebind(response_format=None) switches the output type back to str.
    A left-out field keeps its current value; inference_params is replaced whole, never merged field-wise.
    """
    llm = openai_model("gpt-5.6-terra")
    base = llm.bind(
        system_prompt="Answer in one sentence.",
        inference_params=InferenceParams(max_completion_tokens=256),
    )
    longer = base.rebind(inference_params=InferenceParams(max_completion_tokens=2048))
    response = await longer.generate_one("Explain how a suspension bridge carries load.")
    print(response.output)


async def batch_to_rows() -> None:
    """Run an order-aligned batch with generate_many; flatten every result to one table shape with to_row.

    A terminal per-item failure comes back as a GenerationError in its slot rather than raising, so the batch finishes;
    to_row renders a success and a failure to the same keys, so the mixed list is one table.
    Concurrency is bounded by the shared RateLimiter's max_in_flight (see 05_rate_limiting_and_errors.py).
    """
    llm = openai_model("gpt-5.6-terra")
    summarizer = llm.bind(system_prompt="Summarize in five words.")
    documents = [
        "The quarterly report shows revenue up twelve percent on strong subscription growth.",
        "Heavy rain closed three mountain passes and stranded weekend hikers overnight.",
        "The new compiler release cuts build times roughly in half on large projects.",
    ]
    results = await summarizer.generate_many(documents)
    rows = [to_row(result) for result in results]
    for row in rows:
        print(row["output"], "|", row["cost_in_usd"])
    # rows is a list of flat dicts; pandas.DataFrame(rows) turns it into a dataframe for eval logging.


async def main() -> None:
    """Run every snippet in this file."""
    await plain_text()
    await structured_output()
    await rebind_to_change_a_parameter()
    await batch_to_rows()


if __name__ == "__main__":
    asyncio.run(main())
