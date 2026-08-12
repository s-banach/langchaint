"""Construct `OpenAI`, bind models, and inspect generation results."""

from typing import Literal

from pydantic import BaseModel

from langchaint import InferenceParams, Response, to_tables
from langchaint.openai import OpenAI


class Sentiment(BaseModel):
    """Classify one text and report confidence."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def basics() -> None:
    """Generate text, parse structured output, rebind, and run a batch.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: A `generate_one` call failed.
    """
    openai = OpenAI()
    llm = openai.model("gpt-5.6-terra")

    assistant = llm.bind(system_prompt="Be terse.", automatic_prompt_caching=False)
    colors = await assistant.generate_one("Name three primary colors.")
    print(f"answer: {colors.output}")
    print(f"model: {colors.call.model}")
    print(f"provider: {colors.call.provider_name}")
    print(f"attempts: {len(colors.call.attempt_records)}")

    classifier = llm.bind(response_format=Sentiment, automatic_prompt_caching=False)
    classification = await classifier.generate_one("Best day I have had in months.")
    print(f"{classification.output.label}: {classification.output.confidence}")

    detailed = assistant.rebind(
        system_prompt="Explain the answer in one paragraph.",
        inference_params=InferenceParams(max_completion_tokens=2048),
    )
    bridge = await detailed.generate_one("How does a suspension bridge carry load?")
    print(bridge.output)

    results = await assistant.generate_many(["Define entropy.", "Define enthalpy."])
    successful_count = sum(isinstance(result, Response) for result in results)
    calls, attempts = to_tables(results)
    print(f"{successful_count} successful results")
    print(f"{len(calls)} calls over {len(attempts)} attempts")
