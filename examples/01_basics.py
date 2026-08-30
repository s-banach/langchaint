"""Demonstrate OpenAI generation results."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from langchaint import InferenceParams, to_tables
from langchaint.openai import OpenAI


class Sentiment(BaseModel):
    """Classify one text and report confidence."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def basics() -> None:
    """Demonstrate text, structured, rebound, and batch generation.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: A `generate_one` call fails.
    """
    openai = OpenAI()
    llm = openai.model("gpt-5.6-terra")

    assistant = llm.bind(system_prompt="Be terse.")
    colors = await assistant.generate_one("Name three primary colors.")
    print(f"answer: {colors.output}")
    print(f"model: {colors.call.model}")
    print(f"provider: {colors.call.provider_name}")
    print(f"attempts: {len(colors.call.attempt_records)}")

    classifier = llm.bind(response_format=Sentiment)
    classification = await classifier.generate_one("Best day I have had in months.")
    print(f"{classification.output.label}: {classification.output.confidence}")

    detailed = assistant.rebind(
        system_prompt="Explain the answer in one paragraph.",
        inference_params=InferenceParams(max_completion_tokens=2048),
    )
    bridge = await detailed.generate_one("How does a suspension bridge carry load?")
    print(bridge.output)

    result_records = await assistant.generate_many_records(
        ["Define entropy.", "Define enthalpy."],
        resume_path=Path("definition-records.json"),
    )
    calls, attempts = to_tables(result_records)
    print(f"{len(calls)} calls over {len(attempts)} attempts")
