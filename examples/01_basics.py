"""Demonstrate OpenAI generations and their records."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from langchaint import to_tables
from langchaint.openai import OpenAI


class Sentiment(BaseModel):
    """Classify one text and report confidence."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def basics() -> None:
    """Demonstrate text generation, structured generation, `BoundLLM.bind()`, and batch generation.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: A `generate_one` call fails.
    """
    openai = OpenAI()
    llm = openai.llm("gpt-5.6-terra")

    assistant = llm.bind(system_prompt="Be terse.")
    colors = await assistant.generate_one("Name three primary colors.")
    print(f"answer: {colors.output}")
    print(f"model: {colors.request_history.model}")
    print(f"provider: {colors.request_history.provider_name}")
    print(f"requests: {colors.request_count}")

    classifier = llm.bind(response_format=Sentiment)
    classification = await classifier.generate_one("Best day I have had in months.")
    print(f"{classification.output.label}: {classification.output.confidence}")

    detailed = assistant.bind(
        system_prompt="Explain the answer in one paragraph.",
        max_completion_tokens=2048,
    )
    bridge = await detailed.generate_one("How does a suspension bridge carry load?")
    print(bridge.output)

    outcome_records = await assistant.generate_many_records(
        ["Define entropy.", "Define enthalpy."],
        resume_path=Path("definition-records.json"),
    )
    outcome_rows, request_rows = to_tables(outcome_records)
    print(f"{len(outcome_rows)} outcomes over {len(request_rows)} requests")
