"""Demonstrate OpenAI generation results."""

from typing import Literal

from pydantic import BaseModel, TypeAdapter

from langchaint import CallResultRecord, GenerationError, InferenceParams, to_tables
from langchaint.openai import OpenAI


class Sentiment(BaseModel):
    """Classify one text and report confidence."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def basics() -> None:
    """Demonstrate text, structured, rebound, and batch generation.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        GenerationError: Generation fails.
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

    results = await assistant.generate_many(["Define entropy.", "Define enthalpy."])
    generation_error_count = sum(isinstance(result, GenerationError) for result in results)
    result_record_adapter = TypeAdapter(list[CallResultRecord[str]])
    result_records_json = result_record_adapter.dump_json([result.record for result in results])
    restored_result_records = result_record_adapter.validate_json(result_records_json)
    calls, attempts = to_tables(restored_result_records)
    print(f"{generation_error_count} generation errors")
    print(f"{len(calls)} calls over {len(attempts)} attempts")
