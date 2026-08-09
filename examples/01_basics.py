"""Construct an account, build a model, bind a prompt prefix, and generate."""

from typing import Literal

from pydantic import BaseModel

from langchaint import ImagePart, InferenceParams, Message, TextPart, UserMessage, to_tables
from langchaint.openai import OpenAIAccount


# bind(response_format=...) sends this docstring to the provider as the schema description.
class Sentiment(BaseModel):
    """The sentiment of one text, and how confident you are."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float


async def basics() -> None:
    """Bind a prefix, generate text, parse structured output, rebind, then run a batch.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable during account construction.
        Exception: An owned resource close operation fails.
        GenerationError: any terminal outcome of a generate_one call.
    """
    async with OpenAIAccount() as openai:
        llm = openai.model("gpt-5.6-terra")

        assistant = llm.bind(system_prompt="Be terse.", automatic_prompt_caching=False)

        colors = await assistant.generate_one("Name three primary colors.")
        print(f"answer: {colors.output}")
        print(f"paid: {colors.usage.cost_in_usd} USD")

        image_question: list[Message] = [
            UserMessage(
                content=[
                    TextPart(text="What color is this image?"),
                    ImagePart(data=b"<png bytes>", media_type="image/png"),
                ]
            )
        ]
        image_answer = await assistant.generate_one(image_question)
        print(f"about the image: {image_answer.output}")

        classifier = llm.bind(response_format=Sentiment, automatic_prompt_caching=False)
        sentiment = (await classifier.generate_one("Best day I have had in months.")).output
        print(f"{sentiment.label} at {sentiment.confidence} confidence")

        # inference_params replaces the bound one whole: a field left out of it is None.
        longer = assistant.rebind(inference_params=InferenceParams(max_completion_tokens=2048))
        bridge = await longer.generate_one("How does a suspension bridge carry load?")
        print(f"longer answer: {bridge.output}")

        results = await assistant.generate_many(["Define entropy.", "Define enthalpy."])
        calls, attempts = to_tables(results)
        print(f"{len(calls)} calls over {len(attempts)} attempts")
