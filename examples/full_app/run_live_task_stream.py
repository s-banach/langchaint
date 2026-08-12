"""Run the progress-event app with one long-lived OpenAI."""

import asyncio

from config import build_configs
from events import Event
from opentelemetry import trace
from render import render
from task_stream import App

from langchaint.openai import OpenAI

MODEL = "gpt-5.6-terra"
APP_TIMEOUT_SECONDS = 120.0


def print_event(event: Event) -> None:
    """Print one progress event."""
    print(render(event))


async def main() -> None:
    """Run one live app and print its final answer.

    Raises:
        openai.OpenAIError: OpenAI credentials are unavailable.
        TimeoutError: APP_TIMEOUT_SECONDS expires.
        ExceptionGroup: a concurrent tool function raises.
        asyncio.CancelledError: the caller cancels the app.
    """
    openai = OpenAI()
    app = App(
        llm=openai.model(MODEL, regional_processing=False),
        configs=build_configs(),
        tracer=trace.get_tracer("examples.full_app.live"),
        on_event=print_event,
        capture_message_content=False,
    )
    async with asyncio.timeout(APP_TIMEOUT_SECONDS):
        await app.run()
    print(f"final answer: {app.final_answer!r}")


if __name__ == "__main__":
    asyncio.run(main())
