"""Define the observer that follows every input and every tool dispatch.

A backend constructor passes one observer to every `LLM` it creates.
Each `LLM` passes it to its bindings, their streams, and each `ToolManager` that `bind` builds from a tool sequence.
`langchaint.tracing.OtelObserver` implements this protocol with OpenTelemetry spans.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchaint.adapter import Binding
from langchaint.common.messages import Message
from langchaint.common.observed_operation import ObservedOperation
from langchaint.generation.request_history import AbandonedStreamRecord
from langchaint.generation.response import GenerationOutcome
from langchaint.tools import DispatchObserver


@dataclass(frozen=True, kw_only=True)
class GenerationStart:
    """What langchaint knows when handling one input starts.

    `messages` is the input, with a bare `str` input as one `UserMessage`.
    `stream` is `True` for `stream_one`.
    """

    provider_name: str
    model: str
    binding: Binding
    response_format: type[object] | None
    messages: Sequence[Message]
    stream: bool


class Observer(DispatchObserver, Protocol):
    """Follows each input and each tool dispatch."""

    def generation_started(
        self, start: GenerationStart
    ) -> ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord]:
        """Start following one input.

        `generate_one` and each generated `generate_many` or `generate_many_records` item start one input.
        Their handle receives the `Generation` or `GenerationError`, and is current during the retry loop.
        `stream_one` starts one input when its handle is entered.
        Its handle receives the conclusion, or the `GenerationError` of an expired `timeout_seconds`.
        A block left before the conclusion gives the handle the stream's `abandoned` record.
        langchaint logs an exception this method raises and handles the input unobserved.
        """
        ...
