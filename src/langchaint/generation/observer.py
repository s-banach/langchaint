"""Define the observer that follows generation calls and tool dispatches.

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
from langchaint.generation.call import AbandonedCallRecord
from langchaint.generation.response import CallResult
from langchaint.tools import DispatchObserver


@dataclass(frozen=True, kw_only=True)
class GenerationStart:
    """What langchaint knows when one generation call starts.

    `messages` is the call's input, with a bare `str` input as one `UserMessage`.
    `stream` is `True` for `stream_one`.
    """

    provider_name: str
    model: str
    binding: Binding
    response_format: type[object] | None
    messages: Sequence[Message]
    stream: bool


class Observer(DispatchObserver, Protocol):
    """Follows each generation call and tool dispatch."""

    def generation_started(
        self, start: GenerationStart
    ) -> ObservedOperation[CallResult[object] | AbandonedCallRecord]:
        """Start following one generation call.

        `generate_one` and each generated `generate_many` or `generate_many_records` item start one call.
        Their handle receives the result or `GenerationError`, and is current during the retry loop.
        `stream_one` starts a call when its handle is entered.
        Its handle receives the conclusion, or the `GenerationError` of an expired `timeout_seconds`.
        A block left before the conclusion gives the handle the stream's `abandoned` record.
        langchaint logs an exception this method raises and runs the call unobserved.
        """
        ...
