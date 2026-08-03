"""A scriptable offline adapter so the example runs with no network.

A script is a list of turns keyed by a tag the binding carries in its system prompt,
so one adapter serves every agent in the graph and each agent gets its own scripted turns.
Each turn is either text (ends that agent's loop) or tool calls (the loop dispatches and comes back).
delay_seconds on a turn makes its open suspend, which is how the timeout layers get exercised.
"""

import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, override

from pydantic import BaseModel

from langchaint import (
    LLM,
    AssistantMessage,
    Billing,
    DoNotRetry,
    Message,
    SharedBackoff,
    StreamItem,
    TextPart,
    ToolCall,
    TransientError,
    Usage,
    Verdict,
)
from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    BoundAdapter,
    ErrorClassification,
    RequestParams,
    verdict_from_transient_error,
)
from langchaint.call import ResponseIdentity


class FakeRaw(BaseModel):
    """Stands in for the SDK response model an adapter holds in raw.

    turn_index names the scripted turn this response came from, standing in for the fields a real
    adapter reads the turn and the counters off.
    """

    turn_index: int


def _turn_index(raw: BaseModel) -> int:
    """Narrow a raw response to the scripted one and return the turn it came from.

    Raises:
        TypeError: raw is not a FakeRaw.
    """
    if not isinstance(raw, FakeRaw):
        raise TypeError(f"expected a FakeRaw, got {type(raw).__name__}")
    return raw.turn_index


_TURN_USAGE = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=100,
    output_tokens=20,
    output_tokens_reasoning=0,
    input_tokens_cache_read_cost_in_usd=0.0,
    input_tokens_cache_write_cost_in_usd=0.0,
    input_tokens_cache_none_cost_in_usd=0.006,
    output_tokens_cost_in_usd=0.004,
)
"""What one scripted turn bills, so a lost fold is visible as a round number of cents.

The costs are stated, not priced from the counters.
A real adapter prices what the provider reported; this one reports round numbers.
"""


_TURN_BILLING = Billing(
    usage=_TURN_USAGE,
    service_tier="scripted",
    usage_raw=None,
    input_cache_none_usd_per_million_tokens=60.00,
    cache_read_usd_per_million_tokens=6.00,
    cache_write_usd_per_million_tokens=75.00,
    output_usd_per_million_tokens=200.00,
)
"""The rates behind _TURN_USAGE's costs, so counter times rate reproduces each stated cost."""


@dataclass
class Turn:
    """One scripted assistant turn.

    text ends the agent loop; tool_calls make the loop dispatch and generate again.
    delay_seconds suspends inside open_stream, which is what a per-call timeout races against.
    error, when set, is raised instead of returning, after the delay.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    delay_seconds: float = 0.0
    error: Exception | None = None


@dataclass
class Script:
    """The turns one agent tag plays, in order, plus a count of opens it received."""

    turns: list[Turn]
    opens: int = 0


class ScriptedAdapter(Adapter):
    """One adapter serving every agent; the binding's system prompt selects the script."""

    def __init__(self, scripts: dict[str, list[Turn]]) -> None:
        """Store one Script per agent tag."""
        super().__init__(client=None, model="fake-model", provider_name="fake")
        self.scripts = {tag: Script(turns=list(turns)) for tag, turns in scripts.items()}

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Hand out a bound adapter reading the script the system prompt names."""
        return _ScriptedBoundAdapter(self, _tag_of(binding))

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT]:
        """Reject a structured binding: the example reads structured output from tool arguments."""
        raise NotImplementedError

    failure_types: ClassVar[tuple[type[Exception], ...]] = (TransientError,)

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Map a TransientError with the shared rule; the example scripts nothing else transient."""
        if isinstance(failure, TransientError):
            return verdict_from_transient_error(failure)
        return DoNotRetry()

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify every error as unknown_exception so nothing silently retries in the example."""
        return "unknown_exception"


def _tag_of(binding: Binding) -> str:
    """Read the agent tag out of the binding's system prompt, which every binding in the example starts with."""
    system_prompt = binding.system_prompt
    if isinstance(system_prompt, str) and system_prompt.startswith("["):
        return system_prompt[1 : system_prompt.index("]")]
    return "default"


@dataclass(frozen=True, kw_only=True)
class _ScriptedRequest(RequestParams):
    """What a scripted attempt would have put on the wire, which is the messages and nothing else."""

    messages: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        """Render the messages as a JSON array of each message's dump."""
        return json.dumps([message.model_dump(mode="json") for message in self.messages])


class _ScriptedBoundAdapter(BoundAdapter[str]):
    """Plays one agent's scripted turns in order."""

    def __init__(self, adapter: ScriptedAdapter, tag: str) -> None:
        self._adapter = adapter
        self._tag = tag

    @override
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Report what one scripted turn bills, the same round numbers for every turn."""
        return _TURN_BILLING

    @override
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Name the scripted model and derive both ids from the turn this response came from.

        Raises:
            TypeError: raw is not a FakeRaw.
        """
        return ResponseIdentity(
            model_served="scripted-model",
            response_id=f"turn-{_turn_index(raw)}",
            request_id=f"req-{_turn_index(raw)}",
        )

    @override
    def interpret(self, raw: BaseModel) -> AdapterResult[str]:
        """Build the result the scripted turn this response names describes.

        Raises:
            TypeError: raw is not a FakeRaw.
        """
        turn = self._adapter.scripts[self._tag].turns[_turn_index(raw)]
        if turn.tool_calls:
            return AdapterResult(
                output="",
                assistant_message=AssistantMessage(turn=turn.tool_calls),
                stop_reason="tool_use",
            )
        assert turn.text is not None
        return AdapterResult(
            output=turn.text,
            assistant_message=AssistantMessage(turn=(TextPart(text=turn.text),)),
            stop_reason="end_turn",
        )

    @override
    def build_request(self, messages: Sequence[Message]) -> RequestParams:
        """Build the request the scripted attempts ignore; the script decides what comes back."""
        return _ScriptedRequest(messages=tuple(messages))

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open a stream over the next scripted turn for this tag, after its delay.

        Raises:
            Exception: the turn's scripted error, whatever type it carries.
            RuntimeError: the script for this tag ran out of turns.
        """
        script = self._adapter.scripts[self._tag]
        if script.opens >= len(script.turns):
            raise RuntimeError(f"script {self._tag!r} exhausted after {script.opens} turns")
        turn_index = script.opens
        turn = script.turns[turn_index]
        script.opens += 1
        if turn.delay_seconds:
            await asyncio.sleep(turn.delay_seconds)
        if turn.error is not None:
            raise turn.error
        return _ScriptedTurnStream(raw=FakeRaw(turn_index=turn_index))


class _ScriptedTurnStream(AdapterStream):
    """One scripted attempt's stream: it yields no items and assembles the turn's response.

    The example never iterates a stream itself; the retry loop drains this privately, so items()
    yielding nothing loses nothing.
    """

    def __init__(self, *, raw: FakeRaw) -> None:
        """Hold the response final() returns."""
        self._raw = raw

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        return
        yield

    @override
    async def final(self) -> FakeRaw:
        """Return the response naming the scripted turn."""
        return self._raw

    @override
    def billing_reported(self) -> None:
        """None: the scripted turn bills only through its assembled response."""

    @override
    def request_id(self) -> str | None:
        """None: identity_from_raw derives the request id from the response itself."""
        return None

    @override
    async def close(self) -> None:
        """Nothing to release."""


_CALL_IDS = itertools.count(1)


def call(name: str, args_json: str) -> ToolCall:
    """Build a ToolCall with a fresh id, so scripted calls never collide."""
    return ToolCall(id=f"call-{next(_CALL_IDS)}", name=name, args_json=args_json)


def build_llm(scripts: dict[str, list[Turn]]) -> LLM:
    """Wrap a ScriptedAdapter in an LLM with a fast, generous backpressure domain."""
    adapter = ScriptedAdapter(scripts)
    return LLM(
        adapter,
        shared_backoff=SharedBackoff(
            parse=adapter.parse,
            failure_types=adapter.failure_types,
            capacity=16,
            minimum_wait_ceiling=0.001,
            longest_wait=0.01,
        ),
        max_attempts=2,
    )
