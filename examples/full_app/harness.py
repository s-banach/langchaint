"""A scriptable offline adapter so the example runs with no network.

A script is a list of turns keyed by a tag the binding carries in its system prompt,
so one adapter serves every agent in the graph and each agent gets its own scripted turns.
Each turn is either text (ends that agent's loop) or tool calls (the loop dispatches and comes back).
delay_seconds on a turn makes send suspend, which is how the timeout layers get exercised.
"""

import asyncio
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from pydantic import BaseModel

from langchaint import (
    LLM,
    AssistantMessage,
    Message,
    PricingTable,
    RateLimiter,
    TextPart,
    ToolCall,
    Usage,
)
from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    BoundAdapter,
    ErrorClassification,
)

_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)


class FakeRaw(BaseModel):
    """Stands in for the SDK response model an adapter holds in raw."""

    id: str = "fake"


class FakeRawUsage(BaseModel):
    """Stands in for the SDK usage object an adapter holds in usage_raw."""


_TURN_USAGE = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=100,
    output_tokens=20,
    output_tokens_reasoning=0,
    cost_in_usd=0.01,
)
"""What one scripted turn bills, so a lost fold is visible as a round number of cents."""


@dataclass
class Turn:
    """One scripted assistant turn.

    text ends the agent loop; tool_calls make the loop dispatch and generate again.
    delay_seconds suspends inside send, which is what a per-call timeout races against.
    error, when set, is raised instead of returning, after the delay.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    delay_seconds: float = 0.0
    error: Exception | None = None


@dataclass
class Script:
    """The turns one agent tag plays, in order, plus a count of sends it received."""

    turns: list[Turn]
    sends: int = 0


class ScriptedAdapter(Adapter):
    """One adapter serving every agent; the binding's system prompt selects the script."""

    def __init__(self, scripts: dict[str, list[Turn]]) -> None:
        """Store one Script per agent tag."""
        super().__init__(client=None, model="fake-model", pricing=_PRICING, provider_name="fake")
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

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify every error as fatal so nothing silently retries in the example."""
        return "fatal"


def _tag_of(binding: Binding) -> str:
    """Read the agent tag out of the binding's system prompt, which every binding in the example starts with."""
    system_prompt = binding.system_prompt
    if isinstance(system_prompt, str) and system_prompt.startswith("["):
        return system_prompt[1 : system_prompt.index("]")]
    return "default"


class _ScriptedBoundAdapter(BoundAdapter[str]):
    """Plays one agent's scripted turns in order."""

    def __init__(self, adapter: ScriptedAdapter, tag: str) -> None:
        self._adapter = adapter
        self._tag = tag

    @override
    async def send(self, conversation: Sequence[Message]) -> AdapterResult[str]:
        """Return the next scripted turn for this tag, after its delay.

        Raises:
            Exception: the turn's scripted error, whatever type it carries.
            RuntimeError: the script for this tag ran out of turns.
        """
        script = self._adapter.scripts[self._tag]
        if script.sends >= len(script.turns):
            raise RuntimeError(f"script {self._tag!r} exhausted after {script.sends} turns")
        turn = script.turns[script.sends]
        script.sends += 1
        if turn.delay_seconds:
            await asyncio.sleep(turn.delay_seconds)
        if turn.error is not None:
            raise turn.error
        if turn.tool_calls:
            return AdapterResult(
                output="",
                assistant_message=AssistantMessage(turn=turn.tool_calls),
                usage=_TURN_USAGE,
                usage_raw=FakeRawUsage(),
                stop_reason="tool_use",
                raw=FakeRaw(),
            )
        assert turn.text is not None
        return AdapterResult(
            output=turn.text,
            assistant_message=AssistantMessage(turn=(TextPart(text=turn.text),)),
            usage=_TURN_USAGE,
            usage_raw=FakeRawUsage(),
            stop_reason="end_turn",
            raw=FakeRaw(),
        )

    @override
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream[str]:
        """Reject a stream open: the example uses generate_one only."""
        raise NotImplementedError


_CALL_IDS = itertools.count(1)


def call(name: str, args_json: str) -> ToolCall:
    """Build a ToolCall with a fresh id, so scripted calls never collide."""
    return ToolCall(id=f"call-{next(_CALL_IDS)}", name=name, args_json=args_json)


def build_llm(scripts: dict[str, list[Turn]]) -> LLM:
    """Wrap a ScriptedAdapter in an LLM with a fast, generous limiter."""
    return LLM(
        ScriptedAdapter(scripts),
        rate_limiter=RateLimiter(max_attempts=2, backoff_base_seconds=0.001, max_in_flight=16),
    )
