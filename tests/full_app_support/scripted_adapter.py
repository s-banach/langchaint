"""Provide a deterministic adapter for full-app tests.

Each binding's complete system prompt selects a list of turns.
Text ends a loop, and tool calls continue it.
delay_seconds suspends open_stream to exercise timeouts.
"""

import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Mapping, Sequence
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
    ProviderBilling,
    RequestParams,
    verdict_from_transient_error,
)
from langchaint.call import ResponseIdentity


class FakeRaw(BaseModel):
    """Identify the scripted turn represented by raw."""

    turn_index: int


def _turn_index(raw: BaseModel) -> int:
    """Return the scripted turn index from raw.

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
    provider_executed_tool_cost_in_usd=0.0,
)
"""Bill each scripted turn $0.01."""


_TURN_BILLING = Billing(
    usage=_TURN_USAGE,
    service_tier="scripted",
    input_cache_none_usd_per_million_tokens=60.00,
    cache_read_usd_per_million_tokens=6.00,
    cache_write_usd_per_million_tokens=75.00,
    output_usd_per_million_tokens=200.00,
)
"""Price the counters in _TURN_USAGE."""


@dataclass
class Turn:
    """Configure one scripted assistant turn.

    text ends the loop, and tool_calls continue it.
    delay_seconds suspends open_stream before error is raised or output is returned.
    started signals entry into open_stream before the delay.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    delay_seconds: float = 0.0
    started: asyncio.Event | None = None
    error: Exception | None = None


@dataclass
class Script:
    """Track one system prompt's turns and opens."""

    turns: list[Turn]
    opens: int = 0


class ScriptedAdapter(Adapter):
    """Serve every agent from scripts selected by system prompt."""

    def __init__(self, scripts: dict[str, list[Turn]]) -> None:
        """Store one Script per system prompt."""
        super().__init__(
            client=None,
            model="fake-model",
            provider_name="fake",
            automatic_cache_breakpoints_default=False,
        )
        self.scripts: dict[str, Script] = {
            system_prompt: Script(turns=list(turns)) for system_prompt, turns in scripts.items()
        }

    @override
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return the scripted adapter's stored request configuration."""
        return {}

    @override
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind to the script named by the system prompt.

        Raises:
            TypeError: The binding does not contain a string system prompt.
        """
        system_prompt = binding.system_prompt
        if not isinstance(system_prompt, str):
            raise TypeError("ScriptedAdapter requires a string system prompt")
        return _ScriptedBoundAdapter(self, system_prompt)

    @override
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT]:
        """Reject structured bindings because tools carry structured output."""
        raise NotImplementedError

    failure_types: ClassVar[tuple[type[Exception], ...]] = (TransientError,)

    @override
    def parse(self, failure: Exception) -> Verdict:
        """Map TransientError with verdict_from_transient_error."""
        if isinstance(failure, TransientError):
            return verdict_from_transient_error(failure)
        return DoNotRetry()

    @override
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify every error as unknown_exception."""
        return "unknown_exception"


@dataclass(frozen=True, kw_only=True)
class _ScriptedRequest(RequestParams):
    """Store the messages for one scripted attempt."""

    messages: tuple[Message, ...]

    @override
    def as_json(self) -> str:
        """Serialize the messages as JSON."""
        return json.dumps([message.model_dump(mode="json") for message in self.messages])


class _ScriptedBoundAdapter(BoundAdapter[str]):
    """Play one agent's scripted turns in order."""

    def __init__(self, adapter: ScriptedAdapter, system_prompt: str) -> None:
        self._adapter = adapter
        self._system_prompt = system_prompt

    @override
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Return the constant billing for one turn."""
        return ProviderBilling(billing=_TURN_BILLING, usage_raw=None)

    @override
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Build the scripted response identity.

        Raises:
            TypeError: `raw` is not a `FakeRaw`.
        """
        return ResponseIdentity(
            model_served="scripted-model",
            response_id=f"turn-{_turn_index(raw)}",
            request_id=request_id,
        )

    @override
    def interpret(self, raw: BaseModel) -> AdapterResult[str]:
        """Build the result for the scripted turn.

        Raises:
            TypeError: `raw` is not a `FakeRaw`.
        """
        turn = self._adapter.scripts[self._system_prompt].turns[_turn_index(raw)]
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
        """Build a request containing the messages."""
        return _ScriptedRequest(messages=tuple(messages))

    @override
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open the next scripted turn after its delay.

        Raises:
            Exception: the turn's scripted error, whatever type it carries.
            RuntimeError: the script for this system prompt ran out of turns.
        """
        script = self._adapter.scripts[self._system_prompt]
        if script.opens >= len(script.turns):
            raise RuntimeError(
                f"script for {self._system_prompt!r} exhausted after {script.opens} turns"
            )
        turn_index = script.opens
        turn = script.turns[turn_index]
        script.opens += 1
        if turn.started is not None:
            turn.started.set()
        if turn.delay_seconds:
            await asyncio.sleep(turn.delay_seconds)
        if turn.error is not None:
            raise turn.error
        return _ScriptedTurnStream(raw=FakeRaw(turn_index=turn_index))


class _ScriptedTurnStream(AdapterStream):
    """Assemble one scripted response without yielding items."""

    def __init__(self, *, raw: FakeRaw) -> None:
        """Store the response returned by final."""
        self._raw = raw

    @override
    async def items(self) -> AsyncIterator[StreamItem]:
        return
        yield

    @override
    async def final(self) -> FakeRaw:
        """Return the scripted response."""
        return self._raw

    @override
    def billing_reported(self) -> None:
        """Report no billing before final."""

    @override
    def request_id(self) -> str | None:
        """Return no request ID."""
        return None

    @override
    async def close(self) -> None:
        """Release no resources."""


_CALL_IDS = itertools.count(1)


def call(name: str, args_json: str) -> ToolCall:
    """Build a ToolCall with a unique ID."""
    return ToolCall(id=f"call-{next(_CALL_IDS)}", name=name, args_json=args_json)


def build_llm(scripts: dict[str, list[Turn]]) -> LLM:
    """Build an LLM with ScriptedAdapter and fast request pacing."""
    adapter = ScriptedAdapter(scripts)
    return LLM(
        adapter,
        shared_backoff=SharedBackoff(
            parse=adapter.parse,
            failure_types=adapter.failure_types,
            max_concurrent_requests=16,
            max_request_starts_per_second=10_000.0,
            minimum_wait_ceiling_seconds=0.001,
            longest_wait_seconds=0.01,
        ),
    )
