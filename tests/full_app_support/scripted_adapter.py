"""Provide a deterministic adapter for full-app tests.

Each binding's complete system prompt selects a list of turns.
Text ends a loop, and tool calls continue it.
delay_seconds suspends open_stream to exercise timeouts.
"""

import asyncio
import itertools
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import override

from pydantic import BaseModel, TypeAdapter

from langchaint import (
    LLM,
    AssistantMessage,
    Billing,
    Message,
    SharedBackoff,
    StreamItem,
    TextPart,
    TokenRates,
    ToolCall,
    ToolChoice,
    Usage,
    llm_from_adapter,
)
from langchaint.adapter import (
    Adapter,
    AdapterStream,
    Binding,
    BoundAdapter,
    FinishedResponse,
    ProviderBilling,
    RequestFailure,
    RequestParams,
    ResponseIdentity,
)
from langchaint.generation.observer import Observer


class FakeRaw(BaseModel):
    """Identify the scripted turn represented by raw."""

    turn_index: int


class TerminalProviderError(Exception):
    """A scripted provider failure that ends handling the input."""


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
    usd_per_million_tokens=TokenRates(
        input_tokens_cache_read=6.00,
        input_tokens_cache_write=75.00,
        input_tokens_cache_none=60.00,
        output_tokens=200.00,
    ),
)
"""Price the counters in _TURN_USAGE."""


@dataclass
class Turn:
    """Configure one scripted turn.

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
    requests: list["_ScriptedRequestParams"] = field(default_factory=list)


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
    def bind_text(self, binding: Binding) -> BoundAdapter:
        """Bind to the script named by the system prompt.

        Raises:
            TypeError: The binding does not contain a string system prompt.
        """
        system_prompt = binding.system_prompt
        if not isinstance(system_prompt, str):
            raise TypeError("ScriptedAdapter requires a string system prompt")
        return _ScriptedBoundAdapter(self, system_prompt, binding.tool_choice)

    @override
    def bind_structured(self, binding: Binding, response_format: type[BaseModel]) -> BoundAdapter:
        """Reject structured bindings because tools carry structured output."""
        raise NotImplementedError

    @override
    def request_failure(self, error: Exception) -> RequestFailure:
        """Classify TerminalProviderError and preserve other errors as unknown_exception."""
        return RequestFailure(
            kind=(
                "provider_failed_terminally"
                if isinstance(error, TerminalProviderError)
                else "unknown_exception"
            ),
            pauses_quota=False,
            retry_after_seconds=None,
        )


@dataclass(frozen=True, kw_only=True)
class _ScriptedRequestParams(RequestParams):
    """Store the messages for one scripted request."""

    messages: tuple[Message, ...]
    tool_choice: ToolChoice

    @override
    def as_json(self) -> str:
        """Serialize the request as JSON."""
        return TypeAdapter(_ScriptedRequestParams).dump_json(self).decode()


class _ScriptedBoundAdapter(BoundAdapter):
    """Play one agent's scripted turns in order."""

    def __init__(
        self, adapter: ScriptedAdapter, system_prompt: str, tool_choice: ToolChoice
    ) -> None:
        self._adapter = adapter
        self._system_prompt = system_prompt
        self._tool_choice = tool_choice

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
    def interpret(self, raw: BaseModel) -> FinishedResponse:
        """Build the finished response for the scripted turn.

        Raises:
            TypeError: `raw` is not a `FakeRaw`.
        """
        turn = self._adapter.scripts[self._system_prompt].turns[_turn_index(raw)]
        if turn.tool_calls:
            return FinishedResponse(
                assistant_message=AssistantMessage(content=turn.tool_calls),
                stop_reason="tool_call",
                cannot_continue=None,
            )
        assert turn.text is not None
        return FinishedResponse(
            assistant_message=AssistantMessage(content=(TextPart(text=turn.text),)),
            stop_reason="stop",
            cannot_continue=None,
        )

    @override
    def build_request_params(self, messages: Sequence[Message]) -> RequestParams:
        """Build request params containing the messages."""
        return _ScriptedRequestParams(messages=tuple(messages), tool_choice=self._tool_choice)

    @override
    async def open_stream(
        self, request_params: RequestParams, *, request_count: int
    ) -> AdapterStream:
        """Open the next scripted turn after its delay.

        Raises:
            Exception: the turn's scripted error, whatever type it carries.
            RuntimeError: the script for this system prompt ran out of turns.
        """
        script = self._adapter.scripts[self._system_prompt]
        assert isinstance(request_params, _ScriptedRequestParams)
        script.requests.append(request_params)
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
    def provider_billing(self) -> None:
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


def build_llm(
    scripts: dict[str, list[Turn]], *, system_prompt: str, observer: Observer | None = None
) -> LLM:
    """Share the scripted adapter and admission limits across configured models."""
    return llm_from_adapter(
        ScriptedAdapter(scripts),
        system_prompt=system_prompt,
        observer=observer,
        shared_backoff=SharedBackoff(
            max_concurrent_requests=16,
            max_request_starts_per_second=10_000.0,
            min_wait_ceiling_seconds=0.001,
            max_wait_seconds=0.01,
        ),
    )
