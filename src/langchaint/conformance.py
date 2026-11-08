"""The conformance test classes for adapter implementations.

Subclass `AdapterConformance` or `PartWireConformance` as their docstrings specify, and implement the fixture methods.
The inherited tests validate each neutral adapter invariant against SDK response objects.
Return degenerate fixtures for unsupported capabilities so each invariant still runs.
"""

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    AdapterStream,
    Binding,
    BoundAdapter,
    FinishedResponse,
    RejectedMessages,
    RequestFailure,
    RequestParams,
)
from langchaint.billing.pricing import Billing
from langchaint.billing.usage import ZERO_USAGE
from langchaint.common.exceptions import StreamProtocolError
from langchaint.common.messages import (
    AssistantMessage,
    Message,
    TextPart,
    UserMessage,
    messages_from_json,
    messages_to_json,
)
from langchaint.generation.request_history import (
    AbandonedStreamRecord,
    RequestHistory,
    SettledRequestRecord,
)
from langchaint.generation.tables import RowValue, to_tables

_PLAIN_BINDING = Binding(
    system_prompt=None,
    tool_schemas=(),
    provider_executed_tools=(),
    tool_choice="auto",
    parallel_tool_calls=True,
    max_completion_tokens=1024,
    reasoning_level=None,
    temperature=None,
    automatic_cache_breakpoints=True,
)
"""The binding used by the conformance tests: no system prompt, no tools, and a completion limit.

The limit is stated because `AnthropicMessagesAdapter` rejects a binding without one.
"""


def _costs_agree(actual: float, expected: float) -> bool:
    """Compare costs with rounding tolerance, counting NaN as equal to NaN."""
    if math.isnan(actual) and math.isnan(expected):
        return True
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _row_number(row: Mapping[str, RowValue], column: str) -> float:
    """Read one numeric cell of a `Tables.requests` row.

    This suite reads only rows backed by `Billing`, so each selected column is filled.
    """
    value = row[column]
    assert isinstance(value, int | float), f"{column} is {value!r}, not a number"
    return value


class AdapterConformance(ABC):
    """Subclass once per adapter that represents an assistant message through individual provider fields.

    Subclass `PartWireConformance` for adapters that represent assistant messages as lists of provider parts.
    langchaint supplies every test method.
    Use a pytest-compatible name such as `TestOpenAIChatCompletionsConformance`.
    Put the subclass in the adapter test module and implement the fixture methods below.
    Return a fresh value so tests cannot share mutations.
    """

    @abstractmethod
    def make_adapter(self) -> Adapter:
        """Build the adapter under test.

        Give the pricing table rates for every fixture tier except the tier of `response_at_an_unpriced_tier`.
        """
        ...

    @abstractmethod
    def response_with_cache_writes(self) -> BaseModel:
        """Return an SDK response reporting nonzero counters in every category the provider bills.

        Report zero cache writes when the provider lacks prompt caching.
        """
        ...

    @abstractmethod
    def response_without_usage(self) -> BaseModel:
        """Return an SDK response carrying no usage."""
        ...

    @abstractmethod
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return an SDK response reporting nonzero counters at a tier without rates.

        Use nonzero counters because zero tokens cost zero at every rate.
        """
        ...

    @abstractmethod
    def response_with_impossible_counters(self) -> BaseModel:
        """Return an SDK response whose counters cannot be partitioned, leaving one negative.

        Use a negative count or a cache count above its input total.
        """
        ...

    @abstractmethod
    def response_with_text(self, text: str) -> BaseModel:
        """Return an SDK response for a finished assistant message whose only answer text is `text`."""
        ...

    @abstractmethod
    def response_with_tool_call(self, text: str) -> BaseModel:
        """Construct a finished SDK response containing text and an application tool call.

        Args:
            text: The text accompanying the tool call.
        """
        ...

    @abstractmethod
    def response_with_reasoning(self) -> BaseModel:
        """Return an SDK response whose assistant message holds one ReasoningPart and one TextPart.

        The reasoning must carry a key absent from the installed SDK.
        Rebuilding the payload would drop that key.
        Each AssistantPart must map to one wire part.
        Avoid adjacent parts the adapter joins.
        """
        ...

    @abstractmethod
    def assistant_wire_parts(self, request_params: RequestParams) -> Sequence[object]:
        """Read the assistant message's wire parts from request params, in wire order.

        The request params contain one `UserMessage` before the tested assistant message.
        Skip the wire content produced by that `UserMessage`.

        Args:
            request_params: The adapter-specific request params to inspect.
        """
        ...

    @abstractmethod
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return the same assistant message as the SDK's assembled stream type and whole-response type."""
        ...

    @abstractmethod
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream that ends before its terminal event."""
        ...

    @abstractmethod
    def sdk_errors_and_request_failures(self) -> Mapping[Exception, RequestFailure]:
        """SDK exceptions against the exact `RequestFailure` that `request_failure` returns.

        Cover decisions owned by the adapter, including mid-stream errors and neutral failure names.
        Test SDK delegation against the SDK's answer without encoding its retry policy.
        """
        ...

    def _bound_adapter(self) -> BoundAdapter:
        return self.make_adapter().bind_text(_PLAIN_BINDING)

    def _finished_response(self, raw: BaseModel) -> FinishedResponse:
        interpretation = self._bound_adapter().interpret(raw)
        assert interpretation.kind == "finished_response"
        return interpretation

    def _text_parts(self, assistant_message: AssistantMessage) -> list[str]:
        assert isinstance(assistant_message.content, tuple)
        return [part.text for part in assistant_message.content if part.kind == "text"]

    def _assistant_wire_parts_of(
        self, bound_adapter: BoundAdapter, messages: Sequence[Message]
    ) -> Sequence[object]:
        request_params = bound_adapter.build_request_params(messages)
        assert not isinstance(request_params, RejectedMessages)
        return self.assistant_wire_parts(request_params)

    def _billings(self) -> list[Billing]:
        bound_adapter = self._bound_adapter()
        return [
            bound_adapter.billing_from_raw(raw).billing
            for raw in (
                self.response_with_cache_writes(),
                self.response_without_usage(),
                self.response_at_an_unpriced_tier(),
            )
        ]

    def test_counters_that_cannot_be_partitioned_raise_at_arrival(self) -> None:
        """Reject negative usage categories before a wrong cost reaches the caller."""
        raw = self.response_with_impossible_counters()
        bound_adapter = self._bound_adapter()
        try:
            _ = bound_adapter.billing_from_raw(raw)
        except ValueError:
            return
        raise AssertionError("billing_from_raw accepted counters that cannot be partitioned")

    def test_a_response_reporting_no_usage_bills_zero_at_a_named_tier(self) -> None:
        """Preserve a response's billing tier when usage is absent."""
        billing = self._bound_adapter().billing_from_raw(self.response_without_usage()).billing
        assert billing.usage == ZERO_USAGE
        assert billing.service_tier

    def test_a_served_tier_without_rates_prices_nan(self) -> None:
        """Represent unknown prices with NaN so paid output does not appear free."""
        billing = (
            self._bound_adapter().billing_from_raw(self.response_at_an_unpriced_tier()).billing
        )
        assert math.isnan(billing.usd_per_million_tokens.input_tokens_cache_none)
        assert math.isnan(billing.usd_per_million_tokens.input_tokens_cache_read)
        assert math.isnan(billing.usd_per_million_tokens.input_tokens_cache_write)
        assert math.isnan(billing.usd_per_million_tokens.output_tokens)
        assert math.isnan(billing.usage.cost_in_usd)

    def test_reasoning_round_trips_verbatim_in_position(self) -> None:
        """Replay reasoning unchanged in its original position so providers can verify its payload."""
        assistant_message = self._finished_response(
            self.response_with_reasoning()
        ).assistant_message
        content = assistant_message.content
        assert isinstance(content, tuple)
        ((index, reasoning_part),) = [
            (index, part) for index, part in enumerate(content) if part.kind == "reasoning"
        ]
        wire_parts = self._assistant_wire_parts_of(
            self._bound_adapter(), [UserMessage(content="hi"), assistant_message]
        )
        assert len(wire_parts) == len(content)
        assert wire_parts[index] == reasoning_part.raw

    def test_a_json_round_tripped_assistant_message_builds_the_same_wire_request(self) -> None:
        """Preserve provider-verified reasoning payloads through assistant message serialization and replay.

        Every `ReasoningPart.raw` value must be JSON-representable.
        """
        bound_adapter = self._bound_adapter()
        assistant_message = self._finished_response(
            self.response_with_reasoning()
        ).assistant_message
        original: list[Message] = [UserMessage(content="hi"), assistant_message]
        restored = messages_from_json(messages_to_json(original))
        assert self._assistant_wire_parts_of(
            bound_adapter, restored
        ) == self._assistant_wire_parts_of(bound_adapter, original)

    def test_empty_provider_text_preserves_its_text_part(self) -> None:
        """Empty provider text preserves its complete text-part boundary."""
        finished_response = self._finished_response(self.response_with_text(""))
        assert finished_response.assistant_message.content == (TextPart(text=""),)

    def test_an_empty_text_part_is_sent_as_given(self) -> None:
        """Replay sends every assistant `TextPart`, including one with empty text."""
        wire_parts = self._assistant_wire_parts_of(
            self._bound_adapter(),
            [UserMessage(content="hi"), AssistantMessage(content=(TextPart(text=""),))],
        )
        assert len(wire_parts) == 1

    def test_assistant_strings_and_text_parts_build_the_same_wire_request(self) -> None:
        """A string and a tuple containing one TextPart have the same provider representation."""
        bound_adapter = self._bound_adapter()
        for text in ("", "hello"):
            string_parts = self._assistant_wire_parts_of(
                bound_adapter, [UserMessage(content="hi"), AssistantMessage(content=text)]
            )
            tuple_parts = self._assistant_wire_parts_of(
                bound_adapter,
                [UserMessage(content="hi"), AssistantMessage(content=(TextPart(text=text),))],
            )
            assert string_parts == tuple_parts

    def test_tool_call_text_stays_beside_the_tool_call(self) -> None:
        """Text beside an application tool call stays in the assistant message with the call."""
        text = "Checking the temperature."
        assistant_message = self._finished_response(
            self.response_with_tool_call(text)
        ).assistant_message
        assert assistant_message.tool_calls
        assert self._text_parts(assistant_message) == [text]

    def test_every_sdk_exception_maps_and_an_unknown_one_is_unknown_exception(self) -> None:
        adapter = self.make_adapter()
        for error, request_failure in self.sdk_errors_and_request_failures().items():
            assert adapter.request_failure(error) == request_failure
        assert adapter.request_failure(Exception("no adapter has seen this")) == RequestFailure(
            kind="unknown_exception", pauses_quota=False, retry_after_seconds=None
        )

    def test_the_stream_assembled_type_reads_the_same_as_the_whole_response(self) -> None:
        """Interpret and bill the SDK's assembled stream type and whole-response type identically."""
        streamed, whole = self.streamed_and_whole()
        bound_adapter = self._bound_adapter()
        assert bound_adapter.interpret(streamed) == bound_adapter.interpret(whole)
        assert bound_adapter.billing_from_raw(streamed).billing.usage == (
            bound_adapter.billing_from_raw(whole).billing.usage
        )

    def test_a_stream_missing_its_terminal_event_raises(self) -> None:
        """Reject unfinished streams so truncated output cannot appear complete."""
        stream = self.stream_without_its_terminal_event()

        async def drain() -> None:
            async for _item in stream.items():
                pass

        try:
            asyncio.run(drain())
        except StreamProtocolError:
            return
        raise AssertionError("a stream missing its terminal event drained without raising")

    def test_to_tables_preserves_billing_fields_in_each_request_row(self) -> None:
        """Compare exported counters, costs, and rates with the source Billing."""
        adapter = self.make_adapter()
        for billing in self._billings():
            (row,) = to_tables(_carrier_of(billing, adapter)).requests
            usage = billing.usage
            for column, expected in (
                ("input_tokens_cache_read", usage.input_tokens_cache_read),
                ("input_tokens_cache_write", usage.input_tokens_cache_write),
                ("input_tokens_cache_none", usage.input_tokens_cache_none),
                ("output_tokens", usage.output_tokens),
                ("input_tokens_cache_read_cost_in_usd", usage.input_tokens_cache_read_cost_in_usd),
                (
                    "input_tokens_cache_write_cost_in_usd",
                    usage.input_tokens_cache_write_cost_in_usd,
                ),
                ("input_tokens_cache_none_cost_in_usd", usage.input_tokens_cache_none_cost_in_usd),
                ("output_tokens_cost_in_usd", usage.output_tokens_cost_in_usd),
                (
                    "input_tokens_cache_read_usd_per_million_tokens",
                    billing.usd_per_million_tokens.input_tokens_cache_read,
                ),
                (
                    "input_tokens_cache_write_usd_per_million_tokens",
                    billing.usd_per_million_tokens.input_tokens_cache_write,
                ),
                (
                    "input_tokens_cache_none_usd_per_million_tokens",
                    billing.usd_per_million_tokens.input_tokens_cache_none,
                ),
                (
                    "output_tokens_usd_per_million_tokens",
                    billing.usd_per_million_tokens.output_tokens,
                ),
            ):
                assert _costs_agree(_row_number(row, column), expected)


class PartWireConformance(AdapterConformance, ABC):
    """Conformance tests for adapters that represent assistant messages as lists of provider parts.

    Using `AdapterConformance` for this representation can hide a dropped `RawPart` or merged text parts.
    """

    @abstractmethod
    def response_with_text_parts(self, texts: Sequence[str]) -> BaseModel:
        """Construct a finished SDK response with separate complete text parts in order.

        Args:
            texts: The complete text parts, including empty text.
        """
        ...

    @abstractmethod
    def response_with_raw_part(self) -> BaseModel:
        """Return an SDK response whose assistant message holds one RawPart.

        Beside it, include at least one other AssistantPart.
        Each AssistantPart must map to one wire part.
        """
        ...

    def test_raw_part_round_trips_verbatim_in_position(self) -> None:
        """RawPart.raw returns unchanged in its original position.

        Require a RawPart because dropping it during interpretation shortens both compared sequences.
        """
        assistant_message = self._finished_response(
            self.response_with_raw_part()
        ).assistant_message
        content = assistant_message.content
        assert isinstance(content, tuple)
        assert any(part.kind == "raw" for part in content)
        wire_parts = self._assistant_wire_parts_of(
            self._bound_adapter(), [UserMessage(content="hi"), assistant_message]
        )
        assert len(wire_parts) == len(content)
        for index, part in enumerate(content):
            if part.kind == "raw":
                assert wire_parts[index] == part.raw

    def test_text_parts_keep_their_boundaries_and_order(self) -> None:
        """Every provider text part becomes its own `TextPart`, so the final one gives output."""
        texts = ("Checking the temperature.", '{"city":"Lima","celsius":19}', "")
        assistant_message = self._finished_response(
            self.response_with_text_parts(texts)
        ).assistant_message
        assert self._text_parts(assistant_message) == list(texts)


def _carrier_of(billing: Billing, adapter: Adapter) -> AbandonedStreamRecord:
    return AbandonedStreamRecord(
        request_history=RequestHistory(
            model=adapter.model,
            provider_name=adapter.provider_name,
            records=(
                SettledRequestRecord(
                    started_after_seconds=0.0,
                    elapsed_seconds=1.0,
                    first_item_after_seconds=None,
                    error=None,
                    billing=billing,
                    assistant_message=None,
                    model_served=None,
                    response_id=None,
                    request_id=None,
                ),
            ),
            elapsed_seconds=1.0,
        ),
    )
