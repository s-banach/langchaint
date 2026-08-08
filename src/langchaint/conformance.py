"""The invariants every adapter holds to, as a test class an adapter author inherits.

Subclass AdapterConformance once per adapter and implement its fixture methods; pytest collects
langchaint's own test methods against that adapter. The audience is adapter authors, so this module
sits off the top-level __all__, on the same tier as langchaint.adapter.

The inversion is forced: a test must supply an input, the input to an adapter is an SDK response
object, and there is no neutral response every adapter accepts. langchaint supplies the assertions
and the author supplies the fixtures.
The assertions are plain assert statements, and the ones needing an event loop run through
asyncio.run, so inheriting this suite pulls in nothing beyond the runner already collecting it.

Every invariant here is public API: changing one changes what every third-party suite asserts.

An adapter lacking a capability supplies a degenerate fixture rather than skipping. A provider with
no prompt caching returns zero cache-write tokens from response_with_cache_writes, and the cache
invariants run and pass, because zero counters cost zero at any rate. There is no skip hook, which
is what would let a divergence report green while covering less.
"""

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import get_args

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    Binding,
    BoundAdapter,
    ErrorClassification,
    InvalidRequest,
    RequestParams,
)
from langchaint.call import AttemptRecord, CallRecord
from langchaint.exceptions import AbandonedCallError, StreamProtocolError, TransientError
from langchaint.inference_params import InferenceParams
from langchaint.messages import (
    Message,
    ReasoningTrace,
    UserMessage,
    messages_from_json,
    messages_to_json,
)
from langchaint.pricing import Billing, category_cost
from langchaint.response import RowValue, to_tables
from langchaint.shared_backoff import Verdict
from langchaint.usage import ZERO_USAGE

_PLAIN_TEXT_BINDING = Binding(
    system_prompt=None,
    tool_schemas=(),
    tool_choice="auto",
    parallel_tool_calls=True,
    inference_params=InferenceParams(),
    automatic_prompt_caching=True,
)
"""The binding every invariant here binds under: text output and nothing else stated."""


def _costs_agree(actual: float, expected: float) -> bool:
    """Whether two costs are the same number, counting NaN as the same as NaN.

    NaN is a cost the rate table could not price, and it compares equal to nothing including itself,
    so an invariant reading a cost off an unpriced category needs this rather than ==.
    """
    if math.isnan(actual) and math.isnan(expected):
        return True
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _row_number(row: Mapping[str, RowValue], column: str) -> float:
    """Read one numeric cell of an attempts row.

    Every column this suite reads is filled, because the rows come from a Billing rather than from
    an attempt the provider reported nothing for.
    """
    value = row[column]
    assert isinstance(value, int | float), f"{column} is {value!r}, not a number"
    return value


class AdapterConformance(ABC):
    """Subclass once per adapter; langchaint supplies every test method.

    Name the subclass so pytest collects it (TestAnthropicMessagesConformance), put it in the
    adapter's own test module, and implement the fixture methods below. Each returns a freshly built
    value, so one test cannot see what another did to it.
    """

    @abstractmethod
    def make_adapter(self) -> Adapter:
        """Build the adapter under test.

        It holds a rate table for every tier the fixtures report except the one
        response_at_an_unpriced_tier reports.
        """
        ...

    @abstractmethod
    def response_with_cache_writes(self) -> BaseModel:
        """Return an SDK response reporting nonzero counters in every category the provider bills.

        A provider without prompt caching reports zero cache writes here; the cache invariants pass
        on zero counters at any rate.
        """
        ...

    @abstractmethod
    def response_without_usage(self) -> BaseModel:
        """Return an SDK response carrying no usage at all, which a provider may answer with."""
        ...

    @abstractmethod
    def response_at_an_unpriced_tier(self) -> BaseModel:
        """Return an SDK response reporting nonzero counters at a tier no table of make_adapter prices.

        The counters must be nonzero, since a category billing nothing costs zero at any rate and
        would say nothing about an unpriced one.
        """
        ...

    @abstractmethod
    def response_with_impossible_counters(self) -> BaseModel:
        """Return an SDK response whose counters cannot be partitioned, leaving one negative.

        Whatever shape that takes for the provider: a count below zero, or a cache count above the
        input total it is a share of.
        """
        ...

    @abstractmethod
    def response_with_reasoning(self) -> BaseModel:
        """Return an SDK response whose turn holds one reasoning element and one text part.

        The reasoning must carry a key the installed SDK does not name, which is what an adapter
        that rebuilt the payload from its own model would drop.
        The turn's elements must map one to one onto the wire elements a request built from that
        turn carries, so no two adjacent parts of a kind the adapter joins into one.
        """
        ...

    @abstractmethod
    def assistant_wire_elements(self, request: RequestParams) -> Sequence[object]:
        """Read the assistant turn's elements out of a request this adapter built, in wire order.

        The one fixture that is not an input: what an adapter puts on the wire has no neutral shape,
        so the author supplies the reader and langchaint supplies the comparison.
        The request comes from a Sequence[Message] of one UserMessage followed by the assistant turn
        under test, so everything the user message put on the wire is what to skip past.
        """
        ...

    @abstractmethod
    def streamed_and_whole(self) -> tuple[BaseModel, BaseModel]:
        """Return one turn twice: as the type the SDK's stream assembles into, and whole.

        The same turn in both shapes, so a difference between the two readings is the adapter's and
        not the fixture's.
        """
        ...

    @abstractmethod
    def stream_without_its_terminal_event(self) -> AdapterStream:
        """Return a stream whose events end before the one that closes the turn."""
        ...

    @abstractmethod
    def sdk_errors_and_classifications(self) -> Mapping[Exception, ErrorClassification]:
        """Every SDK exception this adapter places, against the classification it places it as."""
        ...

    @abstractmethod
    def sdk_errors_and_verdicts(self) -> Mapping[Exception, Verdict]:
        """Every failure this adapter's parse maps, against the exact verdict it returns.

        Cover at least one exception per verdict kind this adapter's parse can return, one carrying
        a server-stated retry_after, and one status the provider's table does not list.
        """
        ...

    def _bound_adapter(self) -> BoundAdapter[str]:
        """Bind a fresh adapter for plain text under the one binding these invariants use."""
        return self.make_adapter().bind_text(_PLAIN_TEXT_BINDING)

    def _assistant_wire_elements_of(
        self, bound_adapter: BoundAdapter[str], messages: Sequence[Message]
    ) -> Sequence[object]:
        """Build a request from messages and read its assistant turn's wire elements.

        Asserts the request is valid, so callers compare elements without repeating the guard.
        """
        request = bound_adapter.build_request(messages)
        assert not isinstance(request, InvalidRequest)
        return self.assistant_wire_elements(request)

    def _billings(self) -> list[Billing]:
        """Return the billing of each fixture response the cost invariants price."""
        bound_adapter = self._bound_adapter()
        return [
            bound_adapter.billing_from_raw(raw)
            for raw in (
                self.response_with_cache_writes(),
                self.response_without_usage(),
                self.response_at_an_unpriced_tier(),
            )
        ]

    def test_each_cost_is_its_counter_times_the_price_stored_beside_it(self) -> None:
        """A Billing's costs and the prices stored beside them reproduce each other.

        The price beside a cost is the price that produced it, blended where the provider wrote at
        more than one cache TTL, so a stored row reproduces its own arithmetic without the rate
        table that made it.
        """
        for billing in self._billings():
            usage = billing.usage
            assert _costs_agree(
                usage.input_tokens_cache_read_cost_in_usd,
                category_cost(
                    usage.input_tokens_cache_read, billing.cache_read_usd_per_million_tokens
                ),
            )
            assert _costs_agree(
                usage.input_tokens_cache_write_cost_in_usd,
                category_cost(
                    usage.input_tokens_cache_write, billing.cache_write_usd_per_million_tokens
                ),
            )
            assert _costs_agree(
                usage.input_tokens_cache_none_cost_in_usd,
                category_cost(
                    usage.input_tokens_cache_none, billing.input_cache_none_usd_per_million_tokens
                ),
            )
            assert _costs_agree(
                usage.output_tokens_cost_in_usd,
                category_cost(usage.output_tokens, billing.output_usd_per_million_tokens),
            )

    def test_counters_that_cannot_be_partitioned_raise_at_arrival(self) -> None:
        """A response whose counters leave one category negative fails where it arrives.

        Usage's counters are non-negative by validation, so a defective response fails at arrival
        rather than producing a wrong cost that reaches a caller as a number.
        """
        raw = self.response_with_impossible_counters()
        bound_adapter = self._bound_adapter()
        try:
            _ = bound_adapter.billing_from_raw(raw)
        except ValueError:
            return
        raise AssertionError("billing_from_raw accepted counters that cannot be partitioned")

    def test_a_response_reporting_no_usage_bills_zero_at_a_named_tier(self) -> None:
        """A response carrying no usage bills ZERO_USAGE, never None and never absent.

        A Billing exists whenever a response did, and it names the tier that would have priced it.
        """
        billing = self._bound_adapter().billing_from_raw(self.response_without_usage())
        assert billing.usage == ZERO_USAGE
        assert billing.service_tier

    def test_a_served_tier_the_adapter_holds_no_table_for_prices_nan(self) -> None:
        """Every price and the total cost are NaN at a tier no table prices.

        Not zero and not a raise: the response was paid for, so an unknown cost must not read as a
        free request and must not destroy the output.
        """
        billing = self._bound_adapter().billing_from_raw(self.response_at_an_unpriced_tier())
        assert math.isnan(billing.input_cache_none_usd_per_million_tokens)
        assert math.isnan(billing.cache_read_usd_per_million_tokens)
        assert math.isnan(billing.cache_write_usd_per_million_tokens)
        assert math.isnan(billing.output_usd_per_million_tokens)
        assert math.isnan(billing.usage.cost_in_usd)

    def test_reasoning_round_trips_verbatim_in_position(self) -> None:
        """The reasoning an adapter read off a turn goes back on the wire unchanged, where it sat.

        A provider rejects a turn whose reasoning it cannot verify, and langchaint models nothing
        inside that payload, so the only sound thing to send back is the bytes that arrived.
        """
        bound_adapter = self._bound_adapter()
        outcome = bound_adapter.interpret(self.response_with_reasoning())
        assert isinstance(outcome, AdapterResult)
        turn = outcome.assistant_message.turn
        ((index, trace),) = [
            (index, element)
            for index, element in enumerate(turn)
            if isinstance(element, ReasoningTrace)
        ]
        elements = self._assistant_wire_elements_of(
            bound_adapter, [UserMessage(content="hi"), outcome.assistant_message]
        )
        assert len(elements) == len(turn)
        assert elements[index] == trace.raw

    def test_a_json_round_tripped_turn_builds_the_same_wire_request(self) -> None:
        """A turn restored by messages_from_json puts the same elements on the wire as the original.

        This is what makes the round trip a persistence format: serialization must not lose or
        reshape a raw payload the provider verifies, and every value this adapter puts in
        ReasoningTrace.raw must serialize, which is where a non-JSON-representable value fails.
        """
        bound_adapter = self._bound_adapter()
        outcome = bound_adapter.interpret(self.response_with_reasoning())
        assert isinstance(outcome, AdapterResult)
        original: list[Message] = [UserMessage(content="hi"), outcome.assistant_message]
        restored = messages_from_json(messages_to_json(original))
        assert self._assistant_wire_elements_of(
            bound_adapter, restored
        ) == self._assistant_wire_elements_of(bound_adapter, original)

    def test_every_sdk_exception_classifies_and_an_unknown_one_still_does(self) -> None:
        """Every listed exception takes its stated classification, and an unlisted one still gets one.

        A listed exception must take the classification the author states. Totality over the SDK's
        own exception tree is not something an author-supplied mapping establishes, so the second
        half asserts only that a bare Exception comes back as some classification rather than as a
        raise: an adapter that cannot place an exception says unknown_exception.
        """
        adapter = self.make_adapter()
        for error, classification in self.sdk_errors_and_classifications().items():
            assert adapter.classify(error) == classification
        assert adapter.classify(Exception("no adapter has seen this")) in get_args(
            ErrorClassification.__value__
        )

    def test_every_listed_failure_parses_and_an_unknown_one_still_does(self) -> None:
        """Every listed failure takes its stated verdict, and an unlisted one still gets one.

        parse must return a verdict for every input without raising, statuses and error types the
        provider adds later included: SharedBackoff turns a raise into ParserContractError, which
        fails the request with no verdict at all.
        """
        adapter = self.make_adapter()
        for failure, verdict in self.sdk_errors_and_verdicts().items():
            assert adapter.parse(failure) == verdict
        unknown = adapter.parse(Exception("no adapter has seen this"))
        assert isinstance(unknown, get_args(Verdict.__value__))

    def test_failure_types_names_only_exception_subclasses_and_carries_transient_error(
        self,
    ) -> None:
        """failure_types entries are strict Exception subclasses, and TransientError is one.

        SharedBackoff rejects anything else at construction, so a violation here fails before any
        request. TransientError is required because the retry loop raises it inside the admitted()
        block for a billable response reporting a transient failure; an adapter without it would
        leave those failures unrecorded.
        """
        adapter = self.make_adapter()
        for failure_type in adapter.failure_types:
            assert issubclass(failure_type, Exception)
            assert failure_type is not Exception
        assert TransientError in adapter.failure_types

    def test_the_stream_assembled_type_reads_the_same_as_the_whole_response(self) -> None:
        """One turn read off the type a stream assembles into and off a whole response agree.

        Both request paths go through interpret, so the two readings are the same turn or the
        adapter reads one of the two shapes wrongly.
        """
        streamed, whole = self.streamed_and_whole()
        bound_adapter = self._bound_adapter()
        assert bound_adapter.interpret(streamed) == bound_adapter.interpret(whole)
        assert bound_adapter.billing_from_raw(streamed).usage == (
            bound_adapter.billing_from_raw(whole).usage
        )

    def test_a_stream_missing_its_terminal_event_raises(self) -> None:
        """Draining a stream whose turn never closed is a protocol violation, not an empty answer.

        The events that arrived are not the turn, and presenting them as one would hand a caller a
        truncated answer as if the model had finished.
        """
        stream = self.stream_without_its_terminal_event()

        async def drain() -> None:
            """Iterate the stream to its end, which is where the violation surfaces."""
            async for _item in stream.items():
                pass

        try:
            asyncio.run(drain())
        except StreamProtocolError:
            return
        raise AssertionError("a stream missing its terminal event drained without raising")

    def test_each_attempt_row_reproduces_its_cost_from_its_own_prices(self) -> None:
        """The archive's arithmetic closes without reaching back into any object.

        The same claim as test_each_cost_is_its_counter_times_the_price_stored_beside_it, read at
        the table boundary, because a row that dropped a price or paired it with the wrong counter
        would satisfy the first and fail here.
        """
        adapter = self.make_adapter()
        for billing in self._billings():
            (row,) = to_tables(_carrier_of(billing, adapter)).attempts
            for counter_column, cost_column, price_column in (
                (
                    "input_tokens_cache_read",
                    "input_tokens_cache_read_cost_in_usd",
                    "cache_read_usd_per_million_tokens",
                ),
                (
                    "input_tokens_cache_write",
                    "input_tokens_cache_write_cost_in_usd",
                    "cache_write_usd_per_million_tokens",
                ),
                (
                    "input_tokens_cache_none",
                    "input_tokens_cache_none_cost_in_usd",
                    "input_cache_none_usd_per_million_tokens",
                ),
                (
                    "output_tokens",
                    "output_tokens_cost_in_usd",
                    "output_usd_per_million_tokens",
                ),
            ):
                assert _costs_agree(
                    _row_number(row, cost_column),
                    category_cost(
                        int(_row_number(row, counter_column)), _row_number(row, price_column)
                    ),
                )


def _carrier_of(billing: Billing, adapter: Adapter) -> AbandonedCallError:
    """Wrap one Billing in a result carrier, to reach to_tables with a single attempts row."""
    return AbandonedCallError(
        call=CallRecord(
            model=adapter.model,
            provider_name=adapter.provider_name,
            attempt_records=(
                AttemptRecord(
                    started_at_monotonic_seconds=0.0,
                    ended_at_monotonic_seconds=1.0,
                    first_item_at_monotonic_seconds=None,
                    error=None,
                    billing=billing,
                    assistant_message=None,
                    raw=None,
                    model_served=None,
                    response_id=None,
                    request_id=None,
                ),
            ),
            started_at_monotonic_seconds=0.0,
            elapsed_seconds=1.0,
        ),
        billing_in_flight=None,
        in_flight_attempt_started_at_monotonic_seconds=None,
    )
