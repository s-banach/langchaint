"""OpenAI Responses adapter helpers over constructed SDK objects.

These pin behavior the type checker cannot:
the usage partition derived by subtracting cache counts from input_tokens and its cross-check, cost arithmetic,
input-item placement, tool-choice translation, stop-reason derivation (the API reports no finish reason),
the zero-usage fallback when a response omits usage, and the precomputed request the binding determines.
"""

import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
from typing import override

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from openai.lib.streaming.responses import AsyncResponseStream, ResponseStreamEvent
from openai.lib.streaming.responses import (
    ResponseTextDeltaEvent as AccumulatedResponseTextDeltaEvent,
)
from openai.lib.streaming.responses._events import (
    ResponseCompletedEvent as AccumulatedResponseCompletedEvent,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputRefusal,
    ResponseUsage,
)
from openai.types.responses.parsed_response import (
    ParsedResponse,
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
)
from openai.types.responses.response import IncompleteDetails, ResponseStatus
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel, ValidationError

from langchaint import (
    AssistantMessage,
    ExceededMaxCompletionTokensError,
    ImagePart,
    InferenceParams,
    PricingTable,
    ReasoningTrace,
    RefusalError,
    SpecificTool,
    StreamItem,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from langchaint.exceptions import StreamProtocolError, TransientError
from langchaint.openai import OpenAIResponsesProvider, cost_breakdown
from langchaint.openai.responses_provider import (
    _assistant_items,
    _assistant_message_from,
    _BoundOpenAIStructured,
    _BoundOpenAIText,
    _cost_in_usd,
    _normalized_stop_reason,
    _normalized_usage,
    _OpenAIStream,
    _provider_result,
    _wire_input,
    _wire_tool_choice,
)
from langchaint.provider import Binding

_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=2.5,
    output_usd_per_million_tokens=10.0,
    cache_read_usd_per_million_tokens=1.25,
    cache_write_usd_per_million_tokens=3.125,
)

_TEXT_OUTPUT_ITEM: dict[str, object] = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hey", "annotations": []}],
}

_FUNCTION_CALL_OUTPUT_ITEM = {
    "type": "function_call",
    "id": "fc1",
    "call_id": "call1",
    "name": "lookup",
    "arguments": '{"q": 1}',
}

_REASONING_OUTPUT_ITEM: dict[str, object] = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "enc-1",
}


def _usage_with_cache() -> ResponseUsage:
    """Return usage whose input_tokens includes both cache counts."""
    return ResponseUsage(
        input_tokens=1000,
        input_tokens_details=InputTokensDetails(cached_tokens=600, cache_write_tokens=100),
        output_tokens=40,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=1040,
    )


def _response(
    *,
    usage: ResponseUsage | None,
    output: list[object] | None = None,
    status: str = "completed",
    incomplete_details: IncompleteDetails | None = None,
) -> OpenAIResponse:
    """Build a response carrying the given usage, output items, and status."""
    return OpenAIResponse.model_validate({
        "id": "r1",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": output if output is not None else [_TEXT_OUTPUT_ITEM],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "incomplete_details": incomplete_details,
        "usage": usage,
    })


def test_normalized_usage_subtracts_cache_from_input_tokens_and_prices() -> None:
    """The uncached counter is input_tokens minus both cache counts, and the priced cost rides on it."""
    usage = _normalized_usage(_usage_with_cache(), _PRICING)
    assert usage.input_tokens_cache_read == 600
    assert usage.input_tokens_cache_write == 100
    assert usage.input_tokens_cache_none == 300
    assert usage.input_tokens_total == 1000
    assert usage.cost_in_usd == _cost_in_usd(_usage_with_cache(), _PRICING)


def test_normalized_usage_reads_reasoning_tokens() -> None:
    """output_tokens_reasoning reads the required reasoning_tokens counter."""
    usage = _normalized_usage(
        ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens=20,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
            total_tokens=30,
        ),
        _PRICING,
    )
    assert usage.output_tokens_reasoning == 8


def test_normalized_usage_rejects_cache_counts_exceeding_input_tokens() -> None:
    """Cache counters summing past input_tokens raise instead of going negative.

    The subtraction derives input_tokens_cache_none, so the guard is its non-negativity constraint.
    """
    with pytest.raises(ValidationError):
        _normalized_usage(
            ResponseUsage(
                input_tokens=1000,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=900, cache_write_tokens=200
                ),
                output_tokens=40,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=1040,
            ),
            _PRICING,
        )


def test_cost_prices_each_cache_tier_and_the_remainder() -> None:
    """Cache reads, cache writes, the uncached remainder, and output each bill at their rate."""
    cost = _cost_in_usd(_usage_with_cache(), _PRICING)
    expected = (300 * 2.5 + 600 * 1.25 + 100 * 3.125 + 40 * 10.0) / 1e6
    assert abs(cost - expected) < 1e-12


def test_cost_breakdown_splits_categories_and_matches_the_stored_cost() -> None:
    """Each category cost is its own product, the 1h slot is 0, and the total equals the stored cost."""
    usage = _usage_with_cache()
    breakdown = cost_breakdown(usage, _PRICING)
    assert breakdown.counts.input_tokens_cache_none == 300
    assert breakdown.counts.input_tokens_cache_read == 600
    assert breakdown.counts.input_tokens_cache_write == 100
    assert breakdown.counts.input_tokens_cache_write_1h == 0
    assert breakdown.counts.output_tokens == 40
    assert breakdown.input_tokens_cache_none_cost_in_usd == 300 * 2.5 / 1e6
    assert breakdown.input_tokens_cache_read_cost_in_usd == 600 * 1.25 / 1e6
    assert breakdown.input_tokens_cache_write_cost_in_usd == 100 * 3.125 / 1e6
    assert breakdown.input_tokens_cache_write_1h_cost_in_usd == 0.0
    assert breakdown.output_tokens_cost_in_usd == 40 * 10.0 / 1e6
    assert breakdown.total_cost_in_usd == _normalized_usage(usage, _PRICING).cost_in_usd


def test_stop_reason_is_tool_use_with_a_function_call_item() -> None:
    """Any function_call output item derives tool_use, whatever the status."""
    response = _response(
        usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]
    )
    assert _normalized_stop_reason(response) == "tool_use"


def test_stop_reason_completed_is_end_turn() -> None:
    """Status completed without tool calls derives end_turn."""
    assert _normalized_stop_reason(_response(usage=None)) == "end_turn"


def test_stop_reason_refusal_block_is_refusal() -> None:
    """A refusal content block derives refusal, ahead of the status and tool-call checks."""
    refusal_message: dict[str, object] = {
        "type": "message",
        "id": "m1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I can't help with that"}],
    }
    assert _normalized_stop_reason(_response(usage=None, output=[refusal_message])) == "refusal"


def test_stop_reason_incomplete_for_max_output_tokens_is_max_tokens() -> None:
    """Status incomplete with reason max_output_tokens derives max_tokens."""
    response = _response(
        usage=None,
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="max_output_tokens"),
    )
    assert _normalized_stop_reason(response) == "max_tokens"


def test_stop_reason_other_statuses_are_other() -> None:
    """Content-filter incompleteness and failure both derive other."""
    content_filtered = _response(
        usage=None,
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="content_filter"),
    )
    assert _normalized_stop_reason(content_filtered) == "other"
    assert _normalized_stop_reason(_response(usage=None, status="failed")) == "other"


def test_assistant_message_collects_text_and_tool_calls() -> None:
    """The assistant turn carries the concatenated text and every function call."""
    response = _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM])
    assistant_message = _assistant_message_from(response)
    assert assistant_message.text == "hey"
    assert assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )


def test_reasoning_round_trips_verbatim_in_position() -> None:
    """A reasoning item round-trips verbatim and in its original position.

    Produce yields one ReasoningTrace where the reasoning item sat.
    Consume re-emits the stored dict unchanged, in the same position, with one input item per modeled output item.
    """
    response = _response(
        usage=None,
        output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM],
    )
    assistant_message = _assistant_message_from(response)
    assert [type(element) for element in assistant_message.turn] == [
        ReasoningTrace,
        TextPart,
        ToolCall,
    ]
    reasoning_trace = assistant_message.turn[0]
    assert isinstance(reasoning_trace, ReasoningTrace)
    assert reasoning_trace.reasoning == _REASONING_OUTPUT_ITEM
    assert assistant_message.text == "hey"
    assert assistant_message.tool_calls == (
        ToolCall(id="call1", name="lookup", args_json='{"q": 1}'),
    )
    items = _assistant_items(assistant_message)
    assert len(items) == len(response.output)
    assert items[0] == reasoning_trace.reasoning
    assert items[1] == {"role": "assistant", "content": "hey"}
    assert items[2] == {
        "type": "function_call",
        "call_id": "call1",
        "name": "lookup",
        "arguments": '{"q": 1}',
    }


def test_two_text_parts_stay_split_on_produce_and_rejoin_into_one_message_item() -> None:
    """A message item with two text parts yields two adjacent TextParts.

    On consume, the maximal adjacent run re-joins into one assistant message item.
    """
    two_part_message: dict[str, object] = {
        "type": "message",
        "id": "m1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "he", "annotations": []},
            {"type": "output_text", "text": "y", "annotations": []},
        ],
    }
    assistant_message = _assistant_message_from(_response(usage=None, output=[two_part_message]))
    assert assistant_message.turn == (TextPart(text="he"), TextPart(text="y"))
    assert _assistant_items(assistant_message) == [{"role": "assistant", "content": "hey"}]


def test_reasoning_with_a_key_the_installed_sdk_lacks_survives_the_wire_builder() -> None:
    """A stored dict carrying a field newer than the installed SDK param re-emits unchanged.

    A consume step that reshaped the dict to the pinned param keys would corrupt the payload
    the API re-reads across an SDK upgrade.
    """
    reasoning: dict[str, object] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "enc-1",
        "field_newer_than_sdk": "x",
    }
    assistant_message = AssistantMessage(turn=(ReasoningTrace(reasoning=reasoning),))
    assert _assistant_items(assistant_message) == [reasoning]


def test_foreign_reasoning_goes_to_the_wire_unchanged() -> None:
    """An anthropic-produced trace emits its dict as-is; the API rejects the unknown type key, not this adapter."""
    reasoning: dict[str, object] = {"type": "thinking", "thinking": "t", "signature": "s"}
    assistant_message = AssistantMessage(
        turn=(ReasoningTrace(reasoning=reasoning), TextPart(text="hi"))
    )
    assert _assistant_items(assistant_message) == [
        reasoning,
        {"role": "assistant", "content": "hi"},
    ]


def test_provider_result_normalizes_a_response_with_usage() -> None:
    """A response with usage yields the normalized partition, cost, and stop reason.

    raw must be the SDK response object itself (identity, not equality):
    an equal copy would silently reintroduce the per-request deep copy the no-rewrap rule bans.
    """
    response = _response(usage=_usage_with_cache())
    result = _provider_result(response=response, output="hey", pricing=_PRICING)
    assert result.output == "hey"
    assert result.usage.input_tokens_total == 1000
    assert result.usage.cost_in_usd == _cost_in_usd(_usage_with_cache(), _PRICING)
    assert result.usage_raw is response.usage
    assert result.stop_reason == "end_turn"
    assert result.raw is response


def test_provider_result_falls_back_to_zero_usage_without_usage() -> None:
    """A response missing usage normalizes to zero counters and zero cost."""
    result = _provider_result(response=_response(usage=None), output="hey", pricing=_PRICING)
    assert result.usage.input_tokens_total == 0
    assert result.usage.output_tokens == 0
    assert result.usage.cost_in_usd == 0.0
    assert result.usage_raw is None


def test_wire_input_converts_each_message_kind() -> None:
    """User, assistant (text plus tool calls), and tool messages each map to their items."""
    wire = _wire_input([
        UserMessage(content="q"),
        AssistantMessage(
            turn=(
                TextPart(text="thinking"),
                ToolCall(id="call1", name="lookup", args_json="{}"),
            ),
        ),
        ToolMessage(tool_call_id="call1", content="r"),
    ])
    assert wire == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking"},
        {"type": "function_call", "call_id": "call1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call1", "output": "r"},
    ]


def test_wire_input_converts_tool_result_parts_to_structured_output_content() -> None:
    """A ToolMessage carrying parts becomes a function_call_output whose output is the content list.

    The installed openai SDK's output field accepts input_text and input_image content params,
    so an image reaches the provider as a data: URI.
    A dropped part or mis-encoded image changes this list.
    """
    wire = _wire_input([
        ToolMessage(
            tool_call_id="call1",
            content=(TextPart(text="saw"), ImagePart(data=b"png", media_type="image/png")),
        )
    ])
    assert wire == [
        {
            "type": "function_call_output",
            "call_id": "call1",
            "output": [
                {"type": "input_text", "text": "saw"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}",
                    "detail": "auto",
                },
            ],
        }
    ]


def test_wire_input_has_no_system_item() -> None:
    """The system prompt travels as the instructions parameter, never as an item."""
    request = _provider()._request(_binding(automatic_prompt_caching=True, system_prompt="sys"))
    assert request.instructions == "sys"
    assert _wire_input([UserMessage(content="q")]) == [{"role": "user", "content": "q"}]


def test_wire_tool_choice_passes_strings_through_and_names_specific_tools() -> None:
    """The neutral strings pass through unchanged; SpecificTool becomes the function form."""
    assert _wire_tool_choice("auto") == "auto"
    assert _wire_tool_choice("required") == "required"
    assert _wire_tool_choice("none") == "none"
    assert _wire_tool_choice(SpecificTool(tool_name="x")) == {"type": "function", "name": "x"}


def _provider() -> OpenAIResponsesProvider:
    """Build an adapter over a keyless client, valid because no request is sent."""
    return OpenAIResponsesProvider(
        client=AsyncOpenAI(api_key="test"), model="m", pricing=_PRICING
    )


def _binding(
    *,
    automatic_prompt_caching: bool,
    system_prompt: str | tuple[TextPart, ...] | None = None,
) -> Binding:
    """Assemble a toolless binding varying only caching and the system prompt."""
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(),
        automatic_prompt_caching=automatic_prompt_caching,
    )


def test_request_omits_prompt_cache_options_under_automatic_caching() -> None:
    """Automatic caching leaves prompt_cache_options at the omit sentinel."""
    request = _provider()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(request.prompt_cache_options, openai.Omit)


def test_request_requests_explicit_mode_when_caching_disabled() -> None:
    """Disabled caching sends explicit mode with no breakpoints."""
    request = _provider()._request(_binding(automatic_prompt_caching=False))
    assert request.prompt_cache_options == {"mode": "explicit"}


def test_request_maps_temperature_and_omits_it_when_unset() -> None:
    """A bound temperature lands on the request; None leaves the omit sentinel."""
    unset = _provider()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(unset.temperature, openai.Omit)
    binding = Binding(
        system_prompt=None,
        tool_schemas=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        inference_params=InferenceParams(temperature=0.2),
        automatic_prompt_caching=True,
    )
    assert _provider()._request(binding).temperature == 0.2


def test_request_omits_tool_fields_without_tools() -> None:
    """No tools leaves tools, tool_choice, and parallel_tool_calls at the omit sentinel."""
    request = _provider()._request(_binding(automatic_prompt_caching=True))
    assert isinstance(request.tools, openai.Omit)
    assert isinstance(request.tool_choice, openai.Omit)
    assert isinstance(request.parallel_tool_calls, openai.Omit)


class _FakeSDKStream(AsyncResponseStream[None]):
    """Replays constructed events without a connection.

    Overrides exactly the surface _OpenAIStream uses (iteration and close);
    the base __init__ is deliberately not called, so the untouched base machinery stays unusable.
    """

    def __init__(self, replay_events: Sequence[ResponseStreamEvent]) -> None:
        self._replay_events = list(replay_events)

    @override
    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        for replay_event in self._replay_events:
            yield replay_event

    @override
    async def close(self) -> None:
        return


def _stream(replay_events: Sequence[ResponseStreamEvent]) -> _OpenAIStream[str]:
    """Build a text-content adapter stream over replayed events."""
    return _OpenAIStream(
        sdk_stream=_FakeSDKStream(replay_events),
        pricing=_PRICING,
        output_from_response=lambda response: response.output_text,
    )


def _collected_items(replay_events: Sequence[ResponseStreamEvent]) -> list[StreamItem]:
    """Drain the translated items into a list."""

    async def scenario() -> list[StreamItem]:
        return [item async for item in _stream(replay_events).items()]

    return asyncio.run(scenario())


def _text_delta_event(delta: str, sequence_number: int) -> AccumulatedResponseTextDeltaEvent:
    """Build one accumulated text-delta event."""
    return AccumulatedResponseTextDeltaEvent(
        type="response.output_text.delta",
        delta=delta,
        snapshot=delta,
        content_index=0,
        item_id="m1",
        output_index=0,
        logprobs=[],
        sequence_number=sequence_number,
    )


def _completed_event(
    response: OpenAIResponse, sequence_number: int
) -> AccumulatedResponseCompletedEvent[None]:
    """Wrap a response in the terminal completed event the SDK stream yields."""
    return AccumulatedResponseCompletedEvent[None](
        type="response.completed",
        response=ParsedResponse[None].model_validate(response.model_dump()),
        sequence_number=sequence_number,
    )


def test_stream_passes_text_deltas_through_as_bare_strings() -> None:
    """Text deltas pass through in order as the SDK's own strings; nothing follows them."""
    translated = _collected_items([
        _text_delta_event("he", 1),
        _text_delta_event("y", 2),
        _completed_event(_response(usage=_usage_with_cache()), 3),
    ])
    assert translated == ["he", "y"]


def test_stream_yields_one_complete_tool_call_and_ignores_message_items() -> None:
    """A function_call done event yields one complete ToolCall; message item lifecycles are dropped."""
    message_added = ResponseOutputItemAddedEvent.model_validate({
        "type": "response.output_item.added",
        "item": _TEXT_OUTPUT_ITEM,
        "output_index": 0,
        "sequence_number": 1,
    })
    function_call_added = ResponseOutputItemAddedEvent.model_validate({
        "type": "response.output_item.added",
        "item": _FUNCTION_CALL_OUTPUT_ITEM,
        "output_index": 1,
        "sequence_number": 2,
    })
    message_done = ResponseOutputItemDoneEvent.model_validate({
        "type": "response.output_item.done",
        "item": _TEXT_OUTPUT_ITEM,
        "output_index": 0,
        "sequence_number": 3,
    })
    function_call_done = ResponseOutputItemDoneEvent.model_validate({
        "type": "response.output_item.done",
        "item": _FUNCTION_CALL_OUTPUT_ITEM,
        "output_index": 1,
        "sequence_number": 4,
    })
    translated = _collected_items([
        message_added,
        function_call_added,
        message_done,
        function_call_done,
        _completed_event(
            _response(usage=None, output=[_TEXT_OUTPUT_ITEM, _FUNCTION_CALL_OUTPUT_ITEM]), 5
        ),
    ])
    assert translated == [ToolCall(id="call1", name="lookup", args_json='{"q": 1}')]


def test_stream_incomplete_terminal_still_assembles_final() -> None:
    """An incomplete terminal yields no item, and final() must not raise.

    The SDK's get_final_response() raises RuntimeError unless the terminal event is response.completed,
    so final() assembles from the captured terminal response instead.
    """

    async def scenario() -> None:
        incomplete_response = _response(
            usage=_usage_with_cache(),
            status="incomplete",
            incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        )
        adapter_stream = _stream([
            _text_delta_event("he", 1),
            ResponseIncompleteEvent(
                type="response.incomplete", response=incomplete_response, sequence_number=2
            ),
        ])
        translated = [item async for item in adapter_stream.items()]
        assert translated == ["he"]
        result = await adapter_stream.final()
        assert result.output == "hey"
        assert result.stop_reason == "max_tokens"
        assert result.usage.input_tokens_total == 1000

    asyncio.run(scenario())


def test_final_after_completed_terminal_assembles_from_the_parsed_response() -> None:
    """The completed terminal already carries a ParsedResponse; final() assembles from it."""

    async def scenario() -> None:
        adapter_stream = _stream([
            _text_delta_event("he", 1),
            _completed_event(_response(usage=_usage_with_cache()), 2),
        ])
        translated = [item async for item in adapter_stream.items()]
        assert translated == ["he"]
        result = await adapter_stream.final()
        assert result.output == "hey"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens_total == 1000
        assert result.usage.cost_in_usd == _cost_in_usd(_usage_with_cache(), _PRICING)

    asyncio.run(scenario())


def test_stream_final_turn_carries_reasoning() -> None:
    """final()'s assistant turn includes the ReasoningTrace from the terminal response's output."""

    async def scenario() -> None:
        adapter_stream = _stream([
            _completed_event(
                _response(usage=None, output=[_REASONING_OUTPUT_ITEM, _TEXT_OUTPUT_ITEM]), 1
            ),
        ])
        async for _item in adapter_stream.items():
            pass
        result = await adapter_stream.final()
        reasoning_trace = result.assistant_message.turn[0]
        assert isinstance(reasoning_trace, ReasoningTrace)
        assert reasoning_trace.reasoning == _REASONING_OUTPUT_ITEM

    asyncio.run(scenario())


def test_stream_failed_terminal_is_terminal() -> None:
    """A failed terminal is terminal: no StreamProtocolError, and final() reports other."""

    async def scenario() -> None:
        adapter_stream = _stream([
            ResponseFailedEvent(
                type="response.failed",
                response=_response(usage=None, status="failed"),
                sequence_number=1,
            ),
        ])
        translated = [item async for item in adapter_stream.items()]
        assert translated == []
        result = await adapter_stream.final()
        assert result.stop_reason == "other"

    asyncio.run(scenario())


def test_stream_without_terminal_raises() -> None:
    """Ending without any terminal event is a protocol violation."""
    with pytest.raises(StreamProtocolError):
        _collected_items([_text_delta_event("he", 1)])


def test_final_before_items_are_exhausted_raises() -> None:
    """final() needs the captured terminal response, so it demands drained items."""

    async def scenario() -> None:
        adapter_stream = _stream([_completed_event(_response(usage=None), 1)])
        with pytest.raises(StreamProtocolError):
            await adapter_stream.final()

    asyncio.run(scenario())


class _StructuredReport(BaseModel):
    """The response_format the structured bind path parses into."""

    city: str
    celsius: int


def _structured_bound() -> _BoundOpenAIStructured[_StructuredReport]:
    """Build a structured-bound provider over a keyless client; no request is sent."""
    provider = _provider()
    request = provider._request(_binding(automatic_prompt_caching=False, system_prompt="sys"))
    return _BoundOpenAIStructured(
        adapter=provider, request=request, response_format=_StructuredReport
    )


def _parsed_response(
    parsed: _StructuredReport | None,
    *,
    refuse: bool = False,
    status: ResponseStatus = "completed",
    incomplete_details: IncompleteDetails | None = None,
    usage: ResponseUsage | None = None,
) -> ParsedResponse[_StructuredReport]:
    """Build the SDK parse result whose message carries the parsed instance, or a refusal block."""
    content = (
        [ResponseOutputRefusal(type="refusal", refusal="I can't help with that")]
        if refuse
        else [
            ParsedResponseOutputText[_StructuredReport](
                type="output_text", text="{}", annotations=[], parsed=parsed
            )
        ]
    )
    message = ParsedResponseOutputMessage[_StructuredReport](
        id="m1", role="assistant", status="completed", type="message", content=content
    )
    return ParsedResponse[_StructuredReport](
        id="r1",
        created_at=0,
        model="m",
        object="response",
        output=[message],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status=status,
        incomplete_details=incomplete_details,
        usage=usage,
    )


def test_structured_bind_returns_the_sdk_parsed_instance() -> None:
    """The structured bound provider hands back the SDK-parsed response_format instance as content."""
    report = _StructuredReport(city="Nairobi", celsius=25)
    assert _structured_bound()._parsed_output(_parsed_response(report)) == report


def test_structured_bind_raises_transient_without_parsed_output() -> None:
    """A turn with no parsed output is transient: a later attempt may still produce it."""
    with pytest.raises(TransientError) as raised:
        _structured_bound()._parsed_output(_parsed_response(None))
    assert raised.value.stop_reason == "end_turn"


def test_structured_bind_raises_refusal_on_a_refusal_block() -> None:
    """A response carrying a refusal content block is the terminal refusal leaf, carrying its billing."""
    with pytest.raises(RefusalError) as raised:
        _structured_bound()._parsed_output(
            _parsed_response(None, refuse=True, usage=_usage_with_cache())
        )
    assert raised.value.stop_reason == "refusal"
    assert raised.value.usage.cost_in_usd > 0.0


def test_structured_bind_raises_truncation_on_a_max_output_tokens_incomplete() -> None:
    """An incomplete response for max_output_tokens is the terminal truncation leaf."""
    with pytest.raises(ExceededMaxCompletionTokensError) as raised:
        _structured_bound()._parsed_output(
            _parsed_response(
                None,
                status="incomplete",
                incomplete_details=IncompleteDetails(reason="max_output_tokens"),
                usage=_usage_with_cache(),
            )
        )
    assert raised.value.stop_reason == "max_tokens"
    assert raised.value.usage.cost_in_usd > 0.0


def test_every_request_carries_the_reasoning_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create, parse, and stream (text and structured) all send include=["reasoning.encrypted_content"].

    The offline round-trip tests cannot catch a dropped include:
    the API populates encrypted_content only when asked,
    so without this parameter every replayed reasoning item would be silently empty.
    """
    provider = _provider()
    request = provider._request(_binding(automatic_prompt_caching=True))
    includes: list[object] = []

    async def fake_create(**request_kwargs: object) -> OpenAIResponse:
        includes.append(request_kwargs["include"])
        return _response(usage=None)

    async def fake_parse(**request_kwargs: object) -> ParsedResponse[_StructuredReport]:
        includes.append(request_kwargs["include"])
        return _parsed_response(_StructuredReport(city="Nairobi", celsius=25))

    class _FakeStreamManager:
        async def __aenter__(self) -> _FakeSDKStream:
            return _FakeSDKStream([])

    def fake_stream(**request_kwargs: object) -> _FakeStreamManager:
        includes.append(request_kwargs["include"])
        return _FakeStreamManager()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)
    monkeypatch.setattr(provider.client.responses, "parse", fake_parse)
    monkeypatch.setattr(provider.client.responses, "stream", fake_stream)
    text_bound = _BoundOpenAIText(adapter=provider, request=request)
    structured_bound = _BoundOpenAIStructured(
        adapter=provider, request=request, response_format=_StructuredReport
    )

    async def scenario() -> None:
        conversation = [UserMessage(content="q")]
        await text_bound.send(conversation)
        await text_bound.open_stream(conversation)
        await structured_bound.send(conversation)
        await structured_bound.open_stream(conversation)

    asyncio.run(scenario())
    assert includes == [["reasoning.encrypted_content"]] * 4


def _rate_limit_error(headers: dict[str, str]) -> openai.RateLimitError:
    """Build the SDK's 429 exception around a constructed httpx response."""
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_retry_after_seconds_prefers_the_millisecond_header() -> None:
    """retry-after-ms wins over retry-after because it is more precise."""
    error = _rate_limit_error({"retry-after-ms": "1500", "retry-after": "49"})
    assert _provider().retry_after_seconds(error) == 1.5


def test_retry_after_seconds_parses_the_seconds_header() -> None:
    """Without retry-after-ms, retry-after is parsed as float seconds."""
    error = _rate_limit_error({"retry-after": "49"})
    assert _provider().retry_after_seconds(error) == 49.0


def test_retry_after_seconds_is_none_without_headers_or_status() -> None:
    """No headers, an unparseable value, and a non-SDK error all yield None."""
    provider = _provider()
    assert provider.retry_after_seconds(_rate_limit_error({})) is None
    assert (
        provider.retry_after_seconds(
            _rate_limit_error({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        )
        is None
    )
    assert provider.retry_after_seconds(ValueError("boom")) is None


def test_adapter_pins_sdk_retries_off() -> None:
    """The stored client copy carries max_retries=0 so only the package retries."""
    assert _provider().client.max_retries == 0


def test_wire_input_marks_marked_user_and_tool_parts() -> None:
    """A marked part carries prompt_cache_breakpoint on its wire part; unmarked siblings carry none."""
    wire = _wire_input([
        UserMessage(
            content=(TextPart(text="shared context", cache_breakpoint=True), TextPart(text="question"))
        ),
        ToolMessage(
            tool_call_id="c1",
            content=(
                TextPart(text="saw"),
                ImagePart(data=b"png", media_type="image/png", cache_breakpoint=True),
            ),
        ),
    ])
    assert wire == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "shared context",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "input_text", "text": "question"},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [
                {"type": "input_text", "text": "saw"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}",
                    "detail": "auto",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
            ],
        },
    ]


def test_wire_input_sends_every_mark_without_a_client_side_cap() -> None:
    """The server keeps the latest breakpoints itself, so all five marks go to the wire."""
    wire = _wire_input([
        UserMessage(
            content=tuple(TextPart(text=f"m{index}", cache_breakpoint=True) for index in range(5))
        ),
    ])
    assert wire == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"m{index}",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
                for index in range(5)
            ],
        }
    ]


def test_request_system_parts_become_a_developer_input_message() -> None:
    """A parts system_prompt travels as a developer-role input message; instructions stays unset."""
    request = _provider()._request(
        _binding(
            automatic_prompt_caching=True,
            system_prompt=(
                TextPart(text="stable instructions", cache_breakpoint=True),
                TextPart(text="semi-stable context"),
            ),
        )
    )
    assert request.instructions is None
    assert request.input_prefix == [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "stable instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {"type": "input_text", "text": "semi-stable context"},
            ],
        }
    ]


def test_request_str_system_travels_as_instructions_with_an_empty_prefix() -> None:
    """A str system_prompt keeps the instructions mapping and sends no prefix item."""
    request = _provider()._request(_binding(automatic_prompt_caching=True, system_prompt="sys"))
    assert request.instructions == "sys"
    assert request.input_prefix == []
