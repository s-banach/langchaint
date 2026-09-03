"""Verify that kind narrows each tagged union.

Each exhaustive match reads variant-specific fields.
Each partial match requires a non-exhaustive-match suppression.
Runtime tests exercise each branch of the match statements.
"""

from typing import assert_type

from langchaint import (
    AbandonedCallErrorRecord,
    AssistantMessage,
    AudioPart,
    ContentPart,
    ContextWindowExceededErrorRecord,
    CutOffAttemptRecord,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyOutcome,
    DispatchOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    DoNotRetry,
    EmptyTurnErrorRecord,
    EscapedExceptionErrorRecord,
    GenerateResult,
    GenerationError,
    GenerationErrorKind,
    GenerationErrorRecord,
    ImagePart,
    ImageUrlPart,
    InvalidRequestErrorRecord,
    InvalidToolArgsDetail,
    MaxCompletionTokensExceededErrorRecord,
    Message,
    PauseAll,
    PauseAllDoNotRetry,
    ProviderDeclaredFinalErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    RawPart,
    ReasoningDelta,
    ReasoningPart,
    RefusalErrorRecord,
    Response,
    ResponseRecord,
    RetriesExhaustedErrorRecord,
    RetryThisOne,
    RetryUnavailableErrorRecord,
    SchemaViolationErrorRecord,
    StreamItem,
    TextPart,
    TimedOutErrorRecord,
    ToolCall,
    ToolCallDelta,
    ToolCallTurn,
    ToolCallTurnRecord,
    ToolMessage,
    TransientErrorRecord,
    TurnPart,
    UnfinishedTurnErrorRecord,
    UnknownExceptionErrorRecord,
    UserMessage,
    Verdict,
)
from langchaint.adapter import (
    AdapterResult,
    ContextWindowExceeded,
    EmptyTurn,
    MaxCompletionTokensExceeded,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    Refusal,
    ResponseOutcome,
    SchemaViolation,
    UnfinishedTurn,
)
from langchaint.call import AttemptProviderData
from tests.helpers import StubRaw, attempt_record, call_record

_TURN = AssistantMessage(turn="hi")
_TOOL_MESSAGE = ToolMessage(tool_call_id="c1", content="ok")


_CALL = call_record((attempt_record(error=None),), elapsed_seconds=1.0)
"""Provide shared CallRecord data for GenerateResult values."""


def _by_message_kind(message: Message) -> object:
    match message.kind:
        case "user":
            return message.content
        case "assistant":
            return message.turn
        case "tool":
            return message.tool_call_id


def _by_message_kind_missing_a_variant(message: Message) -> object:
    match message.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "user":
            return message.content


def _by_content_part_kind(part: ContentPart) -> object:
    match part.kind:
        case "text":
            return part.text
        case "image":
            return part.media_type
        case "image_url":
            return part.url
        case "audio":
            return part.data


def _by_content_part_kind_missing_a_variant(part: ContentPart) -> object:
    match part.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "text":
            return part.text


def _by_turn_part_kind(part: TurnPart) -> object:
    match part.kind:
        case "reasoning_part":
            return part.text
        case "text":
            return part.cache_breakpoint
        case "tool_call":
            return part.args_json
        case "raw_part":
            return part.raw


def _by_turn_part_kind_missing_a_variant(part: TurnPart) -> object:
    match part.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "reasoning_part":
            return part.raw


def _by_dispatch_outcome_kind(outcome: DispatchOutcome) -> object:
    match outcome.kind:
        case "handled":
            return outcome.app_data
        case "invalid_tool_args":
            return outcome.details
        case "unknown_tool":
            return outcome.called_name


def _by_dispatch_outcome_kind_missing_a_variant(outcome: DispatchOutcome) -> object:
    match outcome.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "handled":
            return outcome.app_data


def _by_dispatch_many_outcome_kind(outcome: DispatchManyOutcome) -> object:
    match outcome.kind:
        case "handled":
            return outcome.app_data
        case "invalid_tool_args":
            return outcome.details
        case "unknown_tool":
            return outcome.called_name
        case "precomputed":
            return outcome.tool_message


def _by_dispatch_many_outcome_kind_missing_a_variant(outcome: DispatchManyOutcome) -> object:
    match outcome.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "handled":
            return outcome.app_data


def _by_response_outcome_kind(outcome: ResponseOutcome[str]) -> object:
    """Exercise an exhaustive match on ResponseOutcome."""
    match outcome.kind:
        case "adapter_result":
            return outcome.output
        case "refusal":
            return outcome.assistant_message
        case "max_completion_tokens_exceeded":
            return outcome.assistant_message
        case "empty_turn":
            return outcome.assistant_message
        case "context_window_exceeded":
            return outcome.assistant_message
        case "schema_violation":
            return outcome.validation_error_json
        case "unfinished_turn":
            return outcome.reason
        case "provider_failed_terminally":
            return outcome.reason
        case "provider_failed_transiently":
            return outcome.is_rate_limit


def _by_response_outcome_kind_missing_a_variant(outcome: ResponseOutcome[str]) -> object:
    match outcome.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "adapter_result":
            return outcome.output


def _by_generate_result_kind(result: GenerateResult[int]) -> object:
    """Verify that kind narrows GenerateResult.output."""
    match result.kind:
        case "response":
            assert_type(result.output, int)
            return result.output
        case "tool_call_turn":
            assert_type(result.output, int | None)
            return result.tool_calls


def _by_generate_result_kind_missing_a_variant(result: GenerateResult[int]) -> object:
    match result.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "response":
            return result.output


def _by_generation_error_record_kind(  # noqa: PLR0911 (each discriminator requires one branch)
    record: GenerationErrorRecord,
) -> object:
    """Verify that `kind` narrows every normalized generation error record."""
    match record.kind:
        case "retries_exhausted_error":
            assert_type(record, RetriesExhaustedErrorRecord)
            return record.errors_from_attempts
        case "retry_unavailable_error":
            assert_type(record, RetryUnavailableErrorRecord)
            return record.error_text
        case "refusal_error":
            assert_type(record, RefusalErrorRecord)
            return record.stop_reason
        case "max_completion_tokens_exceeded_error":
            assert_type(record, MaxCompletionTokensExceededErrorRecord)
            return record.stop_reason
        case "empty_turn_error":
            assert_type(record, EmptyTurnErrorRecord)
            return record.stop_reason
        case "schema_violation_error":
            assert_type(record, SchemaViolationErrorRecord)
            return record.validation_error_json
        case "context_window_exceeded_error":
            assert_type(record, ContextWindowExceededErrorRecord)
            return record.stop_reason
        case "unfinished_turn_error":
            assert_type(record, UnfinishedTurnErrorRecord)
            return record.error_text
        case "provider_failed_terminally_error":
            assert_type(record, ProviderFailedTerminallyErrorRecord)
            return record.error_text
        case "invalid_request_error":
            assert_type(record, InvalidRequestErrorRecord)
            return record.error_text
        case "provider_declared_final_error":
            assert_type(record, ProviderDeclaredFinalErrorRecord)
            return record.error_text
        case "unknown_exception_error":
            assert_type(record, UnknownExceptionErrorRecord)
            return record.error_text
        case "escaped_exception_error":
            assert_type(record, EscapedExceptionErrorRecord)
            return record.error_text
        case "abandoned_call_error":
            assert_type(record, AbandonedCallErrorRecord)
            return record.attempts
        case "timed_out_error":
            assert_type(record, TimedOutErrorRecord)
            return record.attempts


def _by_stream_item_kind(item: StreamItem) -> object:
    """Exercise an exhaustive match on StreamItem."""
    if isinstance(item, str):
        return item
    match item.kind:
        case "reasoning_delta":
            return item.text
        case "tool_call_delta":
            return item.partial_args_json
        case "tool_call":
            return item.args_json


def _by_stream_item_kind_missing_a_variant(item: StreamItem) -> object:
    if isinstance(item, str):
        return item
    match item.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "reasoning_delta":
            return item.text


def _by_verdict_kind(verdict: Verdict) -> object:
    """Cover Verdict, whose DoNotRetry variant is the one carrying no retry_after."""
    match verdict.kind:
        case "pause_all":
            return verdict.retry_after
        case "pause_all_do_not_retry":
            return verdict.retry_after
        case "retry_this_one":
            return verdict.retry_after
        case "do_not_retry":
            return verdict.kind


def _by_verdict_kind_missing_a_variant(verdict: Verdict) -> object:
    match verdict.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "pause_all":
            return verdict.retry_after


def test_a_message_kind_selects_the_variant_that_carries_the_field_read() -> None:
    """Each Message variant's tag reaches a field the other variants do not carry."""
    assert _by_message_kind(UserMessage(content="hi")) == "hi"
    assert _by_message_kind(_TURN) == (TextPart(text="hi"),)
    assert _by_message_kind(_TOOL_MESSAGE) == "c1"
    assert _by_message_kind_missing_a_variant(_TOOL_MESSAGE) is None


def test_a_content_part_kind_selects_the_variant_that_carries_the_field_read() -> None:
    """Each ContentPart tag reaches a variant-specific field."""
    assert _by_content_part_kind(TextPart(text="hi")) == "hi"
    assert _by_content_part_kind(ImagePart(data=b"png", media_type="image/png")) == "image/png"
    assert _by_content_part_kind(ImageUrlPart(url="https://example.com/a.png")) == (
        "https://example.com/a.png"
    )
    assert _by_content_part_kind(AudioPart(data=b"wav", media_type="audio/wav")) == b"wav"
    assert (
        _by_content_part_kind_missing_a_variant(ImagePart(data=b"png", media_type="image/png"))
        is None
    )


def test_a_turn_part_kind_selects_the_variant_that_carries_the_field_read() -> None:
    """Each TurnPart tag reaches a variant-specific field."""
    assert _by_turn_part_kind(ReasoningPart(raw={"id": "rs_1"}, text="hm")) == "hm"
    assert _by_turn_part_kind(TextPart(text="hi")) is False
    assert _by_turn_part_kind(ToolCall(id="c1", name="probe", args_json="{}")) == "{}"
    assert _by_turn_part_kind(RawPart(raw={"id": "ws_1"})) == {"id": "ws_1"}
    tool_call = ToolCall(id="c1", name="probe", args_json="{}")
    assert _by_turn_part_kind_missing_a_variant(tool_call) is None


def test_a_dispatch_outcome_kind_selects_the_variant_that_carries_the_field_read() -> None:
    """Each DispatchOutcome variant's tag reaches a field the other variants do not carry."""
    detail = InvalidToolArgsDetail(path=("city",), message="required")
    unknown_tool = DispatchUnknownTool(tool_message=_TOOL_MESSAGE, called_name="off_list")
    handled = DispatchHandled(tool_message=_TOOL_MESSAGE, app_data={"seen": 1})
    assert _by_dispatch_outcome_kind(handled) == {"seen": 1}
    assert _by_dispatch_outcome_kind(
        DispatchInvalidToolArgs(tool_message=_TOOL_MESSAGE, details=(detail,))
    ) == (detail,)
    assert _by_dispatch_outcome_kind(unknown_tool) == "off_list"
    assert _by_dispatch_outcome_kind_missing_a_variant(unknown_tool) is None


def test_a_dispatch_many_outcome_kind_selects_its_own_extra_variant() -> None:
    """DispatchManyOutcome's precomputed variant has its own tag, and the shared variants keep theirs."""
    precomputed = DispatchPrecomputed(tool_message=_TOOL_MESSAGE)
    assert _by_dispatch_many_outcome_kind(precomputed) is _TOOL_MESSAGE
    assert (
        _by_dispatch_many_outcome_kind(
            DispatchUnknownTool(tool_message=_TOOL_MESSAGE, called_name="off_list")
        )
        == "off_list"
    )
    assert _by_dispatch_many_outcome_kind_missing_a_variant(precomputed) is None


def test_a_response_outcome_kind_reaches_a_case_that_reads_a_field_its_variant_carries() -> None:
    """Each ResponseOutcome variant reaches a case that reads a field the variant carries."""
    outcomes: list[tuple[ResponseOutcome[str], object]] = [
        (AdapterResult(output="hi", assistant_message=_TURN, stop_reason="end_turn"), "hi"),
        (Refusal(assistant_message=_TURN), _TURN),
        (MaxCompletionTokensExceeded(assistant_message=_TURN), _TURN),
        (EmptyTurn(assistant_message=_TURN), _TURN),
        (ContextWindowExceeded(assistant_message=_TURN), _TURN),
        (SchemaViolation(assistant_message=_TURN, validation_error_json="[]"), "[]"),
        (UnfinishedTurn(assistant_message=_TURN, reason="paused"), "paused"),
        (ProviderFailedTerminally(assistant_message=_TURN, reason="overloaded"), "overloaded"),
        (
            ProviderFailedTransiently(assistant_message=_TURN, reason="busy", is_rate_limit=True),
            True,
        ),
    ]
    assert [_by_response_outcome_kind(outcome) for outcome, _ in outcomes] == [
        expected for _, expected in outcomes
    ]
    assert _by_response_outcome_kind_missing_a_variant(Refusal(assistant_message=_TURN)) is None


def test_a_generate_result_kind_narrows_the_output_type_the_variants_share_a_name_for() -> None:
    """The response case returns the non-optional output. The tool_call_turn case reads tool_calls."""
    tool_call = ToolCall(id="c1", name="probe", args_json="{}")
    tool_call_message = AssistantMessage(turn=(tool_call,))
    provider_attempts = (AttemptProviderData(raw=StubRaw(), usage_raw=None),)
    tool_call_turn: ToolCallTurn[int] = ToolCallTurn(
        record=ToolCallTurnRecord(
            output=None,
            call=call_record(
                (attempt_record(error=None, turn=tool_call_message),), elapsed_seconds=1.0
            ),
            stop_reason="tool_use",
        ),
        provider_attempts=provider_attempts,
    )
    response = Response(
        record=ResponseRecord(
            output=7,
            call=call_record((attempt_record(error=None, turn=_TURN),), elapsed_seconds=1.0),
            stop_reason="end_turn",
        ),
        provider_attempts=provider_attempts,
    )
    assert _by_generate_result_kind(response) == 7
    assert _by_generate_result_kind(tool_call_turn) == (tool_call,)
    assert _by_generate_result_kind_missing_a_variant(tool_call_turn) is None


def test_a_generation_error_record_kind_narrows_every_record_variant() -> None:
    """Each `GenerationErrorRecord.kind` branch reads its narrowed record type."""
    failed_call = call_record(
        (attempt_record(error=TransientErrorRecord(message="retry"), turn=None),),
        elapsed_seconds=1.0,
    )
    completed_call = call_record((attempt_record(error=None, turn=_TURN),), elapsed_seconds=1.0)
    terminal_call = call_record((attempt_record(error=None, turn=None),), elapsed_seconds=1.0)
    empty_call = call_record((), elapsed_seconds=0.0)
    cut_off_call = call_record(
        (CutOffAttemptRecord(started_after_seconds=0.0, billing=None),),
        elapsed_seconds=1.0,
    )
    records: tuple[GenerationErrorRecord, ...] = (
        RetriesExhaustedErrorRecord(call=failed_call),
        RetryUnavailableErrorRecord(call=failed_call),
        RefusalErrorRecord(call=completed_call),
        MaxCompletionTokensExceededErrorRecord(call=completed_call),
        EmptyTurnErrorRecord(call=completed_call),
        SchemaViolationErrorRecord(call=completed_call, validation_error_json="[]"),
        ContextWindowExceededErrorRecord(call=completed_call),
        UnfinishedTurnErrorRecord(call=completed_call, error_text="unfinished"),
        ProviderFailedTerminallyErrorRecord(call=completed_call, error_text="failed"),
        InvalidRequestErrorRecord(call=empty_call, error_text="invalid"),
        ProviderDeclaredFinalErrorRecord(call=terminal_call, error_text="terminal"),
        UnknownExceptionErrorRecord(call=empty_call, error_text="unknown"),
        EscapedExceptionErrorRecord(call=empty_call, error_text="escaped"),
        AbandonedCallErrorRecord(call=cut_off_call),
        TimedOutErrorRecord(call=cut_off_call),
    )
    assert all(_by_generation_error_record_kind(record) is not None for record in records)


def test_generation_error_kind_has_the_public_closed_type() -> None:
    """GenerationError.kind exposes the record category through GenerationErrorKind."""
    error = GenerationError(
        record=InvalidRequestErrorRecord(call=call_record((), elapsed_seconds=0.0), error_text=""),
        request=None,
        provider_attempts=(),
    )
    assert_type(error.kind, GenerationErrorKind)
    assert error.kind == "invalid_request_error"


def test_a_verdict_kind_selects_the_variant_that_carries_the_field_read() -> None:
    """Three variants' tags reach retry_after, which DoNotRetry alone does not carry."""
    assert _by_verdict_kind(PauseAll(retry_after=7.0)) == 7.0
    assert _by_verdict_kind(PauseAllDoNotRetry(retry_after=5.0)) == 5.0
    assert _by_verdict_kind(RetryThisOne(retry_after=2.0)) == 2.0
    assert _by_verdict_kind(DoNotRetry()) == "do_not_retry"
    assert _by_verdict_kind_missing_a_variant(DoNotRetry()) is None


def test_a_stream_item_kind_selects_the_variant_that_carries_the_field_read() -> None:
    """A str is selected by isinstance, and the tag selects among the three classes."""
    tool_call = ToolCall(id="c1", name="probe", args_json="{}")
    assert _by_stream_item_kind("hi") == "hi"
    assert _by_stream_item_kind(ReasoningDelta(text="weighing")) == "weighing"
    assert (
        _by_stream_item_kind(ToolCallDelta(id="c1", name="probe", partial_args_json='{"de'))
        == '{"de'
    )
    assert _by_stream_item_kind(tool_call) == "{}"
    assert _by_stream_item_kind_missing_a_variant(tool_call) is None
