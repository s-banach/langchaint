"""Verify that kind narrows each tagged union.

Each exhaustive match reads variant-specific fields.
Each partial match requires a non-exhaustive-match suppression.
"""

from typing import assert_type

from langchaint import (
    AbandonedCallErrorRecord,
    ContentPart,
    ContextWindowExceededErrorRecord,
    DispatchManyOutcome,
    DispatchOutcome,
    EmptyTurnErrorRecord,
    EscapedExceptionErrorRecord,
    GenerateResult,
    GenerationErrorRecord,
    InvalidRequestErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    Message,
    ProviderDeclaredFinalErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    RefusalErrorRecord,
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    SchemaViolationErrorRecord,
    StreamItem,
    TimedOutErrorRecord,
    TurnPart,
    UnfinishedTurnErrorRecord,
    UnknownExceptionErrorRecord,
    Verdict,
)
from langchaint.adapter import (
    ResponseOutcome,
)


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
