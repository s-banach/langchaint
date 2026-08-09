"""The kind tag as a type-checked contract, on every union whose classes carry one.

Each union gets a match on kind whose cases read a field only some variants carry: that body fails to
type-check unless the tag narrows the subject, so a regressed discriminator is a check-time error
rather than a runtime AttributeError.
GenerateResult's variants share every field name, so its match asserts the narrowed type of output,
which only the tag makes non-optional in the response case.
Each union also gets a one-case match carrying a non-exhaustive-match suppression.
pyrefly reports an unused suppression as an error, so if the tag ever stops driving exhaustiveness
the suppression goes unused and the check fails; a union that gains a variant fails in the full match
beside it.
The full matches are exercised at runtime too, so a subject reaching no case fails an assertion.
That no two variants share a tag, which is what keeps a variant out of a sibling's case, is checked in
test_kind_tag_shape.py.
"""

from typing import assert_type

from langchaint import (
    AssistantMessage,
    ContentPart,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyOutcome,
    DispatchOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    DoNotRetry,
    GenerateResult,
    ImagePart,
    InvalidToolArgsDetail,
    Message,
    PauseAll,
    PauseAllDoNotRetry,
    RawPart,
    ReasoningDelta,
    ReasoningPart,
    Response,
    RetryThisOne,
    StreamItem,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolCallTurn,
    ToolMessage,
    TurnPart,
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
from tests.helpers import StubRaw, attempt_record, call_record

_TURN = AssistantMessage(turn="hi")
_TOOL_MESSAGE = ToolMessage(tool_call_id="c1", content="ok")


_CALL = call_record((attempt_record(error=None),), elapsed_seconds=1.0)
"""One successful attempt's history, the fixed filler both GenerateResult variants carry."""


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
    """Cover ResponseOutcome, the union every retry loop matches an attempt's outcome over.

    A match over it is exhaustive only if a match over NoOutputOutcome is, since those are its
    no-output variants.
    """
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
    """Cover GenerateResult, whose variants share every field name.

    The tag's work is output's optionality, so each case asserts the type the tag narrowed output
    to, standing where the other functions read a variant-only field.
    """
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


def _by_stream_item_kind(item: StreamItem) -> object:
    """Cover StreamItem, whose str variant carries no tag because a builtin cannot hold one."""
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
    """The response case returns the non-optional output; the tool_call_turn case reads tool_calls."""
    tool_call = ToolCall(id="c1", name="probe", args_json="{}")
    tool_call_turn: ToolCallTurn[int] = ToolCallTurn(
        output=None,
        call=_CALL,
        raw=StubRaw(),
        stop_reason="tool_use",
        assistant_message=AssistantMessage(turn=(tool_call,)),
    )
    response = Response(
        output=7, call=_CALL, raw=StubRaw(), stop_reason="end_turn", assistant_message=_TURN
    )
    assert _by_generate_result_kind(response) == 7
    assert _by_generate_result_kind(tool_call_turn) == (tool_call,)
    assert _by_generate_result_kind_missing_a_variant(tool_call_turn) is None


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
