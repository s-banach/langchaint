"""The kind tag as a type-checked contract, on every union whose classes carry one.

Each union gets a match on kind whose cases read a field only some arms carry: that body fails to
type-check unless the tag narrows the subject, so a regressed discriminator is a check-time error
rather than a runtime AttributeError.
Each union also gets a one-case match carrying a non-exhaustive-match suppression.
pyrefly reports an unused suppression as an error, so if the tag ever stops driving exhaustiveness
the suppression goes unused and the check fails; a union that gains an arm fails in the full match
beside it.
The full matches are exercised at runtime too, so a subject reaching no case fails an assertion.
That no two arms share a tag, which is what keeps an arm out of a sibling's case, is checked in
test_kind_tag_shape.py.
"""

from langchaint import (
    AssistantMessage,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyOutcome,
    DispatchOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    ImagePart,
    InvalidToolArgsDetail,
    Message,
    Part,
    ReasoningDelta,
    ReasoningTrace,
    StreamItem,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnElement,
    UserMessage,
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

_TURN = AssistantMessage(turn="hi")
_TOOL_MESSAGE = ToolMessage(tool_call_id="c1", content="ok")


def _by_message_kind(message: Message) -> object:
    match message.kind:
        case "user":
            return message.content
        case "assistant":
            return message.turn
        case "tool":
            return message.tool_call_id


def _by_message_kind_missing_an_arm(message: Message) -> object:
    match message.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "user":
            return message.content


def _by_part_kind(part: Part) -> object:
    match part.kind:
        case "text":
            return part.text
        case "image":
            return part.media_type


def _by_part_kind_missing_an_arm(part: Part) -> object:
    match part.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "text":
            return part.text


def _by_turn_element_kind(element: TurnElement) -> object:
    match element.kind:
        case "reasoning_trace":
            return element.raw
        case "text":
            return element.cache_breakpoint
        case "tool_call":
            return element.args_json


def _by_turn_element_kind_missing_an_arm(element: TurnElement) -> object:
    match element.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "reasoning_trace":
            return element.raw


def _by_dispatch_outcome_kind(outcome: DispatchOutcome) -> object:
    match outcome.kind:
        case "handled":
            return outcome.app_data
        case "invalid_tool_args":
            return outcome.details
        case "unknown_tool":
            return outcome.called_name


def _by_dispatch_outcome_kind_missing_an_arm(outcome: DispatchOutcome) -> object:
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


def _by_dispatch_many_outcome_kind_missing_an_arm(outcome: DispatchManyOutcome) -> object:
    match outcome.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "handled":
            return outcome.app_data


def _by_response_outcome_kind(outcome: ResponseOutcome[str]) -> object:
    """Cover ResponseOutcome, the union every retry loop matches an attempt's outcome over.

    A match over it is exhaustive only if a match over NoOutputOutcome is, since those are its
    no-output arms.
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


def _by_response_outcome_kind_missing_an_arm(outcome: ResponseOutcome[str]) -> object:
    match outcome.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "adapter_result":
            return outcome.output


def _by_stream_item_kind(item: StreamItem) -> object:
    """Cover StreamItem, whose str arm carries no tag because a builtin cannot hold one."""
    if isinstance(item, str):
        return item
    match item.kind:
        case "reasoning_delta":
            return item.text
        case "tool_call":
            return item.args_json


def _by_stream_item_kind_missing_an_arm(item: StreamItem) -> object:
    if isinstance(item, str):
        return item
    match item.kind:  # pyrefly: ignore[non-exhaustive-match]
        case "reasoning_delta":
            return item.text


def test_a_message_kind_selects_the_arm_that_carries_the_field_read() -> None:
    """Each Message arm's tag reaches a field the other arms do not carry."""
    assert _by_message_kind(UserMessage(content="hi")) == "hi"
    assert _by_message_kind(_TURN) == (TextPart(text="hi"),)
    assert _by_message_kind(_TOOL_MESSAGE) == "c1"
    assert _by_message_kind_missing_an_arm(_TOOL_MESSAGE) is None


def test_a_part_kind_selects_the_arm_that_carries_the_field_read() -> None:
    """Each Part arm's tag reaches a field the other arm does not carry."""
    assert _by_part_kind(TextPart(text="hi")) == "hi"
    assert _by_part_kind(ImagePart(data=b"png", media_type="image/png")) == "image/png"
    assert _by_part_kind_missing_an_arm(ImagePart(data=b"png", media_type="image/png")) is None


def test_a_turn_element_kind_selects_the_arm_that_carries_the_field_read() -> None:
    """Each TurnElement arm's tag reaches a field the other arms do not carry."""
    assert _by_turn_element_kind(ReasoningTrace(raw={"id": "rs_1"})) == {"id": "rs_1"}
    assert _by_turn_element_kind(TextPart(text="hi")) is False
    assert _by_turn_element_kind(ToolCall(id="c1", name="probe", args_json="{}")) == "{}"
    tool_call = ToolCall(id="c1", name="probe", args_json="{}")
    assert _by_turn_element_kind_missing_an_arm(tool_call) is None


def test_a_dispatch_outcome_kind_selects_the_arm_that_carries_the_field_read() -> None:
    """Each DispatchOutcome arm's tag reaches a field the other arms do not carry."""
    detail = InvalidToolArgsDetail(path=("city",), message="required")
    unknown_tool = DispatchUnknownTool(tool_message=_TOOL_MESSAGE, called_name="off_list")
    handled = DispatchHandled(tool_message=_TOOL_MESSAGE, app_data={"seen": 1})
    assert _by_dispatch_outcome_kind(handled) == {"seen": 1}
    assert _by_dispatch_outcome_kind(
        DispatchInvalidToolArgs(tool_message=_TOOL_MESSAGE, details=(detail,))
    ) == (detail,)
    assert _by_dispatch_outcome_kind(unknown_tool) == "off_list"
    assert _by_dispatch_outcome_kind_missing_an_arm(unknown_tool) is None


def test_a_dispatch_many_outcome_kind_selects_its_own_extra_arm() -> None:
    """DispatchManyOutcome's precomputed arm has its own tag, and the shared arms keep theirs."""
    precomputed = DispatchPrecomputed(tool_message=_TOOL_MESSAGE)
    assert _by_dispatch_many_outcome_kind(precomputed) is _TOOL_MESSAGE
    assert (
        _by_dispatch_many_outcome_kind(
            DispatchUnknownTool(tool_message=_TOOL_MESSAGE, called_name="off_list")
        )
        == "off_list"
    )
    assert _by_dispatch_many_outcome_kind_missing_an_arm(precomputed) is None


def test_a_response_outcome_kind_reaches_a_case_that_reads_a_field_its_arm_carries() -> None:
    """Each ResponseOutcome arm reaches a case that reads a field the arm carries."""
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
    assert _by_response_outcome_kind_missing_an_arm(Refusal(assistant_message=_TURN)) is None


def test_a_stream_item_kind_selects_the_arm_that_carries_the_field_read() -> None:
    """A str is selected by isinstance, and the tag selects between the two classes."""
    tool_call = ToolCall(id="c1", name="probe", args_json="{}")
    assert _by_stream_item_kind("hi") == "hi"
    assert _by_stream_item_kind(ReasoningDelta(text="weighing")) == "weighing"
    assert _by_stream_item_kind(tool_call) == "{}"
    assert _by_stream_item_kind_missing_an_arm(tool_call) is None
