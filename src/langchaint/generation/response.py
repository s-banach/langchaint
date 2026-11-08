"""Live generations, their records, and the outcome unions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Generic, Literal, Self, TypeVar, override

from pydantic import BaseModel, Field, SerializeAsAny, ValidationError, model_validator

from langchaint.adapter import (
    ContextWindowExceeded,
    EmptyAssistantMessage,
    FinishedResponse,
    MaxCompletionTokensExceeded,
    PlainResponse,
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    ToolCallResponse,
    UnexpectedResponse,
    UnusableResponse,
)
from langchaint.billing.usage import Usage
from langchaint.common.messages import (
    AssistantMessage,
    StopReason,
    ToolCall,
    final_text,
    final_text_index,
)
from langchaint.generation.errors import (
    GenerationError,
    PlainErrorRecord,
    SchemaViolationErrorRecord,
)
from langchaint.generation.request_history import (
    AbandonedStreamRecord,
    RequestHistory,
    RequestProviderData,
    SettledRequestRecord,
    _InputOutcomeRecordBase,
    _RequestLedger,
    _require_finished_assistant_message,
    _settled_request_records,
)

# Declared covariant because pyrefly infers a PEP 695 parameter invariant through the record's `output` field.
# `ChatAttributeTransform` takes `GenerationOutcome[object]`.
_OutputT_co = TypeVar("_OutputT_co", covariant=True)


class _GenerationRecordBase(_InputOutcomeRecordBase):
    """Require settled requests with a final, error-free request holding billing and an assistant message.

    Every earlier request must contain an error.
    """

    stop_reason: StopReason

    @model_validator(mode="after")
    def _validate_generation(self) -> Self:
        _require_finished_assistant_message(self.request_history)
        return self

    @property
    def request_records(self) -> tuple[SettledRequestRecord, ...]:
        """Return `request_history.records`, which validation guarantees are settled."""
        return _settled_request_records(self.request_history)

    @property
    def kept_request(self) -> SettledRequestRecord:
        """Return the final request."""
        final = self.request_history.records[-1]
        assert final.kind == "settled"
        return final

    @property
    @override
    def assistant_message(self) -> AssistantMessage:
        """Return the kept assistant message, from the final request."""
        assistant_message = self.kept_request.assistant_message
        assert assistant_message is not None
        return assistant_message

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the kept assistant message's tool calls."""
        return self.assistant_message.tool_calls

    @property
    def commentary(self) -> str | None:
        """Return plain text before the final text, joined with one newline, or `None` when absent.

        An assistant message with application tool calls contributes every text part.
        Reasoning and other provider content remain in `assistant_message.content`.
        """
        final_index = final_text_index(self.assistant_message)
        texts = [
            x.text
            for i, x in enumerate(self.assistant_message.content_parts)
            if x.kind == "text" and i != final_index
        ]
        return "\n".join(texts) if texts else None


class PlainGenerationRecord[OutputT](_GenerationRecordBase):
    """The normalized record of one `PlainGeneration`.

    The kept assistant message must contain no tool calls.
    """

    output: OutputT
    kind: Literal["plain"] = "plain"

    @model_validator(mode="after")
    def _validate_no_tool_call(self) -> Self:
        if self.assistant_message.tool_calls:
            raise ValueError("a PlainGenerationRecord must contain no tool calls")
        return self


class ToolCallGenerationRecord(_GenerationRecordBase):
    """The normalized record of one `ToolCallGeneration`.

    The kept assistant message must contain at least one tool call.
    """

    kind: Literal["tool_call"] = "tool_call"

    @model_validator(mode="after")
    def _validate_tool_call(self) -> Self:
        if not self.assistant_message.tool_calls:
            raise ValueError("a ToolCallGenerationRecord must contain at least one tool call")
        return self


@dataclass(frozen=True, kw_only=True)
class _LiveGenerationBase[RecordT: _GenerationRecordBase]:
    """Delegate live generation properties to one normalized record."""

    record: RecordT
    request_provider_data: tuple[RequestProviderData, ...]

    def __post_init__(self) -> None:
        """Require aligned provider data with a final provider response."""
        if len(self.request_provider_data) != len(self.record.request_history.records):
            raise ValueError("request_provider_data must align with request_history.records")
        _ = _final_raw(self.request_provider_data)

    @property
    def raw(self) -> BaseModel:
        """Return the final provider SDK response."""
        return _final_raw(self.request_provider_data)

    @property
    def request_history(self) -> RequestHistory:
        """Return the normalized request history."""
        return self.record.request_history

    @property
    def stop_reason(self) -> StopReason:
        """Return the normalized stop reason."""
        return self.record.stop_reason

    @property
    def assistant_message(self) -> AssistantMessage:
        """Return the kept assistant message."""
        return self.record.assistant_message

    @property
    def request_records(self) -> tuple[SettledRequestRecord, ...]:
        """Return `request_history.records`, which validation guarantees are settled."""
        return self.record.request_records

    @property
    def kept_request(self) -> SettledRequestRecord:
        """Return the final request."""
        return self.record.kept_request

    @property
    def usage(self) -> Usage:
        """Return normalized usage across every request."""
        return self.record.usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the kept assistant message's tool calls."""
        return self.record.tool_calls

    @property
    def commentary(self) -> str | None:
        """Return the kept assistant message's commentary, preserving text within each part.

        Text before the final text is joined with one newline, or `None` when absent.
        An assistant message with application tool calls contributes every text part.
        """
        return self.record.commentary


@dataclass(frozen=True, kw_only=True)
class PlainGeneration(
    _LiveGenerationBase[PlainGenerationRecord[_OutputT_co]],
    Generic[_OutputT_co],  # noqa: UP046
):
    """A generation with output and no application tool calls."""

    kind: Literal["plain"] = "plain"

    @property
    def output(self) -> _OutputT_co:
        """Return the assistant text or validated caller model.

        A text binding returns the final `TextPart` text of `assistant_message`.
        To continue the conversation, replay `assistant_message`.
        """
        return self.record.output


@dataclass(frozen=True, kw_only=True)
class ToolCallGeneration(_LiveGenerationBase[ToolCallGenerationRecord]):
    """A generation with application tool calls.

    Only a binding with tools produces one.
    """

    kind: Literal["tool_call"] = "tool_call"


def _final_raw(request_provider_data: tuple[RequestProviderData, ...]) -> BaseModel:
    final_raw = request_provider_data[-1].raw
    if final_raw is None:
        raise ValueError("a live generation requires a final provider response")
    return final_raw


type Generation[OutputT] = PlainGeneration[OutputT] | ToolCallGeneration
"""What an input produces when it succeeds."""
type GenerationRecord[OutputT] = PlainGenerationRecord[OutputT] | ToolCallGenerationRecord
"""The normalized record of one `PlainGeneration` or `ToolCallGeneration`."""
type GenerationOutcome[OutputT] = Generation[OutputT] | GenerationError
"""Every live outcome of an input: a `Generation` or a `GenerationError`.

`InputOutcomeRecord` also covers abandonment, which has no live form.
"""

type GenerationOutcomeRecord[OutputT] = Annotated[
    SerializeAsAny[PlainGenerationRecord[OutputT]]
    | ToolCallGenerationRecord
    | PlainErrorRecord
    | SchemaViolationErrorRecord,
    Field(discriminator="kind"),
]
"""The normalized record of one `GenerationOutcome`."""

type InputOutcomeRecord[OutputT] = GenerationOutcomeRecord[OutputT] | AbandonedStreamRecord
"""Every record of an input's outcome, including a stream the application abandoned."""


type OutputReader[OutputT] = Callable[[AssistantMessage], OutputT | ValidationError | None]
"""Read a binding's output from an assistant message.

`None` means no text part gives output, and a `ValidationError` means the text does not validate.
A text binding reads with `final_text`, and a structured binding with a `partial` of `structured_output`.
"""


def structured_output[ModelT: BaseModel](
    assistant_message: AssistantMessage, response_format: type[ModelT]
) -> ModelT | ValidationError | None:
    """Validate the text at `final_text_index` as `response_format` JSON.

    The result is the `ValidationError` when the text does not validate, and `None` when no text part gives output.
    """
    text = final_text(assistant_message)
    if text is None:
        return None
    try:
        return response_format.model_validate_json(text)
    except ValidationError as validation_error:
        return validation_error


def _stop_reason_outcome(
    assistant_message: AssistantMessage, stop_reason: StopReason
) -> Refusal | MaxCompletionTokensExceeded | ContextWindowExceeded | None:
    """Return the outcome for `refusal`, `max_completion_tokens`, or `context_window_exceeded`.

    Return `None` for another stop reason.
    """
    match stop_reason:
        case "refusal":
            return Refusal(assistant_message=assistant_message)
        case "max_completion_tokens":
            return MaxCompletionTokensExceeded(assistant_message=assistant_message)
        case "context_window_exceeded":
            return ContextWindowExceeded(assistant_message=assistant_message)
        case "stop" | "tool_call" | "other":
            return None


def response_outcome[OutputT](
    interpretation: FinishedResponse | UnusableResponse, read_output: OutputReader[OutputT]
) -> ResponseOutcome[OutputT]:
    """Return the outcome of one response, reading a finished response's output with `read_output`.

    An `UnusableResponse` is the outcome.
    For a `FinishedResponse`, the first matching rule decides:
    1. `cannot_continue` gives `UnexpectedResponse`.
    2. A refusal or truncation stop reason at a final tool call gives its outcome, because the call may be cut off.
    3. Application tool calls give `ToolCallResponse`, because the model waits for their results.
    4. Output gives `PlainResponse`, whose `stop_reason` tells whether a refusal or truncation ended the text.
    5. A refusal or truncation stop reason gives its outcome, which explains the missing or invalid output.
    6. A validation error gives `SchemaViolation`, and anything else gives `EmptyAssistantMessage`.
    """
    if interpretation.kind != "finished_response":
        return interpretation
    assistant_message = interpretation.assistant_message
    stop_reason = interpretation.stop_reason
    if interpretation.cannot_continue is not None:
        return UnexpectedResponse(
            error_text=interpretation.cannot_continue, assistant_message=assistant_message
        )
    stop_reason_outcome = _stop_reason_outcome(assistant_message, stop_reason)
    content_parts = assistant_message.content_parts
    if stop_reason_outcome is not None and content_parts and content_parts[-1].kind == "tool_call":
        return stop_reason_outcome
    if assistant_message.tool_calls:
        return ToolCallResponse(assistant_message=assistant_message, stop_reason=stop_reason)
    output = read_output(assistant_message)
    if output is not None and not isinstance(output, ValidationError):
        return PlainResponse(
            output=output, assistant_message=assistant_message, stop_reason=stop_reason
        )
    if stop_reason_outcome is not None:
        return stop_reason_outcome
    if isinstance(output, ValidationError):
        return SchemaViolation(
            validation_error_json=output.json(include_url=False),
            assistant_message=assistant_message,
        )
    return EmptyAssistantMessage(assistant_message=assistant_message)


def _generation_outcome_from_response_outcome[OutputT](
    outcome: ResponseOutcome[OutputT],
    *,
    request_history: RequestHistory,
    request_provider_data: tuple[RequestProviderData, ...],
    request_params: RequestParams | None,
    splits_on_tool_calls: bool,
) -> GenerationOutcome[OutputT]:
    match outcome.kind:
        case "plain_response":
            return PlainGeneration(
                record=PlainGenerationRecord(
                    output=outcome.output,
                    request_history=request_history,
                    stop_reason=outcome.stop_reason,
                ),
                request_provider_data=request_provider_data,
            )
        case "tool_call_response" if splits_on_tool_calls:
            return ToolCallGeneration(
                record=ToolCallGenerationRecord(
                    request_history=request_history, stop_reason=outcome.stop_reason
                ),
                request_provider_data=request_provider_data,
            )
        case "tool_call_response":
            record = PlainErrorRecord(
                kind="unexpected_response_error",
                error_text="the response has application tool calls, but the binding has no ToolManager",
                request_history=request_history,
            )
        case "refusal":
            record = PlainErrorRecord(kind="refusal_error", request_history=request_history)
        case "max_completion_tokens_exceeded":
            record = PlainErrorRecord(
                kind="max_completion_tokens_exceeded_error", request_history=request_history
            )
        case "empty_assistant_message":
            record = PlainErrorRecord(
                kind="empty_assistant_message_error", request_history=request_history
            )
        case "schema_violation":
            record = SchemaViolationErrorRecord(
                validation_error_json=outcome.validation_error_json,
                request_history=request_history,
            )
        case "context_window_exceeded":
            record = PlainErrorRecord(
                kind="context_window_exceeded_error", request_history=request_history
            )
        case "unexpected_response":
            record = PlainErrorRecord(
                kind="unexpected_response_error",
                error_text=outcome.error_text,
                request_history=request_history,
            )
        case "provider_failed_terminally":
            record = PlainErrorRecord(
                kind="provider_failed_terminally_error",
                error_text=outcome.error_text,
                request_history=request_history,
            )
        case "provider_failed_transiently":
            raise ValueError("ProviderFailedTransiently requires the caller's retry policy")
    return GenerationError(
        record=record,
        request_params=request_params,
        request_provider_data=request_provider_data,
    )


def _timed_out_error(ledger: _RequestLedger) -> GenerationError:
    """Build the expired deadline's failure with one normalized cut-off request."""
    request_history, request_provider_data = ledger.freeze_with_cut_off()
    return GenerationError(
        record=PlainErrorRecord(kind="timed_out_error", request_history=request_history),
        request_params=None,
        request_provider_data=request_provider_data,
    )


def _escaped_error(ledger: _RequestLedger, escaped: Exception) -> GenerationError:
    """Build the failure for an `Exception` that escaped failure handling, with one normalized cut-off request.

    The cut-off request carries the billing noted on `ledger` for the request in flight.
    The caller sets `escaped` as the cause.
    """
    request_history, request_provider_data = ledger.freeze_with_cut_off()
    return GenerationError(
        record=PlainErrorRecord(
            kind="unknown_exception_error",
            error_text=str(escaped),
            request_history=request_history,
        ),
        request_params=None,
        request_provider_data=request_provider_data,
    )
