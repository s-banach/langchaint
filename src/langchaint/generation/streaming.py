"""The stream handle.

`StreamHandle` opens inside `async with` and yields `StreamItem` values.
`final` drains remaining items and returns the assembled result.
Open failures can retry; failures after opening cannot retry.
One `SharedBackoff.admitted` block spans the open stream's lifetime.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from types import TracebackType
from typing import Literal, Never, overload

from langchaint.adapter import (
    Adapter,
    AdapterStream,
    BoundAdapter,
    InvalidRequest,
    RequestParams,
    ResponseOutcome,
    StreamItem,
)
from langchaint.billing.pricing import ProviderBilling
from langchaint.common.exceptions import TransientError
from langchaint.common.messages import Message
from langchaint.common.observed_operation import ObservedOperation
from langchaint.concurrency.shared_backoff import (
    Admission,
    PrivateBackoff,
    SharedBackoff,
    Verdict,
    _exit_admission,
)
from langchaint.failure_step import (
    _failure_step,
    _FailureStep,
    _RetryStep,
    _transient_error_for_step,
)
from langchaint.generation.call import AbandonedCallRecord, _CallLedger
from langchaint.generation.errors import (
    GenerationError,
    InvalidRequestErrorRecord,
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    _terminal_generation_error,
)
from langchaint.generation.response import (
    CallResult,
    Response,
    ToolCallTurn,
    _call_result_from_response_outcome,
    _escaped_error,
    _timed_out_error,
)

type _State = Literal["unopened", "open", "finished"]

_logger = logging.getLogger("langchaint.streaming")

_UNOPENED_MESSAGE = "stream not open: enter the handle with `async with` before using it"
_FINISHED_MESSAGE = "stream is finished: call stream_one again for a new one"
_ALREADY_ENTERED_MESSAGE = "stream already entered: call stream_one again for a new one"


async def _close_stream_quietly(
    adapter_stream: AdapterStream, *, failure_log_message: str
) -> None:
    """Close one attempt's stream, logging a close failure rather than raising it.

    The request has already ended.
    A close exception would displace its result or error, so this function logs the exception.
    `failure_log_message` names the preserved result.

    Raises:
        BaseException: `adapter_stream.close()` raises a value outside `Exception`.
    """
    try:
        await adapter_stream.close()
    except Exception:
        _logger.warning(failure_log_message, exc_info=True)


class StreamHandle[OutputT, ToolTurnT: ToolCallTurn[object] = Never]:
    """An async context manager and iterator for one streamed call.

    Entry opens the request. `final` drains items. `final` returns `Response` or `ToolTurnT`.
    `max_attempts` applies only before the stream opens.
    An open-stream transient failure raises `GenerationError`.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT],
        messages: Sequence[Message],
        shared_backoff: SharedBackoff,
        max_attempts: int,
        timeout_seconds: float | None,
        splits_tool_call_turns: bool,
        generation_started: Callable[
            [], ObservedOperation[CallResult[object] | AbandonedCallRecord]
        ],
    ) -> None:
        """Store the request.

        `generation_started` starts following the call when the handle is entered.
        """
        self._generation_started = generation_started
        self._operation: ObservedOperation[CallResult[object] | AbandonedCallRecord] | None = None
        self._adapter = adapter
        self._bound_adapter = bound_adapter
        self._messages = messages
        self._shared_backoff = shared_backoff
        self._max_attempts = max_attempts
        self._private_backoff = PrivateBackoff(shared_backoff)
        self._timeout_seconds = timeout_seconds
        self._splits_tool_call_turns = splits_tool_call_turns
        self._deadline: asyncio.Timeout | None = None
        self.abandoned: AbandonedCallRecord | None = None
        """The account of a call that no result or `GenerationError` records, or `None`.

        Leaving the block before the conclusion, an exception raised inside it, and cancellation each set this value.
        It holds the billing and first-item time of the request the exit cut off.
        Cancellation sets it before the caller receives `asyncio.CancelledError`.
        A success or `GenerationError` leaves this value as `None`.
        An expired `timeout_seconds` raises `GenerationError` and leaves this value as `None`.
        """
        self._adapter_stream: AdapterStream | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._ledger: _CallLedger
        """Built by `__aenter__`, which starts the call."""
        self._admission: Admission | None = None
        self._ended_at_monotonic_seconds: float | None = None
        self._conclusion: CallResult[OutputT] | None = None
        self._state: _State = "unopened"
        self._request: RequestParams | None = None

    async def __aenter__(self) -> "StreamHandle[OutputT, ToolTurnT]":
        """Open the request and return self.

        Raises:
            GenerationError: `build_request` returns `InvalidRequest`.
                The provider rejects the request.
                The provider declares the open failure final.
                The adapter cannot classify the open failure.
                The open attempts consume `max_attempts`.
                `timeout_seconds` expires before the request opens.
                An `Exception` escapes failure handling.
            RuntimeError: This handle was already entered.
        """
        if self._state != "unopened":
            raise RuntimeError(_ALREADY_ENTERED_MESSAGE)
        self._operation = self._generation_started()
        try:
            await self._open()
        except GenerationError as failure:
            self._end_operation(failure)
            raise
        except Exception as escaped:
            failure = _escaped_error(self._ledger, escaped)
            self._end_operation(failure)
            raise failure from escaped
        except BaseException:
            self._end_operation(None)
            raise
        return self

    async def _open(self) -> None:
        """Open the request under the deadline.

        Raises what `__aenter__` documents.
        """
        self._state = "open"
        self._ledger = _CallLedger(
            model=self._adapter.model, provider_name=self._adapter.provider_name
        )
        self._deadline = asyncio.timeout(self._timeout_seconds)
        await self._deadline.__aenter__()
        try:
            await self._open_stream_with_retries()
        except BaseException as exc:
            # `__aexit__` does not run when `__aenter__` raises.
            # Finish the call, exit admission, and close the deadline here.
            # An open deadline would retain a timer that could cancel this task after this operation.
            # Record abandonment here because no other frame sees cancellation during the open.
            self._state = "finished"
            _ = self._exit_admission(None)
            billing_in_flight = self._billing_reported()
            if await self._close_deadline(exc):
                raise _timed_out_error(self._ledger, billing_in_flight) from None
            if isinstance(exc, asyncio.CancelledError):
                self._abandon(billing_in_flight)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the deadline, connection, and admission.

        Leaving the block sets `abandoned` unless a conclusion already records the call.
        An expired `timeout_seconds` raises `GenerationError` instead.
        A call without a conclusion concludes with the expiry's `GenerationError` or with `abandoned`.

        Raises:
            GenerationError: `timeout_seconds` expires before the block finishes.
            BaseException: Closing the adapter stream raises a non-`Exception` value.
        """
        try:
            await self._close(exc)
        except GenerationError as timed_out:
            self._end_operation(timed_out)
            raise
        finally:
            self._end_operation(None)

    def _end_operation(self, conclusion: CallResult[OutputT] | AbandonedCallRecord | None) -> None:
        """Give the observer the call's conclusion, when there is one, and end the operation once."""
        operation, self._operation = self._operation, None
        if operation is None:
            return
        if conclusion is not None:
            operation.conclude(conclusion)
        operation.end()

    async def _close(self, exc: BaseException | None) -> None:
        """Close the deadline, connection, and admission after the block ends with `exc`.

        Raises what `__aexit__` documents.
        """
        self._state = "finished"
        # Read before the close, which drops the stream that reports it.
        billing_in_flight = self._billing_reported()
        timed_out = await self._close_deadline(exc)
        try:
            await self._close_adapter_stream()
        finally:
            if not timed_out:
                self._abandon(billing_in_flight)
        if timed_out:
            raise _timed_out_error(self._ledger, billing_in_flight) from None

    async def _close_deadline(self, exc: BaseException | None) -> bool:
        """Close the deadline and return whether it caused the current cancellation.

        Repeated calls return `False`.
        """
        if self._deadline is None:
            return False
        deadline, self._deadline = self._deadline, None
        try:
            await deadline.__aexit__(
                type(exc) if exc is not None else None,
                exc,
                exc.__traceback__ if exc is not None else None,
            )
        except TimeoutError:
            return True
        return False

    def _billing_reported(self) -> ProviderBilling | None:
        """Ask the open stream what the provider has reported, or None where it reported nothing.

        Call before the connection closes, because closing drops the stream this asks.
        """
        if self._adapter_stream is None:
            return None
        return self._adapter_stream.billing_reported()

    def _abandon(self, billing_in_flight: ProviderBilling | None) -> None:
        """Set `abandoned` and conclude the operation with it, when the call has no conclusion.

        Include billing reported by the interrupted attempt.
        """
        if self._conclusion is not None:
            return
        call, _ = self._ledger.freeze_with_cut_off(billing_in_flight)
        self.abandoned = AbandonedCallRecord(call=call)
        self._end_operation(self.abandoned)

    def _exit_admission(self, exc: BaseException | None) -> Verdict | None:
        """Exit the held admission and return its `Verdict`.

        Repeated calls return `None`.
        """
        if self._admission is None:
            return None
        admission, self._admission = self._admission, None
        return _exit_admission(admission, exc)

    async def _close_adapter_stream(self) -> None:
        """Close the provider connection and release admission.

        Release admission even when closing raises.
        """
        adapter_stream = self._adapter_stream
        self._adapter_stream = None
        self._items = None
        try:
            if adapter_stream is not None:
                await _close_stream_quietly(
                    adapter_stream,
                    failure_log_message=(
                        "closing the provider stream raised; the permit was returned"
                    ),
                )
        finally:
            _ = self._exit_admission(None)

    def _step_after_failure(self, exc: Exception, verdict: Verdict | None) -> _FailureStep:
        """Note the failed attempt's request id and decide how the call continues."""
        request_id = self._adapter.request_id_from_error(exc)
        if request_id is None and self._adapter_stream is not None:
            request_id = self._adapter_stream.request_id()
        self._ledger.note_request_id(request_id)
        return _failure_step(exc, verdict=verdict, classify=self._adapter.classify)

    async def _backoff_or_exhaust(self, exc: Exception, step: _RetryStep) -> None:
        """Wait before the next open attempt, as `step` asks.

        `_RetryAfterSharedPause` relies on the next `admitted()` wait.

        Raises:
            GenerationError: the recorded failure spent the last attempt.
        """
        if self._ledger.attempts >= self._max_attempts:
            raise GenerationError(
                record=RetriesExhaustedErrorRecord(call=self._ledger.freeze()),
                request=self._request,
                provider_attempts=self._ledger.provider_attempts,
            ) from exc
        if step.kind == "retry_after_private_wait":
            await asyncio.sleep(self._private_backoff.next_wait(step.retry_after))

    def __aiter__(self) -> "StreamHandle[OutputT, ToolTurnT]":
        """Return `self` because the handle is its own iterator."""
        return self

    async def _open_stream_with_retries(self) -> None:
        """Open an adapter stream and retry transient open failures.

        Each attempt uses one `admitted` block and releases it before backoff.
        A successful stream holds admission until completion.

        Raises:
            GenerationError: The adapter or provider rejects the request.
                The provider declares the open failure terminal.
                The adapter cannot classify the open failure.
                Open failures consume `max_attempts`.
        """
        built = self._bound_adapter.build_request(self._messages)
        if isinstance(built, InvalidRequest):
            raise GenerationError(
                record=InvalidRequestErrorRecord(
                    error_text=built.reason, call=self._ledger.freeze()
                ),
                request=None,
                provider_attempts=self._ledger.provider_attempts,
            ) from None
        request = built
        self._request = request
        while self._adapter_stream is None:
            self._admission = await self._shared_backoff.admitted().__aenter__()
            self._ledger.start_attempt()
            try:
                opened = await self._bound_adapter.open_stream(request)
            except Exception as exc:
                step = self._step_after_failure(exc, self._exit_admission(exc))
                if step.kind == "terminal":
                    raise _terminal_generation_error(
                        step,
                        reason=str(exc),
                        ledger=self._ledger,
                        billing=None,
                        request=request,
                        stream_opened=self._adapter_stream is not None,
                    ) from exc
                self._ledger.record(
                    error=_transient_error_for_step(exc, str(exc), step), assistant_message=None
                )
                await self._backoff_or_exhaust(exc, step)
                continue
            except BaseException:
                # `CancelledError` is a `BaseException` that the clause above does not catch.
                # Exit here to return the permit at the same point on each failing path.
                _ = self._exit_admission(None)
                raise
            self._adapter_stream = opened
            self._items = self._adapter_stream.items()

    async def __anext__(self) -> StreamItem:
        """Return the next item.

        Raises:
            GenerationError: The open stream fails transiently.
                The event stream violates its event contract, such as ending without a terminal event.
                The adapter classifies an item error as an invalid request.
                The provider declares an item error terminal.
                The adapter cannot classify an item exception.
                An `Exception` escapes failure handling.
            StopAsyncIteration: The stream is exhausted.
            RuntimeError: The handle is unopened or finished.
        """
        if self._state != "open":
            raise RuntimeError(
                _UNOPENED_MESSAGE if self._state == "unopened" else _FINISHED_MESSAGE
            )
        try:
            return await self._next_item()
        except StopAsyncIteration:
            raise
        except BaseException as exc:
            self._state = "finished"
            if self._conclusion is not None or not isinstance(exc, Exception):
                # A stored conclusion already records the call.
                # Cancellation destroys the frames that could observe the call, so `abandoned` records it.
                raise
            failure = (
                exc if isinstance(exc, GenerationError) else _escaped_error(self._ledger, exc)
            )
            self._conclusion = failure
            self._end_operation(failure)
            await self._close_deadline(exc)
            if failure is exc:
                raise
            raise failure from exc

    async def _next_item(self) -> StreamItem:
        """Pull the next item without retrying the request.

        Raises what __anext__ documents.
        """
        assert self._items is not None
        try:
            item = await self._items.__anext__()
        except StopAsyncIteration:
            if self._ended_at_monotonic_seconds is None:
                self._ended_at_monotonic_seconds = time.monotonic()
            _ = self._exit_admission(None)
            raise
        except Exception as exc:  # noqa: BLE001 (every failure of an open stream settles its attempt)
            raise await self._error_after_stream_failure(exc, stage="iteration")  # noqa: B904 (the returned error already holds its cause)
        except BaseException:
            _ = self._exit_admission(None)
            raise
        self._ledger.stamp_first_item()
        return item

    async def _error_after_stream_failure(
        self, exc: Exception, *, stage: Literal["iteration", "assembly"]
    ) -> GenerationError:
        """Settle the open stream's failed attempt and build the call's error with its cause set.

        An open stream cannot retry, so a transient failure ends the call as `retry_unavailable_error`.
        `stage` names the stream step that failed in the transient error text.
        """
        self._state = "finished"
        stream_billing = self._billing_reported()
        self._ledger.note_billing_in_flight(stream_billing)
        step = self._step_after_failure(exc, self._exit_admission(exc))
        if step.kind == "terminal":
            error = _terminal_generation_error(
                step,
                reason=str(exc),
                ledger=self._ledger,
                billing=stream_billing,
                request=self._request,
                stream_opened=self._adapter_stream is not None,
            )
            error.__cause__ = exc
        else:
            wrapped = _transient_error_for_step(
                exc, f"open stream failed during {stage}: {exc}", step
            )
            self._ledger.record(error=wrapped, assistant_message=None, billing=stream_billing)
            error = GenerationError(
                record=RetryUnavailableErrorRecord(call=self._ledger.freeze()),
                request=self._request,
                provider_attempts=self._ledger.provider_attempts,
            )
            error.__cause__ = wrapped
        await self._close_adapter_stream()
        return error

    @overload
    async def final(self: "StreamHandle[OutputT, Never]") -> Response[OutputT]: ...
    @overload
    async def final(self) -> "Response[OutputT] | ToolTurnT": ...
    async def final(self) -> Response[OutputT] | ToolCallTurn[object]:
        """Drain remaining items and return the stored result.

        Repeated calls return or raise the same conclusion without reading the stream again.
        A response with no output becomes a terminal `GenerationError`.

        Raises:
            GenerationError: The adapter or provider reports a terminal failure.
                The assembled response reports a terminal result.
                The open stream fails transiently.
                The event stream violates its event contract, such as ending without a terminal event.
                The adapter cannot classify an item exception.
                The adapter fails to assemble or interpret the drained stream.
                An `Exception` escapes failure handling.
            RuntimeError: The handle is unopened or finished without a stored conclusion.
        """
        if self._conclusion is None:
            if self._state != "open":
                raise RuntimeError(
                    _UNOPENED_MESSAGE if self._state == "unopened" else _FINISHED_MESSAGE
                )
            async for _ in self:
                pass
            adapter_stream = self._adapter_stream
            assert adapter_stream is not None
            ended_at_monotonic_seconds = (
                time.monotonic()
                if self._ended_at_monotonic_seconds is None
                else self._ended_at_monotonic_seconds
            )
            try:
                self._conclusion = await self._assembled_conclusion(
                    adapter_stream, ended_at_monotonic_seconds=ended_at_monotonic_seconds
                )
            except Exception as escaped:  # noqa: BLE001 (an escaped Exception becomes the stored escaped_exception_error)
                # Store every conclusion, because `_conclude` records the attempt before it can raise.
                # A second `final()` call would otherwise record the attempt again.
                escaped_error = _escaped_error(self._ledger, escaped)
                escaped_error.__cause__ = escaped
                self._conclusion = escaped_error
            self._end_operation(self._conclusion)
            await self._close_deadline(None)
        if isinstance(self._conclusion, (Response, ToolCallTurn)):
            return self._conclusion
        raise self._conclusion

    async def _assembled_conclusion(
        self, adapter_stream: AdapterStream, *, ended_at_monotonic_seconds: float
    ) -> CallResult[OutputT]:
        """Assemble and interpret the drained stream's response, and build the call's conclusion.

        A failure to assemble or interpret settles the attempt like a failure during iteration.
        """
        try:
            raw = await adapter_stream.final()
            self._ledger.stage_response(
                raw=raw,
                billing=self._bound_adapter.billing_from_raw(raw),
                identity=self._bound_adapter.identity_from_raw(
                    raw, request_id=adapter_stream.request_id()
                ),
            )
            outcome = self._bound_adapter.interpret(raw)
        except Exception as exc:  # noqa: BLE001 (an assembly failure settles the attempt like an iteration failure)
            return await self._error_after_stream_failure(exc, stage="assembly")
        return self._conclude(outcome, ended_at_monotonic_seconds=ended_at_monotonic_seconds)

    def _conclude(
        self,
        outcome: ResponseOutcome[OutputT],
        *,
        ended_at_monotonic_seconds: float,
    ) -> CallResult[OutputT]:
        """Build the call result for this outcome.

        Returns the error rather than raising it, so no case can conclude the call without being stored.
        Every outcome records the staged response before building the result.
        The frozen call therefore includes the terminal response and billing.
        """
        if outcome.kind == "provider_failed_transiently":
            failure = TransientError(outcome.reason, is_rate_limit=outcome.is_rate_limit)
            self._ledger.record_ending_at(
                ended_at_monotonic_seconds,
                error=failure,
                assistant_message=outcome.assistant_message,
            )
            retry_unavailable = GenerationError(
                record=RetryUnavailableErrorRecord(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds)
                ),
                request=self._request,
                provider_attempts=self._ledger.provider_attempts,
            )
            retry_unavailable.__cause__ = failure
            return retry_unavailable
        self._ledger.record_ending_at(
            ended_at_monotonic_seconds,
            error=None,
            assistant_message=outcome.assistant_message,
        )
        return _call_result_from_response_outcome(
            outcome,
            call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
            provider_attempts=self._ledger.provider_attempts,
            request=self._request,
            splits_tool_call_turns=self._splits_tool_call_turns,
        )
