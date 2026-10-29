"""Consume provider streams while recording input outcomes."""

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
    ReasoningDelta,
    RejectedMessages,
    RequestParams,
    ResponseOutcome,
    StreamItem,
)
from langchaint.billing.pricing import ProviderBilling
from langchaint.common.exceptions import TransientError
from langchaint.common.messages import Message
from langchaint.common.observed_operation import ObservedOperation
from langchaint.common.request_failure import RequestFailure, _request_failure_of
from langchaint.concurrency.shared_backoff import (
    Admission,
    PrivateBackoff,
    SharedBackoff,
)
from langchaint.generation.errors import (
    GenerationError,
    PlainErrorRecord,
    _terminal_generation_error,
    _transient_error_for,
)
from langchaint.generation.request_history import AbandonedStreamRecord, _RequestLedger
from langchaint.generation.response import (
    GenerationOutcome,
    PlainGeneration,
    ToolCallGeneration,
    _escaped_error,
    _generation_outcome_from_response_outcome,
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
    """Log close failures to preserve the request's generation or error.

    `failure_log_message` names the preserved outcome.

    Raises:
        BaseException: `adapter_stream.close()` raises a value outside `Exception`.
    """
    try:
        await adapter_stream.close()
    except Exception:
        _logger.warning(failure_log_message, exc_info=True)


class StreamHandle[OutputT, ToolCallGenerationT: ToolCallGeneration = Never]:
    """Handle one streamed input inside `async with`.

    Entry opens the request; iteration yields items, and `final` drains remaining items.
    Only opening can retry, up to `max_requests`; an open-stream failure raises `GenerationError`.
    An open stream holds admission until iteration ends or the connection closes.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT | None],
        messages: Sequence[Message],
        shared_backoff: SharedBackoff,
        max_requests: int,
        timeout_seconds: float | None,
        splits_on_tool_calls: bool,
        generation_started: Callable[
            [], ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord]
        ],
    ) -> None:
        """Prepare the input without opening a request.

        Args:
            adapter: The provider adapter for request metadata and failure handling.
            bound_adapter: The bound adapter for request construction and response handling.
            messages: The messages to send.
            shared_backoff: Shared admission and backoff for the rate-limit quota.
            max_requests: The request limit for opening the stream.
            timeout_seconds: The wall-clock budget from entry until an outcome or block exit, or `None`.
            splits_on_tool_calls: Whether application tool calls produce a `ToolCallGeneration`.
            generation_started: Start observing the input when the handle is entered.
        """
        self._generation_started = generation_started
        self._operation: (
            ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord] | None
        ) = None
        self._adapter = adapter
        self._bound_adapter = bound_adapter
        self._messages = messages
        self._shared_backoff = shared_backoff
        self._max_requests = max_requests
        self._private_backoff = PrivateBackoff(shared_backoff)
        self._timeout_seconds = timeout_seconds
        self._splits_on_tool_calls = splits_on_tool_calls
        self._timeout_scope: asyncio.Timeout | None = None
        self.abandoned: AbandonedStreamRecord | None = None
        """The input's record when the block exits without a stored `GenerationOutcome`, or `None`.

        Early exit, an application exception, or cancellation sets this value.
        The record includes the interrupted request's billing and first-item time.
        Cancellation sets it before the caller receives `asyncio.CancelledError`.
        An expired `timeout_seconds` raises `GenerationError` and leaves this value as `None`.
        """
        self._adapter_stream: AdapterStream | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._ledger: _RequestLedger
        self._admission: Admission | None = None
        self._ended_at_monotonic_seconds: float | None = None
        self._outcome: GenerationOutcome[OutputT] | None = None
        self._state: _State = "unopened"
        self._request_params: RequestParams | None = None

    async def __aenter__(self) -> "StreamHandle[OutputT, ToolCallGenerationT]":
        """Open the request and return self.

        Raises:
            GenerationError: `build_request_params` returns `RejectedMessages`.
                The provider rejects the request.
                The provider declares the open failure final.
                The adapter cannot place the open failure.
                The requests that open the stream consume `max_requests`.
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
        self._state = "open"
        self._ledger = _RequestLedger(
            model=self._adapter.model, provider_name=self._adapter.provider_name
        )
        self._timeout_scope = asyncio.timeout(self._timeout_seconds)
        await self._timeout_scope.__aenter__()
        try:
            await self._open_stream_with_retries()
        except BaseException as exc:
            # Failed entry skips `__aexit__`, so release admission and cancel the timer here.
            # Cancellation during entry must also record abandonment here.
            self._state = "finished"
            self._exit_admission()
            provider_billing_in_flight = self._provider_billing()
            if await self._close_timeout_scope(exc):
                raise _timed_out_error(self._ledger, provider_billing_in_flight) from None
            if isinstance(exc, asyncio.CancelledError):
                self._abandon(provider_billing_in_flight)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the timeout scope, connection, and admission.

        Exiting without a stored outcome sets `abandoned`, unless `timeout_seconds` expired.

        Raises:
            GenerationError: `timeout_seconds` expires before the input's outcome or block exit.
            BaseException: Closing the adapter stream raises a non-`Exception` value.
        """
        try:
            await self._close(exc)
        except GenerationError as timed_out:
            self._end_operation(timed_out)
            raise
        finally:
            self._end_operation(None)

    def _end_operation(
        self, outcome: GenerationOutcome[OutputT] | AbandonedStreamRecord | None
    ) -> None:
        """Give the observer the input's outcome, when there is one, and end the operation once."""
        operation, self._operation = self._operation, None
        if operation is None:
            return
        if outcome is not None:
            operation.conclude(outcome)
        operation.end()

    async def _close(self, exc: BaseException | None) -> None:
        self._state = "finished"
        # Read before the close, which drops the stream that reports it.
        provider_billing_in_flight = self._provider_billing()
        timed_out = await self._close_timeout_scope(exc)
        try:
            await self._close_adapter_stream()
        finally:
            if not timed_out:
                self._abandon(provider_billing_in_flight)
        if timed_out:
            raise _timed_out_error(self._ledger, provider_billing_in_flight) from None

    async def _close_timeout_scope(self, exc: BaseException | None) -> bool:
        """Close the timeout scope and return whether it caused the current cancellation.

        Repeated calls return `False`.
        """
        if self._timeout_scope is None:
            return False
        timeout_scope, self._timeout_scope = self._timeout_scope, None
        try:
            await timeout_scope.__aexit__(
                type(exc) if exc is not None else None,
                exc,
                exc.__traceback__ if exc is not None else None,
            )
        except TimeoutError:
            return True
        return False

    def _provider_billing(self) -> ProviderBilling | None:
        if self._adapter_stream is None:
            return None
        return self._adapter_stream.provider_billing()

    def _abandon(self, provider_billing_in_flight: ProviderBilling | None) -> None:
        """Record abandonment with the interrupted request's billing unless an outcome exists."""
        if self._outcome is not None:
            return
        request_history, _ = self._ledger.freeze_with_cut_off(provider_billing_in_flight)
        self.abandoned = AbandonedStreamRecord(request_history=request_history)
        self._end_operation(self.abandoned)

    def _exit_admission(self) -> None:
        """Return the held permit. Repeated calls do nothing."""
        if self._admission is None:
            return
        admission, self._admission = self._admission, None
        admission._release()

    def _request_failure_exiting_admission(self, exc: Exception) -> RequestFailure:
        """Record the failure before releasing admission, so its quota pause starts first.

        Assembly failures can arrive after iteration returned admission.
        Release admission even when `request_failure` raises.

        Raises:
            Exception: The adapter's `request_failure` raises, which is an adapter defect.
        """
        request_id = self._adapter.request_id_from_error(exc)
        if request_id is None and self._adapter_stream is not None:
            request_id = self._adapter_stream.request_id()
        self._ledger.note_request_id(request_id)
        try:
            return self._shared_backoff._recorded(
                _request_failure_of(exc, self._adapter.request_failure)
            )
        finally:
            self._exit_admission()

    async def _close_adapter_stream(self) -> None:
        """Close the provider connection and release admission even when closing raises."""
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
            self._exit_admission()

    async def _backoff_or_exhaust(self, exc: Exception, request_failure: RequestFailure) -> None:
        """Wait before retrying an open failure unless the request limit is reached.

        A failure that pauses the quota relies on the next `admitted()` wait.

        Raises:
            GenerationError: the recorded failure used the last of `max_requests`.
        """
        if self._ledger.request_count >= self._max_requests:
            raise GenerationError(
                record=PlainErrorRecord(
                    kind="retries_exhausted_error", request_history=self._ledger.freeze()
                ),
                request_params=self._request_params,
                request_provider_data=self._ledger.request_provider_data,
            ) from exc
        if not request_failure.pauses_quota:
            await asyncio.sleep(
                self._private_backoff.next_wait_seconds(request_failure.retry_after_seconds)
            )

    def __aiter__(self) -> "StreamHandle[OutputT, ToolCallGenerationT]":
        """Return `self` because the handle is its own iterator."""
        return self

    async def _open_stream_with_retries(self) -> None:
        """Open an adapter stream and retry transient open failures.

        Release admission before backoff; successful opening keeps admission for iteration.
        """
        request_params = self._bound_adapter.build_request_params(self._messages)
        if isinstance(request_params, RejectedMessages):
            raise GenerationError(
                record=PlainErrorRecord(
                    kind="rejected_error",
                    error_text=request_params.error_text,
                    request_history=self._ledger.freeze(),
                ),
                request_params=None,
                request_provider_data=self._ledger.request_provider_data,
            ) from None
        self._request_params = request_params
        while self._adapter_stream is None:
            self._admission = await self._shared_backoff.admitted().__aenter__()
            self._ledger.start_request()
            try:
                self._adapter_stream = await self._bound_adapter.open_stream(request_params)
            except Exception as exc:
                request_failure = self._request_failure_exiting_admission(exc)
                if request_failure.kind != "transient":
                    raise _terminal_generation_error(
                        request_failure.kind,
                        error_text=str(exc),
                        ledger=self._ledger,
                        provider_billing=None,
                        request_params=request_params,
                        stream_opened=self._adapter_stream is not None,
                    ) from exc
                self._ledger.record(
                    error=_transient_error_for(exc, str(exc), request_failure),
                    assistant_message=None,
                )
                await self._backoff_or_exhaust(exc, request_failure)
                continue
            except BaseException:
                self._exit_admission()
                raise
            self._items = self._adapter_stream.items()

    @overload
    async def __anext__(self: "StreamHandle[OutputT, Never]") -> str | ReasoningDelta: ...
    @overload
    async def __anext__(self) -> StreamItem: ...
    async def __anext__(self) -> StreamItem:
        """Return the next item.

        Raises:
            GenerationError: The open stream fails transiently.
                The event stream violates its event contract, such as ending without a terminal event.
                The adapter places an item error as rejected.
                The provider declares an item error terminal.
                The adapter cannot place an item exception.
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
            if self._outcome is not None or not isinstance(exc, Exception):
                # Context exit records abandonment for cancellation without a stored outcome.
                raise
            failure = (
                exc if isinstance(exc, GenerationError) else _escaped_error(self._ledger, exc)
            )
            self._outcome = failure
            self._end_operation(failure)
            await self._close_timeout_scope(exc)
            if failure is exc:
                raise
            raise failure from exc

    async def _next_item(self) -> StreamItem:
        assert self._items is not None
        try:
            item = await self._items.__anext__()
        except StopAsyncIteration:
            if self._ended_at_monotonic_seconds is None:
                self._ended_at_monotonic_seconds = time.monotonic()
            self._exit_admission()
            raise
        except Exception as exc:  # noqa: BLE001 (every failure of an open stream settles its request)
            raise await self._error_after_stream_failure(exc, stage="iteration")  # noqa: B904 (the returned error already holds its cause)
        except BaseException:
            self._exit_admission()
            raise
        self._ledger.stamp_first_item()
        return item

    async def _error_after_stream_failure(
        self, exc: Exception, *, stage: Literal["iteration", "assembly"]
    ) -> GenerationError:
        """Settle the open stream's failed request and build the input's error with its cause set.

        An open stream cannot retry, so a transient failure ends the input as `retry_unavailable_error`.
        `stage` names the stream step that failed in the transient error text.
        """
        self._state = "finished"
        provider_billing_in_flight = self._provider_billing()
        self._ledger.note_provider_billing_in_flight(provider_billing_in_flight)
        request_failure = self._request_failure_exiting_admission(exc)
        if request_failure.kind != "transient":
            error = _terminal_generation_error(
                request_failure.kind,
                error_text=str(exc),
                ledger=self._ledger,
                provider_billing=provider_billing_in_flight,
                request_params=self._request_params,
                stream_opened=self._adapter_stream is not None,
            )
            error.__cause__ = exc
        else:
            wrapped = _transient_error_for(
                exc, f"open stream failed during {stage}: {exc}", request_failure
            )
            self._ledger.record(
                error=wrapped, assistant_message=None, provider_billing=provider_billing_in_flight
            )
            error = GenerationError(
                record=PlainErrorRecord(
                    kind="retry_unavailable_error", request_history=self._ledger.freeze()
                ),
                request_params=self._request_params,
                request_provider_data=self._ledger.request_provider_data,
            )
            error.__cause__ = wrapped
        await self._close_adapter_stream()
        return error

    @overload
    async def final(
        self: "StreamHandle[OutputT, Never]",
    ) -> PlainGeneration[OutputT]: ...
    @overload
    async def final(self) -> "PlainGeneration[OutputT] | ToolCallGenerationT": ...
    async def final(self) -> PlainGeneration[OutputT] | ToolCallGeneration:
        """Drain remaining items and return the stored generation.

        Repeated calls return or raise the same outcome without reading the stream again.

        Raises:
            GenerationError: The adapter or provider reports a terminal failure.
                The assembled response is not usable.
                The open stream fails transiently.
                The event stream violates its event contract, such as ending without a terminal event.
                The adapter cannot place an item exception.
                The adapter fails to assemble or interpret the drained stream.
                An `Exception` escapes failure handling.
            RuntimeError: The handle is unopened or finished without a stored outcome.
        """
        if self._outcome is None:
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
                self._outcome = await self._assembled_outcome(
                    adapter_stream, ended_at_monotonic_seconds=ended_at_monotonic_seconds
                )
            except Exception as escaped:  # noqa: BLE001 (an escaped Exception becomes the stored unknown_exception_error)
                # Cache the error because `_outcome_from_response` can record the request before raising.
                escaped_error = _escaped_error(self._ledger, escaped)
                escaped_error.__cause__ = escaped
                self._outcome = escaped_error
            self._end_operation(self._outcome)
            await self._close_timeout_scope(None)
        if isinstance(self._outcome, (PlainGeneration, ToolCallGeneration)):
            return self._outcome
        raise self._outcome

    async def _assembled_outcome(
        self, adapter_stream: AdapterStream, *, ended_at_monotonic_seconds: float
    ) -> GenerationOutcome[OutputT]:
        try:
            raw = await adapter_stream.final()
            self._ledger.stage_response(
                raw=raw,
                provider_billing=self._bound_adapter.billing_from_raw(raw),
                identity=self._bound_adapter.identity_from_raw(
                    raw, request_id=adapter_stream.request_id()
                ),
            )
            response_outcome = self._bound_adapter.interpret(raw)
        except Exception as exc:  # noqa: BLE001 (an assembly failure settles the request like an iteration failure)
            return await self._error_after_stream_failure(exc, stage="assembly")
        return self._outcome_from_response(
            response_outcome, ended_at_monotonic_seconds=ended_at_monotonic_seconds
        )

    def _outcome_from_response(
        self,
        response_outcome: ResponseOutcome[OutputT | None],
        *,
        ended_at_monotonic_seconds: float,
    ) -> GenerationOutcome[OutputT]:
        """Record the staged response before building its outcome, preserving the response and billing.

        Return errors so `final` stores every outcome before concluding the input.
        """
        if response_outcome.kind == "provider_failed_transiently":
            error = TransientError(
                response_outcome.error_text, pauses_quota=response_outcome.pauses_quota
            )
            self._shared_backoff._record(
                RequestFailure(
                    kind="transient",
                    pauses_quota=response_outcome.pauses_quota,
                    retry_after_seconds=None,
                )
            )
            self._ledger.record_ending_at(
                ended_at_monotonic_seconds,
                error=error,
                assistant_message=response_outcome.assistant_message,
            )
            retry_unavailable = GenerationError(
                record=PlainErrorRecord(
                    kind="retry_unavailable_error",
                    request_history=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                ),
                request_params=self._request_params,
                request_provider_data=self._ledger.request_provider_data,
            )
            retry_unavailable.__cause__ = error
            return retry_unavailable
        self._ledger.record_ending_at(
            ended_at_monotonic_seconds,
            error=None,
            assistant_message=response_outcome.assistant_message,
        )
        return _generation_outcome_from_response_outcome(
            response_outcome,
            request_history=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
            request_provider_data=self._ledger.request_provider_data,
            request_params=self._request_params,
            splits_on_tool_calls=self._splits_on_tool_calls,
        )
