"""The stream handle.

`StreamHandle` opens inside `async with` and yields `StreamItem` values.
`final` drains remaining items and returns the assembled result.
Open failures can retry; failures after opening cannot retry.
One `SharedBackoff.admitted` block spans the open stream's lifetime.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
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
from langchaint.call import _CallLedger
from langchaint.exceptions import (
    AbandonedCallErrorRecord,
    GenerationError,
    InvalidRequestErrorRecord,
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    StreamProtocolError,
    TimedOutErrorRecord,
    TransientError,
    _terminal_error_record,
)
from langchaint.messages import Message
from langchaint.pricing import ProviderBilling
from langchaint.response import (
    CallResult,
    GenerateResult,
    Response,
    ToolCallTurn,
    _abandoned_call_error,
    _call_result_from_response_outcome,
)
from langchaint.shared_backoff import (
    Admission,
    PauseAll,
    PauseAllDoNotRetry,
    PrivateBackoff,
    RetryThisOne,
    SharedBackoff,
    Verdict,
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


class StreamHandle[OutputT, ToolTurnT = Never]:
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
    ) -> None:
        """Store the request."""
        self._adapter = adapter
        self._bound_adapter = bound_adapter
        self._messages = messages
        self._shared_backoff = shared_backoff
        self._max_attempts = max_attempts
        self._private_backoff = PrivateBackoff(shared_backoff)
        self._timeout_seconds = timeout_seconds
        self._splits_tool_call_turns = splits_tool_call_turns
        self._deadline: asyncio.Timeout | None = None
        self.abandoned: AbandonedCallErrorRecord | None = None
        """The interrupted call account, or `None`.

        Cancellation sets this value before the caller receives `asyncio.CancelledError`.
        A success or `GenerationError` leaves this value as `None`.
        An expired `timeout_seconds` raises `GenerationError` and leaves this value as `None`.
        """
        self._adapter_stream: AdapterStream | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._ledger = _CallLedger(model=adapter.model, provider_name=adapter.provider_name)
        self._admission: Admission | None = None
        self._ended_at_monotonic_seconds: float | None = None
        self._conclusion: GenerateResult[OutputT] | Exception | None = None
        self._conclusion_carried_the_call = False
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
            RuntimeError: This handle was already entered.
            ParserContractError: the adapter's parse violated its contract on a failed open.
        """
        if self._state != "unopened":
            raise RuntimeError(_ALREADY_ENTERED_MESSAGE)
        self._state = "open"
        self._deadline = asyncio.timeout(self._timeout_seconds)
        await self._deadline.__aenter__()
        self._ledger.start_call()
        try:
            await self._open_stream_with_retries()
        except BaseException as exc:
            # `__aexit__` does not run when `__aenter__` raises.
            # Finish the call, exit admission, and close the deadline here.
            # An open deadline would retain a timer that could cancel this task after this operation.
            # Record abandonment here because no other frame sees cancellation during the open.
            self._state = "finished"
            await self._exit_admission(None)
            billing_in_flight = self._billing_reported()
            if await self._close_deadline(exc):
                raise self._timed_out_error(billing_in_flight) from None
            if isinstance(exc, asyncio.CancelledError):
                self._set_abandoned(billing_in_flight)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the deadline, connection, and admission.

        Cancellation sets `abandoned` unless a result already records the call.
        An expired `timeout_seconds` raises `GenerationError` instead.

        Raises:
            GenerationError: `timeout_seconds` expires before the block finishes.
            BaseException: Closing the adapter stream raises a non-`Exception` value.
        """
        self._state = "finished"
        # Read before the close, which drops the stream that reports it.
        billing_in_flight = (
            self._billing_reported() if isinstance(exc, asyncio.CancelledError) else None
        )
        timed_out = await self._close_deadline(exc)
        try:
            await self._close_adapter_stream()
        finally:
            if isinstance(exc, asyncio.CancelledError) and not timed_out:
                self._set_abandoned(billing_in_flight)
        if timed_out:
            raise self._timed_out_error(billing_in_flight) from None

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

    def _timed_out_error(self, billing_in_flight: ProviderBilling | None) -> GenerationError:
        """Freeze the ledger into this call's deadline account."""
        return _abandoned_call_error(TimedOutErrorRecord, self._ledger, billing_in_flight)

    def _billing_reported(self) -> ProviderBilling | None:
        """Ask the open stream what the provider has reported, or None where it reported nothing.

        Call before the connection closes, because closing drops the stream this asks.
        """
        if self._adapter_stream is None:
            return None
        return self._adapter_stream.billing_reported()

    def _set_abandoned(self, billing_in_flight: ProviderBilling | None) -> None:
        """Set `abandoned` when no conclusion already accounts for the call.

        Include billing reported by the interrupted attempt.
        """
        if self._conclusion_carried_the_call:
            return
        call, _ = self._ledger.freeze_with_cut_off(billing_in_flight)
        self.abandoned = AbandonedCallErrorRecord(call=call)

    async def _exit_admission(self, exc: BaseException | None) -> Verdict | None:
        """Exit the held admission and return its `Verdict`.

        Repeated calls return `None`.

        Raises:
            ParserContractError: `Adapter.parse` violates its contract.
        """
        if self._admission is None:
            return None
        admission, self._admission = self._admission, None
        _ = await admission.__aexit__(
            type(exc) if exc is not None else None,
            exc,
            exc.__traceback__ if exc is not None else None,
        )
        return admission.verdict

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
            _ = await self._exit_admission(None)

    def _transient_error(
        self, exc: Exception, message: str, verdict: Verdict | None
    ) -> TransientError:
        """Wrap one attempt failure as the TransientError its attempt record carries.

        Return an existing `TransientError` unchanged.
        This preserves its `retry_after_seconds`, `is_rate_limit`, and message.
        """
        if isinstance(exc, TransientError):
            return exc
        match verdict:
            case PauseAll(retry_after=retry_after):
                wrapped = TransientError(
                    message, retry_after_seconds=retry_after, is_rate_limit=True
                )
            case RetryThisOne(retry_after=retry_after):
                wrapped = TransientError(message, retry_after_seconds=retry_after)
            case _:
                wrapped = TransientError(message)
        wrapped.__cause__ = exc
        return wrapped

    def _record_transient_error(
        self, wrapped: TransientError, billing: ProviderBilling | None = None
    ) -> None:
        """Record one transient failure as an attempt.

        The `admitted()` block already recorded the failure's verdict.
        A rate limit pauses every request using this `SharedBackoff`.
        This stream may still lack a safe retry.
        `billing` is the provider-reported attempt billing.
        """
        self._ledger.record(error=wrapped, assistant_message=None, billing=billing)

    async def _backoff_or_exhaust(self, exc: Exception, verdict: Verdict | None) -> None:
        """Wait before the next open attempt, as the failure's verdict asks.

        `PauseAll` relies on the next `admitted()` wait.
        Every other failure waits for `PrivateBackoff`.
        `RetryThisOne.retry_after` sets that wait's floor.

        Raises:
            GenerationError: the recorded failure spent the last attempt.
        """
        if self._ledger.attempts >= self._max_attempts:
            raise GenerationError(
                record=RetriesExhaustedErrorRecord(call=self._ledger.freeze()),
                request=self._request,
                provider_attempts=self._ledger.provider_attempts,
            ) from exc
        match verdict:
            case PauseAll():
                return
            case RetryThisOne(retry_after=retry_after):
                await asyncio.sleep(self._private_backoff.next_wait(retry_after))
            case _:
                await asyncio.sleep(self._private_backoff.next_wait(None))

    def _terminal_error_or_none(
        self,
        exc: Exception,
        *,
        verdict: Verdict | None,
        stream_billing: ProviderBilling | None,
    ) -> GenerationError | None:
        """Map one attempt failure to its terminal error.

        Record `stream_billing` when the stream reached the provider.
        """
        request_id = self._adapter.request_id_from_error(exc)
        if request_id is None and self._adapter_stream is not None:
            request_id = self._adapter_stream.request_id()
        self._ledger.note_request_id(request_id)
        if verdict is not None:
            if verdict.kind not in ("do_not_retry", "pause_all_do_not_retry"):
                return None
        elif isinstance(exc, TransientError) or self._adapter.classify(exc) == "transient":
            return None
        # `PauseAllDoNotRetry` becomes `declared_final` without `classify()`.
        # A provider directive states this request will not succeed.
        # `GenerationError` names that outcome.
        # `classify()` could otherwise return `invalid_request` for status 429.
        classification = (
            "declared_final"
            if isinstance(verdict, PauseAllDoNotRetry)
            else self._adapter.classify(exc)
        )
        if (
            classification in ("invalid_request", "declared_final")
            or self._adapter_stream is not None
        ):
            self._ledger.record(error=None, assistant_message=None, billing=stream_billing)
        return GenerationError(
            record=_terminal_error_record(
                classification, reason=str(exc), call=self._ledger.freeze()
            ),
            request=self._request,
            provider_attempts=self._ledger.provider_attempts,
        )

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
            ParserContractError: `Adapter.parse` violates its contract.
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
                verdict = await self._exit_admission(exc)
                terminal = self._terminal_error_or_none(exc, verdict=verdict, stream_billing=None)
                if terminal is not None:
                    raise terminal from exc
                self._record_transient_error(self._transient_error(exc, str(exc), verdict))
                await self._backoff_or_exhaust(exc, verdict)
                continue
            except BaseException:
                # `CancelledError` is a `BaseException` that the clause above does not catch.
                # Exit here to return the permit at the same point on each failing path.
                _ = await self._exit_admission(None)
                raise
            self._adapter_stream = opened
            self._items = self._adapter_stream.items()

    async def __anext__(self) -> StreamItem:
        """Return the next item.

        Raises:
            GenerationError: The open stream fails transiently.
                The adapter classifies an item error as an invalid request.
                The provider declares an item error terminal.
                The adapter cannot classify an item exception.
            StreamProtocolError: The event stream ends without a terminal event.
            ParserContractError: `Adapter.parse` violates its contract.
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
            if self._conclusion is None and isinstance(exc, Exception):
                # Cancellation destroys the frames that could observe the call.
                # `abandoned` records that condition.
                self._conclusion = exc
                self._conclusion_carried_the_call = isinstance(exc, GenerationError)
                await self._close_deadline(exc)
            raise

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
            _ = await self._exit_admission(None)
            raise
        except StreamProtocolError:
            await self._close_adapter_stream()
            raise
        except Exception as exc:
            stream_billing = self._billing_reported()
            verdict = await self._exit_admission(exc)
            terminal = self._terminal_error_or_none(
                exc, verdict=verdict, stream_billing=stream_billing
            )
            if terminal is not None:
                await self._close_adapter_stream()
                raise terminal from exc
            wrapped = self._transient_error(
                exc, f"open stream failed during iteration: {exc}", verdict
            )
            self._record_transient_error(wrapped, stream_billing)
            await self._close_adapter_stream()
            raise GenerationError(
                record=RetryUnavailableErrorRecord(call=self._ledger.freeze()),
                request=self._request,
                provider_attempts=self._ledger.provider_attempts,
            ) from wrapped
        except BaseException:
            _ = await self._exit_admission(None)
            raise
        self._ledger.stamp_first_item()
        return item

    @overload
    async def final(self: "StreamHandle[OutputT, Never]") -> Response[OutputT]: ...
    @overload
    async def final(self) -> "Response[OutputT] | ToolCallTurn[OutputT]": ...
    async def final(self) -> Response[OutputT] | ToolCallTurn[OutputT]:
        """Drain remaining items and return the stored result.

        Repeated calls return or raise the same conclusion without reading the stream again.
        A response with no output becomes a terminal `GenerationError`.

        Raises:
            StreamProtocolError: The event stream ends without a terminal event.
            GenerationError: The adapter or provider reports a terminal failure.
                The assembled response reports a terminal result.
                The open stream fails transiently.
                The adapter cannot classify an item exception.
            ParserContractError: `Adapter.parse` violates its contract.
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
                raw = await adapter_stream.final()
                self._ledger.stage_response(
                    raw=raw,
                    billing=self._bound_adapter.billing_from_raw(raw),
                    identity=self._bound_adapter.identity_from_raw(
                        raw, request_id=adapter_stream.request_id()
                    ),
                )
                self._conclusion = self._conclude(
                    self._bound_adapter.interpret(raw),
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                )
            except BaseException as exc:
                # Keep interpretation inside the `try` because `_conclude` records before returning.
                # A raise after that record could let a second call record the attempt again.
                if isinstance(exc, Exception):
                    self._conclusion = exc
                    await self._close_deadline(exc)
                raise
            self._conclusion_carried_the_call = True
            await self._close_deadline(None)
        if isinstance(self._conclusion, (Response, ToolCallTurn)):
            return self._conclusion
        raise self._conclusion

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
