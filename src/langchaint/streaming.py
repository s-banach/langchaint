"""The stream handle.

A StreamHandle is three things at once: an async iterator of stream items (text chunks and completed tool calls),
the source of the assembled Response via final(),
and an async context manager whose entry opens the request and whose exit closes it.
A handle is unusable outside its `async with` block, so neither iterating nor final() can start a request.
Assembling the turn and reading what it produced live behind AdapterStream.final();
the handle owns retry, pacing, and accounting.
Connection failures before the first yielded item are retried under the RateLimiter;
after the first yielded item nothing is retried,
because replaying items the caller already consumed would duplicate output.
A transient failure the item iterator raises is recorded and fed back to the RateLimiter on both
paths, so a rate limit paces the account whether or not the stream that hit it could still reopen.
A transient failure past the first item ends the call as RetryUnavailableError, as does one the
assembled response reports.
An open stream holds one RateLimiter in-flight slot from opening until the stream closes or exhausts,
so long-lived streams count against max_in_flight for their whole life.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Literal, assert_never

from pydantic import BaseModel

from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    BoundAdapter,
    ContextWindowExceeded,
    EmptyTurn,
    InvalidRequest,
    MaxCompletionTokensExceeded,
    ProviderFailedTerminally,
    ProviderFailedTransiently,
    Refusal,
    ResponseOutcome,
    SchemaViolation,
    StreamItem,
    UnfinishedTurn,
)
from langchaint.call import _CallLedger
from langchaint.exceptions import (
    ContextWindowExceededError,
    EmptyTurnError,
    GenerationError,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    ProviderDeclaredFinalError,
    ProviderFailedTerminallyError,
    RefusalError,
    RetriesExhaustedError,
    RetryUnavailableError,
    SchemaViolationError,
    StreamProtocolError,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
    _extract_transient_errors,
)
from langchaint.messages import Message
from langchaint.rate_limiter import Admission, Backoff, RateLimiter
from langchaint.response import AbandonedCallLog, Response, _append_abandoned_call
from langchaint.usage import ZERO_USAGE, Usage

type _State = Literal["unopened", "open", "finished"]

_logger = logging.getLogger("langchaint.streaming")

_UNOPENED_MESSAGE = "stream not open: enter the handle with `async with` before using it"
_FINISHED_MESSAGE = "stream is finished: call stream_one again for a new one"
_ALREADY_ENTERED_MESSAGE = "stream already entered: call stream_one again for a new one"


class StreamHandle[OutputT]:
    """One stream: an item iterator, a Response source, a context manager.

    Iterate for items as they arrive; await final() at any point in the block to drain silently and get the Response.
    The request opens on entry, so open failures surface there rather than at the first item.

    max_attempts bounds one phase only: reaching the first item. Past it nothing is retried,
    whatever the budget still holds: a transient failure an item pull raises, and one the assembled
    response reports, each end the call as RetryUnavailableError. Reopening is the application's to
    do, by calling stream_one again.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT],
        conversation: Sequence[Message],
        rate_limiter: RateLimiter,
        abandoned_call_log: AbandonedCallLog | None,
    ) -> None:
        """Store the request; called by BoundLLM.stream_one only."""
        self._adapter = adapter
        self._bound_adapter = bound_adapter
        self._conversation = conversation
        self._rate_limiter = rate_limiter
        self._abandoned_call_log = abandoned_call_log
        self._adapter_stream: AdapterStream | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._ledger = _CallLedger(model=adapter.model, provider_name=adapter.provider_name)
        self._admission: Admission | None = None
        self._yielded_any = False
        self._ended_at_monotonic_seconds: float | None = None
        self._conclusion: Response[OutputT] | Exception | None = None
        """What concluded the call: the Response, or the error that ended it; None until it ends."""
        self._conclusion_carried_the_call = False
        """Whether that conclusion gave the caller an account of this call; see _append_abandoned_call."""
        self._state: _State = "unopened"

    async def __aenter__(self) -> "StreamHandle[OutputT]":
        """Open the request and return self.

        Raises:
            InvalidRequestError: the adapter reported the conversation as InvalidRequest, or the open
                failure was classified as a rejection of the request.
            ProviderDeclaredFinalError: the provider declared the open failure final.
            UnknownExceptionError: the adapter could not place the open failure.
            RetriesExhaustedError: the opens spent the retry budget.
            RuntimeError: this handle was already entered; build a new one with stream_one.
        """
        if self._state != "unopened":
            raise RuntimeError(_ALREADY_ENTERED_MESSAGE)
        self._state = "open"
        self._ledger.start_call()
        try:
            await self._open_stream_with_retries()
        except BaseException as exc:
            # __aexit__ does not run when __aenter__ raises, so finish and release here.
            # _open_stream_with_retries returns the slot on every path that raises, so this release
            # covers only the case where it never acquired one; it is idempotent either way.
            # The abandonment is recorded here because no other frame sees a cancellation that lands
            # during the open.
            self._state = "finished"
            self._release_slot()
            if isinstance(exc, asyncio.CancelledError):
                self._append_abandoned_call(self._usage_reported())
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying connection and finish the handle.

        A CancelledError exiting the block appends an AbandonedCall to the handle's
        abandoned_call_log, when one was given and no conclusion gave the caller an account of the
        call (_append_abandoned_call states which conclusions do).
        A consumer that leaves the block early without an exception chose to walk away in live
        code, so only the cancellation, which destroys the frames that could have observed the
        stream, gets a record.

        Raises:
            BaseException: the adapter stream's close raised something that is not an Exception;
                it propagates in place of whatever was unwinding the block.
        """
        self._state = "finished"
        # Read before the close, which drops the stream that reports it.
        usage_in_flight = (
            self._usage_reported() if isinstance(exc, asyncio.CancelledError) else ZERO_USAGE
        )
        try:
            await self._close_adapter_stream()
        finally:
            # The append runs after the close whatever the close does, so the record is written
            # even when a BaseException comes out of it, which _close_adapter_stream does not catch.
            if isinstance(exc, asyncio.CancelledError):
                self._append_abandoned_call(usage_in_flight)

    def _usage_reported(self) -> Usage:
        """Ask the open stream what the provider has reported, folding nothing reported to ZERO_USAGE.

        Call before the connection closes, because closing drops the stream this asks.
        An adapter stream that reports None folds to ZERO_USAGE, the same fold usage_from_raw
        applies to a response that reports no counters: both mean the provider stated nothing.
        """
        if self._adapter_stream is None:
            return ZERO_USAGE
        usage_reported = self._adapter_stream.usage_reported()
        return usage_reported if usage_reported is not None else ZERO_USAGE

    def _append_abandoned_call(self, usage_in_flight: Usage) -> None:
        """Record the abandonment, unless no log was given or the conclusion accounted for the call.

        A Response and a GenerationError each hand the caller this call's CallRecord, naming
        the model and the attempts to reconcile against.
        Appending after one would report the same call twice and mislabel a concluded call as an
        in-flight abandonment.
        A StreamProtocolError hands over neither, so a cancellation following one still gets a
        record: it is the only account of the stream that was opened.
        usage_settled holds what the settled records carried, and usage_in_flight what the adapter
        could state of the attempt still open when the cancellation arrived.
        """
        if self._conclusion_carried_the_call:
            return
        _append_abandoned_call(self._abandoned_call_log, self._ledger.freeze(), usage_in_flight)

    def _release_slot(self) -> None:
        if self._admission is not None:
            self._rate_limiter.release(self._admission)
            self._admission = None

    async def _close_adapter_stream(self) -> None:
        """Close the provider connection and return the in-flight slot, whatever the close does.

        A close failure is logged rather than raised, because the request it belonged to has already
        ended and the exception would only displace the Response or the error the caller came for.
        The release sits in a finally rather than after the handler, so a BaseException out of the
        close returns the slot too. Either way, an admission this method skips is gone from the
        shared budget for the process's life.
        The stream is dropped before the close is awaited, so a teardown that fails is attempted
        once: __aexit__, which closes again after the paths that close mid-stream, finds nothing to
        close.
        """
        adapter_stream = self._adapter_stream
        self._adapter_stream = None
        self._items = None
        try:
            if adapter_stream is not None:
                await adapter_stream.close()
        except Exception:
            _logger.warning(
                "closing the provider stream raised; the in-flight slot was returned",
                exc_info=True,
            )
        finally:
            self._release_slot()

    def _transient_error(self, exc: Exception, message: str) -> TransientError:
        """Wrap one attempt error as the TransientError that carries its retry directive.

        An exception that already is a TransientError is its own wrapper, so an adapter that stated
        retry_after_seconds and is_rate_limit itself keeps them; message is then unused, because
        replacing the adapter's own text would lose what it said.
        """
        if isinstance(exc, TransientError):
            return exc
        wrapped = TransientError(
            message,
            retry_after_seconds=self._adapter.retry_after_seconds(exc),
            is_rate_limit=self._adapter.classify(exc) == "rate_limit",
        )
        wrapped.__cause__ = exc
        return wrapped

    def _record_transient_error(
        self, wrapped: TransientError, usage: Usage = ZERO_USAGE
    ) -> Backoff:
        """Record one transient failure as an attempt and register it with the RateLimiter.

        Called while the failing attempt's admission is still held;
        register_transient_error raises RuntimeError for one already released.
        A failure this stream cannot retry still registers, because the pause a rate limit sets
        protects the whole account: losing it because this one stream is past reopening would leave
        every other caller sending into the limit.

        usage is what the provider reported for the attempt.

        Returns:
            The Backoff to sleep before the next open attempt;
            its delay is drawn once, so it equals any account-wide pause it set.
            A caller that will not reopen drops it.
        """
        self._ledger.record(error=wrapped, assistant_message=None, usage=usage)
        assert self._admission is not None
        return self._rate_limiter.register_transient_error(
            self._admission, _extract_transient_errors(self._ledger.attempt_records)
        )

    async def _backoff_or_exhaust(self, exc: Exception, backoff: Backoff) -> None:
        """Back off before the next open attempt; call after the failed attempt's release.

        backoff is the value _record_transient_error returned for this failure,
        so the sleep matches the account-wide pause the same draw set.

        Raises:
            RetriesExhaustedError: the recorded failure spent the last attempt.
        """
        if self._ledger.attempts >= self._rate_limiter.max_attempts:
            raise RetriesExhaustedError(call=self._ledger.freeze()) from exc
        await backoff.sleep()

    def _non_retriable_or_none(
        self, exc: Exception, stream_usage: Usage | None
    ) -> GenerationError | None:
        """Map one attempt error to the non-retriable error to propagate, or None when transient.

        Reached only for exceptions, which by the adapter contract are attempts the adapter read no
        outcome from: what it did read it reports as an AttemptOutcome member, which this handle
        matches instead.

        stream_usage is what the open stream reported, and None where the failure preceded any
        stream. A provider that reports counters before its first item has already billed by the
        time one of these errors arrives, so the reading is what keeps that attempt accountable.
        """
        if isinstance(exc, TransientError):
            return None
        usage = ZERO_USAGE if stream_usage is None else stream_usage
        classification = self._adapter.classify(exc)
        if classification == "invalid_request":
            # Adapter.classify returns invalid_request only for a request the provider rejected,
            # so it went out and gets a record.
            self._ledger.record(error=None, assistant_message=None, usage=usage)
            return self._invalid_request_error(f"the provider rejected the request: {exc}", exc)
        if classification == "declared_final":
            # The provider answered, so the attempt gets a record.
            self._ledger.record(error=None, assistant_message=None, usage=usage)
            return ProviderDeclaredFinalError(error=exc, call=self._ledger.freeze())
        if classification == "unknown_exception":
            if stream_usage is not None:
                # The stream was open, so langchaint can say the attempt reached the provider and
                # what that provider reported for it; the class records nothing where it cannot.
                self._ledger.record(error=None, assistant_message=None, usage=stream_usage)
            return UnknownExceptionError(error=exc, call=self._ledger.freeze())
        return None

    def _invalid_request_error(self, reason: str, cause: Exception | None) -> InvalidRequestError:
        """Build the row-shaped InvalidRequestError for this handle, chained to cause when there is one.

        cause is None for a InvalidRequest outcome: the adapter reported that the conversation cannot be
        sent, and no exception was involved.
        """
        invalid_request = InvalidRequestError(reason=reason, call=self._ledger.freeze())
        invalid_request.__cause__ = cause
        return invalid_request

    def __aiter__(self) -> "StreamHandle[OutputT]":
        """Return self; the handle is its own iterator."""
        return self

    async def _open_stream_with_retries(self) -> None:
        """Open one adapter stream, retrying transient failures under the limiter.

        A fresh admission is acquired for each attempt and released before the backoff sleep,
        so a waiting task never holds capacity while this one backs off.
        A successful open registers the admission with the limiter,
        ending any recovery this handle's probe was serving,
        so a stream slow to first token cannot stall the shared account's admission.
        The slot stays held for the stream's whole life; only recovery ends here, not the in-flight hold.
        Every failing path out of an attempt returns the admission, cancellation included.

        Raises:
            InvalidRequestError: the adapter reported the conversation as InvalidRequest, or the open
                failure was classified as a rejection of the request.
            ProviderDeclaredFinalError: the provider declared the open failure final.
            UnknownExceptionError: the adapter could not place the open failure.
            RetriesExhaustedError: the attempts spent the retry budget.
        """
        while self._adapter_stream is None:
            self._admission = await self._rate_limiter.acquire()
            self._ledger.start_attempt()
            try:
                opened = await self._bound_adapter.open_stream(self._conversation)
            except Exception as exc:
                non_retriable = self._non_retriable_or_none(exc, None)
                if non_retriable is not None:
                    self._release_slot()
                    raise non_retriable from exc
                backoff = self._record_transient_error(self._transient_error(exc, str(exc)))
                self._release_slot()
                await self._backoff_or_exhaust(exc, backoff)
                continue
            except BaseException:
                # CancelledError is a BaseException the clause above does not catch. Releasing here
                # returns the admission at the same point on every failing path, so no caller's
                # unwind is what the shared budget depends on.
                self._release_slot()
                raise
            if isinstance(opened, InvalidRequest):
                self._release_slot()
                raise self._invalid_request_error(opened.reason, None)
            self._adapter_stream = opened
            self._items = self._adapter_stream.items()
            assert self._admission is not None
            self._rate_limiter.register_success(self._admission)

    async def __anext__(self) -> StreamItem:
        """Return the next item.

        Every error but StopAsyncIteration finishes the handle, so nothing later reopens the request,
        and every such error but a cancellation is the call's conclusion, which final() replays.

        Raises:
            RetryUnavailableError: the stream failed after items were yielded, so no retry was
                available; __cause__ is the adapter's verdict on that failure.
            InvalidRequestError: the adapter reported a reopened conversation as InvalidRequest, or
                classified an item or reopen error as a rejection of the request.
            ProviderDeclaredFinalError: the provider declared an item or reopen error final.
            UnknownExceptionError: the adapter could not place an item or reopen exception.
            RetriesExhaustedError: a pre-first-item failure spent the retry budget.
            StreamProtocolError: the provider's event stream ended without a terminal event; propagates unchanged.
            StopAsyncIteration: the stream is exhausted.
            RuntimeError: the handle is unopened or finished.
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
                # A cancellation is not a conclusion: it destroys the frames that could have
                # observed the call, which is what an AbandonedCall records instead.
                self._conclusion = exc
                self._conclusion_carried_the_call = isinstance(exc, GenerationError)
            raise

    async def _next_item(self) -> StreamItem:
        """Pull the next item, reopening for a transient failure that precedes the first item.

        Raises what __anext__ documents.
        """
        while True:
            assert self._items is not None
            try:
                item = await self._items.__anext__()
            except StopAsyncIteration:
                if self._ended_at_monotonic_seconds is None:
                    self._ended_at_monotonic_seconds = time.monotonic()
                self._release_slot()
                raise
            except StreamProtocolError:
                await self._close_adapter_stream()
                raise
            except Exception as exc:
                # Read before any close, which drops the stream that reports it. A provider that
                # reports counters before its first item has already billed for this attempt,
                # whichever of the paths below it takes.
                stream_usage = self._usage_reported()
                non_retriable = self._non_retriable_or_none(exc, stream_usage)
                if non_retriable is not None:
                    await self._close_adapter_stream()
                    raise non_retriable from exc
                if self._yielded_any:
                    wrapped = self._transient_error(
                        exc, f"stream failed after items were yielded: {exc}"
                    )
                    self._record_transient_error(wrapped, stream_usage)
                    await self._close_adapter_stream()
                    raise RetryUnavailableError(call=self._ledger.freeze()) from wrapped
                backoff = self._record_transient_error(
                    self._transient_error(exc, str(exc)), stream_usage
                )
                await self._close_adapter_stream()
                await self._backoff_or_exhaust(exc, backoff)
                await self._open_stream_with_retries()
                continue
            except BaseException:
                # CancelledError is a BaseException the clauses above do not catch.
                # Cancelling an item pull in its own task leaves the block open, so waiting for __aexit__
                # would strand this slot. Return it, then let the cancellation propagate.
                self._release_slot()
                raise
            self._yielded_any = True
            return item

    async def final(self) -> Response[OutputT]:
        """Drain any remaining items silently and return the Response.

        Idempotent once a conclusion exists: the call's conclusion is stored once, whether this
        method produced it, a caller's own iteration produced it, or the adapter stream raised it.
        Every later call returns or raises it again without asking the adapter stream anything.
        Without that store, a second call would append a second AttemptRecord for the one request made.
        A 200 that produced no output is detected only here, when the adapter reads the assembled
        message: the adapter reports which one it was and this method builds the GenerationError from
        it, without retrying;
        it reaches the caller carrying the attempt records this handle built.

        Raises:
            StreamProtocolError: the provider's event stream ended without a terminal event.
            InvalidRequestError: draining the stream hit an item or reopen error the adapter
                classified as a rejection of the request, or a reopened conversation the adapter
                reported as InvalidRequest.
            ProviderDeclaredFinalError: draining hit an item or reopen error the provider declared final.
            UnknownExceptionError: draining hit an item or reopen exception the adapter could not place.
            RetriesExhaustedError: draining the stream spent the retry budget on a pre-first-item failure.
            RefusalError: the adapter reported the assembled response as Refusal,
                carrying this handle's attempt records.
            MaxCompletionTokensExceededError: the adapter reported it as MaxCompletionTokensExceeded; likewise.
            EmptyTurnError: the adapter reported it as EmptyTurn; likewise.
            SchemaViolationError: the adapter reported it as SchemaViolation; likewise.
            ContextWindowExceededError: the adapter reported it as ContextWindowExceeded; likewise.
            UnfinishedTurnError: the adapter reported it as UnfinishedTurn; likewise.
            ProviderFailedTerminallyError: the adapter reported it as ProviderFailedTerminally; likewise.
            RetryUnavailableError: the adapter reported the assembled response as
                ProviderFailedTransiently; that response ends the stream, so no retry was available.
                That 200's billing and turn are on its attempt record.
            RuntimeError: the handle is unopened, or it is finished with no conclusion stored
                (drained to exhaustion, then left the block).
        """
        if self._conclusion is None:
            if self._state != "open":
                raise RuntimeError(
                    _UNOPENED_MESSAGE if self._state == "unopened" else _FINISHED_MESSAGE
                )
            async for _ in self:
                pass
            assert self._adapter_stream is not None
            ended_at_monotonic_seconds = (
                time.monotonic()
                if self._ended_at_monotonic_seconds is None
                else self._ended_at_monotonic_seconds
            )
            try:
                raw = await self._adapter_stream.final()
                # Staged before interpret reads the response, so what the attempt billed is on the
                # ledger from the moment it is known.
                self._ledger.stage_receipt(raw=raw, usage=self._bound_adapter.usage_from_raw(raw))
                self._conclusion = self._conclude(
                    self._bound_adapter.interpret(raw),
                    raw=raw,
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                )
            except BaseException as exc:
                # A cancellation is not a conclusion, the same rule __anext__ applies:
                # it destroys the frames that could have observed the call rather than ending it.
                # The interpretation is inside the try because every _conclude case records the
                # attempt first, so a raise past that record would let a second call record it again.
                if isinstance(exc, Exception):
                    self._conclusion = exc
                raise
            # Every _conclude case builds its result off the frozen CallRecord, so the caller has an
            # account of the call whichever one it was.
            self._conclusion_carried_the_call = True
        if isinstance(self._conclusion, Response):
            return self._conclusion
        raise self._conclusion

    def _conclude(
        self,
        outcome: ResponseOutcome[OutputT],
        *,
        raw: BaseModel,
        ended_at_monotonic_seconds: float,
    ) -> Response[OutputT] | GenerationError:
        """Build what this outcome concludes the call with: the Response, or the error to raise.

        Returns the error rather than raising it, so no case can conclude the call without being stored.
        Every case closes the staged receipt into this attempt's record first, so the call it freezes
        into the result holds the terminal response and what it billed.
        raw is that response, which only a success carries onward.
        """
        match outcome:
            case AdapterResult():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return Response(
                    output=outcome.output,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    raw=raw,
                    stop_reason=outcome.stop_reason,
                    assistant_message=outcome.assistant_message,
                )
            case Refusal():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return RefusalError(call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds))
            case MaxCompletionTokensExceeded():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return MaxCompletionTokensExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds)
                )
            case EmptyTurn():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return EmptyTurnError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds)
                )
            case SchemaViolation():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return SchemaViolationError(
                    validation_error_json=outcome.validation_error_json,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                )
            case ContextWindowExceeded():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return ContextWindowExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds)
                )
            case UnfinishedTurn():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return UnfinishedTurnError(
                    reason=outcome.reason,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                )
            case ProviderFailedTerminally():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return ProviderFailedTerminallyError(
                    reason=outcome.reason,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                )
            case ProviderFailedTransiently():
                failure = TransientError(outcome.reason, is_rate_limit=outcome.is_rate_limit)
                self._ledger.record_ending_at(
                    ended_at_monotonic_seconds,
                    error=failure,
                    assistant_message=outcome.assistant_message,
                )
                retry_unavailable = RetryUnavailableError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds)
                )
                retry_unavailable.__cause__ = failure
                return retry_unavailable
            case _ as unhandled:
                assert_never(unhandled)

    def _record_completed_attempt(
        self,
        outcome: AdapterResult[OutputT]
        | Refusal
        | MaxCompletionTokensExceeded
        | EmptyTurn
        | SchemaViolation
        | ContextWindowExceeded
        | UnfinishedTurn
        | ProviderFailedTerminally,
        ended_at_monotonic_seconds: float,
    ) -> None:
        """Record the attempt that reached a billable 200, whatever the adapter made of it.

        error is None on every member here: the request itself succeeded, and what the adapter made of
        the response is the item's outcome, not this attempt's failure.
        ProviderFailedTransiently is the one member _conclude records itself, because its record
        carries the TransientError the failure was classified as.
        """
        self._ledger.record_ending_at(
            ended_at_monotonic_seconds,
            error=None,
            assistant_message=outcome.assistant_message,
        )
