"""The stream handle.

A StreamHandle is three things at once: an async iterator of stream items (text chunks and completed tool calls),
the source of the assembled Response via final(),
and an async context manager whose entry opens the request and whose exit closes it.
A handle is unusable outside its `async with` block, so neither iterating nor final() can start a request.
Assembly and structured-output parsing live in the SDK behind AdapterStream.final();
the handle owns retry, pacing, and accounting.
Connection failures before the first yielded item are retried under the RateLimiter;
after the first yielded item nothing is retried,
because replaying items the caller already consumed would duplicate output.
A transient failure the item iterator raises is recorded and fed back to the RateLimiter on both
paths, so a rate limit paces the account whether or not the stream that hit it could still reopen.
An open stream holds one RateLimiter in-flight slot from opening until the stream closes or exhausts,
so long-lived streams count against max_in_flight for their whole life.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Literal, assert_never

from langchaint.adapter import (
    Adapter,
    AdapterResult,
    AdapterStream,
    BoundAdapter,
    NotSendable,
    Refused,
    ResponseOutcome,
    StreamItem,
    Truncated,
    Unparsed,
)
from langchaint.call import _CallLedger
from langchaint.exceptions import (
    GenerationError,
    InvalidRequestError,
    MaxCompletionTokensExceededError,
    RefusalError,
    RetriesExhaustedError,
    StreamProtocolError,
    TransientError,
    UnrecognizedError,
    _extract_transient_errors,
)
from langchaint.messages import Message
from langchaint.rate_limiter import Admission, RateLimiter
from langchaint.response import AbandonedCallLog, Response, _append_abandoned_call
from langchaint.usage import ZERO_USAGE

type _State = Literal["unopened", "open", "finished"]

_logger = logging.getLogger("langchaint.streaming")

_UNOPENED_MESSAGE = "stream not open: enter the handle with `async with` before using it"
_FINISHED_MESSAGE = "stream is finished: call stream_one again for a new one"
_ALREADY_ENTERED_MESSAGE = "stream already entered: call stream_one again for a new one"


class StreamHandle[OutputT]:
    """One stream: an item iterator, a Response source, a context manager.

    Iterate for items as they arrive; await final() at any point in the block to drain silently and get the Response.
    The request opens on entry, so open failures surface there rather than at the first item.
    A transient failure after the first yielded item propagates as TransientError; retry by calling stream_one again.
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
        self._adapter_stream: AdapterStream[OutputT] | None = None
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
            InvalidRequestError: the adapter reported the conversation as NotSendable, or the open
                failure was classified as a rejection of the request.
            UnrecognizedError: the open failure was classified as unrecognized.
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
                self._append_abandoned_call()
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
        try:
            await self._close_adapter_stream()
        finally:
            # The append runs after the close whatever the close does, so the record is written
            # even when a BaseException comes out of it, which _close_adapter_stream does not catch.
            if isinstance(exc, asyncio.CancelledError):
                self._append_abandoned_call()

    def _append_abandoned_call(self) -> None:
        """Record the abandonment, unless no log was given or the conclusion accounted for the call.

        A Response and a GenerationError each hand the caller this call's CallRecord, naming
        the model and the attempts to reconcile against, and the TransientError an Unparsed 200
        raises carries that 200's billing.
        Appending after one would report the same call twice and mislabel a concluded call as an
        in-flight abandonment.
        A StreamProtocolError and a failure after the first item hand over neither, so a
        cancellation following one still gets a record: it is the only account of the stream that
        was opened.
        Every settled record here is a failed attempt, whether an open that failed before the first
        item or a failure after one, so usage_settled holds only what those failures carried and
        what the stream billed for the items it already delivered is unobservable client-side.
        """
        if self._conclusion_carried_the_call:
            return
        _append_abandoned_call(self._abandoned_call_log, self._ledger.freeze())

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

    def _record_transient_error(self, wrapped: TransientError) -> float:
        """Record one transient failure as an attempt and register it with the RateLimiter.

        Call while the failing attempt's admission is still held,
        so a rate-limit pause is in place before the release admits anyone else.
        Every transient failure goes through here, whether or not the stream had already yielded
        items: the pause a rate limit sets protects the whole account, so losing it because this one
        stream is past reopening would leave every other caller sending into the limit.

        Returns:
            The backoff delay to sleep before the next open attempt, in seconds;
            register_transient_error draws it once so it equals any account-wide pause it set.
            A caller that will not reopen ignores it.
        """
        self._ledger.record(error=wrapped, usage=wrapped.usage, usage_raw=wrapped.usage_raw)
        return self._rate_limiter.register_transient_error(
            _extract_transient_errors(self._ledger.attempt_records)
        )

    async def _backoff_or_exhaust(self, exc: Exception, delay_seconds: float) -> None:
        """Back off before the next open attempt; call after the failed attempt's release.

        delay_seconds is the value _record_transient_error returned for this failure,
        so the sleep matches the account-wide pause the same draw set.

        Raises:
            RetriesExhaustedError: the recorded failure spent the last attempt.
        """
        if self._ledger.attempts >= self._rate_limiter.max_attempts:
            raise RetriesExhaustedError(call=self._ledger.freeze()) from exc
        await asyncio.sleep(delay_seconds)

    def _non_retriable_or_none(self, exc: Exception) -> GenerationError | None:
        """Map one attempt error to the non-retriable error to propagate, or None when transient.

        Reached only for exceptions, which by the adapter contract are attempts the adapter read no
        outcome from: what it did read it reports as an AttemptOutcome arm, which this handle
        matches instead.
        """
        if isinstance(exc, TransientError):
            return None
        classification = self._adapter.classify(exc)
        if classification == "invalid_request":
            # Adapter.classify returns invalid_request only for a request the provider rejected,
            # so it went out, and a rejection reports no usage, so ZERO_USAGE is what it billed.
            self._ledger.record(error=None, usage=ZERO_USAGE, usage_raw=None)
            return self._invalid_request_error(f"the provider rejected the request: {exc}", exc)
        if classification == "unrecognized":
            unrecognized = UnrecognizedError(error=exc, call=self._ledger.freeze())
            unrecognized.__cause__ = exc
            return unrecognized
        return None

    def _invalid_request_error(self, reason: str, cause: Exception | None) -> InvalidRequestError:
        """Build the row-shaped InvalidRequestError for this handle, chained to cause when there is one.

        cause is None for a NotSendable outcome: the adapter reported that the conversation cannot be
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
            InvalidRequestError: the adapter reported the conversation as NotSendable, or the open
                failure was classified as a rejection of the request.
            UnrecognizedError: the open failure was classified as unrecognized.
            RetriesExhaustedError: the attempts spent the retry budget.
        """
        while self._adapter_stream is None:
            self._admission = await self._rate_limiter.acquire()
            self._ledger.start_attempt()
            try:
                opened = await self._bound_adapter.open_stream(self._conversation)
            except Exception as exc:
                non_retriable = self._non_retriable_or_none(exc)
                if non_retriable is not None:
                    self._release_slot()
                    raise non_retriable from exc
                delay_seconds = self._record_transient_error(self._transient_error(exc, str(exc)))
                self._release_slot()
                await self._backoff_or_exhaust(exc, delay_seconds)
                continue
            except BaseException:
                # CancelledError is a BaseException the clause above does not catch. Releasing here
                # returns the admission at the same point on every failing path, so no caller's
                # unwind is what the shared budget depends on.
                self._release_slot()
                raise
            if isinstance(opened, NotSendable):
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
            TransientError: the stream failed after items were yielded; it carries the adapter's
                verdict on that failure, so a rate limit reaches the caller as one.
            InvalidRequestError: the adapter reported a reopened conversation as NotSendable, or
                classified an item or reopen error as a rejection of the request.
            UnrecognizedError: the adapter classified an item or reopen error as unrecognized.
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
                non_retriable = self._non_retriable_or_none(exc)
                if non_retriable is not None:
                    await self._close_adapter_stream()
                    raise non_retriable from exc
                if self._yielded_any:
                    wrapped = self._transient_error(
                        exc, f"stream failed after items were yielded: {exc}"
                    )
                    self._record_transient_error(wrapped)
                    await self._close_adapter_stream()
                    if wrapped is exc:
                        # The adapter raised the TransientError itself; `from exc` would make it its
                        # own cause.
                        raise
                    raise wrapped from exc
                delay_seconds = self._record_transient_error(self._transient_error(exc, str(exc)))
                await self._close_adapter_stream()
                await self._backoff_or_exhaust(exc, delay_seconds)
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
        A structured refusal or truncation is detected only here, when the SDK parses the assembled
        message: the adapter reports it as a Refused or Truncated outcome and this method builds the
        GenerationError from it, without retrying (the stream already yielded items to the caller);
        it reaches the caller carrying the attempt records this handle built.

        Raises:
            StreamProtocolError: the provider's event stream ended without a terminal event.
            InvalidRequestError: draining the stream hit an item or reopen error the adapter
                classified as a rejection of the request, or a reopened conversation the adapter
                reported as NotSendable.
            UnrecognizedError: draining the stream hit an item or reopen error the adapter classified as unrecognized.
            RetriesExhaustedError: draining the stream spent the retry budget on a pre-first-item failure.
            RefusalError: the adapter reported the assembled response as Refused,
                carrying this handle's attempt records.
            MaxCompletionTokensExceededError: the adapter reported it as Truncated; likewise.
            TransientError: the adapter reported it as Unparsed; not retried, because the stream
                already yielded items to the caller. The error carries that 200's billing, the only
                channel this outcome has.
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
                outcome = await self._adapter_stream.final()
                self._conclusion = self._conclude(outcome, ended_at_monotonic_seconds)
            except BaseException as exc:
                # A cancellation is not a conclusion, the same rule __anext__ applies:
                # it destroys the frames that could have observed the call rather than ending it.
                # _conclude is inside the try because three of its arms record the attempt first,
                # so a raise past that record would let a second call record it again.
                if isinstance(exc, Exception):
                    self._conclusion = exc
                raise
            # Every _conclude arm accounts for the call: three build their result off the frozen
            # CallRecord, and Unparsed puts the 200's billing on the TransientError it returns.
            self._conclusion_carried_the_call = True
        if isinstance(self._conclusion, Response):
            return self._conclusion
        raise self._conclusion

    def _conclude(
        self, outcome: ResponseOutcome[OutputT], ended_at_monotonic_seconds: float
    ) -> Response[OutputT] | GenerationError | TransientError:
        """Build what this outcome concludes the call with: the Response, or the error to raise.

        Returns the error rather than raising it, so no arm can conclude the call without being stored.
        """
        match outcome:
            case AdapterResult():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return Response(
                    output=outcome.output,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    raw=outcome.raw,
                    stop_reason=outcome.stop_reason,
                    assistant_message=outcome.assistant_message,
                )
            case Refused():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return RefusalError(call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds))
            case Truncated():
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return MaxCompletionTokensExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds)
                )
            case Unparsed():
                return TransientError(
                    "response carried no usable output",
                    usage=outcome.usage,
                    usage_raw=outcome.usage_raw,
                )
            case _ as unhandled:
                assert_never(unhandled)

    def _record_completed_attempt(
        self,
        outcome: AdapterResult[OutputT] | Refused | Truncated,
        ended_at_monotonic_seconds: float,
    ) -> None:
        """Record the attempt that reached a billable 200, whatever the adapter made of it.

        error is None on all three: the request itself succeeded, and what the adapter made of the
        response is the item's outcome, not this attempt's failure.
        An Unparsed's billing reaches the caller on its TransientError, so it gets no record.
        """
        self._ledger.record_ending_at(
            ended_at_monotonic_seconds,
            error=None,
            usage=outcome.usage,
            usage_raw=outcome.usage_raw,
        )
