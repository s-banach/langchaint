"""The stream handle.

A StreamHandle is three things at once:
an async iterator of stream items (answer text chunks, reasoning text deltas, and completed tool calls),
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
from typing import Literal

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
    Refusal,
    RequestParams,
    ResponseOutcome,
    SchemaViolation,
    StreamItem,
    UnfinishedTurn,
)
from langchaint.call import _CallLedger
from langchaint.exceptions import (
    AbandonedCallError,
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
    TimedOutError,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
    _extract_transient_errors,
)
from langchaint.messages import Message
from langchaint.pricing import Billing
from langchaint.rate_limiter import Admission, Backoff, RateLimiter
from langchaint.response import CallResult, Response, _abandoned_call_error

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
        timeout_seconds: float | None,
    ) -> None:
        """Store the request; called by BoundLLM.stream_one only."""
        self._adapter = adapter
        self._bound_adapter = bound_adapter
        self._conversation = conversation
        self._rate_limiter = rate_limiter
        self._timeout_seconds = timeout_seconds
        self._deadline: asyncio.Timeout | None = None
        """The deadline scope while the call is in progress, None once it is not.

        Opened in __aenter__, so it bounds everything the block does until the call concludes: the
        open, the item pulls, and the caller's own work between them. Closed the moment a conclusion
        is stored, so a caller's own work after the call has a result is not charged to the call and
        cannot raise a second account of it; __aexit__ closes whatever is left. It is built at entry
        rather than here because asyncio.timeout fixes its instant when it is built, and stream_one
        hands the handle over before the block starts.
        """
        self.abandoned: AbandonedCallError | None = None
        """The account of this call where a cancellation cut it off, None otherwise.

        Set on the way out of the block, before the BaseException that cut the call off resumes
        propagating, so the caller reads it in its own except or finally. stream_one hands the
        handle back before anything suspends, so the caller holds it whenever a request is in
        flight, which is why this needs no log.
        None where a Response or a GenerationError already gave the caller an account of the call,
        and None where timeout_seconds expired, which raises TimedOutError instead: both account for
        the same call, so reporting both would double what the archive says the call spent.
        """
        self._adapter_stream: AdapterStream | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._ledger = _CallLedger(model=adapter.model, provider_name=adapter.provider_name)
        self._admission: Admission | None = None
        self._yielded_any = False
        self._ended_at_monotonic_seconds: float | None = None
        self._conclusion: Response[OutputT] | Exception | None = None
        """What concluded the call: the Response, or the error that ended it; None until it ends."""
        self._conclusion_carried_the_call = False
        """Whether that conclusion gave the caller an account of this call; see _set_abandoned."""
        self._state: _State = "unopened"
        self._request: RequestParams | None = None
        """What every attempt of this call sends: None before entry and where none was built.

        The value a GenerationError this handle raises carries, so its None cases are that field's.
        """

    def _request_for_this_call(self) -> RequestParams:
        """Return the request every attempt of this call sends, building it on the first ask.

        A reopen after a pre-first-item failure asks again and gets the same request, so one call
        puts one request on the wire however many streams it opens.

        Raises:
            InvalidRequestError: build_request reported the conversation InvalidRequest, so nothing
                can go out for this call.
        """
        if self._request is None:
            built = self._bound_adapter.build_request(self._conversation)
            if isinstance(built, InvalidRequest):
                raise self._invalid_request_error(built.reason, None)
            self._request = built
        return self._request

    async def __aenter__(self) -> "StreamHandle[OutputT]":
        """Open the request and return self.

        Raises:
            InvalidRequestError: the adapter reported the conversation InvalidRequest, or the open
                failure was classified as a rejection of the request.
            ProviderDeclaredFinalError: the provider declared the open failure final.
            UnknownExceptionError: the adapter could not place the open failure.
            RetriesExhaustedError: the opens spent the retry budget.
            TimedOutError: timeout_seconds expired before the request opened.
            RuntimeError: this handle was already entered; build a new one with stream_one.
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
            # __aexit__ does not run when __aenter__ raises, so finish, release, and close the
            # deadline here. Leaving the deadline open would leave a live timer that cancels this
            # task at an arbitrary later point, outside any block this handle governs.
            # _open_stream_with_retries returns the slot on every path that raises, so this release
            # covers only the case where it never acquired one; it is idempotent either way.
            # The abandonment is recorded here because no other frame sees a cancellation that lands
            # during the open.
            self._state = "finished"
            self._release_slot()
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
        """Close the underlying connection and finish the handle.

        A CancelledError exiting the block sets abandoned, unless a conclusion already gave the
        caller an account of the call (_set_abandoned states which conclusions do).
        A consumer that leaves the block early without an exception chose to walk away in live
        code, so only the cancellation, which destroys the frames that could have observed the
        stream, gets a record.
        An expired timeout_seconds raises TimedOutError in place of that cancellation, and leaves
        abandoned unset, so one cut-off call produces one account.

        The deadline closes before the connection does. A timer still armed while the close is
        awaited could cancel the close itself, and _close_adapter_stream catches only Exception, so
        the connection would be left open.

        Raises:
            TimedOutError: timeout_seconds expired before the block finished.
            BaseException: the adapter stream's close raised something that is not an Exception;
                it propagates in place of whatever was unwinding the block, an expired deadline
                included, because a call whose account nobody will read is not worth an account.
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
            # The set runs after the close whatever the close does, so the account is there
            # even when a BaseException comes out of it, which _close_adapter_stream does not catch.
            if isinstance(exc, asyncio.CancelledError) and not timed_out:
                self._set_abandoned(billing_in_flight)
        if timed_out:
            raise self._timed_out_error(billing_in_flight) from None

    async def _close_deadline(self, exc: BaseException | None) -> bool:
        """Close the deadline scope, reporting whether it was this deadline that expired.

        Disarms the timer, so nothing this handle opened can cancel the caller's task once the call
        is over. Idempotent: a second call finds no scope and reports False, which is what lets the
        conclusion sites, __aenter__'s failure path, and __aexit__ each close it without checking
        what the others did.

        True means the cancellation now unwinding is this deadline's own, which asyncio.timeout
        reports by raising TimeoutError from its __aexit__. An outer cancellation leaves the scope
        with nothing to absorb, so it reports False and propagates untouched.
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

    def _timed_out_error(self, billing_in_flight: Billing | None) -> TimedOutError:
        """Freeze the ledger into this call's deadline account."""
        return _abandoned_call_error(TimedOutError, self._ledger, billing_in_flight)

    def _billing_reported(self) -> Billing | None:
        """Ask the open stream what the provider has reported, or None where it reported nothing.

        Call before the connection closes, because closing drops the stream this asks.
        """
        if self._adapter_stream is None:
            return None
        return self._adapter_stream.billing_reported()

    def _set_abandoned(self, billing_in_flight: Billing | None) -> None:
        """Set abandoned, unless the conclusion already accounted for the call.

        A Response and a GenerationError each hand the caller this call's CallRecord, naming
        the model and the attempts to reconcile against.
        Setting it after one would report the same call twice and mislabel a concluded call as an
        in-flight abandonment.
        A StreamProtocolError hands over neither, so a cancellation following one still gets an
        account: it is the only one of the stream that was opened.
        billing_in_flight is what the adapter could state of the attempt still open when the
        cancellation arrived.
        """
        if self._conclusion_carried_the_call:
            return
        self.abandoned = _abandoned_call_error(AbandonedCallError, self._ledger, billing_in_flight)

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
        self, wrapped: TransientError, billing: Billing | None = None
    ) -> Backoff:
        """Record one transient failure as an attempt and register it with the RateLimiter.

        Called while the failing attempt's admission is still held;
        register_transient_error raises RuntimeError for one already released.
        A failure this stream cannot retry still registers, because the pause a rate limit sets
        protects the whole account: losing it because this one stream is past reopening would leave
        every other caller sending into the limit.

        billing is what the provider reported for the attempt.

        Returns:
            The Backoff to sleep before the next open attempt;
            its delay is drawn once, so it equals any account-wide pause it set.
            A caller that will not reopen drops it.
        """
        self._ledger.record(error=wrapped, assistant_message=None, billing=billing)
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
            raise RetriesExhaustedError(call=self._ledger.freeze(), request=self._request) from exc
        await backoff.sleep()

    def _non_retriable_or_none(
        self, exc: Exception, stream_billing: Billing | None
    ) -> GenerationError | None:
        """Map one attempt error to the non-retriable error to propagate, or None when transient.

        Reached only for exceptions, which by the adapter contract are attempts the adapter read no
        outcome from: what it did read it reports as a ResponseOutcome member, which this handle
        matches instead.

        stream_billing is what the open stream has reported, None where no stream is open and None
        again where an open one has reported nothing. A provider that reports counters before its
        first item has already billed by the time one of these errors arrives, so the reading is
        what keeps that attempt accountable.

        The request id goes to the ledger before the transient check, so an attempt this handle will
        retry carries it too. An open stream is the second source: a connection that drops mid-turn
        raises an error carrying no id, while the stream still holds the headers the 200 arrived with.
        """
        request_id = self._adapter.request_id_from_error(exc)
        if request_id is None and self._adapter_stream is not None:
            request_id = self._adapter_stream.request_id()
        self._ledger.note_request_id(request_id)
        if isinstance(exc, TransientError):
            return None
        classification = self._adapter.classify(exc)
        if classification == "invalid_request":
            # Adapter.classify returns invalid_request only for a request the provider rejected,
            # so it went out and gets a record.
            self._ledger.record(error=None, assistant_message=None, billing=stream_billing)
            return self._invalid_request_error(f"the provider rejected the request: {exc}", exc)
        if classification == "declared_final":
            # The provider answered, so the attempt gets a record.
            self._ledger.record(error=None, assistant_message=None, billing=stream_billing)
            return ProviderDeclaredFinalError(
                error=exc, call=self._ledger.freeze(), request=self._request
            )
        if classification == "unknown_exception":
            if self._adapter_stream is not None:
                # The stream was open, so langchaint can say the attempt reached the provider and
                # what that provider reported for it; the class records nothing where it cannot.
                # The open test is the stream itself, not its billing, which is None on an open
                # stream that has reported nothing yet.
                self._ledger.record(error=None, assistant_message=None, billing=stream_billing)
            return UnknownExceptionError(
                error=exc, call=self._ledger.freeze(), request=self._request
            )
        return None

    def _invalid_request_error(self, reason: str, cause: Exception | None) -> InvalidRequestError:
        """Build the row-shaped InvalidRequestError for this handle, chained to cause when there is one.

        cause is None where build_request reported the conversation InvalidRequest: nothing went out
        and no exception was involved.
        """
        invalid_request = InvalidRequestError(
            reason=reason, call=self._ledger.freeze(), request=self._request
        )
        invalid_request.__cause__ = cause
        return invalid_request

    def __aiter__(self) -> "StreamHandle[OutputT]":
        """Return self; the handle is its own iterator."""
        return self

    async def _open_stream_with_retries(self) -> None:
        """Take the call's request, then open one adapter stream, retrying transient failures.

        A fresh admission is acquired for each attempt and released before the backoff sleep,
        so a waiting task never holds capacity while this one backs off.
        A successful open registers the admission with the limiter,
        ending any recovery this handle's probe was serving,
        so a stream slow to first token cannot stall the shared account's admission.
        The slot stays held for the stream's whole life; only recovery ends here, not the in-flight hold.
        Every failing path out of an attempt returns the admission, cancellation included.

        Raises:
            InvalidRequestError: the adapter reported the conversation InvalidRequest, or the open
                failure was classified as a rejection of the request.
            ProviderDeclaredFinalError: the provider declared the open failure final.
            UnknownExceptionError: the adapter could not place the open failure.
            RetriesExhaustedError: the attempts spent the retry budget.
        """
        request = self._request_for_this_call()
        while self._adapter_stream is None:
            self._admission = await self._rate_limiter.acquire()
            self._ledger.start_attempt()
            try:
                opened = await self._bound_adapter.open_stream(request)
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
            InvalidRequestError: the adapter classified an item or reopen error as a rejection of
                the request.
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
                # observed the call, which is what abandoned records instead.
                self._conclusion = exc
                self._conclusion_carried_the_call = isinstance(exc, GenerationError)
                await self._close_deadline(exc)
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
                stream_billing = self._billing_reported()
                non_retriable = self._non_retriable_or_none(exc, stream_billing)
                if non_retriable is not None:
                    await self._close_adapter_stream()
                    raise non_retriable from exc
                if self._yielded_any:
                    wrapped = self._transient_error(
                        exc, f"stream failed after items were yielded: {exc}"
                    )
                    self._record_transient_error(wrapped, stream_billing)
                    await self._close_adapter_stream()
                    raise RetryUnavailableError(
                        call=self._ledger.freeze(), request=self._request
                    ) from wrapped
                backoff = self._record_transient_error(
                    self._transient_error(exc, str(exc)), stream_billing
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
            self._ledger.stamp_first_item()
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
                classified as a rejection of the request.
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
            adapter_stream = self._adapter_stream
            assert adapter_stream is not None
            ended_at_monotonic_seconds = (
                time.monotonic()
                if self._ended_at_monotonic_seconds is None
                else self._ended_at_monotonic_seconds
            )
            try:
                raw = await adapter_stream.final()
                identity = self._bound_adapter.identity_from_raw(raw)
                if identity.request_id is None:
                    # An assembled response need not carry the header its HTTP response did; the
                    # stream is what still has it.
                    identity = identity._replace(request_id=adapter_stream.request_id())
                # Staged before interpret reads the response, so what the attempt billed and what
                # the response says about itself are on the ledger from the moment they are known.
                self._ledger.stage_response(
                    raw=raw,
                    billing=self._bound_adapter.billing_from_raw(raw),
                    identity=identity,
                )
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
                    await self._close_deadline(exc)
                raise
            # Every _conclude case builds its result off the frozen CallRecord, so the caller has an
            # account of the call whichever one it was.
            self._conclusion_carried_the_call = True
            await self._close_deadline(None)
        if isinstance(self._conclusion, Response):
            return self._conclusion
        raise self._conclusion

    def _conclude(
        self,
        outcome: ResponseOutcome[OutputT],
        *,
        raw: BaseModel,
        ended_at_monotonic_seconds: float,
    ) -> CallResult[OutputT]:
        """Build what this outcome concludes the call with: the Response, or the error to raise.

        Returns the error rather than raising it, so no case can conclude the call without being stored.
        Every case closes the staged response into this attempt's record first, so the call it freezes
        into the result holds the terminal response and what it billed.
        raw is that response, which only a success carries onward.
        """
        match outcome.kind:
            case "adapter_result":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return Response(
                    output=outcome.output,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    raw=raw,
                    stop_reason=outcome.stop_reason,
                    assistant_message=outcome.assistant_message,
                )
            case "refusal":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return RefusalError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "max_completion_tokens_exceeded":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return MaxCompletionTokensExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "empty_turn":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return EmptyTurnError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "schema_violation":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return SchemaViolationError(
                    validation_error_json=outcome.validation_error_json,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "context_window_exceeded":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return ContextWindowExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "unfinished_turn":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return UnfinishedTurnError(
                    reason=outcome.reason,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "provider_failed_terminally":
                self._record_completed_attempt(outcome, ended_at_monotonic_seconds)
                return ProviderFailedTerminallyError(
                    reason=outcome.reason,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "provider_failed_transiently":
                failure = TransientError(outcome.reason, is_rate_limit=outcome.is_rate_limit)
                self._ledger.record_ending_at(
                    ended_at_monotonic_seconds,
                    error=failure,
                    assistant_message=outcome.assistant_message,
                )
                retry_unavailable = RetryUnavailableError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
                retry_unavailable.__cause__ = failure
                return retry_unavailable

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
