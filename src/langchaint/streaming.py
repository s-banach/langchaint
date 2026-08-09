"""The stream handle.

A StreamHandle is three things at once:
an async iterator of StreamItem values,
the source of the assembled success variant via final(),
and an async context manager whose entry opens the request and whose exit closes it.
A handle is unusable outside its `async with` block, so neither iterating nor final() can start a request.
Assembling the turn and reading what it produced live behind AdapterStream.final();
the handle owns retry, pacing, and accounting.
Transient failures before open_stream() returns are retried.
Failures from an open stream are never retried.
Each open attempt is its own SharedBackoff admitted() block, and a successful open leaves the
block held, so the admission spans the stream from opening until it closes or exhausts and a
long-lived stream holds its permit for its whole life.
An iterator failure exits the block with its verdict.
A transient iterator failure ends the call as RetryUnavailableError.
An assembled response can also end the call that way.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Literal, Never, overload

from pydantic import BaseModel

from langchaint.account_state import AccountState
from langchaint.adapter import (
    Adapter,
    AdapterStream,
    BoundAdapter,
    ErrorClassification,
    InvalidRequest,
    RequestParams,
    ResponseOutcome,
    StreamItem,
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
)
from langchaint.messages import Message
from langchaint.pricing import Billing
from langchaint.response import (
    CallResult,
    GenerateResult,
    Response,
    ToolCallTurn,
    _abandoned_call_error,
    _success_variant,
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

    The request the stream belonged to has already ended, so an exception out of the close would
    only displace the result or the error the caller came for; failure_log_message names what
    stands despite the failure. A BaseException propagates.
    """
    try:
        await adapter_stream.close()
    except Exception:
        _logger.warning(failure_log_message, exc_info=True)


class StreamHandle[OutputT, ToolTurnT: ToolCallTurn[object] = Never]:
    """One stream: an item iterator, a final-result source, a context manager.

    Iterate for items as they arrive; await final() at any point in the block to drain silently and get the result.
    The request opens on entry, so open failures surface there rather than at the first item.

    ToolTurnT is the type final() can return besides Response[OutputT]:
    ToolCallTurn[OutputT] on a structured tool-bound binding, and the default Never everywhere else,
    which stream_one's overloads state so only that binding's callers see the variant.

    max_attempts bounds requests sent before open_stream() returns.
    Failures from an open stream are never retried.
    Transient iterator failures end the call as RetryUnavailableError.
    Call stream_one again to retry after that error.
    """

    def __init__(  # noqa: PLR0913 (the handle carries request, retry, lifecycle, and deadline state)
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT],
        messages: Sequence[Message],
        shared_backoff: SharedBackoff,
        max_attempts: int,
        account_state: AccountState | None,
        timeout_seconds: float | None,
        splits_tool_call_turns: bool,
    ) -> None:
        """Store the request; called by BoundLLM.stream_one only.

        splits_tool_call_turns is whether the binding is structured and tool-bound, which is what
        _success_variant needs to conclude a tool-call turn as ToolCallTurn.
        """
        self._adapter = adapter
        self._bound_adapter = bound_adapter
        self._messages = messages
        self._shared_backoff = shared_backoff
        self._max_attempts = max_attempts
        self._account_state = account_state
        self._private_backoff = PrivateBackoff(shared_backoff)
        self._timeout_seconds = timeout_seconds
        self._splits_tool_call_turns = splits_tool_call_turns
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
        None where a success variant or a GenerationError already gave the caller an account of the call,
        and None where timeout_seconds expired, which raises TimedOutError instead: both account for
        the same call, so reporting both would double what the archive says the call spent.
        """
        self._adapter_stream: AdapterStream | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._ledger = _CallLedger(model=adapter.model, provider_name=adapter.provider_name)
        self._admission: Admission | None = None
        self._ended_at_monotonic_seconds: float | None = None
        self._conclusion: GenerateResult[OutputT] | Exception | None = None
        """What concluded the call: the success variant, or the error that ended it; None until it ends."""
        self._conclusion_carried_the_call = False
        """Whether that conclusion gave the caller an account of this call; see _set_abandoned."""
        self._state: _State = "unopened"
        self._request: RequestParams | None = None
        """What every attempt of this call sends: None before entry and where none was built.

        The value a GenerationError this handle raises carries, so its None cases are that field's.
        """

    def _request_for_this_call(self) -> RequestParams:
        """Return the request every attempt of this call sends, building it on the first ask.

        Every open attempt gets the same request.

        Raises:
            InvalidRequestError: build_request returned InvalidRequest, so nothing
                can go out for this call.
        """
        if self._request is None:
            built = self._bound_adapter.build_request(self._messages)
            if isinstance(built, InvalidRequest):
                raise self._invalid_request_error(built.reason, None)
            self._request = built
        return self._request

    async def __aenter__(self) -> "StreamHandle[OutputT, ToolTurnT]":
        """Open the request and return self.

        Raises:
            InvalidRequestError: build_request returned InvalidRequest, or the open
                failure was classified as a rejection of the request.
            RuntimeError: This stream's account is closed or this handle was already entered.
            ProviderDeclaredFinalError: the provider declared the open failure final.
            UnknownExceptionError: the adapter could not place the open failure.
            RetriesExhaustedError: the opens spent the retry budget.
            TimedOutError: timeout_seconds expired before the request opened.
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
            # __aexit__ does not run when __aenter__ raises, so finish, exit the admission, and
            # close the deadline here. Leaving the deadline open would leave a live timer that
            # cancels this task at an arbitrary later point, outside any block this handle governs.
            # _open_stream_with_retries exits the admission on every path that raises, so this exit
            # covers only the case where it never entered one; it is idempotent either way.
            # The abandonment is recorded here because no other frame sees a cancellation that lands
            # during the open.
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

        A success variant and a GenerationError each hand the caller this call's CallRecord, naming
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

    async def _exit_admission(self, exc: BaseException | None) -> Verdict | None:
        """Exit the held admitted() block and return the verdict it recorded, None without one.

        exc is what the block ends with: a failure_types entry is parsed and its verdict
        recorded, and anything else, None included, only returns the permit.
        Idempotent: a later call finds no admission and returns None, which is what lets every
        closing path exit without checking what the others did.

        Raises:
            ParserContractError: the adapter's parse violated its contract on the failure exc carried.
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
        """Close the provider connection and exit the admitted() block, whatever the close does.

        The admission exit sits in a finally rather than after the close, so a BaseException out
        of the close returns the permit too.
        A failing path that already exited the block with its failure leaves this exit nothing to
        do, _exit_admission being idempotent.
        The stream is dropped before the close is awaited, so a teardown that fails is attempted
        once: __aexit__, which closes again after the paths that close mid-stream, finds nothing to
        close.
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

        An exception that already is a TransientError is its own wrapper, so an adapter that stated
        retry_after_seconds and is_rate_limit itself keeps them; message is then unused, because
        replacing the adapter's own text would lose what it said.
        The wrapper takes the verdict's capped retry_after and calls a PauseAll a rate limit;
        a failure outside failure_types has no verdict and carries neither.
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
        self, wrapped: TransientError, billing: Billing | None = None
    ) -> None:
        """Record one transient failure as an attempt.

        The failure's verdict is already recorded: the admitted() block's exit did that, so the
        pause a rate limit sets protects the whole domain even for a failure this stream cannot
        retry, whose loss would otherwise leave every other caller sending into the limit.

        billing is what the provider reported for the attempt.
        """
        self._ledger.record(error=wrapped, assistant_message=None, billing=billing)

    async def _backoff_or_exhaust(self, exc: Exception, verdict: Verdict | None) -> None:
        """Wait before the next open attempt, as the failure's verdict asks.

        On PauseAll there is no wait of our own, because the next admitted() entry already holds
        until the shared pause ends. Every other failure waits out the PrivateBackoff, floored by
        a RetryThisOne's stated retry_after.

        Raises:
            RetriesExhaustedError: the recorded failure spent the last attempt.
        """
        if self._ledger.attempts >= self._max_attempts:
            raise RetriesExhaustedError(call=self._ledger.freeze(), request=self._request) from exc
        match verdict:
            case PauseAll():
                return
            case RetryThisOne(retry_after=retry_after):
                await asyncio.sleep(self._private_backoff.next_wait(retry_after))
            case _:
                await asyncio.sleep(self._private_backoff.next_wait(None))

    def _terminal_error_or_none(
        self, exc: Exception, *, verdict: Verdict | None, stream_billing: Billing | None
    ) -> GenerationError | None:
        """Map one attempt failure to its terminal error.

        Return None when the adapter classifies the failure as transient.

        Reached only for exceptions, which by the adapter contract are attempts the adapter read no
        outcome from: what it did read it reports as a ResponseOutcome variant, which this handle
        matches instead.
        verdict is what the failure's admitted() block recorded: a failure_types exc is terminal
        on a terminal verdict, and an exc outside failure_types has no verdict and is terminal unless
        classify calls it transient (a transport failure that produced nothing parseable).
        A terminal failure's name is mapped exactly as BoundLLM._terminal_error maps it; anything
        classify cannot place lands on UnknownExceptionError.

        stream_billing is what the open stream has reported.
        It is None when no stream is open.
        It is also None when the open stream reported nothing.

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
        # A PauseAllDoNotRetry is declared_final without consulting classify: only a provider
        # directive that this request will not succeed produces one, which is what
        # ProviderDeclaredFinalError names. Reading the status instead would call a throttled
        # account's 429 a rejection of the request.
        classification: ErrorClassification = (
            "declared_final"
            if isinstance(verdict, PauseAllDoNotRetry)
            else self._adapter.classify(exc)
        )
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
        if self._adapter_stream is not None:
            # The stream was open, so langchaint can say the attempt reached the provider and
            # what that provider reported for it; the class records nothing where it cannot.
            # The open test is the stream itself, not its billing, which is None on an open
            # stream that has reported nothing yet.
            self._ledger.record(error=None, assistant_message=None, billing=stream_billing)
        return UnknownExceptionError(error=exc, call=self._ledger.freeze(), request=self._request)

    def _invalid_request_error(self, reason: str, cause: Exception | None) -> InvalidRequestError:
        """Build this handle's InvalidRequestError, chained to cause when there is one.

        cause is None where build_request returned InvalidRequest: nothing went out
        and no exception was involved.
        """
        invalid_request = InvalidRequestError(
            reason=reason, call=self._ledger.freeze(), request=self._request
        )
        invalid_request.__cause__ = cause
        return invalid_request

    def __aiter__(self) -> "StreamHandle[OutputT, ToolTurnT]":
        """Return self; the handle is its own iterator."""
        return self

    async def _open_stream_with_retries(self) -> None:
        """Take the call's request, then open one adapter stream, retrying transient failures.

        Each attempt is its own admitted() block, exited with the failure before any wait,
        so a waiting task never holds a permit while this one backs off and the failure's
        verdict reaches the shared domain either way.
        A successful open leaves the block held for the stream's whole life,
        so the stream holds its permit until it closes or exhausts.
        Every failing path out of an attempt exits the block, cancellation included.

        Raises:
            InvalidRequestError: build_request returned InvalidRequest, or the open
                failure was classified as a rejection of the request.
            RuntimeError: This stream's account is closed.
            ProviderDeclaredFinalError: the provider declared the open failure final.
            UnknownExceptionError: the adapter could not place the open failure.
            RetriesExhaustedError: the attempts spent the retry budget.
            ParserContractError: the adapter's parse violated its contract on a failed open.
        """
        if self._account_state is not None:
            self._account_state.ensure_open()
        request = self._request_for_this_call()
        while self._adapter_stream is None:
            self._admission = await self._shared_backoff.admitted().__aenter__()
            if self._account_state is not None:
                try:
                    self._account_state.ensure_open()
                except RuntimeError:
                    _ = await self._exit_admission(None)
                    raise
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
                # CancelledError is a BaseException the clause above does not catch. Exiting here
                # returns the permit at the same point on every failing path, so no caller's
                # unwind is what the shared budget depends on.
                _ = await self._exit_admission(None)
                raise
            self._adapter_stream = opened
            self._items = self._adapter_stream.items()

    async def __anext__(self) -> StreamItem:
        """Return the next item.

        Every error except StopAsyncIteration finishes the handle.
        Every Exception becomes the conclusion final() replays.

        Raises:
            RetryUnavailableError: an open stream failed transiently.
            InvalidRequestError: the adapter classified an item error as an invalid request.
            ProviderDeclaredFinalError: the provider declared an item error final.
            UnknownExceptionError: the adapter could not classify an item exception.
            StreamProtocolError: the provider's event stream ended without a terminal event; propagates unchanged.
            ParserContractError: the adapter's parse violated its contract on an item failure.
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
            raise RetryUnavailableError(
                call=self._ledger.freeze(), request=self._request
            ) from wrapped
        except BaseException:
            _ = await self._exit_admission(None)
            raise
        self._ledger.stamp_first_item()
        return item

    @overload
    async def final(self: "StreamHandle[OutputT, Never]") -> Response[OutputT]: ...
    @overload
    async def final(self) -> "Response[OutputT] | ToolTurnT": ...
    async def final(self) -> Response[OutputT] | ToolCallTurn[object]:
        """Drain any remaining items silently and return the success variant.

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
            InvalidRequestError: the adapter classified an item error as an invalid request.
            ProviderDeclaredFinalError: the provider declared an item error final.
            UnknownExceptionError: the adapter could not classify an item exception.
            RefusalError: the adapter reported the assembled response as Refusal,
                carrying this handle's attempt records.
            MaxCompletionTokensExceededError: the adapter reported it as MaxCompletionTokensExceeded; likewise.
            EmptyTurnError: the adapter reported it as EmptyTurn; likewise.
            SchemaViolationError: the adapter reported it as SchemaViolation; likewise.
            ContextWindowExceededError: the adapter reported it as ContextWindowExceeded; likewise.
            UnfinishedTurnError: the adapter reported it as UnfinishedTurn; likewise.
            ProviderFailedTerminallyError: the adapter reported it as ProviderFailedTerminally; likewise.
            RetryUnavailableError: iteration failed transiently after open_stream() returned.
                An assembled response can also report ProviderFailedTransiently.
            ParserContractError: draining hit a failure whose parse violated its contract.
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
                # Staged before interpret reads the response, so what the attempt billed and what
                # the response says about itself are on the ledger from the moment they are known.
                self._ledger.stage_response(
                    raw=raw,
                    billing=self._bound_adapter.billing_from_raw(raw),
                    identity=self._bound_adapter.identity_from_raw(raw).with_request_id_fallback(
                        adapter_stream.request_id()
                    ),
                )
                self._conclusion = self._conclude(
                    self._bound_adapter.interpret(raw),
                    raw=raw,
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                )
            except BaseException as exc:
                # A cancellation is not a conclusion, the same rule __anext__ applies:
                # it destroys the frames that could have observed the call rather than ending it.
                # The interpretation is inside the try because _conclude records the attempt before
                # it returns, so a raise past that record would let a second call record it again.
                if isinstance(exc, Exception):
                    self._conclusion = exc
                    await self._close_deadline(exc)
                raise
            # _conclude builds every result off the frozen CallRecord, so the caller has an
            # account of the call whichever one it was.
            self._conclusion_carried_the_call = True
            await self._close_deadline(None)
        if isinstance(self._conclusion, (Response, ToolCallTurn)):
            return self._conclusion
        raise self._conclusion

    def _conclude(
        self,
        outcome: ResponseOutcome[OutputT],
        *,
        raw: BaseModel,
        ended_at_monotonic_seconds: float,
    ) -> CallResult[OutputT]:
        """Build what this outcome concludes the call with: the success variant, or the error to raise.

        Returns the error rather than raising it, so no case can conclude the call without being stored.
        Every outcome closes the staged response into this attempt's record before the result is
        built, so the call it freezes into the result holds the terminal response and what it billed.
        raw is that response, which only a success carries onward.
        """
        if outcome.kind == "provider_failed_transiently":
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
        # error is None on every variant reaching here: the request succeeded, and what the adapter
        # made of the response is the item's outcome, not this attempt's failure.
        self._ledger.record_ending_at(
            ended_at_monotonic_seconds,
            error=None,
            assistant_message=outcome.assistant_message,
        )
        match outcome.kind:
            case "adapter_result":
                return _success_variant(
                    splits_tool_call_turns=self._splits_tool_call_turns,
                    output=outcome.output,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    raw=raw,
                    stop_reason=outcome.stop_reason,
                    assistant_message=outcome.assistant_message,
                )
            case "refusal":
                return RefusalError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "max_completion_tokens_exceeded":
                return MaxCompletionTokensExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "empty_turn":
                return EmptyTurnError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "schema_violation":
                return SchemaViolationError(
                    validation_error_json=outcome.validation_error_json,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "context_window_exceeded":
                return ContextWindowExceededError(
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "unfinished_turn":
                return UnfinishedTurnError(
                    reason=outcome.reason,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
            case "provider_failed_terminally":
                return ProviderFailedTerminallyError(
                    reason=outcome.reason,
                    call=self._ledger.freeze_ending_at(ended_at_monotonic_seconds),
                    request=self._request,
                )
