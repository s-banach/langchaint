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
An open stream holds one RateLimiter in-flight slot from opening until the stream closes or exhausts,
so long-lived streams count against max_in_flight for their whole life.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Literal

from langchaint.adapter import Adapter, AdapterStream, BoundAdapter, StreamItem
from langchaint.exceptions import (
    AttemptRecord,
    FatalError,
    GenerationError,
    RetriesExhaustedError,
    StreamProtocolError,
    TransientError,
    UnrecognizedError,
    _extract_transient_errors,
)
from langchaint.messages import Message
from langchaint.rate_limiter import Admission, RateLimiter
from langchaint.response import AbandonedCall, AbandonedCallLog, Response

type _State = Literal["unopened", "open", "finished"]

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
        self._attempt_records: list[AttemptRecord] = []
        self._admission: Admission | None = None
        self._yielded_any = False
        self._attempt_started_at_monotonic_seconds: float | None = None
        self._started_at_monotonic_seconds: float | None = None
        self._ended_at_monotonic_seconds: float | None = None
        self._response: Response[OutputT] | None = None
        self._final_concluded = False
        self._state: _State = "unopened"

    async def __aenter__(self) -> "StreamHandle[OutputT]":
        """Open the request and return self.

        Raises:
            FatalError: the adapter classified the open failure as fatal.
            GenerationError: the adapter raised one of its leaves while opening,
                or the open failure was classified unrecognized (an UnrecognizedError).
            RetriesExhaustedError: the opens spent the retry budget.
            RuntimeError: this handle was already entered; build a new one with stream_one.
        """
        if self._state != "unopened":
            raise RuntimeError(_ALREADY_ENTERED_MESSAGE)
        self._state = "open"
        try:
            await self._open_stream_with_retries()
        except BaseException as exc:
            # __aexit__ does not run when __aenter__ raises, so finish and release here.
            # _open_stream_with_retries returns the slot on every Exception path but not on a
            # CancelledError, which is a BaseException: cancelling a suspended open would otherwise
            # strand this admission for the process's life, and a stranded probe freezes the whole
            # limiter's recovery, not just one slot. The abandonment is recorded here for the same
            # reason: no other frame sees a cancellation that lands during the open.
            self._state = "finished"
            if isinstance(exc, asyncio.CancelledError):
                self._append_abandoned_call()
            self._release_slot()
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
        abandoned_call_log, when one was given and final() never concluded the call:
        a Response and a raised GenerationError both reached the caller through final() carrying
        their own usage, and a consumer that leaves the block early without an exception chose to
        walk away in live code, so only the cancellation, which destroys the frames that could
        have observed the stream, gets a record.
        """
        self._state = "finished"
        if isinstance(exc, asyncio.CancelledError):
            self._append_abandoned_call()
        await self._close_adapter_stream()

    def _append_abandoned_call(self) -> None:
        """Record the abandonment, unless no log was given or final() already concluded the call.

        A call final() concluded is excluded on both exits: a Response and a raised GenerationError
        each carry their own usage to the caller, so appending here would double-count it and
        mislabel a concluded call as an in-flight abandonment.
        The settled attempt records are only pre-first-item open failures (nothing is retried after
        the first yielded item), so usage_settled is usually zero here and the record's value is
        the count; the streaming request itself is the unobservable in-flight attempt.
        """
        if self._abandoned_call_log is None or self._final_concluded:
            return
        self._abandoned_call_log.append(
            AbandonedCall(
                attempt_records=tuple(self._attempt_records),
                model=self._adapter.model,
                provider_name=self._adapter.provider_name,
            )
        )

    def _release_slot(self) -> None:
        if self._admission is not None:
            self._rate_limiter.release(self._admission)
            self._admission = None

    async def _close_adapter_stream(self) -> None:
        if self._adapter_stream is not None:
            await self._adapter_stream.close()
            self._adapter_stream = None
            self._items = None
        self._release_slot()

    def _record_transient_error(self, exc: Exception) -> float:
        """Record one pre-first-item transient failure and register it with the RateLimiter.

        Call while the failing attempt's admission is still held,
        so a rate-limit pause is in place before the release admits anyone else.

        Returns:
            The backoff delay to sleep before the next open attempt, in seconds;
            register_transient_error draws it once so it equals any account-wide pause it set.
        """
        if isinstance(exc, TransientError):
            wrapped = exc
        else:
            wrapped = TransientError(
                str(exc),
                retry_after_seconds=self._adapter.retry_after_seconds(exc),
                is_rate_limit=self._adapter.classify(exc) == "rate_limit",
            )
            wrapped.__cause__ = exc
        assert self._attempt_started_at_monotonic_seconds is not None
        self._attempt_records.append(
            AttemptRecord(
                started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                ended_at_monotonic_seconds=time.monotonic(),
                error=wrapped,
                usage=wrapped.usage,
                usage_raw=wrapped.usage_raw,
            )
        )
        return self._rate_limiter.register_transient_error(
            _extract_transient_errors(self._attempt_records)
        )

    async def _backoff_or_exhaust(self, exc: Exception, delay_seconds: float) -> None:
        """Back off before the next open attempt; call after the failed attempt's release.

        delay_seconds is the value _record_transient_error returned for this failure,
        so the sleep matches the account-wide pause the same draw set.

        Raises:
            RetriesExhaustedError: the recorded failure spent the last attempt.
        """
        if len(self._attempt_records) >= self._rate_limiter.max_attempts:
            assert self._started_at_monotonic_seconds is not None
            raise RetriesExhaustedError(
                attempt_records=tuple(self._attempt_records),
                model=self._adapter.model,
                provider_name=self._adapter.provider_name,
                elapsed_seconds=time.monotonic() - self._started_at_monotonic_seconds,
                stop_reason=None,
            ) from exc
        await asyncio.sleep(delay_seconds)

    def _non_retriable_or_none(self, exc: Exception) -> FatalError | GenerationError | None:
        """Map one attempt error to the non-retriable error to propagate, or None when transient."""
        if isinstance(exc, TransientError):
            return None
        if isinstance(exc, FatalError):
            exc.attempt_records = tuple(self._attempt_records)
            return exc
        if isinstance(exc, GenerationError):
            return exc
        classification = self._adapter.classify(exc)
        if classification == "fatal":
            fatal_error = FatalError(
                f"fatal provider error: {exc}",
                attempt_records=tuple(self._attempt_records),
            )
            fatal_error.__cause__ = exc
            return fatal_error
        if classification == "unrecognized":
            assert self._started_at_monotonic_seconds is not None
            unrecognized = UnrecognizedError(
                error=exc,
                attempt_records=tuple(self._attempt_records),
                model=self._adapter.model,
                provider_name=self._adapter.provider_name,
                elapsed_seconds=time.monotonic() - self._started_at_monotonic_seconds,
            )
            unrecognized.__cause__ = exc
            return unrecognized
        return None

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

        Raises:
            FatalError: the adapter classified the failure as fatal.
            GenerationError: the adapter raised one of its leaves directly,
                or the failure was classified unrecognized (an UnrecognizedError).
            RetriesExhaustedError: the attempts spent the retry budget.
        """
        while self._adapter_stream is None:
            self._admission = await self._rate_limiter.acquire()
            self._attempt_started_at_monotonic_seconds = time.monotonic()
            if self._started_at_monotonic_seconds is None:
                self._started_at_monotonic_seconds = self._attempt_started_at_monotonic_seconds
            try:
                self._adapter_stream = await self._bound_adapter.open_stream(self._conversation)
            except Exception as exc:
                non_retriable = self._non_retriable_or_none(exc)
                if non_retriable is not None:
                    self._release_slot()
                    raise non_retriable from exc
                delay_seconds = self._record_transient_error(exc)
                self._release_slot()
                await self._backoff_or_exhaust(exc, delay_seconds)
                continue
            self._items = self._adapter_stream.items()
            assert self._admission is not None
            self._rate_limiter.register_success(self._admission)

    async def __anext__(self) -> StreamItem:
        """Return the next item.

        Every error but StopAsyncIteration finishes the handle, so nothing later reopens the request.

        Raises:
            TransientError: the stream failed after items were yielded.
            FatalError: the adapter classified an item or reopen error as fatal.
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
        except BaseException:
            self._state = "finished"
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
                    await self._close_adapter_stream()
                    raise TransientError(f"stream failed after items were yielded: {exc}") from exc
                delay_seconds = self._record_transient_error(exc)
                await self._close_adapter_stream()
                await self._backoff_or_exhaust(exc, delay_seconds)
                await self._open_stream_with_retries()
                continue
            except BaseException:
                # CancelledError is a BaseException the clauses above do not catch.
                # Cancelling an item pull in its own task leaves the block open, so waiting for __aexit__
                # would strand this slot, and because a stranded probe leaves _probe_admission set it freezes
                # the whole limiter's recovery, not just one slot. Return the slot, then let it propagate.
                self._release_slot()
                raise
            self._yielded_any = True
            return item

    async def final(self) -> Response[OutputT]:
        """Drain any remaining items silently and return the Response.

        Idempotent: the assembled Response is cached on first completion.
        A structured refusal or truncation is detected only here,
        when the SDK parses the assembled message, so its GenerationError leaf surfaces from final() and is not retried
        (the stream already yielded items to the caller);
        the enriched leaf carries the attempt records this handle built.

        Raises:
            StreamProtocolError: the provider's event stream ended without a terminal event.
            FatalError: draining the stream hit an item or reopen error the adapter classified as fatal.
            UnrecognizedError: draining the stream hit an item or reopen error the adapter classified as unrecognized.
            RetriesExhaustedError: draining the stream spent the retry budget on a pre-first-item failure.
            RefusalError: the structured parse found a refusal; enriched with this handle's attempt records.
            MaxCompletionTokensExceededError: the structured response hit the token cap; enriched likewise.
            TransientError: the structured parse produced no instance for another reason; not retried,
                because the stream already yielded items to the caller.
            RuntimeError: the handle is unopened or finished.
        """
        if self._response is not None:
            return self._response
        if self._state != "open":
            raise RuntimeError(
                _UNOPENED_MESSAGE if self._state == "unopened" else _FINISHED_MESSAGE
            )
        async for _ in self:
            pass
        assert self._adapter_stream is not None
        assert self._attempt_started_at_monotonic_seconds is not None
        assert self._started_at_monotonic_seconds is not None
        ended_at_monotonic_seconds = (
            time.monotonic()
            if self._ended_at_monotonic_seconds is None
            else self._ended_at_monotonic_seconds
        )
        try:
            adapter_result = await self._adapter_stream.final()
        except GenerationError as exc:
            self._attempt_records.append(
                AttemptRecord(
                    started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                    error=None,
                    usage=exc.usage,
                    usage_raw=exc.usage_raw,
                )
            )
            self._final_concluded = True
            raise type(exc)(
                attempt_records=tuple(self._attempt_records),
                model=self._adapter.model,
                provider_name=self._adapter.provider_name,
                elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
                stop_reason=exc.stop_reason,
            ) from exc
        response = Response(
            output=adapter_result.output,
            model=self._adapter.model,
            provider_name=self._adapter.provider_name,
            attempt_records=(
                *self._attempt_records,
                AttemptRecord(
                    started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                    error=None,
                    usage=adapter_result.usage,
                    usage_raw=adapter_result.usage_raw,
                ),
            ),
            elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
            raw=adapter_result.raw,
            stop_reason=adapter_result.stop_reason,
            assistant_message=adapter_result.assistant_message,
        )
        self._response = response
        self._final_concluded = True
        return response
