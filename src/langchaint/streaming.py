"""The stream handle.

A StreamHandle is three things at once: an async iterator of stream items (text chunks and completed tool calls),
the source of the assembled Response via final(),
and an async context manager so an abandoned stream closes its connection deterministically.
Assembly and structured-output parsing live in the SDK behind ProviderStream.final();
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

from langchaint.exceptions import (
    AbortBatchError,
    AttemptRecord,
    GenerationError,
    RetriesExhaustedError,
    StreamProtocolError,
    TransientError,
    _extract_transient_errors,
)
from langchaint.messages import Message
from langchaint.provider import BoundProvider, Provider, ProviderStream, StreamItem
from langchaint.rate_limiter import Admission, RateLimiter
from langchaint.response import Response


class StreamHandle[OutputT]:
    """One stream: an item iterator, a Response source, a context manager.

    Iterate for items as they arrive; await final() at any point to drain silently and get the assembled Response;
    use async with so leaving the block closes the connection.
    Nothing starts until the first item is requested;
    a RateLimiter slot is acquired then, held while the stream is open, and reacquired before each pre-first-item retry.
    A transient failure after the first yielded item propagates as TransientError; retry by calling stream_one again.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        bound_provider: BoundProvider[OutputT],
        conversation: Sequence[Message],
        rate_limiter: RateLimiter,
    ) -> None:
        """Store the request; called by BoundLLM.stream_one only."""
        self._provider = provider
        self._bound_provider = bound_provider
        self._conversation = conversation
        self._rate_limiter = rate_limiter
        self._provider_stream: ProviderStream[OutputT] | None = None
        self._items: AsyncIterator[StreamItem] | None = None
        self._attempt_records: list[AttemptRecord] = []
        self._admission: Admission | None = None
        self._yielded_any = False
        self._attempt_started_at_monotonic_seconds: float | None = None
        self._started_at_monotonic_seconds: float | None = None
        self._ended_at_monotonic_seconds: float | None = None
        self._response: Response[OutputT] | None = None

    async def __aenter__(self) -> "StreamHandle[OutputT]":
        """Return self; opening is deferred until the first item."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying connection if one is open."""
        await self._close_provider_stream()

    def _release_slot(self) -> None:
        """Return the held RateLimiter admission, if this handle holds one."""
        if self._admission is not None:
            self._rate_limiter.release(self._admission)
            self._admission = None

    async def _close_provider_stream(self) -> None:
        """Close and forget the current provider stream, releasing its slot."""
        if self._provider_stream is not None:
            await self._provider_stream.close()
            self._provider_stream = None
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
                retry_after_seconds=self._provider.retry_after_seconds(exc),
                is_rate_limit=self._provider.classify(exc) == "rate_limit",
            )
            wrapped.__cause__ = exc
        assert self._attempt_started_at_monotonic_seconds is not None
        self._attempt_records.append(
            AttemptRecord(
                started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                ended_at_monotonic_seconds=time.monotonic(),
                error=wrapped,
                usage=wrapped.usage,
                cost_in_usd=wrapped.cost_in_usd,
            )
        )
        return self._rate_limiter.register_transient_error(_extract_transient_errors(self._attempt_records))

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
                model=self._provider.model,
                provider_name=self._provider.name,
                elapsed_seconds=time.monotonic() - self._started_at_monotonic_seconds,
                stop_reason=None,
            ) from exc
        await asyncio.sleep(delay_seconds)

    def _non_retriable_or_none(self, exc: Exception) -> AbortBatchError | GenerationError | None:
        """Map one attempt error to the non-retriable error to propagate, or None when transient.

        A TransientError retries (None).
        An AbortBatchError or a GenerationError leaf raised directly by the adapter propagates as itself.
        An unrecognized exception the adapter classifies "abort" becomes an AbortBatchError; anything else is transient.
        """
        if isinstance(exc, TransientError):
            return None
        if isinstance(exc, (AbortBatchError, GenerationError)):
            return exc
        if self._provider.classify(exc) == "abort":
            abort = AbortBatchError(f"abort provider error: {exc}")
            abort.__cause__ = exc
            return abort
        return None

    def __aiter__(self) -> "StreamHandle[OutputT]":
        """Return self; the handle is its own iterator."""
        return self

    async def _ensure_stream_open(self) -> None:
        """Open the provider stream if none is held, retrying pre-first-item failures under the limiter.

        A fresh admission is acquired for each open attempt and released before the backoff sleep,
        so a waiting task never holds capacity while this one backs off.
        A non-retriable open failure (an AbortBatchError,
        or an abort classification) propagates and pre-first-item transient exhaustion raises RetriesExhaustedError,
        both through the shared helpers, as when the open ran inline in __anext__.
        A successful open registers the admission with the limiter:
        the open is a completed request that already cleared the quota,
        so it ends any recovery this handle's probe was serving,
        so a stream slow to first token cannot stall the shared account's admission.
        The slot stays held for the stream's whole life; only recovery ends here, not the in-flight hold.
        """
        while self._provider_stream is None:
            self._admission = await self._rate_limiter.acquire()
            self._attempt_started_at_monotonic_seconds = time.monotonic()
            if self._started_at_monotonic_seconds is None:
                self._started_at_monotonic_seconds = self._attempt_started_at_monotonic_seconds
            try:
                self._provider_stream = await self._bound_provider.open_stream(self._conversation)
            except Exception as exc:
                non_retriable = self._non_retriable_or_none(exc)
                if non_retriable is not None:
                    self._release_slot()
                    raise non_retriable from exc
                delay_seconds = self._record_transient_error(exc)
                self._release_slot()
                await self._backoff_or_exhaust(exc, delay_seconds)
                continue
            self._items = self._provider_stream.items()
            assert self._admission is not None
            self._rate_limiter.register_success(self._admission)

    async def __anext__(self) -> StreamItem:
        """Return the next item, opening the stream on demand.

        Attempt errors propagate as AbortBatchError when the adapter classifies them so,
        and pre-first-item transient exhaustion raises RetriesExhaustedError (both from the shared helpers).

        Raises:
            TransientError: the stream failed after items were yielded.
            AbortBatchError: the adapter classified an open or item error as abort.
            RetriesExhaustedError: a pre-first-item failure spent the retry budget while opening the stream.
            StreamProtocolError: the provider stream violated the stream contract; propagates unchanged.
            StopAsyncIteration: the stream is exhausted.
        """
        while True:
            await self._ensure_stream_open()
            assert self._items is not None
            try:
                item = await self._items.__anext__()
            except StopAsyncIteration:
                if self._ended_at_monotonic_seconds is None:
                    self._ended_at_monotonic_seconds = time.monotonic()
                self._release_slot()
                raise
            except StreamProtocolError:
                await self._close_provider_stream()
                raise
            except Exception as exc:
                non_retriable = self._non_retriable_or_none(exc)
                if non_retriable is not None:
                    await self._close_provider_stream()
                    raise non_retriable from exc
                if self._yielded_any:
                    await self._close_provider_stream()
                    raise TransientError(
                        f"stream failed after items were yielded: {exc}"
                    ) from exc
                delay_seconds = self._record_transient_error(exc)
                await self._close_provider_stream()
                await self._backoff_or_exhaust(exc, delay_seconds)
                continue
            except BaseException:
                # CancelledError is a BaseException the clauses above do not catch.
                # A cancellation while iterating outside `async with` would otherwise strand this slot,
                # and because a stranded probe leaves _probe_admission set it freezes the whole limiter's recovery,
                # not just one slot. Return the slot, then let the cancellation propagate.
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
            StreamProtocolError: the stream ended without a terminal response.
            AbortBatchError: draining the stream hit an open or item error the adapter classified as abort.
            RetriesExhaustedError: draining the stream spent the retry budget on a pre-first-item failure.
            RefusalError: the structured parse found a refusal; enriched with this handle's attempt records.
            ExceededMaxCompletionTokensError: the structured response hit the token cap; enriched likewise.
            TransientError: the structured parse produced no instance for another reason; not retried,
                because the stream already yielded items to the caller.
        """
        if self._response is not None:
            return self._response
        async for _ in self:
            pass
        if self._provider_stream is None:
            raise StreamProtocolError("stream ended without producing a result")
        assert self._attempt_started_at_monotonic_seconds is not None
        assert self._started_at_monotonic_seconds is not None
        ended_at_monotonic_seconds = time.monotonic() if self._ended_at_monotonic_seconds is None else self._ended_at_monotonic_seconds
        try:
            provider_result = await self._provider_stream.final()
        except GenerationError as exc:
            self._attempt_records.append(
                AttemptRecord(
                    started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                    error=None,
                    usage=exc.usage,
                    cost_in_usd=exc.cost_in_usd,
                )
            )
            raise type(exc)(
                attempt_records=tuple(self._attempt_records),
                model=self._provider.model,
                provider_name=self._provider.name,
                elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
                stop_reason=exc.stop_reason,
            ) from exc
        response = Response(
            output=provider_result.output,
            usage=provider_result.usage,
            cost_in_usd=provider_result.cost_in_usd,
            model=self._provider.model,
            provider_name=self._provider.name,
            attempt_records=(
                *self._attempt_records,
                AttemptRecord(
                    started_at_monotonic_seconds=self._attempt_started_at_monotonic_seconds,
                    ended_at_monotonic_seconds=ended_at_monotonic_seconds,
                    error=None,
                    usage=provider_result.usage,
                    cost_in_usd=provider_result.cost_in_usd,
                ),
            ),
            elapsed_seconds=ended_at_monotonic_seconds - self._started_at_monotonic_seconds,
            raw=provider_result.raw,
            stop_reason=provider_result.stop_reason,
            assistant_message=provider_result.assistant_message,
        )
        self._response = response
        return response
