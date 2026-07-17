"""OTel span tracing for langchaint, as a thin wrapper that never fakes an event boundary it traces.

TracedLLM wraps an LLM and mirrors bind / rebind so every binding stays traced;
TracedBoundLLM wraps a BoundLLM and opens one span per generate call (CLIENT for generate_one,
which wraps one outbound call, INTERNAL for generate_many's aggregate span, which makes no call of its own);
TracedStreamHandle wraps a StreamHandle and opens one CLIENT span across the stream's life.

The wrapper owns the span lifecycle (lazy start, exactly-once end, error status on every exception path)
and never fakes an event boundary a span is supposed to measure:
TracedStreamHandle iterates so it can record langchaint.time_to_first_token_seconds
and close the span on a failing or abandoned stream, rather than delegating the iteration it needs to witness.
generate_many is the exception by design: it is a bulk convenience,
so it delegates to BoundLLM.generate_many under one aggregate span
and leaves per-item detail to the returned rows (to_row).
The mapper owns only attribute names and values, the part that varies by convention;
a mapper cannot change the span name, kind, or status, and a raising mapper is caught and logged, never propagated.

Importing this subpackage requires opentelemetry-api (install langchaint[otel]);
the import below raises a ModuleNotFoundError naming the extra to install.
The wrapper imports only opentelemetry-api, so a production app installs the api and wires its own SDK.

Every GenAI semantic-convention attribute key is verified against the pinned revision, never asserted from memory:
opentelemetry-semantic-conventions 0.64b0.
The chat-completion operation value is "chat" (GenAiOperationNameValues.CHAT).
A convention change is a deliberate edit to this module.
"""

import importlib.metadata
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, overload

from pydantic import BaseModel

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
except ModuleNotFoundError as exc:
    if exc.name is not None and not exc.name.startswith("opentelemetry"):
        raise
    raise ModuleNotFoundError(
        "langchaint's tracing subpackage requires opentelemetry-api; install langchaint[otel]."
    ) from exc

from langchaint.exceptions import GenerationError
from langchaint.inference_params import InferenceParams
from langchaint.llm import (
    LLM,
    UNCHANGED,
    BoundLLM,
    SequenceNotStr,
    Unchanged,
)
from langchaint.messages import Message, TextPart
from langchaint.provider import Binding, Provider, StreamItem, ToolChoice
from langchaint.rate_limiter import RateLimiter
from langchaint.response import Response
from langchaint.streaming import StreamHandle
from langchaint.tools import ToolManager

type SpanAttributes = Mapping[str, str | bool | int | float | Sequence[str]]
"""A span's attributes, keyed by name.

The value union is the subset of OTel's AttributeValue this package emits;
the Sequence[str] arm exists because OTel attribute values include homogeneous string arrays
and the GenAI convention's finish-reason key gen_ai.response.finish_reasons is one.
"""

type AttributeMapper = Callable[[Response[object] | GenerationError], SpanAttributes]
"""Maps one generate result to its span attributes.

The parameter is Response[object] | GenerationError
because the mapper reads the shared Response/GenerationError fields; Response[object] accepts any Response[OutputT]
because Response's OutputT is inferred covariant (frozen dataclass, PEP 695 inference).
The mapper deliberately never receives the conversation, so no mapper can leak a prompt;
capturing prompt content requires subclassing TracedBoundLLM, whose methods have the conversation in scope.
"""

type AttributeFormat = Literal["gen_ai"]
"""The built-in attribute-mapper names; named after the attribute_format parameter, one name for the concept."""

_PACKAGE_VERSION = importlib.metadata.version("langchaint")
_CHAT_OPERATION = "chat"
"""The GenAI operation value for a chat completion (GenAiOperationNameValues.CHAT)."""

_logger = logging.getLogger("langchaint.tracing")


def _gen_ai_attributes(result: Response[object] | GenerationError) -> SpanAttributes:
    """Map a generate result to GenAI-convention span attributes plus langchaint scalars.

    Reads only the shared Response/GenerationError fields, so it cannot leak a prompt and cannot meaningfully fail.
    Keys the GenAI convention has not defined stay under the langchaint.* prefix,
    including the three-way cache partition (the convention has no cache_none counterpart,
    so the whole partition is kept together rather than split across two prefixes).
    gen_ai.response.finish_reasons is the plural array the convention defines;
    it is omitted when stop_reason is None (no completed turn).
    The usage and cost attributes are the call's paid totals across every attempt (result.usage is that scope),
    not one request's counts; per-attempt detail stays visible as the langchaint.attempt_failed span events.
    """
    usage = result.usage
    attributes: dict[str, str | bool | int | float | Sequence[str]] = {
        "gen_ai.provider.name": result.provider_name,
        "gen_ai.request.model": result.model,
        "gen_ai.usage.input_tokens": usage.input_tokens_total,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "gen_ai.usage.reasoning.output_tokens": usage.output_tokens_reasoning,
        "langchaint.attempts": result.attempts,
        "langchaint.cost_in_usd": usage.cost_in_usd,
        "langchaint.input_tokens_cache_read": usage.input_tokens_cache_read,
        "langchaint.input_tokens_cache_write": usage.input_tokens_cache_write,
        "langchaint.input_tokens_cache_none": usage.input_tokens_cache_none,
    }
    if result.stop_reason is not None:
        attributes["gen_ai.response.finish_reasons"] = [result.stop_reason]
    return attributes


_BUILTIN_MAPPERS: dict[AttributeFormat, AttributeMapper] = {"gen_ai": _gen_ai_attributes}
"""The built-in mappers, one per AttributeFormat literal."""


def _resolve_attribute_mapper(
    attribute_format: AttributeFormat | AttributeMapper,
) -> AttributeMapper:
    """Resolve the attribute_format parameter to one AttributeMapper.

    A callable passes through as the mapper; a literal selects its built-in.

    Raises:
        KeyError: attribute_format is a string outside the AttributeFormat set (a static type error for a typed caller;
            this guards an untyped one).
    """
    if callable(attribute_format):
        return attribute_format
    try:
        return _BUILTIN_MAPPERS[attribute_format]
    except KeyError:
        raise KeyError(
            f"unknown attribute_format {attribute_format!r}; "
            f"known formats: {sorted(_BUILTIN_MAPPERS)}"
        ) from None


def _record_attempt_failed_events(span: Span, result: Response[object] | GenerationError) -> None:
    """Add one langchaint.attempt_failed event per failed attempt in the result's records.

    Each event carries the attempt's error text and its own elapsed_seconds;
    events are stamped at recording time and need no wall-clock origin (the records carry only monotonic brackets).
    They answer the first question a slow traced call raises: was it the request or the retries.
    """
    for record in result.attempt_records:
        if record.error is not None:
            span.add_event(
                "langchaint.attempt_failed",
                {"error_text": str(record.error), "elapsed_seconds": record.elapsed_seconds},
            )


def _apply_result_attributes(
    span: Span,
    result: Response[object] | GenerationError,
    attribute_mapper: AttributeMapper,
) -> None:
    """Set the mapper's attributes and the langchaint.attempt_failed events on a recording span.

    Called on the success and the GenerationError paths, both of which carry the shared Response/GenerationError fields;
    never on the other-exception path, which has no such fields.
    Skipped entirely when the span is not recording (no TracerProvider configured, a sampler drop, or an ended span),
    the OTel guard for not computing attributes a non-recording span discards;
    the guard matters because a user AttributeMapper can be arbitrarily expensive.
    A mapper exception is caught and logged at warning level and never propagated,
    so a telemetry bug never discards a paid result; the langchaint.attempt_failed events are added first,
    so they survive a raising mapper, and the span keeps whatever attributes were already set.
    """
    if not span.is_recording():
        return
    _record_attempt_failed_events(span, result)
    try:
        attributes = attribute_mapper(result)
    except Exception:
        _logger.warning(
            "attribute_format mapper raised; leaving span attributes partial", exc_info=True
        )
        return
    span.set_attributes(attributes)


def _set_generation_error_status(span: Span, error: GenerationError) -> None:
    """Set error status from a terminal GenerationError, whose attributes are set separately."""
    span.set_status(Status(StatusCode.ERROR, error.error_text))


def _record_other_exception(span: Span, exc: Exception) -> None:
    """Record a non-GenerationError exception and set error status; no shared-field attributes exist."""
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


class TracedLLM:
    """Wraps an LLM so every binding it produces is traced.

    Wrapping is unconditional: an app wraps every LLM at construction and types its signatures as the Traced classes.
    Enabling, disabling, or routing tracing is OTel SDK configuration (a TracerProvider, a sampler, an exporter),
    never an application code change; an app that never configures an SDK gets non-recording no-op spans.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        attribute_format: AttributeFormat | AttributeMapper = "gen_ai",
        tracer: Tracer | None = None,
    ) -> None:
        """Resolve the mapper and the tracer once, at construction.

        tracer None resolves trace.get_tracer("langchaint.tracing", <package version>) now, not at import.
        attribute_format is resolved to one AttributeMapper and passed down unchanged to every binding.

        Raises:
            KeyError: attribute_format is a string outside the AttributeFormat set.
        """
        self._llm = llm
        self._attribute_mapper = _resolve_attribute_mapper(attribute_format)
        self._tracer = (
            tracer
            if tracer is not None
            else trace.get_tracer("langchaint.tracing", _PACKAGE_VERSION)
        )

    @property
    def provider(self) -> Provider:
        """The wrapped LLM's provider, so an app never reaches for a private field."""
        return self._llm.provider

    @property
    def rate_limiter(self) -> RateLimiter:
        """The wrapped LLM's RateLimiter, so an app sharing a budget never reaches for a private field."""
        return self._llm.rate_limiter

    @overload
    def bind[ModelT: BaseModel](
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: ToolManager | None = ...,
        response_format: type[ModelT],
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        automatic_prompt_caching: bool,
    ) -> "TracedBoundLLM[ModelT]": ...
    @overload
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tool_manager: ToolManager | None = ...,
        response_format: None = ...,
        inference_params: InferenceParams | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        automatic_prompt_caching: bool,
    ) -> "TracedBoundLLM[str]": ...
    def bind(
        self,
        *,
        system_prompt: str | Sequence[TextPart] | None = None,
        tool_manager: ToolManager | None = None,
        response_format: type[BaseModel] | None = None,
        inference_params: InferenceParams | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        automatic_prompt_caching: bool,
    ) -> "TracedBoundLLM[Any]":
        """Mirror LLM.bind and wrap its BoundLLM in a TracedBoundLLM carrying this tracer and mapper.

        The overloads re-declare LLM.bind's response_format split
        so a bound structured model gives TracedBoundLLM[Model] and an absent one gives TracedBoundLLM[str].
        """
        return TracedBoundLLM(
            bound_llm=self._llm.bind(
                system_prompt=system_prompt,
                tool_manager=tool_manager,
                response_format=response_format,
                inference_params=inference_params,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                automatic_prompt_caching=automatic_prompt_caching,
            ),
            tracer=self._tracer,
            attribute_mapper=self._attribute_mapper,
        )


class TracedBoundLLM[OutputT]:
    """Wraps a BoundLLM so every generate call opens a span.

    generate_one opens one CLIENT span (one outbound call);
    generate_many opens one INTERNAL span around the delegated batch (no per-item child spans,
    no outbound call of its own).
    The span name is the GenAI convention {operation} {model}, wrapper-owned,
    so a custom mapper changes attributes only, never the name, kind, or status.
    There is no langchaint.elapsed_seconds attribute:
    the span brackets the same interval elapsed_seconds measures (request start to completion,
    RateLimiter slot waits and backoff included), so the span's own duration already carries it.
    """

    def __init__(
        self,
        *,
        bound_llm: BoundLLM[OutputT],
        tracer: Tracer,
        attribute_mapper: AttributeMapper,
    ) -> None:
        """Store the wrapped BoundLLM, the tracer, and the mapper; compute the span name once."""
        self._bound_llm = bound_llm
        self._tracer = tracer
        self._attribute_mapper = attribute_mapper
        self._span_name = f"{_CHAT_OPERATION} {bound_llm.provider.model}"

    @property
    def provider(self) -> Provider:
        """The wrapped BoundLLM's provider."""
        return self._bound_llm.provider

    @property
    def binding(self) -> Binding:
        """The wrapped BoundLLM's frozen binding."""
        return self._bound_llm.binding

    @property
    def response_format(self) -> type[OutputT] | None:
        """The wrapped BoundLLM's response_format."""
        return self._bound_llm.response_format

    @property
    def tool_manager(self) -> ToolManager | None:
        """The wrapped BoundLLM's ToolManager, so the manual tool loop reads it as it read bound.tool_manager."""
        return self._bound_llm.tool_manager

    @property
    def rate_limiter(self) -> RateLimiter:
        """The wrapped BoundLLM's RateLimiter."""
        return self._bound_llm.rate_limiter

    @overload
    def rebind[NewModelT: BaseModel](
        self,
        *,
        response_format: type[NewModelT],
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_manager: ToolManager | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "TracedBoundLLM[NewModelT]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: None,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_manager: ToolManager | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "TracedBoundLLM[str]": ...
    @overload
    def rebind(
        self,
        *,
        response_format: Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_manager: ToolManager | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        inference_params: InferenceParams | Unchanged = ...,
        automatic_prompt_caching: bool | Unchanged = ...,
    ) -> "TracedBoundLLM[OutputT]": ...
    def rebind(
        self,
        *,
        response_format: type[BaseModel] | None | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tool_manager: ToolManager | None | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        inference_params: InferenceParams | Unchanged = UNCHANGED,
        automatic_prompt_caching: bool | Unchanged = UNCHANGED,
    ) -> "TracedBoundLLM[Any]":
        """Mirror BoundLLM.rebind and re-wrap the plain BoundLLM in a TracedBoundLLM.

        rebind is mirrored so a rebound object stays traced;
        a traced object whose rebind returned an untraced one would silently drop tracing,
        the worst failure mode available here.
        The three overloads re-declare BoundLLM.rebind's response_format split.
        """
        return TracedBoundLLM(
            bound_llm=self._bound_llm.rebind(
                response_format=response_format,
                system_prompt=system_prompt,
                tool_manager=tool_manager,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                inference_params=inference_params,
                automatic_prompt_caching=automatic_prompt_caching,
            ),
            tracer=self._tracer,
            attribute_mapper=self._attribute_mapper,
        )

    async def generate_one(self, conversation: str | Sequence[Message]) -> Response[OutputT]:
        """Open a span around the whole generate_one call, delegate, attribute, and end the span.

        The span brackets the same interval as elapsed_seconds (slot waits and backoff included).
        A GenerationError sets error status and the shared-field attributes and re-raises;
        any other exception sets error status and record_exception and re-raises;
        a success sets OK status and the attributes.
        The span ends exactly once, in the finally.

        Raises:
            GenerationError: the wrapped generate_one raised a terminal per-item result (retries exhausted,
                a refusal, or a truncation); the span is attributed and closed first.
            AbortBatchError: the wrapped generate_one classified an error as abort;
                the span records the exception and closes first.
        """
        span = self._tracer.start_span(self._span_name, kind=SpanKind.CLIENT)
        try:
            try:
                response = await self._bound_llm.generate_one(conversation)
            except GenerationError as exc:
                _apply_result_attributes(span, exc, self._attribute_mapper)
                _set_generation_error_status(span, exc)
                raise
            except Exception as exc:
                _record_other_exception(span, exc)
                raise
            _apply_result_attributes(span, response, self._attribute_mapper)
            span.set_status(Status(StatusCode.OK))
            return response
        finally:
            span.end()

    async def generate_many(
        self,
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = False,
    ) -> list[Response[OutputT] | GenerationError]:
        """Order-aligned batch under one INTERNAL span; per-item detail lives in the returned rows.

        warm_cache passes through to BoundLLM.generate_many, which documents it;
        the span brackets the warming first item and the rest alike.

        Delegates to BoundLLM.generate_many rather than re-gathering into per-item child spans:
        generate_many is a bulk convenience (dataset passes, evals),
        not the agent loop, which traces individual generate_one / stream_one turns;
        a batch user reads per-item cost and timing from the returned Response / GenerationError rows (to_row),
        for which spans add nothing.
        The one span here brackets the whole batch's wall time;
        kind is INTERNAL because it makes no outbound call of its own, the delegated items do;
        a CLIENT batch span would register as a phantom outbound call in APM dependency graphs.
        The mapper is not invoked (there is no single result to map);
        the span carries only langchaint.batch_item_count, the number of conversations
        (a langchaint.* attribute because the GenAI convention defines no batch-size key).
        The span stays OK on mixed per-item results, which come back as rows,
        and takes error status only when an AbortBatchError propagates.

        Raises:
            TypeError: conversations is a bare str (the whole-batch guard, in the delegated method).
            AbortBatchError: one item classified an error as abort;
                the delegated method cancels the in-flight siblings and the span records the exception
                before re-raising.
        """
        span = self._tracer.start_span(self._span_name, kind=SpanKind.INTERNAL)
        try:
            try:
                results = await self._bound_llm.generate_many(conversations, warm_cache=warm_cache)
            except Exception as exc:
                _record_other_exception(span, exc)
                raise
            if span.is_recording():
                span.set_attribute("langchaint.batch_item_count", len(results))
            span.set_status(Status(StatusCode.OK))
            return results
        finally:
            span.end()

    def stream_one(self, conversation: str | Sequence[Message]) -> "TracedStreamHandle[OutputT]":
        """Wrap the BoundLLM's StreamHandle in a TracedStreamHandle; no I/O and no span yet.

        The span opens lazily at the first item or at final(), matching StreamHandle's own contract that
        nothing starts until the stream is first driven.
        """
        return TracedStreamHandle(
            stream_handle=self._bound_llm.stream_one(conversation),
            tracer=self._tracer,
            attribute_mapper=self._attribute_mapper,
            span_name=self._span_name,
        )


class TracedStreamHandle[OutputT]:
    """Wraps a StreamHandle, owning one span across the stream's life.

    Items pass through by reference;
    nothing is rewrapped (the no-rewrap rule bans copying data into same-shape containers;
    observing an iterator is unaffected).
    The span opens lazily at the first __anext__ or at final(),
    records langchaint.time_to_first_token_seconds at the first item,
    takes error status on a failing or abandoned stream, and ends exactly once.
    """

    def __init__(
        self,
        *,
        stream_handle: StreamHandle[OutputT],
        tracer: Tracer,
        attribute_mapper: AttributeMapper,
        span_name: str,
    ) -> None:
        """Store the wrapped handle and the span pieces; the span is not started here."""
        self._stream_handle = stream_handle
        self._tracer = tracer
        self._attribute_mapper = attribute_mapper
        self._span_name = span_name
        self._span: Span | None = None
        self._span_started_at_monotonic_seconds: float | None = None
        self._span_ended = False
        self._first_item_seen = False

    def _ensure_span(self) -> Span:
        """Start the span on first use, recording its start time for langchaint.time_to_first_token_seconds."""
        if self._span is None:
            self._span = self._tracer.start_span(self._span_name, kind=SpanKind.CLIENT)
            self._span_started_at_monotonic_seconds = time.monotonic()
        return self._span

    def _end_span(self) -> None:
        """End the span if one is open and it has not already ended; ends at most once."""
        if self._span is not None and not self._span_ended:
            self._span.end()
            self._span_ended = True

    def _mark_first_item(self, span: Span) -> None:
        """Record the langchaint.time_to_first_token_seconds attribute on the first item's arrival, once.

        The value is the monotonic seconds from the span's start (the first __anext__,
        which is when the underlying request begins) to the first item.
        It is a langchaint.* attribute because the GenAI convention defines no time-to-first-token key.
        Set only when a first item passes through this iterator,
        so a stream drained by final() without iteration carries no such attribute.
        """
        if self._first_item_seen:
            return
        self._first_item_seen = True
        if span.is_recording() and self._span_started_at_monotonic_seconds is not None:
            span.set_attribute(
                "langchaint.time_to_first_token_seconds",
                time.monotonic() - self._span_started_at_monotonic_seconds,
            )

    def __aiter__(self) -> "TracedStreamHandle[OutputT]":
        """Return self; the wrapper is its own iterator."""
        return self

    async def __anext__(self) -> StreamItem:
        """Delegate to the inner handle, opening the span lazily and observing the first item and failures.

        StopAsyncIteration passes through without ending the span,
        so a following final() can still set attributes and end it.
        Any other exception ends the span with error status (a failing stream, or a cancellation, does not leak a span).

        Raises:
            StopAsyncIteration: the inner stream is exhausted; the span is left open for final().
            Exception: the inner stream raised (a transient failure after items, an abort, a protocol violation);
                the span records it, takes error status, and ends before the re-raise.
        """
        span = self._ensure_span()
        try:
            item = await self._stream_handle.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as exc:
            _record_other_exception(span, exc)
            self._end_span()
            raise
        except BaseException:
            self._end_span()
            raise
        self._mark_first_item(span)
        return item

    async def __aenter__(self) -> "TracedStreamHandle[OutputT]":
        """Enter the inner handle's context and return self; opening is still deferred."""
        await self._stream_handle.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the inner handle, then end the span if it is still open.

        A never-started span (constructed and abandoned without iterating or calling final()) ends nothing.
        A span already ended by a mid-iteration failure or by final() is left alone.
        An open span abandoned with an in-flight exception (a consuming loop body that raised) records that exception
        and takes error status before ending.
        """
        try:
            await self._stream_handle.__aexit__(exc_type, exc, traceback)
        finally:
            if self._span is not None and not self._span_ended:
                if isinstance(exc, Exception):
                    _record_other_exception(self._span, exc)
                self._end_span()

    async def final(self) -> Response[OutputT]:
        """Drain the inner stream, attribute the span from the Response, and end the span.

        A GenerationError from the inner final() (a structured refusal or truncation detected
        when the SDK parses the assembled message) sets error status and the shared-field attributes and re-raises;
        any other exception records it and takes error status; a success sets OK status and the attributes.
        The span ends exactly once: if a prior final(), a mid-iteration failure, or __aexit__ already ended it,
        this delegates to the inner final() (which re-raises or returns its cached Response)
        without touching the span again.

        Raises:
            GenerationError: the inner final() raised a terminal per-item result (a refusal or a truncation
                on the structured path, or retries exhausted while draining); the span is attributed and closed first.
            AbortBatchError: draining hit an error classified as abort; the span records it and closes.
            StreamProtocolError: the inner stream violated the event contract; the span records it and closes.
        """
        span = self._ensure_span()
        if self._span_ended:
            return await self._stream_handle.final()
        try:
            response = await self._stream_handle.final()
        except GenerationError as exc:
            _apply_result_attributes(span, exc, self._attribute_mapper)
            _set_generation_error_status(span, exc)
            self._end_span()
            raise
        except Exception as exc:
            _record_other_exception(span, exc)
            self._end_span()
            raise
        except BaseException:
            self._end_span()
            raise
        _apply_result_attributes(span, response, self._attribute_mapper)
        span.set_status(Status(StatusCode.OK))
        self._end_span()
        return response


__all__ = [
    "AttributeFormat",
    "AttributeMapper",
    "SpanAttributes",
    "TracedBoundLLM",
    "TracedLLM",
    "TracedStreamHandle",
]
