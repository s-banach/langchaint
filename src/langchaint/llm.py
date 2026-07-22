"""The client; generation happens only through a binding.

LLM composes an adapter and a RateLimiter.
LLM has no generate methods.
bind() freezes everything that determines the cacheable prompt prefix,
fixes the output type, and precomputes SDK keyword arguments once;
the returned BoundLLM takes only the per-request conversation.
There are no per-call parameter overrides; changing parameters is rebind().
The RateLimiter slot gates every request start on every path, retries included;
the retry loop feeds every failure and every success back
so a rate-limit error pauses admission account-wide until a request succeeds again.
"""

import asyncio
import time
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, SupportsIndex, overload

from pydantic import BaseModel

from langchaint.adapter import Adapter, Binding, BoundAdapter, ToolChoice
from langchaint.exceptions import (
    AbortBatchError,
    AttemptRecord,
    GenerationError,
    RetriesExhaustedError,
    TransientError,
    _extract_transient_errors,
)
from langchaint.inference_params import InferenceParams
from langchaint.messages import Message, TextPart, UserMessage
from langchaint.rate_limiter import RateLimiter
from langchaint.response import AbandonedCall, AbandonedCallLog, Response
from langchaint.streaming import StreamHandle
from langchaint.tools import ToolManager


class Unchanged:
    """Sentinel type for rebind parameters the caller leaves as bound.

    Not in __all__: a caller never constructs or passes it, since omitting the keyword is the interface;
    it appears only in the rebind signature the caller reads.
    """

    def __repr__(self) -> str:
        """Render the default as UNCHANGED in signatures and help() output."""
        return "UNCHANGED"


UNCHANGED = Unchanged()


class SequenceNotStr[T_co](Protocol):
    """A Sequence that a type checker rejects a bare str for.

    str satisfies Sequence[str | Sequence[Message]] (a str is a sequence of str),
    so a plain Sequence batch parameter statically accepts generate_many("hi"),
    which would run one request per character.
    This protocol structurally matches list and tuple but not str,
    because typeshed's str.__contains__ accepts only str while the protocol requires __contains__(value: object).
    Being covariant, it also accepts a caller's list[str] or list[list[UserMessage]],
    which the invariant list[str | Sequence[Message]] would reject.
    Same shape as openai._types.SequenceNotStr, originally from the useful_types library;
    index() and count() are omitted deliberately, matching it.
    If typeshed ever widens str.__contains__,
    the static rejection lapses and _reject_bare_str_batch remains the backstop.
    """

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[T_co]: ...
    def __contains__(self, value: object, /) -> bool:
        """Accept object, which str's str-only __contains__ cannot satisfy."""
        ...

    def __len__(self) -> int:
        """Match Sequence."""
        ...

    def __iter__(self) -> Iterator[T_co]:
        """Match Sequence."""
        ...

    def __reversed__(self) -> Iterator[T_co]:
        """Match Sequence."""
        ...


def _reject_bare_str_batch(conversations: SequenceNotStr[str | Sequence[Message]]) -> None:
    """Reject a bare str passed as the whole batch.

    The SequenceNotStr parameter type makes the type checker reject a bare str;
    this runtime guard is the backstop for untyped callers.

    Raises:
        TypeError: conversations is a bare str.
    """
    if isinstance(conversations, str):
        raise TypeError(
            "conversations is a bare str; wrap it in a list, or use generate_one"
            " for a single conversation"
        )


def _as_conversation(conversation: str | Sequence[Message]) -> Sequence[Message]:
    if isinstance(conversation, str):
        return (UserMessage(content=conversation),)
    return conversation


def _build_binding(
    *,
    system_prompt: str | Sequence[TextPart] | None,
    tool_manager: ToolManager | None,
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    inference_params: InferenceParams,
    automatic_prompt_caching: bool,
) -> Binding:
    """Convert bind arguments to the frozen Binding.

    Tool schema conversion happens here, once per binding.

    Raises:
        ValueError: system_prompt is an empty sequence of parts; pass None to bind no system prompt.
    """
    if system_prompt is not None and not isinstance(system_prompt, str):
        if not system_prompt:
            raise ValueError(
                "system_prompt is an empty sequence of parts; pass None to bind no system prompt"
            )
        system_prompt = tuple(system_prompt)
    return Binding(
        system_prompt=system_prompt,
        tool_schemas=() if tool_manager is None else tool_manager.schemas(),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        inference_params=inference_params,
        automatic_prompt_caching=automatic_prompt_caching,
    )


def _bind_adapter(
    adapter: Adapter, binding: Binding, response_format: type[Any] | None
) -> BoundAdapter[Any]:
    """Dispatch to the adapter bind method the response_format selects.

    The caller-visible output type comes from the bind / rebind overloads,
    so this returns BoundAdapter[Any] and the Any is confined here.
    The parameter is type[Any] | None, not type[BaseModel] | None,
    because rebind feeds it the stored response_format typed type[OutputT] | None:
    type[OutputT] with OutputT unbounded is not assignable to a BaseModel-bounded parameter,
    and narrowing with is None narrows the value, not OutputT.
    """
    if response_format is None:
        return adapter.bind_text(binding)
    return adapter.bind_structured(binding, response_format)


class LLM:
    """The un-bound client; holds what is shared across bindings."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Store the shared pieces; rate_limiter None means the defaults."""
        self.adapter = adapter
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()

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
    ) -> "BoundLLM[ModelT]": ...
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
    ) -> "BoundLLM[str]": ...
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
    ) -> "BoundLLM[Any]":
        """Freeze the prompt prefix and fix the output type.

        response_format=Model gives BoundLLM[Model] whose output is the SDK-parsed instance;
        absent gives BoundLLM[str] whose output is the assistant text.
        automatic_prompt_caching has no default: caching changes billing,
        so langchaint never chooses a caching configuration for the caller.
        Ad-hoc use is llm.bind(automatic_prompt_caching=False).generate_one(...).
        """
        binding = _build_binding(
            system_prompt=system_prompt,
            tool_manager=tool_manager,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            inference_params=(
                inference_params if inference_params is not None else InferenceParams()
            ),
            automatic_prompt_caching=automatic_prompt_caching,
        )
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, binding, response_format),
            response_format=response_format,
            binding=binding,
            tool_manager=tool_manager,
            rate_limiter=self.rate_limiter,
        )


class BoundLLM[OutputT]:
    """One frozen prefix plus the request methods; constructed by LLM.bind.

    tool_manager is kept for tool dispatch (the manual tool loop reads it);
    the provider only ever sees the converted schemas inside the binding.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        bound_adapter: BoundAdapter[OutputT],
        response_format: type[OutputT] | None,
        binding: Binding,
        tool_manager: ToolManager | None,
        rate_limiter: RateLimiter,
    ) -> None:
        """Store the frozen pieces; called by LLM.bind and rebind only."""
        self.adapter = adapter
        self.binding = binding
        self.response_format = response_format
        self.tool_manager = tool_manager
        self.rate_limiter = rate_limiter
        self._bound_adapter = bound_adapter

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
    ) -> "BoundLLM[NewModelT]": ...
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
    ) -> "BoundLLM[str]": ...
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
    ) -> "BoundLLM[OutputT]": ...
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
    ) -> "BoundLLM[Any]":
        """Replace bound fields; a left-out field keeps its current value.

        response_format is the one field whose change alters the static output type,
        so it drives the overload return type.
        Replace semantics: a passed inference_params replaces the bound one whole, never field-wise.
        Every rebind converts the binding to SDK keyword arguments again, a pure conversion with no I/O.
        Whether a rebind preserves the provider's prompt cache is provider-specific and partly undocumented
        (Anthropic documents the prefix order tools -> system -> messages),
        and it depends on which field a rebind changes and on which value that field moves between,
        so measure it on the deployment you ship on.
        langchaint owns no cache-safety matrix over this.
        A matrix carried in the code goes stale the moment a provider changes a model.
        """
        new_tool_manager = (
            self.tool_manager if isinstance(tool_manager, Unchanged) else tool_manager
        )
        new_binding = _build_binding(
            system_prompt=(
                self.binding.system_prompt
                if isinstance(system_prompt, Unchanged)
                else system_prompt
            ),
            tool_manager=new_tool_manager,
            tool_choice=(
                self.binding.tool_choice if isinstance(tool_choice, Unchanged) else tool_choice
            ),
            parallel_tool_calls=(
                self.binding.parallel_tool_calls
                if isinstance(parallel_tool_calls, Unchanged)
                else parallel_tool_calls
            ),
            inference_params=(
                self.binding.inference_params
                if isinstance(inference_params, Unchanged)
                else inference_params
            ),
            automatic_prompt_caching=(
                self.binding.automatic_prompt_caching
                if isinstance(automatic_prompt_caching, Unchanged)
                else automatic_prompt_caching
            ),
        )
        new_response_format = (
            self.response_format if isinstance(response_format, Unchanged) else response_format
        )
        return BoundLLM(
            adapter=self.adapter,
            bound_adapter=_bind_adapter(self.adapter, new_binding, new_response_format),
            response_format=new_response_format,
            binding=new_binding,
            tool_manager=new_tool_manager,
            rate_limiter=self.rate_limiter,
        )

    async def _generate_with_retries(
        self,
        conversation: Sequence[Message],
        *,
        attempt_records: list[AttemptRecord] | None = None,
    ) -> Response[OutputT]:
        """Run the retry loop every generate method shares.

        attempt_records, when given, is the caller's own empty list (the retry budget counts its
        length), appended to in place as each attempt settles: generate_one passes one so a
        cancellation that kills this frame leaves the settled attempts' records readable outside it.
        Appends happen only between awaits, so the list never holds a partial record.

        TransientError, AbortBatchError,
        and the GenerationError leaves raised directly by the adapter are honored without classification.
        Each attempt holds a RateLimiter slot for the request only;
        backoff sleeps outside the slot so a waiting task does not hold capacity.
        Every failure and every success is registered with the limiter while the slot is still held,
        so a rate-limit error pauses admission account-wide before anyone else is admitted and a completed request
        (a success, or a refusal or truncation that reached a 200) ends recovery.
        Every attempt is timed onto an AttemptRecord whose bracket is the send only,
        excluding the slot wait and the backoff sleep,
        so a slow request is distinguishable from time spent rate limited;
        a completed attempt's record carries its usage (with cost_in_usd) and usage_raw,
        a transport failure's usage is ZERO_USAGE and usage_raw None.

        Raises:
            AbortBatchError: the adapter classified an attempt's error as abort.
            RefusalError: the model refused on the structured path;
                the adapter's leaf is re-raised enriched with the attempt records, on the first attempt without a retry.
            MaxCompletionTokensExceededError: the structured response hit the token cap;
                re-raised enriched on the first attempt without a retry.
            RetriesExhaustedError: every attempt failed transiently and the budget ran out.
        """
        if attempt_records is None:
            attempt_records = []
        started_at_monotonic_seconds = time.monotonic()
        while len(attempt_records) < self.rate_limiter.max_attempts:
            async with self.rate_limiter.slot() as admission:
                attempt_started_at_monotonic_seconds = time.monotonic()
                try:
                    adapter_result = await self._bound_adapter.send(conversation)
                except AbortBatchError:
                    raise
                except GenerationError as exc:
                    self.rate_limiter.register_success(admission)
                    attempt_records.append(
                        AttemptRecord(
                            started_at_monotonic_seconds=attempt_started_at_monotonic_seconds,
                            ended_at_monotonic_seconds=time.monotonic(),
                            error=None,
                            usage=exc.usage,
                            usage_raw=exc.usage_raw,
                        )
                    )
                    raise type(exc)(
                        attempt_records=tuple(attempt_records),
                        model=self.adapter.model,
                        provider_name=self.adapter.provider_name,
                        elapsed_seconds=time.monotonic() - started_at_monotonic_seconds,
                        stop_reason=exc.stop_reason,
                    ) from exc
                except TransientError as exc:
                    error: TransientError = exc
                except Exception as exc:
                    error_class = self.adapter.classify(exc)
                    if error_class == "abort":
                        raise AbortBatchError(f"abort provider error: {exc}") from exc
                    error = TransientError(
                        str(exc),
                        retry_after_seconds=self.adapter.retry_after_seconds(exc),
                        is_rate_limit=error_class == "rate_limit",
                    )
                    error.__cause__ = exc
                else:
                    self.rate_limiter.register_success(admission)
                    attempt_records.append(
                        AttemptRecord(
                            started_at_monotonic_seconds=attempt_started_at_monotonic_seconds,
                            ended_at_monotonic_seconds=time.monotonic(),
                            error=None,
                            usage=adapter_result.usage,
                            usage_raw=adapter_result.usage_raw,
                        )
                    )
                    return Response(
                        output=adapter_result.output,
                        model=self.adapter.model,
                        provider_name=self.adapter.provider_name,
                        attempt_records=tuple(attempt_records),
                        elapsed_seconds=time.monotonic() - started_at_monotonic_seconds,
                        raw=adapter_result.raw,
                        stop_reason=adapter_result.stop_reason,
                        assistant_message=adapter_result.assistant_message,
                    )
                attempt_records.append(
                    AttemptRecord(
                        started_at_monotonic_seconds=attempt_started_at_monotonic_seconds,
                        ended_at_monotonic_seconds=time.monotonic(),
                        error=error,
                        usage=error.usage,
                        usage_raw=error.usage_raw,
                    )
                )
                delay_seconds = self.rate_limiter.register_transient_error(
                    _extract_transient_errors(attempt_records)
                )
            if len(attempt_records) < self.rate_limiter.max_attempts:
                await asyncio.sleep(delay_seconds)
        raise RetriesExhaustedError(
            attempt_records=tuple(attempt_records),
            model=self.adapter.model,
            provider_name=self.adapter.provider_name,
            elapsed_seconds=time.monotonic() - started_at_monotonic_seconds,
            stop_reason=None,
        )

    async def generate_one(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = None,
    ) -> Response[OutputT]:
        """Generate one response under the retry loop.

        A bare str is shorthand for a conversation of one UserMessage holding that text.
        Every non-success outcome propagates: RetriesExhaustedError on transient exhaustion,
        RefusalError or MaxCompletionTokensExceededError on the structured path,
        and AbortBatchError on an abort classification;
        the first three share the GenerationError base a caller can catch at once.

        abandoned_call_log, when given, receives one AbandonedCall if a cancellation (a caller's
        asyncio.timeout, a TaskGroup sibling failing, shutdown) cuts this call off. The append is
        the only channel that path has: no value reaches the caller, and the settled attempts'
        records live in this frame, which the cancellation unwinds. Every other outcome appends
        nothing, because its usage travels on the Response or the raised GenerationError, which the
        caller records itself.

        Raises:
            asyncio.CancelledError: an outer scope cancelled this call; when abandoned_call_log is
                given, the AbandonedCall is appended first.
        """
        attempt_records: list[AttemptRecord] = []
        try:
            return await self._generate_with_retries(
                _as_conversation(conversation), attempt_records=attempt_records
            )
        except asyncio.CancelledError:
            if abandoned_call_log is not None:
                abandoned_call_log.append(
                    AbandonedCall(
                        attempt_records=tuple(attempt_records),
                        model=self.adapter.model,
                        provider_name=self.adapter.provider_name,
                    )
                )
            raise

    async def _generate_or_failure(
        self, conversation: str | Sequence[Message]
    ) -> Response[OutputT] | GenerationError:
        """One batch item: the Response, or the GenerationError caught as the failure row.

        An AbortBatchError is not caught here, so it propagates out of the batch and cancels the siblings.
        """
        try:
            return await self._generate_with_retries(_as_conversation(conversation))
        except GenerationError as failure:
            return failure

    async def generate_many(
        self,
        conversations: SequenceNotStr[str | Sequence[Message]],
        *,
        warm_cache: bool = False,
    ) -> list[Response[OutputT] | GenerationError]:
        """Order-aligned batch: result i belongs to conversations[i].

        Each conversation may be a bare str, shorthand for a conversation of one UserMessage holding that text.
        A bare str as the whole batch is rejected: str satisfies the item Sequence type,
        so it would silently become one request per character.
        Continues past items that end in a GenerationError (retries exhausted, a refusal, a truncation);
        those come back as that error in their slot,
        which to_row renders to a failure row so the batch stays table-ready.
        Concurrency is bounded by rate_limiter.max_in_flight,
        which gates every request start and is shared with everything else using the same RateLimiter instance.
        An AbortBatchError in any item cancels the in-flight siblings and raises, because the batch is misconfigured.

        warm_cache runs conversations[0] to completion before starting the rest,
        because a provider cache entry is readable only after the response that writes it begins,
        so a batch sharing a cached prefix otherwise pays one cold cache write per in-flight item.
        It costs one item of serial latency and warms unconditionally,
        whether or not the binding places any cache marker.
        A first item ending in a GenerationError still admits the rest:
        a rejected 200 (a refusal, a truncation) wrote the prefix on the provider side,
        and after a transport failure the rest simply run against a cold cache; there is no second warmer.
        There is no warmup ladder: after the first item settles, every remaining item is admitted at once.
        An AbortBatchError from the first item raises before any sibling starts.
        """
        _reject_bare_str_batch(conversations)
        # The slices also convert the SequenceNotStr protocol to the Sequence _gather_or_cancel takes.
        if warm_cache and conversations:
            first_result = await self._generate_or_failure(conversations[0])
            return [first_result, *await self._gather_or_cancel(conversations[1:])]
        return await self._gather_or_cancel(conversations[0:])

    async def _gather_or_cancel(
        self, conversations: Sequence[str | Sequence[Message]]
    ) -> list[Response[OutputT] | GenerationError]:
        tasks = [
            asyncio.create_task(self._generate_or_failure(conversation))
            for conversation in conversations
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    def stream_one(
        self,
        conversation: str | Sequence[Message],
        *,
        abandoned_call_log: AbandonedCallLog | None = None,
    ) -> StreamHandle[OutputT]:
        """Build the stream handle; entering it with `async with` opens the request.

        A bare str is shorthand for a conversation of one UserMessage holding that text.
        Sync because nothing suspends until the handle is entered;
        see StreamHandle for the retry, close, and abandoned_call_log contracts.
        """
        return StreamHandle(
            adapter=self.adapter,
            bound_adapter=self._bound_adapter,
            conversation=_as_conversation(conversation),
            rate_limiter=self.rate_limiter,
            abandoned_call_log=abandoned_call_log,
        )
