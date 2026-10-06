"""The provider-neutral adapter contract."""

import email.utils
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, Literal, NamedTuple

from pydantic import BaseModel

from langchaint.billing.pricing import (
    ProviderBilling,
    category_cost_in_usd,
    invocation_cost_in_usd,
    require_finite_nonnegative_rate,
)
from langchaint.common.exceptions import StreamProtocolError, TransientError
from langchaint.common.messages import AssistantMessage, Message, StopReason, TextPart, ToolCall
from langchaint.common.request_failure import RequestFailure, _TerminalRequestFailureKind
from langchaint.tools import ToolSchema

_logger = logging.getLogger(__name__)


class ResponseIdentity(NamedTuple):
    """Provider response identifiers recorded on one settled request."""

    model_served: str
    response_id: str
    request_id: str | None


AUTH_STATUSES: frozenset[int] = frozenset({401, 403})
"""The HTTP statuses for failed authentication and denied permission."""


def retry_after_seconds_from_headers(headers: Mapping[str, str]) -> float | None:
    """Parse provider wait headers.

    `retry-after-ms` contains milliseconds.
    `retry-after` contains seconds or a GMT HTTP date.
    Return `None` when neither header contains a positive delay.
    Never raises, so a `request_failure` method can call it on any provider headers.

    Args:
        headers: The provider response headers.
    """
    retry_after_ms_header = headers.get("retry-after-ms")
    if retry_after_ms_header is not None:
        try:
            retry_after_seconds = float(retry_after_ms_header) / 1000.0
        except ValueError:
            pass
        else:
            if retry_after_seconds > 0:
                return retry_after_seconds
    retry_after_header = headers.get("retry-after")
    if retry_after_header is None:
        return None
    try:
        retry_after_seconds = float(retry_after_header)
    except ValueError:
        return _retry_after_seconds_from_http_date(retry_after_header)
    if retry_after_seconds > 0:
        return retry_after_seconds
    return None


def _retry_after_seconds_from_http_date(retry_after_header: str) -> float | None:
    """Return the seconds until a GMT HTTP date, or None for a past or unconvertible date.

    On Python 3.14, `email.utils.mktime_tz` raises `ValueError` for a year outside 1..9999.
    It raises `OverflowError` for a year above 2**31 - 1.
    A numeric field with hundreds of digits gives an integer whose difference with `time.time()` raises `OverflowError`.
    """
    if not retry_after_header.endswith("GMT"):
        return None
    parsed = email.utils.parsedate_tz(retry_after_header)
    if parsed is None:
        return None
    try:
        retry_after_seconds = email.utils.mktime_tz(parsed) - time.time()
    except (OverflowError, ValueError):
        return None
    if retry_after_seconds > 0:
        return retry_after_seconds
    return None


def should_retry_from_headers(headers: Mapping[str, str]) -> bool | None:
    """Read the provider's own retry directive from response headers.

    `None` defers the decision to the status.

    Args:
        headers: The provider response headers.
    """
    should_retry_header = headers.get("x-should-retry")
    if should_retry_header == "true":
        return True
    if should_retry_header == "false":
        return False
    return None


def request_failure_from_response(
    *, status_code: int, headers: Mapping[str, str], retries: bool, pauses_quota: bool
) -> RequestFailure:
    """Build the `RequestFailure` of one error response from what the provider's tables give its status.

    `retries` and `pauses_quota` are the table answers.
    Status 200 identifies a mid-stream error event, and the table answers stand.
    On every other status, the provider's `x-should-retry` directive decides whether the request retries.
    Both SDK clients read `x-should-retry` before status rules.
    Anthropic 0.120.2 and OpenAI 2.51.0 verify this behavior.
    `pauses_quota` stands whatever the directive says.
    A failure that pauses the quota and does not retry is `provider_failed_terminally`.
    Another failure that does not retry takes the first kind that applies:
    - `provider_failed_terminally` for status 200.
    - `auth` for `AUTH_STATUSES`.
    - `rejected` for another 4xx status.
    - `provider_failed_terminally` when the directive forbids a retry.
    - `unknown_exception` otherwise.
    `retry_after_seconds` comes from the headers when the failure retries or pauses the quota.

    Args:
        status_code: The response status code.
        headers: The provider response headers.
        retries: Whether the tables retry this status.
        pauses_quota: Whether the tables pause the rate-limit quota for this status.
    """
    retry_after_seconds = retry_after_seconds_from_headers(headers)
    directive = None if status_code == 200 else should_retry_from_headers(headers)
    if directive is True or (directive is None and retries):
        return RequestFailure(
            kind="transient", pauses_quota=pauses_quota, retry_after_seconds=retry_after_seconds
        )
    if pauses_quota:
        return RequestFailure(
            kind="provider_failed_terminally",
            pauses_quota=True,
            retry_after_seconds=retry_after_seconds,
        )
    return RequestFailure(
        kind=_terminal_kind(status_code=status_code, directive=directive),
        pauses_quota=False,
        retry_after_seconds=None,
    )


def _terminal_kind(*, status_code: int, directive: bool | None) -> _TerminalRequestFailureKind:
    if status_code == 200:
        return "provider_failed_terminally"
    if status_code in AUTH_STATUSES:
        return "auth"
    if 400 <= status_code < 500:
        return "rejected"
    if directive is False:
        return "provider_failed_terminally"
    return "unknown_exception"


def record_request_failure_fallthrough(
    fallthrough_counts: Counter[str],
    *,
    function_name: str,
    status_code: object,
    error_type: object,
) -> None:
    """Count and log one request failure that fell through to a status-family default.

    A provider's `request_failure` function calls this when no listed row matched the failure.
    The count and the warning make a new provider status or error type visible.

    Args:
        fallthrough_counts: The counter to increment by failure description.
        function_name: The provider function name used in the warning.
        status_code: The status code used in the failure description.
        error_type: The provider error type used in the failure description.
    """
    tag = f"status={status_code} type={error_type}"
    fallthrough_counts[tag] += 1
    _logger.warning("%s fell through to a status-family default for %s", function_name, tag)


REASONING_PART_SEPARATOR = "\n\n"
"""Text inserted between provider-delimited reasoning parts."""


@dataclass(frozen=True, kw_only=True)
class ReasoningDelta:
    """A chunk of the model's readable reasoning.

    Append `text` to the preceding reasoning text.
    """

    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True, kw_only=True)
class ToolCallDelta:
    """A chunk of one forming tool call's argument JSON.

    `id` and `name` match the completed `ToolCall`.
    Append `partial_args_json` values by `id`.
    """

    id: str
    name: str
    partial_args_json: str
    kind: Literal["tool_call_delta"] = "tool_call_delta"


type StreamItem = str | ReasoningDelta | ToolCallDelta | ToolCall
"""What a stream yields: answer text chunks, reasoning text deltas, tool-call argument deltas, and completed tool calls.

Answer text chunks are the provider SDK's own strings, passed through without a wrapper class or copy.
`ReasoningDelta` distinguishes reasoning text from answer text.
Each tool call is yielded once, complete, when its block closes.
A call's `ToolCallDelta` values all precede its completed `ToolCall`.
Two forming calls' deltas may interleave, so a consumer accumulates per id.
When the completed call's `args_json` is valid JSON, its concatenated deltas parse to the same JSON value.
An adapter may re-serialize the arguments it accumulated, so text equality is not promised.
Providers that deliver complete calls or empty arguments can produce no deltas.
Usage, cost, and stop reason are available on the `Generation` from `final()` instead of the stream.
"""


@dataclass(frozen=True, kw_only=True)
class SpecificToolChoice:
    """Tool choice that forces the model to call the named tool."""

    tool_name: str
    kind: Literal["specific_tool"] = "specific_tool"


@dataclass(frozen=True, kw_only=True)
class AllowedToolsChoice:
    """Tool choice restricted to named application tools.

    `tool_names` names entries in `Binding.tool_schemas`.
    `tool_names` permits no calls to `Binding.provider_executed_tools`.
    `mode="auto"` permits text or a named tool call.
    `mode="required"` requires a named tool call.

    Raises:
        ValueError: `tool_names` is empty.
    """

    tool_names: tuple[str, ...]
    mode: Literal["auto", "required"]
    kind: Literal["allowed_tools"] = "allowed_tools"

    def __post_init__(self) -> None:
        """Reject empty `tool_names`.

        Raises:
            ValueError: `tool_names` is empty.
        """
        if not self.tool_names:
            raise ValueError("AllowedToolsChoice.tool_names must not be empty")


type ToolChoice = Literal["auto", "required", "none"] | SpecificToolChoice | AllowedToolsChoice
"""Provider-neutral tool choice.

`"auto"` lets the model decide, `"required"` requires a tool call, and `"none"` forbids tool calls.
`SpecificToolChoice` forces one named tool.
`AllowedToolsChoice` restricts calls without changing the bound tool definitions.
Adapters that do not support `AllowedToolsChoice` reject it at bind time.
"""


def validated_provider_executed_tool_types(
    provider_executed_tools: tuple[Mapping[str, object], ...],
    *,
    supported_types: frozenset[str],
    adapter_name: str,
) -> frozenset[str]:
    """Validate each `type` discriminator and return its distinct values.

    Args:
        provider_executed_tools: The provider-shaped tool definitions.
        supported_types: The supported `type` values.
        adapter_name: The adapter name used in the error message.

    Raises:
        ValueError: A mapping lacks a supported string `type` value.
    """
    tool_types: set[str] = set()
    for provider_executed_tool in provider_executed_tools:
        tool_type = provider_executed_tool.get("type")
        if not isinstance(tool_type, str) or tool_type not in supported_types:
            raise ValueError(
                f"{adapter_name} provider_executed_tools require a supported string type"
            )
        tool_types.add(tool_type)
    return frozenset(tool_types)


@dataclass(frozen=True, kw_only=True)
class Binding:
    """The frozen prefix of one BoundLLM, in langchaint terms only.

    Every field determines the provider's cacheable prompt prefix or stays fixed for one binding.
    The `messages` argument of each `BoundAdapter` method contains the per-request data.
    """

    system_prompt: str | tuple[TextPart, ...] | None
    """The bound system prompt.

    A parts value carries `cache_breakpoint` values inside the system prompt.
    `AnthropicMessagesAdapter` renders one system text block per part.
    `OpenAIResponsesAdapter` sends the parts as a developer-role input message before the `Sequence[Message]`.
    """

    tool_schemas: tuple[ToolSchema, ...]
    provider_executed_tools: tuple[Mapping[str, object], ...]
    """Provider-shaped tool definitions executed by the provider."""

    tool_choice: ToolChoice
    parallel_tool_calls: bool
    max_completion_tokens: int | None
    reasoning_level: str | None
    temperature: float | None
    automatic_cache_breakpoints: bool
    """Whether `automatic_cache_breakpoints` is enabled.

    `automatic_cache_breakpoints=True` lets the adapter or provider select cache boundaries.
    `automatic_cache_breakpoints=False` forbids that selection where supported.
    `cache_breakpoint=True` requests an explicit breakpoint under either value.
    OpenAI models lacking `prompt_cache_options` require `automatic_cache_breakpoints=True`.
    Gemini cannot control implicit caching, so both values build identical requests.
    """

    extra_body: Mapping[str, object] | None = None
    """Provider wire-body fields sent verbatim on every request.

    Provider wire names map to values passed through by reference.
    Each adapter rejects keys that it populates.
    """

    def __post_init__(self) -> None:
        """Reject `AllowedToolsChoice` names absent from `tool_schemas`.

        Raises:
            ValueError: `AllowedToolsChoice.tool_names` contains an unknown name.
        """
        if not isinstance(self.tool_choice, AllowedToolsChoice):
            return
        bound_tool_names = {tool_schema.name for tool_schema in self.tool_schemas}
        unknown_tool_names = tuple(
            tool_name
            for tool_name in self.tool_choice.tool_names
            if tool_name not in bound_tool_names
        )
        if unknown_tool_names:
            raise ValueError(
                f"AllowedToolsChoice.tool_names contains names absent from "
                f"Binding.tool_schemas: {unknown_tool_names!r}"
            )


def reject_extra_body_keys_the_adapter_populates(
    extra_body: Mapping[str, object] | None,
    *,
    populated_keys: frozenset[str],
    normalized_key: Callable[[str], str] = str,
) -> None:
    """Reject colliding `extra_body` keys.

    `normalized_key` maps caller keys to the form used by `populated_keys`.

    Args:
        extra_body: The provider wire-body fields, or `None`.
        populated_keys: The request field names that the adapter populates.
        normalized_key: The function that normalizes a caller key before comparison.

    Raises:
        ValueError: An `extra_body` key normalizes into `populated_keys`.
    """
    if extra_body is None:
        return
    colliding = sorted(key for key in extra_body if normalized_key(key) in populated_keys)
    if colliding:
        raise ValueError(
            f"extra_body keys {colliding} collide with request fields the adapter populates. "
            "The adapter refuses these keys to prevent the SDK merge from silently overriding the binding"
        )


@dataclass(frozen=True, kw_only=True)
class UsableResponse[OutputT]:
    """A response whose assistant message is usable: it gives output or has tool calls.

    `output` contains text or a validated `response_format` instance.
    A structured binding's assistant message with tool calls and no instance has `output=None`.
    """

    output: OutputT
    assistant_message: AssistantMessage
    stop_reason: StopReason
    kind: Literal["usable_response"] = "usable_response"


@dataclass(frozen=True, kw_only=True)
class _UnusableResponseBase:
    """The base of every `UnusableResponse` variant: a response whose `assistant_message` is not usable."""

    assistant_message: AssistantMessage


@dataclass(frozen=True, kw_only=True)
class Refusal(_UnusableResponseBase):
    """A completed response that a refusal ended before it produced output."""

    kind: Literal["refusal"] = "refusal"


@dataclass(frozen=True, kw_only=True)
class MaxCompletionTokensExceeded(_UnusableResponseBase):
    """A completed 200 that reached the token cap before its JSON closed."""

    kind: Literal["max_completion_tokens_exceeded"] = "max_completion_tokens_exceeded"


@dataclass(frozen=True, kw_only=True)
class SchemaViolation(_UnusableResponseBase):
    """A finished assistant message whose text fails `response_format` validation.

    `validation_error_json` preserves pydantic's error details and rejected values without documentation URLs.
    """

    validation_error_json: str
    kind: Literal["schema_violation"] = "schema_violation"


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTransiently(_UnusableResponseBase):
    """A billable response reporting a transient provider failure.

    Generation records the request and retries.
    `error_text` becomes the request's `TransientError` text.
    `pauses_quota=True` pauses the rate-limit quota.
    Streaming records the request and raises `GenerationError`.
    Streaming cannot retry because the response stream already ended.
    """

    error_text: str
    pauses_quota: bool
    kind: Literal["provider_failed_transiently"] = "provider_failed_transiently"


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTerminally(_UnusableResponseBase):
    """A billable response containing a terminal provider failure.

    `error_text` preserves the provider's description.
    """

    error_text: str
    kind: Literal["provider_failed_terminally"] = "provider_failed_terminally"


@dataclass(frozen=True, kw_only=True)
class EmptyAssistantMessage(_UnusableResponseBase):
    """A finished assistant message with neither a `response_format` instance nor a ToolCall.

    The retry loop records the request and raises `GenerationError` without retrying.
    A retry would request a new sample.
    """

    kind: Literal["empty_assistant_message"] = "empty_assistant_message"


@dataclass(frozen=True, kw_only=True)
class ContextWindowExceeded(_UnusableResponseBase):
    """A 200 reporting that the request overflowed the model's context window.

    The retry loop records the request and raises `GenerationError` without retrying.
    The same request always overflows.
    """

    kind: Literal["context_window_exceeded"] = "context_window_exceeded"


@dataclass(frozen=True, kw_only=True)
class UnfinishedAssistantMessage(_UnusableResponseBase):
    """A response whose partial content is not a finished answer.

    `error_text` preserves the provider's description.
    """

    error_text: str
    kind: Literal["unfinished_assistant_message"] = "unfinished_assistant_message"


@dataclass(frozen=True, kw_only=True)
class RejectedMessages:
    """A `Sequence[Message]` the adapter will not put on the wire.

    The retry loop sends no request and raises `GenerationError` with this `error_text`.
    Nothing was sent or billed.
    """

    error_text: str
    kind: Literal["rejected_messages"] = "rejected_messages"


class _RejectedMessagesError(Exception):
    """Carry `RejectedMessages.error_text` through nested request conversion functions."""


@dataclass(frozen=True, kw_only=True)
class RequestParams(ABC):
    """Request params, built once per input and sent with every request for that input.

    Each adapter defines a subclass that `narrowed_request_params` validates.
    The SDK client holds credentials.
    `GenerationError.request_params` carries them for application storage.
    """

    @abstractmethod
    def as_json(self) -> str:
        """Render the request as a JSON object for an archive to hold as one cell."""
        ...


def narrowed_request_params[RequestParamsT: RequestParams](
    request_params: RequestParams, request_params_class: type[RequestParamsT]
) -> RequestParamsT:
    """Narrow neutral request params to the subclass one adapter builds for `open_stream`.

    Args:
        request_params: The neutral request params to narrow.
        request_params_class: The required adapter-specific `RequestParams` subclass.

    Raises:
        TypeError: `request_params` is not an instance of `request_params_class`.
    """
    if not isinstance(request_params, request_params_class):
        raise TypeError(
            f"expected request params this adapter built, got {type(request_params).__name__}"
        )
    return request_params


def request_params_json(request_params: RequestParams, *, omitted_class: type) -> str:
    """Render request params as JSON after removing `omitted_class` values.

    Convert unsupported JSON values to text.

    Args:
        request_params: The request params to render.
        omitted_class: The omit sentinel class whose values to remove.
    """
    return json.dumps(
        _without_omitted(asdict(request_params), omitted_class), default=_json_default
    )


def _without_omitted(value: object, omitted_class: type) -> object:
    if isinstance(value, dict):
        mapping: Mapping[object, object] = value
        return {
            key: _without_omitted(item, omitted_class)
            for key, item in mapping.items()
            if not isinstance(item, omitted_class)
        }
    if isinstance(value, list):
        items: Sequence[object] = value
        return [_without_omitted(item, omitted_class) for item in items]
    return value


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


type UnusableResponse = (
    Refusal
    | MaxCompletionTokensExceeded
    | ContextWindowExceeded
    | EmptyAssistantMessage
    | ProviderFailedTerminally
    | ProviderFailedTransiently
    | SchemaViolation
    | UnfinishedAssistantMessage
)
"""Every way reading a billable 200 can end when its assistant message is not usable.

Such an assistant message gives no output and has no tool calls.

The concrete variants make a `kind` match over `UnusableResponse` or `ResponseOutcome` exhaustive.
Each variant has distinct retry-loop behavior.
"""

type ResponseOutcome[OutputT] = UsableResponse[OutputT] | UnusableResponse
"""Every way reading one response can end: a usable response, or the reason it is not usable."""


class AdapterStream(ABC):
    """One open stream, backed by the SDK's stream manager."""

    @abstractmethod
    def items(self) -> AsyncIterator[StreamItem]:
        """Yield `StreamItem` values in arrival order.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.

        Raises:
            StreamProtocolError: The stream violates its event contract, such as ending without a terminal event.
        """
        ...

    @abstractmethod
    async def final(self) -> BaseModel:
        """Return the SDK response the stream's events assembled into, after the stream ends.

        Callable only after items() is exhausted; the adapter delegates assembly to the SDK stream manager.
        Returning the response lets the retry loop record billing before interpretation.
        """
        ...

    @abstractmethod
    def provider_billing(self) -> ProviderBilling | None:
        """Return currently reported billing, or `None` before the SDK reports any."""
        ...

    @abstractmethod
    def request_id(self) -> str | None:
        """Return the request-id header of the response this stream is reading, when the SDK has one.

        The header arrives before the first event, so this method works throughout the stream.
        This method performs no I/O.
        None where the SDK exposes no route to those headers, and None where the provider sent none.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection idempotently."""
        ...


class BoundAdapter[OutputT](ABC):
    """One adapter bound to a frozen prefix."""

    @abstractmethod
    def build_request_params(
        self, messages: Sequence[Message]
    ) -> RequestParams | RejectedMessages:
        """Convert messages and the binding into the request params every request for this input sends.

        Called once before the first request.
        Returns `RejectedMessages` before any request or retry budget use.
        Performs no I/O.

        Args:
            messages: The per-request messages.
        """
        ...

    @abstractmethod
    def billing_from_raw(self, raw: BaseModel) -> ProviderBilling:
        """Price one response's reported counters before `interpret`.

        Args:
            raw: The provider SDK response.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
            ValueError: Reported counters produce a negative normalized counter.
        """
        ...

    @abstractmethod
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Build `ResponseIdentity`.

        Args:
            raw: The provider SDK response.
            request_id: The request id from the response headers, or `None`.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
        """
        ...

    @abstractmethod
    def interpret(self, raw: BaseModel) -> ResponseOutcome[OutputT]:
        """Return a `UsableResponse` or an `UnusableResponse`.

        Args:
            raw: The provider SDK response.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
        """
        ...

    @abstractmethod
    async def open_stream(self, request_params: RequestParams) -> AdapterStream:
        """Open a streaming request.

        Args:
            request_params: The adapter-specific request params.

        Raises:
            TypeError: `request_params` has the wrong `RequestParams` subclass.
            Exception: The SDK fails before returning the stream.
        """
        ...


def _require_provider_name(
    client: object,
    *,
    provider_name: str,
    provider_name_by_client_class: Mapping[type, str],
) -> None:
    """Reject a client whose class fixes another provider name.

    Raises:
        ValueError: The client class contradicts `provider_name`.
    """
    reached = next(
        (
            name
            for client_class, name in provider_name_by_client_class.items()
            if isinstance(client, client_class)
        ),
        None,
    )
    if reached is not None and reached != provider_name:
        raise ValueError(
            f"provider_name={provider_name!r} contradicts the client: "
            f"{type(client).__name__} reaches {reached!r}"
        )


class Adapter(ABC):
    """Base class for a provider SDK adapter.

    The SDK client owns credentials and endpoints.
    """

    provider_name: str
    """The serving provider recorded on each generation and error."""

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = {}
    """SDK client classes that fix the serving provider.

    Exclude base client classes that accept arbitrary endpoints.
    """

    def __init__(
        self,
        *,
        client: object,
        model: str,
        provider_name: str,
        automatic_cache_breakpoints_default: bool,
    ) -> None:
        """Validate `provider_name` and store adapter-wide values.

        Args:
            client: The provider SDK client.
            model: The model id to send verbatim.
            provider_name: The serving provider recorded on generations and errors.
            automatic_cache_breakpoints_default: The default for automatic prompt-cache boundaries.

        Raises:
            ValueError: `client` fixes a provider that differs from `provider_name`.
        """
        _require_provider_name(
            client,
            provider_name=provider_name,
            provider_name_by_client_class=self.provider_name_by_client_class,
        )
        self.model: str = model
        self.provider_name = provider_name
        self.automatic_cache_breakpoints_default: bool = automatic_cache_breakpoints_default

    @abstractmethod
    def config_fingerprint_data(self) -> Mapping[str, object]:
        """Return a snapshot of stored adapter configuration that can form provider requests.

        Exclude the SDK client, credentials, pricing, and response-accounting configuration.
        `BoundLLM.config_fingerprint` adds the adapter class, model, and provider.
        `BoundLLM.config_fingerprint` also adds the binding and response format.
        An adapter may include a non-secret endpoint identity when its contract treats that identity as configuration.
        """
        ...

    @abstractmethod
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output.

        Pure conversion of the binding to SDK keyword arguments; no I/O.

        Args:
            binding: The provider-neutral binding.

        Raises:
            ValueError: `binding` asks for something this adapter cannot send.
        """
        ...

    @abstractmethod
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind structured output parsed into `response_format`.

        An assistant message with tool calls and no structured instance gives `output=None`.

        Args:
            binding: The provider-neutral binding.
            response_format: The pydantic model used to validate structured output.

        Raises:
            ValueError: `binding` contains unsupported values.
            pydantic.PydanticInvalidForJsonSchema: `response_format` cannot produce a JSON schema.
            pydantic.PydanticUserError: `response_format` is not fully defined.
        """
        ...

    @abstractmethod
    def request_failure(self, error: Exception) -> RequestFailure:
        """Return what one failed request means for its retry loop and its rate-limit quota.

        The retry loops pass every `Exception` a request raises, except two they map themselves.
        A `TransientError` is transient and pauses the quota when its `pauses_quota` is true.
        A `StreamProtocolError` is transient and pauses nothing.
        Return `kind="unknown_exception"` for an exception the adapter cannot place.
        Never raise, because a raise escapes the retry loop as an adapter defect.
        Each provider's function documents its tables and defaults for unknown statuses and error types.

        Args:
            error: The exception the failed request raised.
        """
        ...

    def request_id_from_error(self, error: Exception) -> str | None:  # noqa: ARG002
        """Return the request-id header carried by error, when the SDK exposes one.

        The base implementation returns `None` because it knows no SDK types.
        Adapters override it to read the SDK exception attribute.

        Args:
            error: The provider SDK exception.
        """
        return None


__all__ = [
    "AUTH_STATUSES",
    "REASONING_PART_SEPARATOR",
    "Adapter",
    "AdapterStream",
    "AllowedToolsChoice",
    "Binding",
    "BoundAdapter",
    "ContextWindowExceeded",
    "EmptyAssistantMessage",
    "MaxCompletionTokensExceeded",
    "ProviderBilling",
    "ProviderFailedTerminally",
    "ProviderFailedTransiently",
    "ReasoningDelta",
    "Refusal",
    "RejectedMessages",
    "RequestFailure",
    "RequestParams",
    "ResponseIdentity",
    "ResponseOutcome",
    "SchemaViolation",
    "SpecificToolChoice",
    "StreamItem",
    "StreamProtocolError",
    "ToolCallDelta",
    "ToolChoice",
    "TransientError",
    "UnfinishedAssistantMessage",
    "UnusableResponse",
    "UsableResponse",
    "category_cost_in_usd",
    "invocation_cost_in_usd",
    "narrowed_request_params",
    "record_request_failure_fallthrough",
    "reject_extra_body_keys_the_adapter_populates",
    "request_failure_from_response",
    "request_params_json",
    "require_finite_nonnegative_rate",
    "retry_after_seconds_from_headers",
    "should_retry_from_headers",
    "validated_provider_executed_tool_types",
]
