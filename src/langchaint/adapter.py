"""The provider-neutral adapter contract."""

import email.utils
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel

from langchaint.call import ResponseIdentity
from langchaint.exceptions import TransientError
from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, StopReason, TextPart, ToolCall
from langchaint.pricing import ProviderBilling as ProviderBilling  # noqa: PLC0414
from langchaint.shared_backoff import (
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    RetryThisOne,
    Verdict,
)
from langchaint.tools import ToolSchema

_logger = logging.getLogger(__name__)

type ErrorClassification = Literal[
    "transient", "invalid_request", "declared_final", "unknown_exception"
]
"""The retry loop's action for an unparsed failure or `DoNotRetry`.

`transient` retries this request.
`invalid_request`, `declared_final`, and `unknown_exception` fail this request with distinct errors.
"""


def retry_after_seconds_from_headers(headers: Mapping[str, str]) -> float | None:
    """Parse provider wait headers.

    `retry-after-ms` contains milliseconds.
    `retry-after` contains seconds or a GMT HTTP date.
    Return `None` when neither header contains a positive delay.

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
    if not retry_after_header.endswith("GMT"):
        return None
    parsed = email.utils.parsedate_tz(retry_after_header)
    if parsed is None:
        return None
    retry_after_seconds = email.utils.mktime_tz(parsed) - time.time()
    if retry_after_seconds > 0:
        return retry_after_seconds
    return None


def terminal_classification_from_response(
    *, status_code: int, headers: Mapping[str, str]
) -> Literal["invalid_request", "declared_final", "unknown_exception"]:
    """Name one terminal error using its status and retry directive.

    Args:
        status_code: The response status code.
        headers: The provider response headers.
    """
    if status_code == 200:
        return "declared_final"
    if 400 <= status_code < 500:
        return "invalid_request"
    if should_retry_from_headers(headers) is False:
        return "declared_final"
    return "unknown_exception"


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


def verdict_under_retry_directive(
    verdict: Verdict, *, headers: Mapping[str, str], retry_after: float | None
) -> Verdict:
    """Apply the provider's `x-should-retry` directive to `verdict`.

    Both SDK clients read `x-should-retry` before status rules.
    Anthropic 0.120.2 and OpenAI 2.51.0 verify this behavior.
    The directive decides whether this request retries.
    The status decides whether `SharedBackoff` pauses the rate-limit quota.
    Callers exclude status-200 mid-stream error events.

    Args:
        verdict: The status-based verdict.
        headers: The provider response headers.
        retry_after: The normalized retry delay for a forced retry.
    """
    directive = should_retry_from_headers(headers)
    if directive is False:
        if isinstance(verdict, PauseAll | PauseAllDoNotRetry):
            return PauseAllDoNotRetry(retry_after=verdict.retry_after)
        return DoNotRetry()
    if directive is True and isinstance(verdict, DoNotRetry):
        return RetryThisOne(retry_after=retry_after)
    return verdict


def verdict_from_transient_error(error: TransientError) -> PauseAll | RetryThisOne:
    """Map one `TransientError` to its `Verdict`.

    Args:
        error: The transient failure to map.
    """
    if error.is_rate_limit:
        return PauseAll(retry_after=error.retry_after_seconds)
    return RetryThisOne(retry_after=error.retry_after_seconds)


def record_parse_fallthrough(
    fallthrough_counts: Counter[str], *, parse_name: str, status_code: object, error_type: object
) -> None:
    """Count and log one parse fallthrough to a status-family default.

    A provider parse function calls this when no listed row matched the failure.
    The count and the warning make a new provider status or error type visible.

    Args:
        fallthrough_counts: The counter to increment by failure description.
        parse_name: The provider parse function name used in the warning.
        status_code: The status code used in the failure description.
        error_type: The provider error type used in the failure description.
    """
    tag = f"status={status_code} type={error_type}"
    fallthrough_counts[tag] += 1
    _logger.warning("%s fell through to a status-family default for %s", parse_name, tag)


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
Usage, cost, and stop reason are available on the `Response` from `final()` instead of the stream.
"""


@dataclass(frozen=True, kw_only=True)
class SpecificToolChoice:
    """Tool choice that forces the model to call the named tool."""

    tool_name: str
    kind: Literal["specific"] = "specific"


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
    inference_params: InferenceParams
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
class AdapterResult[OutputT]:
    """A successful provider turn normalized to langchaint values.

    `output` contains text or a validated `response_format` instance.
    A structured tool-call turn has `output=None` and carries its calls on `assistant_message`.
    """

    output: OutputT
    assistant_message: AssistantMessage
    stop_reason: StopReason
    kind: Literal["adapter_result"] = "adapter_result"


@dataclass(frozen=True, kw_only=True)
class NoOutput:
    """The `assistant_message` from a response that produced no output."""

    assistant_message: AssistantMessage


@dataclass(frozen=True, kw_only=True)
class Refusal(NoOutput):
    """A completed response whose structured parse found a refusal."""

    kind: Literal["refusal"] = "refusal"


@dataclass(frozen=True, kw_only=True)
class MaxCompletionTokensExceeded(NoOutput):
    """A completed 200 that reached the token cap before its JSON closed."""

    kind: Literal["max_completion_tokens_exceeded"] = "max_completion_tokens_exceeded"


@dataclass(frozen=True, kw_only=True)
class SchemaViolation(NoOutput):
    """A completed turn whose text fails `response_format` validation.

    `validation_error_json` preserves pydantic's error details and rejected values without documentation URLs.
    """

    validation_error_json: str
    kind: Literal["schema_violation"] = "schema_violation"


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTransiently(NoOutput):
    """A billable response reporting a transient provider failure.

    Generation records the attempt and retries.
    `reason` becomes the attempt's `TransientError` text.
    `is_rate_limit=True` produces `PauseAll` during generation.
    `PauseAll` pauses the rate-limit quota.
    Streaming records the attempt and raises `GenerationError`.
    Streaming cannot retry because the response stream already ended.
    """

    reason: str
    is_rate_limit: bool
    kind: Literal["provider_failed_transiently"] = "provider_failed_transiently"


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTerminally(NoOutput):
    """A billable response containing a terminal provider failure.

    `reason` preserves the provider's description.
    """

    reason: str
    kind: Literal["provider_failed_terminally"] = "provider_failed_terminally"


@dataclass(frozen=True, kw_only=True)
class EmptyTurn(NoOutput):
    """A completed turn that produced no instance and no ToolCall.

    The retry loop records the attempt and raises `GenerationError` without retrying.
    A retry would request a new sample.
    """

    kind: Literal["empty_turn"] = "empty_turn"


@dataclass(frozen=True, kw_only=True)
class ContextWindowExceeded(NoOutput):
    """A 200 reporting that the request overflowed the model's context window.

    The retry loop records the attempt and raises `GenerationError` without retrying.
    The same request always overflows.
    """

    kind: Literal["context_window_exceeded"] = "context_window_exceeded"


@dataclass(frozen=True, kw_only=True)
class UnfinishedTurn(NoOutput):
    """A response whose partial content is not a finished answer.

    `reason` preserves the provider's description.
    """

    reason: str
    kind: Literal["unfinished_turn"] = "unfinished_turn"


@dataclass(frozen=True, kw_only=True)
class InvalidRequest:
    """A `Sequence[Message]` the adapter will not put on the wire.

    The retry loop records no attempt and raises `GenerationError` with `reason`.
    Nothing was sent or billed.
    """

    reason: str
    kind: Literal["invalid_request"] = "invalid_request"


@dataclass(frozen=True, kw_only=True)
class RequestParams(ABC):
    """One request, built once per call and sent once per attempt.

    Each adapter defines a subclass that `narrowed_request` validates.
    The SDK client holds credentials.
    Failures carry the request for application storage.
    """

    @abstractmethod
    def as_json(self) -> str:
        """Render the request as a JSON object for an archive to hold as one cell."""
        ...


def narrowed_request[RequestT: RequestParams](
    request: RequestParams, request_class: type[RequestT]
) -> RequestT:
    """Narrow the neutral request to the subclass one adapter builds for `open_stream`.

    Args:
        request: The neutral request to narrow.
        request_class: The required adapter-specific request class.

    Raises:
        TypeError: `request` is not an instance of `request_class`.
    """
    if not isinstance(request, request_class):
        raise TypeError(f"expected a request this adapter built, got {type(request).__name__}")
    return request


def request_json(request: RequestParams, *, omitted_class: type) -> str:
    """Render a request as JSON after removing `omitted_class` values.

    Convert unsupported JSON values to text.

    Args:
        request: The request to render.
        omitted_class: The omit sentinel class whose values to remove.
    """
    return json.dumps(_without_omitted(asdict(request), omitted_class), default=_json_default)


def _without_omitted(value: object, omitted_class: type) -> object:
    if isinstance(value, dict):
        without_omitted: dict[object, object] = {}
        for key, item in value.items():
            key_object: object = key
            item_object: object = item
            if not isinstance(item_object, omitted_class):
                without_omitted[key_object] = _without_omitted(item_object, omitted_class)
        return without_omitted
    if isinstance(value, list):
        without_omitted_items: list[object] = []
        for item in value:
            item_object: object = item
            without_omitted_items.append(_without_omitted(item_object, omitted_class))
        return without_omitted_items
    return value


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


type NoOutputOutcome = (
    Refusal
    | MaxCompletionTokensExceeded
    | ContextWindowExceeded
    | EmptyTurn
    | ProviderFailedTerminally
    | ProviderFailedTransiently
    | SchemaViolation
    | UnfinishedTurn
)
"""Every outcome of a billable 200 that produced no output.

The concrete variants make a `kind` match over `NoOutputOutcome` or `ResponseOutcome` exhaustive.
Each variant has distinct retry-loop behavior.
"""

type ResponseOutcome[OutputT] = AdapterResult[OutputT] | NoOutputOutcome
"""What one completed 200 produced: the turn or the reason it yielded no output."""


class AdapterStream(ABC):
    """One open stream, backed by the SDK's stream manager."""

    @abstractmethod
    def items(self) -> AsyncIterator[StreamItem]:
        """Yield `StreamItem` values in arrival order.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.
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
    def billing_reported(self) -> ProviderBilling | None:
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
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Convert messages and the binding into the request every attempt of this call sends.

        Called once before the first attempt.
        Returns `InvalidRequest` before any request or retry budget use.
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
        """Return a normalized turn or a `NoOutputOutcome`.

        Args:
            raw: The provider SDK response.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
        """
        ...

    @abstractmethod
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open a streaming request.

        Args:
            request: The adapter-specific request.

        Raises:
            TypeError: `request` has the wrong `RequestParams` subclass.
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
    """The serving provider recorded on each result and error."""

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
            provider_name: The serving provider recorded on results and errors.
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

        A successful tool-call turn returns `None` because it contains no structured instance.

        Args:
            binding: The provider-neutral binding.
            response_format: The pydantic model used to validate structured output.

        Raises:
            ValueError: `binding` contains unsupported values.
            pydantic.PydanticInvalidForJsonSchema: `response_format` cannot produce a JSON schema.
            pydantic.PydanticUserError: `response_format` is not fully defined.
        """
        ...

    failure_types: ClassVar[tuple[type[Exception], ...]]
    """The exception types `parse` maps to a `Verdict` for `SharedBackoff.failure_types`.

    `parse` handles SDK status errors and `TransientError`.
    `classify` handles other transport failures.
    Every entry must be a strict subclass of `Exception`.
    """

    @abstractmethod
    def parse(self, failure: Exception) -> Verdict:
        """Map one `failure_types` exception to its `Verdict` for `SharedBackoff.parse`.

        A retry-after header sets only `retry_after`.
        Return a verdict for every input without raising.
        Each provider's `parse` documents defaults for unknown statuses and error types.

        Args:
            failure: The exception to parse.
        """
        ...

    @abstractmethod
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify an unparsed exception or name a terminal `DoNotRetry` error.

        Return `unknown_exception` for an unrecognized exception.

        Args:
            error: The exception to classify.
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
