"""The adapter contract.

An `Adapter` converts a `Binding`, sends requests, translates events, reports billing, and classifies failures.
`AdapterStream.final` returns the SDK response so the retry loop records billing before `BoundAdapter.interpret` runs.
`Adapter.bind_text` and `Adapter.bind_structured` precompute provider fields.
`BoundAdapter.build_request` adds per-call `messages` and rejects unsendable input before a request starts.
"""

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
from langchaint.pricing import Billing
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
    """Convert a GMT-suffixed HTTP-date retry-after into seconds from now, or None.

    None for a date that does not parse, one not ending in "GMT", and one at or before the current time.
    """
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
    """Name one terminal error using its status and retry directive."""
    if status_code == 200:
        return "declared_final"
    if 400 <= status_code < 500:
        return "invalid_request"
    if should_retry_from_headers(headers) is False:
        return "declared_final"
    return "unknown_exception"


def should_retry_from_headers(headers: Mapping[str, str]) -> bool | None:
    """Read the provider's own retry directive from response headers.

    x-should-retry is non-standard and both SDK clients obey it ahead of every status rule.
    None means the provider stated nothing and the status decides.
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

    Retry loops create `TransientError` for transient failures in billable responses.
    They raise it inside `SharedBackoff.admitted()`.
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
    Without them the default would absorb it silently.
    """
    tag = f"status={status_code} type={error_type}"
    fallthrough_counts[tag] += 1
    _logger.warning("%s fell through to a status-family default for %s", parse_name, tag)


REASONING_PART_SEPARATOR = "\n\n"
"""What an adapter puts between two reasoning parts a provider breaks structurally.

Providers delimit the parts of a turn's reasoning by structure, not by text: neither the boundary
nor any whitespace for it arrives on the wire. One constant keeps the two adapters from separating
parts differently.
"""


@dataclass(frozen=True, kw_only=True)
class ReasoningDelta:
    """A chunk of the model's readable reasoning.

    text is literal characters to append: a consumer concatenates the deltas and renders the result.
    An adapter supplies REASONING_PART_SEPARATOR as its own delta at each part boundary,
    so the concatenation reads as prose rather than running two parts together.
    """

    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True, kw_only=True)
class ToolCallDelta:
    """A chunk of one forming tool call's argument JSON.

    id and name are the values the completed ToolCall carries,
    so a consumer keys a buffer by id and labels it by name from the first delta.
    partial_args_json is literal characters to append to that buffer.
    """

    id: str
    name: str
    partial_args_json: str
    kind: Literal["tool_call_delta"] = "tool_call_delta"


type StreamItem = str | ReasoningDelta | ToolCallDelta | ToolCall
"""What a stream yields: answer text chunks, reasoning text deltas, tool-call argument deltas, and completed tool calls.

Answer text chunks are the provider SDK's own strings, passed through without a wrapper class or copy.
Reasoning is wrapped because it is the turn's second kind of text and a bare string could not be told from the answer;
a consumer routes the two to different places.
Each tool call is yielded once, complete, when its block closes.
A call's ToolCallDelta items all precede its completed ToolCall.
Two forming calls' deltas may interleave, so a consumer accumulates per id.
Where the completed call's args_json is valid JSON, its concatenated deltas parse to the same JSON value;
text equality is not promised, because an adapter may re-serialize the arguments it accumulated.
Zero deltas for a call is allowed:
an adapter whose provider delivers calls whole yields none, and empty arguments stream no fragments anywhere.
Usage, cost, and stop reason are not streamed; they live on the Response from final().
"""


@dataclass(frozen=True, kw_only=True)
class SpecificToolChoice:
    """Tool choice that forces the model to call the named tool."""

    tool_name: str
    kind: Literal["specific"] = "specific"


@dataclass(frozen=True, kw_only=True)
class AllowedToolsChoice:
    """Tool choice restricted to named application tools.

    `tool_names` names entries in `Binding.tool_schemas` and permits no calls to
    `Binding.provider_executed_tools`.
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

"auto" lets the model decide, "required" forces some tool call (Anthropic's "any"), and "none" forbids tool calls.
SpecificToolChoice forces one named tool.
AllowedToolsChoice restricts calls without changing the bound tool definitions, preserving their cacheable prefix.
Adapters without `AllowedToolsChoice` support reject `AllowedToolsChoice` at bind time.
"""


def validated_provider_executed_tool_types(
    provider_executed_tools: tuple[Mapping[str, object], ...],
    *,
    supported_types: frozenset[str],
    adapter_name: str,
) -> frozenset[str]:
    """Validate each `type` discriminator and return its distinct values.

    Raises:
        ValueError: a mapping lacks a supported string `type` value.
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

    Every field here determines the provider's cacheable prompt prefix or is fixed per binding by design;
    per-request data is the messages argument of the BoundAdapter methods, nothing else.
    """

    system_prompt: str | tuple[TextPart, ...] | None
    """The bound system prompt; None binds none.

    The parts form exists to carry cache_breakpoint marks inside the system prompt:
    the anthropic adapter renders one system text block per part,
    and the openai Responses adapter sends the parts as a developer-role input message ahead of the Sequence[Message]
    (the SDK documents `instructions` as "a system (or developer) message inserted into the model's context",
    and only input message parts carry prompt_cache_breakpoint).
    A plain str renders as one anthropic system block and as the Responses adapter's instructions parameter.
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
    """Provider wire-body fields sent verbatim on every request; None binds none.

    Keys are the provider's own wire names, and values pass through by reference.
    Each SDK merges extra_body over the named request parameters, extra_body winning on a
    duplicate key, so each adapter raises at bind time for a
    key it populates itself instead of letting the merge silently override the binding.
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

    Raises:
        ValueError: An `extra_body` key normalizes into `populated_keys`.
    """
    if extra_body is None:
        return
    colliding = sorted(key for key in extra_body if normalized_key(key) in populated_keys)
    if colliding:
        raise ValueError(
            f"extra_body keys {colliding} collide with request fields the adapter populates; "
            "the SDK merge would silently override the binding, so they are refused"
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
    """A completed 200 that reached the token cap before its JSON closed.

    The retry loop records the attempt and fails the item with a MaxCompletionTokensExceededError,
    without retrying.
    """

    kind: Literal["max_completion_tokens_exceeded"] = "max_completion_tokens_exceeded"


@dataclass(frozen=True, kw_only=True)
class SchemaViolation(NoOutput):
    """A completed turn whose text fails `response_format` validation.

    `validation_error_json` preserves pydantic's error details and rejected values without documentation URLs.
    It stays outside the exception message because validator text can contain generated content.
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
    Streaming records the attempt and raises `RetryUnavailableError`.
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

    The retry loop records the attempt and raises `EmptyTurnError` without retrying.
    A retry would request a new sample.
    """

    kind: Literal["empty_turn"] = "empty_turn"


@dataclass(frozen=True, kw_only=True)
class ContextWindowExceeded(NoOutput):
    """A 200 reporting that the request overflowed the model's context window.

    The retry loop records the attempt and raises `ContextWindowExceededError` without retrying.
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
    """A Sequence[Message] the adapter will not put on the wire; nothing was sent.

    The retry loop records no attempt and raises `InvalidRequestError` with `reason`.
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
        """Render the request as a JSON object, for an archive to hold as one cell. No I/O."""
        ...


def narrowed_request[RequestT: RequestParams](
    request: RequestParams, request_class: type[RequestT]
) -> RequestT:
    """Narrow the neutral request to the subclass one adapter builds, for its open_stream.

    Raises:
        TypeError: request was built by another adapter, which is a defect in langchaint.
    """
    if not isinstance(request, request_class):
        raise TypeError(f"expected a request this adapter built, got {type(request).__name__}")
    return request


def request_json(request: RequestParams, *, omitted_class: type) -> str:
    """Render a request as JSON after removing `omitted_class` values.

    Convert unsupported JSON values to text.
    """
    return json.dumps(_without_omitted(asdict(request), omitted_class), default=_json_default)


def _without_omitted(value: object, omitted_class: type) -> object:
    """Return value with every mapping key whose value is an omit sentinel dropped, recursively."""
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
    """Render one value the json module rejects: an SDK model as its fields, anything else as text."""
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

Spelled as the concrete variants rather than as NoOutput so that a match on
kind over it, or over ResponseOutcome, is provably exhaustive: a variant added here without a match
case in each retry loop is a type error rather than a silent fall-through.
The variants differ in what the retry loop does with them, which is why they are separate types rather
than one type carrying a reason: see each class.
"""

type ResponseOutcome[OutputT] = AdapterResult[OutputT] | NoOutputOutcome
"""What one completed 200 produced: the turn, or the reason it yielded no output.

Also what one attempt produced, because build_request decides whether a request can be sent, so
every attempt sends one.
"""


class AdapterStream(ABC):
    """One open stream, backed by the SDK's stream manager.

    The adapter translates SDK events into StreamItem values as they pass through;
    assembly stays in the SDK.
    The stream yields items and returns the assembled response.
    `BoundAdapter.interpret` produces the output.
    """

    @abstractmethod
    def items(self) -> AsyncIterator[StreamItem]:
        """Yield StreamItem values in arrival order; StreamItem's docstring enumerates them.

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
    def billing_reported(self) -> Billing | None:
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
        """Close the underlying connection; idempotent."""
        ...


class BoundAdapter[OutputT](ABC):
    """One adapter bound to a frozen prefix.

    Constructed by Adapter.bind_text or Adapter.bind_structured, which precompute the SDK keyword arguments once;
    build_request adds the per-request messages, and open_stream takes what it built.
    """

    @abstractmethod
    def build_request(self, messages: Sequence[Message]) -> RequestParams | InvalidRequest:
        """Convert messages and the binding into the request every attempt of this call sends.

        Called once before the first attempt.
        Returns `InvalidRequest` before any request or retry budget use.
        Performs no I/O.
        """
        ...

    @abstractmethod
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Price one response's reported counters before `interpret`.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
            ValueError: Reported counters produce a negative normalized counter.
        """
        ...

    @abstractmethod
    def identity_from_raw(self, raw: BaseModel, *, request_id: str | None) -> ResponseIdentity:
        """Build `ResponseIdentity`.

        Use `raw` and the stream's `request_id`.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
        """
        ...

    @abstractmethod
    def interpret(self, raw: BaseModel) -> ResponseOutcome[OutputT]:
        """Return a normalized turn or a `NoOutputOutcome`.

        Raises:
            TypeError: `raw` has the wrong SDK response type.
        """
        ...

    @abstractmethod
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open a streaming request.

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
    def bind_text(self, binding: Binding) -> BoundAdapter[str]:
        """Bind for plain-text output.

        Pure conversion of the binding to SDK keyword arguments; no I/O.

        Raises:
            ValueError: the binding asks for something this adapter cannot send,
                which is a defect to report before any request rather than a request to spend.
        """
        ...

    @abstractmethod
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind structured output parsed into `response_format`.

        A successful tool-call turn returns `None` because it contains no structured instance.

        Raises:
            ValueError: `binding` contains unsupported values.
        """
        ...

    failure_types: ClassVar[tuple[type[Exception], ...]]
    """The exception types parse maps to a verdict, for SharedBackoff's failure_types.

    Each concrete adapter states its own: its SDK's status-error class, whose response carries the
    status and error type parse reads, plus TransientError, which the retry loop raises for a 200
    whose body reports a transient failure. Transport failures that produced nothing parseable (a
    connection drop, a timeout) stay out, propagating unparsed for classify to sort.
    Every entry must be a strict subclass of Exception.
    """

    @abstractmethod
    def parse(self, failure: Exception) -> Verdict:
        """Map one failure_types exception to its verdict, for SharedBackoff's parse.

        A retry-after header never sets the verdict, only its retry_after.
        Return a verdict for every input without raising.
        Each provider's `parse` documents defaults for unknown statuses and error types.
        """
        ...

    @abstractmethod
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify an unparsed exception or name a terminal `DoNotRetry` error.

        Return `unknown_exception` for an unrecognized exception.
        """
        ...

    def request_id_from_error(self, error: Exception) -> str | None:  # noqa: ARG002
        """Return the request-id header carried by error, when the SDK exposes one.

        Provider support uses this id for failed attempts.
        `identity_from_raw` handles responses.
        The base implementation returns `None` because it knows no SDK types.
        Adapters override it to read the SDK exception attribute.
        """
        return None
