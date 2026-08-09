"""The adapter contract.

An adapter wraps one official SDK client and is the only place provider knowledge lives:
converting the binding to SDK keyword arguments, sending, translating stream events, normalizing usage, computing cost,
and reading failures (`parse` for the retry-or-pause verdict, `classify` for the terminal name).
Adapters delegate stream assembly to the SDK and validate a structured response's text themselves:
the SDKs validate inside the call that returns the response, where a rejection reaches the caller as
an exception carrying neither the response nor its billing, and an adapter that validates in its own
frame answers a rejection with a variant carrying both.

Reporting model: an adapter reports one attempt; only the retry loop knows the call.
So `open_stream` and `AdapterStream.final` return what came back, and an adapter never
constructs a `GenerationError`, which is a verdict about a call it cannot see.
What no variant can describe is an attempt the adapter read no outcome from: those stay SDK exceptions
that propagate through the admitted() block, whose exit parses the `failure_types` ones.
`AdapterStream.items` is the exception to the return contract, because an async iterator can only
raise: a mid-stream failure reaches the retry loop as an exception and exits the block the same way.

Billing before interpretation: `AdapterStream.final` hands back the SDK response object
itself, and `billing_from_raw`, `identity_from_raw`, and `interpret` read it in separate calls the
retry loop makes. The loop records the response, its `Billing`, and its `ResponseIdentity` the
moment they arrive, so a raise from `interpret` still leaves the attempt and what it billed on the
call's record.

Binding model: `Adapter.bind_text` and `Adapter.bind_structured` convert the frozen prefix
(system_prompt, tool_schemas, tool_choice, parallel_tool_calls, inference_params, automatic_prompt_caching)
to precomputed SDK keyword arguments once;
`BoundAdapter.build_request` adds the per-call `messages` to them, and `open_stream`
takes the `RequestParams` it built, so every attempt of one call sends the same request and a
`Sequence[Message]` the adapter will not put on the wire is found before the first attempt.
The split into two bind methods is what fixes the output type at bind time:
each method is monomorphic in its output type, so no sentinel value has to imply a type downstream.
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
"""What a retry loop does with a failure Adapter.parse gave no verdict, or verdicted DoNotRetry.

A string classification, not an exception class; the retry loop maps it onto one.
The retry-or-pause decision for a parseable provider failure is Adapter.parse's, so classify is
consulted only for an exception outside Adapter.failure_types, and to name the terminal error for
one parse verdicted DoNotRetry.
"transient" is a transport failure that produced nothing parseable (a connection drop, a timeout):
the retry loop retries it alone, as RetryThisOne with no retry_after.
"invalid_request" is not retried: the provider rejected this request, so sending it again
would be rejected again (the retry loop raises InvalidRequestError).
"declared_final" is not retried: the provider answered with an error and named it,
whether on an error status marked final or on a mid-stream error event
(the retry loop raises ProviderDeclaredFinalError).
"unknown_exception" is not retried either, and says the adapter could not place the exception at all
(the retry loop raises UnknownExceptionError).
"""


def retry_after_seconds_from_headers(headers: Mapping[str, str]) -> float | None:
    """Parse the server-stated wait from response headers.

    Three readings, in order, taking the first that yields a positive number: the non-standard
    retry-after-ms header (milliseconds), which is the most precise; retry-after as float seconds;
    and retry-after as an HTTP-date, accepted only in the form ending in "GMT" and converted by
    subtracting the current wall-clock time. Both SDK clients read the three in this order
    (anthropic 0.120.2, openai 2.51.0).
    The HTTP-date reading is the one place langchaint reads wall-clock time, because a timestamp can
    only be compared against wall-clock time; a clock skewed against the provider's misreads it, and
    `SharedBackoff` caps this value at `longest_wait_seconds`.
    None means no usable server-stated delay; non-positive values are treated as absent.
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


def request_id_from_raw(raw: BaseModel) -> str | None:
    """Read the request-id header both SDKs attach to a response they parsed from an HTTP body.

    Absent rather than None on a response the streaming helper assembled from events, and None where
    the response arrived without the header (anthropic 0.120.0, openai 2.48.0).
    """
    return getattr(raw, "_request_id", None)


def terminal_classification_from_response(
    *, status_code: int, headers: Mapping[str, str]
) -> Literal["invalid_request", "declared_final", "unknown_exception"]:
    """Name the terminal error for one error response, by its status and its retry directive.

    Whether the failure is retried is Adapter.parse's verdict, so this only names what a
    DoNotRetry failure becomes: a 200 is a mid-stream error event raised on the live response, a
    failure the provider named itself; a 4xx is this request's rejection, whoever issued it; a
    status the non-standard x-should-retry header declared final is declared_final; and anything
    left is a status langchaint has no account of.
    The 4xx test comes before the header test, so a rejected request the directive marked
    final keeps the rejection name. declared_final states a disposition and never what failed,
    which the carried exception's own text names.
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
    """Let the provider's own x-should-retry directive override the verdict the status tables gave.

    Both SDK clients read x-should-retry ahead of every status rule, so on an error response the
    provider itself would judge, the directive decides whether this request is retried and no status
    overrides it (anthropic 0.120.2, openai 2.51.0 BaseClient._should_retry).
    It decides nothing about the account: _should_retry returns a bool, and the status is what says
    the whole domain is throttled. So "false" over a pausing verdict gives PauseAllDoNotRetry, which
    stops this request and still pauses the domain, and "true" promotes DoNotRetry to RetryThisOne,
    the other verdicts already retrying.
    retry_after is the wait the same response's headers named, which a promoted RetryThisOne carries.
    Callers exclude a status-200 failure before calling: that is a mid-stream error event raised on
    a response the provider accepted, whose headers the SDK never consults _should_retry about.
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
    """Map a TransientError raised inside an admitted() block to its verdict.

    The shared rule both provider parse functions apply to the one failure_types entry langchaint
    itself raises: a billable 200 whose body reports a transient provider-side failure, which the
    retry loop re-raises as a TransientError so the block's exit records it.
    is_rate_limit says the provider named a rate limit, so the whole domain pauses; anything else is
    the one request's to retry. Either verdict carries the error's own retry_after_seconds.
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


type ToolChoice = Literal["auto", "required", "none"] | SpecificToolChoice
"""Provider-neutral tool choice.

"auto" lets the model decide, "required" forces some tool call (Anthropic's "any"), and "none" forbids tool calls.
SpecificToolChoice forces one named tool.
OpenAI's allowed-tools subset form is deliberately unmapped: the binding already pins the tool list.
"""


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
    tool_choice: ToolChoice
    parallel_tool_calls: bool
    inference_params: InferenceParams
    automatic_prompt_caching: bool
    """Whether the adapter manages prompt caching automatically.

    True: the anthropic adapter marks the frozen prefix and each request's last message block as cache breakpoints;
    the openai adapter leaves the provider's implicit caching in place.
    False: the anthropic adapter writes no breakpoints of its own,
    and the openai adapter requests explicit-mode caching with no breakpoints,
    so a Sequence[Message] without marked parts caches nothing and pays no cache writes.
    Under either value, a part with cache_breakpoint True adds a breakpoint at exactly that boundary,
    so False plus marked parts is the fully user-specified caching configuration.
    On openai, False requires an adapter built with supports_prompt_cache_options True,
    which openai documents as gpt-5.6 and later; on any other model it raises at bind time.
    """

    extra_body: Mapping[str, object] | None = None
    """Provider wire-body fields sent verbatim on every request; None binds none.

    Keys are the provider's own wire names, and values pass through by reference.
    Each SDK merges extra_body over the named request parameters, extra_body winning on a
    duplicate key, so each adapter raises at bind time for a
    key it populates itself instead of letting the merge silently override the binding.
    """


def reject_extra_body_keys_the_adapter_populates(
    extra_body: Mapping[str, object] | None,
    *,
    populated_keys: frozenset[str],
    normalized_key: Callable[[str], str] = str,
) -> None:
    """Refuse a Binding.extra_body key the calling adapter populates itself.

    An adapter's bind path calls this with the wire names it sets as explicit keywords,
    because the SDK merge would let extra_body silently override them.
    normalized_key maps a caller's key to the form populated_keys holds, for an SDK whose merge
    matches extra_body keys to wire keys loosely; the default compares keys as written.

    Raises:
        ValueError: extra_body holds a key that normalizes into populated_keys.
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
    """One successful provider turn, normalized to langchaint terms.

    Everything interpret read off the response, and nothing the response itself already holds: the
    retry loop keeps the response it passed to interpret, and that is the one a Response carries.

    output is the assistant text (text bindings) or the response_format instance validated from the
    turn's text (structured bindings).
    A structured binding's output is None when the turn parsed no instance, which here means the model
    called tools: the calls are on assistant_message, and no failure occurred.
    A turn can both parse an instance and call tools, so tool_calls is what says whether a tool result
    is owed.
    assistant_message is the full turn including tool calls, for appending to a Sequence[Message].
    """

    output: OutputT
    assistant_message: AssistantMessage
    stop_reason: StopReason
    kind: Literal["adapter_result"] = "adapter_result"


@dataclass(frozen=True, kw_only=True)
class NoOutput:
    """The shared shape of a 200 that produced no output: the turn it produced instead.

    Declares assistant_message once, and is the runtime test that narrows to NoOutputOutcome:
    isinstance rejects a type alias, so an adapter helper whose result is an output value or a variant
    tests against this base and annotates the variants as NoOutputOutcome.
    Every subclass must be a variant of that alias; that is what makes the test sound.

    assistant_message is whatever turn the response carried, which every variant has one of and none
    of them can return as the output: the sentences a refusal wrote, the text a schema violation
    rejected, the fragment a run that failed had emitted. The retry loop puts it on the attempt's
    record, so what the request bought reaches the caller even where the item fails. A response
    carrying no turn at all gives an empty one.
    """

    assistant_message: AssistantMessage


@dataclass(frozen=True, kw_only=True)
class Refusal(NoOutput):
    """A completed 200 whose structured parse found the model refusing.

    The retry loop records the attempt and fails the item with a RefusalError, without retrying.
    Neither provider puts its explanation in one neutral place, so a caller wanting more than
    assistant_message narrows the attempt's raw: anthropic reports it on Message.stop_details, and
    openai in a refusal content part, which its other refusal condition
    (incomplete_details.reason "content_filter") does not produce.
    """

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
    """A completed turn whose text is not an instance of the binding's response_format.

    The retry loop records the attempt and fails the item with a SchemaViolationError, without
    retrying.

    validation_error_json is ValidationError.json(include_url=False), which travels to the caller on
    that error. Every field pydantic reports is in it, the rejected value included, because a caller
    reading it is deciding whether to change the model or the prompt. That is also why no part of it
    reaches SchemaViolationError's message: pydantic writes the caller's own validator text into each
    msg, so a validator naming the value it rejected would put generated content in every span.
    The URLs are dropped, pointing at pydantic's error documentation rather than at this rejection.
    """

    validation_error_json: str
    kind: Literal["schema_violation"] = "schema_violation"


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTransiently(NoOutput):
    """A billable 200 whose body reports a provider-side failure a resend may get past.

    The retry loop records the attempt and sends another, carrying reason as that attempt's
    TransientError text. A stream handle records the same attempt and fails the item with
    RetryUnavailableError: this outcome is read from the assembled response, so that stream is over
    and the handle opens no other.
    reason is the provider's own description of the failure.
    is_rate_limit says the provider named a rate limit. The retry loop's TransientError carries it
    through the admitted() block's exit, whose PauseAll verdict pauses the whole SharedBackoff
    domain, exactly as a 429 status does.
    A stream handle sets no such pause, so a sibling task learns of the limit from its own next
    request: this outcome arrives with the stream already over, concluded inside the handle rather
    than raised through the block.
    """

    reason: str
    is_rate_limit: bool
    kind: Literal["provider_failed_transiently"] = "provider_failed_transiently"


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTerminally(NoOutput):
    """A billable 200 whose body reports a provider-side failure a resend would hit again.

    The retry loop records the attempt and fails the item with a ProviderFailedTerminallyError,
    without retrying: what the body names is a property of the request, so the retry budget would buy
    the same body at full price each time.
    reason is the provider's own description of the failure, and travels to the caller on that error.
    """

    reason: str
    kind: Literal["provider_failed_terminally"] = "provider_failed_terminally"


@dataclass(frozen=True, kw_only=True)
class EmptyTurn(NoOutput):
    """A completed turn that produced no instance and no ToolCall.

    The retry loop records the attempt and fails the item with an EmptyTurnError, without retrying:
    the model finished on its own terms and emitted nothing a structured binding can return, so a
    resend is a fresh sample charged to a budget that means error recovery.
    """

    kind: Literal["empty_turn"] = "empty_turn"


@dataclass(frozen=True, kw_only=True)
class ContextWindowExceeded(NoOutput):
    """A 200 reporting that the request overflowed the model's context window.

    The retry loop records the attempt and fails the item with a ContextWindowExceededError, without
    retrying: the same request overflows identically every time.
    """

    kind: Literal["context_window_exceeded"] = "context_window_exceeded"


@dataclass(frozen=True, kw_only=True)
class UnfinishedTurn(NoOutput):
    """A 200 that is not a finished turn, so its content is not the answer.

    Continuing such a turn means sending its content back for the provider to resume, which langchaint
    has no code for, so presenting the partial content as the answer would be silently wrong data.
    The retry loop records the attempt and fails the item with an UnfinishedTurnError.
    reason states what the provider reported, naming the provider's own word, and becomes that error's
    message.
    """

    reason: str
    kind: Literal["unfinished_turn"] = "unfinished_turn"


@dataclass(frozen=True, kw_only=True)
class InvalidRequest:
    """A Sequence[Message] the adapter will not put on the wire; nothing was sent.

    Raised by no one and billed by no one: the retry loop records no attempt and fails the item with
    an InvalidRequestError. reason states what cannot be sent, and becomes that error's message.
    """

    reason: str
    kind: Literal["invalid_request"] = "invalid_request"


@dataclass(frozen=True, kw_only=True)
class RequestParams(ABC):
    """One request, built once per call and sent once per attempt.

    Each adapter declares its own subclass and narrows back to it with narrowed_request on the way
    in, the same way it narrows a raw response.
    Holds no credentials: the SDK client carries those, and a failure's request travels to wherever
    the application archives its failures.
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
    """Render request as a JSON object, dropping every field holding an omit sentinel.

    omitted_class is the SDK's class for the value meaning "send no such field", so a dropped field
    is one the request body does not carry, told apart from a field explicitly sent as null.
    A value the json module cannot render becomes its str(), because a table build must not fail over
    one cell of one row.
    """
    return json.dumps(_without_omitted(asdict(request), omitted_class), default=_json_default)


def _without_omitted(value: object, omitted_class: type) -> object:
    """Return value with every mapping key whose value is an omit sentinel dropped, recursively."""
    if isinstance(value, dict):
        mapping: dict[object, object] = value
        return {
            key: _without_omitted(item, omitted_class)
            for key, item in mapping.items()
            if not isinstance(item, omitted_class)
        }
    if isinstance(value, list):
        items: list[object] = value
        return [_without_omitted(item, omitted_class) for item in items]
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

The type an adapter helper returns beside its output value, tested at runtime with
isinstance(x, NoOutput). Spelled as the concrete variants rather than as NoOutput so that a match on
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
    Carries no output type: the stream yields items and hands back the assembled response, and what
    that response produced is BoundAdapter.interpret's answer.
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
        Handing back the response rather than what it produced is what lets the retry loop record
        the billing before interpreting it.
        """
        ...

    @abstractmethod
    def billing_reported(self) -> Billing | None:
        """Return what the provider has reported billing, or None where the SDK reports nothing yet.

        A counter the provider sends late is missing from it, so a caller that can still reach the
        assembled response must read that instead. The callers are those that cannot: a stream cut
        off by a cancellation, and one that broke before reaching its terminal event.
        Callable at any point in the stream's life, including before its first item. No I/O.
        """
        ...

    @abstractmethod
    def request_id(self) -> str | None:
        """Return the request-id header of the response this stream is reading, when the SDK has one.

        Callable at any point in the stream's life: the header arrives with the status line, before
        the first event. No I/O.
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

        Called once per call, before the first attempt, so messages the adapter will not put on
        the wire are found without a request going out and without a retry budget being spent on it.
        Returns InvalidRequest with the reason in that case. No I/O.
        """
        ...

    @abstractmethod
    def billing_from_raw(self, raw: BaseModel) -> Billing:
        """Build what one response billed: its reported counters, priced at the tier it reports.

        Called on every response the moment it arrives, before interpret. No I/O.
        A response reporting no counters at all bills zero counters, at the served tier's prices.

        Raises:
            TypeError: raw is not the SDK response type this bound adapter produces, which is a
                defect in langchaint rather than anything a provider did.
            ValueError: the response's reported counters are inconsistent, leaving a counter
                negative.
        """
        ...

    @abstractmethod
    def identity_from_raw(self, raw: BaseModel) -> ResponseIdentity:
        """Read what the response says about itself: its ids and the model that served it.

        Called on every response the moment it arrives, beside billing_from_raw. No I/O.
        Return response_id as a str, converting it here where a provider's own is not one:
        the adapter is the only place that knows the format, and langchaint invents no id.
        request_id is None where the response carries no request-id header.

        Raises:
            TypeError: raw is not the SDK response type this bound adapter produces, which is a
                defect in langchaint rather than anything a provider did.
        """
        ...

    @abstractmethod
    def interpret(self, raw: BaseModel) -> ResponseOutcome[OutputT]:
        """Read one SDK response as the turn it produced, or as the reason it produced none. No I/O.

        Every response goes through here, assembled by the SDK from its stream's events, so one
        function per binding decides what a response means.
        A response that produced no output is a returned variant, never a raise, so the retry loop
        decides the item's fate with the attempt already recorded.

        raw is typed BaseModel rather than the SDK response type because BoundAdapter is what
        BoundLLM holds, and the neutral core imports no SDK.

        Raises:
            TypeError: raw is not the SDK response type this bound adapter produces, which is a
                defect in langchaint rather than anything a provider did.
        """
        ...

    @abstractmethod
    async def open_stream(self, request: RequestParams) -> AdapterStream:
        """Open one streaming request and return the live stream.

        Opening performs the connection I/O, so a connection failure raises here, before any event is yielded.

        Raises:
            TypeError: request is not the RequestParams subclass this bound adapter builds, which is
                a defect in langchaint rather than anything a provider did.
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
                They are the attempts this adapter read no outcome from.
        """
        ...


class Adapter(ABC):
    """Base class for one adapter per provider SDK.

    An adapter is constructed with the SDK client to use and the provider_name that client reaches,
    which together are how Bedrock support arrives: pass AsyncAnthropicBedrock or AsyncBedrockOpenAI
    instead of the direct client, with provider_name "aws.bedrock".
    Credentials and endpoints belong to the SDK client.
    """

    provider_name: str
    """Which provider served the request, recorded on every Response and GenerationError.

    The value comes from the OpenTelemetry GenAI convention's gen_ai.provider.name value set,
    whose values include the four langchaint's own constructors write
    ("anthropic", "openai", "aws.bedrock", "gcp.gemini"), and the tracing subpackage emits it
    under that key, so a backend groups langchaint spans with any other instrumented client's.
    Whoever constructs the adapter states it, because the SDK client class does not determine it:
    one AsyncOpenAI carrying a base_url reaches any of several providers. For the client classes
    langchaint does know, Adapter.__init__ raises ValueError on a stated value contradicting
    provider_name_by_client_class.
    When the company that trained the model and the platform serving it differ, the platform is the
    value: the convention states the attribute may differ from the actual model provider, and its
    worked example sets "aws.bedrock" for Bedrock spans (opentelemetry-semantic-conventions 0.64b0,
    gen_ai.provider.name). Which company trained the model is read from the model identifier.
    """

    provider_name_by_client_class: ClassVar[Mapping[type, str]] = {}
    """The SDK client classes whose own auth and URL scheme fixes the provider they reach.

    Deliberately partial, and never a source for provider_name:
    it holds only the platform client classes (the Bedrock and Azure ones),
    because a base client reaches whatever its base_url points at.
    Pointing an AsyncOpenAI at another vendor's OpenAI-compatible endpoint is how Groq, DeepSeek, and xAI are reached,
    all of them gen_ai.provider.name values,
    so a base client in this map would make __init__ raise for every one of them.
    A client matching nothing here takes the caller's value.

    Never enter a base client class: that invariant is what lets the lookup use isinstance.
    An application's own subclass of a platform client is then still recognized as reaching that platform,
    whatever it adds (headers, auth, instrumentation).
    Enter AsyncOpenAI and isinstance would match AsyncBedrockOpenAI and AsyncAzureOpenAI through it,
    since both subclass it.
    """

    def __init__(self, *, client: object, model: str, provider_name: str) -> None:
        """Check client against the stated provider_name, then store model and provider_name.

        client is checked here and not stored.
        Its object annotation is the price of checking every adapter in one place.

        Rates are not stored here. Each provider's service tiers are its own words, so an adapter
        holds a mapping from the tier its responses report to the table that prices that tier,
        and no neutral shape spans the two key types. What the contract requires of an adapter is a
        Billing, not that it hold a table for every tier its responses can report.

        Raises:
            ValueError: client is an instance of a class in provider_name_by_client_class
                listed under a provider other than provider_name.
                Such a request succeeds and bills normally
                while every span it produces carries the wrong provider,
                a defect nothing surfaces until telemetry is grouped by provider,
                so the constructor raises before the first request.
        """
        reached = next(
            (
                name
                for client_class, name in self.provider_name_by_client_class.items()
                if isinstance(client, client_class)
            ),
            None,
        )
        if reached is not None and reached != provider_name:
            raise ValueError(
                f"provider_name={provider_name!r} contradicts the client: "
                f"{type(client).__name__} reaches {reached!r}"
            )
        self.model = model
        self.provider_name = provider_name

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
        """Bind for structured output parsed into response_format.

        The output type admits None because a turn holding tool calls parses to no instance while
        being a success: the tool calls are the turn. Both SDKs type their parsed instance Optional
        for the same reason. Every other turn that yields no instance is a NoOutputOutcome variant.

        Pure conversion of the binding to SDK keyword arguments; no I/O.

        Raises:
            ValueError: the binding asks for something this adapter cannot send,
                which is a defect to report before any request rather than a request to spend.
        """
        ...

    failure_types: ClassVar[tuple[type[Exception], ...]]
    """The exception types parse maps to a verdict, for SharedBackoff's failure_types.

    Each concrete adapter states its own: its SDK's status-error class, whose response carries the
    status and error type parse reads, plus TransientError, which the retry loop raises for a 200
    whose body reports a transient failure. Transport failures that produced nothing parseable (a
    connection drop, a timeout) stay out, propagating unparsed for classify to sort.
    Every entry must be a strict subclass of Exception; SharedBackoff rejects anything else at
    construction.
    """

    @abstractmethod
    def parse(self, failure: Exception) -> Verdict:
        """Map one failure_types exception to its verdict, for SharedBackoff's parse.

        A retry-after header never sets the verdict, only its retry_after.
        Return a verdict for every input without raising, unknown statuses and error types
        included, falling through to documented defaults; each provider's parse function names
        its table and its defaults.
        """
        ...

    @abstractmethod
    def classify(self, error: Exception) -> ErrorClassification:
        """Sort an exception parse gave no verdict, or name the terminal error for a DoNotRetry.

        Consulted by the retry loop on the two failures the verdict does not settle: an exception
        outside failure_types, where "transient" marks a transport failure the loop retries alone,
        and a failure parse verdicted DoNotRetry, where the terminal name picks the GenerationError.
        Every classification fails at most its own item.
        A provider states a status, never whether the binding or this one request caused it.
        A binding defect langchaint can detect raises at construction or bind time instead, before any request is sent.
        Anything the adapter cannot place must map to "unknown_exception",
        which fails the one item without a retry, so bugs surface without being retried silently.
        """
        ...

    def request_id_from_error(self, error: Exception) -> str | None:  # noqa: ARG002
        """Return the request-id header carried by error, when the SDK exposes one.

        The id provider support asks for, on an attempt that failed rather than returning a
        response; identity_from_raw carries it on the attempts that returned one.
        The base implementation knows no SDK types and returns None; adapters override it to read
        their SDK exception's own attribute.
        """
        return None
