"""The adapter contract.

An adapter wraps one official SDK client and is the only place provider knowledge lives:
converting the binding to SDK keyword arguments, sending, translating stream events, normalizing usage, computing cost,
and classifying errors.
Adapters delegate stream assembly to the SDK and validate a structured response's text themselves:
the SDKs validate inside the call that returns the response, where a rejection reaches the caller as
an exception carrying neither the response nor its billing, and an adapter that validates in its own
frame answers a rejection with a member carrying both.

Reporting model: an adapter reports one attempt; only the retry loop knows the call.
So `send`, `open_stream`, and `AdapterStream.final` return what came back, and an adapter never
constructs a `GenerationError`, which is a verdict about a call it cannot see.
What no member can describe is an attempt the adapter read no outcome from: those stay SDK exceptions
that propagate for `Adapter.classify` to sort.
`AdapterStream.items` is the exception to the return contract, because an async iterator can only
raise: a mid-stream failure reaches the stream handle as an exception and goes through `classify` too.

Receipt before interpretation: `send` and `AdapterStream.final` hand back the SDK response object
itself, and `usage_from_raw` and `interpret` read it in two separate calls the retry loop makes.
The loop records the response and its price the moment it arrives, so a raise from `interpret` still
leaves the attempt and what it billed on the call's record.

Binding model: `Adapter.bind_text` and `Adapter.bind_structured` convert the frozen prefix
(system_prompt, tool_schemas, tool_choice, parallel_tool_calls, inference_params, automatic_prompt_caching)
to precomputed SDK keyword arguments once;
the returned `BoundAdapter` accepts only the per-request conversation.
The split into two bind methods is what fixes the output type at bind time:
each method is monomorphic in its output type, so no sentinel value has to imply a type downstream.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel

from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, StopReason, TextPart, ToolCall
from langchaint.tools import ToolSchema
from langchaint.usage import Usage

type ErrorClassification = Literal[
    "rate_limit", "transient", "invalid_request", "declared_final", "unknown_exception"
]
"""Whether a retry may fix the error, and what to call it when it cannot.

A string classification, not an exception class; the retry loop maps it onto one.
"rate_limit" is transient and account-wide, so RateLimiter pauses admission for everyone sharing it.
"transient" is retried by the failing task alone.
"invalid_request" is not retried: the provider rejected this request, so sending it again
would be rejected again (the retry loop raises InvalidRequestError).
"declared_final" is not retried: the provider answered with an error and declared it final, so a
resend would fail the same way (the retry loop raises ProviderDeclaredFinalError).
"unknown_exception" is not retried either, and says the adapter could not place the exception at all
(the retry loop raises UnknownExceptionError).
"""


def retry_after_seconds_from_headers(headers: Mapping[str, str]) -> float | None:
    """Parse the server-stated wait from response headers.

    Tries the non-standard retry-after-ms header (milliseconds) first because it is more precise,
    then retry-after as float seconds; both providers send these on rate-limit responses.
    The HTTP-date form of retry-after is not parsed.
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
    if retry_after_header is not None:
        try:
            retry_after_seconds = float(retry_after_header)
        except ValueError:
            return None
        if retry_after_seconds > 0:
            return retry_after_seconds
    return None


def classification_from_response(
    *, status_code: int, headers: Mapping[str, str], rate_limit_statuses: Collection[int]
) -> ErrorClassification:
    """Classify one error response by its status and its retry directive.

    Both SDKs encode the same retry policy, which langchaint owns because it constructs its clients
    with max_retries=0: the non-standard x-should-retry header decides when present, then 408
    (request timeouts), 409 (lock timeouts), 429, and 500 and above retry, and nothing else does.
    What that policy retries is retried here, and outside rate_limit_statuses what it declines is
    named: a 4xx is this request's rejection, whoever issued it, a status the provider declared
    final is declared_final, and anything left is a status langchaint has no account of.
    The 4xx test comes before the declared_final test, so a rejected request the directive marked
    final keeps the rejection name. declared_final states a disposition and never what failed,
    because that is all the directive says.

    rate_limit_statuses is the provider's own set of rate-limit statuses
    (429 for both, plus anthropic's 529), classified ahead of the retry directive because that
    directive speaks for the one request while the pause a rate limit triggers protects the shared
    account. A rate-limit status is therefore classified rate_limit whatever the directive says,
    so the account-wide pause is never lost to a verdict about the one request.
    """
    if status_code in rate_limit_statuses:
        return "rate_limit"
    should_retry = should_retry_from_headers(headers)
    if should_retry:
        return "transient"
    if should_retry is None and (status_code in (408, 409) or status_code >= 500):
        return "transient"
    if 400 <= status_code < 500:
        return "invalid_request"
    if should_retry is False:
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


type StreamItem = str | ToolCall
"""What a stream yields: text chunks and completed tool calls.

Text chunks are the provider SDK's own strings, passed through without a wrapper class or copy.
Each tool call is yielded once, complete, when its block closes;
there are no tool-call delta items because a consumer cannot act on partial argument JSON,
and both SDKs accumulate the arguments and hand over the finished call.
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
    per-request data is the conversation argument of the BoundAdapter methods, nothing else.
    """

    system_prompt: str | tuple[TextPart, ...] | None
    """The bound system prompt; None binds none.

    The parts form exists to carry cache_breakpoint marks inside the system prompt:
    the anthropic adapter renders one system text block per part,
    and the openai adapter sends the parts as a developer-role input message ahead of the conversation
    (the SDK documents `instructions` as "a system (or developer) message inserted into the model's context",
    and only input message parts carry prompt_cache_breakpoint).
    A plain str renders as one anthropic system block and as the openai instructions parameter.
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
    so a conversation without marked parts caches nothing and pays no cache writes.
    Under either value, a part with cache_breakpoint True adds a breakpoint at exactly that boundary,
    so False plus marked parts is the fully user-specified caching configuration.
    On openai, False reaches the wire only through an adapter built with supports_prompt_cache_options True,
    which openai documents as gpt-5.6 and later; on any other model the parameter is unsent
    and the provider's implicit caching stays in place whatever this value says.
    Older openai models cache automatically with free writes, so False buys nothing on them; bind True.
    """


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
    assistant_message is the full turn including tool calls, for appending to a conversation.
    """

    output: OutputT
    assistant_message: AssistantMessage
    stop_reason: StopReason


@dataclass(frozen=True, kw_only=True)
class NoOutput:
    """The shared shape of a 200 that produced no output: the turn it produced instead.

    Declares assistant_message once, and is the runtime test that narrows to NoOutputOutcome:
    isinstance rejects a type alias, so an adapter helper whose result is an output value or a member
    tests against this base and annotates the members as NoOutputOutcome.
    Every subclass must be a member of that alias; that is what makes the test sound.

    assistant_message is whatever turn the response carried, which every member has one of and none
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


@dataclass(frozen=True, kw_only=True)
class MaxCompletionTokensExceeded(NoOutput):
    """A completed 200 that reached the token cap before its JSON closed.

    The retry loop records the attempt and fails the item with a MaxCompletionTokensExceededError,
    without retrying.
    """


@dataclass(frozen=True, kw_only=True)
class SchemaViolation(NoOutput):
    """A completed turn whose text is not an instance of the binding's response_format.

    The retry loop records the attempt and fails the item with a SchemaViolationError, without
    retrying: the turn completed, so nothing about the attempt was transient.

    validation_error_json is ValidationError.json(include_url=False), which travels to the caller on
    that error. Every field pydantic reports is in it, the rejected value included, because a caller
    reading it is deciding whether to change the model or the prompt. That is also why no part of it
    reaches SchemaViolationError's message: pydantic writes the caller's own validator text into each
    msg, so a validator naming the value it rejected would put generated content in every span.
    The URLs are dropped, pointing at pydantic's error documentation rather than at this rejection.
    """

    validation_error_json: str


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTransiently(NoOutput):
    """A billable 200 whose body reports a provider-side failure a resend may get past.

    The retry loop records the attempt and sends another, carrying reason as that attempt's
    TransientError text. A stream handle records the same attempt and fails the item with
    RetryUnavailableError: this outcome is read from the assembled response, so that stream is over
    and the handle opens no other.
    reason is the provider's own description of the failure.
    is_rate_limit says the provider named a rate limit. The retry loop's TransientError carries it to
    the RateLimiter, which pauses admission for every task sharing it, exactly as a 429 status does.
    A stream sets no such pause, so a sibling task learns of the limit from its own next request.
    """

    reason: str
    is_rate_limit: bool


@dataclass(frozen=True, kw_only=True)
class ProviderFailedTerminally(NoOutput):
    """A billable 200 whose body reports a provider-side failure a resend would hit again.

    The retry loop records the attempt and fails the item with a ProviderFailedTerminallyError,
    without retrying: what the body names is a property of the request, so the retry budget would buy
    the same body at full price each time.
    reason is the provider's own description of the failure, and travels to the caller on that error.
    """

    reason: str


@dataclass(frozen=True, kw_only=True)
class EmptyTurn(NoOutput):
    """A completed turn that produced no instance and no tool call.

    The retry loop records the attempt and fails the item with an EmptyTurnError, without retrying:
    the model finished on its own terms and emitted nothing a structured binding can return, so a
    resend is a fresh sample charged to a budget that means error recovery.
    """


@dataclass(frozen=True, kw_only=True)
class ContextWindowExceeded(NoOutput):
    """A 200 reporting that the conversation overflowed the model's context window.

    The retry loop records the attempt and fails the item with a ContextWindowExceededError, without
    retrying: the same conversation overflows identically every time.
    """


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


@dataclass(frozen=True, kw_only=True)
class InvalidRequest:
    """A conversation the adapter will not put on the wire; nothing was sent.

    Raised by no one and billed by no one: the retry loop records no attempt and fails the item with
    an InvalidRequestError. reason states what cannot be sent, and becomes that error's message.
    """

    reason: str


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
isinstance(x, NoOutput). Spelled as the concrete members rather than as NoOutput so that a match over
it, or over ResponseOutcome, is provably exhaustive: a member added here without a match case in each
retry loop is a type error rather than a silent fall-through.
The members differ in what the retry loop does with them, which is why they are separate types rather
than one type carrying a reason: see each class.
"""

type ResponseOutcome[OutputT] = AdapterResult[OutputT] | NoOutputOutcome
"""What one completed 200 produced: the turn, or the reason it yielded no output."""

type AttemptOutcome[OutputT] = ResponseOutcome[OutputT] | InvalidRequest
"""What one attempt produced, whether or not a request went out."""


class AdapterStream(ABC):
    """One open stream, backed by the SDK's stream manager.

    The adapter translates SDK events into StreamItem values as they pass through;
    assembly stays in the SDK.
    Carries no output type: the stream yields items and hands back the assembled response, and what
    that response produced is BoundAdapter.interpret's answer on either request path.
    """

    @abstractmethod
    def items(self) -> AsyncIterator[StreamItem]:
        """Yield text chunks and completed tool calls in arrival order.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.
        """
        ...

    @abstractmethod
    async def final(self) -> BaseModel:
        """Return the SDK response the stream's events assembled into, after the stream ends.

        Callable only after items() is exhausted; the adapter delegates assembly to the SDK stream manager.
        Handing back the response rather than what it produced is what lets the stream handle record
        the receipt before interpreting it, the same order the non-streaming path follows.
        """
        ...

    @abstractmethod
    def usage_reported(self) -> Usage | None:
        """Price what the provider has reported on this stream, or None where the SDK reports nothing yet.

        A counter the provider sends late is missing from it, so a caller that can still reach the
        assembled response must read that instead. The callers are the two that cannot: a cancelled
        stream, and one that broke before reaching its terminal event.
        Callable at any point in the stream's life, including before its first item. No I/O.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        ...


class BoundAdapter[OutputT](ABC):
    """One adapter bound to a frozen prefix.

    Constructed by Adapter.bind_text or Adapter.bind_structured, which precompute the SDK keyword arguments once;
    both request methods take only the per-request conversation.
    """

    @abstractmethod
    async def send(self, conversation: Sequence[Message]) -> BaseModel | InvalidRequest:
        """Send one non-streaming request and return the SDK response object it got back.

        Every 200 comes back this way, whatever it holds, so the retry loop records the response and
        its price before anything is read off it.
        A conversation the adapter will not put on the wire returns InvalidRequest, having sent nothing.

        Raises:
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
                They are the attempts this adapter read no outcome from.
        """
        ...

    @abstractmethod
    def usage_from_raw(self, raw: BaseModel) -> Usage:
        """Price one response's reported counters at the service tier that response reports.

        Called on every response the moment it arrives, before interpret. No I/O.
        A response reporting no counters at all prices to ZERO_USAGE.

        Raises:
            TypeError: raw is not the SDK response type this bound adapter produces, which is a
                defect in langchaint rather than anything a provider did.
        """
        ...

    @abstractmethod
    def interpret(self, raw: BaseModel) -> ResponseOutcome[OutputT]:
        """Read one SDK response as the turn it produced, or as the reason it produced none. No I/O.

        Both request paths go through here, the streaming one on the response the SDK assembled from
        the events, so one function per binding decides what a response means.
        A response that produced no output is a returned member, never a raise, so the retry loop
        decides the item's fate with the attempt already recorded.

        raw is typed BaseModel rather than the SDK response type because BoundAdapter is what
        BoundLLM holds, and the neutral core imports no SDK.

        Raises:
            TypeError: raw is not the SDK response type this bound adapter produces, which is a
                defect in langchaint rather than anything a provider did.
        """
        ...

    @abstractmethod
    async def open_stream(self, conversation: Sequence[Message]) -> AdapterStream | InvalidRequest:
        """Open one streaming request and return the live stream.

        Opening performs the connection I/O, so a connection failure raises here, before any event is yielded.
        A conversation the adapter will not put on the wire returns InvalidRequest instead, having opened nothing.

        Raises:
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
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
    whose members include the three langchaint's own constructors write
    ("anthropic", "openai", "aws.bedrock"), and the tracing subpackage emits it
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

        client is checked here and not stored;
        each adapter stores its own with_options copy.
        Its object annotation is the price of checking every adapter in one place.

        Rates are not stored here. Each provider's service tiers are its own words, so an adapter
        holds a mapping from the tier its responses report to the table that prices that tier,
        and no neutral shape spans the two key types. What the contract requires of an adapter is a
        priced Usage, not that it hold a table at all.

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
        """
        ...

    @abstractmethod
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundAdapter[ModelT | None]:
        """Bind for structured output parsed into response_format.

        The output type admits None because a turn holding tool calls parses to no instance while
        being a success: the tool calls are the turn. Both SDKs type their parsed instance Optional
        for the same reason. Every other turn that yields no instance is a NoOutputOutcome member.

        Pure conversion of the binding to SDK keyword arguments; no I/O.
        """
        ...

    @abstractmethod
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify an exception raised by send, open_stream, or a stream's items().

        Every classification fails at most its own item.
        A provider states a status, never whether the binding or this one conversation caused it.
        A binding defect langchaint can detect raises at construction or bind time instead, before any request is sent.
        Anything the adapter cannot place must map to "unknown_exception",
        which fails the one item without a retry, so bugs surface without being retried silently.
        """
        ...

    def retry_after_seconds(self, error: Exception) -> float | None:  # noqa: ARG002
        """Return the server-stated wait carried by error, when the SDK exposes one.

        The base implementation knows no SDK types and returns None;
        adapters override it to read their SDK exception's response headers via retry_after_seconds_from_headers.
        """
        return None
