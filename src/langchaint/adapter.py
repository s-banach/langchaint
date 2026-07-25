"""The adapter contract.

An adapter wraps one official SDK client and is the only place provider knowledge lives:
converting the binding to SDK keyword arguments, sending, translating stream events, normalizing usage, computing cost,
and classifying errors.
Adapters delegate stream assembly and structured-output parsing to the SDK
(`get_final_response` / `get_final_message`), which is generic in the response format,
so the output type flows from the SDK to the caller without reconstruction.

Reporting model: an adapter reports one attempt; only the retry loop knows the call.
So `send`, `open_stream`, and `AdapterStream.final` return what came back, and an adapter never
constructs a `GenerationError`, which is a verdict about a call it cannot see.
What no arm can describe is an attempt the adapter read no outcome from: those stay SDK exceptions
that propagate for `Adapter.classify` to sort.
`AdapterStream.items` is the exception to the return contract, because an async iterator can only
raise: a mid-stream failure reaches the stream handle as an exception and goes through `classify` too.

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

type ErrorClassification = Literal["rate_limit", "transient", "invalid_request", "unrecognized"]
"""Whether a retry may fix the error, and what to call it when it cannot.

A string classification, not an exception class; the retry loop maps it onto one.
"rate_limit" is transient and account-wide, so RateLimiter pauses admission for everyone sharing it.
"transient" is retried by the failing task alone.
"invalid_request" is not retried: the provider rejected this request, so sending it again
would be rejected again (the retry loop raises InvalidRequestError).
"unrecognized" is not retried either, and says the adapter cannot name the error
(the retry loop raises UnrecognizedError).
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
    named: a 4xx is this request's rejection, whoever issued it, and anything left is a status
    langchaint has no account of.
    A 5xx the provider declares final by sending x-should-retry: false lands there too:
    the directive states the disposition, never what failed.

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
    return "unrecognized"


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
class PricingTable:
    """USD prices per one million tokens, supplied at adapter construction.

    Cost is computed from raw provider counts because providers split counters the normalized Usage collapses.
    input_cache_none_usd_per_million_tokens prices only the uncached input, the partition's input_tokens_cache_none;
    cache reads and writes bill at their own rates.
    cache_write_usd_per_million_tokens applies to OpenAI too: OpenAI bills cache writes
    (reported as input_tokens_details.cache_write_tokens) starting with gpt-5.6.
    cache_write_1h_usd_per_million_tokens exists
    because Anthropic bills 5-minute and 1-hour cache writes at different rates;
    None means the adapter never sees 1-hour writes.

    There is no service-tier axis: one table is one tier's rates, and every cost langchaint reports
    assumes the response was served at the tier those rates are for. Both providers report the tier
    they actually used (anthropic on Usage.service_tier, openai on Response.service_tier), and a
    response served at another one prices here at these rates with no error, so an account on a
    premium or discounted tier reads a cost that is wrong by that tier's multiplier.
    Read the tier off the raw SDK object beside the Usage to detect it.
    """

    input_cache_none_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_usd_per_million_tokens: float
    cache_write_1h_usd_per_million_tokens: float | None = None


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

    output is the assistant text (text bindings) or the SDK-parsed response_format instance (structured bindings).
    assistant_message is the full turn including tool calls, for appending to a conversation.
    usage carries cost_in_usd, priced from raw provider counts against the adapter's PricingTable.
    usage_raw is the raw SDK usage object usage was normalized from, held by reference (no dump, no copy),
    None when the response reported no usage; a caller recovers provider-specific counts from it.
    raw is the SDK's own response model, held by reference (no dump, no copy).
    It is a live, mutable pydantic object, so despite the frozen dataclass around it,
    treat it read-only and raw.model_copy() before mutating.
    """

    output: OutputT
    assistant_message: AssistantMessage
    usage: Usage
    usage_raw: BaseModel | None
    stop_reason: StopReason
    raw: BaseModel


@dataclass(frozen=True, kw_only=True)
class Refused:
    """A completed 200 whose structured parse found the model refusing.

    The retry loop records the attempt and fails the item with a RefusalError, without retrying.
    """

    usage: Usage
    usage_raw: BaseModel | None


@dataclass(frozen=True, kw_only=True)
class Truncated:
    """A completed 200 that reached the token cap before its JSON closed.

    The retry loop records the attempt and fails the item with a MaxCompletionTokensExceededError,
    without retrying.
    """

    usage: Usage
    usage_raw: BaseModel | None


@dataclass(frozen=True, kw_only=True)
class Unparsed:
    """A billable 200 that produced no usable output, for a reason the stop reason does not name.

    The retry loop records the attempt and retries it, because a later attempt may produce output.
    A stream handle instead propagates it as a TransientError carrying this attempt's billing,
    because the stream already yielded items to the caller and is not reopened.
    """

    usage: Usage
    usage_raw: BaseModel | None


@dataclass(frozen=True, kw_only=True)
class NotSendable:
    """A conversation the adapter will not put on the wire; nothing was sent.

    Raised by no one and billed by no one: the retry loop records no attempt and fails the item with
    an InvalidRequestError. reason states what cannot be sent, and becomes that error's message.
    """

    reason: str


type ResponseOutcome[OutputT] = AdapterResult[OutputT] | Refused | Truncated | Unparsed
"""What one completed 200 produced.

Every arm but AdapterResult is a response the adapter read and rejected; all four reached a billable
200, so each carries that attempt's usage (with cost_in_usd inside) and the raw SDK usage object it
was normalized from, held by reference and None when the response reported no usage.
The arms differ in what the retry loop does with them, which is why they are four types rather than
one type carrying a reason: see each class.
"""

type AttemptOutcome[OutputT] = ResponseOutcome[OutputT] | NotSendable
"""What one attempt produced, whether or not a request went out."""


class AdapterStream[OutputT](ABC):
    """One open stream, backed by the SDK's stream manager.

    The adapter translates SDK events into StreamItem values as they pass through;
    assembly and structured-output parsing stay in the SDK.
    """

    @abstractmethod
    def items(self) -> AsyncIterator[StreamItem]:
        """Yield text chunks and completed tool calls in arrival order.

        Yields:
            Stream items; SDK events langchaint does not model are dropped.
        """
        ...

    @abstractmethod
    async def final(self) -> ResponseOutcome[OutputT]:
        """Return what the assembled response produced, after the stream ends.

        Callable only after items() is exhausted; the adapter delegates assembly and parsing to the SDK stream manager.
        A response the adapter reads and rejects is Refused, Truncated, or Unparsed rather than a raise,
        so the stream handle gets this attempt's billing and decides the item's fate itself.
        NotSendable cannot arrive here: the request is already open.
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
    async def send(self, conversation: Sequence[Message]) -> AttemptOutcome[OutputT]:
        """Send one non-streaming request and report what the attempt produced.

        A response the adapter reads and rejects is a returned arm, never a raise, so the retry loop
        records what the attempt billed before deciding the item's fate.

        Raises:
            Exception: the SDK's own exceptions propagate unchanged; Adapter.classify sorts them.
                They are the attempts this adapter read no outcome from.
        """
        ...

    @abstractmethod
    async def open_stream(
        self, conversation: Sequence[Message]
    ) -> AdapterStream[OutputT] | NotSendable:
        """Open one streaming request and return the live stream.

        Opening performs the connection I/O, so a connection failure raises here, before any event is yielded.
        A conversation the adapter will not put on the wire returns NotSendable instead, having opened nothing.

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

    def __init__(
        self, *, client: object, model: str, pricing: PricingTable, provider_name: str
    ) -> None:
        """Check client against the stated provider_name, then store model, pricing, provider_name.

        client is checked here and not stored;
        each adapter stores its own with_options copy.
        Its object annotation is the price of checking every adapter in one place.
        Moving the check into a base helper each adapter calls with its own precisely typed client is rejected:
        it makes the check opt-in, so an adapter whose author forgets the call is silently unguarded,
        which is the failure the check exists to prevent.

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
        self.pricing = pricing
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
    ) -> BoundAdapter[ModelT]:
        """Bind for structured output parsed by the SDK into response_format.

        Pure conversion of the binding to SDK keyword arguments; no I/O.
        """
        ...

    @abstractmethod
    def classify(self, error: Exception) -> ErrorClassification:
        """Classify an exception raised by send or open_stream.

        Every classification fails at most its own item.
        A provider states a status, never whether the binding or this one conversation caused it.
        A binding defect langchaint can detect raises at construction or bind time instead, before any request is sent.
        Anything the adapter cannot name must map to "unrecognized",
        which fails the one item without a retry, so bugs surface without being retried silently.

        Folding this into a request-failed AttemptOutcome arm is rejected.
        items() yields StreamItem values only, so a mid-stream failure reaches the retry loop as a raise anyway.
        The arm would give one event two channels.
        """
        ...

    def retry_after_seconds(self, error: Exception) -> float | None:  # noqa: ARG002
        """Return the server-stated wait carried by error, when the SDK exposes one.

        The base implementation knows no SDK types and returns None;
        adapters override it to read their SDK exception's response headers via retry_after_seconds_from_headers.
        """
        return None
