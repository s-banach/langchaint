"""The provider adapter contract.

An adapter wraps one official SDK client and is the only place provider knowledge lives:
converting the binding to SDK keyword arguments, sending, translating stream events, normalizing usage, computing cost,
and classifying errors.
Adapters delegate stream assembly and structured-output parsing to the SDK
(`get_final_response` / `get_final_message`), which is generic in the response format,
so the output type flows from the SDK to the caller without reconstruction.

Binding model: `Provider.bind_text` and `Provider.bind_structured` convert the frozen prefix
(system_prompt, tool_schemas, tool_choice, parallel_tool_calls, inference_params, automatic_prompt_caching)
to precomputed SDK keyword arguments once;
the returned `BoundProvider` accepts only the per-request conversation.
The split into two bind methods is what fixes the output type at bind time:
each method is monomorphic in its output type, so no sentinel value has to imply a type downstream.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from langchaint.inference_params import InferenceParams
from langchaint.messages import AssistantMessage, Message, StopReason, TextPart, ToolCall
from langchaint.tools import ToolSchema
from langchaint.usage import Usage

type ErrorClass = Literal["rate_limit", "transient", "abort"]
"""Whether a retry may fix the error, and whether it should pause everyone.

"rate_limit" is transient and account-wide: the account or service refuses further requests
right now, so RateLimiter pauses admission for everyone sharing it.
"transient" is retried by the failing task alone; "abort" is not retried and cancels the batch
(the retry loop raises AbortBatchError).
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

type StreamItem = str | ToolCall
"""What a stream yields: text chunks and completed tool calls.

Text chunks are the provider SDK's own strings, passed through without a wrapper class or copy.
Each tool call is yielded once, complete, when its block closes;
there are no tool-call delta items because a consumer cannot act on partial argument JSON,
and both SDKs accumulate the arguments and hand over the finished call.
Usage, cost, and stop reason are not streamed; they live on the Response from final().
"""


@dataclass(frozen=True, kw_only=True)
class SpecificTool:
    """Tool choice that forces the model to call the named tool."""

    tool_name: str


type ToolChoice = Literal["auto", "required", "none"] | SpecificTool
"""Provider-neutral tool choice.

"auto" lets the model decide, "required" forces some tool call (Anthropic's "any"), SpecificTool forces one named tool,
and "none" forbids tool calls.
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
    """

    input_cache_none_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_usd_per_million_tokens: float
    cache_write_1h_usd_per_million_tokens: float | None = None


@dataclass(frozen=True, kw_only=True)
class Binding:
    """The frozen prefix of one BoundLLM, in package terms only.

    Every field here determines the provider's cacheable prompt prefix or is fixed per binding by design;
    per-request data is the conversation argument of the BoundProvider methods, nothing else.
    """

    system_prompt: str | tuple[TextPart, ...] | None
    """The bound system prompt; None binds none.

    The parts form exists to carry cache_breakpoint marks inside the system prompt:
    the anthropic adapter renders one system text block per part,
    and the openai adapter sends the parts as a developer-role input message ahead of the conversation
    (the SDK documents `instructions` as "a system (or developer) message inserted into the model's context",
    and only input message parts carry prompt_cache_breakpoint).
    A plain str renders exactly as before (one anthropic system block; the openai instructions parameter).
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
    On openai, False requires a gpt-5.6 or later model:
    the SDK documents prompt_cache_options as supported only there,
    and the adapter sends it whenever False is bound, so an older model may reject the request.
    Older openai models cache automatically with free writes, so False buys nothing on them; bind True.
    """


@dataclass(frozen=True, kw_only=True)
class ProviderResult[OutputT]:
    """One successful provider turn, normalized to package terms.

    output is the assistant text (text bindings) or the SDK-parsed response_format instance (structured bindings).
    assistant_message is the full turn including tool calls, for appending to a conversation.
    usage carries the adapter-priced cost_in_usd, computed from raw provider counts against its PricingTable.
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


class ProviderStream[OutputT](ABC):
    """One open stream, backed by the SDK's stream manager.

    The adapter translates SDK events into StreamItem values as they pass through;
    assembly and structured-output parsing stay in the SDK.
    """

    @abstractmethod
    def items(self) -> AsyncIterator[StreamItem]:
        """Yield text chunks and completed tool calls in arrival order.

        Yields:
            Stream items; SDK events the package does not model are dropped.
        """
        ...

    @abstractmethod
    async def final(self) -> ProviderResult[OutputT]:
        """Return the SDK-assembled result after the stream ends.

        Callable only after items() is exhausted; the adapter delegates assembly and parsing to the SDK stream manager.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection; idempotent."""
        ...


class BoundProvider[OutputT](ABC):
    """One provider adapter bound to a frozen prefix.

    Constructed by Provider.bind_text or Provider.bind_structured, which precompute the SDK keyword arguments once;
    both request methods take only the per-request conversation.
    """

    @abstractmethod
    async def send(self, conversation: Sequence[Message]) -> ProviderResult[OutputT]:
        """Send one non-streaming request.

        Raises:
            Exception: the SDK's own exceptions propagate unchanged; Provider.classify maps them to transient or abort.
                For defects the SDK reports as data rather than as an exception,
                the adapter raises TransientError, AbortBatchError, or a GenerationError leaf directly,
                and the retry loop honors those without classification.
        """
        ...

    @abstractmethod
    async def open_stream(self, conversation: Sequence[Message]) -> ProviderStream[OutputT]:
        """Open one streaming request and return the live stream.

        Opening performs the connection I/O, so a connection failure raises here, before any event is yielded.

        Raises:
            Exception: the SDK's own exceptions propagate unchanged; Provider.classify maps them to transient or abort.
        """
        ...


class Provider(ABC):
    """Base class for one adapter per provider SDK.

    An adapter is constructed with the SDK client to use, which is how Bedrock support arrives:
    pass AsyncAnthropicBedrock or AsyncBedrockOpenAI instead of the direct client.
    Credentials and endpoints belong to the SDK client.
    """

    name: str
    """The provider identifier recorded on every Response as provider_name."""

    def __init__(self, *, model: str, pricing: PricingTable) -> None:
        """Store the model identifier and the pricing the adapter bills by."""
        self.model = model
        self.pricing = pricing

    @abstractmethod
    def bind_text(self, binding: Binding) -> BoundProvider[str]:
        """Bind for plain-text output.

        Pure conversion of the binding to SDK keyword arguments; no I/O.
        """
        ...

    @abstractmethod
    def bind_structured[ModelT: BaseModel](
        self, binding: Binding, response_format: type[ModelT]
    ) -> BoundProvider[ModelT]:
        """Bind for structured output parsed by the SDK into response_format.

        Pure conversion of the binding to SDK keyword arguments; no I/O.
        """
        ...

    @abstractmethod
    def classify(self, error: Exception) -> ErrorClass:
        """Classify an exception raised by send or open_stream.

        Anything the adapter does not recognize must map to "abort" so bugs are not retried silently.
        """
        ...

    def retry_after_seconds(self, error: Exception) -> float | None:  # noqa: ARG002
        """Return the server-stated wait carried by error, when the SDK exposes one.

        The base implementation knows no SDK types and returns None;
        adapters override it to read their SDK exception's response headers via retry_after_seconds_from_headers.
        """
        return None
