"""Construct OpenAI and Bedrock `LLM` values and OpenAI `EmbeddingModel` values.

Importing this subpackage requires `openai`.
`OpenAI.llm` uses the Responses API and reports `provider_name="openai"`.
`OpenAIBedrock.llm` uses Responses and reports `provider_name="aws.bedrock"`.
Use `OpenAIChatCompletionsAdapter` directly for compatible endpoints.
Use `OpenAIResponsesAdapter` directly for Azure.

Cataloged models receive `OPENAI_PRICING`.
Cataloged embedding models receive `OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS`.
Uncataloged OpenAI models require `pricing` and `supports_cache_breakpoints`.
`OpenAIBedrock.llm` always requires both parameters.
Missing optional rates produce NaN token costs.

Token prices use USD per one million tokens.
Web-search prices use USD per invocation.
File-search prices use USD per invocation.
Token, web-search, and regional-processing price source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json.
File-search price source: https://developers.openai.com/api/docs/pricing.
Embedding batching parameters come from the OpenAI embeddings guide.
Source: https://developers.openai.com/api/docs/guides/embeddings.
Cataloged embedding model dimensions come from the OpenAI model catalog.
Source: https://developers.openai.com/api/docs/models/all.
`OpenAI.llm(pricing=...)` replaces cataloged estimates.
`OpenAIBedrock.llm(pricing=...)` uses caller rates.
`CACHE_BREAKPOINT_MODELS` lists models supporting explicit cache breakpoints independently of pricing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires its dependencies; install "
        "langchaint[openai], langchaint[openai-embedding], or langchaint[openai-bedrock]."
    ) from exc

from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.llm import LLM
from langchaint.openai._generated_pricing import (
    OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS,
    OPENAI_PRICING,
    OPENAI_REGIONAL_PROCESSING_MULTIPLIER,
    OpenAIEmbeddingModelId,
    OpenAIModelId,
)
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.responses_adapter import (
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.shared import (
    OpenAIPricingTable,
    OpenAIResponsesServiceTier,
    OpenAIServiceTier,
    client_without_retries,
)

if TYPE_CHECKING:
    from langchaint.embedding import EmbeddingModel
    from langchaint.generation.observer import Observer

OPENAI_EMBEDDING_MODELS: frozenset[OpenAIEmbeddingModelId] = frozenset(
    OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS
)
"""Cataloged OpenAI embedding model identifiers."""

_PRICING_BY_MODEL_ID = dict[str, OpenAIPricingTable](OPENAI_PRICING.items())
"""`OPENAI_PRICING` with `str` keys for runtime model lookup."""

_REGIONAL_PROCESSING_MULTIPLIER_BY_MODEL_ID = dict[str, float](
    OPENAI_REGIONAL_PROCESSING_MULTIPLIER.items()
)
"""`OPENAI_REGIONAL_PROCESSING_MULTIPLIER` with `str` keys for runtime model lookup."""

CACHE_BREAKPOINT_MODELS: frozenset[OpenAIModelId] = frozenset({
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.6",
})
"""Cataloged models supporting explicit cache breakpoints.

`OpenAI.llm` derives `supports_cache_breakpoints` from this set.
The set stays independent from pricing because parameter availability can change independently.
"""


class OpenAI:
    """Create `LLM` and `EmbeddingModel` values for OpenAI."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `OpenAI` without sending a request.

        `client=None` constructs `AsyncOpenAI()`.
        A passed `client` must reach OpenAI.
        `shared_backoff` admits every request of the created `LLM` and `EmbeddingModel` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer` follows every request of the created `EmbeddingModel` values.
        `observer=None` follows none.

        Raises:
            openai.OpenAIError: `client` is absent and OpenAI credentials are unavailable.
        """
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: AsyncOpenAI = (
            client_without_retries(client) if client is not None else AsyncOpenAI(max_retries=0)
        )

    @overload
    def llm(
        self,
        model: OpenAIModelId,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable | None = ...,
        supports_cache_breakpoints: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable,
        supports_cache_breakpoints: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
    ) -> LLM: ...

    def llm(
        self,
        model: str,
        *,
        regional_processing: bool = False,
        pricing: OpenAIPricingTable | None = None,
        supports_cache_breakpoints: bool | None = None,
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIResponsesServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Responses API model.

        `model` is sent verbatim.
        Cataloged models receive `OPENAI_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        Set `regional_processing=True` when the configured endpoint uses regional processing.
        It multiplies every token rate by the model's `OPENAI_REGIONAL_PROCESSING_MULTIPLIER`.
        An uncataloged model has no multiplier, so every nonzero token category then costs NaN.
        State its regional rates in `pricing` and leave `regional_processing=False`.
        `supports_cache_breakpoints` states whether the provider API supports explicit cache breakpoints.
        False omits explicit markers and cache-breakpoint controls while preserving content.
        Cataloged models derive that value from `CACHE_BREAKPOINT_MODELS`.
        `reasoning_summary` requests readable reasoning summary text.
        `service_tier` sets the requested OpenAI service tier.
        The reported service tier selects pricing.

        `bind` on the returned `LLM`, and on every `BoundLLM` derived from it, raises `ValueError` when:
        - `automatic_cache_breakpoints=False` and `supports_cache_breakpoints` is false.
        - A provider-executed tool's `type` is missing or unsupported.
          The adapter supports specific versions of web search and file search.
        - Web search is bound and `pricing.web_search_usd_per_invocation` is `None`.
        - File search is bound and `pricing.file_search_usd_per_invocation` is `None`.

        Raises:
            ValueError: An uncataloged model lacks required pricing or caching data.
                Also raised when the SDK client contradicts the OpenAI provider.
        """
        catalog_table = _PRICING_BY_MODEL_ID.get(model)
        if catalog_table is None:
            if pricing is None or supports_cache_breakpoints is None:
                missing_arguments: list[str] = []
                if pricing is None:
                    missing_arguments.append("pricing= stating its rates")
                if supports_cache_breakpoints is None:
                    missing_arguments.append(
                        "supports_cache_breakpoints= stating whether it supports explicit cache breakpoints"
                    )
                raise ValueError(
                    f"model {model!r} is not in OPENAI_PRICING; pass "
                    + " and ".join(missing_arguments)
                )
        else:
            pricing = pricing or catalog_table
            if supports_cache_breakpoints is None:
                supports_cache_breakpoints = model in CACHE_BREAKPOINT_MODELS
        if regional_processing:
            pricing = pricing.scaled(
                _REGIONAL_PROCESSING_MULTIPLIER_BY_MODEL_ID.get(model, float("nan"))
            )
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="openai",
            supports_cache_breakpoints=supports_cache_breakpoints,
            reasoning_summary=reasoning_summary,
            service_tier=service_tier,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff, observer=self._observer)

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-3-small"],
        *,
        dimension: int = 1536,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-3-large"],
        *,
        dimension: int = 3072,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-ada-002"],
        *,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    def embedding_model(
        self,
        model: OpenAIEmbeddingModelId,
        *,
        dimension: int | None = None,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel:
        """Build an `EmbeddingModel` for one cataloged OpenAI model.

        Third-generation models accept their documented dimension range.
        `text-embedding-ada-002` has dimension 1536 and accepts no dimension argument.
        Every model counts batching tokens with tiktoken's `cl100k_base` encoding.
        `usd_per_million_tokens` prices input tokens.
        It replaces the model's rate in `OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS`.

        Raises:
            ValueError: `model`, `dimension`, or `max_requests` is invalid.
            ModuleNotFoundError: Either `numpy` or `tiktoken` is unavailable.
        """
        if model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(f"model {model!r} is not in OPENAI_EMBEDDING_MODELS")
        if model == "text-embedding-ada-002":
            if dimension is not None:
                raise ValueError("text-embedding-ada-002 accepts no dimension argument")
            dimension = 1536
        else:
            max_dimension = 1536 if model == "text-embedding-3-small" else 3072
            dimension = max_dimension if dimension is None else dimension
            if isinstance(dimension, bool) or not (1 <= dimension <= max_dimension):
                raise ValueError(
                    f"dimension for {model!r} must be an int from 1 through "
                    f"{max_dimension}, got {dimension!r}"
                )
        try:
            from langchaint.embedding import EmbeddingModel  # noqa: PLC0415 (defer numpy)
            from langchaint.openai.embedding_adapter import (  # noqa: PLC0415 (keep tiktoken outside ordinary imports)
                _OpenAIEmbeddingAdapter,
            )
        except ModuleNotFoundError as exc:
            cause = exc.__cause__
            if exc.name not in ("numpy", "tiktoken") and (
                not isinstance(cause, ModuleNotFoundError) or cause.name != "numpy"
            ):
                raise
            raise ModuleNotFoundError(
                "OpenAI embeddings require numpy and tiktoken; install "
                "langchaint[openai-embedding]."
            ) from exc
        adapter = _OpenAIEmbeddingAdapter(
            client=self.client,
            model=model,
            dimension=dimension,
        )
        return EmbeddingModel(
            adapter=adapter,
            shared_backoff=self._shared_backoff,
            max_requests=max_requests,
            usd_per_million_tokens=(
                OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS[model]
                if usd_per_million_tokens is None
                else usd_per_million_tokens
            ),
            observer=self._observer,
        )


class OpenAIBedrock:
    """Create `LLM` values for OpenAI models on Bedrock."""

    def __init__(
        self,
        *,
        aws_region: str | None = None,
        client: AsyncBedrockOpenAI | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `OpenAIBedrock` without sending a request.

        `aws_region` selects the region for an SDK client created by `OpenAIBedrock`.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` and `aws_region` are both provided.
            openai.OpenAIError: No Bedrock region is available.
        """
        if client is not None and aws_region is not None:
            raise ValueError("Pass at most one of client= or aws_region=")
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: AsyncBedrockOpenAI = (
            client_without_retries(client)
            if client is not None
            else AsyncBedrockOpenAI(aws_region=aws_region, max_retries=0)
        )

    def llm(
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_cache_breakpoints: bool,
        reasoning_summary: ReasoningSummary | None = None,
    ) -> LLM:
        """Build an `LLM` for one OpenAI model served by Bedrock.

        `model` is sent verbatim.
        Bedrock model identifiers have no carried pricing catalog.
        `pricing` states this Bedrock model's rates.
        `supports_cache_breakpoints` states whether the provider API supports explicit cache breakpoints.
        False omits explicit markers and cache-breakpoint controls while preserving content.
        `reasoning_summary` requests readable reasoning summary text.
        Bedrock models accept no OpenAI `service_tier` parameter here.

        `bind` on the returned `LLM`, and on every `BoundLLM` derived from it, raises `ValueError` when:
        - `automatic_cache_breakpoints=False` and `supports_cache_breakpoints=False`.
        - `provider_executed_tools` is nonempty.
        """
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="aws.bedrock",
            supports_cache_breakpoints=supports_cache_breakpoints,
            reasoning_summary=reasoning_summary,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff, observer=self._observer)


__all__ = [
    "CACHE_BREAKPOINT_MODELS",
    "OPENAI_EMBEDDING_MODELS",
    "OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS",
    "OPENAI_PRICING",
    "OPENAI_REGIONAL_PROCESSING_MULTIPLIER",
    "OpenAI",
    "OpenAIBedrock",
    "OpenAIChatCompletionsAdapter",
    "OpenAIEmbeddingModelId",
    "OpenAIModelId",
    "OpenAIPricingTable",
    "OpenAIResponsesAdapter",
    "OpenAIResponsesServiceTier",
    "OpenAIServiceTier",
    "ReasoningSummary",
]
