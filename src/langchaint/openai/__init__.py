"""Construct OpenAI and Bedrock `LLM` values and OpenAI `EmbeddingModel` values.

Importing this subpackage requires `openai`.
`OpenAI.model` uses the Responses API and reports `provider_name="openai"`.
`OpenAIBedrock.model` uses Responses and reports `provider_name="aws.bedrock"`.
Use `OpenAIChatCompletionsAdapter` directly for compatible endpoints.
Use `OpenAIResponsesAdapter` directly for Azure.

Cataloged models receive `OPENAI_PRICING`.
Uncataloged OpenAI models require `pricing` and `supports_prompt_cache_options`.
`OpenAIBedrock.model` always requires both parameters.
Missing optional rates produce NaN token costs.

Token prices use USD per one million tokens.
Web-search prices use USD per invocation.
File-search prices use USD per invocation.
Token price source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json.
Tool and regional price source: https://developers.openai.com/api/docs/pricing.
Embedding batching parameters come from the OpenAI embeddings guide.
Source: https://developers.openai.com/api/docs/guides/embeddings.
Cataloged embedding model dimensions come from the OpenAI model catalog.
Source: https://developers.openai.com/api/docs/models/all.
`OpenAI.model(pricing=...)` replaces cataloged estimates.
`OpenAIBedrock.model(pricing=...)` uses caller rates.
The gpt-5.6 family bills cache writes and accepts `prompt_cache_options`.
`PROMPT_CACHE_OPTIONS_MODELS` lists that family.
"""

from __future__ import annotations

from typing import Literal, overload

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires its dependencies; install "
        "langchaint[openai], langchaint[openai-embedding], or langchaint[openai-bedrock]."
    ) from exc

import langchaint  # noqa: TC001 (required for runtime type introspection)
from langchaint.llm import LLM
from langchaint.openai._generated_pricing import OPENAI_PRICING, OpenAIModelName
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.responses_adapter import (
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.shared import (
    OpenAILongContextPricing,
    OpenAIPricingTable,
    OpenAIRates,
    OpenAIResponsesServiceTier,
    OpenAIServiceTier,
    client_without_retries,
    parse_openai,
)
from langchaint.shared_backoff import SharedBackoff

type OpenAIEmbeddingModelName = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
"""Cataloged OpenAI model identifiers accepted by `embedding_model()`."""

OPENAI_EMBEDDING_MODELS: frozenset[OpenAIEmbeddingModelName] = frozenset({
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
})
"""Cataloged OpenAI embedding model identifiers."""

_PRICING_BY_MODEL_ID = dict[str, OpenAIPricingTable](OPENAI_PRICING.items())
"""`OPENAI_PRICING` with `str` keys for runtime model lookup."""

PROMPT_CACHE_OPTIONS_MODELS: frozenset[OpenAIModelName] = frozenset({
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.6",
})
"""Cataloged models accepting `prompt_cache_options`.

OpenAI 2.45.0 documents this parameter for gpt-5.6-and-later.
It carries `automatic_cache_breakpoints=False` to the request.
`OpenAI.model` derives `supports_prompt_cache_options` from this set.
The set stays independent from pricing because parameter availability can change independently.
"""


class OpenAI:
    """Create `LLM` and `EmbeddingModel` values for OpenAI."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        max_concurrent_requests: int | None = 8,
        max_request_starts_per_second: float = 50.0,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
    ) -> None:
        """Build `OpenAI` without sending a request.

        `client=None` constructs `AsyncOpenAI()`.
        A passed `client` must reach OpenAI.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the initial and minimum wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            openai.OpenAIError: `client` is absent and OpenAI credentials are unavailable.
            ValueError: A `SharedBackoff` setting is invalid.
        """
        self._shared_backoff = SharedBackoff(
            parse=parse_openai,
            failure_types=OpenAIResponsesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client: AsyncOpenAI = (
            client_without_retries(client) if client is not None else AsyncOpenAI(max_retries=0)
        )

    @overload
    def model(
        self,
        model: OpenAIModelName,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable | None = ...,
        supports_prompt_cache_options: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def model(
        self,
        model: str,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
    ) -> LLM: ...

    def model(
        self,
        model: str,
        *,
        regional_processing: bool = False,
        pricing: OpenAIPricingTable | None = None,
        supports_prompt_cache_options: bool | None = None,
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIResponsesServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Responses API model.

        `model` is sent verbatim.
        Cataloged models receive `OPENAI_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `regional_processing=False` uses the standard `1.0` token-price multiplier.
        Set it to `True` when the configured endpoint uses regional processing.
        `supports_prompt_cache_options` states whether the model accepts that request parameter.
        Cataloged models derive that value from `PROMPT_CACHE_OPTIONS_MODELS`.
        `reasoning_summary` requests readable reasoning summary text.
        `service_tier` sets the requested OpenAI service tier.
        The reported service tier selects pricing.

        Raises:
            ValueError: An uncataloged model lacks required pricing or caching data.
                Also raised when the SDK client contradicts the OpenAI provider.
        """
        catalog_table = _PRICING_BY_MODEL_ID.get(model)
        if catalog_table is None:
            if pricing is None:
                raise ValueError(
                    f"model {model!r} is not in OPENAI_PRICING; pass pricing= stating its rates"
                )
            if supports_prompt_cache_options is None:
                raise ValueError(
                    f"model {model!r} is not cataloged, so langchaint cannot know whether it takes "
                    "prompt_cache_options; pass supports_prompt_cache_options= stating that"
                )
        else:
            pricing = pricing or catalog_table
            if supports_prompt_cache_options is None:
                supports_prompt_cache_options = model in PROMPT_CACHE_OPTIONS_MODELS
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="openai",
            regional_processing=regional_processing,
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
            service_tier=service_tier,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-3-small"],
        *,
        dimension: int = 1536,
        max_attempts: int = 3,
    ) -> langchaint.EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-3-large"],
        *,
        dimension: int = 3072,
        max_attempts: int = 3,
    ) -> langchaint.EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-ada-002"],
        *,
        max_attempts: int = 3,
    ) -> langchaint.EmbeddingModel: ...

    def embedding_model(
        self,
        model: OpenAIEmbeddingModelName,
        *,
        dimension: int | None = None,
        max_attempts: int = 3,
    ) -> langchaint.EmbeddingModel:
        """Build an `EmbeddingModel` for one cataloged OpenAI model.

        Third-generation models accept their documented dimension range.
        `text-embedding-ada-002` has dimension 1536 and accepts no dimension argument.
        Every model counts batching tokens with tiktoken's `cl100k_base` encoding.

        Raises:
            ValueError: `model`, `dimension`, or `max_attempts` is invalid.
            ModuleNotFoundError: Either `numpy` or `tiktoken` is unavailable.
        """
        if model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(f"model {model!r} is not in OPENAI_EMBEDDING_MODELS")
        if model == "text-embedding-ada-002":
            if dimension is not None:
                raise ValueError("text-embedding-ada-002 accepts no dimension argument")
            validated_dimension = 1536
        else:
            maximum_dimension = 1536 if model == "text-embedding-3-small" else 3072
            validated_dimension = maximum_dimension if dimension is None else dimension
            if isinstance(validated_dimension, bool) or not (
                1 <= validated_dimension <= maximum_dimension
            ):
                raise ValueError(
                    f"dimension for {model!r} must be an int from 1 through "
                    f"{maximum_dimension}, got {validated_dimension!r}"
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
            dimension=validated_dimension,
        )
        return EmbeddingModel(
            adapter=adapter,
            shared_backoff=self._shared_backoff,
            max_attempts=max_attempts,
        )


class OpenAIBedrock:
    """Create `LLM` values for OpenAI models on Bedrock."""

    def __init__(  # noqa: PLR0913 (each SharedBackoff parameter remains explicit)
        self,
        *,
        aws_region: str | None = None,
        client: AsyncBedrockOpenAI | None = None,
        max_concurrent_requests: int | None = 8,
        max_request_starts_per_second: float = 50.0,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
    ) -> None:
        """Build `OpenAIBedrock` without sending a request.

        `aws_region` selects the region for an SDK client created by `OpenAIBedrock`.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the initial and minimum wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            ValueError: `client` and `aws_region` are both provided.
                Also raised when a `SharedBackoff` setting is invalid.
            openai.OpenAIError: No Bedrock region is available.
        """
        if client is not None and aws_region is not None:
            raise ValueError("Pass at most one of client= or aws_region=")
        self._shared_backoff = SharedBackoff(
            parse=parse_openai,
            failure_types=OpenAIResponsesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client: AsyncBedrockOpenAI = (
            client_without_retries(client)
            if client is not None
            else AsyncBedrockOpenAI(aws_region=aws_region, max_retries=0)
        )

    def model(
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
    ) -> LLM:
        """Build an `LLM` for one OpenAI model served by Bedrock.

        `model` is sent verbatim.
        Bedrock model identifiers have no carried pricing catalog.
        `pricing` states this Bedrock model's rates.
        `supports_prompt_cache_options` states whether the model accepts that request parameter.
        `reasoning_summary` requests readable reasoning summary text.
        Bedrock models accept no OpenAI `service_tier` parameter here.

        """
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="aws.bedrock",
            regional_processing=False,
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)


__all__ = [
    "OPENAI_EMBEDDING_MODELS",
    "OPENAI_PRICING",
    "PROMPT_CACHE_OPTIONS_MODELS",
    "OpenAI",
    "OpenAIBedrock",
    "OpenAIChatCompletionsAdapter",
    "OpenAIEmbeddingModelName",
    "OpenAILongContextPricing",
    "OpenAIModelName",
    "OpenAIPricingTable",
    "OpenAIRates",
    "OpenAIResponsesAdapter",
    "OpenAIResponsesServiceTier",
    "OpenAIServiceTier",
    "ReasoningSummary",
    "parse_openai",
]
