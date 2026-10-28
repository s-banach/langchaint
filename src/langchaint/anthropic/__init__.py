"""Construct Anthropic and Bedrock `LLM` values with cataloged pricing.

Importing this subpackage requires `anthropic`.
`Anthropic.llm` sends the stated model identifier verbatim.
`AnthropicBedrock.llm` sends the stated Bedrock identifier verbatim.
`AnthropicBedrock` selects its SDK client with `api` and defaults to the legacy API in `us-east-1`.

Cataloged Anthropic models receive `ANTHROPIC_PRICING`.
Cataloged Bedrock models receive `ANTHROPIC_BEDROCK_PRICING`.
A Bedrock identifier with a `BEDROCK_CROSS_REGION_MULTIPLIER` prefix resolves through the unprefixed identifier.
`AnthropicBedrock` multiplies those token rates by the prefix multiplier unless `apply_cross_region_multiplier=False`.
Uncataloged Anthropic models require `pricing`.
Uncataloged Bedrock models require `pricing`.
Missing optional rates produce NaN token costs.
Pass `client=AsyncAnthropic(http_client=...)` for custom first-party transports.
`AnthropicBedrock` accepts `http_client` directly.

Token prices use USD per one million tokens.
Web-search prices use USD per invocation.
Token, Batch API, and web-search price source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json.
`inference_geo` price source: https://platform.claude.com/docs/en/manage-claude/data-residency.
Cache reads cost 0.05 times base input on Claude Opus 5.5 and Claude Sonnet 5.5, and 0.025 times on Claude Fable 5.1.
Cache reads cost 0.1 times base input on other models.
Five-minute cache writes cost 1.25 times base input.
One-hour cache writes cost twice base input.
Claude Haiku 5.5 multiplies every token rate by five above 100,000 input tokens, counting cache reads and writes.
`ANTHROPIC_PRICING` web-search rates are public list-price estimates.
A cataloged model without a web-search rate requires `pricing` to bind the web search tool.
`Anthropic.llm(pricing=...)` replaces cataloged estimates.
`AnthropicBedrock.llm(pricing=...)` accepts caller rates.
"""

from typing import Literal, overload

try:
    import httpx2
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
except ModuleNotFoundError as exc:
    if exc.name not in ("anthropic", "httpx2"):
        raise
    raise ModuleNotFoundError(
        "langchaint's anthropic backend requires its dependencies; install "
        "langchaint[anthropic] or langchaint[anthropic-bedrock]."
    ) from exc

from langchaint.anthropic._generated_pricing import (
    ANTHROPIC_BEDROCK_PRICING,
    ANTHROPIC_PRICING,
    BEDROCK_CROSS_REGION_MULTIPLIER,
    AnthropicModelId,
)
from langchaint.anthropic.messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicPricingTable,
    AnthropicRates,
    AnthropicServiceTier,
    CacheTTL,
    client_without_retries,
)
from langchaint.common.bedrock import BedrockCatalogResolution, resolve_bedrock_catalog_id
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.llm import LLM
from langchaint.generation.observer import Observer

_PRICING_BY_MODEL_ID = dict[str, AnthropicPricingTable](ANTHROPIC_PRICING.items())
"""`ANTHROPIC_PRICING` with `str` keys for runtime model lookup."""


type AnthropicBedrockModelId = (
    Literal[
        "anthropic.claude-fable-5-1",
        "anthropic.claude-fable-5",
        "anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-sonnet-5-5",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-haiku-5-5",
        "anthropic.claude-haiku-4-5",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-sonnet-4-6",
    ]
    | str
)
"""Bedrock identifiers accepted by `AnthropicBedrock.llm`.

Each identifier is sent verbatim.
The literal values offer preferred endpoint-specific identifiers.
A `BEDROCK_CROSS_REGION_MULTIPLIER` prefix on a literal value resolves the same catalog pricing.
"""


class Anthropic:
    """Create `LLM` values for Anthropic."""

    def __init__(
        self,
        *,
        client: AsyncAnthropic | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `Anthropic` without sending a request.

        `client=None` constructs `AsyncAnthropic()`.
        A passed `client` must reach Anthropic.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.
        """
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: AsyncAnthropic = (
            client_without_retries(client) if client is not None else AsyncAnthropic(max_retries=0)
        )

    @overload
    def llm(
        self,
        model: AnthropicModelId,
        *,
        pricing: AnthropicPricingTable | None = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
    ) -> LLM: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
    ) -> LLM: ...

    def llm(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
        inference_geo: str | None = None,
    ) -> LLM:
        """Build an `LLM` for one Messages API model.

        `model` is sent verbatim.
        Cataloged models receive `ANTHROPIC_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `cache_ttl` applies to every automatic cache breakpoint and every part with `cache_breakpoint=True`.
        `service_tier` sets the requested Anthropic service tier.
        `inference_geo` requests the inference geography.
        The reported service tier selects pricing.

        `bind` on the returned `LLM`, and on every `BoundLLM` derived from it, raises `ValueError` when:
        - `max_completion_tokens` is `None`, because the Messages API requires `max_tokens`.
        - The binding places more than four cache breakpoints.
          System parts with `cache_breakpoint=True` and automatic cache breakpoints count.
          Provider-executed tools with `cache_control` also count.
        - A provider-executed tool's `type` is missing or unsupported.
          The adapter supports specific versions of web search, web fetch, tool search, and code execution.
        - Code execution is bound without a web search or web fetch version from 20260209 or later.
        - Web search is bound and `pricing.web_search_usd_per_invocation` is `None`.
        `bind` raises `TypeError` when `tool_choice` is `AllowedToolsChoice`.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
        """
        if pricing is None:
            pricing = _PRICING_BY_MODEL_ID.get(model)
        if pricing is None:
            raise ValueError(
                f"model {model!r} is not in ANTHROPIC_PRICING; pass pricing= stating its rates"
            )
        adapter = AnthropicMessagesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="anthropic",
            cache_ttl=cache_ttl,
            service_tier=service_tier,
            inference_geo=inference_geo,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff, observer=self._observer)


class AnthropicBedrock:
    """Create `LLM` values for Anthropic models on Bedrock."""

    def __init__(
        self,
        *,
        api: Literal["legacy", "mantle"] = "legacy",
        aws_region: str = "us-east-1",
        client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle | None = None,
        http_client: httpx2.AsyncClient | None = None,
        apply_cross_region_multiplier: bool = True,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `AnthropicBedrock` without sending a request.

        `api="legacy"` selects `AsyncAnthropicBedrock`; `api="mantle"` selects `AsyncAnthropicBedrockMantle`.
        A passed `client` must match `api` and retains its region and transport.
        `aws_region` selects the region for SDK clients created by `AnthropicBedrock`, defaulting to `us-east-1`.
        `http_client` applies to SDK clients created by `AnthropicBedrock`.
        `apply_cross_region_multiplier=False` leaves a prefixed identifier's catalog rates unmultiplied.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` contradicts `api` or accompanies `http_client`.
        """
        if client is not None and http_client is not None:
            raise ValueError("Pass at most one of client= or http_client=")
        client_class = AsyncAnthropicBedrock if api == "legacy" else AsyncAnthropicBedrockMantle
        if client is not None and not isinstance(client, client_class):
            raise ValueError(
                f"api={api!r} requires a {client_class.__name__} client, "
                f"but a {type(client).__name__} was passed."
            )
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.apply_cross_region_multiplier: bool = apply_cross_region_multiplier
        self.client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle = (
            client_without_retries(client)
            if client is not None
            else client_class(aws_region=aws_region, http_client=http_client, max_retries=0)
        )

    def llm(
        self,
        model: AnthropicBedrockModelId,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
    ) -> LLM:
        """Build an `LLM` for one Bedrock model.

        `model` is sent verbatim.
        An exact catalog identifier selects pricing.
        Otherwise a `BEDROCK_CROSS_REGION_MULTIPLIER` prefix is stripped before the catalog lookups.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `cache_ttl` applies to every automatic cache breakpoint and every part with `cache_breakpoint=True`.
        Bedrock models accept no Anthropic `service_tier` parameter here.

        `bind` on the returned `LLM`, and on every `BoundLLM` derived from it, raises `ValueError` when:
        - `max_completion_tokens` is `None`, because the Messages API requires `max_tokens`.
        - The binding places more than four cache breakpoints.
          System parts with `cache_breakpoint=True` and automatic cache breakpoints count.
        - `provider_executed_tools` is nonempty.
        `bind` raises `TypeError` when `tool_choice` is `AllowedToolsChoice`.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
        """
        resolution = resolve_bedrock_catalog_id(
            model,
            catalog_ids=ANTHROPIC_BEDROCK_PRICING,
            cross_region_multipliers=BEDROCK_CROSS_REGION_MULTIPLIER,
        )
        adapter = AnthropicMessagesAdapter(
            client=self.client,
            model=model,
            pricing=pricing if pricing is not None else self._catalog_pricing(model, resolution),
            provider_name="aws.bedrock",
            cache_ttl=cache_ttl,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff, observer=self._observer)

    def _catalog_pricing(
        self, model: str, resolution: BedrockCatalogResolution
    ) -> AnthropicPricingTable:
        """Return the catalog table for `resolution`.

        `apply_cross_region_multiplier` multiplies the table by the prefix multiplier of `resolution`.

        Raises:
            ValueError: The catalog identifier is not in `ANTHROPIC_BEDROCK_PRICING`.
        """
        catalog_table = ANTHROPIC_BEDROCK_PRICING.get(resolution.catalog_id)
        if catalog_table is None:
            raise ValueError(
                f"model {model!r} is not in ANTHROPIC_BEDROCK_PRICING; pass pricing= stating its rates"
            )
        if resolution.cross_region_multiplier is None or not self.apply_cross_region_multiplier:
            return catalog_table
        return catalog_table.scaled(resolution.cross_region_multiplier)


__all__ = [
    "ANTHROPIC_BEDROCK_PRICING",
    "ANTHROPIC_PRICING",
    "BEDROCK_CROSS_REGION_MULTIPLIER",
    "Anthropic",
    "AnthropicBedrock",
    "AnthropicBedrockModelId",
    "AnthropicMessagesAdapter",
    "AnthropicModelId",
    "AnthropicPricingTable",
    "AnthropicRates",
    "AnthropicServiceTier",
    "CacheTTL",
]
