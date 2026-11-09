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
Per-model `thinking` and effort values: https://platform.claude.com/docs/en/build-with-claude/thinking.

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
        "langchaint's anthropic backend requires its dependencies; install langchaint[anthropic] or langchaint[anthropic-bedrock]."
    ) from exc

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from langchaint.adapter import ToolChoice
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
)
from langchaint.common.bedrock import BedrockCatalogResolution, resolve_bedrock_catalog_id
from langchaint.common.messages import TextPart
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.llm import LLM, llm_from_adapter
from langchaint.generation.observer import Observer
from langchaint.tools import ToolManager, ToolSequence

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
        self.client: AsyncAnthropic = client if client is not None else AsyncAnthropic()

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: AnthropicModelId,
        *,
        pricing: AnthropicPricingTable | None = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: AnthropicModelId,
        *,
        pricing: AnthropicPricingTable | None = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: AnthropicModelId,
        *,
        pricing: AnthropicPricingTable | None = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: AnthropicModelId,
        *,
        pricing: AnthropicPricingTable | None = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[str, None]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[str, None]: ...

    def llm(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
        inference_geo: str | None = None,
        system_prompt: str | Sequence[TextPart] | None = None,
        tools: ToolManager | ToolSequence | None = None,
        provider_executed_tools: Sequence[Mapping[str, object]] = (),
        response_format: type[str] | type[BaseModel] = str,
        max_completion_tokens: int,
        reasoning_level: str | None = None,
        temperature: float | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_requests: int = 3,
        automatic_cache_breakpoints: bool | None = None,
    ) -> LLM[str | BaseModel, ToolManager | None]:
        """Build an `LLM` for one Messages API model.

        See `LLM.bind` for common settings and validation.

        `model` is sent verbatim.
        Cataloged models receive `ANTHROPIC_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `cache_ttl` applies to every automatic cache breakpoint and every part with `cache_breakpoint=True`.
        `service_tier` sets the requested Anthropic service tier.
        `inference_geo` requests the inference geography.
        The reported service tier selects pricing.

        `llm` and `bind` raise `ValueError` when:
        - The binding places more than four cache breakpoints.
          System parts with `cache_breakpoint=True` and automatic cache breakpoints count.
          Provider-executed tools with `cache_control` also count.
        - A provider-executed tool's `type` is missing or unsupported.
          The adapter supports specific versions of web search, web fetch, tool search, and code execution.
        - Code execution is bound without a web search or web fetch version from 20260209 or later.
        - Web search is bound and `pricing.web_search_usd_per_invocation` is `None`.
        `llm` and `bind` raise `TypeError` when `tool_choice` is `AllowedToolsChoice`.

        `reasoning_level` sets `output_config.effort`.
        It enables adaptive `thinking` unless `extra_body` sets `thinking`.
        `max_completion_tokens` is required because the Messages API requires `max_tokens`.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
            ValueError: `tools` contains duplicate names.
            ValueError: `AllowedToolsChoice.tool_names` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence or `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one, or a provider constraint listed above is violated.
            TypeError: `response_format` is a proper subclass of `str`, or the adapter does not support `tool_choice`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's argument model has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's argument model is not fully defined.
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
        return llm_from_adapter(
            adapter,
            shared_backoff=self._shared_backoff,
            observer=self._observer,
            system_prompt=system_prompt,
            tools=tools,
            provider_executed_tools=provider_executed_tools,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            reasoning_level=reasoning_level,
            temperature=temperature,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_body=extra_body,
            max_requests=max_requests,
            automatic_cache_breakpoints=automatic_cache_breakpoints,
        )


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
                f"api={api!r} requires a {client_class.__name__} client, but a {type(client).__name__} was passed."
            )
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.apply_cross_region_multiplier: bool = apply_cross_region_multiplier
        self.client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle = (
            client
            if client is not None
            else client_class(aws_region=aws_region, http_client=http_client)
        )

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: AnthropicBedrockModelId,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: AnthropicBedrockModelId,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: AnthropicBedrockModelId,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: AnthropicBedrockModelId,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> LLM[str, None]: ...

    def llm(
        self,
        model: AnthropicBedrockModelId,
        *,
        pricing: AnthropicPricingTable | None = None,
        cache_ttl: CacheTTL = "5m",
        system_prompt: str | Sequence[TextPart] | None = None,
        tools: ToolManager | ToolSequence | None = None,
        provider_executed_tools: Sequence[Mapping[str, object]] = (),
        response_format: type[str] | type[BaseModel] = str,
        max_completion_tokens: int,
        reasoning_level: str | None = None,
        temperature: float | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_requests: int = 3,
        automatic_cache_breakpoints: bool | None = None,
    ) -> LLM[str | BaseModel, ToolManager | None]:
        """Build an `LLM` for one Bedrock model.

        See `LLM.bind` for common settings and validation.

        `model` is sent verbatim.
        An exact catalog identifier selects pricing.
        Otherwise a `BEDROCK_CROSS_REGION_MULTIPLIER` prefix is stripped before the catalog lookups.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `cache_ttl` applies to every automatic cache breakpoint and every part with `cache_breakpoint=True`.
        Bedrock models accept no Anthropic `service_tier` parameter here.

        `llm` and `bind` raise `ValueError` when:
        - The binding places more than four cache breakpoints.
          System parts with `cache_breakpoint=True` and automatic cache breakpoints count.
        - `provider_executed_tools` is nonempty.
        `llm` and `bind` raise `TypeError` when `tool_choice` is `AllowedToolsChoice`.

        `reasoning_level` sets `output_config.effort`.
        It enables adaptive `thinking` unless `extra_body` sets `thinking`.
        `max_completion_tokens` is required because the Messages API requires `max_tokens`.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
            ValueError: `tools` contains duplicate names.
            ValueError: `AllowedToolsChoice.tool_names` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence or `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one, or a provider constraint listed above is violated.
            TypeError: `response_format` is a proper subclass of `str`, or the adapter does not support `tool_choice`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's argument model has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's argument model is not fully defined.
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
        return llm_from_adapter(
            adapter,
            shared_backoff=self._shared_backoff,
            observer=self._observer,
            system_prompt=system_prompt,
            tools=tools,
            provider_executed_tools=provider_executed_tools,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            reasoning_level=reasoning_level,
            temperature=temperature,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_body=extra_body,
            max_requests=max_requests,
            automatic_cache_breakpoints=automatic_cache_breakpoints,
        )

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
