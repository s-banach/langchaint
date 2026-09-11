"""Construct Anthropic and Bedrock `LLM` values with cataloged pricing.

Importing this subpackage requires `anthropic`.
`Anthropic.model` sends the stated model identifier verbatim.
`AnthropicBedrock.model` sends the stated Bedrock identifier verbatim.
`ANTHROPIC_BEDROCK` lists preferred identifiers and their `BedrockRouting` values.

Cataloged Anthropic models receive `ANTHROPIC_PRICING`.
Cataloged Bedrock models receive `ANTHROPIC_BEDROCK_PRICING`.
A Bedrock identifier with a `BEDROCK_CROSS_REGION_MULTIPLIER` prefix resolves through the unprefixed identifier.
`AnthropicBedrock` multiplies those token rates by the prefix multiplier unless `apply_cross_region_premium=False`.
Uncataloged Anthropic models require `pricing`.
Uncataloged Bedrock models also require a passed `client`.
Missing optional rates produce NaN token costs.
Pass `client=AsyncAnthropic(http_client=...)` for custom first-party transports.
`AnthropicBedrock` accepts `http_client` directly.

Token prices use USD per one million tokens.
Web-search prices use USD per invocation.
Token price source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json.
Tool and modifier price source: https://platform.claude.com/docs/en/about-claude/pricing.
Cache reads cost 0.1 times base input.
Five-minute cache writes cost 1.25 times base input.
One-hour cache writes cost twice base input.
`ANTHROPIC_PRICING` web-search rates are public list-price estimates.
`Anthropic.model(pricing=...)` replaces cataloged estimates.
`AnthropicBedrock.model(pricing=...)` accepts caller rates.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple, overload

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
    AnthropicModelName,
)
from langchaint.anthropic.messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicPricingTable,
    AnthropicRates,
    AnthropicServiceTier,
    CacheTTL,
    client_without_retries,
    parse_anthropic,
)
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.llm import LLM

_PRICING_BY_MODEL_ID = dict[str, AnthropicPricingTable](ANTHROPIC_PRICING.items())
"""`ANTHROPIC_PRICING` with `str` keys for runtime model lookup."""


type AnthropicBedrockModelName = (
    Literal[
        "anthropic.claude-fable-5",
        "anthropic.claude-opus-5",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-haiku-4-5",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-sonnet-4-6",
    ]
    | str
)
"""Bedrock identifiers accepted by `AnthropicBedrock.model`.

Each identifier is sent verbatim.
The literal values offer preferred endpoint-specific identifiers.
A `BEDROCK_CROSS_REGION_MULTIPLIER` prefix on a literal value resolves the same routing and catalog pricing.
"""


class _CatalogResolution(NamedTuple):
    """The catalog identifier a Bedrock identifier resolves through."""

    catalog_id: str
    cross_region_multiplier: float | None
    """The `BEDROCK_CROSS_REGION_MULTIPLIER` entry of a stripped prefix, or `None` when no prefix was stripped."""


def _resolve_catalog_id(model: str) -> _CatalogResolution:
    """Resolve `model` to its catalog identifier.

    An exact catalog identifier resolves to itself.
    Otherwise a `BEDROCK_CROSS_REGION_MULTIPLIER` prefix is stripped.
    """
    if model in ANTHROPIC_BEDROCK_PRICING:
        return _CatalogResolution(catalog_id=model, cross_region_multiplier=None)
    prefix, _, unprefixed = model.partition(".")
    multiplier = BEDROCK_CROSS_REGION_MULTIPLIER.get(prefix)
    if multiplier is None:
        return _CatalogResolution(catalog_id=model, cross_region_multiplier=None)
    return _CatalogResolution(catalog_id=unprefixed, cross_region_multiplier=multiplier)


@dataclass(frozen=True, kw_only=True)
class BedrockRouting:
    """The SDK client API for one Bedrock identifier."""

    api: Literal["mantle", "legacy"]


ANTHROPIC_BEDROCK: dict[AnthropicBedrockModelName, BedrockRouting] = {
    "anthropic.claude-fable-5": BedrockRouting(api="mantle"),
    "anthropic.claude-opus-5": BedrockRouting(api="mantle"),
    "anthropic.claude-opus-4-8": BedrockRouting(api="mantle"),
    "anthropic.claude-opus-4-7": BedrockRouting(api="mantle"),
    "anthropic.claude-sonnet-5": BedrockRouting(api="mantle"),
    "anthropic.claude-haiku-4-5": BedrockRouting(api="mantle"),
    "us.anthropic.claude-opus-4-6-v1": BedrockRouting(api="legacy"),
    "us.anthropic.claude-sonnet-4-6": BedrockRouting(api="legacy"),
}
"""`BedrockRouting` for each preferred `AnthropicBedrockModelName` value."""

_BEDROCK_CLIENT_CLASS: dict[
    Literal["mantle", "legacy"], type[AsyncAnthropicBedrockMantle | AsyncAnthropicBedrock]
] = {
    "mantle": AsyncAnthropicBedrockMantle,
    "legacy": AsyncAnthropicBedrock,
}


class Anthropic:
    """Create `LLM` values for Anthropic."""

    def __init__(
        self,
        *,
        client: AsyncAnthropic | None = None,
        max_concurrent_requests: int | None = 8,
        max_request_starts_per_second: float = 50.0,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
    ) -> None:
        """Build `Anthropic` without sending a request.

        `client=None` constructs `AsyncAnthropic()`.
        A passed `client` must reach Anthropic.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the initial and minimum wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            ValueError: A `SharedBackoff` setting is invalid.
        """
        self._shared_backoff = SharedBackoff(
            parse=parse_anthropic,
            failure_types=AnthropicMessagesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client: AsyncAnthropic = (
            client_without_retries(client) if client is not None else AsyncAnthropic(max_retries=0)
        )

    @overload
    def model(
        self,
        model: AnthropicModelName,
        *,
        pricing: AnthropicPricingTable | None = ...,
        default_max_completion_tokens: int = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
    ) -> LLM: ...

    @overload
    def model(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable,
        default_max_completion_tokens: int = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
        inference_geo: str | None = ...,
    ) -> LLM: ...

    def model(
        self,
        model: str,
        *,
        pricing: AnthropicPricingTable | None = None,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
        inference_geo: str | None = None,
    ) -> LLM:
        """Build an `LLM` for one Messages API model.

        `model` is sent verbatim.
        Cataloged models receive `ANTHROPIC_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `default_max_completion_tokens` fills an unstated bound completion limit.
        `cache_ttl` applies to automatic `cache_control` and every cache marker.
        `service_tier` sets the requested Anthropic service tier.
        `inference_geo` requests the inference geography.
        The reported service tier selects pricing.

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
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
            service_tier=service_tier,
            inference_geo=inference_geo,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)


class AnthropicBedrock:
    """Create `LLM` values for Anthropic models on Bedrock."""

    def __init__(  # noqa: PLR0913 (each SharedBackoff parameter remains explicit)
        self,
        *,
        aws_region: str | None = None,
        client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle | None = None,
        http_client: httpx2.AsyncClient | None = None,
        apply_cross_region_premium: bool = True,
        max_concurrent_requests: int | None = 8,
        max_request_starts_per_second: float = 50.0,
        minimum_wait_ceiling_seconds: float = 1.0,
        longest_wait_seconds: float = 60.0,
        wait_multiplier: float = 2.0,
        quiet_seconds_per_decay_step: float = 60.0,
    ) -> None:
        """Build `AnthropicBedrock` without sending a request.

        `aws_region` selects the region for SDK clients created by `AnthropicBedrock`.
        `http_client` applies to SDK clients created by `AnthropicBedrock`.
        `apply_cross_region_premium=False` leaves a prefixed identifier's catalog rates unmultiplied.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the initial and minimum wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            ValueError: `client` accompanies `aws_region` or `http_client`.
                Also raised when a `SharedBackoff` setting is invalid.
        """
        if client is not None and aws_region is not None:
            raise ValueError("Pass at most one of client= or aws_region=")
        if client is not None and http_client is not None:
            raise ValueError("Pass at most one of client= or http_client=")
        self._shared_backoff = SharedBackoff(
            parse=parse_anthropic,
            failure_types=AnthropicMessagesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.aws_region: str | None = aws_region
        self.http_client: httpx2.AsyncClient | None = http_client
        self.apply_cross_region_premium: bool = apply_cross_region_premium
        self._passed_client = client_without_retries(client) if client is not None else None
        self._clients_by_api: dict[
            Literal["mantle", "legacy"], AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle
        ] = {}

    def model(
        self,
        model: AnthropicBedrockModelName,
        *,
        pricing: AnthropicPricingTable | None = None,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTTL = "5m",
    ) -> LLM:
        """Build an `LLM` for one Bedrock model.

        `model` is sent verbatim.
        An exact catalog identifier selects pricing.
        Otherwise a `BEDROCK_CROSS_REGION_MULTIPLIER` prefix is stripped before the catalog lookups.
        A preferred Mantle identifier selects `AsyncAnthropicBedrockMantle`.
        Other known identifiers select `AsyncAnthropicBedrock`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing` and a passed `client`.
        `default_max_completion_tokens` fills an unstated bound completion limit.
        `cache_ttl` applies to automatic `cache_control` and every cache marker.
        Bedrock models accept no Anthropic `service_tier` parameter here.

        Raises:
            anthropic.AnthropicError: A mantle client cannot resolve its region or base URL.
            ValueError: An uncataloged model lacks `pricing` or a passed `client`.
                Also raised when a passed SDK client cannot serve `model`.
        """
        resolution = _resolve_catalog_id(model)
        routing = ANTHROPIC_BEDROCK.get(resolution.catalog_id)
        client = self._client_for(routing, model)
        adapter = AnthropicMessagesAdapter(
            client=client,
            model=model,
            pricing=pricing if pricing is not None else self._catalog_pricing(model, resolution),
            provider_name="aws.bedrock",
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)

    def _client_for(
        self, routing: BedrockRouting | None, model: str
    ) -> AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle:
        """Return the passed client after checking it serves `routing`, or the client `routing` selects.

        Raises:
            anthropic.AnthropicError: A mantle client cannot resolve its region or base URL.
            ValueError: `routing` is `None` with no passed client, or the passed client cannot
                serve `model`.
        """
        client = self._passed_client
        if client is None:
            return self._owned_client(routing, model)
        if routing is not None and not isinstance(client, _BEDROCK_CLIENT_CLASS[routing.api]):
            raise ValueError(
                f"{model!r} is served by the {routing.api!r} Bedrock API, which requires a "
                f"{_BEDROCK_CLIENT_CLASS[routing.api].__name__} client, but a "
                f"{type(client).__name__} was passed."
            )
        return client

    def _owned_client(
        self, routing: BedrockRouting | None, model: str
    ) -> AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle:
        """Return the client `AnthropicBedrock` creates for `routing`, one per Bedrock API.

        Raises:
            anthropic.AnthropicError: A mantle client cannot resolve its region or base URL.
            ValueError: `routing` is `None`.
        """
        if routing is None:
            raise ValueError(
                f"model {model!r} has no matching BedrockRouting; pass client= with its SDK client class"
            )
        cached_client = self._clients_by_api.get(routing.api)
        if cached_client is not None:
            return cached_client
        created_client = _BEDROCK_CLIENT_CLASS[routing.api](
            aws_region=self.aws_region,
            http_client=self.http_client,
            max_retries=0,
        )
        self._clients_by_api[routing.api] = created_client
        return created_client

    def _catalog_pricing(
        self, model: str, resolution: _CatalogResolution
    ) -> AnthropicPricingTable:
        """Return the catalog table for `resolution`.

        `apply_cross_region_premium` multiplies the table by the prefix multiplier of `resolution`.

        Raises:
            ValueError: The catalog identifier is not in `ANTHROPIC_BEDROCK_PRICING`.
        """
        catalog_table = ANTHROPIC_BEDROCK_PRICING.get(resolution.catalog_id)
        if catalog_table is None:
            raise ValueError(
                f"model {model!r} is not in ANTHROPIC_BEDROCK_PRICING; pass pricing= stating its rates"
            )
        if resolution.cross_region_multiplier is None or not self.apply_cross_region_premium:
            return catalog_table
        return catalog_table.multiplied(resolution.cross_region_multiplier)


__all__ = [
    "ANTHROPIC_BEDROCK",
    "ANTHROPIC_BEDROCK_PRICING",
    "ANTHROPIC_PRICING",
    "BEDROCK_CROSS_REGION_MULTIPLIER",
    "Anthropic",
    "AnthropicBedrock",
    "AnthropicBedrockModelName",
    "AnthropicMessagesAdapter",
    "AnthropicModelName",
    "AnthropicPricingTable",
    "AnthropicRates",
    "AnthropicServiceTier",
    "BedrockRouting",
    "CacheTTL",
    "parse_anthropic",
]
