"""Construct Anthropic and Bedrock `LLM` values with cataloged pricing.

Importing this subpackage requires `anthropic`.
`Anthropic.model` sends the stated model identifier verbatim.
`AnthropicBedrock.model` sends the stated Bedrock identifier verbatim.
`ANTHROPIC_BEDROCK` lists preferred identifiers and their `BedrockRouting` values.

Cataloged Anthropic models receive `ANTHROPIC_PRICING`.
Cataloged Bedrock models receive `ANTHROPIC_BEDROCK_PRICING`.
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
from typing import Literal, overload

import httpx

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
except ModuleNotFoundError as exc:
    if exc.name != "anthropic":
        raise
    raise ModuleNotFoundError(
        "langchaint's anthropic backend requires the anthropic package; install anthropic."
    ) from exc

from langchaint.anthropic._generated_pricing import (
    ANTHROPIC_BEDROCK_PRICING,
    ANTHROPIC_PRICING,
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
from langchaint.llm import LLM
from langchaint.shared_backoff import SharedBackoff

_PRICING_BY_MODEL_ID = dict[str, AnthropicPricingTable](ANTHROPIC_PRICING.items())
"""`ANTHROPIC_PRICING` with `str` keys for runtime model lookup."""


type AnthropicBedrockModelName = (
    Literal[
        "anthropic.claude-fable-5",
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
"""


@dataclass(frozen=True, kw_only=True)
class BedrockRouting:
    """The SDK client API for one Bedrock identifier."""

    api: Literal["mantle", "legacy"]


ANTHROPIC_BEDROCK: dict[AnthropicBedrockModelName, BedrockRouting] = {
    "anthropic.claude-fable-5": BedrockRouting(api="mantle"),
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


def _bedrock_routing(model: str) -> BedrockRouting | None:
    """Return routing for one exact catalog identifier."""
    return ANTHROPIC_BEDROCK.get(model)


def _anthropic_adapter(
    model: str,
    *,
    client: AsyncAnthropic,
    pricing: AnthropicPricingTable | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTTL = "5m",
    service_tier: AnthropicServiceTier | None = None,
    inference_geo: str | None = None,
) -> AnthropicMessagesAdapter:
    """Build the adapter for one Messages API model.

    Raises:
        ValueError: An uncataloged model lacks `pricing`.
    """
    catalog_table = _PRICING_BY_MODEL_ID.get(model)
    if catalog_table is None:
        if pricing is None:
            raise ValueError(
                f"model {model!r} is not in ANTHROPIC_PRICING; pass pricing= stating its rates"
            )
    else:
        pricing = pricing or catalog_table
    return AnthropicMessagesAdapter(
        client=client,
        model=model,
        pricing=pricing,
        provider_name="anthropic",
        default_max_completion_tokens=default_max_completion_tokens,
        cache_ttl=cache_ttl,
        service_tier=service_tier,
        inference_geo=inference_geo,
    )


def _anthropic_bedrock_adapter(
    model: str,
    *,
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
    catalog_table: AnthropicPricingTable | None,
    pricing: AnthropicPricingTable | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTTL = "5m",
) -> AnthropicMessagesAdapter:
    """Build the adapter for one Bedrock model.

    Raises:
        ValueError: An uncataloged model lacks `pricing`.
    """
    if catalog_table is None:
        if pricing is None:
            raise ValueError(
                f"model {model!r} is not in ANTHROPIC_BEDROCK_PRICING; pass pricing= stating its rates"
            )
    else:
        pricing = pricing or catalog_table
    return AnthropicMessagesAdapter(
        client=client,
        model=model,
        pricing=pricing,
        provider_name="aws.bedrock",
        default_max_completion_tokens=default_max_completion_tokens,
        cache_ttl=cache_ttl,
    )


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
        `cache_ttl` applies to every cache marker.
        `service_tier` sets the requested Anthropic service tier.
        `inference_geo` requests the inference geography.
        The reported service tier selects pricing.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
        """
        adapter = _anthropic_adapter(
            model,
            client=self.client,
            pricing=pricing,
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
        http_client: httpx.AsyncClient | None = None,
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
        self.http_client: httpx.AsyncClient | None = http_client
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
        An exact preferred Mantle identifier selects `AsyncAnthropicBedrockMantle`.
        Other known identifiers select `AsyncAnthropicBedrock`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing` and a passed `client`.
        `default_max_completion_tokens` fills an unstated bound completion limit.
        `cache_ttl` applies to every cache marker.
        Bedrock models accept no Anthropic `service_tier` parameter here.

        Raises:
            anthropic.AnthropicError: A mantle client cannot resolve its region or base URL.
            ValueError: An uncataloged model lacks `pricing` or a passed `client`.
                Also raised when a passed SDK client cannot serve `model`.
        """
        routing = _bedrock_routing(model)
        client = self._passed_client
        if client is None:
            if routing is None:
                raise ValueError(
                    f"model {model!r} has no matching BedrockRouting; pass client= with its SDK client class"
                )
            client = self._clients_by_api.get(routing.api)
            if client is None:
                client = _BEDROCK_CLIENT_CLASS[routing.api](
                    aws_region=self.aws_region,
                    http_client=self.http_client,
                    max_retries=0,
                )
                self._clients_by_api[routing.api] = client
        elif routing is not None:
            required_class = _BEDROCK_CLIENT_CLASS[routing.api]
            if not isinstance(client, required_class):
                raise ValueError(
                    f"{model!r} is served by the {routing.api!r} Bedrock API, which requires a "
                    f"{required_class.__name__} client, but a {type(client).__name__} was passed."
                )
        adapter = _anthropic_bedrock_adapter(
            model,
            client=client,
            catalog_table=ANTHROPIC_BEDROCK_PRICING.get(model),
            pricing=pricing,
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)


__all__ = [
    "ANTHROPIC_BEDROCK",
    "ANTHROPIC_BEDROCK_PRICING",
    "ANTHROPIC_PRICING",
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
