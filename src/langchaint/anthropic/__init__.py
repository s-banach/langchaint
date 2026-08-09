"""The anthropic backend provides accounts, the Messages adapter, catalogs, and pricing.

Importing this subpackage requires `anthropic`.
`AnthropicAccount.model` sends the stated model identifier verbatim.
`AnthropicBedrockAccount.model` sends a cataloged Bedrock identifier verbatim.
`ANTHROPIC_BEDROCK` selects its SDK client class and default pricing.
Anthropic 0.120.0 constructs both Bedrock client classes without network requests.

Cataloged models receive standard-tier `ANTHROPIC_PRICING` rates.
Uncataloged Anthropic models require `pricing`.
Responses from uncataloged service tiers cost NaN.
Pass `client=AsyncAnthropic(http_client=...)` for custom first-party transports.
`AnthropicBedrockAccount` accepts `http_client` directly.

Prices use USD per one million tokens.
Source: https://platform.claude.com/docs/en/about-claude/pricing.
Recheck that page before relying on a table.
Cache reads cost 0.1 times base input.
Five-minute cache writes cost 1.25 times base input.
One-hour cache writes cost twice base input.
`ANTHROPIC_PRICING` covers the standard service tier.
Callers state priority, batch, or negotiated rates through `pricing`.
"""

from collections.abc import Mapping
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

from langchaint.account_base import AccountBase
from langchaint.anthropic.messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicPricedServiceTier,
    AnthropicPricingTable,
    AnthropicServiceTier,
    CacheTTL,
    client_without_retries,
    parse_anthropic,
)
from langchaint.llm import LLM

type AnthropicModelName = Literal[
    "claude-fable-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
]
"""Model identifiers with public prices in ANTHROPIC_PRICING."""

ANTHROPIC_PRICING: dict[AnthropicModelName, AnthropicPricingTable] = {
    "claude-fable-5": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=10.00,
        output_usd_per_million_tokens=50.00,
        cache_read_usd_per_million_tokens=1.00,
        cache_write_5m_usd_per_million_tokens=12.50,
        cache_write_1h_usd_per_million_tokens=20.00,
    ),
    "claude-sonnet-4-6": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=3.00,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.30,
        cache_write_5m_usd_per_million_tokens=3.75,
        cache_write_1h_usd_per_million_tokens=6.00,
    ),
    # introductory pricing, through 2026-08-31; standard 3.00/15.00 from 2026-09-01
    "claude-sonnet-5": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=2.00,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.20,
        cache_write_5m_usd_per_million_tokens=2.50,
        cache_write_1h_usd_per_million_tokens=4.00,
    ),
    "claude-opus-4-6": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-opus-4-7": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-opus-4-8": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10.00,
    ),
    "claude-haiku-4-5-20251001": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=5.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_5m_usd_per_million_tokens=1.25,
        cache_write_1h_usd_per_million_tokens=2.00,
    ),
}
"""Public prices per anthropic model; the default pricing lookup, shared by both constructors."""

_PRICING_BY_MODEL_ID = dict[str, AnthropicPricingTable](ANTHROPIC_PRICING.items())
"""`ANTHROPIC_PRICING` with `str` keys for runtime model lookup."""


type AnthropicBedrockModelName = Literal[
    "anthropic.claude-fable-5",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5",
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-sonnet-4-6",
]
"""Bedrock identifiers accepted by `AnthropicBedrockAccount.model`.

The `us.` prefix identifies a cross-region inference profile.
Each identifier is sent verbatim.
"""


@dataclass(frozen=True, kw_only=True)
class BedrockRouting:
    """The SDK client API and default pricing source for one Bedrock identifier.

    `api` selects a class from `_BEDROCK_CLIENT_CLASS`.
    `pricing_key` selects an `ANTHROPIC_PRICING` entry.
    """

    api: Literal["mantle", "legacy"]
    pricing_key: AnthropicModelName


ANTHROPIC_BEDROCK: dict[AnthropicBedrockModelName, BedrockRouting] = {
    "anthropic.claude-fable-5": BedrockRouting(api="mantle", pricing_key="claude-fable-5"),
    "anthropic.claude-opus-4-8": BedrockRouting(api="mantle", pricing_key="claude-opus-4-8"),
    "anthropic.claude-opus-4-7": BedrockRouting(api="mantle", pricing_key="claude-opus-4-7"),
    "anthropic.claude-sonnet-5": BedrockRouting(api="mantle", pricing_key="claude-sonnet-5"),
    "anthropic.claude-haiku-4-5": BedrockRouting(
        api="mantle", pricing_key="claude-haiku-4-5-20251001"
    ),
    "us.anthropic.claude-opus-4-6-v1": BedrockRouting(api="legacy", pricing_key="claude-opus-4-6"),
    "us.anthropic.claude-sonnet-4-6": BedrockRouting(
        api="legacy", pricing_key="claude-sonnet-4-6"
    ),
}
"""Routing for each `AnthropicBedrockModelName` value."""

_BEDROCK_BY_MODEL_ID = dict[str, BedrockRouting](ANTHROPIC_BEDROCK.items())
"""`ANTHROPIC_BEDROCK` with `str` keys for runtime model lookup."""

_BEDROCK_CLIENT_CLASS: dict[
    Literal["mantle", "legacy"], type[AsyncAnthropicBedrockMantle | AsyncAnthropicBedrock]
] = {
    "mantle": AsyncAnthropicBedrockMantle,
    "legacy": AsyncAnthropicBedrock,
}


def _anthropic_adapter(
    model: str,
    *,
    client: AsyncAnthropic,
    pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTTL = "5m",
    service_tier: AnthropicServiceTier | None = None,
) -> AnthropicMessagesAdapter:
    """Build the adapter for one Messages API model.

    Raises:
        ValueError: An uncataloged model lacks `pricing`.
            Also raised when `pricing` lacks its required `"standard"` key.
    """
    catalog_table = _PRICING_BY_MODEL_ID.get(model)
    if catalog_table is None:
        if pricing is None:
            raise ValueError(
                f"model {model!r} is not in ANTHROPIC_PRICING; pass pricing= stating its rates"
            )
    else:
        pricing = {"standard": catalog_table, **(pricing or {})}
    return AnthropicMessagesAdapter(
        client=client,
        model=model,
        pricing=pricing,
        provider_name="anthropic",
        default_max_completion_tokens=default_max_completion_tokens,
        cache_ttl=cache_ttl,
        service_tier=service_tier,
    )


def _anthropic_bedrock_adapter(
    model: str,
    *,
    routing: BedrockRouting,
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
    pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = None,
    default_max_completion_tokens: int = 4096,
    cache_ttl: CacheTTL = "5m",
) -> AnthropicMessagesAdapter:
    """Build the adapter for one cataloged Bedrock model.

    Raises:
        ValueError: `pricing` lacks its required `"standard"` key.
    """
    return AnthropicMessagesAdapter(
        client=client,
        model=model,
        pricing={"standard": ANTHROPIC_PRICING[routing.pricing_key], **(pricing or {})},
        provider_name="aws.bedrock",
        default_max_completion_tokens=default_max_completion_tokens,
        cache_ttl=cache_ttl,
    )


class AnthropicAccount(AccountBase):
    """Anthropic SDK client and `SharedBackoff` shared by Anthropic models."""

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
        """Build an Anthropic account without sending a request.

        `client=None` constructs `AsyncAnthropic()`.
        A passed `client` remains caller-owned.
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
        super().__init__(
            parse=parse_anthropic,
            failure_types=AnthropicMessagesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client = (
            client_without_retries(client) if client is not None else AsyncAnthropic(max_retries=0)
        )
        if client is None:
            self._register_owned_close(self.client.close)

    @overload
    def model(
        self,
        model: AnthropicModelName,
        *,
        pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = ...,
        default_max_completion_tokens: int = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def model(
        self,
        model: str,
        *,
        pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable],
        default_max_completion_tokens: int = ...,
        cache_ttl: CacheTTL = ...,
        service_tier: AnthropicServiceTier | None = ...,
    ) -> LLM: ...

    def model(
        self,
        model: str,
        *,
        pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = None,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTTL = "5m",
        service_tier: AnthropicServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Messages API model.

        `model` is sent verbatim.
        Cataloged models receive standard-tier `ANTHROPIC_PRICING`.
        Stated `pricing` extends or replaces catalog pricing.
        Uncataloged models require `pricing` with a `"standard"` entry.
        `default_max_completion_tokens` fills an unstated bound completion limit.
        `cache_ttl` applies to every cache marker.
        `service_tier` sets the requested Anthropic service tier.
        The reported service tier selects pricing.

        Raises:
            RuntimeError: This account is closed.
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when `pricing` lacks its required `"standard"` key.
        """
        self._state.ensure_open()
        adapter = _anthropic_adapter(
            model,
            client=self.client,
            pricing=pricing,
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
            service_tier=service_tier,
        )
        return self._llm(adapter)


class AnthropicBedrockAccount(AccountBase):
    """Bedrock SDK clients and `SharedBackoff` shared by Anthropic models."""

    def __init__(  # noqa: PLR0913 (the account states every shared request policy)
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
        """Build a Bedrock account without sending a request.

        `aws_region` selects the region for account-created SDK clients.
        A passed `client` remains caller-owned.
        A passed `http_client` becomes account-owned.
        `http_client` applies to account-created SDK clients.
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
        super().__init__(
            parse=parse_anthropic,
            failure_types=AnthropicMessagesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.aws_region = aws_region
        self.http_client = http_client
        self._passed_client = client_without_retries(client) if client is not None else None
        self._clients_by_api: dict[
            Literal["mantle", "legacy"], AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle
        ] = {}
        if http_client is not None:
            self._register_owned_close(http_client.aclose)

    def model(
        self,
        model: AnthropicBedrockModelName,
        *,
        pricing: Mapping[AnthropicPricedServiceTier, AnthropicPricingTable] | None = None,
        default_max_completion_tokens: int = 4096,
        cache_ttl: CacheTTL = "5m",
    ) -> LLM:
        """Build an `LLM` for one cataloged Bedrock model.

        `model` is sent verbatim.
        `ANTHROPIC_BEDROCK` selects its SDK client class and pricing estimate.
        Stated `pricing` extends or replaces the `"standard"` estimate.
        `default_max_completion_tokens` fills an unstated bound completion limit.
        `cache_ttl` applies to every cache marker.
        Bedrock models accept no Anthropic `service_tier` parameter here.

        Raises:
            anthropic.AnthropicError: A mantle client cannot resolve its region or base URL.
            RuntimeError: This account is closed.
            ValueError: `model` is absent from `ANTHROPIC_BEDROCK`.
                Also raised when a passed SDK client cannot serve `model`.
        """
        self._state.ensure_open()
        routing = _BEDROCK_BY_MODEL_ID.get(model)
        if routing is None:
            raise ValueError(f"model {model!r} is not in ANTHROPIC_BEDROCK")
        client = self._passed_client
        if client is None:
            client = self._clients_by_api.get(routing.api)
            if client is None:
                client = _BEDROCK_CLIENT_CLASS[routing.api](
                    aws_region=self.aws_region,
                    http_client=self.http_client,
                    max_retries=0,
                )
                self._clients_by_api[routing.api] = client
                self._register_owned_close(client.close)
        else:
            required_class = _BEDROCK_CLIENT_CLASS[routing.api]
            if not isinstance(client, required_class):
                raise ValueError(
                    f"{model!r} is served by the {routing.api!r} Bedrock API, which requires a "
                    f"{required_class.__name__} client, but a {type(client).__name__} was passed."
                )
        adapter = _anthropic_bedrock_adapter(
            model,
            routing=routing,
            client=client,
            pricing=pricing,
            default_max_completion_tokens=default_max_completion_tokens,
            cache_ttl=cache_ttl,
        )
        return self._llm(adapter)


__all__ = [
    "ANTHROPIC_BEDROCK",
    "ANTHROPIC_PRICING",
    "AnthropicAccount",
    "AnthropicBedrockAccount",
    "AnthropicBedrockModelName",
    "AnthropicMessagesAdapter",
    "AnthropicModelName",
    "AnthropicPricedServiceTier",
    "AnthropicPricingTable",
    "AnthropicServiceTier",
    "BedrockRouting",
    "CacheTTL",
    "parse_anthropic",
]
