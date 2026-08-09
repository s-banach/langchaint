"""The openai backend provides accounts, adapters, model catalogs, and pricing.

Importing this subpackage requires `openai`.
`OpenAIAccount.model` uses the Responses API and reports `provider_name="openai"`.
`OpenAIBedrockAccount.model` uses Responses and reports `provider_name="aws.bedrock"`.
Use `OpenAIChatCompletionsAdapter` directly for compatible endpoints.
Use `OpenAIResponsesAdapter` directly for Azure.

Cataloged models receive default-tier `OPENAI_PRICING` rates.
Uncataloged OpenAI models require `pricing` and `supports_prompt_cache_options`.
`OpenAIBedrockAccount.model` always requires both parameters.
Responses from uncataloged service tiers cost NaN.

Token prices use USD per one million tokens.
Web-search prices use USD per invocation.
File-search prices use USD per invocation.
Source: https://developers.openai.com/api/docs/pricing.
Recheck that page before relying on a table.
`OPENAI_PRICING` covers the default service tier.
Its web-search rates are public list-price estimates.
`OpenAIAccount.model(pricing=...)` replaces cataloged estimates.
`OpenAIBedrockAccount.model(pricing=...)` accepts caller rates.
Earlier models cache automatically and have free cache writes.
The gpt-5.6 family bills cache writes and accepts `prompt_cache_options`.
`PROMPT_CACHE_OPTIONS_MODELS` lists that family.
"""

from collections.abc import Mapping
from typing import Literal, overload

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires the openai package; install openai."
    ) from exc

from langchaint.account_base import AccountBase
from langchaint.llm import LLM
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.responses_adapter import (
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.shared import (
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIServiceTier,
    client_without_retries,
    parse_openai,
)

type OpenAIModelName = Literal[
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]
"""Model identifiers with public prices in OPENAI_PRICING."""

OPENAI_PRICING: dict[OpenAIModelName, OpenAIPricingTable] = {
    "gpt-5.1": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.25,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.125,
        cache_write_usd_per_million_tokens=0.00,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.2": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.75,
        output_usd_per_million_tokens=14.00,
        cache_read_usd_per_million_tokens=0.175,
        cache_write_usd_per_million_tokens=0.00,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.4": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_usd_per_million_tokens=0.00,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.4-mini": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=0.75,
        output_usd_per_million_tokens=4.50,
        cache_read_usd_per_million_tokens=0.075,
        cache_write_usd_per_million_tokens=0.00,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.5": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=30.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=0.00,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.6-luna": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=6.00,
        cache_read_usd_per_million_tokens=0.10,
        cache_write_usd_per_million_tokens=1.25,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.6-terra": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=15.00,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_usd_per_million_tokens=3.125,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
    "gpt-5.6-sol": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=30.00,
        cache_read_usd_per_million_tokens=0.50,
        cache_write_usd_per_million_tokens=6.25,
        web_search_usd_per_invocation=0.01,
        file_search_usd_per_invocation=0.0025,
    ),
}
"""Public prices per openai model; the default pricing lookup."""

_PRICING_BY_MODEL_ID = dict[str, OpenAIPricingTable](OPENAI_PRICING.items())
"""`OPENAI_PRICING` with `str` keys for runtime model lookup."""

PROMPT_CACHE_OPTIONS_MODELS: frozenset[OpenAIModelName] = frozenset({
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
})
"""Cataloged models accepting `prompt_cache_options`.

OpenAI 2.45.0 documents this parameter for gpt-5.6-and-later.
It carries `automatic_prompt_caching=False` to the request.
`OpenAIAccount.model` derives `supports_prompt_cache_options` from this set.
The set stays independent from pricing because parameter availability can change independently.
"""


class OpenAIAccount(AccountBase):
    """One OpenAI SDK client and `SharedBackoff` shared by OpenAI models."""

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
        """Build an OpenAI account without sending a request.

        `client=None` constructs `AsyncOpenAI()`.
        A passed `client` remains caller-owned.
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
        super().__init__(
            parse=parse_openai,
            failure_types=OpenAIResponsesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client = (
            client_without_retries(client) if client is not None else AsyncOpenAI(max_retries=0)
        )
        if client is None:
            self._register_owned_close(self.client.close)

    @overload
    def model(
        self,
        model: OpenAIModelName,
        *,
        pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable] | None = ...,
        supports_prompt_cache_options: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIServiceTier | None = ...,
    ) -> LLM: ...

    @overload
    def model(
        self,
        model: str,
        *,
        pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable],
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIServiceTier | None = ...,
    ) -> LLM: ...

    def model(
        self,
        model: str,
        *,
        pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable] | None = None,
        supports_prompt_cache_options: bool | None = None,
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIServiceTier | None = None,
    ) -> LLM:
        """Build an `LLM` for one Responses API model.

        `model` is sent verbatim.
        Cataloged models receive public default pricing.
        Stated `pricing` replaces or extends catalog pricing.
        Uncataloged models require `pricing` with a `"default"` entry.
        `supports_prompt_cache_options` states whether the model accepts that request parameter.
        Cataloged models derive that value from `PROMPT_CACHE_OPTIONS_MODELS`.
        `reasoning_summary` requests readable reasoning summary text.
        `service_tier` sets the requested OpenAI service tier.
        The reported service tier selects pricing.

        Raises:
            RuntimeError: This account is closed.
            ValueError: An uncataloged model lacks required pricing or caching data.
                Also raised when the SDK client contradicts the OpenAI provider.
        """
        self._state.ensure_open()
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
            pricing = {"default": catalog_table, **(pricing or {})}
            if supports_prompt_cache_options is None:
                supports_prompt_cache_options = model in PROMPT_CACHE_OPTIONS_MODELS
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="openai",
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
            service_tier=service_tier,
        )
        return self._llm(adapter)


class OpenAIBedrockAccount(AccountBase):
    """Bedrock SDK client and `SharedBackoff` shared by OpenAI models."""

    def __init__(  # noqa: PLR0913 (the account states every shared request policy)
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
        """Build a Bedrock account without sending a request.

        `aws_region` selects the region for an account-created SDK client.
        A passed `client` remains caller-owned.
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
        super().__init__(
            parse=parse_openai,
            failure_types=OpenAIResponsesAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        self.client = (
            client_without_retries(client)
            if client is not None
            else AsyncBedrockOpenAI(aws_region=aws_region, max_retries=0)
        )
        if client is None:
            self._register_owned_close(self.client.close)

    def model(
        self,
        model: str,
        *,
        pricing: Mapping[OpenAIPricedServiceTier, OpenAIPricingTable],
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
    ) -> LLM:
        """Build an `LLM` for one OpenAI model served by Bedrock.

        `model` is sent verbatim.
        Bedrock model identifiers have no carried pricing catalog.
        `pricing` requires a `"default"` entry.
        `supports_prompt_cache_options` states whether the model accepts that request parameter.
        `reasoning_summary` requests readable reasoning summary text.
        Bedrock models accept no OpenAI `service_tier` parameter here.

        Raises:
            RuntimeError: This account is closed.
            ValueError: `pricing` lacks its required `"default"` key.
        """
        self._state.ensure_open()
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="aws.bedrock",
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
        )
        return self._llm(adapter)


__all__ = [
    "OPENAI_PRICING",
    "PROMPT_CACHE_OPTIONS_MODELS",
    "OpenAIAccount",
    "OpenAIBedrockAccount",
    "OpenAIChatCompletionsAdapter",
    "OpenAIModelName",
    "OpenAIPricedServiceTier",
    "OpenAIPricingTable",
    "OpenAIResponsesAdapter",
    "OpenAIServiceTier",
    "ReasoningSummary",
    "parse_openai",
]
