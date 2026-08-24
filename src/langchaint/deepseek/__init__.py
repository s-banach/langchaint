"""Construct DeepSeek `LLM` values with cataloged pricing.

Importing this subpackage requires `openai`.
DeepSeek serves Chat Completions at https://api.deepseek.com.

Prices use USD per one million tokens.
Source: https://api-docs.deepseek.com/quick_start/pricing, read 2026-08-03.
Recheck that page before relying on a table.
`DEEPSEEK_PRICING` contains off-peak list prices.
DeepSeek charges twice those prices during its documented peak windows.
Cache hits use `cache_read_usd_per_million_tokens`.
Cache misses use `input_cache_none_usd_per_million_tokens`.
Cache writes cost zero.
"""

import os
from typing import Literal, overload

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's deepseek backend requires the openai package; install langchaint[deepseek]."
    ) from exc

from openai.types.completion_usage import CompletionUsage

from langchaint.llm import LLM
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.shared import (
    OpenAIPricingTable,
    OpenAIRates,
    client_without_retries,
    parse_openai,
)
from langchaint.shared_backoff import SharedBackoff

type DeepSeekModelName = Literal["deepseek-v4-flash", "deepseek-v4-pro"]
"""Model identifiers with public prices in DEEPSEEK_PRICING."""

DEEPSEEK_PRICING: dict[DeepSeekModelName, OpenAIRates] = {
    "deepseek-v4-flash": OpenAIRates(
        input_cache_none_usd_per_million_tokens=0.14,
        output_usd_per_million_tokens=0.28,
        cache_read_usd_per_million_tokens=0.0028,
        cache_write_usd_per_million_tokens=0.0,
    ),
    "deepseek-v4-pro": OpenAIRates(
        input_cache_none_usd_per_million_tokens=0.435,
        output_usd_per_million_tokens=0.87,
        cache_read_usd_per_million_tokens=0.003625,
        cache_write_usd_per_million_tokens=0.0,
    ),
}
"""Public off-peak prices that `DeepSeek.model` uses by default."""

_PRICING_BY_MODEL_ID = dict[str, OpenAIRates](DEEPSEEK_PRICING.items())
"""`DEEPSEEK_PRICING` with `str` keys for runtime model lookup."""

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_API_KEY_ENVIRONMENT_VARIABLE = "DEEPSEEK_API_KEY"


def cache_read_tokens_from_usage_deepseek(usage: CompletionUsage) -> int:
    """Return `prompt_cache_hit_tokens` from `CompletionUsage.model_extra`, or zero when absent.

    DeepSeek documents that cache-hit and cache-miss counters sum to `prompt_tokens`.
    Source: https://api-docs.deepseek.com/guides/kv_cache, read 2026-08-03.
    `CompletionUsage` omits both counters in openai 2.53.0.
    """
    extra = usage.model_extra
    if extra is None:
        return 0
    hit_tokens = extra.get("prompt_cache_hit_tokens")
    if isinstance(hit_tokens, int):
        return hit_tokens
    return 0


class DeepSeek:
    """Create `LLM` values for DeepSeek."""

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
        """Build `DeepSeek` without sending a request.

        A passed `client` must reach DeepSeek.
        `max_concurrent_requests` limits concurrent admitted requests.
        `max_request_starts_per_second` limits starts during queued demand.
        `minimum_wait_ceiling_seconds` sets the initial and minimum wait ceiling.
        `longest_wait_seconds` caps adaptive and provider-stated waits.
        `wait_multiplier` scales wait-ceiling changes.
        `quiet_seconds_per_decay_step` earns one wait-ceiling reduction.

        Raises:
            ValueError: `client` is absent and `DEEPSEEK_API_KEY` is unset.
                Also raised when a `SharedBackoff` setting is invalid.
        """
        self._shared_backoff = SharedBackoff(
            parse=parse_openai,
            failure_types=OpenAIChatCompletionsAdapter.failure_types,
            max_concurrent_requests=max_concurrent_requests,
            max_request_starts_per_second=max_request_starts_per_second,
            minimum_wait_ceiling_seconds=minimum_wait_ceiling_seconds,
            longest_wait_seconds=longest_wait_seconds,
            wait_multiplier=wait_multiplier,
            quiet_seconds_per_decay_step=quiet_seconds_per_decay_step,
        )
        if client is None:
            api_key = os.environ.get(_API_KEY_ENVIRONMENT_VARIABLE)
            if api_key is None:
                raise ValueError(
                    f"client is None and {_API_KEY_ENVIRONMENT_VARIABLE} is unset; set it, or pass "
                    f"client=AsyncOpenAI(base_url={_DEEPSEEK_BASE_URL!r}, api_key=...)."
                )
            client = AsyncOpenAI(
                base_url=_DEEPSEEK_BASE_URL,
                api_key=api_key,
                max_retries=0,
            )
        self.client: AsyncOpenAI = client_without_retries(client)

    @overload
    def model(
        self,
        model: DeepSeekModelName,
        *,
        pricing: OpenAIRates | None = ...,
    ) -> LLM: ...

    @overload
    def model(
        self,
        model: str,
        *,
        pricing: OpenAIRates,
    ) -> LLM: ...

    def model(
        self,
        model: str,
        *,
        pricing: OpenAIRates | None = None,
    ) -> LLM:
        """Build an `LLM` for one DeepSeek Chat Completions model.

        `model` is sent verbatim.
        Cataloged models receive `DEEPSEEK_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when the SDK client contradicts the DeepSeek provider.
        """
        table = pricing if pricing is not None else _PRICING_BY_MODEL_ID.get(model)
        if table is None:
            raise ValueError(
                f"model {model!r} is not in DEEPSEEK_PRICING; pass pricing= stating its rates"
            )
        adapter = OpenAIChatCompletionsAdapter(
            client=self.client,
            model=model,
            pricing=OpenAIPricingTable(default=table),
            provider_name="deepseek",
            supports_prompt_cache_options=False,
            cache_read_tokens_from_usage=cache_read_tokens_from_usage_deepseek,
        )
        return LLM(adapter, shared_backoff=self._shared_backoff)


__all__ = [
    "DEEPSEEK_PRICING",
    "DeepSeek",
    "DeepSeekModelName",
    "cache_read_tokens_from_usage_deepseek",
]
