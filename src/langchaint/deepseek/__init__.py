"""The deepseek backend: deepseek_model, its model catalog, and pricing.

DeepSeek serves an OpenAI-compatible Chat Completions endpoint at https://api.deepseek.com,
so deepseek_model wraps OpenAIChatCompletionsAdapter over an AsyncOpenAI pointed there.
Importing this subpackage requires the openai package;
the import below raises a ModuleNotFoundError naming the package to install.

Prices are USD per one million tokens,
taken from the provider's official pricing page: https://api-docs.deepseek.com/quick_start/pricing
(read 2026-08-03).
The catalog is the off-peak list price: DeepSeek doubles both models' prices during 09:00-12:00
and 14:00-18:00 Beijing time, so a request served in those windows costs twice what langchaint
reports.
Prices are the one provider fact langchaint cannot verify by SDK introspection;
re-check the page before relying on a table for billing.
The page prices cache hits and cache misses: the cache-hit price is the table's cache_read rate
and the cache-miss price its cache_none rate.
Nothing is billed as a cache write, so the cache-write rate is 0.0, and
cache_read_tokens_from_usage_deepseek leaves that counter 0.
"""

import os
from typing import Literal, overload

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's deepseek backend requires the openai package; install openai."
    ) from exc

from openai.types.completion_usage import CompletionUsage

from langchaint.llm import LLM
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.shared import OpenAIPricingTable
from langchaint.shared_backoff import SharedBackoff

type DeepSeekModelName = Literal["deepseek-v4-flash", "deepseek-v4-pro"]
"""Model identifiers with public prices in DEEPSEEK_PRICING."""

DEEPSEEK_PRICING: dict[DeepSeekModelName, OpenAIPricingTable] = {
    "deepseek-v4-flash": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=0.14,
        output_usd_per_million_tokens=0.28,
        cache_read_usd_per_million_tokens=0.0028,
        cache_write_usd_per_million_tokens=0.0,
    ),
    "deepseek-v4-pro": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=0.435,
        output_usd_per_million_tokens=0.87,
        cache_read_usd_per_million_tokens=0.003625,
        cache_write_usd_per_million_tokens=0.0,
    ),
}
"""Public off-peak prices per deepseek model; the default pricing lookup."""

_PRICING_BY_MODEL_ID = dict[str, OpenAIPricingTable](DEEPSEEK_PRICING.items())
"""DEEPSEEK_PRICING under a str key, so deepseek_model can look up a possibly-uncataloged model id."""

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_API_KEY_ENVIRONMENT_VARIABLE = "DEEPSEEK_API_KEY"


def cache_read_tokens_from_usage_deepseek(usage: CompletionUsage) -> int:
    """Read the cache-read counter DeepSeek reports: the prompt_cache_hit_tokens extra field, 0 absent.

    DeepSeek reports prompt_cache_hit_tokens and prompt_cache_miss_tokens, which sum to
    prompt_tokens (https://api-docs.deepseek.com/guides/kv_cache, read 2026-08-03).
    Neither is a field of the installed SDK's CompletionUsage, so the counter is read from the
    model's extra fields, where the SDK lands every response field it does not model.
    deepseek_model passes this as the adapter's cache_read_tokens_from_usage; without it, every
    cache hit would price at the cache-miss rate, a 50x over-report on deepseek-v4-flash.
    """
    extra = usage.model_extra
    if extra is None:
        return 0
    hit_tokens = extra.get("prompt_cache_hit_tokens")
    if isinstance(hit_tokens, int):
        return hit_tokens
    return 0


@overload
def deepseek_model(
    model: DeepSeekModelName,
    *,
    client: AsyncOpenAI | None = ...,
    pricing: OpenAIPricingTable | None = ...,
    shared_backoff: SharedBackoff | None = ...,
    max_attempts: int = ...,
) -> LLM: ...
@overload
def deepseek_model(
    model: str,
    *,
    pricing: OpenAIPricingTable,
    client: AsyncOpenAI | None = ...,
    shared_backoff: SharedBackoff | None = ...,
    max_attempts: int = ...,
) -> LLM: ...
def deepseek_model(
    model: str,
    *,
    client: AsyncOpenAI | None = None,
    pricing: OpenAIPricingTable | None = None,
    shared_backoff: SharedBackoff | None = None,
    max_attempts: int = 3,
) -> LLM:
    """Build a ready LLM for one model on DeepSeek's Chat Completions endpoint.

    model is sent verbatim.
    client None constructs AsyncOpenAI(base_url="https://api.deepseek.com", api_key=...) with the
    key read from DEEPSEEK_API_KEY.
    The key is read here rather than left to the SDK because the SDK's own fallback reads
    OPENAI_API_KEY, which would silently send the OpenAI key to api.deepseek.com.
    A passed client is used as constructed, so its base_url must point at DeepSeek itself.
    pricing is one table rather than a tier mapping: DeepSeek reports no service_tier, so every
    response prices at the "default" key this constructor wraps the table under.
    On a cataloged id, omitting it prices at the public off-peak rates in DEEPSEEK_PRICING, which
    the module docstring states peak windows double; an uncataloged id requires it, there being no
    table to fall back on.
    The adapter is built with supports_prompt_cache_options False: DeepSeek's caching is always on
    and it documents no parameter declining it, so bind(automatic_prompt_caching=False) raises at
    bind time.
    shared_backoff and max_attempts have the LLM.__init__ meanings;
    pass one SharedBackoff across models on the same account so a rate limit pauses them together,
    and note one instance serves one event loop.

    Raises:
        ValueError: model is outside the catalog and pricing is missing, or client is None and
            DEEPSEEK_API_KEY is unset, so there is no credential to reach api.deepseek.com with.
            LLM.__init__ raises it when max_attempts is not a positive int.
    """
    table = pricing if pricing is not None else _PRICING_BY_MODEL_ID.get(model)
    if table is None:
        raise ValueError(
            f"model {model!r} is not in DEEPSEEK_PRICING; pass pricing= stating its rates"
        )
    if client is None:
        api_key = os.environ.get(_API_KEY_ENVIRONMENT_VARIABLE)
        if api_key is None:
            raise ValueError(
                f"client is None and {_API_KEY_ENVIRONMENT_VARIABLE} is unset; set it, or pass "
                f"client=AsyncOpenAI(base_url={_DEEPSEEK_BASE_URL!r}, api_key=...). "
                "Leaving the key to the SDK would read OPENAI_API_KEY instead and silently send "
                "the OpenAI key to api.deepseek.com."
            )
        client = AsyncOpenAI(base_url=_DEEPSEEK_BASE_URL, api_key=api_key)
    return LLM(
        OpenAIChatCompletionsAdapter(
            client=client,
            model=model,
            pricing={"default": table},
            provider_name="deepseek",
            supports_prompt_cache_options=False,
            cache_read_tokens_from_usage=cache_read_tokens_from_usage_deepseek,
        ),
        shared_backoff=shared_backoff,
        max_attempts=max_attempts,
    )


__all__ = [
    "DEEPSEEK_PRICING",
    "DeepSeekModelName",
    "cache_read_tokens_from_usage_deepseek",
    "deepseek_model",
]
