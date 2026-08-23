"""Generated OpenAI pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""

from typing import Literal

from langchaint.openai.shared import (
    OpenAILongContextPricing,
    OpenAIPricingTable,
    OpenAIRates,
)

type OpenAIModelName = Literal[
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]

_GPT_5_6_SOL = OpenAIPricingTable(
    default=OpenAIRates(
        input_cache_none_usd_per_million_tokens=4,
        output_usd_per_million_tokens=20,
        cache_read_usd_per_million_tokens=0.4,
        cache_write_usd_per_million_tokens=5,
    ),
    flex=OpenAIRates(
        input_cache_none_usd_per_million_tokens=2,
        output_usd_per_million_tokens=10,
        cache_read_usd_per_million_tokens=0.2,
        cache_write_usd_per_million_tokens=2.5,
    ),
    fast=OpenAIRates(
        input_cache_none_usd_per_million_tokens=8,
        output_usd_per_million_tokens=40,
        cache_read_usd_per_million_tokens=0.8,
        cache_write_usd_per_million_tokens=10,
    ),
    long_context=OpenAILongContextPricing(
        input_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    regional_processing_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_5_6_TERRA = OpenAIPricingTable(
    default=OpenAIRates(
        input_cache_none_usd_per_million_tokens=2,
        output_usd_per_million_tokens=12,
        cache_read_usd_per_million_tokens=0.2,
        cache_write_usd_per_million_tokens=2.5,
    ),
    flex=OpenAIRates(
        input_cache_none_usd_per_million_tokens=1,
        output_usd_per_million_tokens=6,
        cache_read_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=1.25,
    ),
    fast=OpenAIRates(
        input_cache_none_usd_per_million_tokens=4,
        output_usd_per_million_tokens=24,
        cache_read_usd_per_million_tokens=0.4,
        cache_write_usd_per_million_tokens=5,
    ),
    long_context=OpenAILongContextPricing(
        input_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    regional_processing_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_5_6_LUNA = OpenAIPricingTable(
    default=OpenAIRates(
        input_cache_none_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=1.2,
        cache_read_usd_per_million_tokens=0.02,
        cache_write_usd_per_million_tokens=0.25,
    ),
    flex=OpenAIRates(
        input_cache_none_usd_per_million_tokens=0.1,
        output_usd_per_million_tokens=0.6,
        cache_read_usd_per_million_tokens=0.01,
        cache_write_usd_per_million_tokens=0.125,
    ),
    fast=OpenAIRates(
        input_cache_none_usd_per_million_tokens=0.4,
        output_usd_per_million_tokens=2.4,
        cache_read_usd_per_million_tokens=0.04,
        cache_write_usd_per_million_tokens=0.5,
    ),
    long_context=OpenAILongContextPricing(
        input_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    regional_processing_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

OPENAI_PRICING: dict[OpenAIModelName, OpenAIPricingTable] = {
    "gpt-5.6": _GPT_5_6_SOL,
    "gpt-5.6-sol": _GPT_5_6_SOL,
    "gpt-5.6-terra": _GPT_5_6_TERRA,
    "gpt-5.6-luna": _GPT_5_6_LUNA,
}
