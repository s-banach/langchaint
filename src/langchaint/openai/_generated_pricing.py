"""Generated OpenAI pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""

from typing import Literal

from langchaint.billing.pricing import LongContextPricing, TokenRates
from langchaint.openai.shared import OpenAIPricingTable

type OpenAIModelId = Literal[
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-6.1-sol",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-6-astra",
]

_GPT_5_6_SOL = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=4,
        output_tokens=20,
        input_tokens_cache_read=0.4,
        input_tokens_cache_write=5,
    ),
    flex=TokenRates(
        input_tokens_cache_none=2,
        output_tokens=10,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write=2.5,
    ),
    fast=TokenRates(
        input_tokens_cache_none=8,
        output_tokens=40,
        input_tokens_cache_read=0.8,
        input_tokens_cache_write=10,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_5_6_TERRA = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=2,
        output_tokens=12,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write=2.5,
    ),
    flex=TokenRates(
        input_tokens_cache_none=1,
        output_tokens=6,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write=1.25,
    ),
    fast=TokenRates(
        input_tokens_cache_none=4,
        output_tokens=24,
        input_tokens_cache_read=0.4,
        input_tokens_cache_write=5,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_5_6_LUNA = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=0.2,
        output_tokens=1.2,
        input_tokens_cache_read=0.02,
        input_tokens_cache_write=0.25,
    ),
    flex=TokenRates(
        input_tokens_cache_none=0.1,
        output_tokens=0.6,
        input_tokens_cache_read=0.01,
        input_tokens_cache_write=0.125,
    ),
    fast=TokenRates(
        input_tokens_cache_none=0.4,
        output_tokens=2.4,
        input_tokens_cache_read=0.04,
        input_tokens_cache_write=0.5,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_6_1_SOL = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=2,
        output_tokens=10,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write=2.5,
    ),
    flex=TokenRates(
        input_tokens_cache_none=1,
        output_tokens=5,
        input_tokens_cache_read=0.05,
        input_tokens_cache_write=1.25,
    ),
    fast=TokenRates(
        input_tokens_cache_none=4,
        output_tokens=20,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write=5,
    ),
    ultrafast=TokenRates(
        input_tokens_cache_none=12,
        output_tokens=60,
        input_tokens_cache_read=0.6,
        input_tokens_cache_write=15,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_6_SOL = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=2,
        output_tokens=10,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write=2.5,
    ),
    flex=TokenRates(
        input_tokens_cache_none=1,
        output_tokens=5,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write=1.25,
    ),
    fast=TokenRates(
        input_tokens_cache_none=4,
        output_tokens=20,
        input_tokens_cache_read=0.4,
        input_tokens_cache_write=5,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_6_LUNA = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=0.1,
        output_tokens=0.5,
        input_tokens_cache_read=0.01,
        input_tokens_cache_write=0.125,
    ),
    flex=TokenRates(
        input_tokens_cache_none=0.05,
        output_tokens=0.25,
        input_tokens_cache_read=0.005,
        input_tokens_cache_write=0.0625,
    ),
    fast=TokenRates(
        input_tokens_cache_none=0.2,
        output_tokens=1,
        input_tokens_cache_read=0.02,
        input_tokens_cache_write=0.25,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

_GPT_6_ASTRA = OpenAIPricingTable(
    default=TokenRates(
        input_tokens_cache_none=10,
        output_tokens=50,
        input_tokens_cache_read=1,
        input_tokens_cache_write=12.5,
    ),
    flex=TokenRates(
        input_tokens_cache_none=5,
        output_tokens=25,
        input_tokens_cache_read=0.5,
        input_tokens_cache_write=6.25,
    ),
    fast=TokenRates(
        input_tokens_cache_none=20,
        output_tokens=100,
        input_tokens_cache_read=2,
        input_tokens_cache_write=25,
    ),
    ultrafast=TokenRates(
        input_tokens_cache_none=60,
        output_tokens=300,
        input_tokens_cache_read=6,
        input_tokens_cache_write=75,
    ),
    long_context=LongContextPricing(
        prompt_tokens_above=272000,
        input_multiplier=2.0,
        output_multiplier=1.5,
    ),
    web_search_usd_per_invocation=0.01,
    file_search_usd_per_invocation=0.0025,
)

OPENAI_PRICING: dict[OpenAIModelId, OpenAIPricingTable] = {
    "gpt-5.6": _GPT_5_6_SOL,
    "gpt-5.6-sol": _GPT_5_6_SOL,
    "gpt-5.6-terra": _GPT_5_6_TERRA,
    "gpt-5.6-luna": _GPT_5_6_LUNA,
    "gpt-6.1-sol": _GPT_6_1_SOL,
    "gpt-6-sol": _GPT_6_SOL,
    "gpt-6-luna": _GPT_6_LUNA,
    "gpt-6-astra": _GPT_6_ASTRA,
}

OPENAI_REGIONAL_PROCESSING_MULTIPLIER: dict[OpenAIModelId, float] = {
    "gpt-5.6": 1.1,
    "gpt-5.6-sol": 1.1,
    "gpt-5.6-terra": 1.1,
    "gpt-5.6-luna": 1.1,
    "gpt-6.1-sol": 1.1,
    "gpt-6-sol": 1.1,
    "gpt-6-luna": 1.1,
    "gpt-6-astra": 1.1,
}
"""Token-rate multiplier for each model on an endpoint that uses regional processing."""

type OpenAIEmbeddingModelId = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
"""Cataloged OpenAI model identifiers accepted by `embedding_model()`."""

OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS: dict[OpenAIEmbeddingModelId, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.1,
}
"""USD per million input tokens of each cataloged OpenAI embedding model.

LiteLLM lists no other nonzero rate that a langchaint embedding request incurs.
"""
