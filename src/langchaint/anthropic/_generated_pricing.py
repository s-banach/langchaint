"""Generated Anthropic pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""

from typing import Literal

from langchaint.anthropic.messages_adapter import (
    AnthropicPricingTable,
    AnthropicRates,
)

type AnthropicModelName = Literal[
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
]

_CLAUDE_FABLE_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_cache_none_usd_per_million_tokens=10,
        output_usd_per_million_tokens=50,
        cache_read_usd_per_million_tokens=1,
        cache_write_5m_usd_per_million_tokens=12.5,
        cache_write_1h_usd_per_million_tokens=20,
    ),
    batch=AnthropicRates(
        input_cache_none_usd_per_million_tokens=5,
        output_usd_per_million_tokens=25,
        cache_read_usd_per_million_tokens=0.5,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_OPUS_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_cache_none_usd_per_million_tokens=5,
        output_usd_per_million_tokens=25,
        cache_read_usd_per_million_tokens=0.5,
        cache_write_5m_usd_per_million_tokens=6.25,
        cache_write_1h_usd_per_million_tokens=10,
    ),
    batch=AnthropicRates(
        input_cache_none_usd_per_million_tokens=2.5,
        output_usd_per_million_tokens=12.5,
        cache_read_usd_per_million_tokens=0.25,
        cache_write_5m_usd_per_million_tokens=3.125,
        cache_write_1h_usd_per_million_tokens=5,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_SONNET_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_cache_none_usd_per_million_tokens=2,
        output_usd_per_million_tokens=10,
        cache_read_usd_per_million_tokens=0.2,
        cache_write_5m_usd_per_million_tokens=2.5,
        cache_write_1h_usd_per_million_tokens=4,
    ),
    batch=AnthropicRates(
        input_cache_none_usd_per_million_tokens=1,
        output_usd_per_million_tokens=5,
        cache_read_usd_per_million_tokens=0.1,
        cache_write_5m_usd_per_million_tokens=1.25,
        cache_write_1h_usd_per_million_tokens=2,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_HAIKU_4_5_20251001 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_cache_none_usd_per_million_tokens=1,
        output_usd_per_million_tokens=5,
        cache_read_usd_per_million_tokens=0.1,
        cache_write_5m_usd_per_million_tokens=1.25,
        cache_write_1h_usd_per_million_tokens=2,
    ),
    batch=AnthropicRates(
        input_cache_none_usd_per_million_tokens=0.5,
        output_usd_per_million_tokens=2.5,
        cache_read_usd_per_million_tokens=0.05,
        cache_write_5m_usd_per_million_tokens=0.625,
        cache_write_1h_usd_per_million_tokens=1,
    ),
    inference_geo_us_multiplier=None,
    web_search_usd_per_invocation=0.01,
)

ANTHROPIC_PRICING: dict[AnthropicModelName, AnthropicPricingTable] = {
    "claude-fable-5": _CLAUDE_FABLE_5,
    "claude-opus-5": _CLAUDE_OPUS_5,
    "claude-sonnet-5": _CLAUDE_SONNET_5,
    "claude-haiku-4-5-20251001": _CLAUDE_HAIKU_4_5_20251001,
    "claude-haiku-4-5": _CLAUDE_HAIKU_4_5_20251001,
}

ANTHROPIC_BEDROCK_PRICING: dict[str, AnthropicPricingTable] = {
    "anthropic.claude-fable-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=10,
            output_usd_per_million_tokens=50,
            cache_read_usd_per_million_tokens=1,
            cache_write_5m_usd_per_million_tokens=12.5,
            cache_write_1h_usd_per_million_tokens=20,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-opus-4-8": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=5,
            output_usd_per_million_tokens=25,
            cache_read_usd_per_million_tokens=0.5,
            cache_write_5m_usd_per_million_tokens=6.25,
            cache_write_1h_usd_per_million_tokens=10,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-opus-4-7": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=5,
            output_usd_per_million_tokens=25,
            cache_read_usd_per_million_tokens=0.5,
            cache_write_5m_usd_per_million_tokens=6.25,
            cache_write_1h_usd_per_million_tokens=10,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-sonnet-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=2,
            output_usd_per_million_tokens=10,
            cache_read_usd_per_million_tokens=0.2,
            cache_write_5m_usd_per_million_tokens=2.5,
            cache_write_1h_usd_per_million_tokens=4,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-haiku-4-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=1,
            output_usd_per_million_tokens=5,
            cache_read_usd_per_million_tokens=0.1,
            cache_write_5m_usd_per_million_tokens=1.25,
            cache_write_1h_usd_per_million_tokens=2,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "us.anthropic.claude-opus-4-6-v1": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=5.5,
            output_usd_per_million_tokens=27.5,
            cache_read_usd_per_million_tokens=0.55,
            cache_write_5m_usd_per_million_tokens=6.875,
            cache_write_1h_usd_per_million_tokens=11,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "us.anthropic.claude-sonnet-4-6": AnthropicPricingTable(
        standard=AnthropicRates(
            input_cache_none_usd_per_million_tokens=3.3,
            output_usd_per_million_tokens=16.5,
            cache_read_usd_per_million_tokens=0.33,
            cache_write_5m_usd_per_million_tokens=4.125,
            cache_write_1h_usd_per_million_tokens=6.6,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
}
