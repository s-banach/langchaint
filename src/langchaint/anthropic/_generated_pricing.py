"""Generated Anthropic pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""

from typing import Literal

from langchaint.anthropic.messages_adapter import (
    AnthropicPricingTable,
    AnthropicRates,
)

type AnthropicModelId = Literal[
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
]

_CLAUDE_FABLE_5_1 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=10,
        output_tokens=50,
        input_tokens_cache_read=0.25,
        input_tokens_cache_write_5m=12.5,
        input_tokens_cache_write_1h=20,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=5,
        output_tokens=25,
        input_tokens_cache_read=0.125,
        input_tokens_cache_write_5m=6.25,
        input_tokens_cache_write_1h=10,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_FABLE_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=10,
        output_tokens=50,
        input_tokens_cache_read=1,
        input_tokens_cache_write_5m=12.5,
        input_tokens_cache_write_1h=20,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=5,
        output_tokens=25,
        input_tokens_cache_read=0.5,
        input_tokens_cache_write_5m=6.25,
        input_tokens_cache_write_1h=10,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_OPUS_5_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=4,
        output_tokens=20,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write_5m=5,
        input_tokens_cache_write_1h=8,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=2,
        output_tokens=10,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write_5m=2.5,
        input_tokens_cache_write_1h=4,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_OPUS_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=5,
        output_tokens=25,
        input_tokens_cache_read=0.5,
        input_tokens_cache_write_5m=6.25,
        input_tokens_cache_write_1h=10,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=2.5,
        output_tokens=12.5,
        input_tokens_cache_read=0.25,
        input_tokens_cache_write_5m=3.125,
        input_tokens_cache_write_1h=5,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_SONNET_5_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=2,
        output_tokens=10,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write_5m=2.5,
        input_tokens_cache_write_1h=4,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=1,
        output_tokens=5,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write_5m=1.25,
        input_tokens_cache_write_1h=2,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_SONNET_5 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=2,
        output_tokens=10,
        input_tokens_cache_read=0.2,
        input_tokens_cache_write_5m=2.5,
        input_tokens_cache_write_1h=4,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=1,
        output_tokens=5,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write_5m=1.25,
        input_tokens_cache_write_1h=2,
    ),
    inference_geo_us_multiplier=1.1,
    web_search_usd_per_invocation=0.01,
)

_CLAUDE_HAIKU_4_5_20251001 = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=1,
        output_tokens=5,
        input_tokens_cache_read=0.1,
        input_tokens_cache_write_5m=1.25,
        input_tokens_cache_write_1h=2,
    ),
    batch=AnthropicRates(
        input_tokens_cache_none=0.5,
        output_tokens=2.5,
        input_tokens_cache_read=0.05,
        input_tokens_cache_write_5m=0.625,
        input_tokens_cache_write_1h=1,
    ),
    inference_geo_us_multiplier=None,
    web_search_usd_per_invocation=None,
)

ANTHROPIC_PRICING: dict[AnthropicModelId, AnthropicPricingTable] = {
    "claude-fable-5-1": _CLAUDE_FABLE_5_1,
    "claude-fable-5": _CLAUDE_FABLE_5,
    "claude-opus-5-5": _CLAUDE_OPUS_5_5,
    "claude-opus-5": _CLAUDE_OPUS_5,
    "claude-sonnet-5-5": _CLAUDE_SONNET_5_5,
    "claude-sonnet-5": _CLAUDE_SONNET_5,
    "claude-haiku-4-5-20251001": _CLAUDE_HAIKU_4_5_20251001,
    "claude-haiku-4-5": _CLAUDE_HAIKU_4_5_20251001,
}

ANTHROPIC_BEDROCK_PRICING: dict[str, AnthropicPricingTable] = {
    "anthropic.claude-fable-5-1": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=10,
            output_tokens=50,
            input_tokens_cache_read=0.25,
            input_tokens_cache_write_5m=12.5,
            input_tokens_cache_write_1h=20,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-fable-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=10,
            output_tokens=50,
            input_tokens_cache_read=1,
            input_tokens_cache_write_5m=12.5,
            input_tokens_cache_write_1h=20,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-opus-5-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=4,
            output_tokens=20,
            input_tokens_cache_read=0.2,
            input_tokens_cache_write_5m=5,
            input_tokens_cache_write_1h=8,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-opus-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=5,
            output_tokens=25,
            input_tokens_cache_read=0.5,
            input_tokens_cache_write_5m=6.25,
            input_tokens_cache_write_1h=10,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-opus-4-8": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=5,
            output_tokens=25,
            input_tokens_cache_read=0.5,
            input_tokens_cache_write_5m=6.25,
            input_tokens_cache_write_1h=10,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-opus-4-7": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=5,
            output_tokens=25,
            input_tokens_cache_read=0.5,
            input_tokens_cache_write_5m=6.25,
            input_tokens_cache_write_1h=10,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-sonnet-5-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=2,
            output_tokens=10,
            input_tokens_cache_read=0.2,
            input_tokens_cache_write_5m=2.5,
            input_tokens_cache_write_1h=4,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-sonnet-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=2,
            output_tokens=10,
            input_tokens_cache_read=0.2,
            input_tokens_cache_write_5m=2.5,
            input_tokens_cache_write_1h=4,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "anthropic.claude-haiku-4-5": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=1,
            output_tokens=5,
            input_tokens_cache_read=0.1,
            input_tokens_cache_write_5m=1.25,
            input_tokens_cache_write_1h=2,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=None,
    ),
    "us.anthropic.claude-opus-4-6-v1": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=5.5,
            output_tokens=27.5,
            input_tokens_cache_read=0.55,
            input_tokens_cache_write_5m=6.875,
            input_tokens_cache_write_1h=11,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
    "us.anthropic.claude-sonnet-4-6": AnthropicPricingTable(
        standard=AnthropicRates(
            input_tokens_cache_none=3.3,
            output_tokens=16.5,
            input_tokens_cache_read=0.33,
            input_tokens_cache_write_5m=4.125,
            input_tokens_cache_write_1h=6.6,
        ),
        inference_geo_us_multiplier=None,
        web_search_usd_per_invocation=0.01,
    ),
}

BEDROCK_CROSS_REGION_MULTIPLIER: dict[str, float] = {
    "us": 1.1,
    "eu": 1.1,
    "au": 1.1,
    "jp": 1.1,
    "apac": 1.1,
    "global": 1.0,
    "us-gov": 1.2,
}
"""Token-rate multiplier for each Bedrock cross-region inference profile prefix."""
