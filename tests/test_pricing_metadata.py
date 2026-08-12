"""Pricing metadata selection and generation tests."""

import json
import math
from pathlib import Path

import pytest

from langchaint.anthropic import ANTHROPIC_BEDROCK_PRICING, ANTHROPIC_PRICING
from langchaint.anthropic.messages_adapter import (
    AnthropicPricingTable,
    AnthropicRates,
)
from langchaint.openai import OPENAI_PRICING
from langchaint.openai.shared import (
    OpenAILongContextPricing,
    OpenAIPricingTable,
    OpenAIRates,
)
from scripts.update_pricing_metadata import (
    ANTHROPIC_OUTPUT_PATH,
    METADATA_PATH,
    OPENAI_OUTPUT_PATH,
    SNAPSHOT_PATH,
    _anthropic_module,
    _metadata_items,
    _metadata_multiplier,
    _million_rate,
    _openai_module,
    _ProviderMetadata,
    _required_dict,
    _typed_entries,
)


def _openai_rates() -> OpenAIRates:
    return OpenAIRates(
        input_cache_none_usd_per_million_tokens=10.0,
        output_usd_per_million_tokens=20.0,
        cache_read_usd_per_million_tokens=1.0,
        cache_write_usd_per_million_tokens=12.5,
    )


def _openai_table() -> OpenAIPricingTable:
    return OpenAIPricingTable(
        default=_openai_rates(),
        flex=_openai_rates().multiplied(input_multiplier=0.5, output_multiplier=0.5),
        fast=_openai_rates().multiplied(input_multiplier=2.0, output_multiplier=2.0),
        long_context=OpenAILongContextPricing(
            input_tokens_above=272_000,
            input_multiplier=2.0,
            output_multiplier=1.5,
        ),
        regional_processing_multiplier=1.1,
    )


def test_openai_long_context_starts_above_the_threshold() -> None:
    """The threshold remains inclusive of base rates."""
    table = _openai_table()
    base = table.rates_for(
        service_tier="default",
        input_tokens_total=272_000,
        regional_processing=False,
    )
    long_context = table.rates_for(
        service_tier="default",
        input_tokens_total=272_001,
        regional_processing=False,
    )
    assert base.input_cache_none_usd_per_million_tokens == 10.0
    assert long_context.input_cache_none_usd_per_million_tokens == 20.0
    assert long_context.cache_read_usd_per_million_tokens == 2.0
    assert long_context.cache_write_usd_per_million_tokens == 25.0
    assert long_context.output_usd_per_million_tokens == 30.0


def test_openai_modifiers_compose_after_service_tier_selection() -> None:
    """Fast, long-context, and regional modifiers compose."""
    rates = _openai_table().rates_for(
        service_tier="priority",
        input_tokens_total=272_001,
        regional_processing=True,
    )
    assert rates.input_cache_none_usd_per_million_tokens == 44.0
    assert rates.output_usd_per_million_tokens == 66.0


def test_openai_missing_optional_rates_produce_nan() -> None:
    """Missing scale and regional rates produce NaN."""
    rates = OpenAIPricingTable(default=_openai_rates()).rates_for(
        service_tier="scale",
        input_tokens_total=1,
        regional_processing=False,
    )
    assert math.isnan(rates.input_cache_none_usd_per_million_tokens)
    regional = OpenAIPricingTable(default=_openai_rates()).rates_for(
        service_tier="default",
        input_tokens_total=1,
        regional_processing=True,
    )
    assert math.isnan(regional.output_usd_per_million_tokens)


@pytest.mark.parametrize("value", [True, 0, -1])
def test_openai_long_context_rejects_invalid_thresholds(value: int) -> None:
    """Long-context thresholds must be positive integers."""
    with pytest.raises(ValueError, match="input_tokens_above"):
        _ = OpenAILongContextPricing(
            input_tokens_above=value,
            input_multiplier=2.0,
            output_multiplier=1.5,
        )


def _anthropic_rates() -> AnthropicRates:
    return AnthropicRates(
        input_cache_none_usd_per_million_tokens=10.0,
        output_usd_per_million_tokens=20.0,
        cache_read_usd_per_million_tokens=1.0,
        cache_write_5m_usd_per_million_tokens=12.5,
        cache_write_1h_usd_per_million_tokens=20.0,
    )


def test_anthropic_geo_and_service_tier_select_rates() -> None:
    """US pricing multiplies the selected service-tier rates."""
    table = AnthropicPricingTable(
        standard=_anthropic_rates(),
        batch=_anthropic_rates().multiplied(0.5),
        inference_geo_us_multiplier=1.1,
    )
    rates = table.rates_for(service_tier="batch", inference_geo="us")
    assert rates.input_cache_none_usd_per_million_tokens == 5.5
    assert rates.output_usd_per_million_tokens == 11.0
    global_rates = table.rates_for(service_tier=None, inference_geo="global")
    assert global_rates is table.standard


def test_anthropic_missing_modifier_rates_produce_nan() -> None:
    """Missing priority and US rates produce NaN."""
    table = AnthropicPricingTable(standard=_anthropic_rates())
    priority = table.rates_for(service_tier="priority", inference_geo="global")
    assert math.isnan(priority.output_usd_per_million_tokens)
    regional = table.rates_for(service_tier="standard", inference_geo="us")
    assert math.isnan(regional.input_cache_none_usd_per_million_tokens)


@pytest.mark.parametrize("value", [True, 0.0, -1.0, math.nan, math.inf])
def test_regional_multipliers_must_be_positive_and_finite(value: float) -> None:
    """Both provider tables validate regional multipliers."""
    with pytest.raises(ValueError, match="regional_processing_multiplier"):
        _ = OpenAIPricingTable(
            default=_openai_rates(),
            regional_processing_multiplier=value,
        )
    with pytest.raises(ValueError, match="inference_geo_us_multiplier"):
        _ = AnthropicPricingTable(
            standard=_anthropic_rates(),
            inference_geo_us_multiplier=value,
        )


def test_catalog_aliases_share_canonical_tables() -> None:
    """Each alias references its canonical pricing table."""
    assert OPENAI_PRICING["gpt-5.6"] is OPENAI_PRICING["gpt-5.6-sol"]
    assert ANTHROPIC_PRICING["claude-haiku-4-5"] is ANTHROPIC_PRICING["claude-haiku-4-5-20251001"]


def test_bedrock_pricing_uses_bedrock_entries() -> None:
    """Regional Bedrock rates remain independent from direct rates."""
    bedrock = ANTHROPIC_BEDROCK_PRICING["us.anthropic.claude-opus-4-6-v1"]
    direct = ANTHROPIC_PRICING["claude-opus-5"]
    assert bedrock.standard.input_cache_none_usd_per_million_tokens == 5.5
    assert direct.standard.input_cache_none_usd_per_million_tokens == 5.0


def test_vendored_inputs_reproduce_generated_modules() -> None:
    """Vendored inputs reproduce both generated modules offline."""
    snapshot_payload: object = json.loads(SNAPSHOT_PATH.read_text())
    entries = _typed_entries(snapshot_payload)
    metadata_payload: object = json.loads(METADATA_PATH.read_text())
    raw_metadata = _required_dict(metadata_payload, "provider metadata")
    metadata: _ProviderMetadata = {
        "anthropic": _metadata_items(raw_metadata.get("anthropic"), "anthropic metadata"),
        "openai": _metadata_items(raw_metadata.get("openai"), "openai metadata"),
    }
    assert _openai_module(entries, metadata) == OPENAI_OUTPUT_PATH.read_text()
    assert _anthropic_module(entries, metadata) == ANTHROPIC_OUTPUT_PATH.read_text()


def test_generated_paths_are_inside_the_repository() -> None:
    """Generation targets tracked repository paths."""
    repository = Path(__file__).resolve().parents[1]
    assert OPENAI_OUTPUT_PATH.is_relative_to(repository)
    assert ANTHROPIC_OUTPUT_PATH.is_relative_to(repository)


@pytest.mark.parametrize("value", [True, -1, math.nan, math.inf])
def test_litellm_rates_must_be_nonnegative_and_finite(value: float) -> None:
    """LiteLLM rates reject unusable numeric values."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _ = _million_rate({"input_cost_per_token": value}, "input_cost_per_token")


def test_litellm_rates_accept_zero() -> None:
    """LiteLLM rates accept free pricing categories."""
    assert _million_rate({"input_cost_per_token": 0}, "input_cost_per_token") == "0"


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_provider_metadata_must_be_positive_and_finite(value: float) -> None:
    """Provider metadata rejects unusable numeric values."""
    metadata: _ProviderMetadata = {
        "anthropic": {},
        "openai": {
            "regional_processing_multiplier": {
                "source_url": "https://example.com",
                "value": value,
                "verified_on": "2026-08-11",
            }
        },
    }
    with pytest.raises(ValueError, match="is invalid"):
        _ = _metadata_multiplier(metadata, "openai", "regional_processing_multiplier")


def test_ci_keeps_event_triggers_and_adds_weekly_schedule() -> None:
    """CI retains events and adds weekly dependency testing."""
    repository = Path(__file__).resolve().parents[1]
    workflow = (repository / ".github/workflows/ci.yml").read_text()
    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert 'cron: "17 9 * * 1"' in workflow


def test_refresh_workflow_creates_reviewable_pull_requests() -> None:
    """Pricing refreshes use unique branches and preserve failed CI results."""
    repository = Path(__file__).resolve().parents[1]
    workflow = (repository / ".github/workflows/refresh_pricing_metadata.yml").read_text()
    body = (repository / ".github/pricing-refresh-body.md").read_text()
    assert 'cron: "23 9 1 * *"' in workflow
    assert "continue-on-error: true" in workflow
    assert "pricing-metadata-refresh-${{ github.run_id }}" in workflow
    assert "--force" not in workflow
    assert "Check Anthropic Bedrock model mappings." in body
