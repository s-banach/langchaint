"""Pricing metadata selection and generation tests."""

import math

import pytest

from langchaint.anthropic.messages_adapter import (
    AnthropicPricingTable,
    AnthropicRates,
)
from langchaint.openai.shared import (
    OpenAILongContextPricing,
    OpenAIPricingTable,
    OpenAIRates,
    OpenAIResponsesServiceTier,
)
from scripts.update_pricing_metadata import (
    _LITELLM_FILE,
    ANTHROPIC_ALIASES,
    ANTHROPIC_LITELLM_KEYS,
    ANTHROPIC_OUTPUT_PATH,
    METADATA_PATH,
    OPENAI_ALIASES,
    OPENAI_KNOWN_TIERS,
    OPENAI_LITELLM_KEYS,
    OPENAI_OUTPUT_PATH,
    SNAPSHOT_PATH,
    _anthropic_module,
    _LiteLLMEntry,
    _million_rate,
    _openai_module,
    _openai_rates,
    _ProviderMetadata,
    _selected_entries,
    _snapshot_json,
    _untracked_model_keys,
)


def _sample_openai_rates() -> OpenAIRates:
    return OpenAIRates(
        input_tokens_cache_none=10.0,
        output_tokens=20.0,
        input_tokens_cache_read=1.0,
        input_tokens_cache_write=12.5,
    )


def _openai_table() -> OpenAIPricingTable:
    return OpenAIPricingTable(
        default=_sample_openai_rates(),
        flex=_sample_openai_rates().multiplied(input_multiplier=0.5, output_multiplier=0.5),
        fast=_sample_openai_rates().multiplied(input_multiplier=2.0, output_multiplier=2.0),
        long_context=OpenAILongContextPricing(
            input_tokens_total_above=272_000,
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
    assert base.input_tokens_cache_none == 10.0
    assert long_context.input_tokens_cache_none == 20.0
    assert long_context.input_tokens_cache_read == 2.0
    assert long_context.input_tokens_cache_write == 25.0
    assert long_context.output_tokens == 30.0


def test_openai_modifiers_compose_after_service_tier_selection() -> None:
    """Fast, long-context, and regional modifiers compose."""
    rates = _openai_table().rates_for(
        service_tier="priority",
        input_tokens_total=272_001,
        regional_processing=True,
    )
    assert rates.input_tokens_cache_none == 44.0
    assert rates.output_tokens == 66.0


def test_openai_ultrafast_rates_are_selected() -> None:
    """Keep pricing coverage independent of the installed openai SDK types."""
    ultrafast = _sample_openai_rates().multiplied(input_multiplier=3.0, output_multiplier=3.0)
    rates = OpenAIPricingTable(default=_sample_openai_rates(), ultrafast=ultrafast).rates_for(
        service_tier="ultrafast",
        input_tokens_total=1,
        regional_processing=False,
    )
    assert rates is ultrafast


@pytest.mark.parametrize("service_tier", ["scale", "ultrafast"])
def test_openai_missing_optional_tier_rates_produce_nan(
    service_tier: OpenAIResponsesServiceTier,
) -> None:
    """Missing optional tier rates produce NaN."""
    rates = OpenAIPricingTable(default=_sample_openai_rates()).rates_for(
        service_tier=service_tier,
        input_tokens_total=1,
        regional_processing=False,
    )
    assert math.isnan(rates.input_tokens_cache_none)


def test_openai_missing_regional_rates_produce_nan() -> None:
    """Missing regional rates produce NaN."""
    regional = OpenAIPricingTable(default=_sample_openai_rates()).rates_for(
        service_tier="default",
        input_tokens_total=1,
        regional_processing=True,
    )
    assert math.isnan(regional.output_tokens)


@pytest.mark.parametrize("value", [True, 0, -1])
def test_openai_long_context_rejects_invalid_thresholds(value: int) -> None:
    """Long-context thresholds must be positive integers."""
    with pytest.raises(ValueError, match="input_tokens_total_above"):
        _ = OpenAILongContextPricing(
            input_tokens_total_above=value,
            input_multiplier=2.0,
            output_multiplier=1.5,
        )


def _anthropic_rates() -> AnthropicRates:
    return AnthropicRates(
        input_tokens_cache_none=10.0,
        output_tokens=20.0,
        input_tokens_cache_read=1.0,
        input_tokens_cache_write_5m=12.5,
        input_tokens_cache_write_1h=20.0,
    )


def test_anthropic_geo_and_service_tier_select_rates() -> None:
    """US pricing multiplies the selected service-tier rates."""
    table = AnthropicPricingTable(
        standard=_anthropic_rates(),
        batch=_anthropic_rates().multiplied(0.5),
        inference_geo_us_multiplier=1.1,
    )
    rates = table.rates_for(service_tier="batch", inference_geo="us")
    assert rates.input_tokens_cache_none == 5.5
    assert rates.output_tokens == 11.0
    global_rates = table.rates_for(service_tier=None, inference_geo="global")
    assert global_rates is table.standard


def test_anthropic_missing_modifier_rates_produce_nan() -> None:
    """Missing priority and US rates produce NaN."""
    table = AnthropicPricingTable(standard=_anthropic_rates())
    priority = table.rates_for(service_tier="priority", inference_geo="global")
    assert math.isnan(priority.output_tokens)
    regional = table.rates_for(service_tier="standard", inference_geo="us")
    assert math.isnan(regional.input_tokens_cache_none)


@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_regional_multipliers_must_be_positive_and_finite(value: float) -> None:
    """Both provider tables validate regional multipliers."""
    with pytest.raises(ValueError, match="regional_processing_multiplier"):
        _ = OpenAIPricingTable(
            default=_sample_openai_rates(),
            regional_processing_multiplier=value,
        )
    with pytest.raises(ValueError, match="inference_geo_us_multiplier"):
        _ = AnthropicPricingTable(
            standard=_anthropic_rates(),
            inference_geo_us_multiplier=value,
        )


def test_vendored_inputs_reproduce_generated_modules() -> None:
    """Vendored inputs reproduce the snapshot and both generated modules offline."""
    entries = _selected_entries(_LITELLM_FILE.validate_json(SNAPSHOT_PATH.read_bytes()))
    metadata = _ProviderMetadata.model_validate_json(METADATA_PATH.read_bytes())
    assert _snapshot_json(entries) == SNAPSHOT_PATH.read_text()
    assert _openai_module(entries, metadata) == OPENAI_OUTPUT_PATH.read_text()
    assert _anthropic_module(entries, metadata) == ANTHROPIC_OUTPUT_PATH.read_text()


def test_untracked_model_keys_report_only_new_direct_api_text_models() -> None:
    """Detection skips generated, aliased, ignored, non-text, and non-direct-API entries."""
    raw_entries = _LITELLM_FILE.validate_python({
        "untracked-openai-responses-model": {"litellm_provider": "openai", "mode": "responses"},
        "untracked-anthropic-chat-model": {"litellm_provider": "anthropic", "mode": "chat"},
        next(iter(OPENAI_LITELLM_KEYS.values())): {"litellm_provider": "openai", "mode": "chat"},
        next(iter(OPENAI_ALIASES)): {"litellm_provider": "openai", "mode": "chat"},
        next(iter(ANTHROPIC_LITELLM_KEYS.values())): {
            "litellm_provider": "anthropic",
            "mode": "chat",
        },
        next(iter(ANTHROPIC_ALIASES)): {"litellm_provider": "anthropic", "mode": "chat"},
        "ignored-openai-chat-model": {"litellm_provider": "openai", "mode": "chat"},
        "untracked-openai-image-model": {"litellm_provider": "openai", "mode": "image_generation"},
        "azure/untracked-chat-model": {"litellm_provider": "azure", "mode": "chat"},
        "sample_spec": {"litellm_provider": "one of https://docs.litellm.ai/docs/providers"},
    })
    assert _untracked_model_keys(raw_entries, frozenset({"ignored-openai-chat-model"})) == [
        "untracked-anthropic-chat-model",
        "untracked-openai-responses-model",
    ]


def test_litellm_rates_accept_zero() -> None:
    """LiteLLM rates accept free pricing categories."""
    entry = _LiteLLMEntry.model_validate({"litellm_provider": "openai", "input_cost_per_token": 0})
    assert (
        _million_rate(entry.openai_rate_fields("default").input_tokens_cache_none_usd_per_token)
        == "0"
    )


def test_anthropic_batch_rates_follow_listed_batch_rates() -> None:
    """Listed batch rates are used, and the 1-hour write takes the 5-minute write's discount."""
    entry = _LiteLLMEntry.model_validate({
        "litellm_provider": "anthropic",
        "cache_creation_input_token_cost": 2.5e-06,
        "cache_creation_input_token_cost_above_1hr": 4e-06,
        "input_cost_per_token_batches": 1.2e-06,
        "output_cost_per_token_batches": 6e-06,
        "cache_read_input_token_cost_batches": 1.2e-07,
        "cache_creation_input_token_cost_batches": 1.5e-06,
    })
    batch = entry.anthropic_batch_rate_fields()
    assert [_million_rate(rate) for rate in batch] == ["1.2", "6", "0.12", "1.5", "2.4"]


def test_rates_for_an_unknown_tier_are_rejected() -> None:
    """A tier the generated tables do not render fails the refresh instead of pricing as NaN."""
    entry = _LiteLLMEntry.model_validate({
        "litellm_provider": "openai",
        "input_cost_per_token_above_272k_tokens_scale": 1e-05,
    })
    with pytest.raises(ValueError, match=r"unknown pricing tiers \['scale'\]"):
        entry.require_known_tiers(OPENAI_KNOWN_TIERS)


def test_partially_listed_tier_is_rejected() -> None:
    """A tier with some rates unlisted fails the refresh instead of pricing as NaN."""
    fields = _LiteLLMEntry.model_validate({
        "litellm_provider": "openai",
        "input_cost_per_token_flex": 1e-06,
    }).openai_rate_fields("flex")
    with pytest.raises(ValueError, match="unlisted"):
        _ = _openai_rates(fields, field_name="flex", indent="")


def test_tier_long_context_rates_must_follow_the_default_multipliers() -> None:
    """A tier whose long-context markup differs from the default tier's fails the refresh."""
    entry = _LiteLLMEntry.model_validate({
        "litellm_provider": "openai",
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
        "input_cost_per_token_above_272k_tokens": 2e-06,
        "output_cost_per_token_above_272k_tokens": 3e-06,
        "input_cost_per_token_flex": 5e-07,
        "input_cost_per_token_above_272k_tokens_flex": 1.5e-06,
    })
    with pytest.raises(ValueError, match="input_cost_per_token_above_272k_tokens_flex"):
        entry.require_shared_long_context()


@pytest.mark.parametrize(
    "prices",
    [
        {"search_context_size_low": 0.01},
        {
            "search_context_size_low": 0.01,
            "search_context_size_medium": 0.01,
            "search_context_size_high": 0.02,
        },
    ],
)
def test_web_search_price_requires_one_price_for_every_context_size(
    prices: dict[str, float],
) -> None:
    """A partially listed or size-dependent price cannot fill the table's single price."""
    entry = _LiteLLMEntry.model_validate({
        "litellm_provider": "anthropic",
        "search_context_cost_per_query": prices,
    })
    with pytest.raises(ValueError, match="web search prices"):
        _ = entry.web_search_usd_per_invocation()


@pytest.mark.parametrize(
    "multipliers",
    [
        {},
        {"regional_processing_uplift_multiplier_us": 1.1},
        {
            "regional_processing_uplift_multiplier_us": 1.1,
            "regional_processing_uplift_multiplier_eu": 1.2,
        },
    ],
)
def test_openai_regional_multiplier_requires_one_listed_value(
    multipliers: dict[str, float],
) -> None:
    """An unlisted or region-dependent multiplier cannot fill the table's single multiplier."""
    entry = _LiteLLMEntry.model_validate({"litellm_provider": "openai", **multipliers})
    with pytest.raises(ValueError, match="regional processing multipliers"):
        _ = entry.regional_processing_multiplier()
