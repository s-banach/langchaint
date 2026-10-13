"""Pricing metadata selection and generation tests."""

import math
from typing import Literal

import pytest

from langchaint.anthropic.messages_adapter import (
    AnthropicPricingTable,
    AnthropicRates,
)
from langchaint.billing.pricing import LongContextPricing, TokenRates
from langchaint.openai.shared import OpenAIPricingTable
from scripts.update_pricing_metadata import (
    _LITELLM_FILE,
    ANTHROPIC_ALIASES,
    ANTHROPIC_LITELLM_KEYS,
    ANTHROPIC_OUTPUT_PATH,
    COHERE_BEDROCK_EMBED_V4_LITELLM_KEYS,
    COHERE_OUTPUT_PATH,
    METADATA_PATH,
    OPENAI_ALIASES,
    OPENAI_EMBEDDING_LITELLM_KEYS,
    OPENAI_KNOWN_TIERS,
    OPENAI_LITELLM_KEYS,
    OPENAI_OUTPUT_PATH,
    OPENAI_RENDERED_TIERS,
    SNAPSHOT_PATH,
    _anthropic_module,
    _cohere_module,
    _cross_region_multiplier,
    _embedding_rate_ratios,
    _LiteLLMEntry,
    _million_rate,
    _openai_module,
    _openai_rates,
    _ProviderMetadata,
    _selected_entries,
    _snapshot_json,
    _untracked_model_keys,
)


def _sample_openai_rates(multiplier: float = 1.0) -> TokenRates:
    return TokenRates(
        input_tokens_cache_none=10.0 * multiplier,
        output_tokens=20.0 * multiplier,
        input_tokens_cache_read=1.0 * multiplier,
        input_tokens_cache_write=12.5 * multiplier,
    )


def _openai_table() -> OpenAIPricingTable:
    return OpenAIPricingTable(
        default=_sample_openai_rates(),
        flex=_sample_openai_rates(0.5),
        fast=_sample_openai_rates(2.0),
        long_context=LongContextPricing(
            prompt_tokens_above=272_000,
            input_multiplier=2.0,
            output_multiplier=1.5,
        ),
    )


def test_openai_ultrafast_rates_are_selected() -> None:
    """Keep pricing coverage independent of the installed openai SDK types."""
    ultrafast = _sample_openai_rates(3.0)
    rates = OpenAIPricingTable(default=_sample_openai_rates(), ultrafast=ultrafast).rates_for(
        service_tier="ultrafast", input_tokens_total=1
    )
    assert rates is ultrafast


@pytest.mark.parametrize("service_tier", ["scale", "ultrafast"])
def test_openai_missing_optional_tier_rates_produce_nan(
    service_tier: Literal["scale", "ultrafast"],
) -> None:
    """Missing optional tier rates produce NaN."""
    rates = OpenAIPricingTable(default=_sample_openai_rates()).rates_for(
        service_tier=service_tier, input_tokens_total=1
    )
    assert math.isnan(rates.input_tokens_cache_none)


def _anthropic_rates() -> AnthropicRates:
    return AnthropicRates(
        input_tokens_cache_none=10.0,
        output_tokens=20.0,
        input_tokens_cache_read=1.0,
        input_tokens_cache_write_5m=12.5,
        input_tokens_cache_write_1h=20.0,
    )


def _anthropic_rates_for(
    table: AnthropicPricingTable,
    *,
    service_tier: Literal["standard", "priority", "batch"],
    inference_geo: str | None,
    input_tokens_total: int = 0,
) -> TokenRates:
    """Select rates for a response that wrote no cache, so the five-minute write rate is reported."""
    return table.rates_for(
        service_tier=service_tier,
        inference_geo=inference_geo,
        input_tokens_cache_write_5m=0,
        input_tokens_cache_write_1h=0,
        input_tokens_total=input_tokens_total,
    )


def test_anthropic_geo_and_service_tier_select_rates() -> None:
    """US pricing multiplies the selected service-tier rates."""
    table = AnthropicPricingTable(
        standard=_anthropic_rates(),
        batch=_anthropic_rates().scaled(0.5),
        inference_geo_us_multiplier=1.1,
    )
    rates = _anthropic_rates_for(table, service_tier="batch", inference_geo="us")
    assert rates.input_tokens_cache_none == 5.5
    assert rates.output_tokens == 11.0
    global_rates = _anthropic_rates_for(table, service_tier="standard", inference_geo="global")
    assert global_rates.input_tokens_cache_none == 10.0


def test_anthropic_long_context_applies_above_threshold_before_geo() -> None:
    """Long-context multipliers apply above the threshold, and the US multiplier stacks on them."""
    table = AnthropicPricingTable(
        standard=_anthropic_rates(),
        long_context=LongContextPricing(
            prompt_tokens_above=100, input_multiplier=5.0, output_multiplier=2.0
        ),
        inference_geo_us_multiplier=2.0,
    )
    above = _anthropic_rates_for(
        table, service_tier="standard", inference_geo="us", input_tokens_total=101
    )
    assert above == TokenRates(
        input_tokens_cache_none=100.0,
        output_tokens=80.0,
        input_tokens_cache_read=10.0,
        input_tokens_cache_write=125.0,
    )


def test_anthropic_cache_write_rate_is_the_written_shares_mean() -> None:
    """One written share reports its own rate, and both shares report their token-weighted mean."""
    rates = _anthropic_rates()
    only_1h = rates.token_rates(input_tokens_cache_write_5m=0, input_tokens_cache_write_1h=7)
    assert only_1h.input_tokens_cache_write == 20.0
    both = rates.token_rates(input_tokens_cache_write_5m=10, input_tokens_cache_write_1h=30)
    assert both.input_tokens_cache_write == (10 * 12.5 + 30 * 20.0) / 40


def test_anthropic_missing_modifier_rates_produce_nan() -> None:
    """Missing priority and US rates produce NaN."""
    table = AnthropicPricingTable(standard=_anthropic_rates())
    priority = _anthropic_rates_for(table, service_tier="priority", inference_geo="global")
    assert math.isnan(priority.output_tokens)
    regional = _anthropic_rates_for(table, service_tier="standard", inference_geo="us")
    assert math.isnan(regional.input_tokens_cache_none)


def test_vendored_inputs_reproduce_generated_modules() -> None:
    """Vendored inputs reproduce the snapshot and every generated module offline."""
    entries = _selected_entries(_LITELLM_FILE.validate_json(SNAPSHOT_PATH.read_bytes()))
    metadata = _ProviderMetadata.model_validate_json(METADATA_PATH.read_bytes())
    assert _snapshot_json(entries) == SNAPSHOT_PATH.read_text()
    assert _openai_module(entries, metadata) == OPENAI_OUTPUT_PATH.read_text()
    assert _anthropic_module(entries, metadata) == ANTHROPIC_OUTPUT_PATH.read_text()
    assert _cohere_module(entries) == COHERE_OUTPUT_PATH.read_text()


def test_untracked_model_keys_report_only_new_models_of_reported_kinds() -> None:
    """Detection reports new text and embedding models.

    It skips generated, aliased, ignored, and unreported-kind entries.
    It also skips a cross-region key whose unprefixed key is generated or ignored.
    """
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
        "untracked-openai-embedding-model": {"litellm_provider": "openai", "mode": "embedding"},
        OPENAI_EMBEDDING_LITELLM_KEYS[0]: {"litellm_provider": "openai", "mode": "embedding"},
        "cohere.untracked-embed": {"litellm_provider": "bedrock", "mode": "embedding"},
        "eu.cohere.untracked-embed": {"litellm_provider": "bedrock", "mode": "embedding"},
        COHERE_BEDROCK_EMBED_V4_LITELLM_KEYS[0]: {
            "litellm_provider": "bedrock",
            "mode": "embedding",
        },
        f"eu.{COHERE_BEDROCK_EMBED_V4_LITELLM_KEYS[0]}": {
            "litellm_provider": "bedrock",
            "mode": "embedding",
        },
        "amazon.untracked-embed": {"litellm_provider": "bedrock", "mode": "embedding"},
        "cohere.ignored-embed": {"litellm_provider": "bedrock", "mode": "embedding"},
        "us.cohere.ignored-embed": {"litellm_provider": "bedrock", "mode": "embedding"},
        "sample_spec": {"litellm_provider": "one of https://docs.litellm.ai/docs/providers"},
    })
    assert _untracked_model_keys(
        raw_entries, frozenset({"ignored-openai-chat-model", "cohere.ignored-embed"})
    ) == [
        "cohere.untracked-embed",
        "eu.cohere.untracked-embed",
        "untracked-anthropic-chat-model",
        "untracked-openai-embedding-model",
        "untracked-openai-responses-model",
    ]


def _bedrock_embedding_entry(usd_per_token: float) -> _LiteLLMEntry:
    return _LiteLLMEntry.model_validate({
        "litellm_provider": "bedrock",
        "input_cost_per_token": usd_per_token,
    })


def test_cross_region_multiplier_is_the_listed_ratio_or_nan() -> None:
    """A listed prefixed entry gives its rate ratio, and an unlisted prefix gives NaN."""
    entries = {
        "cohere.embed-v4:0": _bedrock_embedding_entry(1e-7),
        "eu.cohere.embed-v4:0": _bedrock_embedding_entry(1.1e-7),
    }
    keys = ["cohere.embed-v4:0"]
    assert _cross_region_multiplier(entries, "eu", keys, _embedding_rate_ratios) == 1.1
    assert math.isnan(_cross_region_multiplier(entries, "jp", keys, _embedding_rate_ratios))


def test_cross_region_multiplier_rejects_differing_ratios() -> None:
    """Prefixed entries whose ratios differ across models have no single multiplier."""
    entries = {
        "cohere.embed-v4:0": _bedrock_embedding_entry(1e-7),
        "eu.cohere.embed-v4:0": _bedrock_embedding_entry(1.1e-7),
        "cohere.embed-english-v3": _bedrock_embedding_entry(1e-7),
        "eu.cohere.embed-english-v3": _bedrock_embedding_entry(1.2e-7),
    }
    keys = ["cohere.embed-v4:0", "cohere.embed-english-v3"]
    with pytest.raises(ValueError, match="no single cross-region multiplier"):
        _ = _cross_region_multiplier(entries, "eu", keys, _embedding_rate_ratios)


def test_litellm_rates_accept_zero() -> None:
    """LiteLLM rates accept free pricing categories."""
    entry = _LiteLLMEntry.model_validate({"litellm_provider": "openai", "input_cost_per_token": 0})
    assert (
        _million_rate(entry.openai_rate_fields("default").input_tokens_cache_none_usd_per_token)
        == "0"
    )


@pytest.mark.parametrize(
    "rates",
    [
        {},
        {"input_cost_per_token": 1e-07, "output_cost_per_token": 1e-07},
        {"input_cost_per_token": 1e-07, "cache_read_input_token_cost": 0},
        {"input_cost_per_token": 1e-07, "regional_processing_uplift_multiplier_us": 1.1},
        {"input_cost_per_token": 1e-07, "input_cost_per_character": 2.5e-08},
    ],
)
def test_embedding_rate_rejects_unrendered_prices(rates: dict[str, float]) -> None:
    """An embedding entry lists an input rate, a zero output rate at most, and no other price except batch rates."""
    entry = _LiteLLMEntry.model_validate({"litellm_provider": "openai", **rates})
    with pytest.raises(ValueError, match="LiteLLM lists"):
        _ = entry.embedding_input_tokens_usd_per_token()


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
        _ = entry.long_context_fields(OPENAI_RENDERED_TIERS)


def test_long_context_multipliers_cover_cache_rates_and_rendered_tiers() -> None:
    """Every input-side rate, including the one-hour cache write, shares one input multiplier across rendered tiers.

    An unrendered tier is skipped.
    """
    entry = _LiteLLMEntry.model_validate({
        "litellm_provider": "anthropic",
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 5e-07,
        "cache_read_input_token_cost": 1e-08,
        "cache_creation_input_token_cost_above_1hr": 2e-07,
        "input_cost_per_token_batches": 5e-08,
        "input_cost_per_token_above_100k_tokens": 5e-07,
        "output_cost_per_token_above_100k_tokens": 2.5e-06,
        "cache_read_input_token_cost_above_100k_tokens": 5e-08,
        "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-06,
        "input_cost_per_token_above_100k_tokens_batches": 2.5e-07,
    })
    assert entry.long_context_fields(frozenset({"batches"})) == (100_000, 5, 5)
    mismatched = entry.model_copy(update={"cache_creation_input_token_cost_above_1hr": 4e-07})
    with pytest.raises(ValueError, match="do not share"):
        _ = mismatched.long_context_fields(frozenset())
    unrendered = entry.model_copy(update={"input_cost_per_token_batches": 1e-07})
    assert unrendered.long_context_fields(frozenset()) == (100_000, 5, 5)
    with pytest.raises(ValueError, match="do not share"):
        _ = unrendered.long_context_fields(frozenset({"batches"}))


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
    """An unlisted or region-dependent multiplier cannot fill the model's single multiplier."""
    entry = _LiteLLMEntry.model_validate({"litellm_provider": "openai", **multipliers})
    with pytest.raises(ValueError, match="regional processing multipliers"):
        _ = entry.regional_processing_multiplier()
