"""Offline tests for the neutral pricing arithmetic in pricing.py.

These import no SDK: price consumes already-split PriceableCounts,
and the per-backend extraction from raw SDK usage is tested in the adapter test modules.
"""

import pytest

from langchaint import PriceableCounts, PricingTable, price

_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_usd_per_million_tokens=3.75,
    cache_write_1h_usd_per_million_tokens=6.0,
)
_PRICING_NO_1H = PricingTable(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_usd_per_million_tokens=3.75,
)

_COUNTS = PriceableCounts(
    input_tokens_cache_none=100,
    input_tokens_cache_read=200,
    input_tokens_cache_write=10,
    input_tokens_cache_write_1h=20,
    output_tokens=50,
)


def test_price_computes_one_product_per_category() -> None:
    """Each cost field is its count times its rate over one million."""
    breakdown = price(counts=_COUNTS, pricing=_PRICING)
    assert breakdown.counts is _COUNTS
    assert breakdown.input_tokens_cache_none_cost_in_usd == 100 * 3.0 / 1e6
    assert breakdown.input_tokens_cache_read_cost_in_usd == 200 * 0.3 / 1e6
    assert breakdown.input_tokens_cache_write_cost_in_usd == 10 * 3.75 / 1e6
    assert breakdown.input_tokens_cache_write_1h_cost_in_usd == 20 * 6.0 / 1e6
    assert breakdown.output_tokens_cost_in_usd == 50 * 15.0 / 1e6


def test_price_parts_sum_to_the_derived_totals() -> None:
    """The four input components sum to input_tokens_cost_in_usd, and the total adds output."""
    breakdown = price(counts=_COUNTS, pricing=_PRICING)
    assert breakdown.input_tokens_cost_in_usd == (
        breakdown.input_tokens_cache_none_cost_in_usd
        + breakdown.input_tokens_cache_read_cost_in_usd
        + breakdown.input_tokens_cache_write_cost_in_usd
        + breakdown.input_tokens_cache_write_1h_cost_in_usd
    )
    assert breakdown.total_cost_in_usd == (
        breakdown.input_tokens_cost_in_usd + breakdown.output_tokens_cost_in_usd
    )


def test_price_raises_value_error_when_one_hour_writes_lack_a_rate() -> None:
    """A 1-hour write count with no 1h rate is the plain built-in ValueError, no batch type."""
    with pytest.raises(ValueError, match="cache_write_1h_usd_per_million_tokens"):
        price(counts=_COUNTS, pricing=_PRICING_NO_1H)


def test_price_needs_no_one_hour_rate_when_the_slot_is_zero() -> None:
    """An all-base-write count (the openai shape) prices against a table without the 1h rate."""
    counts = PriceableCounts(
        input_tokens_cache_none=100,
        input_tokens_cache_read=200,
        input_tokens_cache_write=30,
        input_tokens_cache_write_1h=0,
        output_tokens=50,
    )
    breakdown = price(counts=counts, pricing=_PRICING_NO_1H)
    assert breakdown.input_tokens_cache_write_1h_cost_in_usd == 0.0
    assert breakdown.total_cost_in_usd == (
        100 * 3.0 / 1e6 + 200 * 0.3 / 1e6 + 30 * 3.75 / 1e6 + 50 * 15.0 / 1e6
    )
