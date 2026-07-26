"""Offline tests for the neutral pricing arithmetic in pricing.py.

These import no SDK: PricingTable.price consumes already-split counters,
and the per-backend extraction from raw SDK usage is tested in the adapter test modules.
"""

import math

from langchaint import PricingTable, Usage
from langchaint.pricing import UNPRICED, category_cost

_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=3.0,
    output_usd_per_million_tokens=15.0,
    cache_read_usd_per_million_tokens=0.3,
    cache_write_usd_per_million_tokens=3.75,
)


def _priced(pricing: PricingTable) -> Usage:
    """One response's counters priced at the given table."""
    return pricing.price(
        input_tokens_cache_read=200,
        input_tokens_cache_write=10,
        input_tokens_cache_none=100,
        output_tokens=50,
        output_tokens_reasoning=20,
    )


def test_price_computes_one_product_per_category() -> None:
    """Each cost field is its counter times its rate over one million."""
    usage = _priced(_PRICING)
    assert usage.input_tokens_cache_read_cost_in_usd == 200 * 0.3 / 1e6
    assert usage.input_tokens_cache_write_cost_in_usd == 10 * 3.75 / 1e6
    assert usage.input_tokens_cache_none_cost_in_usd == 100 * 3.0 / 1e6
    assert usage.output_tokens_cost_in_usd == 50 * 15.0 / 1e6


def test_price_carries_the_counters_through() -> None:
    """The returned Usage reports the counters it was given, reasoning included."""
    usage = _priced(_PRICING)
    assert usage.input_tokens_cache_read == 200
    assert usage.input_tokens_cache_write == 10
    assert usage.input_tokens_cache_none == 100
    assert usage.output_tokens == 50
    assert usage.output_tokens_reasoning == 20


def test_cost_in_usd_is_the_sum_of_the_four_categories() -> None:
    """The priced total is exactly what the parts add to."""
    usage = _priced(_PRICING)
    assert usage.cost_in_usd == (
        usage.input_tokens_cache_read_cost_in_usd
        + usage.input_tokens_cache_write_cost_in_usd
        + usage.input_tokens_cache_none_cost_in_usd
        + usage.output_tokens_cost_in_usd
    )


def test_unpriced_makes_every_nonzero_category_nan() -> None:
    """A response at a tier with no table keeps its counters and costs NaN."""
    usage = _priced(UNPRICED)
    assert usage.input_tokens_cache_read == 200
    assert math.isnan(usage.input_tokens_cache_read_cost_in_usd)
    assert math.isnan(usage.input_tokens_cache_write_cost_in_usd)
    assert math.isnan(usage.input_tokens_cache_none_cost_in_usd)
    assert math.isnan(usage.output_tokens_cost_in_usd)
    assert math.isnan(usage.cost_in_usd)


def test_a_zero_counter_costs_zero_at_an_unknown_rate() -> None:
    """0 * NaN is NaN, so the zero case is special-cased and the total stays a number."""
    assert category_cost(0, float("nan")) == 0.0
    usage = UNPRICED.price(
        input_tokens_cache_read=0,
        input_tokens_cache_write=0,
        input_tokens_cache_none=0,
        output_tokens=0,
        output_tokens_reasoning=0,
    )
    assert usage.cost_in_usd == 0.0
