"""Offline tests for the neutral pricing arithmetic: category_cost and Billing.

These import no SDK. A rate table is provider-shaped and lives in the backend subpackage whose
adapter spends it, so what a table's price() produces is tested in the adapter test modules.
"""

import math

import pytest

from langchaint import ZERO_USAGE, Usage, category_cost
from langchaint.pricing import invocation_cost_in_usd
from tests.helpers import stated_billing


def _usage(
    *,
    input_tokens_cache_read: int = 0,
    input_tokens_cache_write: int = 0,
    input_tokens_cache_none: int = 0,
    input_tokens_cache_read_cost_in_usd: float = 0.0,
    input_tokens_cache_write_cost_in_usd: float = 0.0,
    input_tokens_cache_none_cost_in_usd: float = 0.0,
) -> Usage:
    """One response's input side; output cancels out of the cache-savings counterfactual."""
    return Usage(
        input_tokens_cache_read=input_tokens_cache_read,
        input_tokens_cache_write=input_tokens_cache_write,
        input_tokens_cache_none=input_tokens_cache_none,
        output_tokens=50,
        output_tokens_reasoning=0,
        input_tokens_cache_read_cost_in_usd=input_tokens_cache_read_cost_in_usd,
        input_tokens_cache_write_cost_in_usd=input_tokens_cache_write_cost_in_usd,
        input_tokens_cache_none_cost_in_usd=input_tokens_cache_none_cost_in_usd,
        output_tokens_cost_in_usd=0.75,
        provider_executed_tool_cost_in_usd=0.0,
    )


def test_category_cost_is_the_counter_times_the_rate_over_one_million() -> None:
    """One multiplication and one division, with no rounding of its own."""
    assert category_cost(200, 3.0) == 200 * 3.0 / 1e6


def test_a_zero_counter_costs_zero_at_an_unknown_rate() -> None:
    """0 * NaN is NaN, so the zero case is special-cased and a total over it stays a number."""
    assert category_cost(0, math.nan) == 0.0


def test_category_cost_preserves_public_keyword_names() -> None:
    """Public keyword calls retain their documented parameter names."""
    assert category_cost(tokens=200, usd_per_million_tokens=3.0) == 0.0006


def test_invocation_cost_uses_nan_only_for_positive_unpriced_counts() -> None:
    """An unavailable rate affects only positive provider invocation counts."""
    assert invocation_cost_in_usd(0, None) == 0.0
    assert math.isnan(invocation_cost_in_usd(1, None))


@pytest.mark.parametrize("invocations", [-1, True])
def test_invocation_cost_rejects_invalid_counts(invocations: int) -> None:
    """Negative and boolean counts cannot produce costs."""
    with pytest.raises(ValueError, match="nonnegative int"):
        _ = invocation_cost_in_usd(invocations, 0.01)


def test_cache_savings_is_what_the_uncached_counterfactual_would_have_added() -> None:
    """Reads at a tenth of the uncached rate save the nine tenths they did not pay."""
    billing = stated_billing(
        _usage(
            input_tokens_cache_read=1000,
            input_tokens_cache_none=100,
            input_tokens_cache_read_cost_in_usd=1000 * 0.3 / 1e6,
            input_tokens_cache_none_cost_in_usd=100 * 3.0 / 1e6,
        ),
        input_cache_none_usd_per_million_tokens=3.0,
    )
    assert billing.cache_savings_in_usd == pytest.approx(1000 * (3.0 - 0.3) / 1e6)


def test_cache_savings_is_negative_where_the_write_premium_exceeded_the_read_discount() -> None:
    """A response that wrote the cache and read little of it paid more than it would have uncached."""
    billing = stated_billing(
        _usage(
            input_tokens_cache_write=1000,
            input_tokens_cache_none=100,
            input_tokens_cache_write_cost_in_usd=1000 * 3.75 / 1e6,
            input_tokens_cache_none_cost_in_usd=100 * 3.0 / 1e6,
        ),
        input_cache_none_usd_per_million_tokens=3.0,
    )
    assert billing.cache_savings_in_usd == pytest.approx(1000 * (3.0 - 3.75) / 1e6)


def test_cache_savings_is_nan_where_a_nonzero_input_counter_had_no_rate() -> None:
    """An unpriced input category makes the saving unknown rather than understated."""
    billing = stated_billing(
        _usage(input_tokens_cache_read=1000, input_tokens_cache_read_cost_in_usd=math.nan),
        input_cache_none_usd_per_million_tokens=math.nan,
    )
    assert math.isnan(billing.cache_savings_in_usd)


def test_cache_savings_is_zero_where_no_input_was_billed() -> None:
    """With no input tokens there is nothing to have saved, whatever the rate."""
    billing = stated_billing(ZERO_USAGE)
    assert billing.cache_savings_in_usd == 0.0
