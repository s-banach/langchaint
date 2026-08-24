"""Test provider-neutral pricing arithmetic without SDK imports."""

import inspect
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
    """One response's input side. Output cancels out of the cache-savings counterfactual."""
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


def test_a_zero_counter_costs_zero_at_an_unknown_rate() -> None:
    """0 * NaN is NaN, so the zero case is special-cased and a total over it stays a number."""
    assert category_cost(tokens=0, usd_per_million_tokens=math.nan) == 0.0


def test_rate_parameters_are_keyword_only() -> None:
    """Both pricing functions require the rate by its unit-bearing parameter name."""
    for function, parameter_name in (
        (category_cost, "usd_per_million_tokens"),
        (invocation_cost_in_usd, "usd_per_invocation"),
    ):
        assert (
            inspect.signature(function).parameters[parameter_name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


def test_invocation_cost_uses_nan_only_for_positive_unpriced_counts() -> None:
    """An unavailable rate affects only positive provider invocation counts."""
    assert invocation_cost_in_usd(0, usd_per_invocation=None) == 0.0
    assert math.isnan(invocation_cost_in_usd(1, usd_per_invocation=None))


@pytest.mark.parametrize("invocations", [-1, True])
def test_invocation_cost_rejects_invalid_counts(invocations: int) -> None:
    """Negative and boolean counts cannot produce costs."""
    with pytest.raises(ValueError, match="nonnegative int"):
        _ = invocation_cost_in_usd(invocations, usd_per_invocation=0.01)


def test_cache_savings_negative_nan_and_zero_boundaries() -> None:
    """Cover negative, NaN, and zero cache savings."""
    for billing, expected in (
        (
            stated_billing(
                _usage(
                    input_tokens_cache_read=100,
                    input_tokens_cache_write=1000,
                    input_tokens_cache_none=100,
                    input_tokens_cache_read_cost_in_usd=100 * 0.3 / 1e6,
                    input_tokens_cache_write_cost_in_usd=1000 * 3.75 / 1e6,
                    input_tokens_cache_none_cost_in_usd=100 * 3.0 / 1e6,
                ),
                input_cache_none_usd_per_million_tokens=3.0,
            ),
            -0.00048,
        ),
        (
            stated_billing(
                _usage(
                    input_tokens_cache_read=1000,
                    input_tokens_cache_read_cost_in_usd=math.nan,
                ),
                input_cache_none_usd_per_million_tokens=math.nan,
            ),
            math.nan,
        ),
        (stated_billing(ZERO_USAGE), 0.0),
    ):
        assert billing.cache_savings_in_usd == pytest.approx(expected, nan_ok=True)
