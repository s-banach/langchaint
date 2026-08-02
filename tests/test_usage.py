"""Usage: the input partition, the non-negativity guard, the cost total, and folding.

The three input counters are a disjoint partition, so input_tokens_total is their sum.
Every counter is non-negative by validation.
The cost fields carry no such constraint, because an unpriceable category stores NaN there.
Usage.sum_of folds counters and per-category costs, and ZERO_USAGE totals an empty batch.
"""

import math

import pytest
from pydantic import ValidationError

from langchaint import ZERO_USAGE, Usage


def _usage(
    *,
    cache_read: int = 0,
    cache_write: int = 0,
    cache_none: int = 0,
    output: int = 0,
    reasoning: int = 0,
    cache_read_cost: float = 0.0,
    cache_write_cost: float = 0.0,
    cache_none_cost: float = 0.0,
    output_cost: float = 0.0,
) -> Usage:
    """Build a Usage from the fields a test cares about, defaulting the rest to zero."""
    return Usage(
        input_tokens_cache_read=cache_read,
        input_tokens_cache_write=cache_write,
        input_tokens_cache_none=cache_none,
        output_tokens=output,
        output_tokens_reasoning=reasoning,
        input_tokens_cache_read_cost_in_usd=cache_read_cost,
        input_tokens_cache_write_cost_in_usd=cache_write_cost,
        input_tokens_cache_none_cost_in_usd=cache_none_cost,
        output_tokens_cost_in_usd=output_cost,
    )


def test_input_tokens_total_sums_the_partition() -> None:
    """input_tokens_total is the sum of the three disjoint input counters."""
    usage = _usage(cache_read=600, cache_write=100, cache_none=300, output=40)
    assert usage.input_tokens_total == 1000


def test_cost_in_usd_sums_the_four_categories() -> None:
    """cost_in_usd is derived, so it cannot disagree with the parts it adds."""
    usage = _usage(
        cache_read_cost=0.01, cache_write_cost=0.02, cache_none_cost=0.03, output_cost=0.04
    )
    assert usage.cost_in_usd == pytest.approx(0.10)


def test_negative_counter_is_rejected() -> None:
    """A negative counter raises, so no carrier holds a Usage claiming negative tokens."""
    with pytest.raises(ValidationError):
        _ = _usage(cache_read=900, cache_write=200, cache_none=-100, output=40)


def test_nan_cost_is_stored() -> None:
    """The cost fields carry no constraint, so an unpriceable response's NaN survives construction.

    A non-negative constraint rejects NaN, which would fail the whole Usage
    and take the paid response's output down with it.
    """
    usage = _usage(output=40, output_cost=float("nan"))
    assert math.isnan(usage.output_tokens_cost_in_usd)
    assert math.isnan(usage.cost_in_usd)


def test_a_sum_containing_a_nan_cost_is_nan() -> None:
    """One unpriceable response makes the batch total NaN, not a silently low number."""
    total = Usage.sum_of([
        _usage(output=40, output_cost=0.25),
        _usage(output=10, output_cost=float("nan")),
    ])
    assert math.isnan(total.cost_in_usd)
    assert total.output_tokens == 50


def test_sum_of_is_fieldwise_over_counters_and_costs() -> None:
    """Every counter and every category cost folds on its own, over a batch of more than two."""
    first = _usage(
        cache_read=1,
        cache_write=2,
        cache_none=3,
        output=4,
        reasoning=1,
        cache_read_cost=0.01,
        cache_write_cost=0.02,
        cache_none_cost=0.03,
        output_cost=0.04,
    )
    second = _usage(
        cache_read=10,
        cache_write=20,
        cache_none=30,
        output=40,
        reasoning=5,
        cache_read_cost=0.10,
        cache_write_cost=0.20,
        cache_none_cost=0.30,
        output_cost=0.40,
    )
    third = _usage(
        cache_read=100,
        cache_write=200,
        cache_none=300,
        output=400,
        reasoning=50,
        cache_read_cost=1.00,
        cache_write_cost=2.00,
        cache_none_cost=3.00,
        output_cost=4.00,
    )
    total = Usage.sum_of([first, second, third])
    assert total.input_tokens_cache_read == 111
    assert total.input_tokens_cache_write == 222
    assert total.input_tokens_cache_none == 333
    assert total.output_tokens == 444
    assert total.output_tokens_reasoning == 56
    assert total.input_tokens_cache_read_cost_in_usd == pytest.approx(1.11)
    assert total.input_tokens_cache_write_cost_in_usd == pytest.approx(2.22)
    assert total.input_tokens_cache_none_cost_in_usd == pytest.approx(3.33)
    assert total.output_tokens_cost_in_usd == pytest.approx(4.44)
    assert total.cost_in_usd == pytest.approx(11.10)


def test_sum_of_with_zero_usage_changes_nothing() -> None:
    """Folding ZERO_USAGE in leaves the other usage's every field alone."""
    usage = _usage(cache_none=7, output=3, cache_none_cost=0.2, output_cost=0.3)
    assert Usage.sum_of([usage, ZERO_USAGE]) == usage
    assert Usage.sum_of([ZERO_USAGE, usage]) == usage


def test_sum_of_empty_is_zero_usage() -> None:
    """Usage.sum_of over an empty iterable returns ZERO_USAGE."""
    assert Usage.sum_of([]) == ZERO_USAGE
