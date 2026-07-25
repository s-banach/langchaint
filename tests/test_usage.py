"""Usage: the input partition, the non-negativity guard, and addition.

The three input counters are a disjoint partition, so input_tokens_total is their sum.
Every counter is non-negative by validation, which the openai adapter's subtraction relies on;
cost_in_usd carries no such constraint, because an unpriceable category stores NaN there.
Usage is summable: __add__ folds counters and cost fieldwise, and Usage.sum_of / ZERO_USAGE total a batch.
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
    cost: float = 0.0,
) -> Usage:
    """Build a Usage from the fields a test cares about, defaulting the rest to zero."""
    return Usage(
        input_tokens_cache_read=cache_read,
        input_tokens_cache_write=cache_write,
        input_tokens_cache_none=cache_none,
        output_tokens=output,
        output_tokens_reasoning=reasoning,
        cost_in_usd=cost,
    )


def test_input_tokens_total_sums_the_partition() -> None:
    """input_tokens_total is the sum of the three disjoint input counters."""
    usage = _usage(cache_read=600, cache_write=100, cache_none=300, output=40)
    assert usage.input_tokens_total == 1000


def test_negative_counter_is_rejected() -> None:
    """A negative counter raises.

    The openai adapter derives input_tokens_cache_none by subtraction,
    so without this constraint a response over-reporting its cache counters would construct a Usage
    with a negative remainder.
    """
    with pytest.raises(ValidationError):
        _usage(cache_read=900, cache_write=200, cache_none=-100, output=40)


def test_nan_cost_is_stored() -> None:
    """cost_in_usd carries no constraint, so an unpriceable response's NaN survives construction.

    A non-negative constraint rejects NaN, which would fail the whole Usage
    and take the paid response's output down with it.
    """
    assert math.isnan(_usage(output=40, cost=float("nan")).cost_in_usd)


def test_a_sum_containing_a_nan_cost_is_nan() -> None:
    """One unpriceable response makes the batch total NaN, not a silently low number."""
    total = Usage.sum_of([_usage(output=40, cost=0.25), _usage(output=10, cost=float("nan"))])
    assert math.isnan(total.cost_in_usd)
    assert total.output_tokens == 50


def test_add_is_fieldwise_including_cost_and_reasoning() -> None:
    """__add__ sums every field of the two usages, cost and reasoning included."""
    left = _usage(cache_read=1, cache_write=2, cache_none=3, output=4, reasoning=1, cost=0.10)
    right = _usage(cache_read=10, cache_write=20, cache_none=30, output=40, reasoning=5, cost=0.25)
    total = left + right
    assert total.input_tokens_cache_read == 11
    assert total.input_tokens_cache_write == 22
    assert total.input_tokens_cache_none == 33
    assert total.output_tokens == 44
    assert total.output_tokens_reasoning == 6
    assert total.cost_in_usd == pytest.approx(0.35)


def test_zero_usage_is_the_additive_identity() -> None:
    """Adding ZERO_USAGE changes nothing."""
    usage = _usage(cache_none=7, output=3, cost=0.5)
    assert (usage + ZERO_USAGE) == usage
    assert (ZERO_USAGE + usage) == usage


def test_sum_of_totals_a_batch() -> None:
    """Usage.sum_of folds several usages into one paid total."""
    usages = [
        _usage(cache_none=1, output=2, cost=0.10),
        _usage(cache_none=3, output=4, cost=0.20),
        _usage(cache_none=5, output=6, cost=0.30),
    ]
    total = Usage.sum_of(usages)
    assert total.input_tokens_cache_none == 9
    assert total.output_tokens == 12
    assert total.cost_in_usd == pytest.approx(0.60)


def test_sum_of_empty_is_zero_usage() -> None:
    """Usage.sum_of over an empty iterable returns ZERO_USAGE."""
    assert Usage.sum_of([]) == ZERO_USAGE
