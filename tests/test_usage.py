"""The Usage partition invariant.

The three input counters must sum to input_tokens_total_provider_reported when a provider reports one;
a disagreement would corrupt every cost and table row built from the object, so construction rejects it.
"""

import pytest
from pydantic import ValidationError

from langchaint import Usage


def test_agreeing_partition_is_accepted() -> None:
    """Counters that sum to the reported total construct without error."""
    usage = Usage(
        input_tokens_cache_read=600,
        input_tokens_cache_write=100,
        input_tokens_cache_none=300,
        output_tokens=40,
        input_tokens_total_provider_reported=1000,
    )
    assert usage.input_tokens_total == 1000


def test_disagreeing_partition_is_rejected() -> None:
    """A partition that sums to a different value than the reported total raises."""
    with pytest.raises(ValidationError):
        Usage(
            input_tokens_cache_read=600,
            input_tokens_cache_write=100,
            input_tokens_cache_none=300,
            output_tokens=40,
            input_tokens_total_provider_reported=999,
        )


def test_negative_counter_is_rejected() -> None:
    """A negative counter raises even when the partition cross-check agrees.

    The openai adapter derives input_tokens_cache_none by subtraction,
    so without this constraint a response over-reporting its cache counters would construct a Usage
    with a negative remainder that still sums to the reported total.
    """
    with pytest.raises(ValidationError):
        Usage(
            input_tokens_cache_read=900,
            input_tokens_cache_write=200,
            input_tokens_cache_none=-100,
            output_tokens=40,
            input_tokens_total_provider_reported=1000,
        )


def test_absent_reported_total_skips_the_cross_check() -> None:
    """With no reported total the partition is not cross-checked."""
    usage = Usage(
        input_tokens_cache_read=1,
        input_tokens_cache_write=2,
        input_tokens_cache_none=3,
        output_tokens=4,
    )
    assert usage.input_tokens_total_provider_reported is None
    assert usage.input_tokens_total == 6
