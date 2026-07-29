"""The arithmetic that spends a rate, and the Billing one attempt carries.

category_cost is the one multiplication: a rate per million tokens against a counter.
Billing is what an adapter reports per response, carrying the priced Usage, the tier that priced it,
the provider's own usage object, and the four prices behind the four costs.
Counter times price reproduces the stored cost, so a record reproduces its own arithmetic without
the rate table that made it.

A rate table is provider-shaped and lives in the backend subpackage whose adapter spends it, since
only that adapter knows what its provider bills for. This module names no table type.

This module imports no SDK and no error class: a category the caller supplied no rate for costs NaN,
so an unpriced category needs no error.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from langchaint.usage import Usage


def category_cost(tokens: int, usd_per_million_tokens: float) -> float:
    """One category's cost, zero for a zero counter whatever the rate.

    The zero case is explicit because 0 * NaN is NaN, and an attempt that billed nothing in a
    category must not make a total unknown when that category has no rate.
    """
    if not tokens:
        return 0.0
    return tokens * usd_per_million_tokens / 1_000_000


@dataclass(frozen=True, kw_only=True)
class Billing:
    """What one attempt billed: the priced counters, what priced them, and the prices that applied.

    The four prices are what this attempt's tokens priced at, so a stored row reproduces its own
    costs by multiplication and a later rate change cannot silently reprice held history.
    A provider that bills cache writes at more than one rate in one response reports the blend of
    what those writes cost, since the counters behind the split do not survive into Usage.
    A price is NaN where no rate stood behind that category, which leaves the response's own
    counters intact and says the cost is unknown.

    service_tier is the tier whose rates priced this attempt, at the provider's own spelling. Where
    a response reports a value that names no tier of its own ("auto", or nothing at all), it is the
    concrete tier that value resolved to, so the tier and the prices beside it always agree.
    usage_raw is the provider's usage object, held by reference: a live, mutable pydantic object
    despite the frozen dataclass around it, so treat it read-only.
    """

    usage: Usage
    service_tier: str
    usage_raw: BaseModel | None
    input_cache_none_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_usd_per_million_tokens: float
    output_usd_per_million_tokens: float

    @property
    def cache_savings_in_usd(self) -> float:
        """What prompt caching saved this attempt against billing every input token uncached.

        The counterfactual prices the whole input at the uncached rate; output cost is identical
        under both and cancels. Negative where write premiums exceeded read discounts, NaN where a
        nonzero input counter has no rate behind it, and 0.0 where no input was billed.
        """
        uncached = category_cost(
            self.usage.input_tokens_total, self.input_cache_none_usd_per_million_tokens
        )
        billed = (
            self.usage.input_tokens_cache_read_cost_in_usd
            + self.usage.input_tokens_cache_write_cost_in_usd
            + self.usage.input_tokens_cache_none_cost_in_usd
        )
        return uncached - billed
