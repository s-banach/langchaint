"""The arithmetic that spends a rate, and the Billing one attempt carries.

category_cost prices token counters.
Billing carries priced Usage, its service tier, raw usage, and applied token prices.
Counter times price reproduces the stored token cost.
Records retain token arithmetic without the originating rate table.

Each backend subpackage defines the rate table its adapter spends.
This module names no rate-table type.

This module imports no SDK or error class.
Missing category rates produce NaN instead of exceptions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, nan

from pydantic import BaseModel

from langchaint.usage import Usage


def require_pricing_key[KeyT](pricing: Mapping[KeyT, object], *, key: KeyT, model: str) -> None:
    """Require the pricing key every response reporting no tier of its own prices at.

    An adapter constructor calls this with the key its provider's tierless responses select
    ("default", "ON_DEMAND"), so a missing key raises before the first request instead of pricing
    every such response as NaN, with nothing said until the cost comes back unknown.

    Raises:
        ValueError: key is not in pricing.
    """
    if key not in pricing:
        raise ValueError(
            f"pricing for model {model!r} has no {key!r} key; "
            f"it prices every response that reports no tier of its own, so it is required"
        )


def category_cost(tokens: int, usd_per_million_tokens: float) -> float:
    """Price one token category, preserving zero when the rate is unknown.

    The zero case is explicit because 0 * NaN is NaN, and an attempt that billed nothing in a
    category must not make a total unknown when that category has no rate.
    """
    if not tokens:
        return 0.0
    return tokens * usd_per_million_tokens / 1_000_000


def invocation_cost_in_usd(invocations: int, usd_per_invocation: float | None) -> float:
    """Price provider invocations, preserving zero when the rate is unavailable.

    Raises:
        ValueError: `invocations` is boolean or negative.
    """
    if isinstance(invocations, bool) or invocations < 0:
        raise ValueError("invocations must be a nonnegative int")
    if not invocations:
        return 0.0
    if usd_per_invocation is None:
        return nan
    return invocations * usd_per_invocation


def require_finite_nonnegative_rate(*, rate_name: str, rate: float | None) -> None:
    """Reject a configured charged rate that cannot produce a finite nonnegative cost.

    Raises:
        ValueError: `rate` is unavailable, boolean, negative, infinite, or NaN.
    """
    if rate is None or isinstance(rate, bool) or not isfinite(rate) or rate < 0:
        raise ValueError(f"{rate_name} must be finite and nonnegative")


@dataclass(frozen=True, kw_only=True)
class Billing:
    """What one attempt billed: the priced counters, what priced them, and the prices that applied.

    The four token prices are what this attempt's tokens priced at.
    A stored row reproduces its token arithmetic without the originating rate table.
    Later rate changes cannot reprice held history.
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
