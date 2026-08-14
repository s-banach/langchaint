"""Pricing arithmetic and per-attempt `Billing`.

Provider subpackages define rate tables.
A nonzero category with no configured rate costs NaN.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, nan

from pydantic import BaseModel

from langchaint.usage import Usage


def require_pricing_key[KeyT](pricing: Mapping[KeyT, object], *, key: KeyT, model: str) -> None:
    """Require the pricing key every response reporting no tier of its own prices at.

    An adapter constructor passes the key selected by tierless responses, such as `"default"` or `"ON_DEMAND"`.
    A missing key then raises before the first request.

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

    `0 * NaN` is NaN.
    A zero-token category must therefore preserve a known zero cost.
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
    """One attempt's priced usage, service tier, raw usage, and applied rates.

    Stored rates reproduce token costs without the original rate table.
    A missing category rate is NaN.
    `usage_raw` holds the mutable provider object by reference; copy it before mutation.
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

        The counterfactual prices every input token at the uncached rate.
        Output cost cancels because it is identical in both totals.
        The result is negative when write premiums exceed read discounts.
        It is NaN when a nonzero input counter lacks a rate, and `0.0` when no input was billed.
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
