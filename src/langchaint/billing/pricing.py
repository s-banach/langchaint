"""Pricing arithmetic and per-attempt `Billing`.

Provider subpackages define rate tables.
A nonzero category with no configured rate costs NaN.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, nan

from pydantic import BaseModel, ConfigDict, field_serializer

from langchaint.billing.usage import Usage, _serialize_nonfinite_float
from langchaint.common.checked_copy import CheckedCopyModel


def require_pricing_key[KeyT](pricing: Mapping[KeyT, object], *, key: KeyT, model: str) -> None:
    """Require the pricing key used for a response that reports no service tier.

    Args:
        pricing: The rate table to check.
        key: The key used for a response with no service tier.
        model: The model id used in the error message.

    Raises:
        ValueError: `key` is absent from `pricing`.
    """
    if key not in pricing:
        raise ValueError(
            f"pricing for model {model!r} has no {key!r} key; "
            f"it prices every response that reports no tier of its own, so it is required"
        )


def category_cost(tokens: int, *, usd_per_million_tokens: float) -> float:
    """Price one token category, preserving zero when the rate is unknown.

    `0 * NaN` is NaN.
    A zero-token category must therefore preserve a known zero cost.

    Args:
        tokens: The token count to price.
        usd_per_million_tokens: The rate in USD per million tokens.
    """
    if not tokens:
        return 0.0
    return tokens * usd_per_million_tokens / 1_000_000


def invocation_cost_in_usd(invocations: int, *, usd_per_invocation: float | None) -> float:
    """Price provider invocations, preserving zero when the rate is unavailable.

    Args:
        invocations: The invocation count to price.
        usd_per_invocation: The rate in USD per invocation, or `None` when unavailable.

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

    Args:
        rate_name: The rate name used in the error message.
        rate: The configured rate.

    Raises:
        ValueError: `rate` is unavailable, boolean, negative, infinite, or NaN.
    """
    if rate is None or isinstance(rate, bool) or not isfinite(rate) or rate < 0:
        raise ValueError(f"{rate_name} must be finite and nonnegative")


class Billing(CheckedCopyModel):
    """One attempt's normalized priced usage, service tier, and applied rates.

    Stored rates reproduce token costs without the original rate table.
    A missing category rate is NaN.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")

    usage: Usage
    service_tier: str
    input_cache_none_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_usd_per_million_tokens: float
    output_usd_per_million_tokens: float

    @field_serializer(
        "input_cache_none_usd_per_million_tokens",
        "cache_read_usd_per_million_tokens",
        "cache_write_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        when_used="json",
    )
    def _serialize_rate(self, rate: float) -> float | str:
        return _serialize_nonfinite_float(rate)

    @property
    def cache_savings_in_usd(self) -> float:
        """What prompt caching saved this attempt against billing every input token uncached.

        The counterfactual prices every input token at the uncached rate.
        Output cost cancels because it is identical in both totals.
        The result is negative when write premiums exceed read discounts.
        It is NaN when a nonzero input counter lacks a rate, and `0.0` when no input was billed.
        """
        uncached = category_cost(
            self.usage.input_tokens_total,
            usd_per_million_tokens=self.input_cache_none_usd_per_million_tokens,
        )
        billed = (
            self.usage.input_tokens_cache_read_cost_in_usd
            + self.usage.input_tokens_cache_write_cost_in_usd
            + self.usage.input_tokens_cache_none_cost_in_usd
        )
        return uncached - billed


@dataclass(frozen=True, kw_only=True)
class ProviderBilling:
    """One attempt's normalized billing and live provider usage."""

    billing: Billing
    usage_raw: BaseModel | None
