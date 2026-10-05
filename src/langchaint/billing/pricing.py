"""Pricing arithmetic and per-request `Billing`.

Provider subpackages define rate tables.
A nonzero category with no configured rate costs NaN.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

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


def category_cost_in_usd(tokens: int, *, usd_per_million_tokens: float) -> float:
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


def invocation_cost_in_usd(invocation_count: int, *, usd_per_invocation: float | None) -> float:
    """Price provider invocations, preserving zero when the rate is unavailable.

    Args:
        invocation_count: The invocation count to price.
        usd_per_invocation: The rate in USD per invocation, or `None` when unavailable.

    Raises:
        ValueError: `invocation_count` is boolean or negative.
    """
    if isinstance(invocation_count, bool) or invocation_count < 0:
        raise ValueError("invocation_count must be a nonnegative int")
    if not invocation_count:
        return 0.0
    if usd_per_invocation is None:
        return float("nan")
    return invocation_count * usd_per_invocation


def require_finite_nonnegative_rate(*, rate_name: str, rate: float | None) -> None:
    """Reject a configured charged rate that cannot produce a finite nonnegative cost.

    Args:
        rate_name: The rate name used in the error message.
        rate: The configured rate.

    Raises:
        ValueError: `rate` is unavailable, boolean, negative, infinite, or NaN.
    """
    if rate is None or isinstance(rate, bool) or not math.isfinite(rate) or rate < 0:
        raise ValueError(f"{rate_name} must be finite and nonnegative")


class TokenRates(CheckedCopyModel):
    """The rates applied to one request's priced `Usage` token counters.

    Each rate is in USD per million tokens and has the name of the `Usage` counter it prices.
    A missing category rate is NaN.
    Pydantic validates stored rates and serializes NaN as a string, so a stored `Billing` reproduces its token costs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")

    input_tokens_cache_read: float
    input_tokens_cache_write: float
    input_tokens_cache_none: float
    output_tokens: float

    @field_serializer(
        "input_tokens_cache_read",
        "input_tokens_cache_write",
        "input_tokens_cache_none",
        "output_tokens",
        when_used="json",
    )
    def _serialize_rate(self, rate: float) -> float | str:
        return _serialize_nonfinite_float(rate)


class Billing(CheckedCopyModel):
    """One request's normalized priced usage, service tier, and applied rates.

    `usd_per_million_tokens` stores the applied rates, which reproduce token costs without the original rate table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    usage: Usage
    service_tier: str
    usd_per_million_tokens: TokenRates

    @property
    def cache_savings_in_usd(self) -> float:
        """What prompt caching saved this request against billing every input token uncached.

        The counterfactual prices every input token at the uncached rate.
        Output cost cancels because it is identical in both totals.
        The value is negative when write premiums exceed read discounts.
        It is NaN when a nonzero input counter lacks a rate, and `0.0` when no input tokens were billed.
        """
        uncached = category_cost_in_usd(
            self.usage.input_tokens_total,
            usd_per_million_tokens=self.usd_per_million_tokens.input_tokens_cache_none,
        )
        billed = (
            self.usage.input_tokens_cache_read_cost_in_usd
            + self.usage.input_tokens_cache_write_cost_in_usd
            + self.usage.input_tokens_cache_none_cost_in_usd
        )
        return uncached - billed


@dataclass(frozen=True, kw_only=True)
class ProviderBilling:
    """One request's normalized billing and live provider usage."""

    billing: Billing
    usage_raw: BaseModel | None
