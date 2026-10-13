"""Pricing arithmetic and per-request `Billing`.

Provider subpackages define rate tables.
A nonzero category with no configured rate costs NaN.
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from langchaint.billing.usage import JSONFloat, Usage
from langchaint.common.checked_copy import CheckedCopyModel


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


def require_stated_rate(*, rate_name: str, rate: float | None) -> None:
    """Require the price of a charged tool the binding enables.

    Without it, every response that uses the tool would cost NaN.

    Args:
        rate_name: The rate name used in the error message.
        rate: The configured rate.

    Raises:
        ValueError: `rate` is None.
    """
    if rate is None:
        raise ValueError(f"{rate_name} is None; state the price of every charged tool you bind")


@dataclass(frozen=True, kw_only=True)
class LongContextPricing:
    """Rate multipliers for a request whose prompt length exceeds `prompt_tokens_above` tokens.

    Providers count prompt length differently:
    - `OpenAIPricingTable` and `AnthropicPricingTable` count `input_tokens_total`, including cache reads and writes.
    - `GeminiPricingTable` counts `prompt_token_count`, which includes cached input and excludes tool-execution input.

    `input_multiplier` scales every input rate, including cache-read and cache-write rates.
    `output_multiplier` scales the output rate.
    """

    prompt_tokens_above: int
    input_multiplier: float
    output_multiplier: float

    def applied_to(self, rates: "TokenRates", *, prompt_tokens: int) -> "TokenRates":
        """Return the rates a request whose prompt length is `prompt_tokens` pays.

        A prompt longer than `prompt_tokens_above` tokens pays `rates` scaled by these multipliers.
        A shorter prompt pays `rates`.
        """
        if prompt_tokens <= self.prompt_tokens_above:
            return rates
        return rates.scaled(
            input_multiplier=self.input_multiplier, output_multiplier=self.output_multiplier
        )


class TokenRates(CheckedCopyModel):
    """Rates for the priced `Usage` token counters.

    Each rate is in USD per million tokens and has the name of the `Usage` counter it prices.
    A missing category rate is NaN.
    A nonzero counter in that category then costs NaN, and a zero counter costs zero.
    Pydantic validates stored rates and serializes NaN as a string, so a stored `Billing` reproduces its token costs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens_cache_read: JSONFloat
    input_tokens_cache_write: JSONFloat
    input_tokens_cache_none: JSONFloat
    output_tokens: JSONFloat

    def scaled(self, *, input_multiplier: float, output_multiplier: float) -> "TokenRates":
        """Multiply every input rate by `input_multiplier` and the output rate by `output_multiplier`."""
        return TokenRates(
            input_tokens_cache_read=self.input_tokens_cache_read * input_multiplier,
            input_tokens_cache_write=self.input_tokens_cache_write * input_multiplier,
            input_tokens_cache_none=self.input_tokens_cache_none * input_multiplier,
            output_tokens=self.output_tokens * output_multiplier,
        )


UNPRICED_TOKEN_RATES: TokenRates = TokenRates(
    input_tokens_cache_read=float("nan"),
    input_tokens_cache_write=float("nan"),
    input_tokens_cache_none=float("nan"),
    output_tokens=float("nan"),
)
"""The rates of a response whose served tier, region, or model has no configured rates.

Every nonzero counter costs NaN, and every zero counter costs zero.
"""


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


def billing_at_rates(
    rates: TokenRates,
    *,
    service_tier: str,
    input_tokens_cache_read: int,
    input_tokens_cache_write: int,
    input_tokens_cache_none: int,
    output_tokens: int,
    output_tokens_reasoning: int,
    provider_executed_tool_cost_in_usd: float,
) -> Billing:
    """Price one response's counters at `rates`.

    `service_tier` is the tier that selected `rates`, stored on the `Billing`.
    Each token category costs its counter at the rate of the same name.
    `output_tokens_reasoning` is the reasoning share of `output_tokens`, so the output rate prices it.
    `provider_executed_tool_cost_in_usd` adds the provider-executed tool charge.

    Raises:
        pydantic.ValidationError: A counter is negative.
    """
    return Billing(
        usage=Usage(
            input_tokens_cache_read=input_tokens_cache_read,
            input_tokens_cache_write=input_tokens_cache_write,
            input_tokens_cache_none=input_tokens_cache_none,
            output_tokens=output_tokens,
            output_tokens_reasoning=output_tokens_reasoning,
            input_tokens_cache_read_cost_in_usd=category_cost_in_usd(
                input_tokens_cache_read, usd_per_million_tokens=rates.input_tokens_cache_read
            ),
            input_tokens_cache_write_cost_in_usd=category_cost_in_usd(
                input_tokens_cache_write, usd_per_million_tokens=rates.input_tokens_cache_write
            ),
            input_tokens_cache_none_cost_in_usd=category_cost_in_usd(
                input_tokens_cache_none, usd_per_million_tokens=rates.input_tokens_cache_none
            ),
            output_tokens_cost_in_usd=category_cost_in_usd(
                output_tokens, usd_per_million_tokens=rates.output_tokens
            ),
            provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
        ),
        service_tier=service_tier,
        usd_per_million_tokens=rates,
    )


@dataclass(frozen=True, kw_only=True)
class ProviderBilling:
    """One request's normalized billing and live provider usage."""

    billing: Billing
    usage_raw: BaseModel | None
