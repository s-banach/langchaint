"""The neutral rate table and the arithmetic that spends it.

A PricingTable is one model's rates at one service tier, one rate per priced category.
price() multiplies each rate by its counter and returns the whole Usage, so a priced Usage has one
constructor per rate shape and no per-category cost object exists between the two.

A provider that bills more finely than these four rates gets its own table with its own price()
in its backend subpackage (AnthropicPricingTable, whose two cache-write rates cover the two TTLs).
What crosses into Usage is the four priced categories, never a rate, so the extra structure is
spent inside the adapter and no neutral type learns about it.

This module imports no SDK and no error class: a category the caller supplied no rate for costs NaN,
so an unpriced category needs no error.
"""

from dataclasses import dataclass

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
class PricingTable:
    """USD prices per one million tokens for one model at one service tier.

    input_cache_none_usd_per_million_tokens prices only the uncached input, the partition's
    input_tokens_cache_none; cache reads and writes bill at their own rates.
    cache_write_usd_per_million_tokens applies to OpenAI too: OpenAI bills cache writes
    (reported as input_tokens_details.cache_write_tokens) starting with gpt-5.6.

    An adapter holds one table per service tier it can price and selects by the tier the response
    reports, so a response served at a tier the caller supplied no table for costs NaN
    rather than these rates.
    """

    input_cache_none_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    cache_read_usd_per_million_tokens: float
    cache_write_usd_per_million_tokens: float

    def price(
        self,
        *,
        input_tokens_cache_read: int,
        input_tokens_cache_write: int,
        input_tokens_cache_none: int,
        output_tokens: int,
        output_tokens_reasoning: int,
    ) -> Usage:
        """Price one response's counters at these rates.

        The counters arrive as arguments rather than in a counts object,
        which would exist only to be unpacked again one call later.
        output_tokens_reasoning buys no cost, being the reasoning share of output_tokens and billed
        at the output rate; it is a parameter because the returned Usage carries it.

        The total is the sum of the four category costs, so the parts are individually meaningful
        and sum to cost_in_usd exactly; that association differs from a fused single-division chain
        only at sub-ULP scale, immaterial once billing rounds to cents.

        Raises:
            pydantic.ValidationError: a counter is negative, which the openai adapter's subtraction
                produces from a response over-reporting its cache counts.
        """
        return Usage(
            input_tokens_cache_read=input_tokens_cache_read,
            input_tokens_cache_write=input_tokens_cache_write,
            input_tokens_cache_none=input_tokens_cache_none,
            output_tokens=output_tokens,
            output_tokens_reasoning=output_tokens_reasoning,
            input_tokens_cache_read_cost_in_usd=category_cost(
                input_tokens_cache_read, self.cache_read_usd_per_million_tokens
            ),
            input_tokens_cache_write_cost_in_usd=category_cost(
                input_tokens_cache_write, self.cache_write_usd_per_million_tokens
            ),
            input_tokens_cache_none_cost_in_usd=category_cost(
                input_tokens_cache_none, self.input_cache_none_usd_per_million_tokens
            ),
            output_tokens_cost_in_usd=category_cost(
                output_tokens, self.output_usd_per_million_tokens
            ),
        )


UNPRICED = PricingTable(
    input_cache_none_usd_per_million_tokens=float("nan"),
    output_usd_per_million_tokens=float("nan"),
    cache_read_usd_per_million_tokens=float("nan"),
    cache_write_usd_per_million_tokens=float("nan"),
)
"""The table an adapter prices at when the response reports a service tier it holds no table for.

Every nonzero counter costs NaN and every zero counter costs zero, so the response's own numbers
survive and the cost says it is unknown. Pricing at some other tier's rates instead would report a
number wrong by that tier's multiplier, and raising would destroy a response that was paid for.
"""
