"""Neutral cost arithmetic: price already-split token counts against a PricingTable.

Every stored Usage.cost_in_usd is routed through price() by the backend adapters,
and each backend's public cost_breakdown function (langchaint.anthropic.cost_breakdown,
langchaint.openai.cost_breakdown) extracts PriceableCounts from the raw SDK usage and calls the same price(),
so a reported breakdown and the stored scalar cannot disagree.
The extraction is per-backend because only the raw SDK usage keeps the 5-minute / 1-hour cache-write split
the neutral Usage collapses into one input_tokens_cache_write counter.

This module imports no SDK and no error class: price raises the built-in ValueError,
so the batch concept AbortBatchError stays out of the neutral core;
only the anthropic adapter's generation path translates the ValueError into an AbortBatchError,
where a shared pricing-table defect genuinely dooms a batch.
"""

from dataclasses import dataclass

from langchaint.adapter import PricingTable


@dataclass(frozen=True, kw_only=True)
class PriceableCounts:
    """Token counts split the way pricing needs, keeping the two cache-write rate tiers apart.

    input_tokens_cache_write holds the tokens priced at cache_write_usd_per_million_tokens
    (anthropic 5-minute writes and every openai write);
    input_tokens_cache_write_1h holds the tokens priced at cache_write_1h_usd_per_million_tokens
    (anthropic 1-hour writes only).
    Their sum is the neutral Usage.input_tokens_cache_write, which collapses the tiers.
    Reasoning tokens are not a field: they are the reasoning share of output_tokens,
    billed at the same output rate, so they are already inside output_tokens and are not a separate cost line.
    """

    input_tokens_cache_none: int
    input_tokens_cache_read: int
    input_tokens_cache_write: int
    input_tokens_cache_write_1h: int
    output_tokens: int


@dataclass(frozen=True, kw_only=True)
class CostBreakdown:
    """Exact per-category cost for one request, plus the counts it priced.

    total_cost_in_usd equals the Usage.cost_in_usd stored for the same request,
    because the adapter routes the stored scalar through the same price() call.
    counts is kept so an application can write its own caching counterfactual against
    whatever baseline it chooses; langchaint ships only the per-category facts,
    because a savings baseline (for example, repricing every cache token at the uncached rate)
    is a billing opinion.
    """

    counts: PriceableCounts
    input_tokens_cache_none_cost_in_usd: float
    input_tokens_cache_read_cost_in_usd: float
    input_tokens_cache_write_cost_in_usd: float
    input_tokens_cache_write_1h_cost_in_usd: float
    output_tokens_cost_in_usd: float

    @property
    def input_tokens_cost_in_usd(self) -> float:
        """The input share of the total."""
        return (
            self.input_tokens_cache_none_cost_in_usd
            + self.input_tokens_cache_read_cost_in_usd
            + self.input_tokens_cache_write_cost_in_usd
            + self.input_tokens_cache_write_1h_cost_in_usd
        )

    @property
    def total_cost_in_usd(self) -> float:
        """The whole request's cost."""
        return self.input_tokens_cost_in_usd + self.output_tokens_cost_in_usd


def price(counts: PriceableCounts, pricing: PricingTable) -> CostBreakdown:
    """Price already-split counts against a table.

    The total is the sum of the per-category products, so the parts are individually meaningful
    and sum to total_cost_in_usd exactly; that association differs from a fused single-division
    chain only at sub-ULP scale, immaterial once billing rounds to cents.

    Raises:
        ValueError: counts carry 1-hour cache writes but pricing has no
            cache_write_1h_usd_per_million_tokens.
    """
    input_tokens_cache_write_1h_cost_in_usd = 0.0
    if counts.input_tokens_cache_write_1h:
        if pricing.cache_write_1h_usd_per_million_tokens is None:
            raise ValueError(
                "the counts carry 1-hour cache writes but the PricingTable "
                "has no cache_write_1h_usd_per_million_tokens"
            )
        input_tokens_cache_write_1h_cost_in_usd = (
            counts.input_tokens_cache_write_1h
            * pricing.cache_write_1h_usd_per_million_tokens
            / 1_000_000
        )
    return CostBreakdown(
        counts=counts,
        input_tokens_cache_none_cost_in_usd=(
            counts.input_tokens_cache_none
            * pricing.input_cache_none_usd_per_million_tokens
            / 1_000_000
        ),
        input_tokens_cache_read_cost_in_usd=(
            counts.input_tokens_cache_read * pricing.cache_read_usd_per_million_tokens / 1_000_000
        ),
        input_tokens_cache_write_cost_in_usd=(
            counts.input_tokens_cache_write
            * pricing.cache_write_usd_per_million_tokens
            / 1_000_000
        ),
        input_tokens_cache_write_1h_cost_in_usd=input_tokens_cache_write_1h_cost_in_usd,
        output_tokens_cost_in_usd=(
            counts.output_tokens * pricing.output_usd_per_million_tokens / 1_000_000
        ),
    )
