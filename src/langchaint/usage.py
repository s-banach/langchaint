"""Token accounting and the adapter-priced cost that travels with it.

The three input counters are a disjoint partition of all input tokens, so their sum is the total;
a bare input_tokens field was rejected because Anthropic's field of that name excludes cache reads
while OpenAI's equivalent includes them.

cost_in_usd rides on Usage rather than beside it so the two can never desynchronize:
they are born together in each adapter's _normalized_usage, and one sum folds both.
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt


class Usage(BaseModel):
    """Token counts for one request, normalized across providers, plus the adapter's cost estimate.

    The counters are provider-reported facts. cost_in_usd is the package's estimate,
    priced by the adapter from the raw provider counts against its PricingTable:
    it is stored rather than derived because the normalized counters collapse the 5-minute / 1-hour
    cache-write split that Anthropic bills at different rates, so cost cannot be recomputed from Usage alone.
    No validator cross-checks cost_in_usd; that would require the pricing table and the raw write split here.

    output_tokens_reasoning is the reasoning share of output_tokens
    (Anthropic thinking_tokens, OpenAI reasoning_tokens); whether a provider counts it inside output_tokens
    is an unverified provider fact, so no validator relates the two.

    Every counter is non-negative by validation, which the openai adapter relies on:
    it derives input_tokens_cache_none by subtracting the cache counters from usage.input_tokens,
    so a response over-reporting its cache counters would otherwise produce a silently negative remainder.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens_cache_read: NonNegativeInt
    input_tokens_cache_write: NonNegativeInt
    input_tokens_cache_none: NonNegativeInt
    output_tokens: NonNegativeInt
    output_tokens_reasoning: NonNegativeInt
    cost_in_usd: NonNegativeFloat

    @property
    def input_tokens_total(self) -> int:
        """Sum of the three disjoint input counters."""
        return (
            self.input_tokens_cache_read
            + self.input_tokens_cache_write
            + self.input_tokens_cache_none
        )

    def __add__(self, other: "Usage") -> "Usage":
        """Fieldwise sum, cost included; the counters and dollars of two attempts combine to one."""
        return Usage(
            input_tokens_cache_read=self.input_tokens_cache_read + other.input_tokens_cache_read,
            input_tokens_cache_write=self.input_tokens_cache_write + other.input_tokens_cache_write,
            input_tokens_cache_none=self.input_tokens_cache_none + other.input_tokens_cache_none,
            output_tokens=self.output_tokens + other.output_tokens,
            output_tokens_reasoning=self.output_tokens_reasoning + other.output_tokens_reasoning,
            cost_in_usd=self.cost_in_usd + other.cost_in_usd,
        )

    @staticmethod
    def sum_of(usages: Iterable["Usage"]) -> "Usage":
        """Aggregate several usages into one; the empty iterable returns ZERO_USAGE.

        The import-free way to total usage across several Responses:
        Usage.sum_of(response.usage for response in responses), no ZERO_USAGE import at the call site.
        """
        return sum(usages, start=ZERO_USAGE)


ZERO_USAGE = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=0,
    output_tokens=0,
    output_tokens_reasoning=0,
    cost_in_usd=0.0,
)
"""The zero of Usage addition: the sum() start value, and what a non-billing attempt or a
200 reporting no usage normalizes to."""
