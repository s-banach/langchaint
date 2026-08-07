"""Token accounting and the per-category costs that travel with it.

The three input counters are a disjoint partition of all input tokens, so their sum is the total;
there is no bare input_tokens field, because Anthropic's field of that name excludes cache reads
while OpenAI's equivalent includes them.

The costs ride on Usage rather than beside it so the two can never desynchronize:
they are born together in one price() call, and one fold sums both.
"""

from collections.abc import Iterable

from pydantic import ConfigDict, NonNegativeInt

from langchaint.checked_copy import CheckedCopyModel


class Usage(CheckedCopyModel):
    """Token counts for one request, normalized across providers, plus what each cost.

    The counters are provider-reported facts. The four costs are langchaint's estimate, priced by
    the adapter from the provider's own counters at the rates it holds for the service tier the
    response reported. They are stored rather than derived because a provider can bill one
    normalized counter at several rates: Anthropic's 5-minute and 1-hour cache writes bill
    differently and input_tokens_cache_write collapses both into one count, so no rate the table
    holds recovers that cost. The provider's own usage object, carried beside this one, holds the
    uncollapsed counters.
    No validator cross-checks a cost against its counter; that would require the table here.

    output_tokens_reasoning is the reasoning share of output_tokens
    (Anthropic thinking_tokens, OpenAI reasoning_tokens); whether a provider counts it inside output_tokens
    is an unverified provider fact, so no validator relates the two. It has no cost of its own:
    reasoning tokens bill at the output rate, inside output_tokens_cost_in_usd.

    A category costs NaN when its counter is nonzero and no rate priced it, which is a response
    served at a service tier the adapter holds no table for. Every sum containing it is NaN, and the
    test for it is math.isnan, because nan > limit and nan < limit are both False.

    Server-side tool use has no per-invocation counter and no cost here, so a provider bill charging
    such a fee exceeds cost_in_usd. anthropic reports its own count on the raw SDK usage beside this
    object (Usage.server_tool_use, anthropic 0.120.0); openai's ResponseUsage carries none.

    Every counter is non-negative by validation, so a defect that computes a negative count
    cannot pass silently.
    The cost fields carry no such constraint: it rejects NaN, and an unpriceable response would fail
    validation and take its output down with it. Nothing checks the rates in a caller's own
    rate table either, so a negative rate arrives here as a negative cost.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens_cache_read: NonNegativeInt
    input_tokens_cache_write: NonNegativeInt
    input_tokens_cache_none: NonNegativeInt
    output_tokens: NonNegativeInt
    output_tokens_reasoning: NonNegativeInt
    input_tokens_cache_read_cost_in_usd: float
    input_tokens_cache_write_cost_in_usd: float
    input_tokens_cache_none_cost_in_usd: float
    output_tokens_cost_in_usd: float

    @property
    def input_tokens_total(self) -> int:
        """Sum of the three disjoint input counters."""
        return (
            self.input_tokens_cache_read
            + self.input_tokens_cache_write
            + self.input_tokens_cache_none
        )

    @property
    def cost_in_usd(self) -> float:
        """What the request cost in total, the sum of the four categories."""
        return (
            self.input_tokens_cache_read_cost_in_usd
            + self.input_tokens_cache_write_cost_in_usd
            + self.input_tokens_cache_none_cost_in_usd
            + self.output_tokens_cost_in_usd
        )

    @staticmethod
    def sum_of(usages: Iterable["Usage"]) -> "Usage":
        """Aggregate several usages into one; the empty iterable returns ZERO_USAGE.

        The import-free way to total usage across several Responses:
        Usage.sum_of(response.usage for response in responses), no ZERO_USAGE import at the call site.

        Counters add. Within one model that is meaningful, and across models the sum is a plausible
        number whose meaning is the caller's to judge. Costs add correctly in every case, including
        across models, service tiers, and tables, so a folded Usage reports a correct breakdown as
        well as a correct total.

        There is no __add__ to reach this through: a + b is unlabeled, so it reads as arithmetic and
        hides the claim being made, where a named call makes the caller state it.
        """
        input_tokens_cache_read = 0
        input_tokens_cache_write = 0
        input_tokens_cache_none = 0
        output_tokens = 0
        output_tokens_reasoning = 0
        input_tokens_cache_read_cost_in_usd = 0.0
        input_tokens_cache_write_cost_in_usd = 0.0
        input_tokens_cache_none_cost_in_usd = 0.0
        output_tokens_cost_in_usd = 0.0
        for usage in usages:
            input_tokens_cache_read += usage.input_tokens_cache_read
            input_tokens_cache_write += usage.input_tokens_cache_write
            input_tokens_cache_none += usage.input_tokens_cache_none
            output_tokens += usage.output_tokens
            output_tokens_reasoning += usage.output_tokens_reasoning
            input_tokens_cache_read_cost_in_usd += usage.input_tokens_cache_read_cost_in_usd
            input_tokens_cache_write_cost_in_usd += usage.input_tokens_cache_write_cost_in_usd
            input_tokens_cache_none_cost_in_usd += usage.input_tokens_cache_none_cost_in_usd
            output_tokens_cost_in_usd += usage.output_tokens_cost_in_usd
        return Usage(
            input_tokens_cache_read=input_tokens_cache_read,
            input_tokens_cache_write=input_tokens_cache_write,
            input_tokens_cache_none=input_tokens_cache_none,
            output_tokens=output_tokens,
            output_tokens_reasoning=output_tokens_reasoning,
            input_tokens_cache_read_cost_in_usd=input_tokens_cache_read_cost_in_usd,
            input_tokens_cache_write_cost_in_usd=input_tokens_cache_write_cost_in_usd,
            input_tokens_cache_none_cost_in_usd=input_tokens_cache_none_cost_in_usd,
            output_tokens_cost_in_usd=output_tokens_cost_in_usd,
        )


ZERO_USAGE = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=0,
    output_tokens=0,
    output_tokens_reasoning=0,
    input_tokens_cache_read_cost_in_usd=0.0,
    input_tokens_cache_write_cost_in_usd=0.0,
    input_tokens_cache_none_cost_in_usd=0.0,
    output_tokens_cost_in_usd=0.0,
)
"""What sum_of returns for an empty iterable, and what a non-billing attempt or a
200 reporting no usage normalizes to."""
