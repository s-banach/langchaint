"""Token accounting and per-category costs.

The three `input_tokens_*` counters partition all input tokens.
"""

from collections.abc import Iterable

from pydantic import ConfigDict, NonNegativeInt

from langchaint.checked_copy import CheckedCopyModel


class Usage(CheckedCopyModel):
    """Provider-reported token counts and estimated costs for one request.

    Validation rejects negative counters.
    `input_tokens_cache_write` combines all cache-write durations.
    `provider_executed_tool_cost_in_usd` aggregates provider-executed tool charges.
    Cost fields accept NaN and negative caller-supplied rates.
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
    provider_executed_tool_cost_in_usd: float

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
        """Sum the five cost categories."""
        return (
            self.input_tokens_cache_read_cost_in_usd
            + self.input_tokens_cache_write_cost_in_usd
            + self.input_tokens_cache_none_cost_in_usd
            + self.output_tokens_cost_in_usd
            + self.provider_executed_tool_cost_in_usd
        )

    @staticmethod
    def sum_of(usages: Iterable["Usage"]) -> "Usage":
        """Sum counters and costs; an empty iterable returns `ZERO_USAGE`."""
        input_tokens_cache_read = 0
        input_tokens_cache_write = 0
        input_tokens_cache_none = 0
        output_tokens = 0
        output_tokens_reasoning = 0
        input_tokens_cache_read_cost_in_usd = 0.0
        input_tokens_cache_write_cost_in_usd = 0.0
        input_tokens_cache_none_cost_in_usd = 0.0
        output_tokens_cost_in_usd = 0.0
        provider_executed_tool_cost_in_usd = 0.0
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
            provider_executed_tool_cost_in_usd += usage.provider_executed_tool_cost_in_usd
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
            provider_executed_tool_cost_in_usd=provider_executed_tool_cost_in_usd,
        )


ZERO_USAGE: Usage = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=0,
    output_tokens=0,
    output_tokens_reasoning=0,
    input_tokens_cache_read_cost_in_usd=0.0,
    input_tokens_cache_write_cost_in_usd=0.0,
    input_tokens_cache_none_cost_in_usd=0.0,
    output_tokens_cost_in_usd=0.0,
    provider_executed_tool_cost_in_usd=0.0,
)
"""Usage for an empty sum or an attempt with no reported billing."""
