"""Token accounting and per-category costs.

The three `input_tokens_*` counters partition all input tokens.
"""

import math
from collections.abc import Iterable

from pydantic import ConfigDict, NonNegativeInt, field_serializer

from langchaint.common.checked_copy import CheckedCopyModel


class Usage(CheckedCopyModel):
    """Provider-reported token counts and estimated costs for one request.

    Validation rejects negative counters.
    `input_tokens_cache_write` combines all cache-write durations.
    `provider_executed_tool_cost_in_usd` aggregates provider-executed tool charges.
    Cost fields accept NaN and negative caller-supplied rates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")

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

    @field_serializer(
        "input_tokens_cache_read_cost_in_usd",
        "input_tokens_cache_write_cost_in_usd",
        "input_tokens_cache_none_cost_in_usd",
        "output_tokens_cost_in_usd",
        "provider_executed_tool_cost_in_usd",
        when_used="json",
    )
    def _serialize_cost(self, cost_in_usd: float) -> float | str:
        return _serialize_nonfinite_float(cost_in_usd)

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
        """Sum counters and costs.

        An empty iterable returns `ZERO_USAGE`.

        Args:
            usages: The usage values to sum.
        """
        usage_values = tuple(usages)
        return Usage(
            input_tokens_cache_read=sum(usage.input_tokens_cache_read for usage in usage_values),
            input_tokens_cache_write=sum(usage.input_tokens_cache_write for usage in usage_values),
            input_tokens_cache_none=sum(usage.input_tokens_cache_none for usage in usage_values),
            output_tokens=sum(usage.output_tokens for usage in usage_values),
            output_tokens_reasoning=sum(usage.output_tokens_reasoning for usage in usage_values),
            input_tokens_cache_read_cost_in_usd=sum(
                usage.input_tokens_cache_read_cost_in_usd for usage in usage_values
            ),
            input_tokens_cache_write_cost_in_usd=sum(
                usage.input_tokens_cache_write_cost_in_usd for usage in usage_values
            ),
            input_tokens_cache_none_cost_in_usd=sum(
                usage.input_tokens_cache_none_cost_in_usd for usage in usage_values
            ),
            output_tokens_cost_in_usd=sum(
                usage.output_tokens_cost_in_usd for usage in usage_values
            ),
            provider_executed_tool_cost_in_usd=sum(
                usage.provider_executed_tool_cost_in_usd for usage in usage_values
            ),
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


def _serialize_nonfinite_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return value
