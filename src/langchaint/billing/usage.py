"""Token accounting and per-category costs.

The three `input_tokens_*` counters partition all input tokens.
"""

import math
from collections.abc import Iterable
from typing import Annotated

from pydantic import ConfigDict, NonNegativeInt, PlainSerializer

from langchaint.common.checked_copy import CheckedCopyModel


def _serialize_nonfinite_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return value


type JSONFloat = Annotated[
    float, PlainSerializer(_serialize_nonfinite_float, return_type=float | str, when_used="json")
]
"""A float that JSON serialization writes as "NaN", "Infinity", or "-Infinity" when it is not finite.

Validation reads those strings back as the same float.
"""


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
    input_tokens_cache_read_cost_in_usd: JSONFloat
    input_tokens_cache_write_cost_in_usd: JSONFloat
    input_tokens_cache_none_cost_in_usd: JSONFloat
    output_tokens_cost_in_usd: JSONFloat
    provider_executed_tool_cost_in_usd: JSONFloat

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
"""Usage for an empty sum or a request with no reported billing."""
