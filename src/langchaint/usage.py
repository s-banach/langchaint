"""Token accounting.

The three input counters are a disjoint partition of all input tokens, so their sum is the total;
a bare input_tokens field was rejected because Anthropic's field of that name excludes cache reads
while OpenAI's equivalent includes them.
"""

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator


class Usage(BaseModel):
    """Token counts for one request, normalized across providers.

    input_tokens_total_provider_reported is the provider's own all-inclusive input count,
    set only when the provider reports one (verified against anthropic 0.116.0 / openai 2.45.0):
    openai's input_tokens includes cached and cache-write tokens,
    so the openai adapter sets it and the partition is cross-checked;
    anthropic's input_tokens excludes cache reads and writes,
    so no all-inclusive total exists and the anthropic adapter leaves it None.

    Every counter is non-negative by validation, which the openai adapter relies on:
    it derives input_tokens_cache_none by subtracting the cache counters from usage.input_tokens,
    so a response over-reporting its cache counters would otherwise produce a silently negative remainder,
    and on that path the partition cross-check below is satisfied by construction and cannot catch it.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens_cache_read: NonNegativeInt
    input_tokens_cache_write: NonNegativeInt
    input_tokens_cache_none: NonNegativeInt
    output_tokens: NonNegativeInt
    input_tokens_total_provider_reported: NonNegativeInt | None = None

    @property
    def input_tokens_total(self) -> int:
        """Sum of the three disjoint input counters."""
        return (
            self.input_tokens_cache_read
            + self.input_tokens_cache_write
            + self.input_tokens_cache_none
        )

    @model_validator(mode="after")
    def _enforce_partition(self) -> "Usage":
        """Reject a partition that disagrees with the provider-reported total.

        A silent mismatch would corrupt every cost and table row built from this object.

        Raises:
            ValueError: the three counters do not sum to input_tokens_total_provider_reported.
        """
        if (
            self.input_tokens_total_provider_reported is not None
            and self.input_tokens_total_provider_reported != self.input_tokens_total
        ):
            raise ValueError(
                "input counter partition sums to "
                f"{self.input_tokens_total} but the provider reported "
                f"{self.input_tokens_total_provider_reported}"
            )
        return self
