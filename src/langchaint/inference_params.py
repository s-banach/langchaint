"""Inference parameters.

Use `Binding.extra_body` for other provider fields.
`rebind(inference_params=...)` replaces the complete value.
"""

from dataclasses import dataclass
from typing import Literal

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""Reasoning effort values translated by each adapter.

Providers validate model and value combinations.
"""


@dataclass(frozen=True, kw_only=True)
class InferenceParams:
    """`None` keeps the provider default.

    The Anthropic adapter uses `default_max_completion_tokens` when `max_completion_tokens` is `None`.
    """

    max_completion_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
