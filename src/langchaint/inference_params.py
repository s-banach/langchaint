"""Inference parameters.

`InferenceParams` carries provider-neutral fields shared across adapters.
Use `Binding.extra_body` for provider wire fields absent from `InferenceParams`.
`rebind(inference_params=...)` replaces the complete `InferenceParams`.
Use `dataclasses.replace(bound_llm.binding.inference_params, ...)` for one-field changes.
"""

from dataclasses import dataclass
from typing import Literal

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""Reasoning effort tiers shared across providers.

The values match anthropic 0.120.0 and openai 2.45.0.
OpenAI accepts every value.
Anthropic accepts `"low"` through `"max"`.
OpenAI sends the value as `reasoning_effort`.
Anthropic enables adaptive thinking and sends `output_config.effort`.
Gemini uppercases the value for `thinking_config.thinking_level`.
Gemini sends `"none"` as `thinking_budget=0`.
Providers validate model and value combinations.
Runtime validation does not inspect the value.
Suppress the type error to send a value missing from this `Literal`.
"""


@dataclass(frozen=True, kw_only=True)
class InferenceParams:
    """`None` keeps the provider default.

    Anthropic requires `max_tokens`.
    Its adapter substitutes `default_max_completion_tokens` when `max_completion_tokens` is `None`.
    """

    max_completion_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
