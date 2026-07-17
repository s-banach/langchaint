"""Inference parameters.

Exactly three fields.
The escape hatch for unmapped provider parameters is subclassing the adapter, never a passthrough dict.
Replacement is whole-object: a rebind(inference_params=...) replaces the bound InferenceParams,
so no field-wise merge rules exist to learn;
partial change is spelled dataclasses.replace(bound_llm.binding.inference_params, ...) at the call site.
"""

from dataclasses import dataclass
from typing import Literal

type ReasoningEffort = Literal["low", "medium", "high"]
"""Reasoning effort tiers,
a common subset of both providers' vocabularies (verified against anthropic 0.116.0 / openai 2.45.0):
the anthropic adapter sends the value as output_config.effort, the openai adapter as reasoning_effort.
"""


@dataclass(frozen=True, kw_only=True)
class InferenceParams:
    """None leaves the provider default in place.

    Exception: the Anthropic API requires max_tokens,
    so its adapter fills a None max_completion_tokens with its default_max_completion_tokens.
    """

    max_completion_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
