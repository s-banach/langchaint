"""Wiring in the langchaint.anthropic and langchaint.openai catalogs.

The pricing values themselves are the one provider fact tests cannot verify;
what tests can catch is a catalog function wiring the wrong model identifier, the wrong prices,
or losing an override, a copy-paste error that would type-check and ship silently.
"""

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from langchaint import PricingTable, RateLimiter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    AnthropicMessagesProvider,
    AnthropicModelName,
    anthropic_model,
)
from langchaint.openai import (
    OPENAI_PRICING,
    OpenAIModelName,
    OpenAIResponsesProvider,
    openai_model,
)


@pytest.mark.parametrize("model", list(ANTHROPIC_PRICING))
def test_anthropic_model_wires_model_and_pricing(model: AnthropicModelName) -> None:
    """anthropic_model returns an LLM whose adapter carries the model's prices."""
    llm = anthropic_model(model, client=AsyncAnthropic(api_key="offline"))
    provider = llm.provider
    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.model == model
    assert provider.pricing is ANTHROPIC_PRICING[model]


@pytest.mark.parametrize("model", list(OPENAI_PRICING))
def test_openai_model_wires_model_and_pricing(model: OpenAIModelName) -> None:
    """openai_model returns an LLM whose adapter carries the model's prices."""
    llm = openai_model(model, client=AsyncOpenAI(api_key="offline"))
    provider = llm.provider
    assert isinstance(provider, OpenAIResponsesProvider)
    assert provider.model == model
    assert provider.pricing is OPENAI_PRICING[model]


def test_adapter_client_never_retries_beneath_the_package() -> None:
    """The stored client is a max_retries=0 copy keeping the caller's credentials."""
    client = AsyncAnthropic(api_key="offline")
    llm = anthropic_model("claude-sonnet-5", client=client)
    provider = llm.provider
    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.client.max_retries == 0
    assert provider.client.api_key == client.api_key


def test_pricing_override_replaces_the_default() -> None:
    """A caller-supplied pricing table lands on the adapter unchanged."""
    custom_pricing = PricingTable(
        input_cache_none_usd_per_million_tokens=2.00,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.20,
        cache_write_usd_per_million_tokens=2.50,
        cache_write_1h_usd_per_million_tokens=4.00,
    )
    llm = anthropic_model(
        "claude-sonnet-5",
        client=AsyncAnthropic(api_key="offline"),
        pricing=custom_pricing,
    )
    assert llm.provider.pricing is custom_pricing


def test_rate_limiter_lands_on_the_llm() -> None:
    """A caller-supplied RateLimiter is the LLM's; None means a fresh default."""
    rate_limiter = RateLimiter(max_attempts=5)
    llm = openai_model(
        "gpt-5.6-terra",
        client=AsyncOpenAI(api_key="offline"),
        rate_limiter=rate_limiter,
    )
    assert llm.rate_limiter is rate_limiter
    defaulted = openai_model("gpt-5.6-terra", client=AsyncOpenAI(api_key="offline"))
    assert isinstance(defaulted.rate_limiter, RateLimiter)
    assert defaulted.rate_limiter is not rate_limiter

def test_cache_ttl_lands_on_the_adapter() -> None:
    """A caller-supplied cache_ttl reaches the adapter; the default is "5m"."""
    llm = anthropic_model("claude-sonnet-5", client=AsyncAnthropic(api_key="offline"), cache_ttl="1h")
    provider = llm.provider
    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.cache_ttl == "1h"
    defaulted = anthropic_model("claude-sonnet-5", client=AsyncAnthropic(api_key="offline"))
    assert isinstance(defaulted.provider, AnthropicMessagesProvider)
    assert defaulted.provider.cache_ttl == "5m"
