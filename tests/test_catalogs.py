"""Wiring in the langchaint.anthropic and langchaint.openai catalogs.

The pricing values themselves are the one provider fact tests cannot verify;
what tests can catch is a catalog function wiring the wrong model identifier, the wrong prices,
or losing an override, a copy-paste error that would type-check and ship silently.
"""

import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrockMantle
from openai import AsyncOpenAI

from langchaint import PricingTable, RateLimiter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    AnthropicMessagesProvider,
    AnthropicModelName,
    anthropic_bedrock_model,
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


# One model per Bedrock surface: claude-opus-4-8 routes to "mantle", claude-opus-4-6 to "legacy".
# The transport-drop bug lives in each surface's own copy() override, so both surfaces are exercised.
@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-opus-4-6"])
def test_bedrock_http_client_survives_the_retry_suppression_copy(model: AnthropicModelName) -> None:
    """A custom httpx client passed to anthropic_bedrock_model reaches the stored adapter client.

    The two Bedrock client classes override copy() without reusing the existing transport (anthropic
    0.116.0), so a plain with_options(max_retries=0) drops it; the adapter re-feeds it, so a caller's
    loaded certs reach the wire. This asserts the injected client survives that copy, not a fresh default.
    """
    http_client = httpx.AsyncClient()
    llm = anthropic_bedrock_model(model, aws_region="us-east-1", http_client=http_client)
    provider = llm.provider
    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.client.max_retries == 0
    assert provider.client._client is http_client


def test_bedrock_rejects_client_and_http_client_together() -> None:
    """Passing both client and http_client raises: a passed client already owns its transport."""
    client = AsyncAnthropicBedrockMantle(aws_region="us-east-1")
    with pytest.raises(ValueError, match="at most one"):
        anthropic_bedrock_model(
            "claude-opus-4-8", client=client, http_client=httpx.AsyncClient()
        )


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
