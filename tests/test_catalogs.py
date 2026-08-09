"""Backend catalog wiring tests.

These tests cover identifiers, pricing objects, overrides, and client routing.
"""

import asyncio
from collections.abc import Callable

import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from google import genai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai_semconv

from langchaint import LLM
from langchaint.adapter import Adapter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    AnthropicAccount,
    AnthropicBedrockAccount,
    AnthropicBedrockModelName,
    AnthropicMessagesAdapter,
    AnthropicModelName,
    AnthropicPricedServiceTier,
    AnthropicPricingTable,
)
from langchaint.deepseek import (
    DEEPSEEK_PRICING,
    DeepSeekAccount,
    DeepSeekModelName,
    cache_read_tokens_from_usage_deepseek,
)
from langchaint.gemini import (
    GEMINI_PRICING,
    GeminiAccount,
    GeminiGenerateContentAdapter,
    GeminiModelName,
    GeminiPricingTable,
    GeminiRates,
)
from langchaint.openai import (
    OPENAI_PRICING,
    OpenAIAccount,
    OpenAIBedrockAccount,
    OpenAIChatCompletionsAdapter,
    OpenAIModelName,
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
)

_ARBITRARY_PRICING: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
    "default": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=1.0,
        cache_read_usd_per_million_tokens=1.0,
        cache_write_usd_per_million_tokens=1.0,
    )
}
"""Stands in where OpenAIBedrockAccount.model requires unrelated pricing.

OpenAIBedrockAccount.model has no default pricing catalog.
"""

_ARBITRARY_ANTHROPIC_PRICING: dict[AnthropicPricedServiceTier, AnthropicPricingTable] = {
    "standard": AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=1.0,
        cache_read_usd_per_million_tokens=1.0,
        cache_write_5m_usd_per_million_tokens=1.0,
        cache_write_1h_usd_per_million_tokens=1.0,
    )
}
"""The anthropic counterpart of _ARBITRARY_PRICING, for adapters built without a catalog."""

_ARBITRARY_GEMINI_PRICING: dict[str, GeminiPricingTable] = {
    "ON_DEMAND": GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=1.0,
            cache_read_usd_per_million_tokens=1.0,
            output_usd_per_million_tokens=1.0,
        )
    )
}
"""The gemini counterpart of _ARBITRARY_PRICING, for adapters built without a catalog."""


@pytest.mark.parametrize("model", list(ANTHROPIC_PRICING))
def test_anthropic_account_model_wires_model_and_pricing(model: AnthropicModelName) -> None:
    """AnthropicAccount.model returns an adapter carrying catalog pricing."""
    llm = AnthropicAccount(client=AsyncAnthropic(api_key="offline")).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == model
    assert adapter.pricing["standard"] is ANTHROPIC_PRICING[model]


@pytest.mark.parametrize("model", list(GEMINI_PRICING))
def test_gemini_account_model_wires_model_and_pricing(model: GeminiModelName) -> None:
    """GeminiAccount.model returns an adapter carrying catalog pricing."""
    llm = GeminiAccount(client=genai.Client(api_key="offline", vertexai=False)).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == model
    assert adapter.pricing["ON_DEMAND"] is GEMINI_PRICING[model]


@pytest.mark.parametrize("model", list(OPENAI_PRICING))
def test_openai_account_model_wires_model_and_pricing(model: OpenAIModelName) -> None:
    """OpenAIAccount.model returns an adapter carrying catalog pricing."""
    llm = OpenAIAccount(client=AsyncOpenAI(api_key="offline")).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == model
    assert adapter.pricing["default"] is OPENAI_PRICING[model]


_PROMPT_CACHE_OPTIONS_SUPPORT: dict[OpenAIModelName, bool] = {
    "gpt-5.1": False,
    "gpt-5.2": False,
    "gpt-5.4": False,
    "gpt-5.4-mini": False,
    "gpt-5.5": False,
    "gpt-5.6-luna": True,
    "gpt-5.6-terra": True,
    "gpt-5.6-sol": True,
}
"""Expected cache support for each cataloged OpenAI model."""


def test_the_prompt_cache_options_expectations_cover_the_catalog() -> None:
    """Require one cache-support expectation for every cataloged model."""
    assert set(_PROMPT_CACHE_OPTIONS_SUPPORT) == set(OPENAI_PRICING)


@pytest.mark.parametrize(("model", "supported"), list(_PROMPT_CACHE_OPTIONS_SUPPORT.items()))
def test_openai_account_model_wires_prompt_cache_options_support(
    model: OpenAIModelName, *, supported: bool
) -> None:
    """Verify `OpenAIAccount.model` reads cataloged cache support."""
    llm = OpenAIAccount(client=AsyncOpenAI(api_key="offline")).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is supported


@pytest.mark.parametrize("supported", [True, False])
def test_openai_account_model_accepts_an_uncataloged_model(*, supported: bool) -> None:
    """Pass uncataloged pricing and cache support unchanged."""
    llm = OpenAIAccount(client=AsyncOpenAI(api_key="offline")).model(
        "ft:gpt-5.6-terra:acme::abc123",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=supported,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == "ft:gpt-5.6-terra:acme::abc123"
    assert adapter.pricing is _ARBITRARY_PRICING
    assert adapter.supports_prompt_cache_options is supported


def test_openai_account_model_honors_a_stated_flag_on_a_cataloged_id() -> None:
    """Honor stated cache support for a cataloged model."""
    llm = OpenAIAccount(client=AsyncOpenAI(api_key="offline")).model(
        "gpt-5.6-terra",
        supports_prompt_cache_options=False,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is False


def test_anthropic_account_model_accepts_an_uncataloged_model() -> None:
    """A non-catalog id builds with caller-stated pricing, passed through rather than merged."""
    llm = AnthropicAccount(client=AsyncAnthropic(api_key="offline")).model(
        "claude-next-preview",
        pricing=_ARBITRARY_ANTHROPIC_PRICING,
    )
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == "claude-next-preview"
    assert adapter.pricing is _ARBITRARY_ANTHROPIC_PRICING


def test_gemini_account_model_accepts_an_uncataloged_model() -> None:
    """A non-catalog id builds with caller-stated pricing, passed through rather than merged."""
    llm = GeminiAccount(client=genai.Client(api_key="offline", vertexai=False)).model(
        "gemini-next-preview",
        pricing=_ARBITRARY_GEMINI_PRICING,
    )
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == "gemini-next-preview"
    assert adapter.pricing is _ARBITRARY_GEMINI_PRICING


def test_gemini_pricing_override_replaces_the_on_demand_rates() -> None:
    """A caller-supplied "ON_DEMAND" table replaces the catalog's, and a caller's tier is added."""
    custom = GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=2.0,
            cache_read_usd_per_million_tokens=0.2,
            output_usd_per_million_tokens=20.0,
        )
    )
    adapter = (
        GeminiAccount(client=genai.Client(api_key="offline", vertexai=False))
        .model(
            "gemini-2.5-flash",
            pricing={"ON_DEMAND": custom, "ON_DEMAND_FLEX": custom},
        )
        .adapter
    )
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.pricing["ON_DEMAND"] is custom
    assert adapter.pricing["ON_DEMAND_FLEX"] is custom


def test_gemini_adapter_requires_on_demand_pricing() -> None:
    """A pricing mapping without "ON_DEMAND" would price every tierless response NaN silently."""
    with pytest.raises(ValueError, match="ON_DEMAND"):
        _ = GeminiGenerateContentAdapter(
            client=genai.Client(api_key="offline", vertexai=False),
            model="gemini-2.5-flash",
            pricing={},
            provider_name="gcp.gemini",
        )


def test_gemini_account_model_raises_on_a_vertex_client() -> None:
    """Reject a Vertex AI client from `GeminiAccount.model`."""
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = GeminiAccount(client=genai.Client(api_key="offline", vertexai=True)).model(
            "gemini-2.5-flash"
        )


def test_the_gemini_adapter_accepts_a_vertex_client_under_its_own_name() -> None:
    """A vertexai client under provider_name "gcp.vertex_ai" is the stated Vertex construction."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=True),
        model="gemini-2.5-flash",
        pricing=_ARBITRARY_GEMINI_PRICING,
        provider_name="gcp.vertex_ai",
    )
    assert adapter.provider_name == "gcp.vertex_ai"


def test_gemini_account_owns_shared_backoff_and_bind_owns_max_attempts() -> None:
    """GeminiAccount shares request policy while bind owns max_attempts."""
    account = GeminiAccount(
        client=genai.Client(api_key="offline", vertexai=False),
        max_concurrent_requests=16,
        max_request_starts_per_second=25.0,
    )
    llm = account.model(
        "gemini-2.5-flash",
        service_tier="flex",
    )
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.service_tier == "flex"
    assert llm.shared_backoff.max_concurrent_requests == 16
    assert llm.shared_backoff.max_request_starts_per_second == 25.0
    assert account.model("gemini-2.5-pro").shared_backoff is llm.shared_backoff
    assert llm.bind(automatic_prompt_caching=False, max_attempts=5).max_attempts == 5
    defaulted = account.model("gemini-2.5-flash")
    defaulted_adapter = defaulted.adapter
    assert isinstance(defaulted_adapter, GeminiGenerateContentAdapter)
    assert defaulted_adapter.service_tier is None
    assert defaulted.bind(automatic_prompt_caching=False).max_attempts == 3


def _deepseek_client() -> AsyncOpenAI:
    """Return a keyless client pointed at DeepSeek, valid because no request is sent."""
    return AsyncOpenAI(api_key="offline", base_url="https://api.deepseek.com")


@pytest.mark.parametrize("model", list(DEEPSEEK_PRICING))
def test_deepseek_account_model_wires_model_pricing_and_the_cache_reader(
    model: DeepSeekModelName,
) -> None:
    """Wire DeepSeek pricing and its cache-read usage reader."""
    llm = DeepSeekAccount(client=_deepseek_client()).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == model
    assert adapter.pricing["default"] is DEEPSEEK_PRICING[model]
    assert adapter.cache_read_tokens_from_usage is cache_read_tokens_from_usage_deepseek
    assert adapter.supports_prompt_cache_options is False
    assert adapter.provider_name == "deepseek"


def test_deepseek_account_model_accepts_an_uncataloged_model() -> None:
    """A non-catalog id builds with caller-stated pricing, wrapped under the "default" tier."""
    table = _ARBITRARY_PRICING["default"]
    llm = DeepSeekAccount(client=_deepseek_client()).model(
        "deepseek-next-preview",
        pricing=table,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == "deepseek-next-preview"
    assert adapter.pricing["default"] is table


def test_deepseek_account_without_a_client_requires_the_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building without a client reads DEEPSEEK_API_KEY, raising when it is unset rather than falling back.

    The SDK's own fallback reads OPENAI_API_KEY, which would silently send the OpenAI key to
    api.deepseek.com.
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _ = DeepSeekAccount()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")
    adapter = DeepSeekAccount().model("deepseek-v4-flash").adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.client.api_key == "offline"
    assert str(adapter.client.base_url).startswith("https://api.deepseek.com")


def test_deepseek_account_owns_shared_backoff_and_bind_owns_max_attempts() -> None:
    """DeepSeekAccount shares request policy while bind owns max_attempts."""
    account = DeepSeekAccount(
        client=_deepseek_client(),
        max_concurrent_requests=16,
        max_request_starts_per_second=25.0,
    )
    llm = account.model("deepseek-v4-flash")
    assert llm.shared_backoff.max_concurrent_requests == 16
    assert llm.shared_backoff.max_request_starts_per_second == 25.0
    assert account.model("deepseek-v4-pro").shared_backoff is llm.shared_backoff
    assert llm.bind(automatic_prompt_caching=True, max_attempts=5).max_attempts == 5


@pytest.mark.parametrize("supported", [True, False])
def test_openai_bedrock_account_model_forwards_prompt_cache_options_support(
    *, supported: bool
) -> None:
    """Forward stated Bedrock cache-options support unchanged."""
    llm = OpenAIBedrockAccount(client=AsyncBedrockOpenAI(aws_region="us-east-1")).model(
        "openai.gpt-oss-120b-1:0",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=supported,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is supported


@pytest.mark.parametrize(
    ("build", "provider_name"),
    [
        (
            lambda: AnthropicMessagesAdapter(
                client=AsyncAnthropic(api_key="offline", base_url="https://example.invalid"),
                model="claude-sonnet-5",
                pricing=_ARBITRARY_ANTHROPIC_PRICING,
                provider_name="groq",
            ),
            "groq",
        ),
        (
            lambda: OpenAIResponsesAdapter(
                client=AsyncOpenAI(api_key="offline", base_url="https://example.invalid"),
                model="gpt-5.6-terra",
                pricing=_ARBITRARY_PRICING,
                provider_name="groq",
                supports_prompt_cache_options=False,
            ),
            "groq",
        ),
    ],
)
def test_a_base_client_takes_the_stated_provider_name(
    build: Callable[[], Adapter], provider_name: str
) -> None:
    """A base client carries no provider of its own, so the caller's value stands unchallenged.

    This is how an OpenAI-compatible endpoint is labeled with the provider it actually reaches.
    Mapping a base client class to its own provider would pass every other test here while
    turning this construction into a ValueError, so the acceptance needs its own assertion.
    """
    assert build().provider_name == provider_name


def test_openai_bedrock_account_model_wires_model_pricing_and_region() -> None:
    """OpenAIBedrockAccount builds AsyncBedrockOpenAI for aws_region.

    Other OpenAIBedrockAccount tests pass a client, so they cannot catch this.
    """
    llm = OpenAIBedrockAccount(aws_region="eu-west-1").model(
        "openai.gpt-oss-120b-1:0",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=False,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == "openai.gpt-oss-120b-1:0"
    assert adapter.pricing is _ARBITRARY_PRICING
    assert isinstance(adapter.client, AsyncBedrockOpenAI)
    assert adapter.client.aws_region == "eu-west-1"


def test_adapter_client_never_retries_beneath_langchaint() -> None:
    """The stored client is a max_retries=0 copy keeping the caller's credentials."""
    client = AsyncAnthropic(api_key="offline")
    llm = AnthropicAccount(client=client).model("claude-sonnet-5")
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.client.max_retries == 0
    assert adapter.client.api_key == client.api_key


# One model per Bedrock API exercises both SDK client classes.
@pytest.mark.parametrize("model", ["anthropic.claude-opus-4-8", "us.anthropic.claude-opus-4-6-v1"])
def test_bedrock_http_client_survives_the_retry_suppression_copy(
    model: AnthropicBedrockModelName,
) -> None:
    """Preserve a passed Bedrock client's custom transport while disabling retries."""
    http_client = httpx.AsyncClient()
    if model == "anthropic.claude-opus-4-8":
        client = AsyncAnthropicBedrockMantle(
            aws_region="us-east-1",
            http_client=http_client,
        )
    else:
        client = AsyncAnthropicBedrock(
            aws_region="us-east-1",
            http_client=http_client,
        )
    llm = AnthropicBedrockAccount(client=client).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert client.max_retries != 0
    assert adapter.client is not client
    assert adapter.client.max_retries == 0
    assert adapter.client._client is http_client
    asyncio.run(client.close())


def test_bedrock_rejects_client_and_http_client_together() -> None:
    """Passing both client and http_client raises: a passed client already owns its transport."""
    client = AsyncAnthropicBedrockMantle(aws_region="us-east-1")
    with pytest.raises(ValueError, match="http_client="):
        _ = AnthropicBedrockAccount(
            client=client,
            http_client=httpx.AsyncClient(),
        )


def test_both_bedrock_accounts_raise_on_a_region_beside_a_client() -> None:
    """Both Bedrock account constructors reject aws_region beside client."""
    with pytest.raises(ValueError, match="aws_region="):
        _ = AnthropicBedrockAccount(
            aws_region="eu-west-1",
            client=AsyncAnthropicBedrockMantle(aws_region="us-east-1"),
        )
    with pytest.raises(ValueError, match="aws_region="):
        _ = OpenAIBedrockAccount(
            aws_region="eu-west-1",
            client=AsyncBedrockOpenAI(aws_region="us-east-1"),
        )


def test_pricing_override_replaces_the_standard_rates() -> None:
    """A caller-supplied "standard" table replaces the catalog's, and a caller's tier is added."""
    custom_standard = AnthropicPricingTable(
        input_cache_none_usd_per_million_tokens=2.00,
        output_usd_per_million_tokens=10.00,
        cache_read_usd_per_million_tokens=0.20,
        cache_write_5m_usd_per_million_tokens=2.50,
        cache_write_1h_usd_per_million_tokens=4.00,
    )
    adapter = (
        AnthropicAccount(
            client=AsyncAnthropic(api_key="offline"),
        )
        .model(
            "claude-sonnet-5",
            pricing={"standard": custom_standard, "batch": custom_standard},
        )
        .adapter
    )
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.pricing["standard"] is custom_standard
    assert adapter.pricing["batch"] is custom_standard


def test_service_tier_reaches_each_first_party_adapter() -> None:
    """Each Account.model forwards its provider's service_tier.

    Neither Bedrock Account.model accepts service_tier.
    """
    anthropic_account = AnthropicAccount(client=AsyncAnthropic(api_key="offline"))
    anthropic_adapter = anthropic_account.model(
        "claude-sonnet-5",
        service_tier="standard_only",
    ).adapter
    assert isinstance(anthropic_adapter, AnthropicMessagesAdapter)
    assert anthropic_adapter.service_tier == "standard_only"
    anthropic_unstated = anthropic_account.model("claude-sonnet-5").adapter
    assert isinstance(anthropic_unstated, AnthropicMessagesAdapter)
    assert anthropic_unstated.service_tier is None
    openai_account = OpenAIAccount(client=AsyncOpenAI(api_key="offline"))
    openai_adapter = openai_account.model("gpt-5.6-terra", service_tier="flex").adapter
    assert isinstance(openai_adapter, OpenAIResponsesAdapter)
    assert openai_adapter.service_tier == "flex"
    unstated = openai_account.model("gpt-5.6-terra").adapter
    assert isinstance(unstated, OpenAIResponsesAdapter)
    assert unstated.service_tier is None


def test_openai_account_owns_shared_backoff_and_bind_owns_max_attempts() -> None:
    """OpenAIAccount shares request policy while bind owns max_attempts."""
    account = OpenAIAccount(
        client=AsyncOpenAI(api_key="offline"),
        max_concurrent_requests=16,
        max_request_starts_per_second=25.0,
    )
    llm = account.model("gpt-5.6-terra")
    assert llm.shared_backoff.max_concurrent_requests == 16
    assert llm.shared_backoff.max_request_starts_per_second == 25.0
    assert account.model("gpt-5.6-sol").shared_backoff is llm.shared_backoff
    assert llm.bind(automatic_prompt_caching=False, max_attempts=5).max_attempts == 5
    assert llm.bind(automatic_prompt_caching=False).max_attempts == 3


def test_reasoning_summary_lands_on_the_adapter() -> None:
    """A caller-supplied reasoning_summary reaches the adapter; the default is None."""
    account = OpenAIAccount(client=AsyncOpenAI(api_key="offline"))
    llm = account.model("gpt-5.6-terra", reasoning_summary="detailed")
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.reasoning_summary == "detailed"
    defaulted = account.model("gpt-5.6-terra")
    assert isinstance(defaulted.adapter, OpenAIResponsesAdapter)
    assert defaulted.adapter.reasoning_summary is None


def test_cache_ttl_lands_on_the_adapter() -> None:
    """A caller-supplied cache_ttl reaches the adapter; the default is "5m"."""
    account = AnthropicAccount(client=AsyncAnthropic(api_key="offline"))
    llm = account.model("claude-sonnet-5", cache_ttl="1h")
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.cache_ttl == "1h"
    defaulted = account.model("claude-sonnet-5")
    assert isinstance(defaulted.adapter, AnthropicMessagesAdapter)
    assert defaulted.adapter.cache_ttl == "5m"


@pytest.mark.parametrize(
    ("build_llm", "expected_provider_name"),
    [
        (
            lambda: AnthropicAccount(client=AsyncAnthropic(api_key="k")).model("claude-sonnet-5"),
            "anthropic",
        ),
        (
            lambda: AnthropicBedrockAccount(
                client=AsyncAnthropicBedrock(aws_region="us-east-1"),
            ).model("us.anthropic.claude-sonnet-4-6"),
            "aws.bedrock",
        ),
        (
            lambda: OpenAIAccount(client=AsyncOpenAI(api_key="k")).model("gpt-5.6-terra"),
            "openai",
        ),
        (
            lambda: GeminiAccount(client=genai.Client(api_key="k", vertexai=False)).model(
                "gemini-2.5-flash"
            ),
            "gcp.gemini",
        ),
        (
            lambda: OpenAIBedrockAccount(client=AsyncBedrockOpenAI(aws_region="us-east-1")).model(
                "openai.gpt-oss-120b-1:0",
                pricing=_ARBITRARY_PRICING,
                supports_prompt_cache_options=False,
            ),
            "aws.bedrock",
        ),
        (
            lambda: DeepSeekAccount(client=_deepseek_client()).model("deepseek-v4-flash"),
            "deepseek",
        ),
    ],
)
def test_each_account_model_states_a_convention_provider_name(
    build_llm: Callable[[], LLM], expected_provider_name: str
) -> None:
    """Require a convention value for each account's `provider_name`."""
    defined = {member.value for member in gen_ai_semconv.GenAiProviderNameValues}
    adapter = build_llm().adapter
    assert adapter.provider_name == expected_provider_name
    assert adapter.provider_name in defined


@pytest.mark.parametrize(
    "client",
    [
        AsyncBedrockOpenAI(aws_region="us-east-1"),
        AsyncAzureOpenAI(
            api_key="k", api_version="2024-02-01", azure_endpoint="https://x.openai.azure.com"
        ),
    ],
)
def test_openai_account_model_rejects_a_client_reaching_another_provider(
    client: AsyncOpenAI,
) -> None:
    """Reject another provider's SDK client from `OpenAIAccount.model`."""
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAIAccount(client=client).model("gpt-5.6-terra")


def test_deepseek_account_model_rejects_a_bedrock_client() -> None:
    """Reject a Bedrock SDK client from `DeepSeekAccount.model`."""
    client = AsyncBedrockOpenAI(aws_region="us-east-1")
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = DeepSeekAccount(client=client).model("deepseek-v4-flash")


@pytest.mark.parametrize(
    "client",
    [
        AsyncAnthropicBedrock(aws_region="us-east-1"),
        AsyncAnthropicBedrockMantle(aws_region="us-east-1"),
    ],
)
def test_the_adapter_raises_on_anthropic_over_a_bedrock_client(
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
) -> None:
    """Both Bedrock client classes contradict provider_name="anthropic".

    The Account.model annotations already stop this path.
    This test covers direct adapter construction.
    """
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = AnthropicMessagesAdapter(
            client=client,
            model="claude-sonnet-5",
            pricing=_ARBITRARY_ANTHROPIC_PRICING,
            provider_name="anthropic",
        )


def test_a_subclass_of_a_platform_client_raises_like_its_base() -> None:
    """Subclassing a platform client to add headers or auth is ordinary application code.

    provider_name_by_client_class holds no base client class, which is what lets the lookup use
    isinstance: matching by exact type instead would let this subclass through with "openai" and
    file every Bedrock-served span under openai.
    """

    class SigV4BedrockOpenAI(AsyncBedrockOpenAI):
        pass

    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAIAccount(client=SigV4BedrockOpenAI(aws_region="us-east-1")).model("gpt-5.6-terra")
