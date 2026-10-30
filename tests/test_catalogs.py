"""Backend catalog wiring tests."""

import math
import pathlib
from collections.abc import Callable
from typing import Literal

import httpx2
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from google import genai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from pydantic import TypeAdapter

from langchaint import LLM, JsonValue, SharedBackoff, TokenRates
from langchaint.adapter import Adapter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    Anthropic,
    AnthropicBedrock,
    AnthropicMessagesAdapter,
    AnthropicPricingTable,
    AnthropicRates,
)
from langchaint.deepseek import (
    DEEPSEEK_PRICING,
    DeepSeek,
    input_tokens_cache_read_from_usage_deepseek,
)
from langchaint.gemini import (
    GEMINI_PRICING,
    Gemini,
    GeminiGenerateContentAdapter,
    GeminiPricingTable,
    GeminiRates,
)
from langchaint.openai import (
    OPENAI_PRICING,
    OPENAI_REGIONAL_PROCESSING_MULTIPLIER,
    OpenAI,
    OpenAIBedrock,
    OpenAIChatCompletionsAdapter,
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
)
from langchaint.openai.embedding_adapter import _OpenAIEmbeddingAdapter
from tests.helpers import run_with_timeout

_ARBITRARY_RATES = TokenRates(
    input_tokens_cache_none=1.0,
    output_tokens=1.0,
    input_tokens_cache_read=1.0,
    input_tokens_cache_write=1.0,
)

_ARBITRARY_PRICING = OpenAIPricingTable(default=_ARBITRARY_RATES)

_ARBITRARY_ANTHROPIC_PRICING = AnthropicPricingTable(
    standard=AnthropicRates(
        input_tokens_cache_none=1.0,
        output_tokens=1.0,
        input_tokens_cache_read=1.0,
        input_tokens_cache_write_5m=1.0,
        input_tokens_cache_write_1h=1.0,
    )
)

_ARBITRARY_GEMINI_PRICING = GeminiPricingTable(
    on_demand=GeminiRates(
        input_tokens_cache_none=1.0,
        input_tokens_cache_read=1.0,
        output_tokens=1.0,
    )
)

_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


@pytest.fixture(scope="module", name="provider_name_values")
def _provider_name_values_fixture() -> set[str]:
    path = pathlib.Path(__file__).parent / "semconv_genai" / "chat-span-attributes.json"
    payload = _JSON_VALUE_ADAPTER.validate_json(path.read_text())
    assert isinstance(payload, dict)
    attributes = payload["attributes"]
    assert isinstance(attributes, list)
    provider_entries = [
        entry
        for entry in attributes
        if isinstance(entry, dict) and entry.get("name") == "gen_ai.provider.name"
    ]
    assert len(provider_entries) == 1
    allowed_values = provider_entries[0]["allowed_values"]
    assert isinstance(allowed_values, list)
    values: set[str] = set()
    for value in allowed_values:
        assert isinstance(value, str)
        values.add(value)
    assert len(values) == len(allowed_values)
    return values


def test_anthropic_llm_wires_model_and_pricing() -> None:
    model = "claude-sonnet-5"
    llm = Anthropic(client=AsyncAnthropic(api_key="offline")).llm(model)
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == model
    assert adapter.pricing is ANTHROPIC_PRICING[model]


def test_gemini_llm_wires_model_and_pricing() -> None:
    model = "gemini-3.5-flash"
    llm = Gemini(client=genai.Client(api_key="offline", vertexai=False)).llm(model)
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == model
    assert adapter.pricing is GEMINI_PRICING[model]


def test_openai_llm_wires_model_and_pricing() -> None:
    model = "gpt-5.6-terra"
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).llm(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == model
    assert adapter.pricing is OPENAI_PRICING[model]


def test_openai_llm_wires_cache_breakpoint_support() -> None:
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).llm(
        "gpt-5.6-terra",
        regional_processing=False,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_cache_breakpoints is True
    assert adapter.automatic_cache_breakpoints_default is False


@pytest.mark.parametrize("supported", [True, False])
def test_openai_llm_accepts_an_uncataloged_model(*, supported: bool) -> None:
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).llm(
        "ft:gpt-5.6-terra:acme::abc123",
        regional_processing=False,
        pricing=_ARBITRARY_PRICING,
        supports_cache_breakpoints=supported,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == "ft:gpt-5.6-terra:acme::abc123"
    assert adapter.pricing is _ARBITRARY_PRICING
    assert adapter.supports_cache_breakpoints is supported
    assert adapter.automatic_cache_breakpoints_default is not supported


def test_openai_llm_honors_a_stated_flag_on_a_cataloged_id() -> None:
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).llm(
        "gpt-5.6-terra",
        regional_processing=False,
        supports_cache_breakpoints=False,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_cache_breakpoints is False


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-next-preview"])
def test_gemini_llm_uses_stated_pricing(model: str) -> None:
    custom_rates = GeminiRates(
        input_tokens_cache_none=2.0,
        input_tokens_cache_read=0.2,
        output_tokens=20.0,
    )
    pricing = GeminiPricingTable(
        on_demand=custom_rates,
        on_demand_flex=custom_rates,
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    )
    adapter = (
        Gemini(client=genai.Client(api_key="offline", vertexai=False))
        .llm(model, pricing=pricing)
        .adapter
    )
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == model
    assert adapter.pricing is pricing


def test_gemini_llm_raises_on_a_vertex_client() -> None:
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = Gemini(client=genai.Client(api_key="offline", vertexai=True)).llm("gemini-3.5-flash")


def test_the_gemini_adapter_accepts_a_vertex_client_under_its_own_name() -> None:
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=True),
        model="gemini-3.5-flash",
        pricing=_ARBITRARY_GEMINI_PRICING,
        provider_name="gcp.vertex_ai",
    )
    assert adapter.provider_name == "gcp.vertex_ai"


def test_gemini_shares_backoff() -> None:
    shared_backoff = SharedBackoff()
    gemini = Gemini(
        client=genai.Client(api_key="offline", vertexai=False), shared_backoff=shared_backoff
    )
    llm = gemini.llm(
        "gemini-3.5-flash",
        service_tier="flex",
    )
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.service_tier == "flex"
    assert llm.shared_backoff is shared_backoff
    assert gemini.llm("gemini-3.1-pro-preview").shared_backoff is shared_backoff
    defaulted = gemini.llm("gemini-3.5-flash")
    defaulted_adapter = defaulted.adapter
    assert isinstance(defaulted_adapter, GeminiGenerateContentAdapter)
    assert defaulted_adapter.service_tier is None


def _deepseek_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="offline", base_url="https://api.deepseek.com")


def test_deepseek_llm_wires_model_pricing_and_the_cache_reader() -> None:
    model = "deepseek-v4-flash"
    llm = DeepSeek(client=_deepseek_client()).llm(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == model
    assert adapter.pricing.default is DEEPSEEK_PRICING[model]
    assert (
        adapter.input_tokens_cache_read_from_usage is input_tokens_cache_read_from_usage_deepseek
    )


def test_deepseek_llm_accepts_an_uncataloged_model() -> None:
    table = _ARBITRARY_PRICING.default
    llm = DeepSeek(client=_deepseek_client()).llm(
        "deepseek-next-preview",
        pricing=table,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == "deepseek-next-preview"
    assert adapter.pricing.default is table


def test_deepseek_without_a_client_requires_the_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building without a client reads DEEPSEEK_API_KEY, raising when it is unset rather than falling back.

    The SDK fallback would send OPENAI_API_KEY to api.deepseek.com.
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _ = DeepSeek()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")
    adapter = DeepSeek().llm("deepseek-v4-flash").adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.client.api_key == "offline"
    assert str(adapter.client.base_url).startswith("https://api.deepseek.com")


def test_deepseek_shares_backoff() -> None:
    shared_backoff = SharedBackoff()
    deepseek = DeepSeek(client=_deepseek_client(), shared_backoff=shared_backoff)
    assert deepseek.llm("deepseek-v4-flash").shared_backoff is shared_backoff
    assert deepseek.llm("deepseek-v4-pro").shared_backoff is shared_backoff


@pytest.mark.parametrize("supported", [True, False])
def test_openai_bedrock_llm_forwards_cache_breakpoint_support(*, supported: bool) -> None:
    llm = OpenAIBedrock(client=AsyncBedrockOpenAI(aws_region="us-east-1")).llm(
        "openai.gpt-oss-120b-1:0",
        pricing=_ARBITRARY_PRICING,
        supports_cache_breakpoints=supported,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_cache_breakpoints is supported
    assert adapter.automatic_cache_breakpoints_default is not supported


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
                supports_cache_breakpoints=False,
            ),
            "groq",
        ),
    ],
)
def test_a_base_client_takes_the_stated_provider_name(
    build: Callable[[], Adapter], provider_name: str
) -> None:
    assert build().provider_name == provider_name


def test_openai_bedrock_llm_wires_model_pricing_and_region() -> None:
    llm = OpenAIBedrock(aws_region="eu-west-1").llm(
        "openai.gpt-oss-120b-1:0",
        pricing=_ARBITRARY_PRICING,
        supports_cache_breakpoints=False,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == "openai.gpt-oss-120b-1:0"
    assert adapter.pricing is _ARBITRARY_PRICING
    assert isinstance(adapter.client, AsyncBedrockOpenAI)
    assert adapter.client.aws_region == "eu-west-1"


def test_adapter_client_never_retries_beneath_langchaint() -> None:
    client = AsyncAnthropic(api_key="offline")
    llm = Anthropic(client=client).llm("claude-sonnet-5")
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.client.max_retries == 0
    assert adapter.client.api_key == client.api_key


@pytest.mark.parametrize(
    ("api", "client_class"),
    [("legacy", AsyncAnthropicBedrock), ("mantle", AsyncAnthropicBedrockMantle)],
)
def test_bedrock_http_client_survives_the_retry_suppression_copy(
    api: Literal["legacy", "mantle"],
    client_class: type[AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle],
) -> None:
    http_client = httpx2.AsyncClient()
    client = client_class(aws_region="us-east-1", http_client=http_client)
    llm = AnthropicBedrock(api=api, client=client).llm("anthropic.claude-opus-4-8")
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert client.max_retries != 0
    assert adapter.client is not client
    assert adapter.client.max_retries == 0
    assert adapter.client._client is http_client
    run_with_timeout(client.close())


def test_bedrock_rejects_client_and_http_client_together() -> None:
    client = AsyncAnthropicBedrockMantle(aws_region="us-east-1")
    with pytest.raises(ValueError, match="http_client="):
        _ = AnthropicBedrock(
            api="mantle",
            client=client,
            http_client=httpx2.AsyncClient(),
        )


def test_openai_bedrock_rejects_a_region_beside_a_client() -> None:
    with pytest.raises(ValueError, match="aws_region="):
        _ = OpenAIBedrock(
            aws_region="eu-west-1",
            client=AsyncBedrockOpenAI(aws_region="us-east-1"),
        )


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-next-preview"])
def test_anthropic_llm_uses_stated_pricing(model: str) -> None:
    custom_standard = AnthropicRates(
        input_tokens_cache_none=2.00,
        output_tokens=10.00,
        input_tokens_cache_read=0.20,
        input_tokens_cache_write_5m=2.50,
        input_tokens_cache_write_1h=4.00,
    )
    replacement = AnthropicPricingTable(
        standard=custom_standard,
        batch=custom_standard,
        web_search_usd_per_invocation=0.01,
    )
    adapter = (
        Anthropic(
            client=AsyncAnthropic(api_key="offline"),
        )
        .llm(
            model,
            pricing=replacement,
        )
        .adapter
    )
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == model
    assert adapter.pricing is replacement


def test_service_tier_reaches_each_first_party_adapter() -> None:
    anthropic = Anthropic(client=AsyncAnthropic(api_key="offline"))
    anthropic_adapter = anthropic.llm(
        "claude-sonnet-5",
        service_tier="standard_only",
        inference_geo="us",
    ).adapter
    assert isinstance(anthropic_adapter, AnthropicMessagesAdapter)
    assert anthropic_adapter.service_tier == "standard_only"
    assert anthropic_adapter.inference_geo == "us"
    anthropic_unstated = anthropic.llm("claude-sonnet-5").adapter
    assert isinstance(anthropic_unstated, AnthropicMessagesAdapter)
    assert anthropic_unstated.service_tier is None
    openai = OpenAI(client=AsyncOpenAI(api_key="offline"))
    openai_adapter = openai.llm(
        "gpt-5.6-terra", regional_processing=False, service_tier="flex"
    ).adapter
    assert isinstance(openai_adapter, OpenAIResponsesAdapter)
    assert openai_adapter.service_tier == "flex"
    unstated = openai.llm("gpt-5.6-terra").adapter
    assert isinstance(unstated, OpenAIResponsesAdapter)
    assert unstated.service_tier is None


def test_openai_llm_regional_processing_scales_every_token_rate() -> None:
    openai = OpenAI(client=AsyncOpenAI(api_key="offline"))
    regional = openai.llm("gpt-5.6-terra", regional_processing=True).adapter
    assert isinstance(regional, OpenAIResponsesAdapter)
    assert regional.pricing == OPENAI_PRICING["gpt-5.6-terra"].scaled(
        OPENAI_REGIONAL_PROCESSING_MULTIPLIER["gpt-5.6-terra"]
    )
    unstated = openai.llm(
        "uncataloged-model",
        regional_processing=True,
        pricing=_ARBITRARY_PRICING,
        supports_cache_breakpoints=False,
    ).adapter
    assert isinstance(unstated, OpenAIResponsesAdapter)
    assert math.isnan(unstated.pricing.default.output_tokens)


def test_openai_shares_client_and_shared_backoff() -> None:
    client = AsyncOpenAI(api_key="offline")
    shared_backoff = SharedBackoff()
    openai = OpenAI(client=client, shared_backoff=shared_backoff)
    terra = openai.llm("gpt-5.6-terra", regional_processing=False)
    sol = openai.llm("gpt-5.6-sol", regional_processing=False)
    embedding_model = openai.embedding_model("text-embedding-3-small")

    assert isinstance(terra.adapter, OpenAIResponsesAdapter)
    assert isinstance(sol.adapter, OpenAIResponsesAdapter)
    assert isinstance(embedding_model._adapter, _OpenAIEmbeddingAdapter)
    assert terra.adapter.client is openai.client
    assert sol.adapter.client is openai.client
    assert embedding_model._adapter.client is openai.client
    assert terra.shared_backoff is shared_backoff
    assert sol.shared_backoff is shared_backoff
    assert embedding_model._shared_backoff is shared_backoff


def test_separate_openai_values_create_separate_shared_backoffs() -> None:
    first = OpenAI(client=AsyncOpenAI(api_key="offline"))
    second = OpenAI(client=AsyncOpenAI(api_key="offline"))

    assert (
        first.llm("gpt-5.6-terra", regional_processing=False).shared_backoff
        is not second.llm("gpt-5.6-terra", regional_processing=False).shared_backoff
    )


def test_reasoning_summary_lands_on_the_adapter() -> None:
    openai = OpenAI(client=AsyncOpenAI(api_key="offline"))
    llm = openai.llm("gpt-5.6-terra", regional_processing=False, reasoning_summary="detailed")
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.reasoning_summary == "detailed"
    defaulted = openai.llm("gpt-5.6-terra", regional_processing=False)
    assert isinstance(defaulted.adapter, OpenAIResponsesAdapter)
    assert defaulted.adapter.reasoning_summary is None


def test_cache_ttl_lands_on_the_adapter() -> None:
    anthropic = Anthropic(client=AsyncAnthropic(api_key="offline"))
    llm = anthropic.llm("claude-sonnet-5", cache_ttl="1h")
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.cache_ttl == "1h"
    defaulted = anthropic.llm("claude-sonnet-5")
    assert isinstance(defaulted.adapter, AnthropicMessagesAdapter)
    assert defaulted.adapter.cache_ttl == "5m"


@pytest.mark.parametrize(
    ("build_llm", "expected_provider_name"),
    [
        (
            lambda: Anthropic(client=AsyncAnthropic(api_key="offline")).llm("claude-sonnet-5"),
            "anthropic",
        ),
        (
            lambda: AnthropicBedrock(
                client=AsyncAnthropicBedrock(aws_region="us-east-1"),
            ).llm("us.anthropic.claude-sonnet-4-6"),
            "aws.bedrock",
        ),
        (
            lambda: OpenAI(client=AsyncOpenAI(api_key="offline")).llm(
                "gpt-5.6-terra", regional_processing=False
            ),
            "openai",
        ),
        (
            lambda: Gemini(client=genai.Client(api_key="offline", vertexai=False)).llm(
                "gemini-3.5-flash"
            ),
            "gcp.gemini",
        ),
        (
            lambda: OpenAIBedrock(client=AsyncBedrockOpenAI(aws_region="us-east-1")).llm(
                "openai.gpt-oss-120b-1:0",
                pricing=_ARBITRARY_PRICING,
                supports_cache_breakpoints=False,
            ),
            "aws.bedrock",
        ),
        (
            lambda: DeepSeek(client=_deepseek_client()).llm("deepseek-v4-flash"),
            "deepseek",
        ),
    ],
)
def test_each_model_states_a_convention_provider_name(
    build_llm: Callable[[], LLM], expected_provider_name: str, provider_name_values: set[str]
) -> None:
    adapter = build_llm().adapter
    assert adapter.provider_name == expected_provider_name
    assert adapter.provider_name in provider_name_values


@pytest.mark.parametrize(
    "build_client",
    [
        lambda: AsyncBedrockOpenAI(aws_region="us-east-1"),
        lambda: AsyncAzureOpenAI(
            api_key="offline",
            api_version="2024-02-01",
            azure_endpoint="https://x.openai.azure.com",
        ),
    ],
    ids=["bedrock", "azure"],
)
def test_openai_rejects_a_client_reaching_another_provider(
    build_client: Callable[[], AsyncOpenAI],
) -> None:
    client = build_client()
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAI(client=client).llm("gpt-5.6-terra", regional_processing=False)
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAI(client=client).embedding_model("text-embedding-3-small")
    run_with_timeout(client.close())


def test_deepseek_llm_rejects_a_bedrock_client() -> None:
    client = AsyncBedrockOpenAI(aws_region="us-east-1")
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = DeepSeek(client=client).llm("deepseek-v4-flash")


@pytest.mark.parametrize(
    "client_class",
    [
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
    ],
)
def test_the_adapter_raises_on_anthropic_over_a_bedrock_client(
    client_class: type[AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle],
) -> None:
    client = client_class(aws_region="us-east-1")
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = AnthropicMessagesAdapter(
            client=client,
            model="claude-sonnet-5",
            pricing=_ARBITRARY_ANTHROPIC_PRICING,
            provider_name="anthropic",
        )


def test_a_subclass_of_a_platform_client_raises_like_its_base() -> None:
    class SigV4BedrockOpenAI(AsyncBedrockOpenAI):
        pass

    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAI(client=SigV4BedrockOpenAI(aws_region="us-east-1")).llm(
            "gpt-5.6-terra", regional_processing=False
        )
