"""Backend catalog wiring tests.

These tests cover identifiers, pricing objects, overrides, and client routing.
"""

import asyncio
import json
import pathlib
from collections.abc import Callable

import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from google import genai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

from langchaint import LLM
from langchaint.adapter import Adapter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    Anthropic,
    AnthropicBedrock,
    AnthropicBedrockModelName,
    AnthropicMessagesAdapter,
    AnthropicModelName,
    AnthropicPricedServiceTier,
    AnthropicPricingTable,
)
from langchaint.cohere import CohereBedrock
from langchaint.deepseek import (
    DEEPSEEK_PRICING,
    DeepSeek,
    DeepSeekModelName,
    cache_read_tokens_from_usage_deepseek,
)
from langchaint.gemini import (
    GEMINI_PRICING,
    Gemini,
    GeminiGenerateContentAdapter,
    GeminiModelName,
    GeminiPricingTable,
    GeminiRates,
)
from langchaint.openai import (
    OPENAI_PRICING,
    OpenAI,
    OpenAIBedrock,
    OpenAIChatCompletionsAdapter,
    OpenAIModelName,
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
)
from langchaint.openai.embedding_adapter import _OpenAIEmbeddingAdapter

_ARBITRARY_PRICING: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
    "default": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=1.0,
        cache_read_usd_per_million_tokens=1.0,
        cache_write_usd_per_million_tokens=1.0,
    )
}
"""Stands in where OpenAIBedrock.model requires unrelated pricing.

OpenAIBedrock.model has no default pricing catalog.
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
        ),
    )
}
"""The gemini counterpart of _ARBITRARY_PRICING, for adapters built without a catalog."""


@pytest.fixture(scope="module", name="provider_name_values")
def _provider_name_values_fixture() -> set[str]:
    path = pathlib.Path(__file__).parent / "semconv_genai" / "provider-name-values.json"
    payload: object = json.loads(path.read_text())
    assert isinstance(payload, list)
    values: set[str] = set()
    for value in payload:
        assert isinstance(value, str)
        values.add(value)
    assert len(values) == len(payload)
    return values


@pytest.mark.parametrize("model", list(ANTHROPIC_PRICING))
def test_anthropic_model_wires_model_and_pricing(model: AnthropicModelName) -> None:
    """Anthropic.model returns an adapter carrying catalog pricing."""
    llm = Anthropic(client=AsyncAnthropic(api_key="offline")).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == model
    assert adapter.pricing["standard"] is ANTHROPIC_PRICING[model]
    web_search_rate = adapter.pricing["standard"].web_search_usd_per_invocation
    assert web_search_rate is not None
    assert web_search_rate > 0


@pytest.mark.parametrize("model", list(GEMINI_PRICING))
def test_gemini_model_wires_model_and_pricing(model: GeminiModelName) -> None:
    """Gemini.model returns an adapter carrying catalog pricing."""
    llm = Gemini(client=genai.Client(api_key="offline", vertexai=False)).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == model
    assert adapter.pricing["ON_DEMAND"] is GEMINI_PRICING[model]
    google_search_rate = adapter.pricing["ON_DEMAND"].google_search_usd_per_query
    google_maps_rate = adapter.pricing["ON_DEMAND"].google_maps_usd_per_query
    assert google_search_rate is not None
    assert google_search_rate > 0
    assert google_maps_rate is not None
    assert google_maps_rate > 0


@pytest.mark.parametrize("model", list(OPENAI_PRICING))
def test_openai_model_wires_model_and_pricing(model: OpenAIModelName) -> None:
    """OpenAI.model returns an adapter carrying catalog pricing."""
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == model
    assert adapter.pricing["default"] is OPENAI_PRICING[model]
    web_search_rate = adapter.pricing["default"].web_search_usd_per_invocation
    assert web_search_rate is not None
    assert web_search_rate > 0


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


def test_gemini_catalog_contains_only_gemini_3_models() -> None:
    """Gemini provider-tool support begins with Gemini 3 models."""
    assert all(model.startswith("gemini-3") for model in GEMINI_PRICING)


@pytest.mark.parametrize(("model", "supported"), list(_PROMPT_CACHE_OPTIONS_SUPPORT.items()))
def test_openai_model_wires_prompt_cache_options_support(
    model: OpenAIModelName, *, supported: bool
) -> None:
    """Verify `OpenAI.model` reads cataloged cache support."""
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is supported


@pytest.mark.parametrize("supported", [True, False])
def test_openai_model_accepts_an_uncataloged_model(*, supported: bool) -> None:
    """Pass uncataloged pricing and cache support unchanged."""
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).model(
        "ft:gpt-5.6-terra:acme::abc123",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=supported,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == "ft:gpt-5.6-terra:acme::abc123"
    assert adapter.pricing is _ARBITRARY_PRICING
    assert adapter.supports_prompt_cache_options is supported


def test_openai_model_honors_a_stated_flag_on_a_cataloged_id() -> None:
    """Honor stated cache support for a cataloged model."""
    llm = OpenAI(client=AsyncOpenAI(api_key="offline")).model(
        "gpt-5.6-terra",
        supports_prompt_cache_options=False,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is False


def test_anthropic_model_accepts_an_uncataloged_model() -> None:
    """A non-catalog id builds with caller-stated pricing, passed through rather than merged."""
    llm = Anthropic(client=AsyncAnthropic(api_key="offline")).model(
        "claude-next-preview",
        pricing=_ARBITRARY_ANTHROPIC_PRICING,
    )
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == "claude-next-preview"
    assert adapter.pricing is _ARBITRARY_ANTHROPIC_PRICING


def test_gemini_model_accepts_an_uncataloged_model() -> None:
    """A non-catalog id builds with caller-stated pricing, passed through rather than merged."""
    llm = Gemini(client=genai.Client(api_key="offline", vertexai=False)).model(
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
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    )
    adapter = (
        Gemini(client=genai.Client(api_key="offline", vertexai=False))
        .model(
            "gemini-3.5-flash",
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
            model="gemini-3.5-flash",
            pricing={},
            provider_name="gcp.gemini",
        )


def test_gemini_model_raises_on_a_vertex_client() -> None:
    """Reject a Vertex AI client from `Gemini.model`."""
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = Gemini(client=genai.Client(api_key="offline", vertexai=True)).model("gemini-3.5-flash")


def test_the_gemini_adapter_accepts_a_vertex_client_under_its_own_name() -> None:
    """A vertexai client under provider_name "gcp.vertex_ai" is the stated Vertex construction."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=True),
        model="gemini-3.5-flash",
        pricing=_ARBITRARY_GEMINI_PRICING,
        provider_name="gcp.vertex_ai",
    )
    assert adapter.provider_name == "gcp.vertex_ai"


def test_gemini_shares_backoff_and_bind_owns_max_attempts() -> None:
    """`Gemini` shares `SharedBackoff`; `LLM.bind()` sets `max_attempts`."""
    gemini = Gemini(
        client=genai.Client(api_key="offline", vertexai=False),
        max_concurrent_requests=16,
        max_request_starts_per_second=25.0,
    )
    llm = gemini.model(
        "gemini-3.5-flash",
        service_tier="flex",
    )
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.service_tier == "flex"
    assert llm.shared_backoff.max_concurrent_requests == 16
    assert llm.shared_backoff.max_request_starts_per_second == 25.0
    assert gemini.model("gemini-3.1-pro-preview").shared_backoff is llm.shared_backoff
    assert llm.bind(automatic_prompt_caching=False, max_attempts=5).max_attempts == 5
    defaulted = gemini.model("gemini-3.5-flash")
    defaulted_adapter = defaulted.adapter
    assert isinstance(defaulted_adapter, GeminiGenerateContentAdapter)
    assert defaulted_adapter.service_tier is None
    assert defaulted.bind(automatic_prompt_caching=False).max_attempts == 3


def _deepseek_client() -> AsyncOpenAI:
    """Return a keyless client pointed at DeepSeek, valid because no request is sent."""
    return AsyncOpenAI(api_key="offline", base_url="https://api.deepseek.com")


@pytest.mark.parametrize("model", list(DEEPSEEK_PRICING))
def test_deepseek_model_wires_model_pricing_and_the_cache_reader(
    model: DeepSeekModelName,
) -> None:
    """Wire DeepSeek pricing and its cache-read usage reader."""
    llm = DeepSeek(client=_deepseek_client()).model(model)
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == model
    assert adapter.pricing["default"] is DEEPSEEK_PRICING[model]
    assert adapter.cache_read_tokens_from_usage is cache_read_tokens_from_usage_deepseek
    assert adapter.supports_prompt_cache_options is False
    assert adapter.provider_name == "deepseek"


def test_deepseek_model_accepts_an_uncataloged_model() -> None:
    """A non-catalog id builds with caller-stated pricing, wrapped under the "default" tier."""
    table = _ARBITRARY_PRICING["default"]
    llm = DeepSeek(client=_deepseek_client()).model(
        "deepseek-next-preview",
        pricing=table,
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == "deepseek-next-preview"
    assert adapter.pricing["default"] is table


def test_deepseek_without_a_client_requires_the_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building without a client reads DEEPSEEK_API_KEY, raising when it is unset rather than falling back.

    The SDK's own fallback reads OPENAI_API_KEY, which would silently send the OpenAI key to
    api.deepseek.com.
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _ = DeepSeek()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")
    adapter = DeepSeek().model("deepseek-v4-flash").adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.client.api_key == "offline"
    assert str(adapter.client.base_url).startswith("https://api.deepseek.com")


def test_deepseek_shares_backoff_and_bind_owns_max_attempts() -> None:
    """`DeepSeek` shares `SharedBackoff`; `LLM.bind()` sets `max_attempts`."""
    deepseek = DeepSeek(
        client=_deepseek_client(),
        max_concurrent_requests=16,
        max_request_starts_per_second=25.0,
    )
    llm = deepseek.model("deepseek-v4-flash")
    assert llm.shared_backoff.max_concurrent_requests == 16
    assert llm.shared_backoff.max_request_starts_per_second == 25.0
    assert deepseek.model("deepseek-v4-pro").shared_backoff is llm.shared_backoff
    assert llm.bind(automatic_prompt_caching=True, max_attempts=5).max_attempts == 5


@pytest.mark.parametrize("supported", [True, False])
def test_openai_bedrock_model_forwards_prompt_cache_options_support(*, supported: bool) -> None:
    """Forward stated Bedrock cache-options support unchanged."""
    llm = OpenAIBedrock(client=AsyncBedrockOpenAI(aws_region="us-east-1")).model(
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


def test_openai_bedrock_model_wires_model_pricing_and_region() -> None:
    """OpenAIBedrock builds AsyncBedrockOpenAI for aws_region.

    Other OpenAIBedrock tests pass a client, so they cannot catch this.
    """
    llm = OpenAIBedrock(aws_region="eu-west-1").model(
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
    llm = Anthropic(client=client).model("claude-sonnet-5")
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
    llm = AnthropicBedrock(client=client).model(model)
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
        _ = AnthropicBedrock(
            client=client,
            http_client=httpx.AsyncClient(),
        )


def test_both_bedrock_classes_raise_on_a_region_beside_a_client() -> None:
    """Both Bedrock constructors reject `aws_region` beside `client`."""
    with pytest.raises(ValueError, match="aws_region="):
        _ = AnthropicBedrock(
            aws_region="eu-west-1",
            client=AsyncAnthropicBedrockMantle(aws_region="us-east-1"),
        )
    with pytest.raises(ValueError, match="aws_region="):
        _ = OpenAIBedrock(
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
        web_search_usd_per_invocation=0.01,
    )
    adapter = (
        Anthropic(
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
    """Each first-party `model()` forwards its provider's `service_tier`.

    Neither Bedrock `model()` accepts `service_tier`.
    """
    anthropic = Anthropic(client=AsyncAnthropic(api_key="offline"))
    anthropic_adapter = anthropic.model(
        "claude-sonnet-5",
        service_tier="standard_only",
    ).adapter
    assert isinstance(anthropic_adapter, AnthropicMessagesAdapter)
    assert anthropic_adapter.service_tier == "standard_only"
    anthropic_unstated = anthropic.model("claude-sonnet-5").adapter
    assert isinstance(anthropic_unstated, AnthropicMessagesAdapter)
    assert anthropic_unstated.service_tier is None
    openai = OpenAI(client=AsyncOpenAI(api_key="offline"))
    openai_adapter = openai.model("gpt-5.6-terra", service_tier="flex").adapter
    assert isinstance(openai_adapter, OpenAIResponsesAdapter)
    assert openai_adapter.service_tier == "flex"
    unstated = openai.model("gpt-5.6-terra").adapter
    assert isinstance(unstated, OpenAIResponsesAdapter)
    assert unstated.service_tier is None


def test_openai_shares_client_and_shared_backoff() -> None:
    """Models from one `OpenAI` share its client and `SharedBackoff`."""
    client = AsyncOpenAI(api_key="offline")
    openai = OpenAI(
        client=client,
        max_concurrent_requests=16,
        max_request_starts_per_second=25.0,
        minimum_wait_ceiling_seconds=0.25,
        longest_wait_seconds=12.0,
        wait_multiplier=3.0,
        quiet_seconds_per_decay_step=9.0,
    )
    terra = openai.model("gpt-5.6-terra")
    sol = openai.model("gpt-5.6-sol")
    embedding_model = openai.embedding_model("text-embedding-3-small")

    assert isinstance(terra.adapter, OpenAIResponsesAdapter)
    assert isinstance(sol.adapter, OpenAIResponsesAdapter)
    assert isinstance(embedding_model._adapter, _OpenAIEmbeddingAdapter)
    assert terra.adapter.client is openai.client
    assert sol.adapter.client is openai.client
    assert embedding_model._adapter.client is openai.client
    assert terra.shared_backoff is sol.shared_backoff
    assert terra.shared_backoff is embedding_model._shared_backoff
    assert terra.shared_backoff.max_concurrent_requests == 16
    assert terra.shared_backoff.max_request_starts_per_second == 25.0
    assert terra.shared_backoff.minimum_wait_ceiling_seconds == 0.25
    assert terra.shared_backoff.longest_wait_seconds == 12.0
    assert terra.shared_backoff.wait_multiplier == 3.0
    assert terra.shared_backoff.quiet_seconds_per_decay_step == 9.0
    assert terra.bind(automatic_prompt_caching=False, max_attempts=5).max_attempts == 5
    assert terra.bind(automatic_prompt_caching=False).max_attempts == 3


def test_separate_openai_values_create_separate_shared_backoffs() -> None:
    """Separate `OpenAI` values create separate `SharedBackoff` values."""
    first = OpenAI(client=AsyncOpenAI(api_key="offline"))
    second = OpenAI(client=AsyncOpenAI(api_key="offline"))

    assert (
        first.model("gpt-5.6-terra").shared_backoff
        is not second.model("gpt-5.6-terra").shared_backoff
    )


@pytest.mark.parametrize(
    "backend_class",
    [Anthropic, AnthropicBedrock, CohereBedrock, DeepSeek, Gemini, OpenAI, OpenAIBedrock],
)
def test_backend_classes_expose_no_lifecycle_methods(backend_class: type[object]) -> None:
    """Backend classes expose no lifecycle methods."""
    assert not hasattr(backend_class, "aclose")
    assert not hasattr(backend_class, "__aenter__")
    assert not hasattr(backend_class, "__aexit__")


def test_reasoning_summary_lands_on_the_adapter() -> None:
    """A caller-supplied reasoning_summary reaches the adapter; the default is None."""
    openai = OpenAI(client=AsyncOpenAI(api_key="offline"))
    llm = openai.model("gpt-5.6-terra", reasoning_summary="detailed")
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.reasoning_summary == "detailed"
    defaulted = openai.model("gpt-5.6-terra")
    assert isinstance(defaulted.adapter, OpenAIResponsesAdapter)
    assert defaulted.adapter.reasoning_summary is None


def test_cache_ttl_lands_on_the_adapter() -> None:
    """A caller-supplied cache_ttl reaches the adapter; the default is "5m"."""
    anthropic = Anthropic(client=AsyncAnthropic(api_key="offline"))
    llm = anthropic.model("claude-sonnet-5", cache_ttl="1h")
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.cache_ttl == "1h"
    defaulted = anthropic.model("claude-sonnet-5")
    assert isinstance(defaulted.adapter, AnthropicMessagesAdapter)
    assert defaulted.adapter.cache_ttl == "5m"


@pytest.mark.parametrize(
    ("build_llm", "expected_provider_name"),
    [
        (
            lambda: Anthropic(client=AsyncAnthropic(api_key="k")).model("claude-sonnet-5"),
            "anthropic",
        ),
        (
            lambda: AnthropicBedrock(
                client=AsyncAnthropicBedrock(aws_region="us-east-1"),
            ).model("us.anthropic.claude-sonnet-4-6"),
            "aws.bedrock",
        ),
        (
            lambda: OpenAI(client=AsyncOpenAI(api_key="k")).model("gpt-5.6-terra"),
            "openai",
        ),
        (
            lambda: Gemini(client=genai.Client(api_key="k", vertexai=False)).model(
                "gemini-3.5-flash"
            ),
            "gcp.gemini",
        ),
        (
            lambda: OpenAIBedrock(client=AsyncBedrockOpenAI(aws_region="us-east-1")).model(
                "openai.gpt-oss-120b-1:0",
                pricing=_ARBITRARY_PRICING,
                supports_prompt_cache_options=False,
            ),
            "aws.bedrock",
        ),
        (
            lambda: DeepSeek(client=_deepseek_client()).model("deepseek-v4-flash"),
            "deepseek",
        ),
    ],
)
def test_each_model_states_a_convention_provider_name(
    build_llm: Callable[[], LLM], expected_provider_name: str, provider_name_values: set[str]
) -> None:
    """Require a convention value for each adapter's `provider_name`."""
    adapter = build_llm().adapter
    assert adapter.provider_name == expected_provider_name
    assert adapter.provider_name in provider_name_values


@pytest.mark.parametrize(
    "client",
    [
        AsyncBedrockOpenAI(aws_region="us-east-1"),
        AsyncAzureOpenAI(
            api_key="k", api_version="2024-02-01", azure_endpoint="https://x.openai.azure.com"
        ),
    ],
)
def test_openai_rejects_a_client_reaching_another_provider(
    client: AsyncOpenAI,
) -> None:
    """Reject another provider's client from both OpenAI request APIs."""
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAI(client=client).model("gpt-5.6-terra")
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = OpenAI(client=client).embedding_model("text-embedding-3-small")
    asyncio.run(client.close())


def test_deepseek_model_rejects_a_bedrock_client() -> None:
    """Reject a Bedrock SDK client from `DeepSeek.model`."""
    client = AsyncBedrockOpenAI(aws_region="us-east-1")
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = DeepSeek(client=client).model("deepseek-v4-flash")


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

    `Anthropic.model()` annotations already stop this path.
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
        _ = OpenAI(client=SigV4BedrockOpenAI(aws_region="us-east-1")).model("gpt-5.6-terra")
