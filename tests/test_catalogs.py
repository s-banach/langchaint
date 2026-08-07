"""Wiring in the langchaint.anthropic, langchaint.deepseek, langchaint.gemini, and langchaint.openai catalogs.

The pricing values themselves are the one provider fact tests cannot verify;
what tests can catch is a catalog function wiring the wrong model identifier, the wrong prices,
or losing an override, a copy-paste error that would type-check and ship silently.
"""

from collections.abc import Callable

import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from google import genai
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai_semconv

from langchaint import LLM, SharedBackoff, TransientError
from langchaint.adapter import Adapter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    AnthropicBedrockModelName,
    AnthropicMessagesAdapter,
    AnthropicModelName,
    AnthropicPricedServiceTier,
    AnthropicPricingTable,
    anthropic_bedrock_model,
    anthropic_model,
)
from langchaint.deepseek import (
    DEEPSEEK_PRICING,
    DeepSeekModelName,
    cache_read_tokens_from_usage_deepseek,
    deepseek_model,
)
from langchaint.gemini import (
    GEMINI_PRICING,
    GeminiGenerateContentAdapter,
    GeminiModelName,
    GeminiPricingTable,
    GeminiRates,
    gemini_model,
    parse_gemini,
)
from langchaint.openai import (
    OPENAI_PRICING,
    OpenAIChatCompletionsAdapter,
    OpenAIModelName,
    OpenAIPricedServiceTier,
    OpenAIPricingTable,
    OpenAIResponsesAdapter,
    openai_bedrock_model,
    openai_model,
    parse_openai,
)

_ARBITRARY_PRICING: dict[OpenAIPricedServiceTier, OpenAIPricingTable] = {
    "default": OpenAIPricingTable(
        input_cache_none_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=1.0,
        cache_read_usd_per_million_tokens=1.0,
        cache_write_usd_per_million_tokens=1.0,
    )
}
"""Stands in wherever an openai constructor requires pricing but the assertion is about something else.

openai_bedrock_model has no catalog to default from, so its callers always supply a mapping.
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
def test_anthropic_model_wires_model_and_pricing(model: AnthropicModelName) -> None:
    """anthropic_model returns an LLM whose adapter carries the model's prices."""
    llm = anthropic_model(model, client=AsyncAnthropic(api_key="offline"))
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == model
    assert adapter.pricing["standard"] is ANTHROPIC_PRICING[model]


@pytest.mark.parametrize("model", list(GEMINI_PRICING))
def test_gemini_model_wires_model_and_pricing(model: GeminiModelName) -> None:
    """gemini_model returns an LLM whose adapter carries the model's prices."""
    llm = gemini_model(model, client=genai.Client(api_key="offline", vertexai=False))
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == model
    assert adapter.pricing["ON_DEMAND"] is GEMINI_PRICING[model]


@pytest.mark.parametrize("model", list(OPENAI_PRICING))
def test_openai_model_wires_model_and_pricing(model: OpenAIModelName) -> None:
    """openai_model returns an LLM whose adapter carries the model's prices."""
    llm = openai_model(model, client=AsyncOpenAI(api_key="offline"))
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
"""What openai_model is expected to pass for each cataloged model.

Spelled out rather than recomputed from PROMPT_CACHE_OPTIONS_MODELS, which would restate the
implementation and pass however that set were edited.
"""


def test_the_prompt_cache_options_expectations_cover_the_catalog() -> None:
    """Every cataloged model has an expected value, so adding one to OPENAI_PRICING fails here.

    Without this, a new model reaches openai_model untested and takes the absent-model branch,
    where bind(automatic_prompt_caching=False) raises for a model that accepts the parameter.
    """
    assert set(_PROMPT_CACHE_OPTIONS_SUPPORT) == set(OPENAI_PRICING)


@pytest.mark.parametrize(("model", "supported"), list(_PROMPT_CACHE_OPTIONS_SUPPORT.items()))
def test_openai_model_wires_prompt_cache_options_support(
    model: OpenAIModelName, *, supported: bool
) -> None:
    """openai_model reads the flag from PROMPT_CACHE_OPTIONS_MODELS, gpt-5.6 and later.

    A model dropped from that set, or misspelled in it, fails here instead of at every bind that
    declines caching on a model taking the parameter.
    """
    llm = openai_model(model, client=AsyncOpenAI(api_key="offline"))
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is supported


@pytest.mark.parametrize("supported", [True, False])
def test_openai_model_accepts_a_model_id_outside_the_catalog(*, supported: bool) -> None:
    """A non-catalog id (here a fine-tune) builds with caller-stated pricing and cache-options support.

    Both facts are required because no catalog maps the id, and the pricing mapping is passed
    through rather than merged, so the caller's "default" table is the one that prices.
    Both flag values are asserted so a truthiness test of the flag cannot pass as the
    is-None test that keeps a stated False.
    """
    llm = openai_model(
        "ft:gpt-5.6-terra:acme::abc123",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=supported,
        client=AsyncOpenAI(api_key="offline"),
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == "ft:gpt-5.6-terra:acme::abc123"
    assert adapter.pricing is _ARBITRARY_PRICING
    assert adapter.supports_prompt_cache_options is supported


def test_openai_model_requires_both_facts_outside_the_catalog() -> None:
    """Omitting pricing or supports_prompt_cache_options on a non-catalog id raises.

    The overloads already reject both calls at check time, hence the suppressions;
    the raises are what an untyped caller hits.
    """
    with pytest.raises(ValueError, match="OPENAI_PRICING"):
        _ = openai_model(
            "ft:gpt-5.6-terra:acme::abc123",  # pyrefly: ignore[bad-argument-type]
            client=AsyncOpenAI(api_key="offline"),
        )
    with pytest.raises(ValueError, match="pass supports_prompt_cache_options"):
        _ = openai_model(  # pyrefly: ignore[no-matching-overload]
            "ft:gpt-5.6-terra:acme::abc123",
            pricing=_ARBITRARY_PRICING,
            client=AsyncOpenAI(api_key="offline"),
        )


def test_openai_model_honors_a_stated_flag_on_a_cataloged_id() -> None:
    """A stated supports_prompt_cache_options wins over PROMPT_CACHE_OPTIONS_MODELS.

    The model is in that set, so the stated False proves the caller's value is kept
    rather than overwritten from the catalog.
    """
    llm = openai_model(
        "gpt-5.6-terra",
        supports_prompt_cache_options=False,
        client=AsyncOpenAI(api_key="offline"),
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is False


def test_anthropic_model_accepts_a_model_id_outside_the_catalog() -> None:
    """A non-catalog id builds with caller-stated pricing, passed through rather than merged."""
    llm = anthropic_model(
        "claude-next-preview",
        pricing=_ARBITRARY_ANTHROPIC_PRICING,
        client=AsyncAnthropic(api_key="offline"),
    )
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == "claude-next-preview"
    assert adapter.pricing is _ARBITRARY_ANTHROPIC_PRICING


def test_anthropic_model_requires_pricing_outside_the_catalog() -> None:
    """Omitting pricing on a non-catalog id raises.

    The overloads already reject the call at check time, hence the suppression;
    the raise is what an untyped caller hits.
    """
    with pytest.raises(ValueError, match="ANTHROPIC_PRICING"):
        _ = anthropic_model(  # pyrefly: ignore[no-matching-overload]
            "claude-next-preview",
            client=AsyncAnthropic(api_key="offline"),
        )


def test_gemini_model_accepts_a_model_id_outside_the_catalog() -> None:
    """A non-catalog id builds with caller-stated pricing, passed through rather than merged."""
    llm = gemini_model(
        "gemini-next-preview",
        pricing=_ARBITRARY_GEMINI_PRICING,
        client=genai.Client(api_key="offline", vertexai=False),
    )
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.model == "gemini-next-preview"
    assert adapter.pricing is _ARBITRARY_GEMINI_PRICING


def test_gemini_model_requires_pricing_outside_the_catalog() -> None:
    """Omitting pricing on a non-catalog id raises.

    The overloads already reject the call at check time, hence the suppression;
    the raise is what an untyped caller hits.
    """
    with pytest.raises(ValueError, match="GEMINI_PRICING"):
        _ = gemini_model(  # pyrefly: ignore[no-matching-overload]
            "gemini-next-preview",
            client=genai.Client(api_key="offline", vertexai=False),
        )


def test_gemini_pricing_override_replaces_the_on_demand_rates() -> None:
    """A caller-supplied "ON_DEMAND" table replaces the catalog's, and a caller's tier is added."""
    custom = GeminiPricingTable(
        rates=GeminiRates(
            input_cache_none_usd_per_million_tokens=2.0,
            cache_read_usd_per_million_tokens=0.2,
            output_usd_per_million_tokens=20.0,
        )
    )
    adapter = gemini_model(
        "gemini-2.5-flash",
        client=genai.Client(api_key="offline", vertexai=False),
        pricing={"ON_DEMAND": custom, "ON_DEMAND_FLEX": custom},
    ).adapter
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


def test_gemini_model_raises_on_a_vertex_client() -> None:
    """gemini_model states provider_name="gcp.gemini", so a vertexai client raises.

    The client is built in Vertex express mode (vertexai=True with an API key), which constructs
    offline; without the check every span of a Vertex-served request would report "gcp.gemini".
    """
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = gemini_model("gemini-2.5-flash", client=genai.Client(api_key="offline", vertexai=True))


def test_the_gemini_adapter_accepts_a_vertex_client_under_its_own_name() -> None:
    """A vertexai client under provider_name "gcp.vertex_ai" is the stated Vertex construction."""
    adapter = GeminiGenerateContentAdapter(
        client=genai.Client(api_key="offline", vertexai=True),
        model="gemini-2.5-flash",
        pricing=_ARBITRARY_GEMINI_PRICING,
        provider_name="gcp.vertex_ai",
    )
    assert adapter.provider_name == "gcp.vertex_ai"


def test_gemini_model_forwards_service_tier_backoff_and_attempts() -> None:
    """service_tier lands on the adapter, shared_backoff and max_attempts on the LLM."""
    shared_backoff = SharedBackoff(
        parse=parse_gemini, failure_types=(TransientError,), max_concurrent_requests=16
    )
    llm = gemini_model(
        "gemini-2.5-flash",
        client=genai.Client(api_key="offline", vertexai=False),
        service_tier="flex",
        shared_backoff=shared_backoff,
        max_attempts=5,
    )
    adapter = llm.adapter
    assert isinstance(adapter, GeminiGenerateContentAdapter)
    assert adapter.service_tier == "flex"
    assert llm.shared_backoff is shared_backoff
    assert llm.max_attempts == 5
    defaulted = gemini_model(
        "gemini-2.5-flash", client=genai.Client(api_key="offline", vertexai=False)
    )
    defaulted_adapter = defaulted.adapter
    assert isinstance(defaulted_adapter, GeminiGenerateContentAdapter)
    assert defaulted_adapter.service_tier is None
    assert defaulted.max_attempts == 3


def _deepseek_client() -> AsyncOpenAI:
    """Return a keyless client pointed at DeepSeek, valid because no request is sent."""
    return AsyncOpenAI(api_key="offline", base_url="https://api.deepseek.com")


@pytest.mark.parametrize("model", list(DEEPSEEK_PRICING))
def test_deepseek_model_wires_model_pricing_and_the_cache_reader(
    model: DeepSeekModelName,
) -> None:
    """deepseek_model returns an LLM whose adapter carries the model's prices and DeepSeek's usage reader.

    The reader assertion is the billing-relevant wiring: over the openai default every DeepSeek
    cache hit would price at the cache-miss rate.
    """
    llm = deepseek_model(model, client=_deepseek_client())
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == model
    assert adapter.pricing["default"] is DEEPSEEK_PRICING[model]
    assert adapter.cache_read_tokens_from_usage is cache_read_tokens_from_usage_deepseek
    assert adapter.supports_prompt_cache_options is False
    assert adapter.provider_name == "deepseek"


def test_deepseek_model_accepts_a_model_id_outside_the_catalog() -> None:
    """A non-catalog id builds with caller-stated pricing, wrapped under the "default" tier."""
    table = _ARBITRARY_PRICING["default"]
    llm = deepseek_model("deepseek-next-preview", pricing=table, client=_deepseek_client())
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.model == "deepseek-next-preview"
    assert adapter.pricing["default"] is table


def test_deepseek_model_requires_pricing_outside_the_catalog() -> None:
    """Omitting pricing on a non-catalog id raises.

    The overloads already reject the call at check time, hence the suppression;
    the raise is what an untyped caller hits.
    """
    with pytest.raises(ValueError, match="DEEPSEEK_PRICING"):
        _ = deepseek_model(  # pyrefly: ignore[no-matching-overload]
            "deepseek-next-preview", client=_deepseek_client()
        )


def test_deepseek_model_without_a_client_requires_the_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building without a client reads DEEPSEEK_API_KEY, raising when it is unset rather than falling back.

    The SDK's own fallback reads OPENAI_API_KEY, which would silently send the OpenAI key to
    api.deepseek.com.
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _ = deepseek_model("deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")
    adapter = deepseek_model("deepseek-v4-flash").adapter
    assert isinstance(adapter, OpenAIChatCompletionsAdapter)
    assert adapter.client.api_key == "offline"
    assert str(adapter.client.base_url).startswith("https://api.deepseek.com")


def test_deepseek_model_forwards_shared_backoff_and_max_attempts() -> None:
    """shared_backoff and max_attempts land on the LLM."""
    shared_backoff = SharedBackoff(
        parse=parse_openai, failure_types=(TransientError,), max_concurrent_requests=16
    )
    llm = deepseek_model(
        "deepseek-v4-flash",
        client=_deepseek_client(),
        shared_backoff=shared_backoff,
        max_attempts=5,
    )
    assert llm.shared_backoff is shared_backoff
    assert llm.max_attempts == 5


@pytest.mark.parametrize("supported", [True, False])
def test_openai_bedrock_model_forwards_prompt_cache_options_support(*, supported: bool) -> None:
    """The caller's value reaches the adapter, no Bedrock id being cataloged to derive it from.

    Both values are asserted because forwarding is the whole contract here: an implementation
    hardcoding either one satisfies every other Bedrock test, and hardcoding False would refuse
    every binding that declines caching.
    """
    llm = openai_bedrock_model(
        "openai.gpt-oss-120b-1:0",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=supported,
        client=AsyncBedrockOpenAI(aws_region="us-east-1"),
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
    """The default-client path builds AsyncBedrockOpenAI on the stated region.

    Dropping aws_region here would send every request to whatever region the AWS environment
    resolves, which no other assertion catches: the constructor's other tests all pass a client.
    """
    llm = openai_bedrock_model(
        "openai.gpt-oss-120b-1:0",
        pricing=_ARBITRARY_PRICING,
        supports_prompt_cache_options=False,
        aws_region="eu-west-1",
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
    llm = anthropic_model("claude-sonnet-5", client=client)
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.client.max_retries == 0
    assert adapter.client.api_key == client.api_key


# One wire model id per Bedrock API: anthropic.claude-opus-4-8 routes to "mantle",
# us.anthropic.claude-opus-4-6-v1 to "legacy".
# The transport-drop bug lives in each client class's own copy() override, so both classes are tested.
@pytest.mark.parametrize("model", ["anthropic.claude-opus-4-8", "us.anthropic.claude-opus-4-6-v1"])
def test_bedrock_http_client_survives_the_retry_suppression_copy(
    model: AnthropicBedrockModelName,
) -> None:
    """A custom httpx client passed to anthropic_bedrock_model reaches the stored adapter client.

    The two Bedrock client classes override copy() without reusing the existing transport (anthropic
    0.120.0), so a plain with_options(max_retries=0) drops it; the adapter re-feeds it, so a caller's
    loaded certs reach the wire. This asserts the injected client survives that copy, not a fresh default.
    """
    http_client = httpx.AsyncClient()
    llm = anthropic_bedrock_model(model, aws_region="us-east-1", http_client=http_client)
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.client.max_retries == 0
    assert adapter.client._client is http_client


def test_bedrock_rejects_client_and_http_client_together() -> None:
    """Passing both client and http_client raises: a passed client already owns its transport."""
    client = AsyncAnthropicBedrockMantle(aws_region="us-east-1")
    with pytest.raises(ValueError, match="http_client="):
        _ = anthropic_bedrock_model(
            "anthropic.claude-opus-4-8", client=client, http_client=httpx.AsyncClient()
        )


def test_both_bedrock_constructors_raise_on_a_region_beside_a_client() -> None:
    """A passed client carries its own region, so the aws_region beside it would be dropped.

    Silently, and every request would go to the client's region.
    Both constructors raise rather than rewrite a client the caller built, and rewriting is not
    uniformly available anyway: AsyncAnthropicBedrockMantle.copy(aws_region=...) sets the attribute
    and leaves base_url pointing at the original region, while AsyncBedrockOpenAI.copy recomputes it
    (anthropic 0.120.0, openai 2.45.0).
    """
    with pytest.raises(ValueError, match="aws_region="):
        _ = anthropic_bedrock_model(
            "anthropic.claude-opus-4-8",
            aws_region="eu-west-1",
            client=AsyncAnthropicBedrockMantle(aws_region="us-east-1"),
        )
    with pytest.raises(ValueError, match="aws_region="):
        _ = openai_bedrock_model(
            "openai.gpt-oss-120b-1:0",
            pricing=_ARBITRARY_PRICING,
            supports_prompt_cache_options=False,
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
    adapter = anthropic_model(
        "claude-sonnet-5",
        client=AsyncAnthropic(api_key="offline"),
        pricing={"standard": custom_standard, "batch": custom_standard},
    ).adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.pricing["standard"] is custom_standard
    assert adapter.pricing["batch"] is custom_standard


def test_service_tier_reaches_the_adapter_from_both_first_party_constructors() -> None:
    """Each constructor forwards the tier its provider's requests can ask for; None forwards None.

    Neither Bedrock constructor takes the parameter: the tiers are each provider's own platform's.
    """
    anthropic_adapter = anthropic_model(
        "claude-sonnet-5", client=AsyncAnthropic(api_key="offline"), service_tier="standard_only"
    ).adapter
    assert isinstance(anthropic_adapter, AnthropicMessagesAdapter)
    assert anthropic_adapter.service_tier == "standard_only"
    anthropic_unstated = anthropic_model(
        "claude-sonnet-5", client=AsyncAnthropic(api_key="offline")
    ).adapter
    assert isinstance(anthropic_unstated, AnthropicMessagesAdapter)
    assert anthropic_unstated.service_tier is None
    openai_adapter = openai_model(
        "gpt-5.6-terra", client=AsyncOpenAI(api_key="offline"), service_tier="flex"
    ).adapter
    assert isinstance(openai_adapter, OpenAIResponsesAdapter)
    assert openai_adapter.service_tier == "flex"
    unstated = openai_model("gpt-5.6-terra", client=AsyncOpenAI(api_key="offline")).adapter
    assert isinstance(unstated, OpenAIResponsesAdapter)
    assert unstated.service_tier is None


def test_shared_backoff_and_max_attempts_land_on_the_llm() -> None:
    """A caller-supplied SharedBackoff and max_attempts are the LLM's; omitting them means the LLM defaults."""
    shared_backoff = SharedBackoff(
        parse=parse_openai, failure_types=(TransientError,), max_concurrent_requests=16
    )
    llm = openai_model(
        "gpt-5.6-terra",
        client=AsyncOpenAI(api_key="offline"),
        shared_backoff=shared_backoff,
        max_attempts=5,
    )
    assert llm.shared_backoff is shared_backoff
    assert llm.max_attempts == 5
    defaulted = openai_model("gpt-5.6-terra", client=AsyncOpenAI(api_key="offline"))
    assert isinstance(defaulted.shared_backoff, SharedBackoff)
    assert defaulted.shared_backoff is not shared_backoff
    assert defaulted.max_attempts == 3


def test_reasoning_summary_lands_on_the_adapter() -> None:
    """A caller-supplied reasoning_summary reaches the adapter; the default is None."""
    llm = openai_model(
        "gpt-5.6-terra", client=AsyncOpenAI(api_key="offline"), reasoning_summary="detailed"
    )
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.reasoning_summary == "detailed"
    defaulted = openai_model("gpt-5.6-terra", client=AsyncOpenAI(api_key="offline"))
    assert isinstance(defaulted.adapter, OpenAIResponsesAdapter)
    assert defaulted.adapter.reasoning_summary is None


def test_cache_ttl_lands_on_the_adapter() -> None:
    """A caller-supplied cache_ttl reaches the adapter; the default is "5m"."""
    llm = anthropic_model(
        "claude-sonnet-5", client=AsyncAnthropic(api_key="offline"), cache_ttl="1h"
    )
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.cache_ttl == "1h"
    defaulted = anthropic_model("claude-sonnet-5", client=AsyncAnthropic(api_key="offline"))
    assert isinstance(defaulted.adapter, AnthropicMessagesAdapter)
    assert defaulted.adapter.cache_ttl == "5m"


@pytest.mark.parametrize(
    ("build_llm", "expected_provider_name"),
    [
        (
            lambda: anthropic_model("claude-sonnet-5", client=AsyncAnthropic(api_key="k")),
            "anthropic",
        ),
        (
            lambda: anthropic_bedrock_model(
                "us.anthropic.claude-sonnet-4-6",
                client=AsyncAnthropicBedrock(aws_region="us-east-1"),
            ),
            "aws.bedrock",
        ),
        (lambda: openai_model("gpt-5.6-terra", client=AsyncOpenAI(api_key="k")), "openai"),
        (
            lambda: gemini_model(
                "gemini-2.5-flash", client=genai.Client(api_key="k", vertexai=False)
            ),
            "gcp.gemini",
        ),
        (
            lambda: openai_bedrock_model(
                "openai.gpt-oss-120b-1:0",
                pricing=_ARBITRARY_PRICING,
                supports_prompt_cache_options=False,
                client=AsyncBedrockOpenAI(aws_region="us-east-1"),
            ),
            "aws.bedrock",
        ),
        (
            lambda: deepseek_model("deepseek-v4-flash", client=_deepseek_client()),
            "deepseek",
        ),
    ],
)
def test_each_constructor_states_a_convention_provider_name(
    build_llm: Callable[[], LLM], expected_provider_name: str
) -> None:
    """Every provider_name langchaint itself writes is a value the convention defines.

    The value reaches a backend as gen_ai.provider.name, whose value set the convention enumerates,
    so a typo like "bedrock" or a withdrawn value files langchaint's spans in their own bucket and
    joins them with no other instrumented client's.
    A direct adapter construction states its own provider_name and is the caller's to get right;
    what this pins is the set of literals langchaint writes on the application's behalf.
    """
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
def test_openai_model_raises_on_a_client_that_does_not_reach_openai(client: AsyncOpenAI) -> None:
    """openai_model states provider_name="openai", so it raises for the clients that reach elsewhere.

    Both classes subclass AsyncOpenAI, so the parameter annotation accepts them and only the
    adapter's provider_name_by_client_class check stops them; without it the adapter reports
    "openai" for a Bedrock- or Azure-served request and nothing surfaces the error until spans are
    grouped by provider.
    """
    with pytest.raises(ValueError, match="contradicts the client"):
        _ = openai_model("gpt-5.6-terra", client=client)


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
    """Both Bedrock client classes are mapped, so stating "anthropic" over either raises.

    Unlike the openai side, the annotations already stop this at the catalog constructors, since
    the Bedrock classes are siblings of AsyncAnthropic rather than subclasses. This covers the
    direct adapter construction, where nothing but the map stands between a Bedrock-served
    request and a span reporting "anthropic".
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
        _ = openai_model("gpt-5.6-terra", client=SigV4BedrockOpenAI(aws_region="us-east-1"))
