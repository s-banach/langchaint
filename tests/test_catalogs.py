"""Wiring in the langchaint.anthropic and langchaint.openai catalogs.

The pricing values themselves are the one provider fact tests cannot verify;
what tests can catch is a catalog function wiring the wrong model identifier, the wrong prices,
or losing an override, a copy-paste error that would type-check and ship silently.
"""

from collections.abc import Callable

import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai_semconv

from langchaint import LLM, PricingTable, RateLimiter
from langchaint.adapter import Adapter
from langchaint.anthropic import (
    ANTHROPIC_PRICING,
    AnthropicBedrockModelName,
    AnthropicMessagesAdapter,
    AnthropicModelName,
    anthropic_bedrock_model,
    anthropic_model,
)
from langchaint.openai import (
    OPENAI_PRICING,
    OpenAIModelName,
    OpenAIResponsesAdapter,
    openai_bedrock_model,
    openai_model,
)

_ARBITRARY_PRICING = PricingTable(
    input_cache_none_usd_per_million_tokens=1.0,
    output_usd_per_million_tokens=1.0,
    cache_read_usd_per_million_tokens=1.0,
    cache_write_usd_per_million_tokens=1.0,
)
"""Stands in wherever a constructor requires pricing but the assertion is about something else.

openai_bedrock_model has no catalog to default from, so its callers always supply a table.
"""


@pytest.mark.parametrize("model", list(ANTHROPIC_PRICING))
def test_anthropic_model_wires_model_and_pricing(model: AnthropicModelName) -> None:
    """anthropic_model returns an LLM whose adapter carries the model's prices."""
    llm = anthropic_model(model, client=AsyncAnthropic(api_key="offline"))
    adapter = llm.adapter
    assert isinstance(adapter, AnthropicMessagesAdapter)
    assert adapter.model == model
    assert adapter.pricing is ANTHROPIC_PRICING[model]


@pytest.mark.parametrize("model", list(OPENAI_PRICING))
def test_openai_model_wires_model_and_pricing(model: OpenAIModelName) -> None:
    """openai_model returns an LLM whose adapter carries the model's prices."""
    llm = openai_model(model, client=AsyncOpenAI(api_key="offline"))
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.model == model
    assert adapter.pricing is OPENAI_PRICING[model]


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

    Without this, a new model reaches openai_model untested and silently takes the absent-model
    branch, sending no caching parameter however openai prices or documents it.
    """
    assert set(_PROMPT_CACHE_OPTIONS_SUPPORT) == set(OPENAI_PRICING)


@pytest.mark.parametrize(("model", "supported"), list(_PROMPT_CACHE_OPTIONS_SUPPORT.items()))
def test_openai_model_wires_prompt_cache_options_support(
    model: OpenAIModelName, *, supported: bool
) -> None:
    """openai_model reads the flag from PROMPT_CACHE_OPTIONS_MODELS, gpt-5.6 and later.

    A model dropped from that set, or misspelled in it, fails here instead of silently sending
    no caching parameter for a model that takes one.
    """
    llm = openai_model(model, client=AsyncOpenAI(api_key="offline"))
    adapter = llm.adapter
    assert isinstance(adapter, OpenAIResponsesAdapter)
    assert adapter.supports_prompt_cache_options is supported


@pytest.mark.parametrize("supported", [True, False])
def test_openai_bedrock_model_forwards_prompt_cache_options_support(*, supported: bool) -> None:
    """The caller's value reaches the adapter, no Bedrock id being cataloged to derive it from.

    Both values are asserted because forwarding is the whole contract here: an implementation
    hardcoding either one satisfies every other Bedrock test, and hardcoding False would leave a
    caller who asked to stop caching paying for it on a model that bills cache writes.
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
                pricing=_ARBITRARY_PRICING,
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
    0.116.0), so a plain with_options(max_retries=0) drops it; the adapter re-feeds it, so a caller's
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
        anthropic_bedrock_model(
            "anthropic.claude-opus-4-8", client=client, http_client=httpx.AsyncClient()
        )


def test_both_bedrock_constructors_refuse_a_region_beside_a_client() -> None:
    """A passed client carries its own region, so the aws_region beside it would be dropped.

    Silently, and every request would go to the client's region.
    Both constructors raise rather than rewrite a client the caller built, and rewriting is not
    uniformly available anyway: AsyncAnthropicBedrockMantle.copy(aws_region=...) sets the attribute
    and leaves base_url pointing at the original region, while AsyncBedrockOpenAI.copy recomputes it
    (anthropic 0.116.0, openai 2.45.0).
    """
    with pytest.raises(ValueError, match="aws_region="):
        anthropic_bedrock_model(
            "anthropic.claude-opus-4-8",
            aws_region="eu-west-1",
            client=AsyncAnthropicBedrockMantle(aws_region="us-east-1"),
        )
    with pytest.raises(ValueError, match="aws_region="):
        openai_bedrock_model(
            "openai.gpt-oss-120b-1:0",
            pricing=_ARBITRARY_PRICING,
            supports_prompt_cache_options=False,
            aws_region="eu-west-1",
            client=AsyncBedrockOpenAI(aws_region="us-east-1"),
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
    assert llm.adapter.pricing is custom_pricing


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
            lambda: openai_bedrock_model(
                "openai.gpt-oss-120b-1:0",
                pricing=_ARBITRARY_PRICING,
                supports_prompt_cache_options=False,
                client=AsyncBedrockOpenAI(aws_region="us-east-1"),
            ),
            "aws.bedrock",
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
def test_openai_model_refuses_a_client_that_does_not_reach_openai(client: AsyncOpenAI) -> None:
    """openai_model states provider_name="openai", so it refuses the clients that reach elsewhere.

    Both classes subclass AsyncOpenAI, so the parameter annotation accepts them and only the
    adapter's provider_name_by_client_class check stops them; without it the adapter reports
    "openai" for a Bedrock- or Azure-served request and nothing surfaces the error until spans are
    grouped by provider.
    """
    with pytest.raises(ValueError, match="contradicts the client"):
        openai_model("gpt-5.6-terra", client=client)


@pytest.mark.parametrize(
    "client",
    [
        AsyncAnthropicBedrock(aws_region="us-east-1"),
        AsyncAnthropicBedrockMantle(aws_region="us-east-1"),
    ],
)
def test_the_adapter_refuses_anthropic_over_a_bedrock_client(
    client: AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle,
) -> None:
    """Both Bedrock client classes are mapped, so stating "anthropic" over either is refused.

    Unlike the openai side, the annotations already stop this at the catalog constructors, since
    the Bedrock classes are siblings of AsyncAnthropic rather than subclasses. This covers the
    direct adapter construction, where nothing but the map stands between a Bedrock-served
    request and a span reporting "anthropic".
    """
    with pytest.raises(ValueError, match="contradicts the client"):
        AnthropicMessagesAdapter(
            client=client,
            model="claude-sonnet-5",
            pricing=_ARBITRARY_PRICING,
            provider_name="anthropic",
        )


def test_a_subclass_of_a_platform_client_is_refused_like_its_base() -> None:
    """Subclassing a platform client to add headers or auth is ordinary application code.

    provider_name_by_client_class holds no base client class, which is what lets the lookup use
    isinstance: matching by exact type instead would let this subclass through with "openai" and
    file every Bedrock-served span under openai.
    """

    class SigV4BedrockOpenAI(AsyncBedrockOpenAI):
        pass

    with pytest.raises(ValueError, match="contradicts the client"):
        openai_model("gpt-5.6-terra", client=SigV4BedrockOpenAI(aws_region="us-east-1"))
