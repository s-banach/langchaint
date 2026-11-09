"""Configured provider model behavior and type inference."""

import asyncio
from collections.abc import Callable

import httpx2
import pytest
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel

from langchaint import LLM, ToolManager
from langchaint.anthropic import Anthropic, AnthropicBedrock
from langchaint.deepseek import DeepSeek
from langchaint.gemini import Gemini
from langchaint.openai import OPENAI_PRICING, OpenAI, OpenAIBedrock
from langchaint.span_parsing import OtelChatSpan, llm_from_otel


class _Answer(BaseModel):
    value: int


@pytest.mark.parametrize(
    "create",
    [
        lambda: OpenAI(client=AsyncOpenAI(api_key="offline")).llm(
            "gpt-5.6-terra", system_prompt="rules", max_requests=2
        ),
        lambda: Anthropic(client=AsyncAnthropic(api_key="offline")).llm(
            "claude-sonnet-5", max_completion_tokens=128, system_prompt="rules", max_requests=2
        ),
        lambda: Gemini(client=genai.Client(api_key="offline")).llm(
            "gemini-3.6-flash", system_prompt="rules", max_requests=2
        ),
        lambda: DeepSeek(
            client=AsyncOpenAI(api_key="offline", base_url="https://api.deepseek.com")
        ).llm("deepseek-v4-flash", system_prompt="rules", max_requests=2),
        lambda: OpenAIBedrock(aws_region="us-east-1").llm(
            "openai.gpt-oss-120b-1:0",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            system_prompt="rules",
            max_requests=2,
        ),
        lambda: AnthropicBedrock().llm(
            "anthropic.claude-sonnet-5",
            max_completion_tokens=128,
            system_prompt="rules",
            max_requests=2,
        ),
    ],
)
def test_bind_preserves_provider_and_omitted_settings(create: Callable[[], LLM]) -> None:
    llm = create()
    assert llm.binding.system_prompt == "rules"
    assert llm.max_requests == 2
    same = llm.bind()
    assert same is not llm
    assert type(same) is LLM
    assert same.binding == llm.binding
    assert same.config_fingerprint() == llm.config_fingerprint()
    assert same.shared_backoff is llm.shared_backoff
    assert same.adapter is llm.adapter
    tool_manager = ToolManager([])
    changed = llm.bind(response_format=_Answer, tools=tool_manager, temperature=0.5)
    assert type(changed) is LLM
    assert changed.response_format is _Answer
    assert changed.tool_manager is tool_manager
    assert changed.binding.system_prompt == "rules"
    assert changed.binding.max_completion_tokens == llm.binding.max_completion_tokens
    assert changed.binding.temperature == 0.5
    assert changed.max_requests == 2
    assert llm.response_format is str
    assert llm.tool_manager is None
    assert llm.binding.temperature is None
    cleared = changed.bind(response_format=str, tools=None, system_prompt=None)
    assert cleared.response_format is str
    assert cleared.tool_manager is None
    assert cleared.binding.system_prompt is None
    assert cleared.binding.temperature == 0.5
    span = OtelChatSpan.model_validate({
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": llm.adapter.provider_name,
        "gen_ai.request.model": llm.adapter.model,
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.temperature": 0.75,
        "gen_ai.output.type": "json",
    })
    reconstructed = llm_from_otel(span, llm=changed)
    assert type(reconstructed) is LLM
    assert reconstructed.response_format is _Answer
    assert reconstructed.tool_manager is tool_manager
    assert reconstructed.shared_backoff is llm.shared_backoff
    assert reconstructed.max_requests == 2
    assert reconstructed.binding.max_completion_tokens == 256
    assert reconstructed.binding.temperature == 0.75
    assert reconstructed.bind(response_format=str).response_format is str


@pytest.mark.parametrize("bedrock", [False, True])
@pytest.mark.parametrize("response_format", [str, _Answer])
def test_anthropic_bind_rejects_an_absent_token_limit_before_a_request(
    *, bedrock: bool, response_format: type[str] | type[_Answer]
) -> None:
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        raise AssertionError("Binding must finish before any request is sent")

    async def scenario() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handle_request)
        ) as http_client:
            llm = (
                AnthropicBedrock(http_client=http_client).llm(
                    "anthropic.claude-sonnet-5", max_completion_tokens=128
                )
                if bedrock
                else Anthropic(
                    client=AsyncAnthropic(api_key="offline", http_client=http_client)
                ).llm("claude-sonnet-5", max_completion_tokens=128)
            )
            with pytest.raises(ValueError, match="Anthropic requires max_completion_tokens"):
                _ = llm.bind(response_format=response_format, max_completion_tokens=None)
            assert requests == []
            assert llm.binding.max_completion_tokens == 128

    asyncio.run(scenario())
