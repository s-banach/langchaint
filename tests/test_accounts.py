"""Account lifecycle, ownership, and shared request policy tests."""

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI

from langchaint import Account
from langchaint.account_base import AccountBase
from langchaint.anthropic import AnthropicAccount, AnthropicBedrockAccount
from langchaint.cohere import CohereBedrockAccount
from langchaint.deepseek import DeepSeekAccount
from langchaint.gemini import GeminiAccount
from langchaint.openai import OpenAIAccount, OpenAIBedrockAccount, OpenAIResponsesAdapter
from langchaint.openai.embedding_adapter import _OpenAIEmbeddingAdapter
from langchaint.shared_backoff import DoNotRetry


class _ProviderError(Exception):
    """A provider failure type for `AccountBase` tests."""


def _do_not_retry(_failure: Exception) -> DoNotRetry:
    """Return one terminal verdict for `AccountBase` tests."""
    return DoNotRetry()


def _primary_accounts_satisfy_protocol(
    openai_account: OpenAIAccount,
    anthropic_account: AnthropicAccount,
    deepseek_account: DeepSeekAccount,
    gemini_account: GeminiAccount,
    cohere_bedrock_account: CohereBedrockAccount,
    openai_bedrock_account: OpenAIBedrockAccount,
    anthropic_bedrock_account: AnthropicBedrockAccount,
) -> tuple[Account, ...]:
    """Make pyrefly check the common `Account` protocol."""
    return (
        openai_account,
        anthropic_account,
        deepseek_account,
        gemini_account,
        cohere_bedrock_account,
        openai_bedrock_account,
        anthropic_bedrock_account,
    )


def test_models_share_the_accounts_request_policy() -> None:
    """Models from one account share one configured `SharedBackoff`."""
    client = AsyncOpenAI(api_key="offline")
    account = OpenAIAccount(
        client=client,
        max_concurrent_requests=3,
        max_request_starts_per_second=7.5,
        minimum_wait_ceiling_seconds=0.25,
        longest_wait_seconds=12.0,
        wait_multiplier=3.0,
        quiet_seconds_per_decay_step=9.0,
    )
    first = account.model("gpt-5.4-mini")
    second = account.model("gpt-5.6-terra")
    embedding_model = account.embedding_model("text-embedding-3-small")

    assert first.shared_backoff is second.shared_backoff
    assert first.shared_backoff is embedding_model._shared_backoff
    shared_backoff = first.shared_backoff
    assert shared_backoff.max_concurrent_requests == 3
    assert shared_backoff.max_request_starts_per_second == 7.5
    assert shared_backoff.minimum_wait_ceiling_seconds == 0.25
    assert shared_backoff.longest_wait_seconds == 12.0
    assert shared_backoff.wait_multiplier == 3.0
    assert shared_backoff.quiet_seconds_per_decay_step == 9.0

    asyncio.run(client.close())


def test_models_share_the_accounts_sdk_client() -> None:
    """Models from one account share one retries-disabled SDK client."""
    passed_client = AsyncOpenAI(api_key="offline")
    account = OpenAIAccount(client=passed_client)
    first = account.model("gpt-5.4-mini")
    second = account.model("gpt-5.6-terra")
    embedding_model = account.embedding_model("text-embedding-3-small")

    assert isinstance(first.adapter, OpenAIResponsesAdapter)
    assert isinstance(second.adapter, OpenAIResponsesAdapter)
    assert isinstance(embedding_model._adapter, _OpenAIEmbeddingAdapter)
    assert first.adapter.client is account.client
    assert second.adapter.client is account.client
    assert embedding_model._adapter.client is account.client
    assert account.client.max_retries == 0

    asyncio.run(passed_client.close())


def test_separate_accounts_have_separate_request_policies() -> None:
    """Separate accounts construct separate `SharedBackoff` instances."""
    first_client = AsyncOpenAI(api_key="offline")
    second_client = AsyncOpenAI(api_key="offline")
    first = OpenAIAccount(client=first_client).model("gpt-5.4-mini")
    second = OpenAIAccount(client=second_client).model("gpt-5.4-mini")
    assert first.shared_backoff is not second.shared_backoff

    async def close_clients() -> None:
        await first_client.close()
        await second_client.close()

    asyncio.run(close_clients())


def test_aclose_closes_an_internally_created_client_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aclose()` owns an internally created SDK client and remains idempotent."""
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    account = OpenAIAccount()
    client = account.client
    assert not client.is_closed()

    async def scenario() -> None:
        await account.aclose()
        assert client.is_closed()
        await account.aclose()

    asyncio.run(scenario())


def test_gemini_aclose_closes_both_internal_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GeminiAccount.aclose()` closes its synchronous and asynchronous transports."""
    monkeypatch.setenv("GEMINI_API_KEY", "offline")
    account = GeminiAccount()
    sync_transport = account.client._api_client._httpx_client
    async_transport = account.client.aio._api_client._async_httpx_client
    assert sync_transport is not None
    assert async_transport is not None

    async def scenario() -> None:
        await account.aclose()
        assert sync_transport.is_closed
        assert async_transport.is_closed

    asyncio.run(scenario())


def test_concurrent_aclose_callers_wait_for_resource_closure() -> None:
    """Concurrent `aclose()` callers finish after shared resource closure."""
    account = AccountBase(
        parse=_do_not_retry,
        failure_types=(_ProviderError,),
        max_concurrent_requests=1,
        max_request_starts_per_second=50.0,
        minimum_wait_ceiling_seconds=1.0,
        longest_wait_seconds=60.0,
        wait_multiplier=2.0,
        quiet_seconds_per_decay_step=60.0,
    )

    async def scenario() -> None:
        close_started = asyncio.Event()
        permit_close = asyncio.Event()

        async def close_resource() -> None:
            close_started.set()
            await permit_close.wait()

        account._register_owned_close(close_resource)
        first = asyncio.create_task(account.aclose())
        await close_started.wait()
        second = asyncio.create_task(account.aclose())
        await asyncio.sleep(0)
        assert not second.done()
        permit_close.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())


def test_aclose_leaves_a_passed_client_open() -> None:
    """A passed SDK client remains caller-owned after `aclose()`."""
    client = AsyncOpenAI(api_key="offline")
    account = OpenAIAccount(client=client)

    async def scenario() -> None:
        await account.aclose()
        assert not client.is_closed()
        await client.close()

    asyncio.run(scenario())


def test_anthropic_bedrock_account_owns_a_passed_http_client() -> None:
    """`AnthropicBedrockAccount` closes its passed `http_client`."""
    http_client = httpx.AsyncClient()
    account = AnthropicBedrockAccount(http_client=http_client)

    async def scenario() -> None:
        await account.aclose()
        assert http_client.is_closed

    asyncio.run(scenario())


def test_context_entry_rejects_reentry_and_closes_on_exit() -> None:
    """One context entry succeeds and its exit closes the account."""
    client = AsyncOpenAI(api_key="offline")
    account = OpenAIAccount(client=client)

    async def scenario() -> None:
        async with account as entered:
            assert entered is account
            with pytest.raises(RuntimeError, match="already entered"):
                await account.__aenter__()
        with pytest.raises(RuntimeError, match="closed"):
            await account.__aenter__()
        await client.close()

    asyncio.run(scenario())


def test_account_model_binds_provider_executed_tools_inside_its_lifecycle() -> None:
    """Account-created models bind provider-executed tools during account use."""
    client = AsyncOpenAI(api_key="offline")
    account = OpenAIAccount(client=client)

    async def scenario() -> None:
        async with account:
            bound = account.model("gpt-5.4-mini").bind(
                provider_executed_tools=({"type": "web_search"},),
                automatic_prompt_caching=True,
            )
            assert bound.binding.provider_executed_tools == ({"type": "web_search"},)
        await client.close()

    asyncio.run(scenario())


def test_closed_account_rejects_models_and_created_bindings() -> None:
    """Closing rejects model construction and request starts."""
    client = AsyncOpenAI(api_key="offline")
    account = OpenAIAccount(client=client)
    bound = account.model("gpt-5.4-mini").bind(automatic_prompt_caching=True)
    embedding_model = account.embedding_model("text-embedding-3-small")

    async def scenario() -> None:
        await account.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            _ = account.model("gpt-5.4-mini")
        with pytest.raises(RuntimeError, match="closed"):
            _ = account.embedding_model("text-embedding-3-small")
        with pytest.raises(RuntimeError, match="closed"):
            _ = await bound.generate_one([])
        with pytest.raises(RuntimeError, match="closed"):
            async with bound.stream_one([]):
                pass
        with pytest.raises(RuntimeError, match="closed"):
            _ = await embedding_model.embed(["text"], task="classification")
        await client.close()

    asyncio.run(scenario())
