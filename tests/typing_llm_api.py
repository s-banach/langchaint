"""Public caller contracts checked by CI's pyrefly run.

Expected rejection uses specific diagnostic codes; pyrefly reports unused ignores when a rejected call becomes accepted.
Functions are type examples and are not executed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, assert_type

from pydantic import BaseModel

from langchaint import (
    LLM,
    AdapterLLM,
    Generation,
    GenerationOutcomeRecord,
    PlainGeneration,
    StreamHandle,
    ToolCallGeneration,
    ToolManager,
    ToolSequence,
    llm_from_adapter,
)
from langchaint.anthropic import Anthropic, AnthropicBedrock, AnthropicBedrockLLM, AnthropicLLM
from langchaint.deepseek import DeepSeek, DeepSeekLLM
from langchaint.gemini import Gemini, GeminiLLM
from langchaint.openai import (
    OPENAI_PRICING,
    OpenAI,
    OpenAIBedrock,
    OpenAIBedrockLLM,
    OpenAILLM,
    OpenAIPricingTable,
)
from langchaint.span_parsing import OtelChatSpan, llm_from_otel

if TYPE_CHECKING:
    from langchaint.adapter import Adapter


class Answer(BaseModel):
    value: int


def provider_conformance[T: str | BaseModel, ToolsT: ToolManager | None](
    llm: OpenAILLM[T, ToolsT]
    | OpenAIBedrockLLM[T, ToolsT]
    | AnthropicLLM[T, ToolsT]
    | AnthropicBedrockLLM[T, ToolsT]
    | GeminiLLM[T, ToolsT]
    | DeepSeekLLM[T, ToolsT]
    | AdapterLLM[T, ToolsT],
) -> LLM[T, ToolsT]:
    return llm


def bind_protocol[T: str | BaseModel, ToolsT: ToolManager | None, NewT: str | BaseModel](
    llm: LLM[T, ToolsT],
    output: type[NewT],
    tools: ToolSequence,
) -> None:
    assert_type(llm.bind(), LLM[T, ToolsT])
    assert_type(llm.bind(tool_choice="none"), LLM[T, ToolsT])
    assert_type(llm.bind(response_format=output), LLM[NewT, ToolsT])
    assert_type(llm.bind(tools=tools), LLM[T, ToolManager])
    assert_type(llm.bind(tools=None), LLM[T, None])
    assert_type(llm.bind(response_format=output, tools=tools), LLM[NewT, ToolManager])
    assert_type(llm.bind(response_format=output, tools=None), LLM[NewT, None])
    assert_type(llm.bind(response_format=str), LLM[str, ToolsT])
    configured = llm.bind(
        response_format=Answer,
        tools=tools,
        system_prompt="Answer.",
        max_requests=2,
        max_completion_tokens=128,
    )
    assert_type(configured, LLM[Answer, ToolManager])
    assert_type(configured.bind(tool_choice="none").tool_manager, ToolManager)


def bind_optional_tools(
    llm: LLM[Answer], tools: ToolManager | None
) -> LLM[Answer, ToolManager | None]:
    return llm.bind(tools=tools)


def reject_bind_context[T: str | BaseModel](llm: LLM[str]) -> None:
    wrong_output: LLM[Answer] = llm.bind()  # pyrefly: ignore[bad-assignment]
    wrong_callable: Callable[[], LLM[Answer]] = llm.bind  # pyrefly: ignore[bad-assignment]
    wrong_generic: Callable[[], LLM[T]] = llm.bind  # pyrefly: ignore[bad-assignment]
    _ = llm.bind(max_completion_tokens=None)  # pyrefly: ignore[no-matching-overload]
    _ = wrong_output, wrong_callable, wrong_generic


def protocol_covariance(llm: LLM[Answer, ToolManager]) -> LLM[BaseModel, ToolManager | None]:
    return llm


def adapter_covariance(llm: AdapterLLM[Answer, ToolManager]) -> LLM[BaseModel, ToolManager | None]:
    return llm


def stream_output_covariance(stream: StreamHandle[Answer]) -> StreamHandle[BaseModel]:
    return stream


def stream_tools_covariance(
    stream: StreamHandle[Answer],
) -> StreamHandle[Answer, ToolCallGeneration]:
    return stream


async def consume[T: str | BaseModel](llm: LLM[T, ToolManager | None]) -> T | None:
    generation = await llm.generate_one("question")
    assert_type(generation, Generation[T])
    assert_type(await llm.generate_many(["question"]), Sequence[Generation[T]])
    assert_type(
        await llm.generate_many_records(["question"]), Sequence[GenerationOutcomeRecord[T]]
    )
    assert_type(await llm.stream_one("question").final(), Generation[T])
    if generation.kind == "plain":
        assert_type(generation.output, T)
        return generation.output
    return None


async def concrete_plain(llm: OpenAILLM[Answer]) -> None:
    assert_type(await llm.generate_one("question"), PlainGeneration[Answer])
    assert_type(await llm.generate_many(["question"]), list[PlainGeneration[Answer]])
    assert_type(await llm.stream_one("question").final(), PlainGeneration[Answer])


def reject_anthropic_token_limits(anthropic: Anthropic, bedrock: AnthropicBedrock) -> None:
    _ = anthropic.llm("claude-sonnet-5")  # pyrefly: ignore[no-matching-overload]
    _ = anthropic.llm("claude-sonnet-5", max_completion_tokens=None)  # pyrefly: ignore[bad-argument-type]
    _ = bedrock.llm("anthropic.claude-sonnet-5")  # pyrefly: ignore[no-matching-overload]
    _ = bedrock.llm("anthropic.claude-sonnet-5", max_completion_tokens=None)  # pyrefly: ignore[bad-argument-type]
    direct = anthropic.llm("claude-sonnet-5", max_completion_tokens=128)
    _ = direct.bind(max_completion_tokens=None)  # pyrefly: ignore[no-matching-overload]
    hosted = bedrock.llm("anthropic.claude-sonnet-5", max_completion_tokens=128)
    _ = hosted.bind(max_completion_tokens=None)  # pyrefly: ignore[no-matching-overload]


def openai_optional_token_limits(openai: OpenAI) -> None:
    assert_type(openai.llm("gpt-5.6-terra"), OpenAILLM[str, None])
    assert_type(openai.llm("gpt-5.6-terra", max_completion_tokens=None), OpenAILLM[str, None])


class _TokenLimitedFactory[ModelIdT: str, LLMT](Protocol):
    def __call__(self, model: ModelIdT, *, max_completion_tokens: int) -> LLMT: ...


class _PricedFactory[LLMT](Protocol):
    def __call__(
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
    ) -> LLMT: ...


def reject_contextual_output(
    *,
    openai: OpenAI,
    anthropic: Anthropic,
    openai_bedrock: OpenAIBedrock,
    anthropic_bedrock: AnthropicBedrock,
    gemini: Gemini,
    deepseek: DeepSeek,
) -> None:
    wrong_openai: Callable[[Literal["gpt-5.6-terra"]], OpenAILLM[Answer]] = openai.llm  # pyrefly: ignore[bad-assignment]
    wrong_anthropic: _TokenLimitedFactory[Literal["claude-sonnet-5"], AnthropicLLM[Answer]] = (
        anthropic.llm  # pyrefly: ignore[bad-assignment]
    )
    wrong_openai_bedrock: _PricedFactory[OpenAIBedrockLLM[Answer]] = openai_bedrock.llm  # pyrefly: ignore[bad-assignment]
    wrong_anthropic_bedrock: _TokenLimitedFactory[
        Literal["anthropic.claude-sonnet-5"], AnthropicBedrockLLM[Answer]
    ] = anthropic_bedrock.llm  # pyrefly: ignore[bad-assignment]
    wrong_gemini: Callable[[Literal["gemini-3.6-flash"]], GeminiLLM[Answer]] = gemini.llm  # pyrefly: ignore[bad-assignment]
    wrong_deepseek: Callable[[Literal["deepseek-v4-flash"]], DeepSeekLLM[Answer]] = deepseek.llm  # pyrefly: ignore[bad-assignment]
    wrong_adapter: Callable[[Adapter], AdapterLLM[Answer]] = llm_from_adapter  # pyrefly: ignore[bad-assignment]
    wrong_direct: OpenAILLM[Answer] = openai.llm("gpt-5.6-terra")  # pyrefly: ignore[bad-assignment]
    _ = (
        wrong_openai,
        wrong_anthropic,
        wrong_openai_bedrock,
        wrong_anthropic_bedrock,
        wrong_gemini,
        wrong_deepseek,
        wrong_adapter,
        wrong_direct,
    )


async def reject_unsound_protocol_use(llm: LLM[Answer], other: LLM[str, ToolManager]) -> None:
    wrong_output: LLM[str] = llm  # pyrefly: ignore[bad-assignment]
    wrong_tools: LLM[str] = other  # pyrefly: ignore[bad-assignment]
    generation = await llm.generate_one("question")
    _ = generation.output  # pyrefly: ignore[missing-attribute]
    batch = await llm.generate_many(["question"])
    batch.append(generation)  # pyrefly: ignore[missing-attribute]
    _ = wrong_output, wrong_tools


def openai_inference[T: str | BaseModel](
    provider: OpenAI, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = provider.llm("gpt-5.6-terra")
    assert_type(text, OpenAILLM[str, None])
    assert_type(provider.llm("gpt-5.6-terra", tools=[]), OpenAILLM[str, ToolManager])
    assert_type(provider.llm("gpt-5.6-terra", response_format=Answer), OpenAILLM[Answer, None])
    assert_type(
        provider.llm("gpt-5.6-terra", response_format=Answer, tools=tools),
        OpenAILLM[Answer, ToolManager],
    )
    generic = provider.llm("gpt-5.6-terra", response_format=output, tools=tools)
    assert_type(generic, OpenAILLM[T, ToolManager])
    assert_type(provider.llm("gpt-5.6-terra", response_format=output), OpenAILLM[T, None])
    assert_type(generic.bind(), OpenAILLM[T, ToolManager])
    assert_type(generic.bind(tools=None), OpenAILLM[T, None])
    assert_type(generic.bind(response_format=str), OpenAILLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), OpenAILLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), OpenAILLM[T, ToolManager])
    assert_type(text.bind(response_format=output), OpenAILLM[T, None])
    assert_type(text.bind(tools=tools), OpenAILLM[str, ToolManager])
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, OpenAILLM[T, ToolManager])
    assert_type(restored.bind(tools=None), OpenAILLM[T, None])
    return generic


def anthropic_inference[T: str | BaseModel](
    provider: Anthropic, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = provider.llm("claude-sonnet-5", max_completion_tokens=128)
    assert_type(text, AnthropicLLM[str, None])
    assert_type(
        provider.llm("claude-sonnet-5", max_completion_tokens=128, tools=[]),
        AnthropicLLM[str, ToolManager],
    )
    assert_type(
        provider.llm("claude-sonnet-5", max_completion_tokens=128, response_format=Answer),
        AnthropicLLM[Answer, None],
    )
    assert_type(
        provider.llm(
            "claude-sonnet-5", max_completion_tokens=128, response_format=Answer, tools=tools
        ),
        AnthropicLLM[Answer, ToolManager],
    )
    generic = provider.llm(
        "claude-sonnet-5", max_completion_tokens=128, response_format=output, tools=tools
    )
    assert_type(generic, AnthropicLLM[T, ToolManager])
    assert_type(
        provider.llm("claude-sonnet-5", max_completion_tokens=128, response_format=output),
        AnthropicLLM[T, None],
    )
    assert_type(generic.bind(), AnthropicLLM[T, ToolManager])
    assert_type(generic.bind(tools=None), AnthropicLLM[T, None])
    assert_type(generic.bind(response_format=str), AnthropicLLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), AnthropicLLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), AnthropicLLM[T, ToolManager])
    assert_type(text.bind(response_format=output), AnthropicLLM[T, None])
    assert_type(text.bind(tools=tools), AnthropicLLM[str, ToolManager])
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, AnthropicLLM[T, ToolManager])
    assert_type(restored.bind(tools=None), AnthropicLLM[T, None])
    return generic


def openai_bedrock_inference[T: str | BaseModel](
    provider: OpenAIBedrock, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = provider.llm(
        "custom", pricing=OPENAI_PRICING["gpt-5.6-terra"], supports_prompt_cache_options=False
    )
    assert_type(text, OpenAIBedrockLLM[str, None])
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            tools=[],
        ),
        OpenAIBedrockLLM[str, ToolManager],
    )
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            response_format=Answer,
        ),
        OpenAIBedrockLLM[Answer, None],
    )
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            response_format=Answer,
            tools=tools,
        ),
        OpenAIBedrockLLM[Answer, ToolManager],
    )
    generic = provider.llm(
        "custom",
        pricing=OPENAI_PRICING["gpt-5.6-terra"],
        supports_prompt_cache_options=False,
        response_format=output,
        tools=tools,
    )
    assert_type(generic, OpenAIBedrockLLM[T, ToolManager])
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            response_format=output,
        ),
        OpenAIBedrockLLM[T, None],
    )
    assert_type(generic.bind(), OpenAIBedrockLLM[T, ToolManager])
    assert_type(generic.bind(tools=None), OpenAIBedrockLLM[T, None])
    assert_type(generic.bind(response_format=str), OpenAIBedrockLLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), OpenAIBedrockLLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), OpenAIBedrockLLM[T, ToolManager])
    assert_type(text.bind(response_format=output), OpenAIBedrockLLM[T, None])
    assert_type(text.bind(tools=tools), OpenAIBedrockLLM[str, ToolManager])
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, OpenAIBedrockLLM[T, ToolManager])
    assert_type(restored.bind(tools=None), OpenAIBedrockLLM[T, None])
    return generic


def anthropic_bedrock_inference[T: str | BaseModel](
    provider: AnthropicBedrock, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = provider.llm("anthropic.claude-sonnet-5", max_completion_tokens=128)
    assert_type(text, AnthropicBedrockLLM[str, None])
    assert_type(
        provider.llm("anthropic.claude-sonnet-5", max_completion_tokens=128, tools=[]),
        AnthropicBedrockLLM[str, ToolManager],
    )
    assert_type(
        provider.llm(
            "anthropic.claude-sonnet-5", max_completion_tokens=128, response_format=Answer
        ),
        AnthropicBedrockLLM[Answer, None],
    )
    assert_type(
        provider.llm(
            "anthropic.claude-sonnet-5",
            max_completion_tokens=128,
            response_format=Answer,
            tools=tools,
        ),
        AnthropicBedrockLLM[Answer, ToolManager],
    )
    generic = provider.llm(
        "anthropic.claude-sonnet-5", max_completion_tokens=128, response_format=output, tools=tools
    )
    assert_type(generic, AnthropicBedrockLLM[T, ToolManager])
    assert_type(
        provider.llm(
            "anthropic.claude-sonnet-5", max_completion_tokens=128, response_format=output
        ),
        AnthropicBedrockLLM[T, None],
    )
    assert_type(generic.bind(), AnthropicBedrockLLM[T, ToolManager])
    assert_type(generic.bind(tools=None), AnthropicBedrockLLM[T, None])
    assert_type(generic.bind(response_format=str), AnthropicBedrockLLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), AnthropicBedrockLLM[str, None])
    assert_type(
        text.bind(response_format=output, tools=tools), AnthropicBedrockLLM[T, ToolManager]
    )
    assert_type(text.bind(response_format=output), AnthropicBedrockLLM[T, None])
    assert_type(text.bind(tools=tools), AnthropicBedrockLLM[str, ToolManager])
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, AnthropicBedrockLLM[T, ToolManager])
    assert_type(restored.bind(tools=None), AnthropicBedrockLLM[T, None])
    return generic


def gemini_inference[T: str | BaseModel](
    provider: Gemini, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = provider.llm("gemini-3.6-flash")
    assert_type(text, GeminiLLM[str, None])
    assert_type(provider.llm("gemini-3.6-flash", tools=[]), GeminiLLM[str, ToolManager])
    assert_type(provider.llm("gemini-3.6-flash", response_format=Answer), GeminiLLM[Answer, None])
    assert_type(
        provider.llm("gemini-3.6-flash", response_format=Answer, tools=tools),
        GeminiLLM[Answer, ToolManager],
    )
    generic = provider.llm("gemini-3.6-flash", response_format=output, tools=tools)
    assert_type(generic, GeminiLLM[T, ToolManager])
    assert_type(provider.llm("gemini-3.6-flash", response_format=output), GeminiLLM[T, None])
    assert_type(generic.bind(), GeminiLLM[T, ToolManager])
    assert_type(generic.bind(tools=None), GeminiLLM[T, None])
    assert_type(generic.bind(response_format=str), GeminiLLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), GeminiLLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), GeminiLLM[T, ToolManager])
    assert_type(text.bind(response_format=output), GeminiLLM[T, None])
    assert_type(text.bind(tools=tools), GeminiLLM[str, ToolManager])
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, GeminiLLM[T, ToolManager])
    assert_type(restored.bind(tools=None), GeminiLLM[T, None])
    return generic


def deepseek_inference[T: str | BaseModel](
    provider: DeepSeek, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = provider.llm("deepseek-v4-flash")
    assert_type(text, DeepSeekLLM[str, None])
    assert_type(provider.llm("deepseek-v4-flash", tools=[]), DeepSeekLLM[str, ToolManager])
    assert_type(
        provider.llm("deepseek-v4-flash", response_format=Answer), DeepSeekLLM[Answer, None]
    )
    assert_type(
        provider.llm("deepseek-v4-flash", response_format=Answer, tools=tools),
        DeepSeekLLM[Answer, ToolManager],
    )
    generic = provider.llm("deepseek-v4-flash", response_format=output, tools=tools)
    assert_type(generic, DeepSeekLLM[T, ToolManager])
    assert_type(provider.llm("deepseek-v4-flash", response_format=output), DeepSeekLLM[T, None])
    assert_type(generic.bind(), DeepSeekLLM[T, ToolManager])
    assert_type(generic.bind(tools=None), DeepSeekLLM[T, None])
    assert_type(generic.bind(response_format=str), DeepSeekLLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), DeepSeekLLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), DeepSeekLLM[T, ToolManager])
    assert_type(text.bind(response_format=output), DeepSeekLLM[T, None])
    assert_type(text.bind(tools=tools), DeepSeekLLM[str, ToolManager])
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, DeepSeekLLM[T, ToolManager])
    assert_type(restored.bind(tools=None), DeepSeekLLM[T, None])
    return generic


def adapter_inference[T: str | BaseModel](
    provider: Adapter, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = llm_from_adapter(provider)
    assert_type(text, AdapterLLM[str, None])
    assert_type(text, AdapterLLM)
    assert_type(text.tool_manager, None)
    assert_type(llm_from_adapter(provider, tools=[]), AdapterLLM[str, ToolManager])
    assert_type(llm_from_adapter(provider, response_format=Answer), AdapterLLM[Answer, None])
    assert_type(
        llm_from_adapter(provider, response_format=Answer, tools=tools),
        AdapterLLM[Answer, ToolManager],
    )
    generic = llm_from_adapter(provider, response_format=output, tools=tools)
    assert_type(generic, AdapterLLM[T, ToolManager])
    assert_type(generic.tool_manager, ToolManager)
    assert_type(llm_from_adapter(provider, response_format=output), AdapterLLM[T, None])
    assert_type(llm_from_adapter(provider, response_format=Answer), AdapterLLM[Answer])
    assert_type(generic.bind(), AdapterLLM[T, ToolManager])
    assert_type(generic.bind(system_prompt="s"), AdapterLLM[T, ToolManager])
    assert_type(generic.bind(tools=None), AdapterLLM[T, None])
    assert_type(generic.bind(response_format=str), AdapterLLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), AdapterLLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), AdapterLLM[T, ToolManager])
    assert_type(text.bind(response_format=output), AdapterLLM[T, None])
    assert_type(text.bind(tools=tools), AdapterLLM[str, ToolManager])
    assert_type(text.bind(tools=[]), AdapterLLM[str, ToolManager])
    assert_type(text.bind(response_format=output, tools=[]), AdapterLLM[T, ToolManager])
    assert_type(text.bind(response_format=output).bind(tools=tools), AdapterLLM[T, ToolManager])
    assert_type(text.bind(response_format=output).bind(tools=[]), AdapterLLM[T, ToolManager])
    provider_executed_tools = ({"type": "web_search"},)
    assert_type(text.bind(provider_executed_tools=provider_executed_tools), AdapterLLM[str, None])
    assert_type(
        generic.bind(tools=None).bind(provider_executed_tools=provider_executed_tools),
        AdapterLLM[T, None],
    )
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, AdapterLLM[T, ToolManager])
    assert_type(restored.bind(tools=None), AdapterLLM[T, None])
    return generic
