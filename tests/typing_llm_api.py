"""Public caller contracts checked by CI's pyrefly run.

Expected rejection uses specific diagnostic codes; pyrefly reports unused ignores when a rejected call becomes accepted.
Functions are type examples and are not executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, assert_type

from pydantic import BaseModel

from langchaint import (
    LLM,
    Generation,
    GenerationError,
    GenerationErrorRecord,
    GenerationErrorStopReason,
    GenerationOutcomeRecord,
    PlainErrorRecord,
    PlainGeneration,
    PlainGenerationRecord,
    StreamHandle,
    ToolCallGeneration,
    ToolManager,
    ToolSequence,
    llm_from_adapter,
)
from langchaint.openai import OPENAI_PRICING, OpenAI, OpenAIBedrock, OpenAIPricingTable
from langchaint.span_parsing import OtelChatSpan, llm_from_otel

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchaint.adapter import Adapter
    from langchaint.anthropic import Anthropic, AnthropicBedrock
    from langchaint.deepseek import DeepSeek
    from langchaint.gemini import Gemini


class Answer(BaseModel):
    value: int


def error_stop_reason(error: GenerationError, record: PlainErrorRecord) -> None:
    assert_type(error.stop_reason, GenerationErrorStopReason | None)
    assert_type(record.stop_reason, GenerationErrorStopReason | None)


def bind[T: str | BaseModel, ToolsT: ToolManager | None, NewT: str | BaseModel](
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
    _ = llm.bind(max_completion_tokens=None)
    _ = wrong_output, wrong_callable, wrong_generic


def llm_covariance(llm: LLM[Answer, ToolManager]) -> LLM[BaseModel, ToolManager | None]:
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
    assert_type(
        await llm.generate_many(["question"]), list[Generation[T]] | list[PlainGeneration[T]]
    )
    assert_type(
        await llm.generate_many_records(["question"]),
        list[GenerationOutcomeRecord[T]] | list[PlainGenerationRecord[T] | GenerationErrorRecord],
    )
    assert_type(await llm.stream_one("question").final(), Generation[T])
    if generation.kind == "plain":
        assert_type(generation.output, T)
        return generation.output
    return None


async def concrete_plain(llm: LLM[Answer]) -> None:
    assert_type(await llm.generate_one("question"), PlainGeneration[Answer])
    assert_type(await llm.generate_many(["question"]), list[PlainGeneration[Answer]])
    assert_type(await llm.stream_one("question").final(), PlainGeneration[Answer])


def reject_anthropic_token_limits(anthropic: Anthropic, bedrock: AnthropicBedrock) -> None:
    _ = anthropic.llm("claude-sonnet-5")  # pyrefly: ignore[no-matching-overload]
    _ = anthropic.llm("claude-sonnet-5", max_completion_tokens=None)  # pyrefly: ignore[bad-argument-type]
    _ = bedrock.llm("anthropic.claude-sonnet-5")  # pyrefly: ignore[no-matching-overload]
    _ = bedrock.llm("anthropic.claude-sonnet-5", max_completion_tokens=None)  # pyrefly: ignore[bad-argument-type]
    direct = anthropic.llm("claude-sonnet-5", max_completion_tokens=128)
    _ = direct.bind(max_completion_tokens=None)
    hosted = bedrock.llm("anthropic.claude-sonnet-5", max_completion_tokens=128)
    _ = hosted.bind(max_completion_tokens=None)


def openai_optional_token_limits(openai: OpenAI) -> None:
    assert_type(openai.llm("gpt-5.6-terra"), LLM[str, None])
    assert_type(openai.llm("gpt-5.6-terra", max_completion_tokens=None), LLM[str, None])


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
    wrong_openai: Callable[[Literal["gpt-5.6-terra"]], LLM[Answer]] = openai.llm  # pyrefly: ignore[bad-assignment]
    wrong_anthropic: _TokenLimitedFactory[Literal["claude-sonnet-5"], LLM[Answer]] = anthropic.llm  # pyrefly: ignore[bad-assignment]
    wrong_openai_bedrock: _PricedFactory[LLM[Answer]] = openai_bedrock.llm  # pyrefly: ignore[bad-assignment]
    wrong_anthropic_bedrock: _TokenLimitedFactory[
        Literal["anthropic.claude-sonnet-5"], LLM[Answer]
    ] = anthropic_bedrock.llm  # pyrefly: ignore[bad-assignment]
    wrong_gemini: Callable[[Literal["gemini-3.6-flash"]], LLM[Answer]] = gemini.llm  # pyrefly: ignore[bad-assignment]
    wrong_deepseek: Callable[[Literal["deepseek-v4-flash"]], LLM[Answer]] = deepseek.llm  # pyrefly: ignore[bad-assignment]
    wrong_adapter: Callable[[Adapter], LLM[Answer]] = llm_from_adapter  # pyrefly: ignore[bad-assignment]
    wrong_direct: LLM[Answer] = openai.llm("gpt-5.6-terra")  # pyrefly: ignore[bad-assignment]
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


async def reject_unsound_llm_use(llm: LLM[Answer], other: LLM[str, ToolManager]) -> None:
    wrong_output: LLM[str] = llm  # pyrefly: ignore[bad-assignment]
    wrong_tools: LLM[str] = other  # pyrefly: ignore[bad-assignment]
    generation = await llm.generate_one("question")
    assert_type(generation.output, Answer)
    batch = await llm.generate_many(["question"])
    batch.append(generation)
    _ = wrong_output, wrong_tools


def openai_inference[T: str | BaseModel](
    provider: OpenAI, output: type[T], tools: ToolManager
) -> LLM[T, ToolManager]:
    text = provider.llm("gpt-5.6-terra")
    assert_type(text, LLM[str, None])
    assert_type(provider.llm("gpt-5.6-terra", tools=[]), LLM[str, ToolManager])
    assert_type(provider.llm("gpt-5.6-terra", response_format=Answer), LLM[Answer, None])
    assert_type(
        provider.llm("gpt-5.6-terra", response_format=Answer, tools=tools),
        LLM[Answer, ToolManager],
    )
    generic = provider.llm("gpt-5.6-terra", response_format=output, tools=tools)
    assert_type(generic, LLM[T, ToolManager])
    assert_type(provider.llm("gpt-5.6-terra", response_format=output), LLM[T, None])
    return generic


def anthropic_inference[T: str | BaseModel](
    provider: Anthropic, output: type[T], tools: ToolManager
) -> LLM[T, ToolManager]:
    text = provider.llm("claude-sonnet-5", max_completion_tokens=128)
    assert_type(text, LLM[str, None])
    assert_type(
        provider.llm("claude-sonnet-5", max_completion_tokens=128, tools=[]),
        LLM[str, ToolManager],
    )
    assert_type(
        provider.llm("claude-sonnet-5", max_completion_tokens=128, response_format=Answer),
        LLM[Answer, None],
    )
    assert_type(
        provider.llm(
            "claude-sonnet-5", max_completion_tokens=128, response_format=Answer, tools=tools
        ),
        LLM[Answer, ToolManager],
    )
    generic = provider.llm(
        "claude-sonnet-5", max_completion_tokens=128, response_format=output, tools=tools
    )
    assert_type(generic, LLM[T, ToolManager])
    assert_type(
        provider.llm("claude-sonnet-5", max_completion_tokens=128, response_format=output),
        LLM[T, None],
    )
    return generic


def openai_bedrock_inference[T: str | BaseModel](
    provider: OpenAIBedrock, output: type[T], tools: ToolManager
) -> LLM[T, ToolManager]:
    text = provider.llm(
        "custom", pricing=OPENAI_PRICING["gpt-5.6-terra"], supports_prompt_cache_options=False
    )
    assert_type(text, LLM[str, None])
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            tools=[],
        ),
        LLM[str, ToolManager],
    )
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            response_format=Answer,
        ),
        LLM[Answer, None],
    )
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            response_format=Answer,
            tools=tools,
        ),
        LLM[Answer, ToolManager],
    )
    generic = provider.llm(
        "custom",
        pricing=OPENAI_PRICING["gpt-5.6-terra"],
        supports_prompt_cache_options=False,
        response_format=output,
        tools=tools,
    )
    assert_type(generic, LLM[T, ToolManager])
    assert_type(
        provider.llm(
            "custom",
            pricing=OPENAI_PRICING["gpt-5.6-terra"],
            supports_prompt_cache_options=False,
            response_format=output,
        ),
        LLM[T, None],
    )
    return generic


def anthropic_bedrock_inference[T: str | BaseModel](
    provider: AnthropicBedrock, output: type[T], tools: ToolManager
) -> LLM[T, ToolManager]:
    text = provider.llm("anthropic.claude-sonnet-5", max_completion_tokens=128)
    assert_type(text, LLM[str, None])
    assert_type(
        provider.llm("anthropic.claude-sonnet-5", max_completion_tokens=128, tools=[]),
        LLM[str, ToolManager],
    )
    assert_type(
        provider.llm(
            "anthropic.claude-sonnet-5", max_completion_tokens=128, response_format=Answer
        ),
        LLM[Answer, None],
    )
    assert_type(
        provider.llm(
            "anthropic.claude-sonnet-5",
            max_completion_tokens=128,
            response_format=Answer,
            tools=tools,
        ),
        LLM[Answer, ToolManager],
    )
    generic = provider.llm(
        "anthropic.claude-sonnet-5", max_completion_tokens=128, response_format=output, tools=tools
    )
    assert_type(generic, LLM[T, ToolManager])
    assert_type(
        provider.llm(
            "anthropic.claude-sonnet-5", max_completion_tokens=128, response_format=output
        ),
        LLM[T, None],
    )
    return generic


def gemini_inference[T: str | BaseModel](
    provider: Gemini, output: type[T], tools: ToolManager
) -> LLM[T, ToolManager]:
    text = provider.llm("gemini-3.6-flash")
    assert_type(text, LLM[str, None])
    assert_type(provider.llm("gemini-3.6-flash", tools=[]), LLM[str, ToolManager])
    assert_type(provider.llm("gemini-3.6-flash", response_format=Answer), LLM[Answer, None])
    assert_type(
        provider.llm("gemini-3.6-flash", response_format=Answer, tools=tools),
        LLM[Answer, ToolManager],
    )
    generic = provider.llm("gemini-3.6-flash", response_format=output, tools=tools)
    assert_type(generic, LLM[T, ToolManager])
    assert_type(provider.llm("gemini-3.6-flash", response_format=output), LLM[T, None])
    return generic


def deepseek_inference[T: str | BaseModel](
    provider: DeepSeek, output: type[T], tools: ToolManager
) -> LLM[T, ToolManager]:
    text = provider.llm("deepseek-v4-flash")
    assert_type(text, LLM[str, None])
    assert_type(provider.llm("deepseek-v4-flash", tools=[]), LLM[str, ToolManager])
    assert_type(provider.llm("deepseek-v4-flash", response_format=Answer), LLM[Answer, None])
    assert_type(
        provider.llm("deepseek-v4-flash", response_format=Answer, tools=tools),
        LLM[Answer, ToolManager],
    )
    generic = provider.llm("deepseek-v4-flash", response_format=output, tools=tools)
    assert_type(generic, LLM[T, ToolManager])
    assert_type(provider.llm("deepseek-v4-flash", response_format=output), LLM[T, None])
    return generic


def adapter_inference[T: str | BaseModel](
    provider: Adapter, output: type[T], tools: ToolManager, span: OtelChatSpan
) -> LLM[T, ToolManager]:
    text = llm_from_adapter(provider)
    assert_type(text, LLM[str, None])
    assert_type(text, LLM)
    assert_type(text.tool_manager, None)
    assert_type(llm_from_adapter(provider, tools=[]), LLM[str, ToolManager])
    assert_type(llm_from_adapter(provider, response_format=Answer), LLM[Answer, None])
    assert_type(
        llm_from_adapter(provider, response_format=Answer, tools=tools),
        LLM[Answer, ToolManager],
    )
    generic = llm_from_adapter(provider, response_format=output, tools=tools)
    assert_type(generic, LLM[T, ToolManager])
    assert_type(generic.tool_manager, ToolManager)
    assert_type(llm_from_adapter(provider, response_format=output), LLM[T, None])
    assert_type(llm_from_adapter(provider, response_format=Answer), LLM[Answer])
    assert_type(generic.bind(), LLM[T, ToolManager])
    assert_type(generic.bind(system_prompt="s"), LLM[T, ToolManager])
    assert_type(generic.bind(tools=None), LLM[T, None])
    assert_type(generic.bind(response_format=str), LLM[str, ToolManager])
    assert_type(generic.bind(response_format=str, tools=None), LLM[str, None])
    assert_type(text.bind(response_format=output, tools=tools), LLM[T, ToolManager])
    assert_type(text.bind(response_format=output), LLM[T, None])
    assert_type(text.bind(tools=tools), LLM[str, ToolManager])
    assert_type(text.bind(tools=[]), LLM[str, ToolManager])
    assert_type(text.bind(response_format=output, tools=[]), LLM[T, ToolManager])
    assert_type(text.bind(response_format=output).bind(tools=tools), LLM[T, ToolManager])
    assert_type(text.bind(response_format=output).bind(tools=[]), LLM[T, ToolManager])
    provider_executed_tools = ({"type": "web_search"},)
    assert_type(text.bind(provider_executed_tools=provider_executed_tools), LLM[str, None])
    assert_type(
        generic.bind(tools=None).bind(provider_executed_tools=provider_executed_tools),
        LLM[T, None],
    )
    restored = llm_from_otel(span, llm=generic)
    assert_type(restored, LLM[T, ToolManager])
    assert_type(restored.bind(tools=None), LLM[T, None])
    return generic
