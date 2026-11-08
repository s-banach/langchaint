"""Construct DeepSeek `LLM` values with cataloged pricing.

Importing this subpackage requires `openai`.
DeepSeek serves Chat Completions at https://api.deepseek.com.
Thinking mode `thinking` values for `extra_body`: https://api-docs.deepseek.com/guides/thinking_mode.

Prices use USD per one million tokens.
Source: https://api-docs.deepseek.com/quick_start/pricing, read 2026-08-03.
Recheck that page before relying on a table.
`DEEPSEEK_PRICING` contains off-peak list prices.
DeepSeek charges twice those prices during its documented peak windows.
Cache hits use the `input_tokens_cache_read` rate.
Cache misses use the `input_tokens_cache_none` rate.
Cache writes cost zero.
"""

import os
from typing import Literal, overload

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's deepseek backend requires the openai package; install langchaint[deepseek]."
    ) from exc

from collections.abc import Mapping, Sequence

from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from langchaint.adapter import ToolChoice
from langchaint.billing.pricing import TokenRates
from langchaint.common.messages import TextPart
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.llm import UNCHANGED, Unchanged, _configure, _ConfiguredLLM
from langchaint.generation.observer import Observer
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.shared import OpenAIPricingTable
from langchaint.tools import ToolManager, ToolSequence

type DeepSeekModelId = Literal["deepseek-v4-flash", "deepseek-v4-pro"]
"""Model identifiers with public prices in DEEPSEEK_PRICING."""

DEEPSEEK_PRICING: dict[DeepSeekModelId, TokenRates] = {
    "deepseek-v4-flash": TokenRates(
        input_tokens_cache_none=0.14,
        output_tokens=0.28,
        input_tokens_cache_read=0.0028,
        input_tokens_cache_write=0.0,
    ),
    "deepseek-v4-pro": TokenRates(
        input_tokens_cache_none=0.435,
        output_tokens=0.87,
        input_tokens_cache_read=0.003625,
        input_tokens_cache_write=0.0,
    ),
}
"""Public off-peak prices that `DeepSeek.llm` uses by default."""

_PRICING_BY_MODEL_ID = dict[str, TokenRates](DEEPSEEK_PRICING.items())
"""`DEEPSEEK_PRICING` with `str` keys for runtime model lookup."""

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_API_KEY_ENVIRONMENT_VARIABLE = "DEEPSEEK_API_KEY"


def input_tokens_cache_read_from_usage_deepseek(usage: CompletionUsage) -> int:
    """Return `prompt_cache_hit_tokens` from `CompletionUsage.model_extra`, or zero when absent.

    DeepSeek documents that cache-hit and cache-miss counters sum to `prompt_tokens`.
    Source: https://api-docs.deepseek.com/guides/kv_cache, read 2026-08-03.
    `CompletionUsage` omits both counters in openai 3.44.0.
    """
    hit_tokens = (usage.model_extra or {}).get("prompt_cache_hit_tokens")
    return hit_tokens if isinstance(hit_tokens, int) else 0


class DeepSeekLLM[OutputT: str | BaseModel = str, ToolManagerT: ToolManager | None = None](
    _ConfiguredLLM[OutputT, ToolManagerT]
):
    """Create instances through `DeepSeek.llm` to validate request settings before generation."""

    @overload
    def bind[NewOutputT: str | BaseModel](
        self,
        *,
        response_format: type[NewOutputT],
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "DeepSeekLLM[NewOutputT, ToolManager]": ...
    @overload
    def bind[NewOutputT: str | BaseModel](
        self,
        *,
        response_format: type[NewOutputT],
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "DeepSeekLLM[NewOutputT, None]": ...
    @overload
    def bind[NewOutputT: str | BaseModel](
        self: "DeepSeekLLM[OutputT, ToolManagerT]",
        *,
        response_format: type[NewOutputT],
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "DeepSeekLLM[NewOutputT, ToolManagerT]": ...
    @overload
    def bind(
        self: "DeepSeekLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "DeepSeekLLM[OutputT, ToolManager]": ...
    @overload
    def bind(
        self: "DeepSeekLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: None,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "DeepSeekLLM[OutputT, None]": ...
    @overload
    def bind(
        self: "DeepSeekLLM[OutputT, ToolManagerT]",
        *,
        response_format: Unchanged = ...,
        tools: Unchanged = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = ...,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = ...,
        tool_choice: ToolChoice | Unchanged = ...,
        parallel_tool_calls: bool | Unchanged = ...,
        max_completion_tokens: int | None | Unchanged = ...,
        reasoning_level: str | None | Unchanged = ...,
        temperature: float | None | Unchanged = ...,
        extra_body: Mapping[str, object] | None | Unchanged = ...,
        max_requests: int | Unchanged = ...,
        automatic_cache_breakpoints: bool | None | Unchanged = ...,
    ) -> "DeepSeekLLM[OutputT, ToolManagerT]": ...
    def bind(
        self,
        *,
        response_format: type[str] | type[BaseModel] | Unchanged = UNCHANGED,
        system_prompt: str | Sequence[TextPart] | None | Unchanged = UNCHANGED,
        tools: ToolManager | ToolSequence | None | Unchanged = UNCHANGED,
        provider_executed_tools: Sequence[Mapping[str, object]] | Unchanged = UNCHANGED,
        tool_choice: ToolChoice | Unchanged = UNCHANGED,
        parallel_tool_calls: bool | Unchanged = UNCHANGED,
        max_completion_tokens: int | None | Unchanged = UNCHANGED,
        reasoning_level: str | None | Unchanged = UNCHANGED,
        temperature: float | None | Unchanged = UNCHANGED,
        extra_body: Mapping[str, object] | None | Unchanged = UNCHANGED,
        max_requests: int | Unchanged = UNCHANGED,
        automatic_cache_breakpoints: bool | None | Unchanged = UNCHANGED,
    ) -> "DeepSeekLLM[BaseModel | str, ToolManager | None]":
        """Return a new LLM, preserving every omitted setting.

        `tools=None` removes `ToolManager`.
        A `tools` sequence constructs `ToolManager` with this binding's observer.
        An existing `ToolManager` retains its identity and its own observer.
        `automatic_cache_breakpoints=None` reads `Adapter.automatic_cache_breakpoints_default`.

        `response_format=str` returns text; a pydantic model validates structured output.
        `max_completion_tokens=None` omits the limit.
        `reasoning_level` accepts exact provider strings; `None` omits the setting.
        `extra_body` supplies provider wire-body fields; `None` clears them.
        Declare tools only through `tools`, because result types follow the configured tools.
        `max_requests` counts all requests per input, including retries.

        Raises:
            ValueError: `tools` contains duplicate names.
            ValueError: `AllowedToolsChoice.tool_names` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence.
            ValueError: `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one.
            ValueError: The binding breaks a rule of the adapter.
                The docstring of the backend's `llm()` lists the rules.
            TypeError: `response_format` is a proper subclass of `str`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        return DeepSeekLLM(
            self._reconfigure(
                system_prompt=system_prompt,
                tools=tools,
                provider_executed_tools=provider_executed_tools,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                reasoning_level=reasoning_level,
                temperature=temperature,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                extra_body=extra_body,
                max_requests=max_requests,
                automatic_cache_breakpoints=automatic_cache_breakpoints,
            )
        )


class DeepSeek:
    """Create `LLM` values for DeepSeek."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `DeepSeek` without sending a request.

        A passed `client` must reach DeepSeek.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` is absent and `DEEPSEEK_API_KEY` is unset.
        """
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        if client is None:
            api_key = os.environ.get(_API_KEY_ENVIRONMENT_VARIABLE)
            if api_key is None:
                raise ValueError(
                    f"client is None and {_API_KEY_ENVIRONMENT_VARIABLE} is unset; set it, or pass client=AsyncOpenAI(base_url={_DEEPSEEK_BASE_URL!r}, api_key=...)."
                )
            client = AsyncOpenAI(base_url=_DEEPSEEK_BASE_URL, api_key=api_key)
        self.client: AsyncOpenAI = client

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: DeepSeekModelId,
        *,
        pricing: TokenRates | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: DeepSeekModelId,
        *,
        pricing: TokenRates | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: DeepSeekModelId,
        *,
        pricing: TokenRates | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: DeepSeekModelId,
        *,
        pricing: TokenRates | None = ...,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[str, None]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: TokenRates,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: TokenRates,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[OutputT],
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: TokenRates,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: ToolManager | ToolSequence,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: TokenRates,
        system_prompt: str | Sequence[TextPart] | None = ...,
        tools: None = ...,
        provider_executed_tools: Sequence[Mapping[str, object]] = ...,
        response_format: type[str] = ...,
        max_completion_tokens: int | None = ...,
        reasoning_level: str | None = ...,
        temperature: float | None = ...,
        tool_choice: ToolChoice = ...,
        parallel_tool_calls: bool = ...,
        extra_body: Mapping[str, object] | None = ...,
        max_requests: int = ...,
        automatic_cache_breakpoints: bool | None = ...,
    ) -> DeepSeekLLM[str, None]: ...

    def llm(
        self,
        model: str,
        *,
        pricing: TokenRates | None = None,
        system_prompt: str | Sequence[TextPart] | None = None,
        tools: ToolManager | ToolSequence | None = None,
        provider_executed_tools: Sequence[Mapping[str, object]] = (),
        response_format: type[str] | type[BaseModel] = str,
        max_completion_tokens: int | None = None,
        reasoning_level: str | None = None,
        temperature: float | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = True,
        extra_body: Mapping[str, object] | None = None,
        max_requests: int = 3,
        automatic_cache_breakpoints: bool | None = None,
    ) -> DeepSeekLLM[str | BaseModel, ToolManager | None]:
        """Build an `LLM` for one DeepSeek Chat Completions model.

        `model` is sent verbatim.
        Cataloged models receive `DEEPSEEK_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.

        `llm` and `bind` raise `ValueError` when:
        - `automatic_cache_breakpoints=False`, because the adapter sends no `prompt_cache_options`.
        - `provider_executed_tools` is nonempty.
        - `extra_body` contains `web_search_options`, which langchaint cannot price.
        - `extra_body` contains `functions`, which declares tools outside `tools`.
        - `extra_body` sets `n` to a value other than 1 or `None`, because the adapter reads one choice per response.

        `reasoning_level` sets `reasoning_effort`.
        `max_completion_tokens=None` omits the token limit.

        `tools` accepts application tools or an existing `ToolManager`.
        A sequence constructs a `ToolManager` with the backend's observer.
        An existing manager retains its identity and observer.
        `provider_executed_tools` contains provider-shaped tool definitions executed by the provider.

        `response_format` selects a pydantic model for structured output, or `str` for final text.
        A response without a text part or tool call gives `GenerationError`.
        For text output, an empty final text part gives `""`.
        `reasoning_level=None` uses the provider's reasoning default.

        `extra_body` adds provider wire-body fields.
        Declare application tools through `tools` so result types match executable tools.
        `max_requests` limits the requests for one input, including retries.
        `automatic_cache_breakpoints=None` uses the adapter default.

        Raises:
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when the SDK client contradicts the DeepSeek provider.
            ValueError: `tools` contains duplicate names.
            ValueError: `AllowedToolsChoice.tool_names` contains a name absent from the bound tool schemas.
            ValueError: `system_prompt` is an empty sequence or `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one, or a provider constraint listed above is violated.
            TypeError: `response_format` is a proper subclass of `str`.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's argument model has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's argument model is not fully defined.
        """
        rates = pricing if pricing is not None else _PRICING_BY_MODEL_ID.get(model)
        if rates is None:
            raise ValueError(
                f"model {model!r} is not in DEEPSEEK_PRICING; pass pricing= stating its rates"
            )
        adapter = OpenAIChatCompletionsAdapter(
            client=self.client,
            model=model,
            pricing=OpenAIPricingTable(default=rates),
            provider_name="deepseek",
            supports_prompt_cache_options=False,
            input_tokens_cache_read_from_usage=input_tokens_cache_read_from_usage_deepseek,
        )
        return DeepSeekLLM(
            _configure(
                adapter,
                shared_backoff=self._shared_backoff,
                observer=self._observer,
                system_prompt=system_prompt,
                tools=tools,
                provider_executed_tools=provider_executed_tools,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                reasoning_level=reasoning_level,
                temperature=temperature,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                extra_body=extra_body,
                max_requests=max_requests,
                automatic_cache_breakpoints=automatic_cache_breakpoints,
            )
        )


__all__ = [
    "DEEPSEEK_PRICING",
    "DeepSeek",
    "DeepSeekLLM",
    "DeepSeekModelId",
    "input_tokens_cache_read_from_usage_deepseek",
]
