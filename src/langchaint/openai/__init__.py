"""Construct OpenAI and Bedrock `LLM` values and OpenAI `EmbeddingModel` values.

Importing this subpackage requires `openai`.
`OpenAI.llm` uses the Responses API and reports `provider_name="openai"`.
`OpenAIBedrock.llm` uses Responses and reports `provider_name="aws.bedrock"`.
Use `OpenAIChatCompletionsAdapter` directly for compatible endpoints.
Use `OpenAIResponsesAdapter` directly for Azure.
Per-model reasoning effort values: https://developers.openai.com/api/docs/guides/reasoning.

Cataloged models receive `OPENAI_PRICING`.
Cataloged embedding models receive `OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS`.
Uncataloged OpenAI models require `pricing` and `supports_prompt_cache_options`.
`OpenAIBedrock.llm` always requires both parameters.
Missing optional rates produce NaN token costs.

Token prices use USD per one million tokens.
Web-search prices use USD per invocation.
File-search prices use USD per invocation.
Token, web-search, and regional-processing price source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json.
File-search price source: https://developers.openai.com/api/docs/pricing.
Embedding batching parameters come from the OpenAI embeddings guide.
Source: https://developers.openai.com/api/docs/guides/embeddings.
Cataloged embedding model dimensions come from the OpenAI model catalog.
Source: https://developers.openai.com/api/docs/models/all.
`OpenAI.llm(pricing=...)` replaces cataloged estimates.
`OpenAIBedrock.llm(pricing=...)` uses caller rates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "langchaint's openai backend requires its dependencies; install "
        "langchaint[openai], langchaint[openai-embedding], or langchaint[openai-bedrock]."
    ) from exc


from pydantic import BaseModel

from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.llm import UNCHANGED, Unchanged, _configure, _ConfiguredLLM
from langchaint.openai._generated_pricing import (
    OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS,
    OPENAI_PRICING,
    OPENAI_REGIONAL_PROCESSING_MULTIPLIER,
    OpenAIEmbeddingModelId,
    OpenAIModelId,
)
from langchaint.openai.chat_completions_adapter import OpenAIChatCompletionsAdapter
from langchaint.openai.responses_adapter import (
    OpenAIResponsesAdapter,
    ReasoningSummary,
)
from langchaint.openai.shared import (
    OpenAIPricingTable,
    OpenAIResponsesServiceTier,
    OpenAIServiceTier,
)
from langchaint.tools import ToolManager, ToolSequence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchaint.adapter import ToolChoice
    from langchaint.common.messages import TextPart
    from langchaint.embedding import EmbeddingModel
    from langchaint.generation.observer import Observer

OPENAI_EMBEDDING_MODELS: frozenset[OpenAIEmbeddingModelId] = frozenset(
    OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS
)
"""Cataloged OpenAI embedding model identifiers."""

_PRICING_BY_MODEL_ID = dict[str, OpenAIPricingTable](OPENAI_PRICING.items())
"""`OPENAI_PRICING` with `str` keys for runtime model lookup."""

_REGIONAL_PROCESSING_MULTIPLIER_BY_MODEL_ID = dict[str, float](
    OPENAI_REGIONAL_PROCESSING_MULTIPLIER.items()
)
"""`OPENAI_REGIONAL_PROCESSING_MULTIPLIER` with `str` keys for runtime model lookup."""


class OpenAILLM[OutputT: str | BaseModel = str, ToolManagerT: ToolManager | None = None](
    _ConfiguredLLM[OutputT, ToolManagerT]
):
    """Create instances through `OpenAI.llm` to validate request settings before generation."""

    @overload
    def rebind[NewOutputT: str | BaseModel](
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
    ) -> OpenAILLM[NewOutputT, ToolManager]: ...
    @overload
    def rebind[NewOutputT: str | BaseModel](
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
    ) -> OpenAILLM[NewOutputT, None]: ...
    @overload
    def rebind[NewOutputT: str | BaseModel](
        self: OpenAILLM[OutputT, ToolManagerT],
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
    ) -> OpenAILLM[NewOutputT, ToolManagerT]: ...
    @overload
    def rebind(
        self: OpenAILLM[OutputT, ToolManagerT],
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
    ) -> OpenAILLM[OutputT, ToolManager]: ...
    @overload
    def rebind(
        self: OpenAILLM[OutputT, ToolManagerT],
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
    ) -> OpenAILLM[OutputT, None]: ...
    @overload
    def rebind(
        self: OpenAILLM[OutputT, ToolManagerT],
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
    ) -> OpenAILLM[OutputT, ToolManagerT]: ...
    def rebind(
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
    ) -> OpenAILLM[BaseModel | str, ToolManager | None]:
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
            TypeError: The adapter does not support `tool_choice`.
            TypeError: `response_format` is a proper subclass of `str`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        return OpenAILLM(
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


class OpenAI:
    """Create `LLM` and `EmbeddingModel` values for OpenAI."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `OpenAI` without sending a request.

        `client=None` constructs `AsyncOpenAI()`.
        A passed `client` must reach OpenAI.
        `shared_backoff` admits every request of the created `LLM` and `EmbeddingModel` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer` follows every request of the created `EmbeddingModel` values.
        `observer=None` follows none.

        Raises:
            openai.OpenAIError: `client` is absent and OpenAI credentials are unavailable.
        """
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: AsyncOpenAI = client if client is not None else AsyncOpenAI()

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: OpenAIModelId,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable | None = ...,
        supports_prompt_cache_options: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: OpenAIModelId,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable | None = ...,
        supports_prompt_cache_options: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: OpenAIModelId,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable | None = ...,
        supports_prompt_cache_options: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: OpenAIModelId,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable | None = ...,
        supports_prompt_cache_options: bool | None = ...,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[str, None]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        regional_processing: bool = ...,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = ...,
        service_tier: OpenAIResponsesServiceTier | None = ...,
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
    ) -> OpenAILLM[str, None]: ...

    def llm(
        self,
        model: str,
        *,
        regional_processing: bool = False,
        pricing: OpenAIPricingTable | None = None,
        supports_prompt_cache_options: bool | None = None,
        reasoning_summary: ReasoningSummary | None = None,
        service_tier: OpenAIResponsesServiceTier | None = None,
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
    ) -> OpenAILLM[str | BaseModel, ToolManager | None]:
        """Build an `LLM` for one Responses API model.

        `model` is sent verbatim.
        Cataloged models receive `OPENAI_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        Set `regional_processing=True` when the configured endpoint uses regional processing.
        It multiplies every token rate by the model's `OPENAI_REGIONAL_PROCESSING_MULTIPLIER`.
        An uncataloged model has no multiplier, so every nonzero token category then costs NaN.
        State its regional rates in `pricing` and leave `regional_processing=False`.
        `supports_prompt_cache_options` states whether the provider API accepts `prompt_cache_options`.
        Cataloged models default to True because every cataloged model is gpt-5.6 or later.
        OpenAI 3.42.1 documents `prompt_cache_options` for gpt-5.6 and later models.
        `reasoning_summary` requests readable reasoning summary text.
        `service_tier` sets the requested OpenAI service tier.
        The reported service tier selects pricing.

        `llm` and `rebind` raise `ValueError` when:
        - `automatic_cache_breakpoints=False` and `supports_prompt_cache_options` is false.
        - A provider-executed tool's `type` is missing or unsupported.
          The adapter supports specific versions of web search and file search.
        - Web search is bound and `pricing.web_search_usd_per_invocation` is `None`.
        - File search is bound and `pricing.file_search_usd_per_invocation` is `None`.

        `reasoning_level` sets `reasoning.effort`.
        `max_completion_tokens=None` omits the token limit.

        `tools` accepts application tools or an existing `ToolManager`.
        A sequence constructs a `ToolManager` with the backend's observer.
        An existing manager retains its identity and observer.
        `provider_executed_tools` contains provider-shaped tool definitions executed by the provider.

        `response_format` selects a pydantic model for structured output, or `str` for final text.
        A response without a text part or tool call gives `GenerationError`.
        An empty final text part gives `""`.
        `reasoning_level=None` uses the provider's reasoning default.

        `extra_body` adds provider wire-body fields.
        Declare application tools through `tools` so result types match executable tools.
        `max_requests` limits the requests for one input, including retries.
        `automatic_cache_breakpoints=None` uses the adapter default.

        Raises:
            ValueError: An uncataloged model lacks required pricing or caching data.
                Also raised when the SDK client contradicts the OpenAI provider.
            ValueError: `tools` contains duplicate names or `tool_choice` names an absent tool.
            ValueError: `system_prompt` is an empty sequence or `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one, or a provider constraint listed above is violated.
            TypeError: `response_format` is a proper subclass of `str`, or the adapter does not support `tool_choice`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's argument model has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's argument model is not fully defined.
        """
        catalog_table = _PRICING_BY_MODEL_ID.get(model)
        if catalog_table is None:
            if pricing is None or supports_prompt_cache_options is None:
                missing_arguments: list[str] = []
                if pricing is None:
                    missing_arguments.append("pricing= stating its rates")
                if supports_prompt_cache_options is None:
                    missing_arguments.append(
                        "supports_prompt_cache_options= stating whether it accepts prompt_cache_options"
                    )
                raise ValueError(
                    f"model {model!r} is not in OPENAI_PRICING; pass "
                    + " and ".join(missing_arguments)
                )
        else:
            pricing = pricing or catalog_table
            if supports_prompt_cache_options is None:
                supports_prompt_cache_options = True
        if regional_processing:
            pricing = pricing.scaled(
                _REGIONAL_PROCESSING_MULTIPLIER_BY_MODEL_ID.get(model, float("nan"))
            )
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="openai",
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
            service_tier=service_tier,
        )
        return OpenAILLM(
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

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-3-small"],
        *,
        dimension: int = 1536,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-3-large"],
        *,
        dimension: int = 3072,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    @overload
    def embedding_model(
        self,
        model: Literal["text-embedding-ada-002"],
        *,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel: ...

    def embedding_model(
        self,
        model: OpenAIEmbeddingModelId,
        *,
        dimension: int | None = None,
        max_requests: int = 3,
        usd_per_million_tokens: float | None = None,
    ) -> EmbeddingModel:
        """Build an `EmbeddingModel` for one cataloged OpenAI model.

        Third-generation models accept their documented dimension range.
        `text-embedding-ada-002` has dimension 1536 and accepts no dimension argument.
        Every model counts batching tokens with tiktoken's `cl100k_base` encoding.
        `usd_per_million_tokens` prices input tokens.
        It replaces the model's rate in `OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS`.

        Raises:
            ValueError: `model`, `dimension`, or `max_requests` is invalid.
            ModuleNotFoundError: Either `numpy` or `tiktoken` is unavailable.
        """
        if model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(f"model {model!r} is not in OPENAI_EMBEDDING_MODELS")
        if model == "text-embedding-ada-002":
            if dimension is not None:
                raise ValueError("text-embedding-ada-002 accepts no dimension argument")
            dimension = 1536
        else:
            max_dimension = 1536 if model == "text-embedding-3-small" else 3072
            dimension = max_dimension if dimension is None else dimension
            if isinstance(dimension, bool) or not (1 <= dimension <= max_dimension):
                raise ValueError(
                    f"dimension for {model!r} must be an int from 1 through "
                    f"{max_dimension}, got {dimension!r}"
                )
        try:
            from langchaint.embedding import EmbeddingModel  # noqa: PLC0415 (defer numpy)
            from langchaint.openai.embedding_adapter import (  # noqa: PLC0415 (keep tiktoken outside ordinary imports)
                _OpenAIEmbeddingAdapter,
            )
        except ModuleNotFoundError as exc:
            cause = exc.__cause__
            if exc.name not in ("numpy", "tiktoken") and (
                not isinstance(cause, ModuleNotFoundError) or cause.name != "numpy"
            ):
                raise
            raise ModuleNotFoundError(
                "OpenAI embeddings require numpy and tiktoken; install "
                "langchaint[openai-embedding]."
            ) from exc
        adapter = _OpenAIEmbeddingAdapter(
            client=self.client,
            model=model,
            dimension=dimension,
        )
        return EmbeddingModel(
            adapter=adapter,
            shared_backoff=self._shared_backoff,
            max_requests=max_requests,
            usd_per_million_tokens=(
                OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS[model]
                if usd_per_million_tokens is None
                else usd_per_million_tokens
            ),
            observer=self._observer,
        )


class OpenAIBedrockLLM[OutputT: str | BaseModel = str, ToolManagerT: ToolManager | None = None](
    _ConfiguredLLM[OutputT, ToolManagerT]
):
    """Create instances through `OpenAIBedrock.llm` to validate request settings before generation."""

    @overload
    def rebind[NewOutputT: str | BaseModel](
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
    ) -> OpenAIBedrockLLM[NewOutputT, ToolManager]: ...
    @overload
    def rebind[NewOutputT: str | BaseModel](
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
    ) -> OpenAIBedrockLLM[NewOutputT, None]: ...
    @overload
    def rebind[NewOutputT: str | BaseModel](
        self: OpenAIBedrockLLM[OutputT, ToolManagerT],
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
    ) -> OpenAIBedrockLLM[NewOutputT, ToolManagerT]: ...
    @overload
    def rebind(
        self: OpenAIBedrockLLM[OutputT, ToolManagerT],
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
    ) -> OpenAIBedrockLLM[OutputT, ToolManager]: ...
    @overload
    def rebind(
        self: OpenAIBedrockLLM[OutputT, ToolManagerT],
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
    ) -> OpenAIBedrockLLM[OutputT, None]: ...
    @overload
    def rebind(
        self: OpenAIBedrockLLM[OutputT, ToolManagerT],
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
    ) -> OpenAIBedrockLLM[OutputT, ToolManagerT]: ...
    def rebind(
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
    ) -> OpenAIBedrockLLM[BaseModel | str, ToolManager | None]:
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
            TypeError: The adapter does not support `tool_choice`.
            TypeError: `response_format` is a proper subclass of `str`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's `args_model` has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's `args_model` is not fully defined.
        """
        return OpenAIBedrockLLM(
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


class OpenAIBedrock:
    """Create `LLM` values for OpenAI models on Bedrock."""

    def __init__(
        self,
        *,
        aws_region: str | None = None,
        client: AsyncBedrockOpenAI | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `OpenAIBedrock` without sending a request.

        `aws_region` selects the region for an SDK client created by `OpenAIBedrock`.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` and `aws_region` are both provided.
            openai.OpenAIError: No Bedrock region is available.
        """
        if client is not None and aws_region is not None:
            raise ValueError("Pass at most one of client= or aws_region=")
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: AsyncBedrockOpenAI = (
            client if client is not None else AsyncBedrockOpenAI(aws_region=aws_region)
        )

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
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
    ) -> OpenAIBedrockLLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
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
    ) -> OpenAIBedrockLLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
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
    ) -> OpenAIBedrockLLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
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
    ) -> OpenAIBedrockLLM[str, None]: ...

    def llm(
        self,
        model: str,
        *,
        pricing: OpenAIPricingTable,
        supports_prompt_cache_options: bool,
        reasoning_summary: ReasoningSummary | None = None,
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
    ) -> OpenAIBedrockLLM[str | BaseModel, ToolManager | None]:
        """Build an `LLM` for one OpenAI model served by Bedrock.

        `model` is sent verbatim.
        Bedrock model identifiers have no carried pricing catalog.
        `pricing` states this Bedrock model's rates.
        `supports_prompt_cache_options` states whether the provider API accepts `prompt_cache_options`.
        `reasoning_summary` requests readable reasoning summary text.
        Bedrock models accept no OpenAI `service_tier` parameter here.

        `llm` and `rebind` raise `ValueError` when:
        - `automatic_cache_breakpoints=False` and `supports_prompt_cache_options=False`.
        - `provider_executed_tools` is nonempty.

        `reasoning_level` sets `reasoning.effort`.
        `max_completion_tokens=None` omits the token limit.

        `tools` accepts application tools or an existing `ToolManager`.
        A sequence constructs a `ToolManager` with the backend's observer.
        An existing manager retains its identity and observer.
        `provider_executed_tools` contains provider-shaped tool definitions executed by the provider.

        `response_format` selects a pydantic model for structured output, or `str` for final text.
        A response without a text part or tool call gives `GenerationError`.
        An empty final text part gives `""`.
        `reasoning_level=None` uses the provider's reasoning default.

        `extra_body` adds provider wire-body fields.
        Declare application tools through `tools` so result types match executable tools.
        `max_requests` limits the requests for one input, including retries.
        `automatic_cache_breakpoints=None` uses the adapter default.

        Raises:
            ValueError: `tools` contains duplicate names or `tool_choice` names an absent tool.
            ValueError: `system_prompt` is an empty sequence or `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one, or a provider constraint listed above is violated.
            TypeError: `response_format` is a proper subclass of `str`, or the adapter does not support `tool_choice`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's argument model has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's argument model is not fully defined.
        """
        adapter = OpenAIResponsesAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="aws.bedrock",
            supports_prompt_cache_options=supports_prompt_cache_options,
            reasoning_summary=reasoning_summary,
        )
        return OpenAIBedrockLLM(
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
    "OPENAI_EMBEDDING_MODELS",
    "OPENAI_EMBEDDING_USD_PER_MILLION_TOKENS",
    "OPENAI_PRICING",
    "OPENAI_REGIONAL_PROCESSING_MULTIPLIER",
    "OpenAI",
    "OpenAIBedrock",
    "OpenAIBedrockLLM",
    "OpenAIChatCompletionsAdapter",
    "OpenAIEmbeddingModelId",
    "OpenAILLM",
    "OpenAIModelId",
    "OpenAIPricingTable",
    "OpenAIResponsesAdapter",
    "OpenAIResponsesServiceTier",
    "OpenAIServiceTier",
    "ReasoningSummary",
]
