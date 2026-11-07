"""Construct Gemini `LLM` values with cataloged pricing.

Importing this subpackage requires `google-genai`.
`Gemini.llm` sends the stated model identifier verbatim.
`Gemini.llm` reaches the Gemini Developer API and reports `provider_name="gcp.gemini"`.
Vertex AI callers construct `GeminiGenerateContentAdapter` directly.
Use `provider_name="gcp.vertex_ai"` and Vertex pricing there.
Per-model thinking defaults and levels: https://aistudio.google.com/docs/thinking.
`generationConfig.thinkingConfig` fields: https://aistudio.google.com/api-reference/generate-content.

Cataloged models receive `GEMINI_PRICING`, which states `ON_DEMAND` rates.
Uncataloged Gemini models require `pricing`.
Responses from traffic types without rates cost NaN.

Token prices use USD per one million tokens.
Google Search prices use USD per query.
Google Maps prices use USD per query.
Source: https://ai.google.dev/gemini-api/docs/pricing, read 2026-08-03.
Maps source: https://ai.google.dev/gemini-api/docs/maps-grounding.
Recheck the sources before relying on a table.
`GEMINI_PRICING` carries text, image, and video rates.
Catalog tool rates estimate post-quota list prices.
`Gemini.llm(pricing=...)` replaces cataloged estimates.
Explicit cache-resource storage charges have no request `Usage` field.
"""

from typing import Literal, overload

try:
    from google import genai
except ModuleNotFoundError as exc:
    if exc.name not in ("google", "google.genai"):
        raise
    raise ModuleNotFoundError(
        "langchaint's gemini backend requires google-genai; install langchaint[gemini]."
    ) from exc

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from langchaint.adapter import ToolChoice
from langchaint.billing.pricing import LongContextPricing
from langchaint.common.messages import TextPart
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.gemini.generate_content_adapter import (
    GeminiGenerateContentAdapter,
    GeminiPricingTable,
    GeminiRates,
    GeminiReportedServiceTier,
    GeminiServiceTier,
    assembled_response,
)
from langchaint.generation.llm import UNCHANGED, Unchanged, _configure, _ConfiguredLLM
from langchaint.generation.observer import Observer
from langchaint.tools import ToolManager, ToolSequence

type GeminiModelId = Literal[
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]
"""Model identifiers with public prices in GEMINI_PRICING."""

GEMINI_PRICING: dict[GeminiModelId, GeminiPricingTable] = {
    "gemini-3.6-flash": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=1.50,
            input_tokens_cache_read=0.15,
            output_tokens=7.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.5-flash": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=1.50,
            input_tokens_cache_read=0.15,
            output_tokens=9.00,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    # The pricing page lists no cache-read price for gemini-3.5-flash-lite, so the rate is NaN.
    "gemini-3.5-flash-lite": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=0.30,
            input_tokens_cache_read=float("nan"),
            output_tokens=2.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-flash-lite": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=0.25,
            input_tokens_cache_read=0.025,
            output_tokens=1.50,
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
    "gemini-3.1-pro-preview": GeminiPricingTable(
        on_demand=GeminiRates(
            input_tokens_cache_none=2.00,
            input_tokens_cache_read=0.20,
            output_tokens=12.00,
        ),
        long_context=LongContextPricing(
            prompt_tokens_above=200_000, input_multiplier=2.0, output_multiplier=1.5
        ),
        google_search_usd_per_query=0.014,
        google_maps_usd_per_query=0.014,
    ),
}
"""Public on-demand prices that `Gemini.llm` uses by default."""

_PRICING_BY_MODEL_ID = dict[str, GeminiPricingTable](GEMINI_PRICING.items())
"""`GEMINI_PRICING` with `str` keys for runtime model lookup."""


class GeminiLLM[OutputT: str | BaseModel = str, ToolManagerT: ToolManager | None = None](
    _ConfiguredLLM[OutputT, ToolManagerT]
):
    """Create instances through `Gemini.llm` to validate request settings before generation."""

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
    ) -> "GeminiLLM[NewOutputT, ToolManager]": ...
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
    ) -> "GeminiLLM[NewOutputT, None]": ...
    @overload
    def rebind[NewOutputT: str | BaseModel](
        self: "GeminiLLM[OutputT, ToolManagerT]",
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
    ) -> "GeminiLLM[NewOutputT, ToolManagerT]": ...
    @overload
    def rebind(
        self: "GeminiLLM[OutputT, ToolManagerT]",
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
    ) -> "GeminiLLM[OutputT, ToolManager]": ...
    @overload
    def rebind(
        self: "GeminiLLM[OutputT, ToolManagerT]",
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
    ) -> "GeminiLLM[OutputT, None]": ...
    @overload
    def rebind(
        self: "GeminiLLM[OutputT, ToolManagerT]",
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
    ) -> "GeminiLLM[OutputT, ToolManagerT]": ...
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
    ) -> "GeminiLLM[BaseModel | str, ToolManager | None]":
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
        return GeminiLLM(
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


class Gemini:
    """Create `LLM` values for Gemini."""

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        shared_backoff: SharedBackoff | None = None,
        observer: Observer | None = None,
    ) -> None:
        """Build `Gemini` without sending a request.

        `client=None` constructs `genai.Client(vertexai=False)`.
        A passed `client` must reach the Gemini Developer API.
        A passed `client` must use the SDK's default httpx2 HTTP client, because langchaint supports no other.
        `shared_backoff` admits every request of the created `LLM` values.
        `shared_backoff=None` creates a `SharedBackoff` with its defaults.
        `observer` follows every input and every tool dispatch of the created `LLM` values.
        `observer=None` follows none.

        Raises:
            ValueError: `client` is absent and no API key is available.
        """
        self._observer = observer
        self._shared_backoff = shared_backoff if shared_backoff is not None else SharedBackoff()
        self.client: genai.Client = client if client is not None else genai.Client(vertexai=False)

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: GeminiModelId,
        *,
        pricing: GeminiPricingTable | None = ...,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: GeminiModelId,
        *,
        pricing: GeminiPricingTable | None = ...,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: GeminiModelId,
        *,
        pricing: GeminiPricingTable | None = ...,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: GeminiModelId,
        *,
        pricing: GeminiPricingTable | None = ...,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[str, None]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: GeminiPricingTable,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[OutputT, ToolManager]: ...

    @overload
    def llm[OutputT: str | BaseModel](
        self,
        model: str,
        *,
        pricing: GeminiPricingTable,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[OutputT, None]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: GeminiPricingTable,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[str, ToolManager]: ...

    @overload
    def llm(
        self,
        model: str,
        *,
        pricing: GeminiPricingTable,
        service_tier: GeminiServiceTier | None = ...,
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
    ) -> GeminiLLM[str, None]: ...

    def llm(
        self,
        model: str,
        *,
        pricing: GeminiPricingTable | None = None,
        service_tier: GeminiServiceTier | None = None,
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
    ) -> GeminiLLM[str | BaseModel, ToolManager | None]:
        """Build an `LLM` for one Gemini Developer API model.

        `model` is sent verbatim.
        Cataloged models receive `GEMINI_PRICING`.
        Stated `pricing` replaces catalog pricing.
        Uncataloged models require `pricing`.
        `service_tier` sets the requested Gemini service tier.
        Responses price at `on_demand` rates because google-genai 2.34.0 drops the served tier.

        `llm` and `rebind` raise `ValueError` when:
        - `parallel_tool_calls=False`, because `generateContent` cannot disable parallel function calls.
        - `reasoning_level` differs only in letter case from a `types.ThinkingLevel` name, such as `"high"`.
          google-genai 2.33.0 would send the uppercase name instead.
        - `extra_body` has a `generationConfig` key whose value is not a mapping with only string keys.
          Every letter case and underscore spelling of `generationConfig` counts.
        - `extra_body` sets `generationConfig.candidateCount` to a value other than 1 or `None`.
          The adapter reads one candidate per response.
          Every letter case and underscore spelling of `candidateCount` counts.
        - `provider_executed_tools` is nonempty and `tool_choice` is not `"auto"`.
        - `provider_executed_tools` is nonempty and the model is not a Gemini 3 model.
        - A provider-executed tool sets no field or sets an unsupported field.
          The supported fields are `code_execution`, `file_search`, `google_maps`, `google_search`, and `url_context`.
        - Google Search enables image search.
        - Google Search is bound and `pricing.google_search_usd_per_query` is `None`.
        - Google Maps is bound and `pricing.google_maps_usd_per_query` is `None`.
        `llm` and `rebind` raise `pydantic.ValidationError` when a provider-executed tool fails `types.Tool` validation.

        `reasoning_level` sets `generationConfig.thinkingConfig.thinkingLevel` and enables `includeThoughts`.
        With `reasoning_level=None`, `extra_body` may set `generationConfig.thinkingConfig`.
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
            ValueError: An uncataloged model lacks `pricing`.
                Also raised when `client` reaches Vertex AI.
            ValueError: `tools` contains duplicate names or `tool_choice` names an absent tool.
            ValueError: `system_prompt` is an empty sequence or `extra_body` contains an adapter-populated key.
            ValueError: `max_requests` is boolean or below one, or a provider constraint listed above is violated.
            TypeError: `response_format` is a proper subclass of `str`, or the adapter does not support `tool_choice`.
            pydantic.ValidationError: The provider SDK rejects a provider-executed tool.
            pydantic.PydanticInvalidForJsonSchema: `response_format` or a tool's argument model has no JSON schema.
            pydantic.PydanticUserError: `response_format` or a tool's argument model is not fully defined.
        """
        if pricing is None:
            pricing = _PRICING_BY_MODEL_ID.get(model)
            if pricing is None:
                raise ValueError(
                    f"model {model!r} is not in GEMINI_PRICING; pass pricing= stating its rates"
                )
        adapter = GeminiGenerateContentAdapter(
            client=self.client,
            model=model,
            pricing=pricing,
            provider_name="gcp.gemini",
            service_tier=service_tier,
        )
        return GeminiLLM(
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
    "GEMINI_PRICING",
    "Gemini",
    "GeminiGenerateContentAdapter",
    "GeminiLLM",
    "GeminiModelId",
    "GeminiPricingTable",
    "GeminiRates",
    "GeminiReportedServiceTier",
    "GeminiServiceTier",
    "assembled_response",
]
