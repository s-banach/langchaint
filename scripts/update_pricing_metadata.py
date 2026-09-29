"""Refresh vendored pricing metadata and generated pricing modules."""

import json
import re
from collections.abc import Iterator, Mapping
from datetime import date
from decimal import Decimal
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, overload
from urllib.request import Request, urlopen

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StringConstraints,
    TypeAdapter,
)

from langchaint.common.checked_copy import CheckedCopyModel

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "scripts/pricing/litellm-pricing-snapshot.json"
METADATA_PATH = ROOT / "scripts/pricing/provider-pricing-metadata.json"
IGNORED_MODEL_KEYS_PATH = ROOT / "scripts/pricing/ignored-litellm-model-keys.json"
"""Hand-maintained direct-API LiteLLM keys that langchaint deliberately does not price."""
OPENAI_OUTPUT_PATH = ROOT / "src/langchaint/openai/_generated_pricing.py"
ANTHROPIC_OUTPUT_PATH = ROOT / "src/langchaint/anthropic/_generated_pricing.py"

LITELLM_COMMIT_URL = "https://api.github.com/repos/BerriAI/litellm/commits/main"
LITELLM_RAW_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/"
    "{revision}/model_prices_and_context_window.json"
)

OPENAI_LITELLM_KEYS = {
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-6.1-sol": "gpt-6.1-sol",
    "gpt-6-sol": "gpt-6-sol",
    "gpt-6-luna": "gpt-6-luna",
    "gpt-6-astra": "gpt-6-astra",
}
OPENAI_ALIASES = {"gpt-5.6": "gpt-5.6-sol"}
ANTHROPIC_LITELLM_KEYS = {
    "claude-fable-5-1": "claude-fable-5-1",
    "claude-fable-5": "claude-fable-5",
    "claude-opus-5-5": "claude-opus-5-5",
    "claude-opus-5": "claude-opus-5",
    "claude-sonnet-5-5": "claude-sonnet-5-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}
ANTHROPIC_ALIASES = {"claude-haiku-4-5": "claude-haiku-4-5-20251001"}
ANTHROPIC_INFERENCE_GEO_MODELS: frozenset[str] = frozenset({
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
})
ANTHROPIC_BEDROCK_LITELLM_KEYS = {
    "anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "anthropic.claude-fable-5": "anthropic.claude-fable-5",
    "anthropic.claude-opus-5-5": "anthropic.claude-opus-5-5",
    "anthropic.claude-opus-5": "anthropic.claude-opus-5",
    "anthropic.claude-opus-4-8": "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7": "anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-5-5": "anthropic.claude-sonnet-5-5",
    "anthropic.claude-sonnet-5": "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-6-v1": "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}
BEDROCK_CROSS_REGION_PREFIXES = ("us", "eu", "au", "jp", "apac", "global", "us-gov")
"""Bedrock cross-region inference profile prefixes whose LiteLLM entries are kept when present."""
_LONG_CONTEXT_INPUT_FIELD = re.compile(r"input_cost_per_token_above_(\d+)k_tokens")
_TIER_RATE_FIELD = re.compile(
    r"(?:input_cost_per_token|output_cost_per_token|cache_read_input_token_cost"
    r"|cache_creation_input_token_cost)(?:_above_\d+k_tokens)?_(?P<tier>[a-z]+)"
)
"""A LiteLLM rate field of one pricing tier, such as `input_cost_per_token_flex`."""
_LONG_CONTEXT_RATE_FIELD = re.compile(
    r"(?P<rate>input_cost_per_token|output_cost_per_token|cache_read_input_token_cost"
    r"|cache_creation_input_token_cost)_above_(?P<thousands>\d+)k_tokens(?:_(?P<tier>[a-z]+))?"
)
"""A LiteLLM long-context rate field, such as `output_cost_per_token_above_272k_tokens_flex`."""
OPENAI_RENDERED_TIERS = frozenset({"flex", "priority", "ultrafast"})
"""LiteLLM tiers the OpenAI tables render besides the default tier."""
OPENAI_KNOWN_TIERS = frozenset({*OPENAI_RENDERED_TIERS, "batches"})
"""`OPENAI_RENDERED_TIERS` plus `batches`, which no langchaint OpenAI adapter sends."""
ANTHROPIC_KNOWN_TIERS = frozenset({"batches"})
"""LiteLLM tiers the Anthropic tables render; Bedrock tables omit batch rates."""


_Rate = Annotated[StrictInt | StrictFloat, Field(ge=0, allow_inf_nan=False)]
"""A finite nonnegative per-token rate; `int` survives so the snapshot reproduces the upstream value."""
_RATE_ADAPTER = TypeAdapter[int | float](_Rate)
_Multiplier = Annotated[StrictInt | StrictFloat, Field(gt=0, allow_inf_nan=False)]
"""A finite positive price multiplier."""


@overload
def _decimal(rate: float) -> Decimal: ...
@overload
def _decimal(rate: float | None) -> Decimal | None: ...
def _decimal(rate: float | None) -> Decimal | None:
    """Return the shortest decimal that parses to `rate`, `None` when the rate is unlisted.

    This equals the literal LiteLLM wrote whenever that literal has at most 15 significant digits,
    as every rate does; `Decimal(rate)` would expand the float's binary approximation instead.
    """
    return None if rate is None else Decimal(str(rate))


class _AnthropicRateFields(NamedTuple):
    """The five Anthropic token rates of one LiteLLM entry, `None` where the entry lists none."""

    input_cache_none: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None
    cache_write_5m: Decimal | None
    cache_write_1h: Decimal | None


class _OpenAIRateFields(NamedTuple):
    """The four OpenAI token rates of one service tier, `None` where the entry lists none."""

    input_cache_none: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None
    cache_write: Decimal | None


class _LongContextFields(NamedTuple):
    """The OpenAI long-context threshold and the rates above it."""

    input_tokens_above: int
    input_cache_none: Decimal
    output: Decimal


class _SearchContextCost(BaseModel):
    """LiteLLM's web search prices per call, one per search context size.

    Validation rejects a listed price that is not a finite nonnegative number.
    Unknown fields are kept so the snapshot reproduces the upstream entry, which is why this model
    does not inherit `CheckedCopyModel`.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    search_context_size_low: _Rate | None = None
    search_context_size_medium: _Rate | None = None
    search_context_size_high: _Rate | None = None


class _LiteLLMEntry(BaseModel):
    """One selected LiteLLM model entry.

    Validation rejects a listed rate that is not a finite nonnegative number, and a listed
    multiplier that is not finite and positive, before any module renders it.
    A value that is absent or `null` upstream is unlisted.
    Unknown fields are kept so the snapshot reproduces the upstream entry, which is why this model
    does not inherit `CheckedCopyModel`.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    litellm_provider: str
    input_cost_per_token: _Rate | None = None
    output_cost_per_token: _Rate | None = None
    cache_read_input_token_cost: _Rate | None = None
    cache_creation_input_token_cost: _Rate | None = None
    cache_creation_input_token_cost_above_1hr: _Rate | None = None
    input_cost_per_token_flex: _Rate | None = None
    output_cost_per_token_flex: _Rate | None = None
    cache_read_input_token_cost_flex: _Rate | None = None
    cache_creation_input_token_cost_flex: _Rate | None = None
    input_cost_per_token_priority: _Rate | None = None
    output_cost_per_token_priority: _Rate | None = None
    cache_read_input_token_cost_priority: _Rate | None = None
    cache_creation_input_token_cost_priority: _Rate | None = None
    input_cost_per_token_ultrafast: _Rate | None = None
    output_cost_per_token_ultrafast: _Rate | None = None
    cache_read_input_token_cost_ultrafast: _Rate | None = None
    cache_creation_input_token_cost_ultrafast: _Rate | None = None
    input_cost_per_token_batches: _Rate | None = None
    output_cost_per_token_batches: _Rate | None = None
    cache_read_input_token_cost_batches: _Rate | None = None
    cache_creation_input_token_cost_batches: _Rate | None = None
    regional_processing_uplift_multiplier_us: _Multiplier | None = None
    regional_processing_uplift_multiplier_eu: _Multiplier | None = None
    search_context_cost_per_query: _SearchContextCost | None = None

    def require_known_tiers(self, known_tiers: frozenset[str]) -> None:
        """Reject rates for a pricing tier that the generated table would silently drop.

        Raises:
            ValueError: The entry lists a rate for a tier outside `known_tiers`.
        """
        listed_tiers = {
            match.group("tier")
            for field in self.model_dump(exclude_none=True)
            if (match := _TIER_RATE_FIELD.fullmatch(field)) is not None
        }
        if unknown_tiers := listed_tiers - known_tiers:
            raise ValueError(
                f"LiteLLM lists rates for unknown pricing tiers {sorted(unknown_tiers)}"
            )

    def web_search_usd_per_invocation(self) -> float | None:
        """Return the web search price per call, `None` when LiteLLM lists none.

        LiteLLM lists one price per search context size, and each pricing table holds one price.

        Raises:
            ValueError: The prices are listed for only some search context sizes, or differ.
        """
        cost = self.search_context_cost_per_query
        if cost is None:
            return None
        prices = {
            cost.search_context_size_low,
            cost.search_context_size_medium,
            cost.search_context_size_high,
        }
        if len(prices) != 1:
            raise ValueError("web search prices are partially listed or differ by context size")
        (price,) = prices
        return price

    def anthropic_rate_fields(self) -> _AnthropicRateFields:
        """Collect the Anthropic token rates without requiring any of them."""
        return _AnthropicRateFields(
            input_cache_none=_decimal(self.input_cost_per_token),
            output=_decimal(self.output_cost_per_token),
            cache_read=_decimal(self.cache_read_input_token_cost),
            cache_write_5m=_decimal(self.cache_creation_input_token_cost),
            cache_write_1h=_decimal(self.cache_creation_input_token_cost_above_1hr),
        )

    def anthropic_batch_rate_fields(self) -> _AnthropicRateFields:
        """Collect the Anthropic Batch API token rates without requiring any of them.

        LiteLLM lists no batch rate for the 1-hour cache write.
        Anthropic's pricing page (https://platform.claude.com/docs/en/about-claude/pricing) states
        that the cache-write multipliers stack with the Batch API discount, so the 1-hour cache
        write rate is scaled by the ratio of the 5-minute cache write's batch rate to its standard
        rate.

        Raises:
            ValueError: The 1-hour and batch 5-minute cache-write rates are listed, but the
                standard 5-minute cache-write rate is unlisted or zero.
        """
        standard = self.anthropic_rate_fields()
        batch_cache_write_5m = _decimal(self.cache_creation_input_token_cost_batches)
        batch_cache_write_1h = (
            None
            if standard.cache_write_1h is None or batch_cache_write_5m is None
            else standard.cache_write_1h
            * batch_cache_write_5m
            / _positive(standard.cache_write_5m)
        )
        return _AnthropicRateFields(
            input_cache_none=_decimal(self.input_cost_per_token_batches),
            output=_decimal(self.output_cost_per_token_batches),
            cache_read=_decimal(self.cache_read_input_token_cost_batches),
            cache_write_5m=batch_cache_write_5m,
            cache_write_1h=batch_cache_write_1h,
        )

    def regional_processing_multiplier(self) -> float:
        """Return the OpenAI token-price multiplier for regional processing endpoints.

        LiteLLM lists one multiplier per region, and `OpenAIPricingTable` holds one for all regions.

        Raises:
            ValueError: A regional multiplier is unlisted, or the US and EU multipliers differ.
        """
        multiplier = self.regional_processing_uplift_multiplier_us
        if multiplier is None or multiplier != self.regional_processing_uplift_multiplier_eu:
            raise ValueError("OpenAI regional processing multipliers are unlisted or differ")
        return multiplier

    def openai_rate_fields(
        self, tier: Literal["default", "flex", "priority", "ultrafast"]
    ) -> _OpenAIRateFields:
        """Collect the OpenAI token rates of one tier without requiring any of them."""
        match tier:
            case "default":
                return _OpenAIRateFields(
                    input_cache_none=_decimal(self.input_cost_per_token),
                    output=_decimal(self.output_cost_per_token),
                    cache_read=_decimal(self.cache_read_input_token_cost),
                    cache_write=_decimal(self.cache_creation_input_token_cost),
                )
            case "flex":
                return _OpenAIRateFields(
                    input_cache_none=_decimal(self.input_cost_per_token_flex),
                    output=_decimal(self.output_cost_per_token_flex),
                    cache_read=_decimal(self.cache_read_input_token_cost_flex),
                    cache_write=_decimal(self.cache_creation_input_token_cost_flex),
                )
            case "priority":
                return _OpenAIRateFields(
                    input_cache_none=_decimal(self.input_cost_per_token_priority),
                    output=_decimal(self.output_cost_per_token_priority),
                    cache_read=_decimal(self.cache_read_input_token_cost_priority),
                    cache_write=_decimal(self.cache_creation_input_token_cost_priority),
                )
            case "ultrafast":
                return _OpenAIRateFields(
                    input_cache_none=_decimal(self.input_cost_per_token_ultrafast),
                    output=_decimal(self.output_cost_per_token_ultrafast),
                    cache_read=_decimal(self.cache_read_input_token_cost_ultrafast),
                    cache_write=_decimal(self.cache_creation_input_token_cost_ultrafast),
                )

    def long_context_fields(self) -> _LongContextFields:
        """Read the single OpenAI long-context threshold and the rates above it.

        Raises:
            ValueError: The threshold is missing or ambiguous, or a rate above it is missing or
                invalid.
        """
        extra_fields = self.model_extra or {}
        matches = [
            match
            for field in extra_fields
            if (match := _LONG_CONTEXT_INPUT_FIELD.fullmatch(field)) is not None
        ]
        if len(matches) != 1:
            raise ValueError("OpenAI long-context threshold is ambiguous")
        (match,) = matches
        thousands = match.group(1)
        return _LongContextFields(
            input_tokens_above=int(thousands) * 1_000,
            input_cache_none=_decimal(_RATE_ADAPTER.validate_python(extra_fields[match.group(0)])),
            output=_decimal(
                _RATE_ADAPTER.validate_python(
                    extra_fields.get(f"output_cost_per_token_above_{thousands}k_tokens")
                )
            ),
        )

    def require_shared_long_context(self) -> None:
        """Require each rendered tier's long-context rates to follow the default tier's multipliers.

        `OpenAIPricingTable.rates_for` applies one long-context threshold to every tier, one input
        multiplier to the input, cache-read, and cache-write rates, and one output multiplier to the
        output rate.
        Rates are compared by cross-multiplication, so the check is exact in decimal.

        Raises:
            ValueError: A rendered tier lists a long-context rate at another threshold, without its
                base rate, or at another multiplier.
        """
        long_context = self.long_context_fields()
        default = self.openai_rate_fields("default")
        fields = self.model_dump(exclude_none=True)
        for field, value in fields.items():
            match = _LONG_CONTEXT_RATE_FIELD.fullmatch(field)
            if match is None or match.group("tier") not in {None, *OPENAI_RENDERED_TIERS}:
                continue
            rate, tier = match.group("rate"), match.group("tier")
            base_field = rate if tier is None else f"{rate}_{tier}"
            base_value = fields.get(base_field)
            if base_value is None:
                raise ValueError(f"LiteLLM lists {field} without {base_field}")
            base = _decimal(_RATE_ADAPTER.validate_python(base_value))
            above = _decimal(_RATE_ADAPTER.validate_python(value))
            default_base, default_above = (
                (_positive(default.output), long_context.output)
                if rate == "output_cost_per_token"
                else (_positive(default.input_cache_none), long_context.input_cache_none)
            )
            threshold = int(match.group("thousands")) * 1_000
            if threshold != long_context.input_tokens_above or (
                above * default_base != base * default_above
            ):
                raise ValueError(
                    f"LiteLLM {field} does not follow the shared long-context pricing"
                )


class _MetadataValue(CheckedCopyModel):
    """The source of one provider-documented value.

    Validation rejects a source that is not https and a verification date that is not a calendar
    date.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: Annotated[str, StringConstraints(pattern=r"^https://")]
    verified_on: date


class _MetadataRate(_MetadataValue):
    """One provider-documented rate; validation rejects a value that is not finite and nonnegative."""

    value: Annotated[StrictInt | StrictFloat, Field(ge=0, allow_inf_nan=False)]


class _MetadataMultiplier(_MetadataValue):
    """One provider-documented multiplier; validation rejects a value that is not finite and positive."""

    value: _Multiplier


class _AnthropicMetadata(CheckedCopyModel):
    """Anthropic-documented values that LiteLLM omits for some priced models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inference_geo_us_multiplier: _MetadataMultiplier


class _OpenAIMetadata(CheckedCopyModel):
    """OpenAI-documented tool prices that LiteLLM lists outside the model entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_search_usd_per_invocation: _MetadataRate


class _ProviderMetadata(CheckedCopyModel):
    """The vendored provider documentation values.

    Validation rejects a missing or misspelled value so a generated module never silently omits
    one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anthropic: _AnthropicMetadata
    openai: _OpenAIMetadata


class _GitHubCommit(BaseModel):
    """The GitHub commit response field the refresh reads.

    The response carries many other fields, so this model ignores extras and does not inherit
    `CheckedCopyModel`.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    sha: Annotated[str, StringConstraints(min_length=1)]


class _LiteLLMModelKind(BaseModel):
    """The LiteLLM entry fields that decide whether new-model detection reports an entry.

    Entries carry many other fields, so this model ignores extras and does not inherit
    `CheckedCopyModel`.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    litellm_provider: str | None = None
    mode: str | None = None

    def is_direct_api_text_model(self) -> bool:
        """Report whether the entry is an OpenAI or Anthropic direct-API text generation model."""
        return self.litellm_provider in {"openai", "anthropic"} and self.mode in {
            "chat",
            "responses",
        }


_LITELLM_FILE = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])
_MODEL_KEYS = TypeAdapter[frozenset[str]](frozenset[str])


def _download(url: str) -> bytes:
    """Download one response body.

    Raises:
        OSError: The download fails.
    """
    request = Request(url, headers={"User-Agent": "langchaint-pricing-refresh"})
    with urlopen(request) as response:
        content: bytes = response.read()
    return content


def _positive(rate: Decimal | None) -> Decimal:
    """Require a listed positive rate.

    Raises:
        ValueError: The rate is unlisted or zero.
    """
    if rate is None or rate == 0:
        raise ValueError("a required rate is unlisted or zero")
    return rate


def _decimal_ratio(rate: Decimal | None, base_rate: Decimal | None) -> float:
    """Divide two listed positive rates in decimal, so `7.5e-05 / 5e-05` renders as `1.5`.

    Raises:
        ValueError: A rate is unlisted or zero.
    """
    return float(_positive(rate) / _positive(base_rate))


def _million_rate(rate: Decimal | None) -> str:
    """Render one per-token rate per million tokens.

    Raises:
        ValueError: The rate is unlisted.
    """
    if rate is None:
        raise ValueError("a required rate is unlisted")
    return format((rate * 1_000_000).normalize(), "f")


def _openai_rates(
    fields: _OpenAIRateFields,
    *,
    field_name: str,
    indent: str,
) -> list[str]:
    """Render one OpenAI rate category, or nothing when the entry lists none of its rates.

    Raises:
        ValueError: The entry lists only some of the category's rates.
    """
    if all(rate is None for rate in fields):
        return []
    return [
        f"{indent}{field_name}=OpenAIRates(",
        f"{indent}    input_cache_none_usd_per_million_tokens={_million_rate(fields.input_cache_none)},",
        f"{indent}    output_usd_per_million_tokens={_million_rate(fields.output)},",
        f"{indent}    cache_read_usd_per_million_tokens={_million_rate(fields.cache_read)},",
        f"{indent}    cache_write_usd_per_million_tokens={_million_rate(fields.cache_write)},",
        f"{indent}),",
    ]


def _long_context(entry: _LiteLLMEntry, *, indent: str) -> list[str]:
    """Render OpenAI long-context pricing.

    Raises:
        ValueError: Long-context fields are missing, invalid, or ambiguous, a base rate is
            unlisted or zero, or a tier's long-context rates follow other multipliers.
    """
    entry.require_shared_long_context()
    fields = entry.long_context_fields()
    default_fields = entry.openai_rate_fields("default")
    input_multiplier = _decimal_ratio(fields.input_cache_none, default_fields.input_cache_none)
    output_multiplier = _decimal_ratio(fields.output, default_fields.output)
    return [
        f"{indent}long_context=OpenAILongContextPricing(",
        f"{indent}    input_tokens_above={fields.input_tokens_above},",
        f"{indent}    input_multiplier={input_multiplier!r},",
        f"{indent}    output_multiplier={output_multiplier!r},",
        f"{indent}),",
    ]


def _openai_table(
    entry: _LiteLLMEntry,
    metadata: _ProviderMetadata,
    *,
    indent: str,
    prefix: str,
    trailing_comma: bool,
) -> list[str]:
    """Render one OpenAI pricing table.

    Raises:
        ValueError: Pricing data or provider metadata is invalid.
    """
    entry.require_known_tiers(OPENAI_KNOWN_TIERS)
    default_fields = entry.openai_rate_fields("default")
    if None in default_fields:
        raise ValueError("OpenAI default rates are missing")
    inner = f"{indent}    "
    closing = f"{indent})," if trailing_comma else f"{indent})"
    return [
        f"{indent}{prefix}OpenAIPricingTable(",
        *_openai_rates(default_fields, field_name="default", indent=inner),
        *_openai_rates(entry.openai_rate_fields("flex"), field_name="flex", indent=inner),
        *_openai_rates(entry.openai_rate_fields("priority"), field_name="fast", indent=inner),
        *_openai_rates(
            entry.openai_rate_fields("ultrafast"), field_name="ultrafast", indent=inner
        ),
        *_long_context(entry, indent=inner),
        f"{inner}regional_processing_multiplier={entry.regional_processing_multiplier()!r},",
        f"{inner}web_search_usd_per_invocation={entry.web_search_usd_per_invocation()!r},",
        f"{inner}file_search_usd_per_invocation={metadata.openai.file_search_usd_per_invocation.value!r},",
        closing,
    ]


def _anthropic_rates(
    fields: _AnthropicRateFields,
    *,
    field_name: str,
    indent: str,
) -> list[str]:
    """Render one Anthropic rate category.

    Raises:
        ValueError: A required rate is unlisted.
    """
    return [
        f"{indent}{field_name}=AnthropicRates(",
        f"{indent}    input_cache_none_usd_per_million_tokens={_million_rate(fields.input_cache_none)},",
        f"{indent}    output_usd_per_million_tokens={_million_rate(fields.output)},",
        f"{indent}    cache_read_usd_per_million_tokens={_million_rate(fields.cache_read)},",
        f"{indent}    cache_write_5m_usd_per_million_tokens={_million_rate(fields.cache_write_5m)},",
        f"{indent}    cache_write_1h_usd_per_million_tokens={_million_rate(fields.cache_write_1h)},",
        f"{indent}),",
    ]


def _anthropic_table(
    entry: _LiteLLMEntry,
    metadata: _ProviderMetadata,
    *,
    indent: str,
    direct: bool,
    inference_geo: bool,
    prefix: str,
    trailing_comma: bool,
) -> list[str]:
    """Render one Anthropic pricing table.

    Raises:
        ValueError: Pricing data or provider metadata is invalid.
    """
    entry.require_known_tiers(ANTHROPIC_KNOWN_TIERS)
    inner = f"{indent}    "
    batch = (
        _anthropic_rates(entry.anthropic_batch_rate_fields(), field_name="batch", indent=inner)
        if direct
        else []
    )
    regional_multiplier = (
        metadata.anthropic.inference_geo_us_multiplier.value if inference_geo else None
    )
    closing = f"{indent})," if trailing_comma else f"{indent})"
    return [
        f"{indent}{prefix}AnthropicPricingTable(",
        *_anthropic_rates(entry.anthropic_rate_fields(), field_name="standard", indent=inner),
        *batch,
        f"{inner}inference_geo_us_multiplier={regional_multiplier!r},",
        f"{inner}web_search_usd_per_invocation={entry.web_search_usd_per_invocation()!r},",
        closing,
    ]


def _constant_name(model: str) -> str:
    """Return the generated constant name for one model ID."""
    return f"_{model.upper().replace('-', '_').replace('.', '_')}"


def _openai_module(entries: Mapping[str, _LiteLLMEntry], metadata: _ProviderMetadata) -> str:
    """Render the generated OpenAI module.

    Raises:
        KeyError: A required model entry is missing.
        ValueError: Pricing data or provider metadata is invalid.
    """
    model_names = [*OPENAI_ALIASES, *OPENAI_LITELLM_KEYS]
    tables = chain.from_iterable(
        [
            *_openai_table(
                entries[key],
                metadata,
                indent="",
                prefix=f"{_constant_name(model)} = ",
                trailing_comma=False,
            ),
            "",
        ]
        for model, key in OPENAI_LITELLM_KEYS.items()
    )
    lines = [
        '"""Generated OpenAI pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""',
        "",
        "from typing import Literal",
        "",
        "from langchaint.openai.shared import (",
        "    OpenAILongContextPricing,",
        "    OpenAIPricingTable,",
        "    OpenAIRates,",
        ")",
        "",
        "type OpenAIModelName = Literal[",
        *[f'    "{model}",' for model in model_names],
        "]",
        "",
        *tables,
        "OPENAI_PRICING: dict[OpenAIModelName, OpenAIPricingTable] = {",
        *[
            f'    "{alias}": {_constant_name(canonical)},'
            for alias, canonical in OPENAI_ALIASES.items()
        ],
        *[f'    "{model}": {_constant_name(model)},' for model in OPENAI_LITELLM_KEYS],
        "}",
        "",
    ]
    return "\n".join(lines)


def _anthropic_module(entries: Mapping[str, _LiteLLMEntry], metadata: _ProviderMetadata) -> str:
    """Render the generated Anthropic module.

    Raises:
        KeyError: A required model entry is missing.
        ValueError: Pricing data or provider metadata is invalid.
    """
    model_names = [*ANTHROPIC_LITELLM_KEYS, *ANTHROPIC_ALIASES]
    direct_tables = chain.from_iterable(
        [
            *_anthropic_table(
                entries[key],
                metadata,
                indent="",
                direct=True,
                inference_geo=model in ANTHROPIC_INFERENCE_GEO_MODELS,
                prefix=f"{_constant_name(model)} = ",
                trailing_comma=False,
            ),
            "",
        ]
        for model, key in ANTHROPIC_LITELLM_KEYS.items()
    )
    bedrock_tables = chain.from_iterable(
        _anthropic_table(
            entries[key],
            metadata,
            indent="    ",
            direct=False,
            inference_geo=False,
            prefix=f'"{model}": ',
            trailing_comma=True,
        )
        for model, key in ANTHROPIC_BEDROCK_LITELLM_KEYS.items()
    )
    lines = [
        '"""Generated Anthropic pricing metadata. Refresh with `uv run python -m scripts.update_pricing_metadata`."""',
        "",
        "from typing import Literal",
        "",
        "from langchaint.anthropic.messages_adapter import (",
        "    AnthropicPricingTable,",
        "    AnthropicRates,",
        ")",
        "",
        "type AnthropicModelName = Literal[",
        *[f'    "{model}",' for model in model_names],
        "]",
        "",
        *direct_tables,
        "ANTHROPIC_PRICING: dict[AnthropicModelName, AnthropicPricingTable] = {",
        *[f'    "{model}": {_constant_name(model)},' for model in ANTHROPIC_LITELLM_KEYS],
        *[
            f'    "{alias}": {_constant_name(canonical)},'
            for alias, canonical in ANTHROPIC_ALIASES.items()
        ],
        "}",
        "",
        "ANTHROPIC_BEDROCK_PRICING: dict[str, AnthropicPricingTable] = {",
        *bedrock_tables,
        "}",
        "",
        "BEDROCK_CROSS_REGION_MULTIPLIER: dict[str, float] = {",
        *[
            f'    "{prefix}": {_cross_region_multiplier(entries, prefix)!r},'
            for prefix in BEDROCK_CROSS_REGION_PREFIXES
        ],
        "}",
        '"""Token-rate multiplier for each Bedrock cross-region inference profile prefix."""',
        "",
    ]
    return "\n".join(lines)


def _prefixed_key(prefix: str, key: str) -> str:
    """Return the LiteLLM key of one cross-region inference profile for one Bedrock key."""
    return f"{prefix}.{key}"


def _cross_region_multiplier(entries: Mapping[str, _LiteLLMEntry], prefix: str) -> float:
    """Return the ratio of prefixed to unprefixed Bedrock token rates for one prefix.

    A rate unlisted in a prefixed entry contributes no ratio.

    Raises:
        ValueError: No prefixed entry exists, a base rate is unlisted or zero, or the ratio
            differs across models or rate fields.
    """
    ratios = {
        ratio
        for key in ANTHROPIC_BEDROCK_LITELLM_KEYS.values()
        if (prefixed_key := _prefixed_key(prefix, key)) in entries
        for ratio in _rate_ratios(entries[prefixed_key], entries[key])
    }
    if len(ratios) != 1:
        raise ValueError(f"{prefix} has no single cross-region multiplier: {sorted(ratios)}")
    return float(ratios.pop())


def _rate_ratios(prefixed: _LiteLLMEntry, base: _LiteLLMEntry) -> Iterator[Decimal]:
    """Compare the token rates of one prefixed LiteLLM entry with its unprefixed entry.

    Yields:
        The exact ratio of each token rate listed in `prefixed` to the same rate in `base`.

    Raises:
        ValueError: A base rate is unlisted or zero.
    """
    for prefixed_rate, base_rate in zip(
        prefixed.anthropic_rate_fields(), base.anthropic_rate_fields(), strict=True
    ):
        if prefixed_rate is not None:
            yield prefixed_rate / _positive(base_rate)


def _expected_providers(key: str) -> frozenset[str]:
    """Return the LiteLLM provider values accepted for one selected key."""
    if key in OPENAI_LITELLM_KEYS.values():
        return frozenset({"openai"})
    if key in ANTHROPIC_LITELLM_KEYS.values():
        return frozenset({"anthropic"})
    return frozenset({"bedrock", "bedrock_converse"})


def _validated_entry(key: str, raw_entry: JsonValue) -> _LiteLLMEntry:
    """Validate one selected LiteLLM entry.

    Raises:
        ValueError: The entry has the wrong shape, an invalid rate, or an unexpected provider.
    """
    entry = _LiteLLMEntry.model_validate(raw_entry)
    if entry.litellm_provider not in _expected_providers(key):
        raise ValueError(f"{key} has an unexpected provider")
    return entry


def _selected_entries(raw_entries: Mapping[str, JsonValue]) -> dict[str, _LiteLLMEntry]:
    """Select and validate the LiteLLM entries the generated modules use.

    Raises:
        KeyError: A required LiteLLM key is missing.
        ValueError: A selected entry is invalid.
    """
    required_keys = {
        *OPENAI_LITELLM_KEYS.values(),
        *ANTHROPIC_LITELLM_KEYS.values(),
        *ANTHROPIC_BEDROCK_LITELLM_KEYS.values(),
    }
    optional_keys = {
        _prefixed_key(prefix, key)
        for prefix in BEDROCK_CROSS_REGION_PREFIXES
        for key in ANTHROPIC_BEDROCK_LITELLM_KEYS.values()
    }
    return {
        key: _validated_entry(key, raw_entries[key])
        for key in sorted(required_keys | (optional_keys & raw_entries.keys()))
    }


def _untracked_model_keys(
    raw_entries: Mapping[str, JsonValue], ignored_keys: frozenset[str]
) -> list[str]:
    """List the direct-API text model keys that are neither generated nor ignored.

    An OpenAI or Anthropic direct-API LiteLLM key equals the model id, so an alias key such as
    `gpt-5.6` counts as generated.
    """
    generated_keys = {
        *OPENAI_LITELLM_KEYS.values(),
        *OPENAI_ALIASES,
        *ANTHROPIC_LITELLM_KEYS.values(),
        *ANTHROPIC_ALIASES,
    }
    return sorted(
        key
        for key, raw_entry in raw_entries.items()
        if key not in generated_keys
        and key not in ignored_keys
        and _LiteLLMModelKind.model_validate(raw_entry).is_direct_api_text_model()
    )


def _snapshot_json(entries: Mapping[str, _LiteLLMEntry]) -> str:
    """Render the selected entries as the vendored snapshot, field for field as upstream lists them."""
    payload = {key: entry.model_dump(exclude_unset=True) for key, entry in entries.items()}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> None:
    """Refresh pricing files from current upstream data.

    Print one line per direct-API text model key that is neither generated nor in
    `IGNORED_MODEL_KEYS_PATH`, so the refresh workflow can report new models.

    Raises:
        OSError: A download or file operation fails.
        KeyError: A required model entry is missing.
        ValueError: Upstream data or metadata is invalid.
        SyntaxError: Generated Python is invalid.
    """
    metadata = _ProviderMetadata.model_validate_json(METADATA_PATH.read_bytes())
    ignored_keys = _MODEL_KEYS.validate_json(IGNORED_MODEL_KEYS_PATH.read_bytes())
    commit = _GitHubCommit.model_validate_json(_download(LITELLM_COMMIT_URL))
    raw_entries = _LITELLM_FILE.validate_json(
        _download(LITELLM_RAW_URL.format(revision=commit.sha))
    )
    entries = _selected_entries(raw_entries)
    untracked_model_keys = _untracked_model_keys(raw_entries, ignored_keys)
    snapshot = _snapshot_json(entries)
    openai_module = _openai_module(entries, metadata)
    anthropic_module = _anthropic_module(entries, metadata)
    _ = compile(openai_module, str(OPENAI_OUTPUT_PATH), "exec")
    _ = compile(anthropic_module, str(ANTHROPIC_OUTPUT_PATH), "exec")
    _ = SNAPSHOT_PATH.write_text(snapshot)
    _ = OPENAI_OUTPUT_PATH.write_text(openai_module)
    _ = ANTHROPIC_OUTPUT_PATH.write_text(anthropic_module)
    for key in untracked_model_keys:
        print(key)


if __name__ == "__main__":
    main()
