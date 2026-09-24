"""Refresh vendored pricing metadata and generated pricing modules."""

import json
import re
from collections.abc import Iterator, Mapping
from datetime import date
from decimal import Decimal
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal, NamedTuple
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
}
OPENAI_ALIASES = {"gpt-5.6": "gpt-5.6-sol"}
ANTHROPIC_LITELLM_KEYS = {
    "claude-fable-5": "claude-fable-5",
    "claude-opus-5": "claude-opus-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}
ANTHROPIC_ALIASES = {"claude-haiku-4-5": "claude-haiku-4-5-20251001"}
ANTHROPIC_INFERENCE_GEO_MODELS: frozenset[str] = frozenset({
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
})
ANTHROPIC_BEDROCK_LITELLM_KEYS = {
    "anthropic.claude-fable-5": "anthropic.claude-fable-5",
    "anthropic.claude-opus-5": "anthropic.claude-opus-5",
    "anthropic.claude-opus-4-8": "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7": "anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-5": "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-6-v1": "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}
BEDROCK_CROSS_REGION_PREFIXES = ("us", "eu", "au", "jp", "apac", "global", "us-gov")
"""Bedrock cross-region inference profile prefixes whose LiteLLM entries are kept when present."""
_LONG_CONTEXT_INPUT_FIELD = re.compile(r"input_cost_per_token_above_(\d+)k_tokens")


_Rate = Annotated[StrictInt | StrictFloat, Field(ge=0, allow_inf_nan=False)]
"""A finite nonnegative per-token rate; `int` survives so the snapshot reproduces the upstream value."""
_RATE_ADAPTER = TypeAdapter[int | float](_Rate)


class _AnthropicRateFields(NamedTuple):
    """The five Anthropic token rates of one LiteLLM entry, `None` where the entry lists none."""

    input_cache_none: float | None
    output: float | None
    cache_read: float | None
    cache_write_5m: float | None
    cache_write_1h: float | None


class _OpenAIRateFields(NamedTuple):
    """The four OpenAI token rates of one service tier, `None` where the entry lists none."""

    input_cache_none: float | None
    output: float | None
    cache_read: float | None
    cache_write: float | None


class _LongContextFields(NamedTuple):
    """The OpenAI long-context threshold and the rates above it."""

    input_tokens_above: int
    input_cache_none: float
    output: float


class _LiteLLMEntry(BaseModel):
    """One selected LiteLLM model entry.

    Validation rejects a listed rate that is not a finite nonnegative number before any module
    renders it.
    A rate that is absent or `null` upstream is unlisted.
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

    def anthropic_rate_fields(self) -> _AnthropicRateFields:
        """Collect the Anthropic token rates without requiring any of them."""
        return _AnthropicRateFields(
            input_cache_none=self.input_cost_per_token,
            output=self.output_cost_per_token,
            cache_read=self.cache_read_input_token_cost,
            cache_write_5m=self.cache_creation_input_token_cost,
            cache_write_1h=self.cache_creation_input_token_cost_above_1hr,
        )

    def openai_rate_fields(
        self, tier: Literal["default", "flex", "priority"]
    ) -> _OpenAIRateFields:
        """Collect the OpenAI token rates of one tier without requiring any of them."""
        match tier:
            case "default":
                return _OpenAIRateFields(
                    input_cache_none=self.input_cost_per_token,
                    output=self.output_cost_per_token,
                    cache_read=self.cache_read_input_token_cost,
                    cache_write=self.cache_creation_input_token_cost,
                )
            case "flex":
                return _OpenAIRateFields(
                    input_cache_none=self.input_cost_per_token_flex,
                    output=self.output_cost_per_token_flex,
                    cache_read=self.cache_read_input_token_cost_flex,
                    cache_write=self.cache_creation_input_token_cost_flex,
                )
            case "priority":
                return _OpenAIRateFields(
                    input_cache_none=self.input_cost_per_token_priority,
                    output=self.output_cost_per_token_priority,
                    cache_read=self.cache_read_input_token_cost_priority,
                    cache_write=self.cache_creation_input_token_cost_priority,
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
            input_cache_none=_RATE_ADAPTER.validate_python(extra_fields[match.group(0)]),
            output=_RATE_ADAPTER.validate_python(
                extra_fields.get(f"output_cost_per_token_above_{thousands}k_tokens")
            ),
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

    value: Annotated[StrictInt | StrictFloat, Field(gt=0, allow_inf_nan=False)]


class _AnthropicMetadata(CheckedCopyModel):
    """Anthropic-documented values that LiteLLM does not list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_multiplier: _MetadataMultiplier
    inference_geo_us_multiplier: _MetadataMultiplier
    web_search_usd_per_invocation: _MetadataRate


class _OpenAIMetadata(CheckedCopyModel):
    """OpenAI-documented values that LiteLLM does not list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_search_usd_per_invocation: _MetadataRate
    regional_processing_multiplier: _MetadataMultiplier
    web_search_usd_per_invocation: _MetadataRate


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


def _positive(rate: float | None) -> float:
    """Require a listed positive rate.

    Raises:
        ValueError: The rate is unlisted or zero.
    """
    if rate is None or rate == 0:
        raise ValueError("a required rate is unlisted or zero")
    return rate


def _million_rate(rate: float | None, multiplier: float = 1.0) -> str:
    """Render one per-token rate per million tokens.

    Raises:
        ValueError: The rate is unlisted.
    """
    if rate is None:
        raise ValueError("a required rate is unlisted")
    decimal_value = Decimal(str(rate)) * Decimal(1_000_000) * Decimal(str(multiplier))
    return format(decimal_value.normalize(), "f")


def _openai_rates(
    fields: _OpenAIRateFields,
    *,
    field_name: str,
    indent: str,
) -> list[str]:
    """Render one OpenAI rate category, or nothing when any of its rates is unlisted."""
    if None in fields:
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
        ValueError: Long-context fields are missing, invalid, or ambiguous, or a base rate is
            unlisted or zero.
    """
    fields = entry.long_context_fields()
    input_multiplier = _positive(fields.input_cache_none) / _positive(entry.input_cost_per_token)
    output_multiplier = _positive(fields.output) / _positive(entry.output_cost_per_token)
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
        *_long_context(entry, indent=inner),
        f"{inner}regional_processing_multiplier={metadata.openai.regional_processing_multiplier.value!r},",
        f"{inner}web_search_usd_per_invocation={metadata.openai.web_search_usd_per_invocation.value!r},",
        f"{inner}file_search_usd_per_invocation={metadata.openai.file_search_usd_per_invocation.value!r},",
        closing,
    ]


def _anthropic_rates(
    fields: _AnthropicRateFields,
    *,
    field_name: str,
    indent: str,
    multiplier: float = 1.0,
) -> list[str]:
    """Render one Anthropic rate category.

    Raises:
        ValueError: A required rate is unlisted.
    """
    return [
        f"{indent}{field_name}=AnthropicRates(",
        f"{indent}    input_cache_none_usd_per_million_tokens={_million_rate(fields.input_cache_none, multiplier)},",
        f"{indent}    output_usd_per_million_tokens={_million_rate(fields.output, multiplier)},",
        f"{indent}    cache_read_usd_per_million_tokens={_million_rate(fields.cache_read, multiplier)},",
        f"{indent}    cache_write_5m_usd_per_million_tokens={_million_rate(fields.cache_write_5m, multiplier)},",
        f"{indent}    cache_write_1h_usd_per_million_tokens={_million_rate(fields.cache_write_1h, multiplier)},",
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
    fields = entry.anthropic_rate_fields()
    inner = f"{indent}    "
    batch = (
        _anthropic_rates(
            fields,
            field_name="batch",
            indent=inner,
            multiplier=metadata.anthropic.batch_multiplier.value,
        )
        if direct
        else []
    )
    regional_multiplier = (
        metadata.anthropic.inference_geo_us_multiplier.value if inference_geo else None
    )
    closing = f"{indent})," if trailing_comma else f"{indent})"
    return [
        f"{indent}{prefix}AnthropicPricingTable(",
        *_anthropic_rates(fields, field_name="standard", indent=inner),
        *batch,
        f"{inner}inference_geo_us_multiplier={regional_multiplier!r},",
        f"{inner}web_search_usd_per_invocation={metadata.anthropic.web_search_usd_per_invocation.value!r},",
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
            yield Decimal(str(prefixed_rate)) / Decimal(str(_positive(base_rate)))


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
