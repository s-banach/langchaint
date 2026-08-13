"""Refresh vendored pricing metadata and generated pricing modules."""

import json
import re
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Literal, TypedDict
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "scripts/pricing/litellm-pricing-snapshot.json"
METADATA_PATH = ROOT / "scripts/pricing/provider-pricing-metadata.json"
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
    "anthropic.claude-opus-4-8": "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-4-7": "anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-5": "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-6-v1": "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}


class _MetadataValue(TypedDict):
    source_url: str
    value: float
    verified_on: str


class _ProviderMetadata(TypedDict):
    anthropic: dict[str, _MetadataValue]
    openai: dict[str, _MetadataValue]


def _download_json(url: str) -> object:
    """Download one JSON value.

    Raises:
        OSError: The download fails.
        ValueError: The response is invalid JSON.
    """
    request = Request(url, headers={"User-Agent": "langchaint-pricing-refresh"})
    with urlopen(request) as response:
        content: bytes = response.read()
    payload: object = json.loads(content)
    return payload


def _required_dict(value: object, name: str) -> dict[str, object]:
    """Validate one string-keyed object.

    Raises:
        TypeError: The value is not an object.
        ValueError: An object key is not a string.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _nonnegative_number(entry: dict[str, object], field: str) -> float:
    """Read one nonnegative numeric field.

    Raises:
        ValueError: The field is missing or invalid.
    """
    value = entry.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field} must be finite and nonnegative")
    return float(value)


def _positive_number(entry: dict[str, object], field: str) -> float:
    """Read one positive numeric field.

    Raises:
        ValueError: The field is missing or invalid.
    """
    value = _nonnegative_number(entry, field)
    if value == 0:
        raise ValueError(f"{field} must be positive")
    return value


def _metadata_items(value: object, name: str) -> dict[str, _MetadataValue]:
    """Validate one provider's documentation metadata.

    Raises:
        TypeError: A metadata value has the wrong type.
        ValueError: A metadata object key is invalid.
    """
    raw_items = _required_dict(value, name)
    items: dict[str, _MetadataValue] = {}
    for field, raw_item in raw_items.items():
        item = _required_dict(raw_item, f"{name}.{field}")
        source_url = item.get("source_url")
        raw_value = item.get("value")
        verified_on = item.get("verified_on")
        if not isinstance(source_url, str):
            raise TypeError(f"{name}.{field}.source_url must be a string")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise TypeError(f"{name}.{field}.value must be numeric")
        if not isinstance(verified_on, str):
            raise TypeError(f"{name}.{field}.verified_on must be a string")
        items[field] = {
            "source_url": source_url,
            "value": float(raw_value),
            "verified_on": verified_on,
        }
    return items


def _million_rate(entry: dict[str, object], field: str) -> str:
    """Render one per-token rate per million tokens.

    Raises:
        ValueError: The rate is missing or invalid.
    """
    value = _nonnegative_number(entry, field)
    decimal_value = Decimal(str(value)) * Decimal(1_000_000)
    return format(decimal_value.normalize(), "f")


def _validated_metadata_value(
    metadata: _ProviderMetadata,
    provider: Literal["anthropic", "openai"],
    field: str,
) -> float:
    """Read one finite provider-documentation value.

    Raises:
        KeyError: The requested metadata is missing.
        ValueError: The requested metadata is invalid.
    """
    provider_metadata = metadata[provider]
    item = provider_metadata[field]
    value = item["value"]
    if not isfinite(value) or not item["source_url"].startswith("https://"):
        raise ValueError(f"{provider}.{field} is invalid")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["verified_on"]):
        raise ValueError(f"{provider}.{field}.verified_on is invalid")
    return value


def _metadata_rate(
    metadata: _ProviderMetadata,
    provider: Literal["anthropic", "openai"],
    field: str,
) -> float:
    """Read one nonnegative provider-documentation rate.

    Raises:
        KeyError: The requested metadata is missing.
        ValueError: The requested metadata is invalid.
    """
    value = _validated_metadata_value(metadata, provider, field)
    if value < 0:
        raise ValueError(f"{provider}.{field} is invalid")
    return value


def _metadata_multiplier(
    metadata: _ProviderMetadata,
    provider: Literal["anthropic", "openai"],
    field: str,
) -> float:
    """Read one positive provider-documentation multiplier.

    Raises:
        KeyError: The requested metadata is missing.
        ValueError: The requested metadata is invalid.
    """
    value = _validated_metadata_value(metadata, provider, field)
    if value <= 0:
        raise ValueError(f"{provider}.{field} is invalid")
    return value


def _openai_rates(
    entry: dict[str, object],
    *,
    field_name: str,
    suffix: str,
    indent: str,
) -> list[str] | None:
    """Render one optional OpenAI rate category.

    Raises:
        ValueError: A present rate is invalid.
    """
    field_suffix = f"_{suffix}" if suffix else ""
    fields = (
        f"input_cost_per_token{field_suffix}",
        f"output_cost_per_token{field_suffix}",
        f"cache_read_input_token_cost{field_suffix}",
        f"cache_creation_input_token_cost{field_suffix}",
    )
    if not all(field in entry for field in fields):
        return None
    return [
        f"{indent}{field_name}=OpenAIRates(",
        f"{indent}    input_cache_none_usd_per_million_tokens={_million_rate(entry, fields[0])},",
        f"{indent}    output_usd_per_million_tokens={_million_rate(entry, fields[1])},",
        f"{indent}    cache_read_usd_per_million_tokens={_million_rate(entry, fields[2])},",
        f"{indent}    cache_write_usd_per_million_tokens={_million_rate(entry, fields[3])},",
        f"{indent}),",
    ]


def _long_context(entry: dict[str, object], *, indent: str) -> list[str]:
    """Render OpenAI long-context pricing.

    Raises:
        ValueError: Long-context fields are missing or invalid.
    """
    pattern = re.compile(r"input_cost_per_token_above_(\d+)k_tokens$")
    threshold_fields = [field for field in entry if pattern.fullmatch(field)]
    if len(threshold_fields) != 1:
        raise ValueError("OpenAI long-context threshold is ambiguous")
    threshold_field = threshold_fields[0]
    match = pattern.fullmatch(threshold_field)
    if match is None:
        raise ValueError("OpenAI long-context threshold is invalid")
    threshold = int(match.group(1)) * 1_000
    input_multiplier = _positive_number(entry, threshold_field) / _positive_number(
        entry,
        "input_cost_per_token",
    )
    output_multiplier = _positive_number(
        entry,
        f"output_cost_per_token_above_{match.group(1)}k_tokens",
    ) / _positive_number(entry, "output_cost_per_token")
    return [
        f"{indent}long_context=OpenAILongContextPricing(",
        f"{indent}    input_tokens_above={threshold},",
        f"{indent}    input_multiplier={input_multiplier!r},",
        f"{indent}    output_multiplier={output_multiplier!r},",
        f"{indent}),",
    ]


def _openai_table(
    entry: dict[str, object],
    metadata: _ProviderMetadata,
    *,
    indent: str,
    prefix: str,
    trailing_comma: bool,
) -> list[str]:
    """Render one OpenAI pricing table.

    Raises:
        KeyError: Required provider metadata is missing.
        ValueError: Pricing data or provider metadata is invalid.
    """
    lines = [f"{indent}{prefix}OpenAIPricingTable("]
    default_rates = _openai_rates(
        entry,
        field_name="default",
        suffix="",
        indent=f"{indent}    ",
    )
    if default_rates is None:
        raise ValueError("OpenAI default rates are missing")
    lines.extend(default_rates)
    for field, suffix in (("flex", "flex"), ("fast", "priority")):
        rates = _openai_rates(
            entry,
            field_name=field,
            suffix=suffix,
            indent=f"{indent}    ",
        )
        if rates is not None:
            lines.extend(rates)
    lines.extend(_long_context(entry, indent=f"{indent}    "))
    regional = _metadata_multiplier(metadata, "openai", "regional_processing_multiplier")
    closing = f"{indent})," if trailing_comma else f"{indent})"
    lines.extend([
        f"{indent}    regional_processing_multiplier={regional!r},",
        f"{indent}    web_search_usd_per_invocation={_metadata_rate(metadata, 'openai', 'web_search_usd_per_invocation')!r},",
        f"{indent}    file_search_usd_per_invocation={_metadata_rate(metadata, 'openai', 'file_search_usd_per_invocation')!r},",
        closing,
    ])
    return lines


def _anthropic_rates(
    entry: dict[str, object],
    *,
    field_name: str,
    indent: str,
    multiplier: float = 1.0,
) -> list[str]:
    """Render one Anthropic rate category.

    Raises:
        ValueError: A required rate is missing or invalid.
    """
    fields = (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
    )
    values = [Decimal(_million_rate(entry, field)) * Decimal(str(multiplier)) for field in fields]
    rendered = [format(value.normalize(), "f") for value in values]
    return [
        f"{indent}{field_name}=AnthropicRates(",
        f"{indent}    input_cache_none_usd_per_million_tokens={rendered[0]},",
        f"{indent}    output_usd_per_million_tokens={rendered[1]},",
        f"{indent}    cache_read_usd_per_million_tokens={rendered[2]},",
        f"{indent}    cache_write_5m_usd_per_million_tokens={rendered[3]},",
        f"{indent}    cache_write_1h_usd_per_million_tokens={rendered[4]},",
        f"{indent}),",
    ]


def _anthropic_table(
    entry: dict[str, object],
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
        KeyError: Required provider metadata is missing.
        ValueError: Pricing data or provider metadata is invalid.
    """
    lines = [f"{indent}{prefix}AnthropicPricingTable("]
    lines.extend(_anthropic_rates(entry, field_name="standard", indent=f"{indent}    "))
    if direct:
        batch_multiplier = _metadata_multiplier(metadata, "anthropic", "batch_multiplier")
        lines.extend(
            _anthropic_rates(
                entry,
                field_name="batch",
                indent=f"{indent}    ",
                multiplier=batch_multiplier,
            )
        )
    regional_multiplier = (
        _metadata_multiplier(metadata, "anthropic", "inference_geo_us_multiplier")
        if inference_geo
        else None
    )
    closing = f"{indent})," if trailing_comma else f"{indent})"
    lines.extend([
        f"{indent}    inference_geo_us_multiplier={regional_multiplier!r},",
        f"{indent}    web_search_usd_per_invocation={_metadata_rate(metadata, 'anthropic', 'web_search_usd_per_invocation')!r},",
        closing,
    ])
    return lines


def _constant_name(model: str) -> str:
    """Return the generated constant name for one model ID."""
    return f"_{model.upper().replace('-', '_').replace('.', '_')}"


def _openai_module(entries: dict[str, dict[str, object]], metadata: _ProviderMetadata) -> str:
    """Render the generated OpenAI module.

    Raises:
        KeyError: A required entry or metadata value is missing.
        ValueError: Pricing data or provider metadata is invalid.
    """
    model_names = [*OPENAI_ALIASES, *OPENAI_LITELLM_KEYS]
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
    ]
    for model, key in OPENAI_LITELLM_KEYS.items():
        variable = _constant_name(model)
        table = _openai_table(
            entries[key],
            metadata,
            indent="",
            prefix=f"{variable} = ",
            trailing_comma=False,
        )
        lines.extend(table)
        lines.append("")
    lines.append("OPENAI_PRICING: dict[OpenAIModelName, OpenAIPricingTable] = {")
    for alias, canonical in OPENAI_ALIASES.items():
        variable = _constant_name(canonical)
        lines.append(f'    "{alias}": {variable},')
    for model in OPENAI_LITELLM_KEYS:
        variable = _constant_name(model)
        lines.append(f'    "{model}": {variable},')
    lines.extend(["}", ""])
    return "\n".join(lines)


def _anthropic_module(entries: dict[str, dict[str, object]], metadata: _ProviderMetadata) -> str:
    """Render the generated Anthropic module.

    Raises:
        KeyError: A required entry or metadata value is missing.
        ValueError: Pricing data or provider metadata is invalid.
    """
    model_names = [*ANTHROPIC_LITELLM_KEYS, *ANTHROPIC_ALIASES]
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
    ]
    for model, key in ANTHROPIC_LITELLM_KEYS.items():
        variable = _constant_name(model)
        table = _anthropic_table(
            entries[key],
            metadata,
            indent="",
            direct=True,
            inference_geo=model in ANTHROPIC_INFERENCE_GEO_MODELS,
            prefix=f"{variable} = ",
            trailing_comma=False,
        )
        lines.extend(table)
        lines.append("")
    lines.append("ANTHROPIC_PRICING: dict[AnthropicModelName, AnthropicPricingTable] = {")
    for model in ANTHROPIC_LITELLM_KEYS:
        variable = _constant_name(model)
        lines.append(f'    "{model}": {variable},')
    for alias, canonical in ANTHROPIC_ALIASES.items():
        variable = _constant_name(canonical)
        lines.append(f'    "{alias}": {variable},')
    lines.extend(["}", "", "ANTHROPIC_BEDROCK_PRICING: dict[str, AnthropicPricingTable] = {"])
    for model, key in ANTHROPIC_BEDROCK_LITELLM_KEYS.items():
        table = _anthropic_table(
            entries[key],
            metadata,
            indent="    ",
            direct=False,
            inference_geo=False,
            prefix=f'"{model}": ',
            trailing_comma=True,
        )
        lines.extend(table)
    lines.extend(["}", ""])
    return "\n".join(lines)


def _typed_entries(payload: object) -> dict[str, dict[str, object]]:
    """Filter and validate selected LiteLLM entries.

    Raises:
        TypeError: LiteLLM data has the wrong shape.
        ValueError: A key or provider is invalid.
    """
    raw_entries = _required_dict(payload, "LiteLLM pricing")
    keys = {
        *OPENAI_LITELLM_KEYS.values(),
        *ANTHROPIC_LITELLM_KEYS.values(),
        *ANTHROPIC_BEDROCK_LITELLM_KEYS.values(),
    }
    entries: dict[str, dict[str, object]] = {}
    for key in sorted(keys):
        entry = _required_dict(raw_entries.get(key), key)
        expected_provider = "openai" if key in OPENAI_LITELLM_KEYS.values() else None
        if expected_provider is not None and entry.get("litellm_provider") != expected_provider:
            raise ValueError(f"{key} has an unexpected provider")
        if key in ANTHROPIC_LITELLM_KEYS.values() and entry.get("litellm_provider") != "anthropic":
            raise ValueError(f"{key} has an unexpected provider")
        if key in ANTHROPIC_BEDROCK_LITELLM_KEYS.values() and entry.get(
            "litellm_provider"
        ) not in {"bedrock", "bedrock_converse"}:
            raise ValueError(f"{key} has an unexpected provider")
        entries[key] = entry
    return entries


def main() -> None:
    """Refresh pricing files from current upstream data.

    Raises:
        OSError: A download or file operation fails.
        TypeError: Upstream data has the wrong shape.
        ValueError: Upstream data or metadata is invalid.
        SyntaxError: Generated Python is invalid.
    """
    metadata_payload = json.loads(METADATA_PATH.read_text())
    raw_metadata = _required_dict(metadata_payload, "provider metadata")
    anthropic_metadata = _metadata_items(raw_metadata.get("anthropic"), "anthropic metadata")
    openai_metadata = _metadata_items(raw_metadata.get("openai"), "openai metadata")
    metadata: _ProviderMetadata = {
        "anthropic": anthropic_metadata,
        "openai": openai_metadata,
    }
    commit_payload = _required_dict(_download_json(LITELLM_COMMIT_URL), "LiteLLM commit")
    revision = commit_payload.get("sha")
    if not isinstance(revision, str) or not revision:
        raise ValueError("LiteLLM revision is missing")
    entries = _typed_entries(_download_json(LITELLM_RAW_URL.format(revision=revision)))
    snapshot = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    openai_module = _openai_module(entries, metadata)
    anthropic_module = _anthropic_module(entries, metadata)
    _ = compile(openai_module, str(OPENAI_OUTPUT_PATH), "exec")
    _ = compile(anthropic_module, str(ANTHROPIC_OUTPUT_PATH), "exec")
    _ = SNAPSHOT_PATH.write_text(snapshot)
    _ = OPENAI_OUTPUT_PATH.write_text(openai_module)
    _ = ANTHROPIC_OUTPUT_PATH.write_text(anthropic_module)


if __name__ == "__main__":
    main()
