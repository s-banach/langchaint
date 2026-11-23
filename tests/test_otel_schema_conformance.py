"""Check the public chat parser against the structured payload schemas in the OTel snapshot."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from jsonschema import Draft7Validator, Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from langchaint.span_parsing import parse_otel_chat
from scripts.refresh_semconv_genai import ATTRIBUTE_SCHEMA_FILES

_SCHEMA_DIRECTORY = Path(__file__).parent / "semconv_genai"
_DRAFT7 = "http://json-schema.org/draft-07/schema#"
_JSON = TypeAdapter(JsonValue)


class _Schema(BaseModel):
    """Keywords for which the test generates examples; new constraints require new examples."""

    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    description: str | None = None
    deprecated: bool = False
    default: JsonValue = None
    format: Literal["binary"] | None = None
    type: Literal["string", "object", "array", "null"] | None = None
    const: str | None = None
    enum: tuple[str, ...] = ()
    excluded: "_Schema | None" = Field(default=None, alias="not")
    required: tuple[str, ...] = ()
    additional_properties: bool = Field(default=True, alias="additionalProperties")
    defs: dict[str, "_Schema"] = Field(default_factory=dict, alias="$defs")
    ref: str | None = Field(default=None, alias="$ref")
    options: tuple["_Schema", ...] = Field(default=(), alias="anyOf")
    properties: dict[str, "_Schema"] = Field(default_factory=dict)
    items: "_Schema | None" = None


def _examples(schema: _Schema, defs: dict[str, _Schema]) -> Iterator[JsonValue]:
    if schema.ref == _DRAFT7:
        yield {"type": "object"}
        yield False
        yield {"pattern": "["}
        yield {"type": 42}
    elif schema.ref is not None:
        assert schema.ref.startswith("#/$defs/")
        yield from _examples(defs[schema.ref.removeprefix("#/$defs/")], defs)
    elif schema.options:
        for option in schema.options:
            yield from _examples(option, defs)
    elif schema.type == "array":
        assert schema.items is not None
        for value in _examples(schema.items, defs):
            yield [value]
        yield []
    elif schema.type == "object":
        yield from _objects(schema, defs)
    elif schema.const is not None:
        yield schema.const
    elif schema.enum:
        yield from schema.enum
    elif schema.type == "string":
        yield "vendor.value"
        if schema.excluded is not None:
            assert schema.excluded.enum
            yield from schema.excluded.enum
        yield "text"
        yield "blob"
        yield "transcription"
    elif schema.type is None:
        yield {"vendor": [1, False]}
        yield "value"
        yield [1, False]
        yield False
        yield 1.5
    yield None
    yield 42
    yield {}


def _objects(schema: _Schema, defs: dict[str, _Schema]) -> Iterator[dict[str, JsonValue]]:
    fields = {name: next(_examples(prop, defs)) for name, prop in schema.properties.items()}
    yield fields
    yield {**fields, "vendor_field": [1, None]}
    if fields.get("type") == "tool_call_response":
        yield {**fields, "is_error": "yes"}
    for name, prop in schema.properties.items():
        yield {key: value for key, value in fields.items() if key != name}
        for value in _examples(prop, defs):
            yield {**fields, name: value}


def _assert_fields(value: JsonValue, stored: JsonValue) -> None:
    """Check ordinary fields and the values retained in additional_properties."""
    if isinstance(value, list) and isinstance(stored, list):
        for item, stored_item in zip(value, stored, strict=True):
            _assert_fields(item, stored_item)
    elif (
        isinstance(value, dict) and isinstance(stored, dict) and "additional_properties" in stored
    ):
        extra = stored["additional_properties"]
        assert isinstance(extra, dict)
        assert set(extra) <= {"vendor_field", "is_error"}
        fields = {
            **{key: val for key, val in stored.items() if key != "additional_properties"},
            **extra,
        }
        assert fields.keys() == value.keys()
        for key, item in value.items():
            _assert_fields(item, fields[key])
    else:
        assert type(stored) is type(value)
        assert stored == value


def _normalize_tools(payload: JsonValue) -> JsonValue:
    if not isinstance(payload, list):
        return payload
    return [{"type": "function", **item} if isinstance(item, dict) else item for item in payload]


@pytest.mark.parametrize(
    "attribute",
    [
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.prompt.variables",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
    ],
)
def test_chat_payload_matches_its_schema(attribute: str) -> None:
    raw = (_SCHEMA_DIRECTORY / ATTRIBUTE_SCHEMA_FILES[attribute]).read_bytes()
    schema = _Schema.model_validate_json(raw)
    validator = Draft202012Validator(
        json.loads(raw), format_checker=Draft7Validator.FORMAT_CHECKER
    )
    seen: set[str] = set()
    for payload in _examples(schema, schema.defs):
        encoded = json.dumps(payload, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        expected = _normalize_tools(payload) if attribute == "gen_ai.tool.definitions" else payload
        valid = validator.is_valid(expected)
        for representation in (payload, encoded):
            attributes: dict[str, JsonValue] = {
                "gen_ai.operation.name": "chat",
                attribute: representation,
            }
            try:
                parsed = parse_otel_chat(attributes)
            except ValidationError:
                assert not valid, attributes
                continue
            assert valid, attributes
            stored = _JSON.validate_python(
                parsed.model_dump(mode="json", by_alias=True, exclude_unset=True)[attribute]
            )
            if attribute == "gen_ai.tool.definitions":
                assert parsed.tool_definitions is not None
                assert isinstance(stored, list)
                for item, definition in zip(stored, parsed.tool_definitions, strict=True):
                    assert isinstance(item, dict)
                    item["type"] = definition.type
            _assert_fields(expected, stored)
