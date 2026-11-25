"""Transform JSON schemas for structured output."""

from collections.abc import Mapping
from copy import deepcopy


def without_schema_titles(schema: Mapping[str, object]) -> dict[str, object]:
    """Copy a Draft 2020-12 JSON Schema object, omitting the `title` annotation from each subschema.

    Visit standard JSON Schema subschemas, including those under `$defs`, without following references.
    Preserve property names, literal values, and unknown keyword values.
    """
    return {key: _copy_keyword(key, value) for key, value in schema.items() if key != "title"}


def _copy_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return without_schema_titles(value)
    return deepcopy(value)


def _copy_keyword(key: str, value: object) -> object:
    match key:
        case (
            "$defs"
            | "definitions"
            | "properties"
            | "patternProperties"
            | "dependentSchemas"
            | "dependencies"
        ) if isinstance(value, Mapping):
            named_schemas: Mapping[str, object] = value
            return {name: _copy_schema(child) for name, child in named_schemas.items()}
        case "allOf" | "anyOf" | "oneOf" | "prefixItems" if isinstance(value, list):
            schemas: list[object] = value
            return [_copy_schema(child) for child in schemas]
        case (
            "additionalProperties"
            | "contains"
            | "contentSchema"
            | "else"
            | "if"
            | "items"
            | "not"
            | "propertyNames"
            | "then"
            | "unevaluatedItems"
            | "unevaluatedProperties"
        ):
            return _copy_schema(value)
        case _:
            return deepcopy(value)
