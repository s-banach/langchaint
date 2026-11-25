"""Canonical encodings for request fingerprints."""

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeIs

import pydantic
from pydantic import BaseModel

from langchaint.adapter import Adapter
from langchaint.common.messages import Message

type _CanonicalValue = None | bool | str | list[_CanonicalValue]
type _ConfigContainer = (
    Mapping[object, object] | list[object] | tuple[object, ...] | set[object] | frozenset[object]
)


@dataclass(frozen=True)
class _ResponseFormatFingerprintError:
    message: str


type ResponseFormatFingerprintData = _CanonicalValue | _ResponseFormatFingerprintError


def capture_response_format_fingerprint_data(
    response_format: type[str] | type[BaseModel],
) -> ResponseFormatFingerprintData:
    """Capture response-format identity and schema while binding."""
    try:
        canonical_value = _Canonicalizer().response_format(response_format)
    except TypeError as error:
        return _ResponseFormatFingerprintError(str(error))
    return canonical_value


def llm_config_fingerprint(
    *,
    adapter_class: type[Adapter],
    provider_name: str,
    prepared_config: Mapping[str, object],
    response_format_fingerprint_data: ResponseFormatFingerprintData,
) -> str:
    """Hash the prepared provider configuration and the output type that reads its responses.

    Raises:
        TypeError: A configuration value has no deterministic encoding or contains a cycle.
    """
    if isinstance(response_format_fingerprint_data, _ResponseFormatFingerprintError):
        raise TypeError(response_format_fingerprint_data.message)
    payload = [
        ["adapter", _class_identity(adapter_class)],
        ["provider_name", provider_name],
        ["prepared_config", dict(prepared_config)],
        ["response_format", response_format_fingerprint_data],
    ]
    try:
        # JSON converts numeric object keys to strings before they can be sorted.
        normalized: object = json.loads(json.dumps(payload, allow_nan=False))
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cannot encode prepared provider configuration as JSON: {exc}") from exc
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return f"sha256:{digest}"


def input_fingerprint(messages: Sequence[Message]) -> str:
    """Hash one normalized message sequence."""
    encoded_messages = _Canonicalizer().value(list(messages), path="messages")
    digest = hashlib.sha256(_canonical_json(encoded_messages).encode()).hexdigest()
    return f"sha256:{digest}"


def _class_identity(class_: type[object]) -> _CanonicalValue:
    return ["class", class_.__module__, class_.__qualname__]


def _canonical_json(value: _CanonicalValue) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _is_config_container(value: object) -> TypeIs[_ConfigContainer]:
    if isinstance(value, Mapping):
        return True
    return isinstance(value, (list, tuple, set, frozenset))


class _Canonicalizer:
    """Encode supported values while tracking active containers for cycle detection."""

    def __init__(self) -> None:
        self._active_container_ids: set[int] = set()

    def response_format(self, response_format: type[str] | type[BaseModel]) -> _CanonicalValue:
        """Encode `str` as text, or a response format's class identity and JSON Schema."""
        if response_format is str:
            return ["text"]
        if not issubclass(response_format, BaseModel):
            raise TypeError("response_format must be str or a pydantic.BaseModel subclass")
        try:
            response_json_schema = response_format.model_json_schema()
        except pydantic.PydanticUserError as error:
            raise TypeError(
                f"response_format.model_json_schema() cannot be serialized deterministically: {error}"
            ) from error
        return [
            "pydantic_model",
            _class_identity(response_format),
            self.value(
                response_json_schema,
                path="response_format.model_json_schema()",
            ),
        ]

    def value(self, value: object, *, path: str) -> _CanonicalValue:
        """Encode one supported value or raise `TypeError` with its path."""
        if value is None:
            return ["none"]
        if isinstance(value, Enum):
            return ["enum", _class_identity(type(value)), ["name", value.name]]
        if type(value) in (bool, int, float, str, bytes):
            return self._scalar(value, path=path)
        if isinstance(value, BaseModel):
            return self._pydantic_model(value, path=path)
        if _is_config_container(value):
            return self._container(value, path=path)
        value_class = type(value)
        raise TypeError(
            f"{path} has unsupported type {value_class.__module__}.{value_class.__qualname__}"
        )

    def _scalar(self, value: object, *, path: str) -> _CanonicalValue:
        if type(value) is bool:
            return ["bool", value]
        if type(value) is int:
            return ["int", str(value)]
        if type(value) is float:
            if not math.isfinite(value):
                raise TypeError(f"{path} contains a non-finite float")
            return ["float", value.hex()]
        if type(value) is str:
            return ["str", value]
        if type(value) is bytes:
            return ["bytes", base64.b64encode(value).decode("ascii")]
        raise AssertionError("_scalar requires a supported scalar")

    def _pydantic_model(self, value: BaseModel, *, path: str) -> _CanonicalValue:
        try:
            dumped = value.model_dump(mode="python", round_trip=True)
        except ValueError as error:
            raise TypeError(f"{path} cannot be serialized deterministically: {error}") from error
        return [
            "pydantic_model",
            _class_identity(type(value)),
            self.value(dumped, path=f"{path}.model_dump()"),
        ]

    def _container(self, value: _ConfigContainer, *, path: str) -> _CanonicalValue:
        container_id = id(value)
        if container_id in self._active_container_ids:
            raise TypeError(f"{path} contains a cycle")
        self._active_container_ids.add(container_id)
        try:
            if isinstance(value, Mapping):
                return self._mapping(value, path=path)
            if isinstance(value, list):
                return self._sequence("list", value, path=path)
            if isinstance(value, tuple):
                return self._sequence("tuple", value, path=path)
            if isinstance(value, set):
                return self._set("set", value, path=path)
            return self._set("frozenset", value, path=path)
        finally:
            self._active_container_ids.remove(container_id)

    def _mapping(self, value: Mapping[object, object], *, path: str) -> _CanonicalValue:
        entries: list[_CanonicalValue] = [
            [key, self.value(value[key], path=f"{path}[{key!r}]")]
            for key in self._sorted_string_keys(value, path=path)
        ]
        return ["mapping", entries]

    def _sequence(
        self,
        kind: str,
        value: list[object] | tuple[object, ...],
        *,
        path: str,
    ) -> _CanonicalValue:
        items = [self.value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
        return [kind, items]

    def _set(
        self,
        kind: str,
        value: set[object] | frozenset[object],
        *,
        path: str,
    ) -> _CanonicalValue:
        items = [self.value(item, path=f"{path}[set item]") for item in value]
        items.sort(key=_canonical_json)
        return [kind, items]

    @staticmethod
    def _sorted_string_keys(value: Mapping[object, object], *, path: str) -> list[str]:
        keys: list[str] = []
        for key in value:
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string mapping key")
            keys.append(key)
        return sorted(keys)
