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

from langchaint.adapter import Adapter, AllowedToolsChoice, Binding, SpecificToolChoice
from langchaint.inference_params import InferenceParams
from langchaint.messages import Message
from langchaint.tools import ToolSchema

type _CanonicalValue = None | bool | str | list[_CanonicalValue]
type _ConfigContainer = (
    Mapping[object, object] | list[object] | tuple[object, ...] | set[object] | frozenset[object]
)


@dataclass(frozen=True)
class _ResponseFormatFingerprintError:
    message: str


type ResponseFormatFingerprintData = _CanonicalValue | _ResponseFormatFingerprintError


def capture_response_format_fingerprint_data(
    response_format: type[object] | None,
) -> ResponseFormatFingerprintData:
    """Capture response-format identity and schema while binding."""
    try:
        canonical_value = _Canonicalizer().response_format(response_format)
    except TypeError as error:
        return _ResponseFormatFingerprintError(str(error))
    return canonical_value


def bound_llm_config_fingerprint(
    *,
    adapter_class: type[Adapter],
    adapter_model: str,
    adapter_provider_name: str,
    adapter_config_fingerprint_data: Mapping[str, object],
    binding: Binding,
    response_format_fingerprint_data: ResponseFormatFingerprintData,
) -> str:
    """Hash the stored configuration that can form provider requests.

    Raises:
        TypeError: A configuration value has no deterministic encoding or contains a cycle.
    """
    if isinstance(response_format_fingerprint_data, _ResponseFormatFingerprintError):
        raise TypeError(response_format_fingerprint_data.message)
    canonicalizer = _Canonicalizer()
    payload: _CanonicalValue = [
        [
            "adapter",
            _class_identity(adapter_class),
            ["model", adapter_model],
            ["provider_name", adapter_provider_name],
            [
                "config_fingerprint_data",
                canonicalizer.value(
                    adapter_config_fingerprint_data,
                    path="adapter.config_fingerprint_data()",
                ),
            ],
        ],
        ["binding", canonicalizer.value(binding, path="binding")],
        ["response_format", response_format_fingerprint_data],
    ]
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"sha256:{digest}"


def generation_input_fingerprint(messages: Sequence[Message]) -> str:
    """Hash one normalized message sequence."""
    encoded_messages = _Canonicalizer().value(list(messages), path="messages")
    digest = hashlib.sha256(_canonical_json(encoded_messages).encode()).hexdigest()
    return f"sha256:{digest}"


def _class_identity(class_: type[object]) -> _CanonicalValue:
    return ["class", class_.__module__, class_.__qualname__]


def _canonical_json(value: _CanonicalValue) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _is_base_model_class(class_: type[object]) -> TypeIs[type[BaseModel]]:
    return issubclass(class_, BaseModel)


def _is_config_container(value: object) -> TypeIs[_ConfigContainer]:
    if isinstance(value, Mapping):
        return True
    return isinstance(value, (list, tuple, set, frozenset))


class _Canonicalizer:
    """Encode supported values while tracking active containers for cycle detection."""

    def __init__(self) -> None:
        self._active_container_ids: set[int] = set()

    def response_format(self, response_format: type[object] | None) -> _CanonicalValue:
        """Encode a response format's class identity and JSON Schema."""
        if response_format is None:
            return ["none"]
        if not _is_base_model_class(response_format):
            raise TypeError("response_format must derive from pydantic.BaseModel")
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
        if isinstance(value, Binding):
            return self._binding(value, path=path)
        if isinstance(value, InferenceParams):
            return self._inference_params(value, path=path)
        if isinstance(value, ToolSchema):
            return self._tool_schema(value, path=path)
        if isinstance(value, SpecificToolChoice):
            return ["SpecificToolChoice", ["tool_name", value.tool_name]]
        if isinstance(value, AllowedToolsChoice):
            return [
                "AllowedToolsChoice",
                self._field(path, "tool_names", value.tool_names),
                ["mode", value.mode],
            ]
        if isinstance(value, BaseModel):
            return self._pydantic_model(value, path=path)
        if _is_config_container(value):
            return self._container(value, path=path)
        value_class = type(value)
        raise TypeError(
            f"{path} has unsupported type {value_class.__module__}.{value_class.__qualname__}"
        )

    def _field(self, path: str, name: str, value: object) -> _CanonicalValue:
        return [name, self.value(value, path=f"{path}.{name}")]

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

    def _binding(self, binding: Binding, *, path: str) -> _CanonicalValue:
        return [
            "Binding",
            self._field(path, "system_prompt", binding.system_prompt),
            self._field(path, "tool_schemas", binding.tool_schemas),
            self._field(path, "provider_executed_tools", binding.provider_executed_tools),
            self._field(path, "tool_choice", binding.tool_choice),
            self._field(path, "parallel_tool_calls", binding.parallel_tool_calls),
            self._field(path, "inference_params", binding.inference_params),
            self._field(
                path,
                "automatic_cache_breakpoints",
                binding.automatic_cache_breakpoints,
            ),
            self._field(path, "extra_body", binding.extra_body),
        ]

    def _inference_params(
        self, inference_params: InferenceParams, *, path: str
    ) -> _CanonicalValue:
        return [
            "InferenceParams",
            self._field(
                path,
                "max_completion_tokens",
                inference_params.max_completion_tokens,
            ),
            self._field(path, "reasoning_effort", inference_params.reasoning_effort),
            self._field(path, "temperature", inference_params.temperature),
        ]

    def _tool_schema(self, tool_schema: ToolSchema, *, path: str) -> _CanonicalValue:
        return [
            "ToolSchema",
            ["name", tool_schema.name],
            ["description", tool_schema.description],
            self._field(path, "args_schema", tool_schema.args_schema),
        ]

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
        if isinstance(value, Mapping):
            return self._mapping(value, path=path)
        if isinstance(value, list):
            return self._sequence("list", value, path=path)
        if isinstance(value, tuple):
            return self._sequence("tuple", value, path=path)
        if isinstance(value, set):
            return self._set("set", value, path=path)
        return self._set("frozenset", value, path=path)

    def _mapping(self, value: Mapping[object, object], *, path: str) -> _CanonicalValue:
        container_id = self._enter_container(value, path=path)
        try:
            entries: list[_CanonicalValue] = [
                [key, self.value(value[key], path=f"{path}[{key!r}]")]
                for key in self._sorted_string_keys(value, path=path)
            ]
            return ["mapping", entries]
        finally:
            self._active_container_ids.remove(container_id)

    def _sequence(
        self,
        kind: str,
        value: list[object] | tuple[object, ...],
        *,
        path: str,
    ) -> _CanonicalValue:
        container_id = self._enter_container(value, path=path)
        try:
            items = [self.value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
            return [kind, items]
        finally:
            self._active_container_ids.remove(container_id)

    def _set(
        self,
        kind: str,
        value: set[object] | frozenset[object],
        *,
        path: str,
    ) -> _CanonicalValue:
        container_id = self._enter_container(value, path=path)
        try:
            items = [self.value(item, path=f"{path}[set item]") for item in value]
            items.sort(key=_canonical_json)
            return [kind, items]
        finally:
            self._active_container_ids.remove(container_id)

    def _enter_container(self, value: object, *, path: str) -> int:
        container_id = id(value)
        if container_id in self._active_container_ids:
            raise TypeError(f"{path} contains a cycle")
        self._active_container_ids.add(container_id)
        return container_id

    @staticmethod
    def _sorted_string_keys(value: Mapping[object, object], *, path: str) -> list[str]:
        keys: list[str] = []
        for key in value:
            if not isinstance(key, str) or type(key) is not str:
                raise TypeError(f"{path} contains a non-string mapping key")
            keys.append(key)
        keys.sort()
        return keys
