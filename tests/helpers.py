"""Helpers shared by more than one test module."""

import asyncio
import functools
import importlib
import json
import pathlib
import pkgutil
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from types import ModuleType
from typing import Literal, override

import httpx2
import jsonschema
import openai
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

import langchaint
from langchaint import (
    ZERO_USAGE,
    AbandonedStreamRecord,
    AssistantMessage,
    Billing,
    CutOffRequestRecord,
    DispatchOutcome,
    GenerationOutcome,
    RequestHistory,
    SettledRequestRecord,
    TokenRates,
    ToolCall,
    TransientErrorRecord,
    Usage,
)
from langchaint.adapter import ProviderBilling, RequestFailure, TransientError
from langchaint.common.observed_operation import ObservedOperation
from langchaint.embedding_requests import EmbeddingRequestRecord, EmbeddingRequestStart
from langchaint.generation.observer import GenerationStart
from scripts import refresh_semconv_genai


class StubRaw(BaseModel):
    """Stand-in for the SDK's own response model a generation carries on raw."""


def stated_billing(
    usage: Usage,
    *,
    input_tokens_cache_none_usd_per_million_tokens: float = float("nan"),
) -> Billing:
    return Billing(
        usage=usage,
        service_tier="stub",
        usd_per_million_tokens=TokenRates(
            input_tokens_cache_read=float("nan"),
            input_tokens_cache_write=float("nan"),
            input_tokens_cache_none=input_tokens_cache_none_usd_per_million_tokens,
            output_tokens=float("nan"),
        ),
    )


def stated_provider_billing(
    usage: Usage,
    *,
    input_tokens_cache_none_usd_per_million_tokens: float = float("nan"),
    usage_raw: BaseModel | None = None,
) -> ProviderBilling:
    return ProviderBilling(
        billing=stated_billing(
            usage,
            input_tokens_cache_none_usd_per_million_tokens=input_tokens_cache_none_usd_per_million_tokens,
        ),
        usage_raw=usage_raw,
    )


def request_record(
    *,
    error: TransientError | TransientErrorRecord | None,
    usage: Usage = ZERO_USAGE,
    reported_billing: bool = True,
    input_tokens_cache_none_usd_per_million_tokens: float = float("nan"),
    started_after_seconds: float = 0.0,
    elapsed_seconds: float = 0.0,
    first_item_after_seconds: float | None = None,
    assistant_message: AssistantMessage | None = None,
    model_served: str | None = None,
    response_id: str | None = None,
    request_id: str | None = None,
) -> SettledRequestRecord:
    normalized_error = (
        TransientErrorRecord(
            error_text=str(error),
            retry_after_seconds=error.retry_after_seconds,
            pauses_quota=error.pauses_quota,
        )
        if isinstance(error, TransientError)
        else error
    )
    return SettledRequestRecord(
        started_after_seconds=started_after_seconds,
        elapsed_seconds=elapsed_seconds,
        first_item_after_seconds=first_item_after_seconds,
        error=normalized_error,
        billing=(
            stated_billing(
                usage,
                input_tokens_cache_none_usd_per_million_tokens=input_tokens_cache_none_usd_per_million_tokens,
            )
            if reported_billing
            else None
        ),
        assistant_message=assistant_message,
        model_served=model_served,
        response_id=response_id,
        request_id=request_id,
    )


def call_record(
    records: tuple[SettledRequestRecord | CutOffRequestRecord, ...], *, elapsed_seconds: float
) -> RequestHistory:
    return RequestHistory(
        model="fake-model",
        provider_name="fake",
        records=records,
        elapsed_seconds=elapsed_seconds,
    )


def package_modules() -> Iterator[ModuleType]:
    """Import every langchaint module, including backends and tracing.

    Backend imports require their provider SDKs.
    Tracing imports require opentelemetry-api.
    The development environment installs those dependencies.

    Yields:
        Each imported module, with langchaint first.
    """
    yield langchaint
    for module_info in pkgutil.walk_packages(langchaint.__path__, prefix="langchaint."):
        yield importlib.import_module(module_info.name)


TEST_TIMEOUT_SECONDS = 5.0


def run_with_timeout[ReturnT](awaitable: Awaitable[ReturnT]) -> ReturnT:
    """Run `awaitable` in a new event loop and return its result.

    The timeout makes a deadlocked test fail instead of hanging the suite.

    Raises:
        TimeoutError: `awaitable` ran longer than `TEST_TIMEOUT_SECONDS`.
    """

    async def guarded() -> ReturnT:
        return await asyncio.wait_for(awaitable, timeout=TEST_TIMEOUT_SECONDS)

    return asyncio.run(guarded())


async def yield_until(condition: Callable[[], bool]) -> None:
    """Yield to the event loop until `condition` holds.

    A test calls this to wait for another task to reach a state, such as joining a queue.
    `asyncio.sleep(0)` lets the other ready tasks run without waiting for time to pass.
    Call it under `run_with_timeout`, whose timeout ends a wait whose condition never holds.
    """
    while not condition():
        await asyncio.sleep(0)


async def time_out_when[ReturnT](
    awaitable: Awaitable[ReturnT], condition: Callable[[], bool]
) -> ReturnT:
    """Await `awaitable` in the current task under a timeout that expires once `condition` holds.

    The expiry cancels `awaitable` at the point it has reached, as a caller's own `asyncio.timeout` would.
    A test calls this instead of a short timeout, so the cancellation lands without waiting for time to pass.

    Raises:
        TimeoutError: the timeout expired and `awaitable` let the cancellation through.
        Exception: whatever `condition` raised, after the timeout cancels `awaitable`.
    """
    async with asyncio.timeout(None) as scope:

        async def expire_once_condition_holds() -> None:
            # A condition that raises also expires the scope, so the test fails at once.
            # The `finally` below then re-raises the condition's exception.
            try:
                await yield_until(condition)
            except Exception:
                scope.reschedule(asyncio.get_running_loop().time())
                raise
            scope.reschedule(asyncio.get_running_loop().time())

        expiry = asyncio.create_task(expire_once_condition_holds())
        try:
            return await awaitable
        finally:
            _ = expiry.cancel()
            if expiry.done() and not expiry.cancelled():
                expiry.result()


def status_error[ErrorT: openai.APIStatusError](
    error_class: type[ErrorT],
    status_code: int,
    headers: dict[str, str] | None = None,
    error_code: str | None = None,
) -> ErrorT:
    response = httpx2.Response(
        status_code,
        request=httpx2.Request("POST", "https://api.openai.com"),
        headers=headers,
    )
    body = (
        None
        if error_code is None
        else {"code": error_code, "type": "insufficient_quota", "message": "boom"}
    )
    return error_class("boom", response=response, body=body)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com"))


def transient(*, pauses_quota: bool, retry_after_seconds: float | None = None) -> RequestFailure:
    return RequestFailure(
        kind="transient", pauses_quota=pauses_quota, retry_after_seconds=retry_after_seconds
    )


def terminal(
    kind: Literal["auth", "rejected", "provider_failed_terminally", "unknown_exception"],
) -> RequestFailure:
    return RequestFailure(kind=kind, pauses_quota=False, retry_after_seconds=None)


def openai_sdk_errors_and_request_failures() -> Mapping[Exception, RequestFailure]:
    return {
        connection_error(): transient(pauses_quota=False),
        openai.APITimeoutError(httpx2.Request("POST", "https://api.openai.com")): transient(
            pauses_quota=False
        ),
        status_error(openai.RateLimitError, 429, {"retry-after": "7"}): transient(
            pauses_quota=True, retry_after_seconds=7.0
        ),
        status_error(
            openai.RateLimitError, 429, error_code="organization_spend_limit_exceeded"
        ): terminal("rejected"),
        status_error(openai.InternalServerError, 503): transient(pauses_quota=True),
        status_error(openai.InternalServerError, 500): transient(pauses_quota=False),
        status_error(openai.APIStatusError, 408): transient(pauses_quota=False),
        status_error(openai.ConflictError, 409): transient(pauses_quota=False),
        status_error(openai.InternalServerError, 599): transient(pauses_quota=False),
        status_error(openai.BadRequestError, 400): terminal("rejected"),
        status_error(openai.AuthenticationError, 401): terminal("auth"),
        status_error(openai.PermissionDeniedError, 403): terminal("auth"),
        status_error(openai.AuthenticationError, 401, {"x-should-retry": "false"}): terminal(
            "auth"
        ),
        status_error(openai.NotFoundError, 404): terminal("rejected"),
        status_error(openai.UnprocessableEntityError, 422): terminal("rejected"),
        status_error(openai.APIStatusError, 413): terminal("rejected"),
        status_error(openai.APIStatusError, 451): terminal("rejected"),
        status_error(openai.BadRequestError, 400, {"x-should-retry": "false"}): terminal(
            "rejected"
        ),
        status_error(openai.InternalServerError, 500, {"x-should-retry": "false"}): terminal(
            "provider_failed_terminally"
        ),
        status_error(openai.RateLimitError, 429, {"x-should-retry": "false"}): RequestFailure(
            kind="provider_failed_terminally", pauses_quota=True, retry_after_seconds=None
        ),
        status_error(openai.APIStatusError, 302): terminal("unknown_exception"),
        status_error(openai.APIStatusError, 200, error_code="invalid_prompt"): terminal(
            "provider_failed_terminally"
        ),
        ValueError("boom"): terminal("unknown_exception"),
    }


SEMCONV_GENAI_DIR = pathlib.Path(__file__).parent / "semconv_genai"
"""The vendored GenAI semantic-convention data that `scripts/refresh_semconv_genai.py` writes."""

_UNVALIDATED_PAYLOAD_ATTRIBUTES = frozenset({"gen_ai.tool.call.arguments"})
"""Skip schema validation for malformed tool-call argument text.

The schema accepts objects, while DispatchInvalidToolArgs preserves non-object text.
_validate_payload_attributes still validates this attribute when it contains an object.
"""


_GENERIC_FALLBACK_DEFINITIONS = frozenset({"GenericPart", "GenericToolDefinition"})
"""The vendored definitions that end a part or tool-definition union and accept any `type`."""


class _TypeProperty(BaseModel, extra="allow"):
    """A definition's `type` property, whose `const` names the one part or tool type the definition accepts."""

    const: str | None = None


class _DefinitionProperties(BaseModel, extra="allow"):
    type: _TypeProperty = _TypeProperty()


class _Definition(BaseModel, extra="allow"):
    properties: _DefinitionProperties = _DefinitionProperties()
    additional_properties: JsonValue = Field(default=None, alias="additionalProperties")
    not_: JsonValue = Field(default=None, alias="not")


class _VendoredSchema(BaseModel, extra="allow"):
    """A vendored payload schema.

    Every model keeps unknown keys, so dumping with `exclude_unset` reproduces the file.
    """

    defs: dict[str, _Definition] = Field(default_factory=dict, alias="$defs")


_LANGCHAINT_EXTENSION_PROPERTIES: Mapping[str, Mapping[str, JsonValue]] = {
    "tool_call_response": {"is_error": {"type": "boolean"}},
}
"""The schemas of the properties langchaint tracing adds to a convention part, keyed by its declared `type`."""


def _strict_definition(
    name: str, definition: _Definition, declared_types: list[str]
) -> _Definition:
    """Return `definition` restricted so that a part or tool definition matches only the definition its `type` declares.

    A generic fallback definition rejects every declared `type`.
    A definition that declares a `type` rejects properties that neither it nor `_LANGCHAINT_EXTENSION_PROPERTIES` lists.
    """
    if name in _GENERIC_FALLBACK_DEFINITIONS:
        return definition.model_copy(
            update={"not_": {"properties": {"type": {"enum": declared_types}}}}
        )
    declared_type = definition.properties.type.const
    if declared_type is None:
        return definition
    properties = definition.properties.model_copy(
        update=_LANGCHAINT_EXTENSION_PROPERTIES.get(declared_type, {})
    )
    return definition.model_copy(update={"properties": properties, "additional_properties": False})


@functools.cache
def _payload_schema(file: str) -> Mapping[str, object]:
    """Load and cache one vendored schema, restricted by `_strict_definition`.

    A declared `type` is the `const` of a definition's `type` property.
    The vendored schemas let a part with a declared `type` validate as a generic part and carry any property.
    With the restriction, a part that lacks a required field or misspells any field fails.
    Dumping with `exclude_unset` keeps every other key of the file.

    Raises:
        OSError: the vendored file could not be read.
        pydantic.ValidationError: the file is not JSON, or does not hold the `_VendoredSchema` shape.
    """
    schema = _VendoredSchema.model_validate_json((SEMCONV_GENAI_DIR / file).read_text())
    if not schema.defs:
        return schema.model_dump(mode="json", by_alias=True, exclude_unset=True)
    declared_types = [
        definition.properties.type.const
        for definition in schema.defs.values()
        if definition.properties.type.const is not None
    ]
    strict_defs = {
        name: _strict_definition(name, definition, declared_types)
        for name, definition in schema.defs.items()
    }
    strict_schema = schema.model_copy(update={"defs": strict_defs})
    return strict_schema.model_dump(mode="json", by_alias=True, exclude_unset=True)


def _validate_payload_attributes(span: ReadableSpan) -> None:
    """Validate one span's structured payload attributes.

    Each payload is a JSON string.
    _UNVALIDATED_PAYLOAD_ATTRIBUTES skips non-object tool arguments.
    Exact-equality assertions test fields that the schemas leave optional.

    Raises:
        AssertionError: A payload does not conform, is not a JSON string, or does not parse.
    """
    for key, value in (span.attributes or {}).items():
        file = refresh_semconv_genai.ATTRIBUTE_SCHEMA_FILES.get(key)
        if file is None:
            continue
        assert isinstance(value, str), f"{span.name}: {key} is not a JSON string"
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(f"{span.name}: {key} is not JSON: {error}") from error
        if key in _UNVALIDATED_PAYLOAD_ATTRIBUTES and not isinstance(payload, dict):
            continue
        try:
            jsonschema.Draft202012Validator(_payload_schema(file)).validate(payload)
        except jsonschema.ValidationError as error:
            raise AssertionError(
                f"{span.name}: {key} violates {file}. "
                f"Path {list(error.absolute_path)}: {error.message}"
            ) from error


class _DeclaredAttribute(BaseModel):
    name: str
    allowed_values: tuple[str, ...] | None


class _ChatSpanDeclaration(BaseModel):
    attributes: tuple[_DeclaredAttribute, ...]


LANGCHAINT_KEYS = frozenset({"langchaint.request_count", "langchaint.cost_in_usd"})
"""The keys langchaint writes on a chat span for values the convention has no attribute for."""


@functools.cache
def _chat_span_declaration() -> _ChatSpanDeclaration:
    return _ChatSpanDeclaration.model_validate_json(
        (SEMCONV_GENAI_DIR / refresh_semconv_genai.CHAT_SPAN_ATTRIBUTES_FILE).read_text()
    )


@functools.cache
def _declared_keys_by_operation() -> Mapping[str, frozenset[str]]:
    """Map each traced gen_ai.operation.name value to the keys its spans may carry, apart from application keys.

    Chat spans may carry the vendored chat span attribute names and the langchaint keys.
    Tool spans may carry the vendored execute_tool span attribute names.
    Embeddings spans may carry the vendored embeddings span attribute names and langchaint.cost_in_usd.

    Raises:
        pydantic.ValidationError: a vendored file does not hold its expected shape.
        AssertionError: a traced operation value is not a vendored gen_ai.operation.name value.
    """
    chat_names = frozenset(attribute.name for attribute in _chat_span_declaration().attributes)
    execute_tool_names = TypeAdapter(frozenset[str]).validate_json(
        (
            SEMCONV_GENAI_DIR / refresh_semconv_genai.EXECUTE_TOOL_SPAN_ATTRIBUTE_NAMES_FILE
        ).read_text()
    )
    embeddings_names = TypeAdapter(frozenset[str]).validate_json(
        (
            SEMCONV_GENAI_DIR / refresh_semconv_genai.EMBEDDINGS_SPAN_ATTRIBUTE_NAMES_FILE
        ).read_text()
    )
    declared = {
        "chat": chat_names | LANGCHAINT_KEYS,
        "execute_tool": execute_tool_names,
        "embeddings": embeddings_names | {"langchaint.cost_in_usd"},
    }
    (operation_attribute,) = (
        attribute
        for attribute in _chat_span_declaration().attributes
        if attribute.name == "gen_ai.operation.name"
    )
    assert operation_attribute.allowed_values is not None
    assert declared.keys() <= set(operation_attribute.allowed_values)
    return declared


def _validate_attribute_keys(span: ReadableSpan, application_keys: frozenset[str]) -> None:
    """Check one span's operation value and keys against the vendored convention.

    A span may also carry any of `application_keys`.
    A span without gen_ai.operation.name is one a test started itself, so it is not checked.
    OTel 1.45.0 allows the operation value to be a string, bool, int, float, bytes, sequence, or mapping.

    Raises:
        AssertionError: the operation value is not a string that langchaint traces.
        AssertionError: the span carries a key that its operation's span does not declare.
    """
    attributes = span.attributes or {}
    declared_keys_by_operation = _declared_keys_by_operation()
    match attributes.get("gen_ai.operation.name"):
        case None:
            return
        case str() as operation if operation in declared_keys_by_operation:
            declared_keys = declared_keys_by_operation[operation]
        case operation:
            raise AssertionError(f"{span.name}: langchaint does not trace operation {operation!r}")
    undeclared = set(attributes) - declared_keys - application_keys
    assert not undeclared, f"{span.name}: undeclared keys {sorted(undeclared)}"


class ValidatingSpanExporter(InMemorySpanExporter):
    """An in-memory exporter that validates attribute keys and payload attributes of the spans a test reads.

    An exception raised while a span ends never reaches the test, because `_GuardedOperation.end` logs it.
    """

    def __init__(self, *, application_keys: frozenset[str] = frozenset()) -> None:
        """Accept `application_keys` on every span, as the keys a test's transform or `extra_attributes` sets."""
        super().__init__()
        self._application_keys = application_keys

    @override
    def get_finished_spans(self) -> tuple[ReadableSpan, ...]:
        """Validate and return the finished spans.

        Raises:
            AssertionError: a span carries an undeclared key or a payload that does not conform to its schema.
        """
        spans = super().get_finished_spans()
        for span in spans:
            _validate_attribute_keys(span, self._application_keys)
            _validate_payload_attributes(span)
        return spans


def in_memory_tracer_provider(
    *, application_keys: frozenset[str] = frozenset()
) -> tuple[TracerProvider, InMemorySpanExporter]:
    """Build a tracer provider whose in-memory exporter validates the spans a test reads.

    The exporter accepts `application_keys` on every span.
    """
    exporter = ValidatingSpanExporter(application_keys=application_keys)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider, exporter


class RaisingOperation:
    """An `ObservedOperation` whose every method raises."""

    def current(self) -> AbstractContextManager[None]:
        raise RuntimeError("observer current boom")

    def conclude(self, outcome: object) -> None:
        raise RuntimeError(f"observer conclude boom for {type(outcome).__name__}")

    def end(self) -> None:
        raise RuntimeError("observer end boom")


class RaisingObserver:
    """An `Observer` that raises from its start methods, or returns operations whose every method raises."""

    def __init__(self, *, raise_at_start: bool) -> None:
        self._raise_at_start = raise_at_start

    def generation_started(
        self, start: GenerationStart
    ) -> ObservedOperation[GenerationOutcome[object] | AbandonedStreamRecord]:
        return self._started(start.model)

    def dispatch_started(
        self, tool_call: ToolCall
    ) -> ObservedOperation[DispatchOutcome | Exception]:
        return self._started(tool_call.name)

    def embedding_request_started(
        self, start: EmbeddingRequestStart
    ) -> ObservedOperation[EmbeddingRequestRecord]:
        return self._started(start.model)

    def _started(self, name: str) -> RaisingOperation:
        if self._raise_at_start:
            raise RuntimeError(f"observer start boom for {name}")
        return RaisingOperation()
