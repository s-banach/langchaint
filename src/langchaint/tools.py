"""Tools.

`PydanticTool` validates arguments with a `BaseModel`.
`JSONSchemaTool` validates JSON-object arguments with `jsonschema`.
`CaptureTool` returns validated arguments without calling a function.
Tool functions return model-facing content and optional application data.
The application owns the tool loop.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Protocol, TypeIs

import jsonschema.exceptions
import jsonschema.protocols
import jsonschema.validators
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint.exceptions import DispatchExceptionGroup, InvalidToolArgsError
from langchaint.messages import MessageContent, ToolCall, ToolMessage
from langchaint.sequence_not_str import SequenceNotStr


@dataclass(frozen=True, kw_only=True)
class ToolOutputExplicit[AppDataT = None]:
    """Model-facing `content`, its error status, and application-only `app_data`.

    `dispatch` copies `is_error` to `ToolMessage.is_error` and returns `app_data` unchanged.
    A bare `MessageContent` means `is_error=False` and `app_data=None`.
    """

    content: MessageContent
    is_error: bool = False
    app_data: AppDataT | None = None


type ToolOutput[AppDataT = None] = MessageContent | ToolOutputExplicit[AppDataT]
"""The model-facing content and optional application data returned by a tool function."""


@dataclass(frozen=True, kw_only=True)
class DispatchHandled[AppDataT = None]:
    """A completed tool call with its `tool_message` and application-only `app_data`.

    `tool_message.is_error` distinguishes success from a tool-reported failure.
    """

    tool_message: ToolMessage
    app_data: AppDataT | None = None
    kind: Literal["handled"] = "handled"


@dataclass(frozen=True, kw_only=True)
class InvalidToolArgsDetail:
    """One argument-validation failure.

    `path` contains object keys and list indexes.
    An empty `path` refers to the complete arguments object.
    `message` preserves the validator's text.
    """

    path: tuple[str | int, ...]
    message: str


@dataclass(frozen=True, kw_only=True)
class DispatchInvalidToolArgs:
    """A tool call rejected before its function ran.

    `tool_message` describes `details` for the model.
    """

    tool_message: ToolMessage
    details: tuple[InvalidToolArgsDetail, ...]
    kind: Literal["invalid_tool_args"] = "invalid_tool_args"


@dataclass(frozen=True, kw_only=True)
class DispatchUnknownTool:
    """A tool call whose `called_name` is absent from `ToolManager`.

    `tool_message` lists the available tool names for the model.
    """

    tool_message: ToolMessage
    called_name: str
    kind: Literal["unknown_tool"] = "unknown_tool"


@dataclass(frozen=True, kw_only=True)
class DispatchPrecomputed:
    """A `ToolMessage` supplied through `ToolManager.dispatch_many` without running a tool."""

    tool_message: ToolMessage
    kind: Literal["precomputed"] = "precomputed"


type DispatchOutcome = (
    DispatchHandled[BaseModel | Mapping[str, object] | None]
    | DispatchInvalidToolArgs
    | DispatchUnknownTool
)
"""The outcomes of `ToolManager.dispatch`.

Every variant carries `tool_message`.
Call a concrete tool's `dispatch` to preserve its `app_data` type.
"""


type DispatchManyOutcome = DispatchOutcome | DispatchPrecomputed
"""One ordered outcome from `ToolManager.dispatch_many`."""


@dataclass(frozen=True, kw_only=True)
class ToolSchema:
    """The provider-neutral description of one tool."""

    name: str
    description: str
    args_schema: Mapping[str, object]


@dataclass(frozen=True, kw_only=True)
class PydanticTool[ArgsT: BaseModel, AppDataT = None]:
    """A callable tool validated by `args_model`."""

    name: str
    description: str
    args_model: type[ArgsT]
    function: Callable[[ArgsT], Awaitable[ToolOutput[AppDataT]]]

    def schema(self) -> ToolSchema:
        """Convert to the provider-neutral schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            args_schema=self.args_model.model_json_schema(),
        )

    def _validated_args(self, args_json: str) -> ArgsT:
        """Validate `args_json` against `args_model`.

        Raises:
            InvalidToolArgsError: `args_json` fails validation.
        """
        try:
            return self.args_model.model_validate_json(args_json)
        except ValidationError as exc:
            raise InvalidToolArgsError(exc) from exc

    async def validate_and_run(self, args_json: str) -> ToolOutput[AppDataT]:
        """Validate `args_json` with `args_model`, then run `function`.

        Args:
            args_json: The model-generated argument JSON.

        Raises:
            InvalidToolArgsError: `args_json` fails validation.
            BaseException: `function` raises it.
        """
        return await self.function(self._validated_args(args_json))

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Return a handled call or its argument-validation failure.

        The caller must match `call.name` before calling this method.

        Args:
            call: The tool call to dispatch.

        Raises:
            BaseException: `function` raises it.
        """
        try:
            args = self._validated_args(call.args_json)
        except InvalidToolArgsError as error:
            return _invalid_args_outcome(call, _details_from_pydantic(error.validation_error))
        return _handled_outcome(call, await self.function(args))


def _is_matching_args_model[ArgsT: BaseModel, AppDataT](
    annotation: object,
    _function: Callable[[ArgsT], Awaitable[ToolOutput[AppDataT]]],
) -> TypeIs[type[ArgsT]]:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


@dataclass(frozen=True, kw_only=True)
class _ToolDecorator:
    description: str
    name: str | None

    def __call__[ArgsT: BaseModel, AppDataT = None](
        self,
        function: Callable[[ArgsT], Awaitable[ToolOutput[AppDataT]]],
    ) -> PydanticTool[ArgsT, AppDataT]:
        """Resolve `args_model` from the parameter.

        Raises:
            TypeError: The decorated value is not a function.
            TypeError: The parameter annotation does not resolve to a `BaseModel` subclass.
        """
        if not inspect.isfunction(function):
            raise TypeError("@tool requires a function")
        parameter = next(iter(inspect.signature(function).parameters.values()))
        args_model: object = parameter.annotation
        if isinstance(args_model, str):
            decorator_frame = inspect.currentframe()
            if decorator_frame is None or decorator_frame.f_back is None:
                local_namespace = function.__globals__
            else:
                local_namespace = decorator_frame.f_back.f_locals
            del decorator_frame
            try:
                args_model = eval(args_model, function.__globals__, local_namespace)
            except Exception as error:
                raise TypeError(
                    f"@tool function {function.__name__!r} parameter {parameter.name!r} "
                    "annotation could not resolve"
                ) from error
        if not _is_matching_args_model(args_model, function):
            raise TypeError(
                f"@tool function {function.__name__!r} parameter {parameter.name!r} "
                "must be annotated with a BaseModel subclass"
            )
        return PydanticTool(
            name=function.__name__ if self.name is None else self.name,
            description=self.description,
            args_model=args_model,
            function=function,
        )


def tool(*, description: str, name: str | None = None) -> _ToolDecorator:
    """Decorate one async function as a PydanticTool.

    Args:
        description: Provider-facing purpose.
        name: ToolSchema.name. None uses function.__name__.
    """
    return _ToolDecorator(description=description, name=name)


@dataclass(frozen=True, kw_only=True)
class DispatchCaptured[CapturedT: BaseModel]:
    """A validated `captured` instance and its acknowledgement `tool_message`."""

    tool_message: ToolMessage
    captured: CapturedT
    kind: Literal["captured"] = "captured"


@dataclass(frozen=True, kw_only=True)
class CaptureTool[CapturedT: BaseModel]:
    """Validate model-generated arguments. Return them without calling a function.

    `acknowledgement` becomes the model-facing tool result.
    `capture` preserves `CapturedT`.
    """

    name: str
    description: str
    args_model: type[CapturedT]
    acknowledgement: str = "Acknowledged"

    def schema(self) -> ToolSchema:
        """Convert to the provider-neutral schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            args_schema=self.args_model.model_json_schema(),
        )

    async def capture(
        self, call: ToolCall
    ) -> DispatchCaptured[CapturedT] | DispatchInvalidToolArgs:
        """Validate `call.args_json` and return its capture or failure.

        The caller must match `call.name` before calling this method.

        Args:
            call: The tool call to validate.
        """
        try:
            captured = self.args_model.model_validate_json(call.args_json)
        except ValidationError as error:
            return _invalid_args_outcome(call, _details_from_pydantic(error))
        return DispatchCaptured(
            tool_message=ToolMessage(tool_call_id=call.id, content=self.acknowledgement),
            captured=captured,
        )

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[CapturedT] | DispatchInvalidToolArgs:
        """Return `capture` as `DispatchHandled.app_data` after validation.

        Args:
            call: The tool call to dispatch.
        """
        outcome = await self.capture(call)
        match outcome.kind:
            case "captured":
                return DispatchHandled(
                    tool_message=outcome.tool_message, app_data=outcome.captured
                )
            case "invalid_tool_args":
                return outcome


_ARGS_OBJECT = TypeAdapter(dict[str, object])


@dataclass(frozen=True, kw_only=True)
class JSONSchemaTool[AppDataT = None]:
    """A function whose raw JSON schema validates its model-generated arguments.

    `args_schema` passes to the provider unchanged.
    """

    name: str
    description: str
    args_schema: Mapping[str, object]
    function: Callable[[dict[str, object]], Awaitable[ToolOutput[AppDataT]]]

    @cached_property
    def _validator(self) -> jsonschema.protocols.Validator:
        return jsonschema.validators.validator_for(self.args_schema)(self.args_schema)

    def schema(self) -> ToolSchema:
        """Convert to the provider-neutral schema."""
        return ToolSchema(
            name=self.name, description=self.description, args_schema=self.args_schema
        )

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Validate `call.args_json`, then run `function`.

        Validation failures skip `function`.

        Args:
            call: The tool call to dispatch.

        Raises:
            BaseException: `function` raises it.
        """
        try:
            args = _ARGS_OBJECT.validate_json(call.args_json)
        except ValidationError as error:
            return _invalid_args_outcome(call, _details_from_pydantic(error))
        # Parsing preserves the JSON types required by `iter_errors`.
        # `object` cannot prove those types to the checker.
        # pyrefly: ignore[bad-argument-type]
        details = _details_from_jsonschema(self._validator.iter_errors(args))
        if details:
            return _invalid_args_outcome(call, details)
        result = await self.function(args)
        return _handled_outcome(call, result)


class Tool[AppDataT](Protocol):
    """The `name`, `schema`, and `dispatch` interface required by `ToolManager`."""

    @property
    def name(self) -> str:
        """Return the tool's dispatch name matched against `ToolCall.name`."""
        ...

    def schema(self) -> ToolSchema:
        """Return the provider-neutral schema of this tool."""
        ...

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Run this tool on `call` and wrap the outcome.

        Args:
            call: The tool call to dispatch.

        Raises:
            BaseException: The tool implementation raises it.
        """
        ...


def render_invalid_tool_args(tool_name: str, details: Sequence[InvalidToolArgsDetail]) -> str:
    """Build the model-facing content for an argument-validation failure.

    Args:
        tool_name: The tool name to include in the content.
        details: The argument-validation failures.

    Raises:
        ValueError: `details` is empty.
    """
    if not details:
        raise ValueError(f"render_invalid_tool_args for {tool_name} received no details to render")
    lines = [f"invalid arguments for {tool_name}:"]
    for detail in details:
        joined_path = (
            ".".join(str(segment) for segment in detail.path) if detail.path else "(root)"
        )
        lines.append(f"  {joined_path}: {detail.message}")
    return "\n".join(lines)


def render_unknown_tool(called_name: str, held_names: SequenceNotStr[str]) -> str:
    """Name an unknown tool and list the available tool names.

    Args:
        called_name: The unknown tool name.
        held_names: The available tool names.
    """
    held = ", ".join(held_names) if held_names else "(none)"
    return f"unknown tool {called_name!r}; available tools: {held}"


def _details_from_pydantic(validation_error: ValidationError) -> tuple[InvalidToolArgsDetail, ...]:
    return tuple(
        InvalidToolArgsDetail(path=tuple(error["loc"]), message=error["msg"])
        for error in validation_error.errors(
            include_url=False, include_context=False, include_input=False
        )
    )


def _details_from_jsonschema(
    errors: Iterable[jsonschema.exceptions.ValidationError],
) -> tuple[InvalidToolArgsDetail, ...]:
    return tuple(
        InvalidToolArgsDetail(path=tuple(error.absolute_path), message=error.message)
        for error in errors
    )


def _invalid_args_outcome(
    call: ToolCall, details: tuple[InvalidToolArgsDetail, ...]
) -> DispatchInvalidToolArgs:
    tool_message = ToolMessage.error(
        call, render_invalid_tool_args(tool_name=call.name, details=details)
    )
    return DispatchInvalidToolArgs(tool_message=tool_message, details=details)


def _split_precomputed(
    tool_calls: Sequence[ToolCall],
    precomputed: Callable[[ToolCall], ToolMessage | None] | None,
) -> tuple[dict[int, DispatchPrecomputed], list[tuple[int, ToolCall]]]:
    """Apply `precomputed` before dispatch and partition the calls by input position.

    Raises:
        ValueError: `precomputed` returns a `ToolMessage` for a different `tool_call_id`.
    """
    answered: dict[int, DispatchPrecomputed] = {}
    to_dispatch: list[tuple[int, ToolCall]] = []
    for index, tool_call in enumerate(tool_calls):
        supplied = precomputed(tool_call) if precomputed is not None else None
        if supplied is None:
            to_dispatch.append((index, tool_call))
        elif supplied.tool_call_id != tool_call.id:
            raise ValueError(
                f"precomputed answered the call with id {tool_call.id!r} with a ToolMessage "
                f"whose tool_call_id is {supplied.tool_call_id!r}"
            )
        else:
            answered[index] = DispatchPrecomputed(tool_message=supplied)
    return answered, to_dispatch


def _handled_outcome[AppDataT](
    call: ToolCall, result: ToolOutput[AppDataT]
) -> DispatchHandled[AppDataT]:
    if isinstance(result, ToolOutputExplicit):
        tool_message = ToolMessage(
            tool_call_id=call.id, content=result.content, is_error=result.is_error
        )
        return DispatchHandled[AppDataT](tool_message=tool_message, app_data=result.app_data)
    tool_message = ToolMessage(tool_call_id=call.id, content=result)
    return DispatchHandled[AppDataT](tool_message=tool_message)


class ToolManager:
    """Index tools by name and route calls to them."""

    def __init__(self, tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]]) -> None:
        """Index the tools by name.

        Args:
            tools: The tools to index.

        Raises:
            ValueError: Two tools share a name.
        """
        self._tools: dict[str, Tool[BaseModel | Mapping[str, object] | None]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> tuple[ToolSchema, ...]:
        """Convert every indexed tool to its provider-neutral schema."""
        return tuple(tool.schema() for tool in self._tools.values())

    async def dispatch(self, call: ToolCall) -> DispatchOutcome:
        """Dispatch `call` or return `DispatchUnknownTool`.

        Args:
            call: The tool call to dispatch.

        Raises:
            BaseException: The matched tool raises it.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            tool_message = ToolMessage.error(
                call, render_unknown_tool(called_name=call.name, held_names=tuple(self._tools))
            )
            return DispatchUnknownTool(tool_message=tool_message, called_name=call.name)
        return await tool.dispatch(call)

    async def dispatch_many(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        precomputed: Callable[[ToolCall], ToolMessage | None] | None = None,
    ) -> tuple[DispatchManyOutcome, ...]:
        """Dispatch calls concurrently. Preserve `tool_calls` order.

        `precomputed` runs for every call before any tool function starts.
        A returned `ToolMessage` skips that call's dispatch.

        Args:
            tool_calls: The tool calls to dispatch.
            precomputed: The optional function that supplies a result without dispatch.

        Raises:
            ValueError: `precomputed` returns a `ToolMessage` for another `tool_call_id`.
            DispatchExceptionGroup: Tool functions raise `Exception` values after sibling calls settle.
            asyncio.CancelledError: The caller cancels this function after sibling calls settle.
            BaseException: `precomputed` raises it before any tool function starts.
            BaseException: A tool function raises a non-`Exception` value after every call settles.
        """
        answered, to_dispatch = _split_precomputed(tool_calls, precomputed)
        settled: dict[int, DispatchManyOutcome | BaseException] = dict(answered)
        tasks = [asyncio.ensure_future(self.dispatch(tool_call)) for _, tool_call in to_dispatch]
        try:
            results: list[DispatchOutcome | BaseException] = await asyncio.gather(
                *tasks, return_exceptions=True
            )
        except asyncio.CancelledError:
            # `gather` cancels sibling tasks without settling them.
            for task in tasks:
                _ = task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for (index, _), result in zip(to_dispatch, results, strict=True):
            settled[index] = result
        completed_outcomes: list[DispatchManyOutcome] = []
        raised_exceptions: list[Exception] = []
        base_exceptions: list[BaseException] = []
        for index in range(len(tool_calls)):
            result = settled[index]
            if isinstance(result, Exception):
                raised_exceptions.append(result)
            elif isinstance(result, BaseException):
                base_exceptions.append(result)
            else:
                completed_outcomes.append(result)
        if raised_exceptions:
            group = DispatchExceptionGroup(
                f"{len(raised_exceptions)} of {len(tool_calls)} tool calls raised during dispatch_many",
                raised_exceptions,
                completed_outcomes=tuple(completed_outcomes),
            )
            if base_exceptions:
                # Preserve concurrent function exceptions as the cause.
                raise base_exceptions[0] from group
            raise group
        if base_exceptions:
            raise base_exceptions[0]
        return tuple(completed_outcomes)
