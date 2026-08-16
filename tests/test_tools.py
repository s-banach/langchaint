"""Test tool validation, dispatch, app_data, and function errors."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, assert_type

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import UnknownType
from pydantic import BaseModel, Field, ValidationError

from langchaint import (
    CaptureTool,
    ContentPart,
    DispatchCaptured,
    DispatchExceptionGroup,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    ImagePart,
    InvalidToolArgsDetail,
    InvalidToolArgsError,
    JSONSchemaTool,
    PydanticTool,
    TextPart,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    tool,
)
from langchaint.tools import _details_from_pydantic, render_invalid_tool_args, render_unknown_tool

if TYPE_CHECKING:
    from jsonschema.protocols import Validator

_WEATHER_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


class _EchoArgs(BaseModel):
    """Arguments of the echo tool."""

    text: str


async def _echo_function(args: _EchoArgs) -> str:
    """Return the validated text unchanged."""
    return args.text


@tool(description="Echo the text without reading this function's docstring.")
async def _decorated_echo(args: "_EchoArgs") -> str:
    """Describe implementation behavior that the provider must not receive."""
    return args.text


class _EchoRecord(BaseModel):
    """Application data produced beside an echo result."""

    text: str


@tool(description="Echo the text and build a later-defined record.")
async def _decorated_later_return(
    args: _EchoArgs,
) -> "ToolOutputExplicit[_LaterEchoRecord]":
    """Return application data whose class follows this function."""
    return ToolOutputExplicit(content=args.text, app_data=_LaterEchoRecord(text=args.text))


class _LaterEchoRecord(BaseModel):
    """Application data defined after its producing function."""

    text: str


@tool(description="Echo the text and record it.", name="record_echo")
async def _decorated_record_echo(args: _EchoArgs) -> ToolOutputExplicit[_EchoRecord]:
    """Return model content and application data."""
    return ToolOutputExplicit(content=args.text, app_data=_EchoRecord(text=args.text))


async def _validation_error_function(args: _EchoArgs) -> str:
    """Fail with a function-internal ValidationError, a user-code defect.

    The model_validate call always raises because the payload lacks the required text field.
    """
    _ = _EchoArgs.model_validate({"wrong": args.text})
    return "unreachable"


def _echo_tool() -> PydanticTool[_EchoArgs]:
    return PydanticTool(
        name="echo",
        description="Echo the text back.",
        args_model=_EchoArgs,
        function=_echo_function,
    )


def test_schema_converts_name_description_and_args_schema() -> None:
    """PydanticTool.schema carries the name, description, and the args JSON schema."""
    schema = _echo_tool().schema()
    assert schema.name == "echo"
    assert schema.description == "Echo the text back."
    assert schema.args_schema == _EchoArgs.model_json_schema()
    assert ToolManager([_echo_tool()]).schemas() == (schema,)


def test_tool_decorator_infers_metadata_and_preserves_types() -> None:
    """Verify tool reads the parameter annotation, function name, and explicit description."""
    assert_type(_decorated_echo, PydanticTool[_EchoArgs, None])
    assert_type(_decorated_record_echo, PydanticTool[_EchoArgs, _EchoRecord])
    assert_type(_decorated_later_return, PydanticTool[_EchoArgs, _LaterEchoRecord])
    assert _decorated_echo.name == "_decorated_echo"
    assert (
        _decorated_echo.description == "Echo the text without reading this function's docstring."
    )
    assert _decorated_echo.args_model is _EchoArgs
    assert _decorated_echo.schema().args_schema == _EchoArgs.model_json_schema()
    assert _decorated_record_echo.name == "record_echo"
    outcome = asyncio.run(
        _decorated_record_echo.dispatch(
            ToolCall(id="call_decorated", name="record_echo", args_json='{"text":"tide"}')
        )
    )
    assert isinstance(outcome, DispatchHandled)
    assert outcome.app_data == _EchoRecord(text="tide")


def test_tool_decorator_resolves_quoted_local_args_model() -> None:
    """Verify tool resolves its parameter annotation from the decorator's local namespace."""

    class _LocalArgs(BaseModel):
        """Arguments defined in the decorator's local namespace."""

        text: str

    @tool(description="Echo locally defined arguments.")
    async def _local_echo(args: "_LocalArgs") -> str:
        return args.text

    assert_type(_local_echo, PydanticTool[_LocalArgs, None])
    assert _local_echo.args_model is _LocalArgs


def test_tool_decorator_rejects_unusable_parameter_annotations() -> None:
    """Verify tool rejects an annotation that does not resolve to a BaseModel subclass at runtime."""

    async def _plain_annotation(args: str) -> str:
        return args

    # pyrefly: ignore[implicit-any-parameter]
    async def _missing_annotation(args) -> str:  # noqa: ANN001
        return str(args)  # pyrefly: ignore[unknown-argument-type]

    async def _type_checking_only_annotation(args: "Validator") -> str:
        return str(args)

    decorator = tool(description="Invalid test function.")
    with pytest.raises(TypeError, match="BaseModel subclass"):
        _ = decorator(_plain_annotation)
    with pytest.raises(TypeError, match="BaseModel subclass"):
        _ = decorator(_missing_annotation)
    with pytest.raises(TypeError, match="annotation could not resolve"):
        _ = decorator(_type_checking_only_annotation)


def test_tool_decorator_rejects_a_callable_instance() -> None:
    """@tool requires a Python function before reading its signature."""

    class CallableTool:
        async def __call__(self, args: _EchoArgs) -> str:
            return args.text

    decorator = tool(description="Invalid callable instance.")
    with pytest.raises(TypeError, match="@tool requires a function"):
        _ = decorator(CallableTool())


def test_validate_and_run_returns_the_function_result() -> None:
    """Valid args_json reaches the function as the validated model."""
    result = asyncio.run(_echo_tool().validate_and_run('{"text": "tide"}'))
    assert result == "tide"


def test_invalid_tool_args_holds_the_validation_error() -> None:
    """InvalidToolArgsError preserves ValidationError and readable text."""
    with pytest.raises(InvalidToolArgsError) as caught:
        _ = asyncio.run(_echo_tool().validate_and_run('{"wrong": "key"}'))
    error = caught.value
    assert isinstance(error.validation_error, ValidationError)
    assert any("text" in entry["loc"] for entry in error.validation_error.errors())
    assert "text" in str(error)


def test_details_from_pydantic_and_renderer_format_per_field() -> None:
    """Pydantic errors render neutral paths and messages."""

    class _Recipient(BaseModel):
        """One recipient with a required email."""

        email: str

    class _SendArgs(BaseModel):
        """Send arguments with a recipient list and a non-empty subject."""

        to: list[_Recipient]
        subject: str = Field(min_length=1)

    with pytest.raises(ValidationError) as caught:
        _ = _SendArgs.model_validate_json('{"to":[{"x":1},5],"subject":""}')
    validation_error = caught.value
    details = _details_from_pydantic(validation_error)
    assert details == tuple(
        InvalidToolArgsDetail(path=tuple(entry["loc"]), message=entry["msg"])
        for entry in validation_error.errors()
    )
    rendered = render_invalid_tool_args("send_email", details)
    expected = "\n".join(
        ["invalid arguments for send_email:"]
        + [
            f"  {'.'.join(str(segment) for segment in detail.path)}: {detail.message}"
            for detail in details
        ]
    )
    assert rendered == expected
    assert "\n  to.0.email: " in rendered
    assert "\n  to.1: " in rendered
    assert "https://" not in rendered
    assert "errors.pydantic.dev" not in rendered
    assert "type=" not in rendered


def test_render_invalid_tool_args_formats_neutral_details() -> None:
    """render_invalid_tool_args preserves detail order and paths."""
    details = [
        InvalidToolArgsDetail(path=(), message="'name' is a required property"),
        InvalidToolArgsDetail(path=("items", 0, "id"), message="'x' is not of type 'integer'"),
        InvalidToolArgsDetail(path=("mode",), message="'fast' is not one of ['safe', 'slow']"),
    ]
    assert render_invalid_tool_args("search", details) == (
        "invalid arguments for search:\n"
        "  (root): 'name' is a required property\n"
        "  items.0.id: 'x' is not of type 'integer'\n"
        "  mode: 'fast' is not one of ['safe', 'slow']"
    )


def test_render_invalid_tool_args_rejects_empty_details() -> None:
    """Empty details raise ValueError: claiming invalid arguments with no listed failure would mislead the model."""
    with pytest.raises(ValueError, match="no details to render"):
        _ = render_invalid_tool_args("search", [])


def test_dispatch_wraps_success_in_a_tool_message() -> None:
    """A valid call comes back as a DispatchHandled non-error ToolMessage with the call id."""
    call = ToolCall(id="call1", name="echo", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.tool_call_id == "call1"
    assert result.tool_message.content == "tide"
    assert result.tool_message.is_error is False
    assert result.app_data is None


def test_dispatch_carries_content_parts_into_tool_message_content() -> None:
    """Sequence[ContentPart] reaches ToolMessage.content as a tuple.

    A success path that dropped or stringified the result would fail this equality.
    """

    async def _content_parts_function(args: _EchoArgs) -> Sequence[ContentPart]:
        """Return ContentPart values built from validated text."""
        return [TextPart(text=args.text), ImagePart(data=b"png", media_type="image/png")]

    tool = PydanticTool(
        name="render",
        description="Return model content.",
        args_model=_EchoArgs,
        function=_content_parts_function,
    )
    call = ToolCall(id="call1", name="render", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([tool]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == (
        TextPart(text="tide"),
        ImagePart(data=b"png", media_type="image/png"),
    )
    assert isinstance(result.tool_message.content, tuple)
    assert result.tool_message.is_error is False
    assert result.app_data is None


def test_dispatch_delegates_invalid_args_content_to_the_renderer() -> None:
    """PydanticTool renders and stores converted validation details."""
    args_json = '{"wrong": "key"}'
    with pytest.raises(InvalidToolArgsError) as caught:
        _ = asyncio.run(_echo_tool().validate_and_run(args_json))
    expected_details = _details_from_pydantic(caught.value.validation_error)
    expected_content = render_invalid_tool_args("echo", expected_details)
    call = ToolCall(id="call1", name="echo", args_json=args_json)
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    match result:
        case DispatchInvalidToolArgs():
            assert result.tool_message.is_error is True
            assert result.tool_message.content == expected_content
            assert result.details == expected_details
        case DispatchHandled() | DispatchUnknownTool():
            pytest.fail("invalid args must return DispatchInvalidToolArgs")


def test_dispatch_returns_unknown_tool_variant_for_off_list_name() -> None:
    """An unknown tool name returns DispatchUnknownTool."""
    call = ToolCall(id="call1", name="missing", args_json="{}")
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchUnknownTool)
    assert result.called_name == "missing"
    assert result.tool_message.tool_call_id == "call1"
    assert result.tool_message.is_error is True
    assert result.tool_message.content == render_unknown_tool(
        called_name="missing", held_names=("echo",)
    )


def test_render_unknown_tool_lists_held_names_and_none_when_empty() -> None:
    """render_unknown_tool names the off-list tool and held_names, rendering (none) for an empty set."""
    assert render_unknown_tool(called_name="x", held_names=("a", "b")) == (
        "unknown tool 'x'; available tools: a, b"
    )
    assert render_unknown_tool(called_name="x", held_names=()) == (
        "unknown tool 'x'; available tools: (none)"
    )


def test_function_validation_error_propagates_as_a_defect() -> None:
    """A ValidationError raised inside the function is not treated as bad args."""
    tool = PydanticTool(
        name="broken",
        description="Raises from its own pydantic use.",
        args_model=_EchoArgs,
        function=_validation_error_function,
    )
    call = ToolCall(id="call1", name="broken", args_json='{"text": "tide"}')
    with pytest.raises(ValidationError):
        _ = asyncio.run(ToolManager([tool]).dispatch(call))


def test_dispatch_carries_a_returned_is_error_result() -> None:
    """A function returning ToolOutputExplicit(is_error=True) becomes an is_error ToolMessage."""

    async def _returned_error_function(args: _EchoArgs) -> ToolOutputExplicit:
        """Report a model-visible failure by returning, not raising."""
        return ToolOutputExplicit(
            content=f"cannot echo {args.text!r}: try a shorter value", is_error=True
        )

    tool = PydanticTool(
        name="picky",
        description="Returns an is_error result.",
        args_model=_EchoArgs,
        function=_returned_error_function,
    )
    call = ToolCall(id="call1", name="picky", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([tool]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.is_error is True
    assert result.tool_message.content == "cannot echo 'tide': try a shorter value"
    assert result.tool_message.tool_call_id == "call1"
    assert result.app_data is None


class _Cites(BaseModel):
    """A pydantic app_data payload naming its own field."""

    citations: list[str]


class _Receipt(BaseModel):
    """A pydantic app_data payload for a record the function persisted before failing."""

    record_id: str


def test_dispatch_preserves_mapping_app_data_identity() -> None:
    """Mapping app_data passes through ToolManager unchanged."""
    mapping = {"citations": ["doc-1"]}

    async def _mapping_function(args: _EchoArgs) -> ToolOutputExplicit[Mapping[str, object]]:
        """Return content plus mapping app_data."""
        return ToolOutputExplicit(content=f"declined {args.text}", is_error=True, app_data=mapping)

    mapping_tool = PydanticTool(
        name="mapping",
        description="Returns a mapping app_data.",
        args_model=_EchoArgs,
        function=_mapping_function,
    )
    mapping_result = asyncio.run(
        ToolManager([mapping_tool]).dispatch(
            ToolCall(id="call2", name="mapping", args_json='{"text": "tide"}')
        )
    )
    assert isinstance(mapping_result, DispatchHandled)
    assert mapping_result.tool_message.is_error is True
    assert mapping_result.app_data is mapping


async def _pin_tool_dispatch_app_data_type(
    tool: PydanticTool[_EchoArgs, _Cites], call: ToolCall
) -> None:
    result = await tool.dispatch(call)
    assert isinstance(result, DispatchHandled)
    assert_type(result.app_data, _Cites | None)


def test_plain_function_exception_propagates_as_a_defect() -> None:
    """Any non-validation function exception propagates unchanged."""

    async def _failing_function(args: _EchoArgs) -> str:
        """Fail with an ordinary user-code exception.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError(f"function broke on {args.text}")

    tool = PydanticTool(
        name="failing",
        description="Raises an ordinary exception.",
        args_model=_EchoArgs,
        function=_failing_function,
    )
    call = ToolCall(id="call1", name="failing", args_json='{"text": "tide"}')
    with pytest.raises(RuntimeError, match="function broke on tide"):
        _ = asyncio.run(ToolManager([tool]).dispatch(call))


def test_a_function_raised_invalid_tool_args_error_propagates_as_a_defect() -> None:
    """A function-raised InvalidToolArgsError propagates as a defect."""

    async def _nested_validation_function(args: _EchoArgs) -> str:
        """Validate a payload of the function's own and fail on it.

        Raises:
            InvalidToolArgsError: always; the nested payload lacks the required text field.
        """
        try:
            _ = _EchoArgs.model_validate({"wrong": args.text})
        except ValidationError as exc:
            raise InvalidToolArgsError(exc) from exc
        return "unreachable"

    tool = PydanticTool(
        name="nested",
        description="Validates a payload of its own.",
        args_model=_EchoArgs,
        function=_nested_validation_function,
    )
    call = ToolCall(id="call1", name="nested", args_json='{"text": "tide"}')
    with pytest.raises(InvalidToolArgsError):
        _ = asyncio.run(tool.dispatch(call))
    with pytest.raises(InvalidToolArgsError):
        _ = asyncio.run(ToolManager([tool]).dispatch(call))


def test_duplicate_tool_names_are_rejected() -> None:
    """Two tools sharing a name raise ValueError at construction."""
    with pytest.raises(ValueError, match="duplicate tool name"):
        _ = ToolManager([_echo_tool(), _echo_tool()])


async def _weather_function(args: dict[str, object]) -> str:
    """Return weather from parsed JSONSchemaTool arguments."""
    return f"sunny in {args['city']}"


def _weather_tool() -> JSONSchemaTool:
    """Build a JSONSchemaTool from a raw JSON schema, the shape an MCP tool arrives in."""
    return JSONSchemaTool(
        name="weather",
        description="Report the weather.",
        args_schema=_WEATHER_SCHEMA,
        function=_weather_function,
    )


def test_schema_tool_schema_passes_the_raw_json_schema_through_unchanged() -> None:
    """JSONSchemaTool.schema preserves args_schema by identity."""
    schema = _weather_tool().schema()
    assert schema.name == "weather"
    assert schema.description == "Report the weather."
    assert schema.args_schema is _WEATHER_SCHEMA


def test_schema_tool_dispatch_returns_invalid_args_for_schema_violations() -> None:
    """Schema violations return converted jsonschema errors."""
    call = ToolCall(id="call1", name="weather", args_json='{"town": "Oslo"}')
    result = asyncio.run(_weather_tool().dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True
    # The Validator protocol types iter_errors as yielding ValidationError, where the concrete
    # Draft202012Validator's own stub yields Incomplete.
    validator: Validator = Draft202012Validator(_WEATHER_SCHEMA)
    expected_details = tuple(
        InvalidToolArgsDetail(path=tuple(error.absolute_path), message=error.message)
        for error in validator.iter_errors({"town": "Oslo"})
    )
    assert result.details == expected_details
    assert any(detail.message == "'city' is a required property" for detail in result.details)
    assert result.tool_message.content == render_invalid_tool_args("weather", expected_details)


def _recording_weather_tool(calls: list[str]) -> JSONSchemaTool:
    """Build the weather JSONSchemaTool with a function recording each run in calls."""

    async def _recording_function(args: dict[str, object]) -> str:
        """Record the run, then report the weather."""
        calls.append("function")
        return f"sunny in {args['city']}"

    return JSONSchemaTool(
        name="weather",
        description="Report the weather.",
        args_schema=_WEATHER_SCHEMA,
        function=_recording_function,
    )


def test_schema_tool_valid_args_run_the_function() -> None:
    """Arguments satisfying args_schema reach the function: validation passed, a DispatchHandled."""
    calls: list[str] = []
    tool = _recording_weather_tool(calls)
    result = asyncio.run(
        tool.dispatch(ToolCall(id="c1", name="weather", args_json='{"city": "Oslo"}'))
    )
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "sunny in Oslo"
    assert result.tool_message.is_error is False
    assert result.app_data is None
    assert calls == ["function"]


@pytest.mark.parametrize(
    "args_json",
    ['{"town": "Oslo"}', "5", '{"city": '],
    ids=["schema_violation", "non_object_json", "malformed_json"],
)
def test_schema_tool_rejects_bad_args_locally_without_running_the_function(args_json: str) -> None:
    """Bad JSONSchemaTool arguments return DispatchInvalidToolArgs without execution."""
    calls: list[str] = []
    tool = _recording_weather_tool(calls)
    result = asyncio.run(tool.dispatch(ToolCall(id="c1", name="weather", args_json=args_json)))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.tool_call_id == "c1"
    assert result.tool_message.is_error is True
    assert "invalid arguments for weather" in result.tool_message.content
    assert len(result.details) >= 1
    assert calls == []


def test_schema_tool_malformed_schema_raises_from_dispatch_as_a_defect() -> None:
    """An invalid args_schema raises jsonschema's exception."""
    tool = JSONSchemaTool(
        name="bad",
        description="Malformed schema.",
        args_schema={"type": "strng"},
        function=_weather_function,
    )
    with pytest.raises(UnknownType):
        _ = asyncio.run(tool.dispatch(ToolCall(id="c1", name="bad", args_json='{"city": "Oslo"}')))


def test_schema_tool_dispatch_carries_a_mapping_app_data_through() -> None:
    """JSONSchemaTool.dispatch preserves ToolOutputExplicit app_data type."""
    raw_result = {"forecast": ["sunny"], "source": "mcp"}

    async def _mcp_function(
        args: Mapping[str, object],
    ) -> ToolOutputExplicit[Mapping[str, object]]:
        """Return model-visible content plus the raw MCP result the model never sees.

        Annotated Mapping[str, object]: accepted against the dict[str, object] parameter by contravariance,
        pinning that the wider annotation keeps typechecking.
        """
        return ToolOutputExplicit(content=f"weather for {args['city']}", app_data=raw_result)

    tool: JSONSchemaTool[Mapping[str, object]] = JSONSchemaTool(
        name="weather",
        description="Report the weather via MCP.",
        args_schema=_WEATHER_SCHEMA,
        function=_mcp_function,
    )
    call = ToolCall(id="call1", name="weather", args_json='{"city": "Oslo"}')
    result = asyncio.run(tool.dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "weather for Oslo"
    raw: Mapping[str, object] | None = result.app_data
    assert raw is raw_result


def test_tool_manager_holds_a_mix_of_tool_and_schema_tool() -> None:
    """One ToolManager routes to a PydanticTool and a JSONSchemaTool side by side.

    ToolManager emits and dispatches both tool forms uniformly.
    """
    manager = ToolManager([_echo_tool(), _weather_tool()])
    names = {schema.name for schema in manager.schemas()}
    assert names == {"echo", "weather"}

    echo_result = asyncio.run(
        manager.dispatch(ToolCall(id="c1", name="echo", args_json='{"text": "hi"}'))
    )
    weather_result = asyncio.run(
        manager.dispatch(ToolCall(id="c2", name="weather", args_json='{"city": "Oslo"}'))
    )
    assert isinstance(echo_result, DispatchHandled)
    assert isinstance(weather_result, DispatchHandled)
    assert echo_result.tool_message.content == "hi"
    assert weather_result.tool_message.content == "sunny in Oslo"


def _raiser_tool() -> PydanticTool[_EchoArgs]:
    """Build a tool whose function raises immediately, the user-code defect of the dispatch_many tests."""

    async def _raiser_function(args: _EchoArgs) -> str:
        """Fail with an ordinary user-code exception naming the call's text.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError(f"broke on {args.text}")

    return PydanticTool(
        name="raiser",
        description="Raises an ordinary exception.",
        args_model=_EchoArgs,
        function=_raiser_function,
    )


def test_dispatch_many_runs_concurrently_and_keeps_call_order() -> None:
    """dispatch_many runs calls concurrently and preserves call order."""

    async def _run() -> tuple[DispatchManyOutcome, ...]:
        gate = asyncio.Event()

        async def _waiter_function(args: _EchoArgs) -> str:
            """Block until the sibling call opens the gate, then echo."""
            await gate.wait()
            return f"waited {args.text}"

        async def _setter_function(args: _EchoArgs) -> str:
            """Open the gate the sibling call blocks on, then echo."""
            gate.set()
            return f"set {args.text}"

        manager = ToolManager([
            PydanticTool(
                name="waiter",
                description="Waits.",
                args_model=_EchoArgs,
                function=_waiter_function,
            ),
            PydanticTool(
                name="setter", description="Sets.", args_model=_EchoArgs, function=_setter_function
            ),
        ])
        tool_calls = [
            ToolCall(id="c1", name="waiter", args_json='{"text": "a"}'),
            ToolCall(id="c2", name="setter", args_json='{"text": "b"}'),
        ]
        return await asyncio.wait_for(manager.dispatch_many(tool_calls), timeout=5)

    outcomes = asyncio.run(_run())
    assert [outcome.tool_message.tool_call_id for outcome in outcomes] == ["c1", "c2"]
    assert [outcome.tool_message.content for outcome in outcomes] == ["waited a", "set b"]


def test_dispatch_many_returns_every_variant_in_call_order() -> None:
    """A handled call, bad args, and an off-list name land as their outcome variants, in call order, no raise."""
    tool_calls = [
        ToolCall(id="c1", name="echo", args_json='{"text": "hi"}'),
        ToolCall(id="c2", name="echo", args_json='{"wrong": "key"}'),
        ToolCall(id="c3", name="missing", args_json="{}"),
    ]
    outcomes = asyncio.run(ToolManager([_echo_tool()]).dispatch_many(tool_calls))
    assert [type(outcome) for outcome in outcomes] == [
        DispatchHandled,
        DispatchInvalidToolArgs,
        DispatchUnknownTool,
    ]
    assert [outcome.tool_message.tool_call_id for outcome in outcomes] == ["c1", "c2", "c3"]


def test_dispatch_many_of_no_calls_returns_empty() -> None:
    """An empty tool_calls returns an empty outcome tuple."""
    assert asyncio.run(ToolManager([_echo_tool()]).dispatch_many([])) == ()


def _running_echo_tool(ran_texts: list[str]) -> PydanticTool[_EchoArgs]:
    """Build an echo tool that appends each call's text to ran_texts, recording which calls executed."""

    async def _recording_function(args: _EchoArgs) -> str:
        """Record the text, then echo it."""
        ran_texts.append(args.text)
        return args.text

    return PydanticTool(
        name="echo",
        description="Echo the text back.",
        args_model=_EchoArgs,
        function=_recording_function,
    )


def test_dispatch_many_returns_a_precomputed_message_at_the_call_index() -> None:
    """Precomputed preserves ToolMessage identity and call order without dispatch."""
    supplied_message = ToolMessage(
        tool_call_id="c2", content="skipped: duplicate call", is_error=True
    )
    tool_calls = [
        ToolCall(id="c1", name="echo", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="echo", args_json='{"text": "b"}'),
        ToolCall(id="c3", name="echo", args_json='{"text": "c"}'),
    ]
    ran_texts: list[str] = []
    manager = ToolManager([_running_echo_tool(ran_texts)])
    outcomes = asyncio.run(
        manager.dispatch_many(
            tool_calls,
            precomputed=lambda tool_call: supplied_message if tool_call.id == "c2" else None,
        )
    )
    assert [type(outcome) for outcome in outcomes] == [
        DispatchHandled,
        DispatchPrecomputed,
        DispatchHandled,
    ]
    assert [outcome.tool_message.tool_call_id for outcome in outcomes] == ["c1", "c2", "c3"]
    assert outcomes[1].tool_message is supplied_message
    assert sorted(ran_texts) == ["a", "c"]


def test_dispatch_many_precomputed_id_mismatch_raises_before_any_dispatch() -> None:
    """Precomputed rejects a mismatched tool_call_id before dispatch."""
    tool_calls = [
        ToolCall(id="c1", name="echo", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="echo", args_json='{"text": "b"}'),
    ]
    ran_texts: list[str] = []
    manager = ToolManager([_running_echo_tool(ran_texts)])

    def _mismatched(tool_call: ToolCall) -> ToolMessage | None:
        if tool_call.id != "c2":
            return None
        return ToolMessage(tool_call_id="other-id", content="skipped")

    with pytest.raises(ValueError, match=r"'c2'.*'other-id'"):
        _ = asyncio.run(manager.dispatch_many(tool_calls, precomputed=_mismatched))
    assert ran_texts == []


def test_dispatch_many_precomputed_raising_propagates_before_any_dispatch() -> None:
    """A precomputed that raises is a user-code defect: the exception propagates ungrouped and no tool ran."""
    tool_calls = [
        ToolCall(id="c1", name="echo", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="echo", args_json='{"text": "b"}'),
    ]
    ran_texts: list[str] = []
    manager = ToolManager([_running_echo_tool(ran_texts)])

    def _broken(tool_call: ToolCall) -> ToolMessage | None:
        raise RuntimeError(f"broken precomputed on {tool_call.id}")

    with pytest.raises(RuntimeError, match="broken precomputed on c1"):
        _ = asyncio.run(manager.dispatch_many(tool_calls, precomputed=_broken))
    assert ran_texts == []


def test_dispatch_many_group_includes_precomputed_outcomes() -> None:
    """A batch that raises still carries a precomputed-answered call in completed_outcomes, at its position."""
    supplied_message = ToolMessage(
        tool_call_id="c1", content="skipped: over the call limit", is_error=True
    )
    tool_calls = [
        ToolCall(id="c1", name="raiser", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="raiser", args_json='{"text": "b"}'),
        ToolCall(id="c3", name="echo", args_json='{"text": "ok"}'),
    ]
    manager = ToolManager([_raiser_tool(), _echo_tool()])
    with pytest.raises(DispatchExceptionGroup) as caught:
        _ = asyncio.run(
            manager.dispatch_many(
                tool_calls,
                precomputed=lambda tool_call: supplied_message if tool_call.id == "c1" else None,
            )
        )
    group = caught.value
    assert [str(error) for error in group.exceptions] == ["broke on b"]
    assert [type(outcome) for outcome in group.completed_outcomes] == [
        DispatchPrecomputed,
        DispatchHandled,
    ]
    assert [outcome.tool_message.tool_call_id for outcome in group.completed_outcomes] == [
        "c1",
        "c3",
    ]


def test_dispatch_many_raises_the_group_after_siblings_settle() -> None:
    """DispatchExceptionGroup preserves settled sibling outcomes and app_data."""
    receipt = _Receipt(record_id="rec-7")

    async def _spender_function(args: _EchoArgs) -> ToolOutputExplicit[_Receipt]:
        """Yield twice so the sibling defect fires first, then return content plus the receipt."""
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return ToolOutputExplicit(content=f"charged {args.text}", app_data=receipt)

    spender = PydanticTool(
        name="spender",
        description="Spends money, then reports.",
        args_model=_EchoArgs,
        function=_spender_function,
    )
    tool_calls = [
        ToolCall(id="c1", name="raiser", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="spender", args_json='{"text": "b"}'),
    ]
    manager = ToolManager([_raiser_tool(), spender])
    with pytest.raises(DispatchExceptionGroup) as caught:
        _ = asyncio.run(manager.dispatch_many(tool_calls))
    group = caught.value
    assert [str(error) for error in group.exceptions] == ["broke on a"]
    assert len(group.completed_outcomes) == 1
    outcome = group.completed_outcomes[0]
    assert isinstance(outcome, DispatchHandled)
    assert outcome.tool_message.tool_call_id == "c2"
    assert outcome.tool_message.content == "charged b"
    assert outcome.app_data is receipt


def test_dispatch_many_collects_every_defect_in_call_order() -> None:
    """Two raising functions land together in exceptions, ordered by call position, siblings settled."""
    tool_calls = [
        ToolCall(id="c1", name="raiser", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="echo", args_json='{"text": "ok"}'),
        ToolCall(id="c3", name="raiser", args_json='{"text": "b"}'),
    ]
    manager = ToolManager([_raiser_tool(), _echo_tool()])
    with pytest.raises(DispatchExceptionGroup) as caught:
        _ = asyncio.run(manager.dispatch_many(tool_calls))
    group = caught.value
    assert [str(error) for error in group.exceptions] == ["broke on a", "broke on b"]
    assert [outcome.tool_message.tool_call_id for outcome in group.completed_outcomes] == ["c2"]


def test_dispatch_exception_group_except_star_subgroup_keeps_completed_outcomes() -> None:
    """except* catches the group, and the derived subgroup still carries completed_outcomes.

    derive preserves completed_outcomes on an except* subgroup.
    """
    tool_calls = [
        ToolCall(id="c1", name="raiser", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="echo", args_json='{"text": "ok"}'),
    ]
    manager = ToolManager([_raiser_tool(), _echo_tool()])
    subgroups: list[ExceptionGroup[RuntimeError]] = []
    try:
        _ = asyncio.run(manager.dispatch_many(tool_calls))
    except* RuntimeError as subgroup:
        subgroups.append(subgroup)
    assert len(subgroups) == 1
    caught = subgroups[0]
    assert isinstance(caught, DispatchExceptionGroup)
    assert [outcome.tool_message.tool_call_id for outcome in caught.completed_outcomes] == ["c2"]


def test_dispatch_many_re_raises_a_sibling_cancelled_error_bare() -> None:
    """A tool-raised CancelledError propagates without grouping."""

    async def _self_cancelling_function(args: _EchoArgs) -> str:
        """Produce a CancelledError without anyone cancelling the enclosing task.

        Raises:
            asyncio.CancelledError: always.
        """
        raise asyncio.CancelledError(args.text)

    manager = ToolManager([
        PydanticTool(
            name="self_cancel",
            description="Produces a CancelledError.",
            args_model=_EchoArgs,
            function=_self_cancelling_function,
        ),
        _echo_tool(),
    ])
    tool_calls = [
        ToolCall(id="c1", name="self_cancel", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="echo", args_json='{"text": "ok"}'),
    ]
    with pytest.raises(asyncio.CancelledError):
        _ = asyncio.run(manager.dispatch_many(tool_calls))


def test_dispatch_many_chains_defects_onto_a_bare_base_exception() -> None:
    """Defects co-occurring with a sibling's CancelledError chain as its __cause__, not vanish.

    Cancellation chains sibling defects and settled outcomes in DispatchExceptionGroup.
    """

    async def _self_cancelling_function(args: _EchoArgs) -> str:
        """Produce a CancelledError without anyone cancelling the enclosing task.

        Raises:
            asyncio.CancelledError: always.
        """
        raise asyncio.CancelledError(args.text)

    manager = ToolManager([
        PydanticTool(
            name="self_cancel",
            description="Produces a CancelledError.",
            args_model=_EchoArgs,
            function=_self_cancelling_function,
        ),
        _raiser_tool(),
        _echo_tool(),
    ])
    tool_calls = [
        ToolCall(id="c1", name="raiser", args_json='{"text": "a"}'),
        ToolCall(id="c2", name="self_cancel", args_json='{"text": "b"}'),
        ToolCall(id="c3", name="echo", args_json='{"text": "ok"}'),
    ]
    with pytest.raises(asyncio.CancelledError) as caught:
        _ = asyncio.run(manager.dispatch_many(tool_calls))
    cause = caught.value.__cause__
    assert isinstance(cause, DispatchExceptionGroup)
    assert [str(error) for error in cause.exceptions] == ["broke on a"]
    assert [outcome.tool_message.tool_call_id for outcome in cause.completed_outcomes] == ["c3"]


def test_dispatch_many_cancellation_settles_siblings_then_propagates() -> None:
    """dispatch_many waits for sibling cancellation before propagating CancelledError."""
    unwound: list[str] = []

    async def _hanging_function(args: _EchoArgs) -> str:
        """Hang until cancelled, recording the unwind in finally."""
        try:
            await asyncio.Event().wait()
        finally:
            unwound.append(args.text)
        return "unreachable"

    async def _run() -> None:
        manager = ToolManager([
            PydanticTool(
                name="hang", description="Hangs.", args_model=_EchoArgs, function=_hanging_function
            )
        ])
        tool_calls = [
            ToolCall(id="c1", name="hang", args_json='{"text": "a"}'),
            ToolCall(id="c2", name="hang", args_json='{"text": "b"}'),
        ]
        dispatch_task = asyncio.ensure_future(manager.dispatch_many(tool_calls))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        _ = dispatch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch_task
        assert sorted(unwound) == ["a", "b"]

    asyncio.run(_run())


class _CapturedAnswer(BaseModel):
    """The captured model of the CaptureTool tests."""

    answer: str


def _answer_capture_tool() -> CaptureTool[_CapturedAnswer]:
    return CaptureTool(
        name="final_response",
        description="Submit the final answer.",
        args_model=_CapturedAnswer,
        acknowledgement="Answer received",
    )


def test_capture_tool_schema_converts_name_description_and_args_schema() -> None:
    """CaptureTool.schema carries the name, description, and args_model's JSON schema, like PydanticTool."""
    schema = _answer_capture_tool().schema()
    assert schema.name == "final_response"
    assert schema.description == "Submit the final answer."
    assert schema.args_schema == _CapturedAnswer.model_json_schema()


@pytest.mark.parametrize(
    ("build_tool", "expected_acknowledgement"),
    [
        (_answer_capture_tool, "Answer received"),
        (
            lambda: CaptureTool(
                name="final_response",
                description="Submit the final answer.",
                args_model=_CapturedAnswer,
            ),
            "Acknowledged",
        ),
    ],
    ids=["acknowledgement_given", "acknowledgement_defaulted"],
)
def test_capture_returns_the_validated_instance_beside_its_acknowledgement(
    build_tool: Callable[[], CaptureTool[_CapturedAnswer]], expected_acknowledgement: str
) -> None:
    """A valid CaptureTool call returns typed output and acknowledgement."""
    call = ToolCall(id="call1", name="final_response", args_json='{"answer": "tide"}')
    outcome = asyncio.run(build_tool().capture(call))
    assert isinstance(outcome, DispatchCaptured)
    assert outcome.captured.answer == "tide"
    assert outcome.tool_message.tool_call_id == "call1"
    assert outcome.tool_message.content == expected_acknowledgement
    assert outcome.tool_message.is_error is False


def test_capture_invalid_args_delegates_to_the_shared_renderer() -> None:
    """An invalid call returns the same DispatchInvalidToolArgs any tool form produces.

    The content and details are exactly the pydantic conversion and rendering,
    so the model reads identical field-level corrections whether it miscalled a CaptureTool or a PydanticTool.
    """
    args_json = '{"wrong": "key"}'
    with pytest.raises(ValidationError) as caught:
        _ = _CapturedAnswer.model_validate_json(args_json)
    expected_details = _details_from_pydantic(caught.value)
    expected_content = render_invalid_tool_args("final_response", expected_details)
    call = ToolCall(id="call1", name="final_response", args_json=args_json)
    outcome = asyncio.run(_answer_capture_tool().capture(call))
    assert isinstance(outcome, DispatchInvalidToolArgs)
    assert outcome.tool_message.is_error is True
    assert outcome.tool_message.tool_call_id == "call1"
    assert outcome.tool_message.content == expected_content
    assert outcome.details == expected_details


def test_capture_malformed_and_non_object_json_return_the_invalid_args_variant() -> None:
    """Malformed JSON and a non-object JSON value land in the same DispatchInvalidToolArgs variant, no raise.

    args_model.model_validate_json raises ValidationError for both shapes,
    so capture returns rendered corrections exactly as it does for a well-formed object with wrong fields.
    """
    for args_json in ("not json", '"scalar"'):
        call = ToolCall(id="call1", name="final_response", args_json=args_json)
        outcome = asyncio.run(_answer_capture_tool().capture(call))
        assert isinstance(outcome, DispatchInvalidToolArgs)
        assert outcome.tool_message.is_error is True
        assert "invalid arguments for final_response" in outcome.tool_message.content
        assert len(outcome.details) >= 1


def test_capture_tool_dispatch_erases_the_capture_onto_app_data() -> None:
    """ToolManager dispatches CaptureTool to DispatchHandled with captured app_data."""
    manager = ToolManager([_echo_tool(), _answer_capture_tool()])
    assert manager.schemas() == (_echo_tool().schema(), _answer_capture_tool().schema())
    call = ToolCall(id="call1", name="final_response", args_json='{"answer": "tide"}')
    result = asyncio.run(manager.dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "Answer received"
    assert result.tool_message.is_error is False
    assert result.app_data == _CapturedAnswer(answer="tide")


def test_capture_tool_dispatch_returns_invalid_args_variant_for_invalid_args() -> None:
    """A manager-routed invalid call comes back as the same DispatchInvalidToolArgs capture returns."""
    call = ToolCall(id="call1", name="final_response", args_json='{"wrong": "key"}')
    result = asyncio.run(ToolManager([_answer_capture_tool()]).dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True
    assert "invalid arguments for final_response" in result.tool_message.content
    assert any("answer" in detail.path for detail in result.details)
