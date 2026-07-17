"""Tool.validate_and_run and ToolManager routing.

The typed validate-then-call pair lives on Tool; these tests pin the error contract:
an invalid args_json becomes an is_error ToolMessage and a function returning a
ToolOutputExplicit(is_error=True) carries that failure with its app_data,
while every function exception (including a function-internal ValidationError) propagates as a user-code defect.
"""

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from pydantic import BaseModel, Field, ValidationError

from langchaint import (
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchUnknownTool,
    ImagePart,
    InvalidToolArgsError,
    Part,
    RawSchemaTool,
    TextPart,
    Tool,
    ToolCall,
    ToolManager,
    ToolOutputExplicit,
)
from langchaint.tools import render_invalid_tool_args, render_unknown_tool

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


async def _validation_error_function(args: _EchoArgs) -> str:
    """Fail with a function-internal ValidationError, a user-code defect.

    The model_validate call always raises because the payload lacks the required text field.
    """
    _EchoArgs.model_validate({"wrong": args.text})
    return "unreachable"


def _echo_tool() -> Tool[_EchoArgs]:
    """Build the echo tool."""
    return Tool(
        name="echo",
        description="Echo the text back.",
        args_model=_EchoArgs,
        function=_echo_function,
    )


def test_schema_converts_name_description_and_args_schema() -> None:
    """Tool.schema carries the name, description, and the args JSON schema."""
    schema = _echo_tool().schema()
    assert schema.name == "echo"
    assert schema.description == "Echo the text back."
    assert schema.args_schema == _EchoArgs.model_json_schema()
    assert ToolManager([_echo_tool()]).schemas() == (schema,)


def test_validate_and_run_returns_the_function_result() -> None:
    """Valid args_json reaches the function as the validated model."""
    result = asyncio.run(_echo_tool().validate_and_run('{"text": "tide"}'))
    assert result == "tide"


def test_validate_and_run_raises_invalid_tool_args_on_bad_json() -> None:
    """An args_json that fails validation raises InvalidToolArgsError."""
    with pytest.raises(InvalidToolArgsError):
        asyncio.run(_echo_tool().validate_and_run('{"text": 5.5}'))


def test_invalid_tool_args_holds_the_validation_error() -> None:
    """The raised InvalidToolArgsError carries the live ValidationError, not just a string.

    validation_error is a pydantic ValidationError whose errors() name the failing field,
    so a caller can read the structured detail.
    str(the error) is non-empty and names the field too,
    which a dropped __str__ override (leaving super().__init__()'s empty message) would fail.
    """
    with pytest.raises(InvalidToolArgsError) as caught:
        asyncio.run(_echo_tool().validate_and_run('{"wrong": "key"}'))
    error = caught.value
    assert isinstance(error.validation_error, ValidationError)
    assert any("text" in entry["loc"] for entry in error.validation_error.errors())
    assert "text" in str(error)


def test_render_invalid_tool_args_formats_per_field() -> None:
    """The renderer emits a header then one loc:msg line per failure, dropping the noise.

    A nested-object path renders dot-joined (to.0.email), a list-index path stringifies the int segment (to.1),
    proving the str(segment) join rather than a bare ".".join(loc) that would TypeError on the int.
    type codes, documentation URLs, and echoed input never appear.
    """

    class _Recipient(BaseModel):
        """One recipient with a required email."""

        email: str

    class _SendArgs(BaseModel):
        """Send arguments with a recipient list and a non-empty subject."""

        to: list[_Recipient]
        subject: str = Field(min_length=1)

    with pytest.raises(ValidationError) as caught:
        _SendArgs.model_validate_json('{"to":[{"x":1},5],"subject":""}')
    validation_error = caught.value
    rendered = render_invalid_tool_args("send_email", validation_error)
    expected = "\n".join(
        ["invalid arguments for send_email:"]
        + [
            f"  {'.'.join(str(segment) for segment in entry['loc'])}: {entry['msg']}"
            for entry in validation_error.errors()
        ]
    )
    assert rendered == expected
    assert "\n  to.0.email: " in rendered
    assert "\n  to.1: " in rendered
    assert "https://" not in rendered
    assert "errors.pydantic.dev" not in rendered
    assert "type=" not in rendered


def test_render_invalid_tool_args_renders_empty_loc_as_root() -> None:
    """A failure with an empty loc (a non-object input) renders as the (root) path.

    An empty loc cannot coexist with per-field locs in one ValidationError,
    so this branch needs its own input: a JSON that is not an object fails at the whole model with loc ().
    """

    class _SendArgs(BaseModel):
        """Send arguments; validated here against a non-object payload."""

        subject: str

    with pytest.raises(ValidationError) as caught:
        _SendArgs.model_validate_json("5")
    rendered = render_invalid_tool_args("send_email", caught.value)
    header, root_line = rendered.splitlines()
    assert header == "invalid arguments for send_email:"
    assert root_line.startswith("  (root): ")


def test_dispatch_wraps_success_in_a_tool_message() -> None:
    """A valid call comes back as a DispatchHandled non-error ToolMessage with the call id."""
    call = ToolCall(id="call1", name="echo", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.tool_call_id == "call1"
    assert result.tool_message.content == "tide"
    assert result.tool_message.is_error is False
    assert result.app_data is None


def test_dispatch_carries_a_parts_result_into_tool_message_content() -> None:
    """A function returning a sequence of parts reaches ToolMessage.content as a tuple of those parts.

    A success path that dropped or stringified the result would fail this equality.
    """
    async def _parts_function(args: _EchoArgs) -> Sequence[Part]:
        """Return content parts built from the validated text instead of a string."""
        return [TextPart(text=args.text), ImagePart(data=b"png", media_type="image/png")]

    tool = Tool(
        name="render",
        description="Return parts.",
        args_model=_EchoArgs,
        function=_parts_function,
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


def test_dispatch_returns_invalid_args_arm_for_invalid_args() -> None:
    """An invalid args_json is model data: a DispatchInvalidToolArgs holding the ValidationError, no raise.

    The tool_message must carry the validation detail (the failing field), or the model has nothing to correct against,
    and validation_error names the failing field for a caller authoring its own reply.
    """
    call = ToolCall(id="call1", name="echo", args_json='{"wrong": "key"}')
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True
    assert "invalid arguments for echo" in result.tool_message.content
    assert "text" in result.tool_message.content
    assert any("text" in entry["loc"] for entry in result.validation_error.errors())


def test_dispatch_delegates_invalid_args_content_to_the_renderer() -> None:
    """The invalid-args content is exactly render_invalid_tool_args of the stored error.

    Catching the InvalidToolArgsError from the same validate_and_run and rendering its
    validation_error reproduces the content, pinning that dispatch passes the tool name and the
    held ValidationError through to the renderer rather than formatting the string itself.
    """
    args_json = '{"wrong": "key"}'
    with pytest.raises(InvalidToolArgsError) as caught:
        asyncio.run(_echo_tool().validate_and_run(args_json))
    expected_content = render_invalid_tool_args("echo", caught.value.validation_error)
    call = ToolCall(id="call1", name="echo", args_json=args_json)
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True
    assert result.tool_message.content == expected_content


def test_dispatch_invalid_args_match_arm_reads_errors_without_assert() -> None:
    """Matching DispatchInvalidToolArgs narrows validation_error, so errors() reads with no assert.

    The match arm reads result.validation_error.errors() directly;
    pyrefly checks that this typechecks with no assert, cast, or type ignore, which is the point of the split arm.
    """
    call = ToolCall(id="call1", name="echo", args_json='{"wrong": "key"}')
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    match result:
        case DispatchInvalidToolArgs():
            fields = [entry["loc"] for entry in result.validation_error.errors()]
            assert any("text" in loc for loc in fields)
        case DispatchHandled():
            pytest.fail("invalid args must return DispatchInvalidToolArgs")


def test_dispatch_returns_unknown_tool_arm_for_off_list_name() -> None:
    """A name the manager does not hold is model-correctable data: a DispatchUnknownTool, no raise.

    The arm carries called_name and a default is_error ToolMessage naming held_names,
    so the loop survives a hallucinated name or a tool_call stranded by a rebind and the model can retry.
    Matching the arm narrows called_name, read with no assert.
    """
    call = ToolCall(id="call1", name="missing", args_json="{}")
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchUnknownTool)
    assert result.called_name == "missing"
    assert result.tool_message.tool_call_id == "call1"
    assert result.tool_message.is_error is True
    assert "missing" in result.tool_message.content
    assert "echo" in result.tool_message.content


def test_dispatch_unknown_tool_content_delegates_to_the_renderer() -> None:
    """The unknown-tool content is exactly render_unknown_tool of called_name and held_names."""
    call = ToolCall(id="call1", name="missing", args_json="{}")
    result = asyncio.run(ToolManager([_echo_tool()]).dispatch(call))
    assert isinstance(result, DispatchUnknownTool)
    assert result.tool_message.content == render_unknown_tool(called_name="missing", held_names=("echo",))


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
    tool = Tool(
        name="broken",
        description="Raises from its own pydantic use.",
        args_model=_EchoArgs,
        function=_validation_error_function,
    )
    call = ToolCall(id="call1", name="broken", args_json='{"text": "tide"}')
    with pytest.raises(ValidationError):
        asyncio.run(ToolManager([tool]).dispatch(call))


def test_dispatch_carries_a_returned_is_error_result() -> None:
    """A function returning ToolOutputExplicit(is_error=True) becomes an is_error ToolMessage."""

    async def _returned_error_function(args: _EchoArgs) -> ToolOutputExplicit:
        """Report a model-visible failure by returning, not raising."""
        return ToolOutputExplicit(
            content=f"cannot echo {args.text!r}: try a shorter value", is_error=True
        )

    tool = Tool(
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


def test_dispatch_passes_a_pydantic_app_data_through_live() -> None:
    """A success ToolOutputExplicit with a pydantic app_data reaches result.app_data by identity."""
    cites = _Cites(citations=["doc-1", "doc-2"])

    async def _cited_function(args: _EchoArgs) -> ToolOutputExplicit[_Cites]:
        """Return model-visible content plus the app_data the model never sees."""
        return ToolOutputExplicit(content=f"echoed {args.text}", app_data=cites)

    tool = Tool(
        name="cited",
        description="Returns content plus app_data.",
        args_model=_EchoArgs,
        function=_cited_function,
    )
    call = ToolCall(id="call1", name="cited", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([tool]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "echoed tide"
    assert result.tool_message.is_error is False
    assert result.app_data is cites


def test_dispatch_passes_a_mapping_app_data_through_unchanged() -> None:
    """A mapping app_data reaches result.app_data unchanged, for MCP tools with unknown schemas."""
    app_data = {"citations": ["doc-1"]}

    async def _mapping_function(args: _EchoArgs) -> ToolOutputExplicit[Mapping[str, object]]:
        """Return content plus a mapping app_data."""
        return ToolOutputExplicit(content=f"echoed {args.text}", app_data=app_data)

    tool = Tool(
        name="mapping",
        description="Returns a mapping app_data.",
        args_model=_EchoArgs,
        function=_mapping_function,
    )
    call = ToolCall(id="call1", name="mapping", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([tool]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.app_data == {"citations": ["doc-1"]}


def test_dispatch_carries_app_data_on_the_error_outcome() -> None:
    """An is_error ToolOutputExplicit still carries its app_data: failure and record ride together."""
    receipt = _Receipt(record_id="rec-7")

    async def _persist_then_fail_function(args: _EchoArgs) -> ToolOutputExplicit[_Receipt]:
        """Persist a record, then report a model-visible failure carrying that record."""
        return ToolOutputExplicit(
            content=f"declined {args.text}", is_error=True, app_data=receipt
        )

    tool = Tool(
        name="persist",
        description="Fails after persisting a record.",
        args_model=_EchoArgs,
        function=_persist_then_fail_function,
    )
    call = ToolCall(id="call1", name="persist", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([tool]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.is_error is True
    assert result.app_data is receipt


def test_dispatch_reads_a_pydantic_app_data_typed_without_revalidation() -> None:
    """The read site narrows result.app_data to its concrete type and reads its field, no model_validate."""

    async def _cited_function(args: _EchoArgs) -> ToolOutputExplicit[_Cites]:
        """Return content plus a pydantic app_data."""
        return ToolOutputExplicit(
            content=f"echoed {args.text}", app_data=_Cites(citations=["doc-1"])
        )

    tool = Tool(
        name="cited",
        description="Returns content plus app_data.",
        args_model=_EchoArgs,
        function=_cited_function,
    )
    call = ToolCall(id="call1", name="cited", args_json='{"text": "tide"}')
    result = asyncio.run(ToolManager([tool]).dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert isinstance(result.app_data, _Cites)
    assert result.app_data.citations == ["doc-1"]


def test_tool_dispatch_carries_the_tools_app_data_type_without_isinstance() -> None:
    """Tool.dispatch returns DispatchHandled[AppDataT], so the read narrows to the concrete type.

    Dispatching a known single tool keeps its own app_data type, so the local annotation cites:
    _Cites | None typechecks (it would be a pyrefly error if app_data were the
    heterogeneous BaseModel | Mapping[str, object] | None the manager surfaces),
    and .citations reads with no isinstance narrowing the discriminator already guarantees.
    """

    async def _cited_function(args: _EchoArgs) -> ToolOutputExplicit[_Cites]:
        """Return content plus a pydantic app_data."""
        return ToolOutputExplicit(
            content=f"echoed {args.text}", app_data=_Cites(citations=["doc-1"])
        )

    tool = Tool(
        name="cited",
        description="Returns content plus app_data.",
        args_model=_EchoArgs,
        function=_cited_function,
    )
    call = ToolCall(id="call1", name="cited", args_json='{"text": "tide"}')
    result = asyncio.run(tool.dispatch(call))
    assert isinstance(result, DispatchHandled)
    cites: _Cites | None = result.app_data
    assert cites is not None
    assert cites.citations == ["doc-1"]


def test_tool_dispatch_returns_invalid_args_arm_for_invalid_args() -> None:
    """Tool.dispatch is the same validate-then-wrap as the manager: bad args are a DispatchInvalidToolArgs.

    The manager delegates to Tool.dispatch, so the argument-validation rendering must live here;
    a bad payload comes back as the invalid-args arm holding the ValidationError, not a raise.
    """
    call = ToolCall(id="call1", name="echo", args_json='{"wrong": "key"}')
    result = asyncio.run(_echo_tool().dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True
    assert "invalid arguments for echo" in result.tool_message.content
    assert any("text" in entry["loc"] for entry in result.validation_error.errors())


def test_plain_function_exception_propagates_as_a_defect() -> None:
    """Any non-validation function exception propagates unchanged."""

    async def _failing_function(args: _EchoArgs) -> str:
        """Fail with an ordinary user-code exception.

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError(f"function broke on {args.text}")

    tool = Tool(
        name="failing",
        description="Raises an ordinary exception.",
        args_model=_EchoArgs,
        function=_failing_function,
    )
    call = ToolCall(id="call1", name="failing", args_json='{"text": "tide"}')
    with pytest.raises(RuntimeError, match="function broke on tide"):
        asyncio.run(ToolManager([tool]).dispatch(call))


def test_duplicate_tool_names_are_rejected() -> None:
    """Two tools sharing a name raise ValueError at construction."""
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolManager([_echo_tool(), _echo_tool()])


async def _weather_function(args: dict[str, object]) -> str:
    """Read the parsed city argument and report it, with no pydantic model in sight.

    Annotated dict[str, object], the declared parameter type of RawSchemaTool.function and the typical user form.
    """
    return f"sunny in {args['city']}"


def _weather_tool() -> RawSchemaTool:
    """Build a RawSchemaTool from a raw JSON schema, the shape an MCP tool arrives in."""
    return RawSchemaTool(
        name="weather",
        description="Report the weather.",
        args_schema=_WEATHER_SCHEMA,
        function=_weather_function,
    )


def test_schema_tool_schema_passes_the_raw_json_schema_through_unchanged() -> None:
    """RawSchemaTool.schema carries the args_schema by identity, not a model_json_schema derivation.

    A pydantic tool derives its schema; a RawSchemaTool already holds the JSON schema the provider wants, so the wire
    schema must be the exact object passed in (an MCP inputSchema), byte-for-byte, with no reshaping.
    """
    schema = _weather_tool().schema()
    assert schema.name == "weather"
    assert schema.description == "Report the weather."
    assert schema.args_schema is _WEATHER_SCHEMA


def test_schema_tool_dispatch_passes_the_parsed_mapping_to_the_function() -> None:
    """A valid call reaches the function as the parsed arguments mapping and comes back a DispatchHandled."""
    call = ToolCall(id="call1", name="weather", args_json='{"city": "Oslo"}')
    result = asyncio.run(_weather_tool().dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "sunny in Oslo"
    assert result.tool_message.is_error is False
    assert result.app_data is None


def test_schema_tool_dispatch_does_no_field_level_validation() -> None:
    """An argument outside the schema still runs: field rules are the tool owner's, not RawSchemaTool's.

    The JSON is a well-formed object, so the only local check passes; the schema requires city and forbids extras,
    yet dispatch runs the function unchanged, proving RawSchemaTool defers field validation to the server.
    """
    call = ToolCall(id="call1", name="weather", args_json='{"town": "Oslo", "city": "Bergen"}')
    result = asyncio.run(_weather_tool().dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "sunny in Bergen"


def test_schema_tool_dispatch_returns_invalid_args_for_non_object_json() -> None:
    """A non-object args_json is the one thing RawSchemaTool rejects locally: a DispatchInvalidToolArgs, no raise.

    A scalar JSON payload cannot be a tool's arguments, so it comes back as the invalid-args arm holding a live
    ValidationError, symmetric with the pydantic tool, so the loop and the model can recover.
    """
    call = ToolCall(id="call1", name="weather", args_json="5")
    result = asyncio.run(_weather_tool().dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True
    assert "invalid arguments for weather" in result.tool_message.content
    assert result.validation_error.error_count() >= 1


def test_schema_tool_dispatch_returns_invalid_args_for_malformed_json() -> None:
    """Malformed args_json is rejected the same way: a DispatchInvalidToolArgs, not a propagating decode error."""
    call = ToolCall(id="call1", name="weather", args_json='{"city": ')
    result = asyncio.run(_weather_tool().dispatch(call))
    assert isinstance(result, DispatchInvalidToolArgs)
    assert result.tool_message.is_error is True


def test_schema_tool_dispatch_carries_a_mapping_app_data_through() -> None:
    """A RawSchemaTool function returning a ToolOutputExplicit rides its app_data through, the MCP result channel."""
    raw_result = {"forecast": ["sunny"], "source": "mcp"}

    async def _mcp_function(args: Mapping[str, object]) -> ToolOutputExplicit[Mapping[str, object]]:
        """Return model-visible content plus the raw MCP result the model never sees.

        Annotated Mapping[str, object]: accepted against the dict[str, object] parameter by contravariance,
        pinning that the wider annotation keeps typechecking.
        """
        return ToolOutputExplicit(content=f"weather for {args['city']}", app_data=raw_result)

    tool: RawSchemaTool[Mapping[str, object]] = RawSchemaTool(
        name="weather",
        description="Report the weather via MCP.",
        args_schema=_WEATHER_SCHEMA,
        function=_mcp_function,
    )
    call = ToolCall(id="call1", name="weather", args_json='{"city": "Oslo"}')
    result = asyncio.run(tool.dispatch(call))
    assert isinstance(result, DispatchHandled)
    assert result.tool_message.content == "weather for Oslo"
    assert result.app_data is raw_result


def test_tool_manager_holds_a_mix_of_tool_and_schema_tool() -> None:
    """One ToolManager routes to a pydantic Tool and a RawSchemaTool side by side.

    schemas() emits both wire schemas and dispatch reaches each tool, proving DispatchableTool lets the manager hold
    the two forms together without either being a special case.
    """
    manager = ToolManager([_echo_tool(), _weather_tool()])
    names = {schema.name for schema in manager.schemas()}
    assert names == {"echo", "weather"}

    echo_result = asyncio.run(manager.dispatch(ToolCall(id="c1", name="echo", args_json='{"text": "hi"}')))
    weather_result = asyncio.run(manager.dispatch(ToolCall(id="c2", name="weather", args_json='{"city": "Oslo"}')))
    assert isinstance(echo_result, DispatchHandled)
    assert isinstance(weather_result, DispatchHandled)
    assert echo_result.tool_message.content == "hi"
    assert weather_result.tool_message.content == "sunny in Oslo"


def test_schema_tool_dispatch_carries_its_app_data_type_without_isinstance() -> None:
    """RawSchemaTool.dispatch returns DispatchHandled[AppDataT], so the read narrows to the concrete type.

    Dispatching the RawSchemaTool directly (not via the manager) keeps its own app_data type, so the local annotation
    raw: Mapping[str, object] | None typechecks with no isinstance.
    This is the RawSchemaTool analogue of the Tool.dispatch test.
    """

    async def _mcp_function(args: Mapping[str, object]) -> ToolOutputExplicit[Mapping[str, object]]:
        """Return content plus a mapping app_data."""
        return ToolOutputExplicit(content=f"weather for {args['city']}", app_data={"source": "mcp"})

    tool: RawSchemaTool[Mapping[str, object]] = RawSchemaTool(
        name="weather",
        description="Report the weather via MCP.",
        args_schema=_WEATHER_SCHEMA,
        function=_mcp_function,
    )
    result = asyncio.run(tool.dispatch(ToolCall(id="c1", name="weather", args_json='{"city": "Oslo"}')))
    assert isinstance(result, DispatchHandled)
    raw: Mapping[str, object] | None = result.app_data
    assert raw is not None
    assert raw["source"] == "mcp"
