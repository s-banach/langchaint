"""Tools.

A tool comes in three forms, all dispatched by ToolManager.
PydanticTool is the pydantic form: its function accepts one validated BaseModel argument, and its args_model is both the
schema source (model_json_schema) and the validator of the model-produced argument JSON.
JSONSchemaTool is the raw-JSON-schema form for a tool whose schema is a plain JSON schema, not a pydantic model
(an MCP tool discovered at runtime): its function accepts the parsed arguments as a dict[str, object], its
args_schema rides through to the provider unchanged, and dispatch validates the arguments against args_schema
with jsonschema (first that they are a JSON object, then the schema's field-level rules), so a schema violation
renders through the same formatter as a PydanticTool failure and the function only ever sees valid arguments.
Both function-bearing forms return str or a Sequence[Part] (text and images the model then sees).
They may instead return a ToolOutputExplicit carrying that content plus is_error and app_data.
CaptureTool is the function-free form: a tool whose whole job is carrying a validated instance to the application.
The archetype is the final-response tool that ends a tool loop.
Its capture returns the instance as a required field of DispatchCaptured.
Its fixed reply is the acknowledgement the model reads.
A tool function returns data and nothing that steers control flow:
stop, route, escalate, and needs-approval are decisions the application makes in its own loop between turns,
reading app_data or is_error.
The application owns that loop and langchaint ships no agent loop of its own,
so a control-flow return channel (a goto or an engine-state update smuggled through a tool's return value) is forbidden.
No signature introspection and no docstring scraping: name, description, and schema are explicit,
so what the provider sees is exactly what the code states.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import jsonschema.exceptions
import jsonschema.protocols
import jsonschema.validators
from pydantic import BaseModel, TypeAdapter, ValidationError

from langchaint.exceptions import DispatchExceptionGroup, InvalidToolArgsError
from langchaint.messages import MessageContent, ToolCall, ToolMessage


@dataclass(frozen=True, kw_only=True)
class ToolOutputExplicit[AppDataT = None]:
    """The explicit arm of ToolOutput: the model-visible outcome plus app_data.

    content is what the model reads, the same shape as a bare function return and as ToolMessage.content:
    MessageContent, a str or a Sequence[Part] (text and images the model sees) and nothing else, because
    content is model-facing and must already be in a form the model reads;
    a function with a typed result serializes it to that form itself.
    is_error marks a model-visible failure the model should read and adapt to;
    it is the value that lands on ToolMessage.is_error.
    app_data is data the model never sees (citations, retrieved chunks, records the function persisted);
    dispatch routes it to the application untouched, on both the success and the error outcome.
    AppDataT is the type of that data, defaulting to None for a function that carries none;
    the checker solves it from the value the function passes to app_data,
    so a function returning app_data=ProfileRecord(...) makes AppDataT ProfileRecord with no annotation,
    and the app reads that concrete type back off DispatchHandled with no isinstance.
    A function that authors its own app_data uses a pydantic model with a named field, e.g. citations: list[Citation],
    which is self-documenting and typed at the read site;
    a Mapping arm on app_data supports MCP tools whose result schema is unknown at typecheck time, riding through as-is.
    A function that returns bare content is sugar for ToolOutputExplicit(content=..., is_error=False,
    app_data=None) and never constructs this.
    """

    content: MessageContent
    is_error: bool = False
    app_data: AppDataT | None = None


type ToolOutput[AppDataT = None] = MessageContent | ToolOutputExplicit[AppDataT]
"""What a tool function returns.

Bare MessageContent is sugar for ToolOutputExplicit(content=..., is_error=False, app_data=None).
AppDataT is the explicit arm's app_data type, defaulting to None for a function that returns bare content.
"""


@dataclass(frozen=True, kw_only=True)
class DispatchHandled[AppDataT = None]:
    """A tool call the tool executed: the model-facing tool_message plus app_data.

    Covers success and a tool-authored failure; tool_message.is_error distinguishes them, the same bool the tool set.
    tool_message is the ToolMessage the application appends to the conversation and the provider sees.
    app_data is what the tool routed to the application, passed through live and read back at its concrete type.
    PydanticTool.dispatch carries the tool's own AppDataT onto this arm, so a known-tool caller needs no isinstance.
    For the function-bearing forms app_data is the function's, None when the function returned bare content.
    For CaptureTool it is the capture, present on every valid manager-routed call.
    It never reaches the provider: only tool_message enters the conversation the adapters convert.
    ToolManager.dispatch dispatches a heterogeneous tool set whose per-call AppDataT is erased,
    so its DispatchHandled is parameterized with the widest app_data the channel allows
    (BaseModel | Mapping[str, object] | None) and the app folds over that union there.
    """

    tool_message: ToolMessage
    app_data: AppDataT | None = None


@dataclass(frozen=True, kw_only=True)
class InvalidToolArgsDetail:
    """One argument-validation failure, provider- and library-neutral.

    One instance is one detail item of the invalid-arguments condition named by InvalidToolArgsError and
    DispatchInvalidToolArgs: one failure at one path.
    path segments are str for an object key and int for a list index, matching both pydantic's loc and
    jsonschema's absolute_path; an empty path means the failure is about the arguments as a whole.
    message is the reason verbatim from the validator that produced it,
    so the tool owner's own words reach the model unrewritten.
    Both producers are boundary conversions inside langchaint (_details_from_pydantic, _details_from_jsonschema),
    so a consumer reads one vocabulary whichever tool form failed.
    A frozen dataclass, not pydantic: it is a constructed detail value, not a persisted message-tree node,
    so serde and validation buy nothing here.
    """

    path: tuple[str | int, ...]
    message: str


@dataclass(frozen=True, kw_only=True)
class DispatchInvalidToolArgs:
    """A tool call whose arguments failed validation before any function ran.

    tool_message is a default is_error ToolMessage rendered from details for the model to read and correct;
    the application appends it as-is or authors its own reply.
    details is the neutral per-failure detail, a required field: matching this arm narrows it,
    so the application reads it with no assert, cast, or type ignore, and reads no pydantic type.
    Every dispatch-produced outcome carries at least one detail; construction does not enforce that,
    the emptiness guard lives in render_invalid_tool_args, which every dispatch path renders through.
    There is no app_data field because no function ran to produce any.
    """

    tool_message: ToolMessage
    details: tuple[InvalidToolArgsDetail, ...]


@dataclass(frozen=True, kw_only=True)
class DispatchUnknownTool:
    """A tool call naming a tool the ToolManager does not hold.

    tool_message is a default is_error ToolMessage naming the held tools for the model to read and correct,
    symmetric with DispatchInvalidToolArgs; the application appends it as-is or authors its own reply.
    called_name is the off-list name the model produced, a required field:
    matching this arm narrows it, so the application reads it with no assert.
    An off-list name is model data the model can correct: a provider can emit a name outside the sent schemas,
    and rebinding to a different tool set can strand an earlier turn's tool_call.
    There is no app_data field because no function ran to produce any.
    """

    tool_message: ToolMessage
    called_name: str


type DispatchOutcome = (
    DispatchHandled[BaseModel | Mapping[str, object] | None]
    | DispatchInvalidToolArgs
    | DispatchUnknownTool
)
"""The three outcomes of ToolManager.dispatch on one tool call.

The manager dispatches a heterogeneous tool set, so app_data is the widest the channel allows,
BaseModel | Mapping[str, object] | None (a bare-content function leaves it None, hence the None arm in the parameter).
A caller that knows the single tool it dispatched keeps the tool's own app_data type by calling that tool's
own dispatch instead.
Every arm carries tool_message, so a consumer that only appends the reply reads result.tool_message with no match.
A consumer that reads the field-level failure detail matches DispatchInvalidToolArgs;
one that reads the off-list name matches DispatchUnknownTool.
"""


@dataclass(frozen=True, kw_only=True)
class ToolSchema:
    """The provider-neutral description of one tool.

    Adapters convert it to their wire shape at bind time.
    args_schema is the JSON schema of the tool's arguments: a PydanticTool supplies its args_model's
    model_json_schema output, a JSONSchemaTool supplies its raw args_schema unchanged.
    """

    name: str
    description: str
    args_schema: Mapping[str, object]


@dataclass(frozen=True, kw_only=True)
class PydanticTool[ArgsT: BaseModel, AppDataT = None]:
    """One callable tool: explicit name, description, args_model, function.

    validate_and_run, dispatch, and schema are the only readers of args_model and function:
    inside these methods the type parameters are concrete, so the validated arguments flow into the function typed,
    which no outside caller could reproduce (a heterogeneous tool collection erases ArgsT and AppDataT).
    AppDataT is the app_data type the function carries, defaulting to None;
    the checker solves it from the function's ToolOutputExplicit return,
    so PydanticTool.dispatch returns DispatchHandled[AppDataT]
    and the caller reads app_data at its concrete type with no isinstance.
    """

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

    async def validate_and_run(self, args_json: str) -> ToolOutput[AppDataT]:
        """Validate args_json against args_model and run the function on it.

        A function-raised exception is a defect and propagates unchanged,
        including any ValidationError the function raises from its own pydantic use.
        A model-visible failure returns a ToolOutputExplicit carrying is_error and app_data instead of raising.
        This method returns the function's result, the same union the function declares;
        dispatch wraps it into the ToolMessage the application appends.

        Raises:
            InvalidToolArgsError: args_json failed validation; model data the model can correct.
        """
        try:
            args = self.args_model.model_validate_json(args_json)
        except ValidationError as exc:
            raise InvalidToolArgsError(exc) from exc
        return await self.function(args)

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Run this tool on call and wrap the outcome as a DispatchHandled or DispatchInvalidToolArgs.

        Assembles the ToolMessage the application appends
        and renders an argument-validation failure with render_invalid_tool_args;
        ToolManager.dispatch delegates here so that assembly exists once.
        The caller must already have matched call.name to this tool, so there is no unknown-tool outcome:
        a single PydanticTool cannot receive an off-list name.
        The returned DispatchHandled carries this tool's AppDataT,
        so the caller reads app_data at its concrete type with no isinstance.
        Every function exception propagates: it is a defect in user code.
        """
        try:
            result = await self.validate_and_run(call.args_json)
        except InvalidToolArgsError as error:
            return _invalid_args_outcome(call, _details_from_pydantic(error.validation_error))
        return _handled_outcome(call, result)


@dataclass(frozen=True, kw_only=True)
class DispatchCaptured[CapturedT: BaseModel]:
    """A capture call whose arguments validated: the acknowledgement tool_message plus the captured instance.

    tool_message is the acknowledgement ToolMessage the application appends to the conversation for the model to read.
    captured is the validated args_model instance, a required field.
    Matching this arm proves the capture happened, so no consumer revalidates or None-guards captured.
    Returned only by CaptureTool.capture; a manager-routed call erases the capture onto DispatchHandled.app_data.
    """

    tool_message: ToolMessage
    captured: CapturedT


@dataclass(frozen=True, kw_only=True)
class CaptureTool[CapturedT: BaseModel]:
    """A tool with no function: it receives one validated CapturedT instance from the model.

    This is the form for a tool whose whole job is carrying structured data from the model to the application.
    The archetypes: a final-response tool ending a tool_choice="required" loop, and a forced side capture.
    As on PydanticTool, args_model is both the schema the provider sees and the validator of the argument JSON.
    The checker solves CapturedT from args_model, so capture returns the instance typed, with no isinstance.
    There is no function field because the behavior is fixed: validate, acknowledge, hand the instance to the caller.
    acknowledgement is the model-facing content answering a valid call; the model reads it as the tool result.
    The application's loop calls capture and matches DispatchCaptured, whose captured field is required.
    dispatch exists for Tool conformance, so a CaptureTool sits in a ToolManager and bind sends its schema.
    A manager-routed call is still answered, with the capture riding the erased app_data channel.
    A CaptureTool returns data, never a control-flow signal.
    Whether a capture ends the loop is the application's decision, made in its own loop between turns.
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
        """Validate call.args_json against args_model and return the typed capture.

        A validation failure returns the same DispatchInvalidToolArgs any tool form produces.
        Its tool_message renders the field-level corrections for the model.
        The caller must already have matched call.name to this tool, as on PydanticTool.dispatch.
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
        """Answer a manager-routed call: capture's outcome with the capture as app_data.

        The manager's channel erases per-tool types, so the capture rides DispatchHandled.app_data there.
        A caller that wants the required captured field calls capture directly.
        """
        outcome = await self.capture(call)
        if isinstance(outcome, DispatchInvalidToolArgs):
            return outcome
        return DispatchHandled(tool_message=outcome.tool_message, app_data=outcome.captured)


_ARGS_OBJECT = TypeAdapter(dict[str, object])
"""Validates that a tool call's args_json is a JSON object, parsing it to a dict without coercing the values.

JSONSchemaTool uses it for the parse step of dispatch: the arguments are a JSON object (not a scalar or malformed JSON),
which is the precondition every JSON-schema tool shares and the shape jsonschema then validates field-by-field.
The `object` value type passes every value through untouched, so no field is silently reshaped.
"""


@dataclass(frozen=True, kw_only=True)
class JSONSchemaTool[AppDataT = None]:
    """One callable tool described by a raw JSON schema instead of a pydantic model.

    This is the form for a tool whose schema arrives as a plain JSON schema, not a pydantic BaseModel: the archetype
    is an MCP tool discovered at runtime, whose inputSchema is JSON and whose server validates its own inputs.
    args_schema is that JSON schema, carried to the provider unchanged (it is already the model_json_schema-shaped
    Mapping the adapters send). function receives the parsed arguments as a dict[str, object], the type dispatch
    actually builds, so a function annotated with either dict[str, object] or Mapping[str, object] is accepted
    (a Callable parameter is contravariant); it does not receive a validated model, because there is no model to
    validate against here.
    dispatch validates call.args_json in two steps: the JSON-object precondition (a malformed or non-object
    args_json becomes a DispatchInvalidToolArgs the model can correct), then jsonschema validation of the parsed
    object against args_schema, so a field-level violation lands in the same DispatchInvalidToolArgs a pydantic
    PydanticTool produces and the function only ever sees valid arguments.
    The validator class comes from jsonschema.validators.validator_for, so a $schema key in args_schema selects
    the tool owner's declared draft and a schema without one validates under Draft 2020-12.
    There is deliberately no construction-time check_schema: jsonschema counts only dict as a JSON object,
    so the metaschema check false-rejects a non-dict Mapping args_schema that validates correctly;
    a malformed schema instead raises jsonschema's own exception (jsonschema.exceptions.UnknownType, typically)
    from dispatch, propagating as a user-code defect like a function exception.
    A semantic rule the schema cannot express is the function's to enforce, returning a
    ToolOutputExplicit with is_error=True.
    AppDataT is the app_data type the function carries, defaulting to None, solved from the function's
    ToolOutputExplicit return exactly as on PydanticTool, so JSONSchemaTool.dispatch returns DispatchHandled[AppDataT].
    There is no validate_and_run counterpart here: that method exists on PydanticTool because the validated
    arguments reach the caller typed, and a JSONSchemaTool caller gains nothing over parsing the JSON itself.
    """

    name: str
    description: str
    args_schema: Mapping[str, object]
    function: Callable[[dict[str, object]], Awaitable[ToolOutput[AppDataT]]]

    @cached_property
    def _validator(self) -> jsonschema.protocols.Validator:
        """The jsonschema validator instance for args_schema, built once on first dispatch.

        cached_property stores the instance in __dict__, which the frozen dataclass permits
        (frozen blocks __setattr__, not direct __dict__ writes).
        """
        return jsonschema.validators.validator_for(self.args_schema)(self.args_schema)

    def schema(self) -> ToolSchema:
        """Convert to the provider-neutral schema, passing args_schema through unchanged."""
        return ToolSchema(
            name=self.name, description=self.description, args_schema=self.args_schema
        )

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Parse call.args_json to a dict, validate it against args_schema, run the function, and wrap the outcome.

        The JSON-object precondition runs first, then jsonschema validation of the parsed object;
        either failure becomes a DispatchInvalidToolArgs without calling the function.
        The returned DispatchHandled carries this tool's AppDataT, read at its concrete type with no isinstance.
        Every function exception propagates, and so does any jsonschema exception for a malformed args_schema
        (jsonschema.exceptions.UnknownType, typically): both are defects in user code.
        """
        try:
            args = _ARGS_OBJECT.validate_json(call.args_json)
        except ValidationError as error:
            return _invalid_args_outcome(call, _details_from_pydantic(error))
        # args came from parsing JSON text, so its values are exactly the JSON types iter_errors's
        # inline recursive-union annotation wants; object cannot prove that to the checker.
        # pyrefly: ignore[bad-argument-type]
        details = _details_from_jsonschema(self._validator.iter_errors(args))
        if details:
            return _invalid_args_outcome(call, details)
        result = await self.function(args)
        return _handled_outcome(call, result)


class Tool[AppDataT](Protocol):
    """The interface ToolManager needs from a tool: a name, a schema, and dispatch.

    PydanticTool, JSONSchemaTool, and CaptureTool all satisfy it structurally, so one ToolManager holds a mix of them.
    An application may add its own tool type by satisfying this interface.
    AppDataT appears only in dispatch's return, so it is covariant:
    a Tool of a concrete app_data type is a Tool of the manager's wider channel type.
    name is a read-only property so a frozen-dataclass name field (all three concrete forms) satisfies it;
    a plain attribute would demand a read-write name the frozen tools do not have.
    """

    @property
    def name(self) -> str:
        """Return the tool's dispatch name, matched against a ToolCall.name."""
        ...

    def schema(self) -> ToolSchema:
        """Return the provider-neutral schema of this tool."""
        ...

    async def dispatch(
        self, call: ToolCall
    ) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Run this tool on call and wrap the outcome."""
        ...


def render_invalid_tool_args(tool_name: str, details: Sequence[InvalidToolArgsDetail]) -> str:
    """Build the model-facing content for an argument-validation failure.

    A header naming the tool, then one line per failure: the dot-joined path and the message.
    A path segment is stringified before the join; an empty path renders as (root).
    Shared by the PydanticTool path (pydantic errors mapped through _details_from_pydantic) and the JSONSchemaTool path
    (jsonschema errors mapped through _details_from_jsonschema), so the two cannot drift.

    Raises:
        ValueError: details is empty; claiming invalid arguments with no listed failure would mislead the model.
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


def render_unknown_tool(called_name: str, held_names: Sequence[str]) -> str:
    """Build the model-facing content for a call naming a tool the manager does not hold.

    Names the off-list tool and lists the held tool names so the model can retry with a valid one.
    An empty tool set renders the held list as (none).
    """
    held = ", ".join(held_names) if held_names else "(none)"
    return f"unknown tool {called_name!r}; available tools: {held}"


def _details_from_pydantic(validation_error: ValidationError) -> tuple[InvalidToolArgsDetail, ...]:
    """Map a pydantic ValidationError to the neutral details at the outcome boundary.

    Only loc and msg carry over; url, ctx, and input are dropped at the source
    so the model reads no type codes, documentation URLs, or echoed input.
    """
    return tuple(
        InvalidToolArgsDetail(path=tuple(error["loc"]), message=error["msg"])
        for error in validation_error.errors(
            include_url=False, include_context=False, include_input=False
        )
    )


def _details_from_jsonschema(
    errors: Iterable[jsonschema.exceptions.ValidationError],
) -> tuple[InvalidToolArgsDetail, ...]:
    """Map jsonschema ValidationErrors to the neutral details at the outcome boundary.

    Only absolute_path and message carry over, in iteration order; the validator keyword and schema path are
    dropped at the source, symmetric with _details_from_pydantic. message rides verbatim, and jsonschema's
    messages name the offending value themselves, so the model sees what to correct.
    """
    return tuple(
        InvalidToolArgsDetail(path=tuple(error.absolute_path), message=error.message)
        for error in errors
    )


def _invalid_args_outcome(
    call: ToolCall, details: tuple[InvalidToolArgsDetail, ...]
) -> DispatchInvalidToolArgs:
    """Build the DispatchInvalidToolArgs for a call whose arguments failed validation.

    Shared by PydanticTool.dispatch, CaptureTool.capture, and both JSONSchemaTool.dispatch failure paths
    (a non-object args_json, a jsonschema violation):
    all render the same is_error ToolMessage and carry the neutral details for a caller reading the failure.
    render_invalid_tool_args's ValueError cannot fire from here: every caller passes a non-empty tuple.
    A pydantic ValidationError carries at least one error, covering the PydanticTool and CaptureTool paths.
    JSONSchemaTool.dispatch checks the mapped jsonschema details for emptiness first.
    The calling methods therefore do not list the ValueError.
    """
    tool_message = ToolMessage(
        tool_call_id=call.id,
        content=render_invalid_tool_args(tool_name=call.name, details=details),
        is_error=True,
    )
    return DispatchInvalidToolArgs(tool_message=tool_message, details=details)


def _handled_outcome[AppDataT](
    call: ToolCall, result: ToolOutput[AppDataT]
) -> DispatchHandled[AppDataT]:
    """Wrap a tool function's result into the DispatchHandled the application appends.

    Shared by PydanticTool.dispatch and JSONSchemaTool.dispatch: a ToolOutputExplicit carries its content,
    is_error, and app_data onto the ToolMessage and the arm; bare content becomes a non-error ToolMessage
    with no app_data.
    """
    if isinstance(result, ToolOutputExplicit):
        tool_message = ToolMessage(
            tool_call_id=call.id, content=result.content, is_error=result.is_error
        )
        return DispatchHandled[AppDataT](tool_message=tool_message, app_data=result.app_data)
    tool_message = ToolMessage(tool_call_id=call.id, content=result)
    return DispatchHandled[AppDataT](tool_message=tool_message)


class ToolManager:
    """Holds the tools of one conversation and routes calls to them.

    Validation and function execution live on the held tool's own dispatch, where its type parameters are concrete;
    the manager only resolves the called name and returns the outcome as a DispatchOutcome.
    """

    def __init__(self, tools: Sequence[Tool[BaseModel | Mapping[str, object] | None]]) -> None:
        """Index the tools by name.

        Each element is any Tool implementation, keyed by name.
        A manager holds a mix of PydanticTool, JSONSchemaTool, CaptureTool, and an application's own tool type.
        The app_data bound BaseModel | Mapping[str, object] | None is the widest the manager surfaces:
        it dispatches a heterogeneous set whose per-call AppDataT is erased,
        so a tool whose function carries any of those app_data types is accepted
        and every other app_data type is out of the manager's channel.
        A caller that needs a tool's own app_data type calls the tool's own dispatch, where AppDataT is concrete.

        Raises:
            ValueError: two tools share a name.
        """
        self._tools: dict[str, Tool[BaseModel | Mapping[str, object] | None]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> tuple[ToolSchema, ...]:
        """Convert every held tool to its provider-neutral schema."""
        return tuple(tool.schema() for tool in self._tools.values())

    async def dispatch(self, call: ToolCall) -> DispatchOutcome:
        """Resolve call.name to a held tool and delegate to that tool's dispatch.

        The ToolMessage assembly and argument-validation rendering live on the tool's dispatch;
        the manager only resolves the name and returns that outcome.
        app_data is erased to the manager's channel type (BaseModel | Mapping[str, object] | None)
        because the set is heterogeneous.
        An off-list name is an expected outcome:
        it returns a DispatchUnknownTool with a default is_error ToolMessage naming the held tools,
        symmetric with the DispatchInvalidToolArgs an argument error returns,
        so the loop survives a hallucinated name or a tool_call stranded by a rebind.
        Every function exception propagates: it is a defect in user code.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            tool_message = ToolMessage(
                tool_call_id=call.id,
                content=render_unknown_tool(called_name=call.name, held_names=tuple(self._tools)),
                is_error=True,
            )
            return DispatchUnknownTool(tool_message=tool_message, called_name=call.name)
        return await tool.dispatch(call)

    async def dispatch_many(self, tool_calls: Sequence[ToolCall]) -> tuple[DispatchOutcome, ...]:
        """Dispatch every call concurrently and return the outcomes ordered by tool_calls position.

        The concurrent counterpart of dispatch for the several tool calls of one assistant turn:
        every call starts at once, and each outcome sits at its call's index regardless of completion order.
        A function exception (a user-code defect, as on dispatch) does not interrupt the siblings:
        every call settles first, then the defects raise together as one DispatchExceptionGroup
        whose completed_outcomes carries the settled calls' outcomes,
        so app_data a completed sibling produced (a billing record for money the tool spent) survives the raise.
        Cancellation is never grouped: cancelling the enclosing task cancels the sibling dispatches
        and re-raises the CancelledError only after they finish unwinding,
        and a CancelledError (or any other non-Exception BaseException) a sibling produces re-raises bare,
        both because ExceptionGroup rejects such members and because grouping one would swallow cancellation.
        Defects co-occurring with such a bare re-raise still surface: they chain as its __cause__,
        a DispatchExceptionGroup carrying them and completed_outcomes as usual.

        Raises:
            DispatchExceptionGroup: one or more tool functions raised;
                its exceptions holds the defects and completed_outcomes the settled calls' outcomes,
                both ordered by tool_calls position.
            asyncio.CancelledError: the enclosing task was cancelled;
                re-raised after the sibling dispatches finish unwinding.
            BaseException: a sibling produced a non-Exception BaseException
                (its own CancelledError, typically); re-raised bare after every sibling settled,
                with any sibling defects chained as its __cause__ in a DispatchExceptionGroup.
                When several siblings produce such BaseExceptions, only the first by tool_calls position surfaces:
                a second CancelledError duplicates the first's only fact (the batch was abandoned),
                and a KeyboardInterrupt or SystemExit tears down the event loop before this function can observe it.
        """
        tasks = [asyncio.ensure_future(self.dispatch(tool_call)) for tool_call in tool_calls]
        try:
            results: list[DispatchOutcome | BaseException] = await asyncio.gather(
                *tasks, return_exceptions=True
            )
        except asyncio.CancelledError:
            # gather already cancelled the sibling tasks but does not wait for them;
            # settle them so no tool task is still unwinding after this raise.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        completed_outcomes: list[DispatchOutcome] = []
        raised_exceptions: list[Exception] = []
        base_exceptions: list[BaseException] = []
        for result in results:
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
                # Cancellation wins, but the co-occurring defects must not vanish:
                # they chain as the __cause__ the traceback prints.
                raise base_exceptions[0] from group
            raise group
        if base_exceptions:
            raise base_exceptions[0]
        return tuple(completed_outcomes)
