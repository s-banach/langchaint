"""Tools.

Every tool is an async callable accepting exactly one pydantic BaseModel argument
and returning str or a Sequence[Part] (text and images the model then sees);
the function may instead return a ToolOutputExplicit carrying that content plus is_error and app_data.
The args_model is the schema source.
No signature introspection and no docstring scraping: name, description, and args_model are explicit,
so what the provider sees is exactly what the code states.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from langchaint.exceptions import InvalidToolArgsError
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
    """A tool call a function ran: the model-facing tool_message plus app_data.

    Covers both success and a function-authored failure;
    tool_message.is_error distinguishes them, the same bool the function set.
    tool_message is the ToolMessage the application appends to the conversation and the provider sees.
    app_data is the function's app_data, passed through live so the application reads it back at its concrete type:
    Tool.dispatch carries the tool's own AppDataT onto this arm,
    so a caller that dispatched a known tool reads app_data with no isinstance.
    It is None when the function returned bare content.
    It never reaches the provider: only tool_message enters the conversation the adapters convert.
    ToolManager.dispatch dispatches a heterogeneous tool set whose per-call AppDataT is erased,
    so its DispatchHandled is parameterized with the widest app_data the channel allows
    (BaseModel | Mapping[str, object] | None) and the app folds over that union there.
    """

    tool_message: ToolMessage
    app_data: AppDataT | None = None


@dataclass(frozen=True, kw_only=True)
class DispatchInvalidToolArgs:
    """A tool call whose arguments failed validation before any function ran.

    tool_message is a default is_error ToolMessage rendered from validation_error for the model to read and correct;
    the application appends it as-is or authors its own reply.
    validation_error is the pydantic ValidationError, a required field:
    matching this arm narrows it,
    so the application reads validation_error.errors() with no assert, cast, or type ignore.
    There is no app_data field because no function ran to produce any.
    """

    tool_message: ToolMessage
    validation_error: ValidationError


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
A caller that knows the single tool it dispatched keeps the tool's own app_data type by calling Tool.dispatch instead.
Every arm carries tool_message, so a consumer that only appends the reply reads result.tool_message with no match.
A consumer that reads the field-level failure detail matches DispatchInvalidToolArgs;
one that reads the off-list name matches DispatchUnknownTool.
"""


@dataclass(frozen=True, kw_only=True)
class ToolSchema:
    """The provider-neutral description of one tool.

    Adapters convert it to their wire shape at bind time.
    args_schema is the JSON schema of the args_model (model_json_schema output).
    """

    name: str
    description: str
    args_schema: Mapping[str, object]


@dataclass(frozen=True, kw_only=True)
class Tool[ArgsT: BaseModel, AppDataT = None]:
    """One callable tool: explicit name, description, args_model, function.

    validate_and_run, dispatch, and schema are the only readers of args_model and function:
    inside these methods the type parameters are concrete, so the validated arguments flow into the function typed,
    which no outside caller could reproduce (a heterogeneous tool collection erases ArgsT and AppDataT).
    AppDataT is the app_data type the function carries, defaulting to None;
    the checker solves it from the function's ToolOutputExplicit return,
    so Tool.dispatch returns DispatchHandled[AppDataT]
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

    async def dispatch(self, call: ToolCall) -> DispatchHandled[AppDataT] | DispatchInvalidToolArgs:
        """Run this tool on call and wrap the outcome as a DispatchHandled or DispatchInvalidToolArgs.

        Assembles the ToolMessage the application appends
        and renders an argument-validation failure with render_invalid_tool_args;
        ToolManager.dispatch delegates here so that assembly exists once.
        The caller must already have matched call.name to this tool, so there is no unknown-tool outcome:
        a single Tool cannot receive an off-list name.
        The returned DispatchHandled carries this tool's AppDataT,
        so the caller reads app_data at its concrete type with no isinstance.
        Every function exception propagates: it is a defect in user code.
        """
        try:
            result = await self.validate_and_run(call.args_json)
        except InvalidToolArgsError as error:
            tool_message = ToolMessage(
                tool_call_id=call.id,
                content=render_invalid_tool_args(
                    tool_name=call.name, validation_error=error.validation_error
                ),
                is_error=True,
            )
            return DispatchInvalidToolArgs(tool_message=tool_message, validation_error=error.validation_error)
        if isinstance(result, ToolOutputExplicit):
            tool_message = ToolMessage(
                tool_call_id=call.id,
                content=result.content,
                is_error=result.is_error,
            )
            return DispatchHandled[AppDataT](tool_message=tool_message, app_data=result.app_data)
        tool_message = ToolMessage(tool_call_id=call.id, content=result)
        return DispatchHandled[AppDataT](tool_message=tool_message)


def render_invalid_tool_args(tool_name: str, validation_error: ValidationError) -> str:
    """Build the model-facing content for an argument-validation failure.

    A header naming the tool, then one line per failure: the dot-joined field path and the pydantic msg.
    loc segments are str for object keys and int for list indices, so each is stringified before the join;
    an empty loc renders as (root).
    Only loc and msg are emitted;
    the url, ctx, and input fields are dropped at the source so the model reads no type codes,
    documentation URLs, or echoed input.
    """
    lines = [f"invalid arguments for {tool_name}:"]
    for error in validation_error.errors(
        include_url=False, include_context=False, include_input=False
    ):
        loc = error["loc"]
        path = ".".join(str(segment) for segment in loc) if loc else "(root)"
        lines.append(f"  {path}: {error['msg']}")
    return "\n".join(lines)


def render_unknown_tool(called_name: str, held_names: Sequence[str]) -> str:
    """Build the model-facing content for a call naming a tool the manager does not hold.

    Names the off-list tool and lists the held tool names so the model can retry with a valid one.
    An empty tool set renders the held list as (none).
    """
    held = ", ".join(held_names) if held_names else "(none)"
    return f"unknown tool {called_name!r}; available tools: {held}"


class ToolManager:
    """Holds the tools of one conversation and routes calls to them.

    Validation and function execution live on Tool (validate_and_run), where the args type parameter is concrete;
    the manager only resolves the called name and returns the outcome as a DispatchOutcome.
    """

    def __init__(
        self, tools: Sequence[Tool[Any, BaseModel | Mapping[str, object] | None]]
    ) -> None:
        """Index the tools by name.

        The app_data bound BaseModel | Mapping[str, object] | None is the widest the manager surfaces:
        it dispatches a heterogeneous set whose per-call AppDataT is erased,
        so a tool whose function carries any of those app_data types is accepted
        and every other app_data type is out of the manager's channel.
        A caller that needs a tool's own app_data type calls Tool.dispatch, where AppDataT is concrete.

        Raises:
            ValueError: two tools share a name.
        """
        self._tools: dict[str, Tool[Any, BaseModel | Mapping[str, object] | None]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> tuple[ToolSchema, ...]:
        """Convert every held tool to its provider-neutral schema."""
        return tuple(tool.schema() for tool in self._tools.values())

    async def dispatch(self, call: ToolCall) -> DispatchOutcome:
        """Resolve call.name to a held tool and delegate to Tool.dispatch.

        The ToolMessage assembly and argument-validation rendering live on Tool.dispatch;
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
                content=render_unknown_tool(
                    called_name=call.name, held_names=tuple(self._tools)
                ),
                is_error=True,
            )
            return DispatchUnknownTool(tool_message=tool_message, called_name=call.name)
        return await tool.dispatch(call)
