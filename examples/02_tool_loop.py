"""A tool loop: deny a call before it runs, dispatch the rest, total their fees.

Two tools, written two ways. get_weather has a pydantic model for its arguments, so the
decorator derives the schema. The MCP tool's schema is JSON the server sent, so JSONSchemaTool
carries it through unchanged. The two MCP calls are stubs.
"""

from collections.abc import Mapping

from pydantic import BaseModel, Field, TypeAdapter

from langchaint import (
    JSONSchemaTool,
    Message,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    UserMessage,
    tool,
)
from langchaint.openai import openai_model


# pydantic puts this docstring in model_json_schema, which PydanticTool sends as args_schema.
class WeatherArgs(BaseModel):
    """Which city to report the weather for."""

    city: str


class ApiFee(BaseModel):
    """What one weather lookup cost the application."""

    fee_in_usd: float


@tool(description="Return the current weather for a city.")
async def get_weather(args: WeatherArgs) -> ToolOutputExplicit[ApiFee]:
    """Report the weather to the model, and the lookup's fee to the application."""
    return ToolOutputExplicit(
        content=f"It is 18C and clear in {args.city}.", app_data=ApiFee(fee_in_usd=0.01)
    )


class McpToolEntry(BaseModel):
    """One entry of an MCP server's tools/list result.

    input_schema is the server's own JSON schema for the tool's arguments, under the wire
    name inputSchema. Nothing here authored it, and nothing here needs to read inside it.
    """

    name: str
    description: str
    input_schema: dict[str, object] = Field(alias="inputSchema")


_TOOLS_LIST_JSON = """
[
  {
    "name": "search_docs",
    "description": "Search the team's documentation and return the matching passages.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20}
      },
      "required": ["query"]
    }
  }
]
"""
"""What the MCP server answered tools/list with, kept as text so the example parses it like a client would."""

_TOOLS_LIST = TypeAdapter(tuple[McpToolEntry, ...])


async def list_mcp_tools() -> tuple[McpToolEntry, ...]:
    """Return the server's tool list; a real client awaits its MCP session here."""
    return _TOOLS_LIST.validate_json(_TOOLS_LIST_JSON)


async def call_mcp_tool(name: str, arguments: Mapping[str, object]) -> str:
    """Return what the server answered tools/call with; a real client awaits its MCP session here."""
    return f"{name} found 3 passages for {arguments['query']!r}."


def tool_from_mcp_entry(entry: McpToolEntry) -> JSONSchemaTool[None]:
    """Wrap one discovered entry as a tool the loop can dispatch.

    args_schema is the server's schema, sent to the provider unchanged.
    JSONSchemaTool validates the model's arguments against it before calling the function, so a
    violation reaches the model as a correctable tool message and never reaches the server.
    """

    async def call_the_server(arguments: dict[str, object]) -> str:
        return await call_mcp_tool(entry.name, arguments)

    return JSONSchemaTool(
        name=entry.name,
        description=entry.description,
        args_schema=entry.input_schema,
        function=call_the_server,
    )


def approve_or_deny(call: ToolCall) -> ToolMessage | None:
    """Deny a transfer_funds call, and approve every other one."""
    if call.name == "transfer_funds":
        return ToolMessage.error(call, "The user declined this action.")
    return None


async def run_agent(
    prompt: str,
    max_turns: int = 10,
) -> tuple[str, float]:
    """Run the tool loop, returning the answer and what the tools charged.

    Raises:
        RuntimeError: the model called tools for max_turns turns without a final answer.
        GenerationError: a bound.generate_one call failed.
        DispatchExceptionGroup: a tool function raised.
    """
    discovered = [tool_from_mcp_entry(entry) for entry in await list_mcp_tools()]
    bound = openai_model("gpt-5.6-terra").bind(
        system_prompt="Use tools to answer the user's question.",
        tool_manager=ToolManager([get_weather, *discovered]),
        automatic_prompt_caching=True,
    )

    messages: list[Message] = [UserMessage(content=prompt)]
    fees_in_usd = 0.0
    for _ in range(max_turns):
        response = await bound.generate_one(messages)
        messages.append(response.assistant_message)
        if not response.tool_calls:
            return response.output, fees_in_usd

        outcomes = await bound.tool_manager.dispatch_many(
            response.tool_calls, precomputed=approve_or_deny
        )
        for outcome in outcomes:
            if outcome.kind == "handled" and isinstance(outcome.app_data, ApiFee):
                fees_in_usd += outcome.app_data.fee_in_usd
            messages.append(outcome.tool_message)
    raise RuntimeError(f"agent did not finish within {max_turns} turns")
