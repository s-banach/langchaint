"""JSONSchemaTool argument validation: dispatch validates the arguments against args_schema.

The schema is the only validation code: dispatch checks that the arguments are a JSON object, then validates
them against args_schema with jsonschema (a langchaint dependency, like pydantic), so an invalid call becomes
the same DispatchInvalidToolArgs house message a PydanticTool's argument failure produces
and the function only ever sees valid arguments.
This example needs no API key: it dispatches constructed ToolCalls, no provider involved.
"""

import asyncio

from langchaint import DispatchInvalidToolArgs, JSONSchemaTool, ToolCall

SEARCH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1},
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def run_search(args: dict[str, object]) -> str:
    """Return a canned search result; dispatch validated args against SEARCH_SCHEMA, so query is present here."""
    return f"3 results for {args['query']!r}"


search_tool = JSONSchemaTool(
    name="search",
    description="Search the index.",
    args_schema=SEARCH_SCHEMA,
    function=run_search,
)


async def main() -> None:
    """Dispatch one valid and one invalid ToolCall and print both outcomes.

    The invalid call returns the house DispatchInvalidToolArgs: an is_error ToolMessage rendered from
    jsonschema's own errors, the same arm and format a PydanticTool's argument failure lands in,
    so the app's loop handles both tool forms identically.
    """
    valid = await search_tool.dispatch(
        ToolCall(id="c1", name="search", args_json='{"query": "tides"}')
    )
    print(valid.tool_message.content)

    invalid = await search_tool.dispatch(
        ToolCall(id="c2", name="search", args_json='{"limit": 0, "extra": true}')
    )
    print(invalid.tool_message.content)
    assert isinstance(invalid, DispatchInvalidToolArgs)
    for detail in invalid.details:
        print("failure at", detail.path or "(root)", "-", detail.message)


if __name__ == "__main__":
    asyncio.run(main())
