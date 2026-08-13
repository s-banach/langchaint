"""Dispatch pydantic and JSON-schema tools through an approval gate."""

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from langchaint import (
    DispatchExceptionGroup,
    DispatchHandled,
    DispatchManyOutcome,
    JSONSchemaTool,
    ToolCall,
    ToolManager,
    ToolMessage,
    ToolOutputExplicit,
    tool,
)


class TransferArgs(BaseModel):
    """Describe one requested funds transfer."""

    transfer_id: str
    recipient: str
    amount_in_usd: float = Field(gt=0, allow_inf_nan=False)


class TransferReceipt(BaseModel):
    """Record one completed funds transfer."""

    transfer_id: str
    amount_in_usd: float


@tool(description="Transfer funds after application approval.")
async def transfer_funds(args: TransferArgs) -> ToolOutputExplicit[TransferReceipt]:
    """Record a transfer and return its application receipt."""
    receipt = TransferReceipt(
        transfer_id=args.transfer_id,
        amount_in_usd=args.amount_in_usd,
    )
    return ToolOutputExplicit(
        content=f"Transferred {args.amount_in_usd} USD to {args.recipient}.",
        app_data=receipt,
    )


async def search_docs(
    arguments: dict[str, object],
) -> ToolOutputExplicit[Mapping[str, object]]:
    """Return an MCP-style result and its application data.

    Raises:
        KeyError: `arguments` lacks `"query"`.
        RuntimeError: The query requests a simulated server defect.
    """
    query = arguments["query"]
    if query == "raise server defect":
        raise RuntimeError("the MCP server failed")
    raw_result: Mapping[str, object] = {"query": query, "matches": 3}
    return ToolOutputExplicit(content=f"Found 3 passages for {query!r}.", app_data=raw_result)


search_docs_tool: JSONSchemaTool[Mapping[str, object]] = JSONSchemaTool(
    name="search_docs",
    description="Search documentation exposed by an MCP server.",
    args_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    function=search_docs,
)


def approval_gate(
    approved_transfer_call_ids: frozenset[str],
) -> Callable[[ToolCall], ToolMessage | None]:
    """Build the `precomputed` approval function."""

    def approve_or_deny(tool_call: ToolCall) -> ToolMessage | None:
        """Deny each unapproved `transfer_funds` call."""
        if (
            tool_call.name == transfer_funds.name
            and tool_call.id not in approved_transfer_call_ids
        ):
            return ToolMessage.error(tool_call, "The user declined this transfer.")
        return None

    return approve_or_deny


async def dispatch_with_approval(
    tool_calls: Sequence[ToolCall],
    approved_transfer_call_ids: frozenset[str],
) -> tuple[ToolMessage, ...]:
    """Dispatch calls and record settled outcomes before propagating defects.

    Raises:
        DispatchExceptionGroup: At least one tool function raised.
    """
    tool_manager = ToolManager([transfer_funds, search_docs_tool])
    try:
        outcomes = await tool_manager.dispatch_many(
            tool_calls,
            precomputed=approval_gate(approved_transfer_call_ids),
        )
    except DispatchExceptionGroup as exception_group:
        print(f"tool defects: {len(exception_group.exceptions)}")
        _record_app_data(exception_group.completed_outcomes)
        raise

    _record_app_data(outcomes)
    return tuple(outcome.tool_message for outcome in outcomes)


def _record_app_data(outcomes: Sequence[DispatchManyOutcome]) -> None:
    """Record application data from settled dispatch outcomes."""
    for outcome in outcomes:
        if not isinstance(outcome, DispatchHandled):
            continue
        if isinstance(outcome.app_data, TransferReceipt):
            print(f"recorded transfer: {outcome.app_data.transfer_id}")
        elif isinstance(outcome.app_data, Mapping):
            print(f"MCP result: {outcome.app_data}")
