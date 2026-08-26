"""Define the example app's fixed local tools.

search_tool returns its Usage through app_data for inclusion in the run total.
"""

from events import current_gui_emitter
from pydantic import BaseModel

from langchaint import PydanticTool, ToolOutputExplicit, Usage, tool

SEARCH_USAGE: Usage = Usage(
    input_tokens_cache_read=0,
    input_tokens_cache_write=0,
    input_tokens_cache_none=200,
    output_tokens=40,
    output_tokens_reasoning=0,
    input_tokens_cache_read_cost_in_usd=0.0,
    input_tokens_cache_write_cost_in_usd=0.0,
    input_tokens_cache_none_cost_in_usd=0.0012,
    output_tokens_cost_in_usd=0.0008,
    provider_executed_tool_cost_in_usd=0.0,
)
"""One search call costs $0.002."""


class SearchArgs(BaseModel):
    """Define the provider schema for search arguments."""

    query: str


class CritiqueArgs(BaseModel):
    """Define the provider schema for critique arguments."""

    draft: str


class DelegateArgs(BaseModel):
    """Define the provider schema for delegate arguments."""

    question: str


@tool(description="Search the corpus for a query.", name="search")
async def search_tool(args: SearchArgs) -> ToolOutputExplicit[Usage]:
    """Return a fixed local result and its Usage through app_data.

    current_gui_emitter() routes progress to the dispatching run's on_event callback.

    Raises:
        LookupError: No `GuiEmitter` is active.
    """
    current_gui_emitter().emit_tool_progress(
        tool_name="search", message=f"searching the corpus for {args.query!r}"
    )
    return ToolOutputExplicit(
        content=f"Top result for {args.query!r}: a paragraph of findings.",
        app_data=SEARCH_USAGE,
    )


_FIRST_VERDICT = "revise: the draft cites no figures."
"""The first rejection that exercises revision."""


class CritiqueVerdict(BaseModel):
    """Carry critique approval through app_data."""

    approved: bool


def build_critique_tool() -> PydanticTool[CritiqueArgs, CritiqueVerdict]:
    """Build a critique tool that rejects once and then approves."""
    pending_rejections = [_FIRST_VERDICT]

    @tool(description="Critique a draft; return approval or a revision instruction.")
    async def critique(args: CritiqueArgs) -> ToolOutputExplicit[CritiqueVerdict]:  # noqa: ARG001  # the first verdict does not inspect the draft
        """Return the next verdict through app_data."""
        if pending_rejections:
            return ToolOutputExplicit(
                content=pending_rejections.pop(0), app_data=CritiqueVerdict(approved=False)
            )
        return ToolOutputExplicit(content="approved", app_data=CritiqueVerdict(approved=True))

    return critique
