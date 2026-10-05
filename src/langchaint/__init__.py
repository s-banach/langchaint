"""Provide provider-neutral LLM and embedding clients.

Generation uses `LLM.bind()` and the returned `BoundLLM`.
Embedding generation uses `EmbeddingModel.embed()`.
`__all__` exports only the SDK-free application surface.
`Tool` and `ToolSchema` support application-defined tool forms.
The `tool` decorator builds `PydanticTool` from an async function annotation.
`run_many` exposes bounded concurrent execution for application work.
"""

from typing import TYPE_CHECKING

from langchaint.adapter import (
    AllowedToolsChoice,
    ReasoningDelta,
    SpecificToolChoice,
    StreamItem,
    ToolCallDelta,
    ToolChoice,
)
from langchaint.billing.pricing import Billing, TokenRates
from langchaint.billing.usage import ZERO_USAGE, Usage
from langchaint.common.exceptions import EmbeddingOutputError
from langchaint.common.messages import (
    AssistantMessage,
    AssistantPart,
    AudioPart,
    ContentPart,
    ImagePart,
    ImageUrlPart,
    JsonValue,
    Message,
    MessageContent,
    RawPart,
    ReasoningPart,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
    messages_from_json,
    messages_to_json,
)
from langchaint.concurrency.run_many import run_many
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.errors import (
    GenerationError,
    GenerationErrorKind,
    GenerationErrorRecord,
    PlainErrorRecord,
    SchemaViolationErrorRecord,
)
from langchaint.generation.llm import LLM, BoundLLM, GenerationInput
from langchaint.generation.request_history import (
    AbandonedStreamRecord,
    CutOffRequestRecord,
    RequestHistory,
    RequestProviderData,
    RequestRecord,
    SettledRequestRecord,
    TransientErrorRecord,
)
from langchaint.generation.response import (
    Generation,
    GenerationOutcome,
    GenerationOutcomeRecord,
    GenerationRecord,
    GenerationWithoutToolCalls,
    GenerationWithoutToolCallsRecord,
    GenerationWithToolCalls,
    GenerationWithToolCallsRecord,
    InputOutcomeRecord,
)
from langchaint.generation.streaming import StreamHandle
from langchaint.generation.tables import RowValue, Tables, to_tables
from langchaint.tools import (
    CaptureTool,
    DispatchCaptured,
    DispatchExceptionGroup,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyItemOutcome,
    DispatchOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    InvalidToolArgsDetail,
    InvalidToolArgsError,
    JSONSchemaTool,
    PydanticTool,
    Tool,
    ToolManager,
    ToolReturn,
    ToolReturnExplicit,
    ToolSchema,
    ToolSequence,
    tool,
)

if TYPE_CHECKING:
    from langchaint.embedding import EmbeddingModel, EmbeddingTask, Float2D


def __getattr__(name: str) -> object:
    """Resolve public embedding attributes through `langchaint.embedding`.

    Raises:
        ModuleNotFoundError: The requested attribute requires unavailable `numpy`.
        AttributeError: `name` is not a deferred public attribute.
    """
    if name == "EmbeddingModel":
        from langchaint.embedding import EmbeddingModel  # noqa: PLC0415 (defer numpy)

        return EmbeddingModel
    if name == "EmbeddingTask":
        from langchaint.embedding import EmbeddingTask  # noqa: PLC0415 (defer numpy)

        return EmbeddingTask
    if name == "Float2D":
        from langchaint.embedding import Float2D  # noqa: PLC0415 (defer numpy)

        return Float2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLM",
    "ZERO_USAGE",
    "AbandonedStreamRecord",
    "AllowedToolsChoice",
    "AssistantMessage",
    "AssistantPart",
    "AudioPart",
    "Billing",
    "BoundLLM",
    "CaptureTool",
    "ContentPart",
    "CutOffRequestRecord",
    "DispatchCaptured",
    "DispatchExceptionGroup",
    "DispatchHandled",
    "DispatchInvalidToolArgs",
    "DispatchManyItemOutcome",
    "DispatchOutcome",
    "DispatchPrecomputed",
    "DispatchUnknownTool",
    "EmbeddingModel",
    "EmbeddingOutputError",
    "EmbeddingTask",
    "Float2D",
    "Generation",
    "GenerationError",
    "GenerationErrorKind",
    "GenerationErrorRecord",
    "GenerationInput",
    "GenerationOutcome",
    "GenerationOutcomeRecord",
    "GenerationRecord",
    "GenerationWithToolCalls",
    "GenerationWithToolCallsRecord",
    "GenerationWithoutToolCalls",
    "GenerationWithoutToolCallsRecord",
    "ImagePart",
    "ImageUrlPart",
    "InputOutcomeRecord",
    "InvalidToolArgsDetail",
    "InvalidToolArgsError",
    "JSONSchemaTool",
    "JsonValue",
    "Message",
    "MessageContent",
    "PlainErrorRecord",
    "PydanticTool",
    "RawPart",
    "ReasoningDelta",
    "ReasoningPart",
    "RequestHistory",
    "RequestProviderData",
    "RequestRecord",
    "RowValue",
    "SchemaViolationErrorRecord",
    "SettledRequestRecord",
    "SharedBackoff",
    "SpecificToolChoice",
    "StopReason",
    "StreamHandle",
    "StreamItem",
    "Tables",
    "TextPart",
    "TokenRates",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolChoice",
    "ToolManager",
    "ToolMessage",
    "ToolReturn",
    "ToolReturnExplicit",
    "ToolSchema",
    "ToolSequence",
    "TransientErrorRecord",
    "Usage",
    "UserMessage",
    "messages_from_json",
    "messages_to_json",
    "run_many",
    "to_tables",
    "tool",
]
