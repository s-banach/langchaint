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
from langchaint.billing.pricing import Billing
from langchaint.billing.usage import ZERO_USAGE, Usage
from langchaint.common.exceptions import EmbeddingOutputError
from langchaint.common.messages import (
    AssistantMessage,
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
    TurnPart,
    UserMessage,
    messages_from_json,
    messages_to_json,
)
from langchaint.concurrency.run_many import run_many
from langchaint.concurrency.shared_backoff import SharedBackoff
from langchaint.generation.call import (
    AbandonedCallRecord,
    AttemptProviderData,
    AttemptRecord,
    CallRecord,
    CutOffAttemptRecord,
    SettledAttemptRecord,
    TransientErrorRecord,
)
from langchaint.generation.errors import (
    AuthErrorRecord,
    ContextWindowExceededErrorRecord,
    EmptyTurnErrorRecord,
    EscapedExceptionErrorRecord,
    GenerationError,
    GenerationErrorKind,
    GenerationErrorRecord,
    InvalidRequestErrorRecord,
    MaxCompletionTokensExceededErrorRecord,
    ProviderDeclaredFinalErrorRecord,
    ProviderFailedTerminallyErrorRecord,
    RefusalErrorRecord,
    RetriesExhaustedErrorRecord,
    RetryUnavailableErrorRecord,
    SchemaViolationErrorRecord,
    TimedOutErrorRecord,
    UnfinishedTurnErrorRecord,
    UnknownExceptionErrorRecord,
)
from langchaint.generation.llm import LLM, BoundLLM, GenerationInput
from langchaint.generation.response import (
    CallResult,
    CallResultRecord,
    GenerateResult,
    GenerationRecord,
    Response,
    ResponseRecord,
    ToolCallTurn,
    ToolCallTurnRecord,
)
from langchaint.generation.streaming import StreamHandle
from langchaint.generation.tables import RowValue, Tables, to_tables
from langchaint.tools import (
    CaptureTool,
    DispatchCaptured,
    DispatchExceptionGroup,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyOutcome,
    DispatchOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    InvalidToolArgsDetail,
    InvalidToolArgsError,
    JSONSchemaTool,
    PydanticTool,
    Tool,
    ToolManager,
    ToolOutput,
    ToolOutputExplicit,
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
    "AbandonedCallRecord",
    "AllowedToolsChoice",
    "AssistantMessage",
    "AttemptProviderData",
    "AttemptRecord",
    "AudioPart",
    "AuthErrorRecord",
    "Billing",
    "BoundLLM",
    "CallRecord",
    "CallResult",
    "CallResultRecord",
    "CaptureTool",
    "ContentPart",
    "ContextWindowExceededErrorRecord",
    "CutOffAttemptRecord",
    "DispatchCaptured",
    "DispatchExceptionGroup",
    "DispatchHandled",
    "DispatchInvalidToolArgs",
    "DispatchManyOutcome",
    "DispatchOutcome",
    "DispatchPrecomputed",
    "DispatchUnknownTool",
    "EmbeddingModel",
    "EmbeddingOutputError",
    "EmbeddingTask",
    "EmptyTurnErrorRecord",
    "EscapedExceptionErrorRecord",
    "Float2D",
    "GenerateResult",
    "GenerationError",
    "GenerationErrorKind",
    "GenerationErrorRecord",
    "GenerationInput",
    "GenerationRecord",
    "ImagePart",
    "ImageUrlPart",
    "InvalidRequestErrorRecord",
    "InvalidToolArgsDetail",
    "InvalidToolArgsError",
    "JSONSchemaTool",
    "JsonValue",
    "MaxCompletionTokensExceededErrorRecord",
    "Message",
    "MessageContent",
    "ProviderDeclaredFinalErrorRecord",
    "ProviderFailedTerminallyErrorRecord",
    "PydanticTool",
    "RawPart",
    "ReasoningDelta",
    "ReasoningPart",
    "RefusalErrorRecord",
    "Response",
    "ResponseRecord",
    "RetriesExhaustedErrorRecord",
    "RetryUnavailableErrorRecord",
    "RowValue",
    "SchemaViolationErrorRecord",
    "SettledAttemptRecord",
    "SharedBackoff",
    "SpecificToolChoice",
    "StopReason",
    "StreamHandle",
    "StreamItem",
    "Tables",
    "TextPart",
    "TimedOutErrorRecord",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallTurn",
    "ToolCallTurnRecord",
    "ToolChoice",
    "ToolManager",
    "ToolMessage",
    "ToolOutput",
    "ToolOutputExplicit",
    "ToolSchema",
    "ToolSequence",
    "TransientErrorRecord",
    "TurnPart",
    "UnfinishedTurnErrorRecord",
    "UnknownExceptionErrorRecord",
    "Usage",
    "UserMessage",
    "messages_from_json",
    "messages_to_json",
    "run_many",
    "to_tables",
    "tool",
]
