"""Provide provider-neutral LLM and embedding clients.

Generation uses `LLM.bind()` and the returned `BoundLLM`.
Embedding generation uses `EmbeddingModel.embed()`.
`__all__` exports only the SDK-free application surface.
`Tool` and `ToolSchema` support application-defined tool forms.
The `tool` decorator builds `PydanticTool` from an async function annotation.
`run_many` exposes bounded concurrent batching for application work.
"""

from typing import TYPE_CHECKING

from langchaint.adapter import (
    ReasoningDelta,
    SpecificToolChoice,
    StreamItem,
    ToolCallDelta,
    ToolChoice,
)
from langchaint.call import AttemptRecord, CallRecord
from langchaint.exceptions import (
    AbandonedCallError,
    ContextWindowExceededError,
    DispatchExceptionGroup,
    EmbeddingOutputError,
    EmptyTurnError,
    EscapedExceptionError,
    GaveUpWaiting,
    GenerationError,
    InvalidRequestError,
    InvalidToolArgsError,
    MaxCompletionTokensExceededError,
    ParserContractError,
    ProviderDeclaredFinalError,
    ProviderFailedTerminallyError,
    RefusalError,
    RetriesExhaustedError,
    RetryUnavailableError,
    SchemaViolationError,
    StreamProtocolError,
    TimedOutError,
    TransientError,
    UnfinishedTurnError,
    UnknownExceptionError,
)
from langchaint.inference_params import InferenceParams, ReasoningEffort
from langchaint.llm import LLM, BoundLLM, GenerationInput
from langchaint.messages import (
    AssistantMessage,
    AudioPart,
    ContentPart,
    ImagePart,
    ImageUrlPart,
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
from langchaint.pricing import Billing, category_cost
from langchaint.response import (
    CallResult,
    GenerateResult,
    Response,
    RowValue,
    Tables,
    ToolCallTurn,
    to_tables,
)
from langchaint.run_many import run_many
from langchaint.shared_backoff import (
    Admission,
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    PrivateBackoff,
    RetryThisOne,
    SharedBackoff,
    Verdict,
)
from langchaint.streaming import StreamHandle
from langchaint.tools import (
    CaptureTool,
    DispatchCaptured,
    DispatchHandled,
    DispatchInvalidToolArgs,
    DispatchManyOutcome,
    DispatchOutcome,
    DispatchPrecomputed,
    DispatchUnknownTool,
    InvalidToolArgsDetail,
    JSONSchemaTool,
    PydanticTool,
    Tool,
    ToolManager,
    ToolOutput,
    ToolOutputExplicit,
    ToolSchema,
    tool,
)
from langchaint.usage import ZERO_USAGE, Usage

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
    "AbandonedCallError",
    "Admission",
    "AssistantMessage",
    "AttemptRecord",
    "AudioPart",
    "Billing",
    "BoundLLM",
    "CallRecord",
    "CallResult",
    "CaptureTool",
    "ContentPart",
    "ContextWindowExceededError",
    "DispatchCaptured",
    "DispatchExceptionGroup",
    "DispatchHandled",
    "DispatchInvalidToolArgs",
    "DispatchManyOutcome",
    "DispatchOutcome",
    "DispatchPrecomputed",
    "DispatchUnknownTool",
    "DoNotRetry",
    "EmbeddingModel",
    "EmbeddingOutputError",
    "EmbeddingTask",
    "EmptyTurnError",
    "EscapedExceptionError",
    "Float2D",
    "GaveUpWaiting",
    "GenerateResult",
    "GenerationError",
    "GenerationInput",
    "ImagePart",
    "ImageUrlPart",
    "InferenceParams",
    "InvalidRequestError",
    "InvalidToolArgsDetail",
    "InvalidToolArgsError",
    "JSONSchemaTool",
    "MaxCompletionTokensExceededError",
    "Message",
    "MessageContent",
    "ParserContractError",
    "PauseAll",
    "PauseAllDoNotRetry",
    "PrivateBackoff",
    "ProviderDeclaredFinalError",
    "ProviderFailedTerminallyError",
    "PydanticTool",
    "RawPart",
    "ReasoningDelta",
    "ReasoningEffort",
    "ReasoningPart",
    "RefusalError",
    "Response",
    "RetriesExhaustedError",
    "RetryThisOne",
    "RetryUnavailableError",
    "RowValue",
    "SchemaViolationError",
    "SharedBackoff",
    "SpecificToolChoice",
    "StopReason",
    "StreamHandle",
    "StreamItem",
    "StreamProtocolError",
    "Tables",
    "TextPart",
    "TimedOutError",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallTurn",
    "ToolChoice",
    "ToolManager",
    "ToolMessage",
    "ToolOutput",
    "ToolOutputExplicit",
    "ToolSchema",
    "TransientError",
    "TurnPart",
    "UnfinishedTurnError",
    "UnknownExceptionError",
    "Usage",
    "UserMessage",
    "Verdict",
    "category_cost",
    "messages_from_json",
    "messages_to_json",
    "run_many",
    "to_tables",
    "tool",
]
