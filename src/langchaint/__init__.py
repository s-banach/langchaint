"""langchaint: a provider-neutral LLM client.

Adapters wrap the official anthropic/openai SDK clients; generation happens only through LLM.bind(...) -> BoundLLM.
__all__ re-exports only the SDK-free application surface.
The backend constructors, their price catalogs, and the adapters stay in their subpackages:
re-exporting them here would force import langchaint through both SDKs.
The adapter-author contract stays in langchaint.adapter.
Internal helpers (Admission, Backoff, SequenceNotStr) are importable but off __all__.
Tool, the protocol an application implements to add its own tool form, and ToolSchema, which that protocol's
schema() returns, are on __all__: both appear in signatures application code writes against.
The tool decorator builds PydanticTool from an async function's parameter annotation.
"""

from langchaint.adapter import (
    ReasoningDelta,
    SpecificToolChoice,
    StreamItem,
    ToolChoice,
)
from langchaint.call import AttemptRecord, CallRecord
from langchaint.exceptions import (
    AbandonedCallError,
    ContextWindowExceededError,
    DispatchExceptionGroup,
    EmptyTurnError,
    EscapedExceptionError,
    GenerationError,
    InvalidRequestError,
    InvalidToolArgsError,
    MaxCompletionTokensExceededError,
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
    ImagePart,
    Message,
    MessageContent,
    Part,
    ReasoningTrace,
    StopReason,
    TextPart,
    ToolCall,
    ToolMessage,
    TurnElement,
    UserMessage,
)
from langchaint.pricing import Billing, category_cost
from langchaint.rate_limiter import RateLimiter
from langchaint.response import (
    CallResult,
    Response,
    RowValue,
    Tables,
    to_tables,
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

__all__ = [
    "LLM",
    "ZERO_USAGE",
    "AbandonedCallError",
    "AssistantMessage",
    "AttemptRecord",
    "Billing",
    "BoundLLM",
    "CallRecord",
    "CallResult",
    "CaptureTool",
    "ContextWindowExceededError",
    "DispatchCaptured",
    "DispatchExceptionGroup",
    "DispatchHandled",
    "DispatchInvalidToolArgs",
    "DispatchManyOutcome",
    "DispatchOutcome",
    "DispatchPrecomputed",
    "DispatchUnknownTool",
    "EmptyTurnError",
    "EscapedExceptionError",
    "GenerationError",
    "GenerationInput",
    "ImagePart",
    "InferenceParams",
    "InvalidRequestError",
    "InvalidToolArgsDetail",
    "InvalidToolArgsError",
    "JSONSchemaTool",
    "MaxCompletionTokensExceededError",
    "Message",
    "MessageContent",
    "Part",
    "ProviderDeclaredFinalError",
    "ProviderFailedTerminallyError",
    "PydanticTool",
    "RateLimiter",
    "ReasoningDelta",
    "ReasoningEffort",
    "ReasoningTrace",
    "RefusalError",
    "Response",
    "RetriesExhaustedError",
    "RetryUnavailableError",
    "RowValue",
    "SchemaViolationError",
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
    "ToolChoice",
    "ToolManager",
    "ToolMessage",
    "ToolOutput",
    "ToolOutputExplicit",
    "ToolSchema",
    "TransientError",
    "TurnElement",
    "UnfinishedTurnError",
    "UnknownExceptionError",
    "Usage",
    "UserMessage",
    "category_cost",
    "to_tables",
    "tool",
]
