"""langchaint: a provider-neutral LLM client.

Adapters wrap the official anthropic/openai SDK clients; generation happens only through LLM.bind(...) -> BoundLLM.
__all__ re-exports only the SDK-free application surface.
The backend constructors, their price catalogs, and the adapters stay in their subpackages:
re-exporting them here would force import langchaint through both SDKs.
The adapter-author contract stays in langchaint.adapter.
Internal helpers (Admission, Backoff, SequenceNotStr) are importable but off __all__.
Tool, the protocol an application implements to add its own tool form, and ToolSchema, which that protocol's
schema() returns, are on __all__: both appear in signatures application code writes against.
"""

from langchaint.adapter import (
    SpecificToolChoice,
    StreamItem,
    ToolChoice,
)
from langchaint.call import AttemptRecord, CallRecord
from langchaint.exceptions import (
    ContextWindowExceededError,
    DispatchExceptionGroup,
    EmptyTurnError,
    GenerationError,
    InvalidRequestError,
    InvalidToolArgsError,
    MaxCompletionTokensExceededError,
    ProviderFailedTerminallyError,
    RefusalError,
    RetriesExhaustedError,
    RetryUnavailableError,
    SchemaViolationError,
    StreamProtocolError,
    TransientError,
    UnfinishedTurnError,
    UnrecognizedError,
)
from langchaint.inference_params import InferenceParams, ReasoningEffort
from langchaint.llm import LLM, BoundLLM, HasTools, NoTools
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
from langchaint.pricing import PricingTable
from langchaint.rate_limiter import RateLimiter
from langchaint.response import AbandonedCall, AbandonedCallLog, Response, RowValue, to_row
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
)
from langchaint.usage import ZERO_USAGE, Usage

__all__ = [
    "LLM",
    "ZERO_USAGE",
    "AbandonedCall",
    "AbandonedCallLog",
    "AssistantMessage",
    "AttemptRecord",
    "BoundLLM",
    "CallRecord",
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
    "GenerationError",
    "HasTools",
    "ImagePart",
    "InferenceParams",
    "InvalidRequestError",
    "InvalidToolArgsDetail",
    "InvalidToolArgsError",
    "JSONSchemaTool",
    "MaxCompletionTokensExceededError",
    "Message",
    "MessageContent",
    "NoTools",
    "Part",
    "PricingTable",
    "ProviderFailedTerminallyError",
    "PydanticTool",
    "RateLimiter",
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
    "TextPart",
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
    "UnrecognizedError",
    "Usage",
    "UserMessage",
    "to_row",
]
