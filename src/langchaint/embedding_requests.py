"""Define embedding request records and the observer that follows each embedding request.

This module imports no numpy, so `Observer` and `langchaint.tracing` use it without numpy installed.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from langchaint.common.observed_operation import ObservedOperation
from langchaint.common.request_failure import _TerminalRequestFailureKind

type EmbeddingRequestErrorKind = (
    Literal["transient", "invalid_output"] | _TerminalRequestFailureKind
)
"""How one embedding request failed.

- `invalid_output`: the response had unusable vectors or reported no token count.
- Every other kind is the `RequestFailure` kind of the request's exception.
"""


@dataclass(frozen=True, kw_only=True)
class EmbeddingUsage:
    """Provider-reported input tokens and their estimated cost.

    Embedding models price input tokens only.
    `cost_in_usd` is NaN when `input_tokens_total` is nonzero and the model has no rate.
    """

    input_tokens_total: int
    cost_in_usd: float


@dataclass(frozen=True, kw_only=True)
class EmbeddingRequestErrorRecord:
    """The failure of one embedding request.

    `error_text` is the text of the exception that ended the request.
    """

    kind: EmbeddingRequestErrorKind
    error_text: str


@dataclass(frozen=True, kw_only=True)
class EmbeddingRequestRecord:
    """One provider request that `EmbeddingModel.embed` sent.

    `input_count` is the number of inputs in the request.
    `started_after_seconds` runs from the start of `embed` until the request was sent, after admission.
    `usage` is `None` when the provider reported no token count, as when the request raised before a response arrived.
    `model_served` is the model the provider reported serving the request.
    It is `None` when the request returned no usable vectors.
    It is also `None` when the provider reports no model, as Cohere on Bedrock does.
    `error` is `None` when the request returned usable vectors.
    """

    input_count: int
    started_after_seconds: float
    elapsed_seconds: float
    usage: EmbeddingUsage | None
    model_served: str | None
    error: EmbeddingRequestErrorRecord | None


@dataclass(frozen=True, kw_only=True)
class EmbeddingRequestStart:
    """What langchaint knows when one embedding request is sent.

    `dimension` is the length of each requested vector.
    """

    provider_name: str
    model: str
    dimension: int


class EmbeddingObserver(Protocol):
    """Follows each embedding request."""

    def embedding_request_started(
        self, start: EmbeddingRequestStart
    ) -> ObservedOperation[EmbeddingRequestRecord]:
        """Start following one request that `EmbeddingModel.embed` sends.

        A request starts after admission, and each retry starts a new request.
        Its handle is current while the request is awaited, and receives the request's `EmbeddingRequestRecord`.
        A cancelled request ends without a record.
        langchaint logs an exception this method raises and sends the request unobserved.
        """
        ...
