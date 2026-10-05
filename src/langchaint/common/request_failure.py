"""`RequestFailure` and the mapping every retry loop shares, without langchaint imports outside `common/`."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchaint.common.exceptions import StreamProtocolError, TransientError

type _TerminalRequestFailureKind = Literal[
    "auth", "rejected", "provider_failed_terminally", "unknown_exception"
]
"""The `RequestFailure` kinds that end handling the input."""


@dataclass(frozen=True, kw_only=True)
class RequestFailure:
    """What one failed request means for its retry loop and its rate-limit quota.

    `kind` decides whether the retry loop sends the request again:
    - `transient`: send it again while `max_requests` permits.
    - `auth`: the provider rejected the client's credentials or permissions.
      `BoundLLM.config_fingerprint()` excludes credentials.
      The same request can succeed after the caller repairs them.
    - `rejected`: the provider rejected the request.
    - `provider_failed_terminally`: the provider reported a failure that sending the request again would repeat.
    - `unknown_exception`: the adapter could not place the exception.
    In generation, each kind except `transient` ends handling the input with a `GenerationError`.
    Its kind appends `_error` to the `RequestFailure` kind.
    `EmbeddingModel` re-raises the exception.

    `pauses_quota` starts or extends the `SharedBackoff` pause of every request on the rate-limit quota.
    It applies whatever `kind` is.
    `retry_after_seconds` is the provider-specified wait, or `None` when the provider specified none.
    It sets the pause length when `pauses_quota` is true.
    Otherwise it is the minimum wait before a `transient` retry.
    """

    kind: Literal["transient"] | _TerminalRequestFailureKind
    pauses_quota: bool
    retry_after_seconds: float | None


def _request_failure_of(
    error: Exception, adapter_request_failure: Callable[[Exception], RequestFailure]
) -> RequestFailure:
    """Return the `RequestFailure` of one exception a request raised.

    A `TransientError` states its own failure, including `pauses_quota`.
    A `StreamProtocolError` is transient and pauses nothing.
    `adapter_request_failure`, the adapter's `request_failure` method, maps every other exception.

    Raises:
        Exception: `adapter_request_failure` raises, which is an adapter defect.
    """
    if isinstance(error, TransientError):
        return RequestFailure(
            kind="transient",
            pauses_quota=error.pauses_quota,
            retry_after_seconds=error.retry_after_seconds,
        )
    if isinstance(error, StreamProtocolError):
        return RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None)
    return adapter_request_failure(error)
