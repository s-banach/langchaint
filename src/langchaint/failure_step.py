"""The retry decision after one failed provider attempt, shared by generation and embedding."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchaint.adapter import ErrorClassification
from langchaint.common.exceptions import StreamProtocolError, TransientError
from langchaint.concurrency.shared_backoff import Verdict


@dataclass(frozen=True, kw_only=True)
class _RetryAfterSharedPause:
    """Retry once the next `admitted()` has waited out the shared pause that a `PauseAll` verdict started or extended.

    `retry_after` is the verdict's `retry_after`.
    """

    retry_after: float | None
    kind: Literal["retry_after_shared_pause"] = "retry_after_shared_pause"


@dataclass(frozen=True, kw_only=True)
class _RetryAfterPrivateWait:
    """Retry after this request's own `PrivateBackoff` wait, which pauses no other request.

    `retry_after` is the wait's minimum in seconds from a `RetryThisOne` verdict.
    `retry_after` is `None` when that verdict carries none or the failure reached no verdict.
    """

    retry_after: float | None
    kind: Literal["retry_after_private_wait"] = "retry_after_private_wait"


@dataclass(frozen=True, kw_only=True)
class _Terminal:
    """Stop the request; `classification` names the terminal failure."""

    classification: ErrorClassification
    kind: Literal["terminal"] = "terminal"


type _RetryStep = _RetryAfterSharedPause | _RetryAfterPrivateWait
type _FailureStep = _RetryStep | _Terminal


def _failure_step(
    failure: Exception,
    *,
    verdict: Verdict | None,
    classify: Callable[[Exception], ErrorClassification],
) -> _FailureStep:
    """Decide how a request continues after one attempt fails with `failure`.

    `verdict` is the verdict the `admitted()` block recorded for `failure`.
    `verdict` is `None` when `failure` is not one of the `SharedBackoff.failure_types`.
    A `TransientError` or `StreamProtocolError` without a verdict retries without `classify`.
    `classify` runs only for another failure without a verdict and for `DoNotRetry`.
    Only a failure without a verdict retries on the classification `"transient"`.
    The caller counts attempts, so a retry step does not promise that an attempt remains.
    """
    if verdict is None:
        if isinstance(failure, (TransientError, StreamProtocolError)):
            return _RetryAfterPrivateWait(retry_after=None)
        classification = classify(failure)
        if classification == "transient":
            return _RetryAfterPrivateWait(retry_after=None)
    elif verdict.kind == "pause_all":
        return _RetryAfterSharedPause(retry_after=verdict.retry_after)
    elif verdict.kind == "retry_this_one":
        return _RetryAfterPrivateWait(retry_after=verdict.retry_after)
    elif verdict.kind == "pause_all_do_not_retry":
        # A provider directive states this request will not succeed, which `declared_final` names.
        # `classify` could otherwise return `invalid_request` for status 429.
        classification = "declared_final"
    else:
        classification = classify(failure)
    return _Terminal(classification=classification)


def _transient_error_for_step(
    failure: Exception, message: str, step: _RetryStep
) -> TransientError:
    """Wrap one retried attempt failure as the `TransientError` its attempt record carries.

    Return an existing `TransientError` unchanged.
    This preserves its `retry_after_seconds`, `is_rate_limit`, and message.
    A `_RetryAfterSharedPause` step marks the wrap `is_rate_limit`.
    """
    if isinstance(failure, TransientError):
        return failure
    wrapped = TransientError(
        message,
        retry_after_seconds=step.retry_after,
        is_rate_limit=step.kind == "retry_after_shared_pause",
    )
    wrapped.__cause__ = failure
    return wrapped
