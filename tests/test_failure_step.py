"""The retry decision after one failed attempt, over constructed failures and verdicts."""

from typing import NamedTuple

import pytest

from langchaint.adapter import ErrorClassification
from langchaint.common.exceptions import StreamProtocolError, TransientError
from langchaint.concurrency.shared_backoff import (
    DoNotRetry,
    PauseAll,
    PauseAllDoNotRetry,
    RetryThisOne,
    Verdict,
)
from langchaint.failure_step import (
    _failure_step,
    _FailureStep,
    _RetryAfterPrivateWait,
    _RetryAfterSharedPause,
    _Terminal,
)


class _Case(NamedTuple):
    failure: Exception
    verdict: Verdict | None
    step: _FailureStep
    classification: ErrorClassification | None = None
    """What `classify` returns, or `None` when the decision must not call `classify`."""


_CASES = {
    "pause_all": _Case(
        TransientError(""), PauseAll(retry_after=2.0), _RetryAfterSharedPause(retry_after=2.0)
    ),
    "retry_this_one": _Case(
        ValueError(), RetryThisOne(retry_after=1.5), _RetryAfterPrivateWait(retry_after=1.5)
    ),
    "pause_all_do_not_retry": _Case(
        TransientError(""),
        PauseAllDoNotRetry(retry_after=None),
        _Terminal(classification="declared_final"),
    ),
    "do_not_retry": _Case(
        ValueError(), DoNotRetry(), _Terminal(classification="invalid_request"), "invalid_request"
    ),
    "do_not_retry_classified_transient": _Case(
        ValueError(), DoNotRetry(), _Terminal(classification="transient"), "transient"
    ),
    "no_verdict_transient_error": _Case(
        TransientError("", retry_after_seconds=3.0), None, _RetryAfterPrivateWait(retry_after=None)
    ),
    "no_verdict_stream_protocol_error": _Case(
        StreamProtocolError(""), None, _RetryAfterPrivateWait(retry_after=None)
    ),
    "no_verdict_classified_transient": _Case(
        ValueError(), None, _RetryAfterPrivateWait(retry_after=None), "transient"
    ),
    "no_verdict_classified_terminal": _Case(
        ValueError(), None, _Terminal(classification="auth"), "auth"
    ),
}


@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_failure_step(case: _Case) -> None:
    """Each failure and verdict decides one step, calling `classify` only where the decision needs it."""
    classified: list[Exception] = []

    def classify(failure: Exception) -> ErrorClassification:
        classified.append(failure)
        assert case.classification is not None, "classify must not run for this failure"
        return case.classification

    step = _failure_step(case.failure, verdict=case.verdict, classify=classify)
    assert step == case.step
    assert classified == ([] if case.classification is None else [case.failure])
