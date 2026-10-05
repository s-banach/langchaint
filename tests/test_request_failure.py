"""The `RequestFailure` of one exception a request raised, over constructed exceptions."""

from typing import NamedTuple

import pytest

from langchaint.common.exceptions import StreamProtocolError, TransientError
from langchaint.common.request_failure import RequestFailure, _request_failure_of

_ADAPTER_REQUEST_FAILURE = RequestFailure(
    kind="auth", pauses_quota=False, retry_after_seconds=None
)


class _Case(NamedTuple):
    error: Exception
    request_failure: RequestFailure
    reaches_adapter: bool


_CASES = {
    "rate_limit_transient_error": _Case(
        TransientError("", retry_after_seconds=3.0, pauses_quota=True),
        RequestFailure(kind="transient", pauses_quota=True, retry_after_seconds=3.0),
        reaches_adapter=False,
    ),
    "transient_error": _Case(
        TransientError(""),
        RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None),
        reaches_adapter=False,
    ),
    "stream_protocol_error": _Case(
        StreamProtocolError(""),
        RequestFailure(kind="transient", pauses_quota=False, retry_after_seconds=None),
        reaches_adapter=False,
    ),
    "other_exception": _Case(ValueError(), _ADAPTER_REQUEST_FAILURE, reaches_adapter=True),
}


@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_request_failure_of(case: _Case) -> None:
    """The neutral exceptions state their own failure, and the adapter maps every other exception."""
    mapped: list[Exception] = []

    def adapter_request_failure(error: Exception) -> RequestFailure:
        mapped.append(error)
        return _ADAPTER_REQUEST_FAILURE

    assert _request_failure_of(case.error, adapter_request_failure) == case.request_failure
    assert mapped == ([case.error] if case.reaches_adapter else [])
