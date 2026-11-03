import json
from dataclasses import dataclass, field
from typing import NamedTuple, override

import pytest
from pydantic import BaseModel

from langchaint.adapter import (
    RequestParams,
    narrowed_request_params,
    request_failure_from_response,
    request_params_json,
    retry_after_seconds_from_headers,
)
from langchaint.common.request_failure import RequestFailure
from tests.helpers import terminal, transient


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, None),
        ({"retry-after": "49"}, 49.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after-ms": "1500", "retry-after": "49"}, 1.5),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, None),
        ({"retry-after": "Thu, 01 Jan 2026 00:00:30 GMT"}, 30.0),
        ({"retry-after": "0"}, None),
        ({"retry-after": "-5"}, None),
        ({"retry-after-ms": "0", "retry-after": "49"}, 49.0),
        ({"retry-after-ms": "-1000", "retry-after": "49"}, 49.0),
        ({"retry-after-ms": "soon", "retry-after": "49"}, 49.0),
        ({"retry-after-ms": "0"}, None),
        ({"retry-after-ms": "soon"}, None),
        ({"retry-after-ms": "0", "retry-after": "soon"}, None),
        ({"retry-after": "Mon, 01 Jan 99999 00:00:00 GMT"}, None),
        ({"retry-after": "Mon, 01 Jan 99999999999999999999 00:00:00 GMT"}, None),
        ({"retry-after": "Mon, 01 Jan 2026 00:00:00 -" + "9" * 400 + " GMT"}, None),
    ],
    ids=[
        "no_headers",
        "seconds_whole",
        "seconds_fractional",
        "milliseconds",
        "milliseconds_preferred_over_seconds",
        "expired_http_date",
        "future_http_date",
        "zero_seconds_is_absent",
        "negative_seconds_is_absent",
        "zero_milliseconds_falls_through",
        "negative_milliseconds_falls_through",
        "unparseable_milliseconds_falls_through",
        "zero_milliseconds_alone",
        "unparseable_milliseconds_alone",
        "unusable_milliseconds_then_unparseable_seconds",
        "http_date_year_past_9999",
        "http_date_year_past_c_long",
        "http_date_offset_past_float",
    ],
)
def test_retry_after_seconds_from_headers(
    headers: dict[str, str], expected: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("langchaint.adapter.time.time", lambda: 1767225600.0)
    assert retry_after_seconds_from_headers(headers) == expected


class _TableAnswers(NamedTuple):
    retries: bool
    pauses_quota: bool


_RETRY_THIS_ONE = _TableAnswers(retries=True, pauses_quota=False)
_PAUSE = _TableAnswers(retries=True, pauses_quota=True)
_DO_NOT_RETRY = _TableAnswers(retries=False, pauses_quota=False)


@pytest.mark.parametrize(
    ("status_code", "headers", "table_answers", "expected"),
    [
        (
            500,
            {"retry-after": "7"},
            _RETRY_THIS_ONE,
            transient(pauses_quota=False, retry_after_seconds=7.0),
        ),
        (400, {"retry-after": "7"}, _DO_NOT_RETRY, terminal("rejected")),
        (
            400,
            {"x-should-retry": "true", "retry-after": "3"},
            _DO_NOT_RETRY,
            transient(pauses_quota=False, retry_after_seconds=3.0),
        ),
        (
            500,
            {"x-should-retry": "false"},
            _RETRY_THIS_ONE,
            terminal("provider_failed_terminally"),
        ),
        (
            429,
            {"x-should-retry": "false", "retry-after": "7"},
            _PAUSE,
            RequestFailure(
                kind="provider_failed_terminally", pauses_quota=True, retry_after_seconds=7.0
            ),
        ),
        (401, {"x-should-retry": "false"}, _DO_NOT_RETRY, terminal("auth")),
        (400, {"x-should-retry": "false"}, _DO_NOT_RETRY, terminal("rejected")),
        (200, {"x-should-retry": "false"}, _PAUSE, transient(pauses_quota=True)),
        (200, {"x-should-retry": "true"}, _DO_NOT_RETRY, terminal("provider_failed_terminally")),
        (302, {}, _DO_NOT_RETRY, terminal("unknown_exception")),
    ],
    ids=[
        "retry_reads_retry_after",
        "no_retry_drops_retry_after",
        "directive_true_overrides_the_table",
        "directive_false_overrides_the_table",
        "directive_false_keeps_the_pause",
        "auth_status_before_directive_false",
        "client_status_before_directive_false",
        "mid_stream_error_ignores_directive_false",
        "mid_stream_error_ignores_directive_true",
        "unlisted_status",
    ],
)
def test_request_failure_from_response(
    status_code: int,
    headers: dict[str, str],
    table_answers: _TableAnswers,
    expected: RequestFailure,
) -> None:
    assert (
        request_failure_from_response(
            status_code=status_code,
            headers=headers,
            retries=table_answers.retries,
            pauses_quota=table_answers.pauses_quota,
        )
        == expected
    )


class _Omit:
    """Stands in for the SDK class whose instances mean "send no such field"."""


class _Nested(BaseModel):
    depth: int


@dataclass(frozen=True, kw_only=True)
class _SampleRequestParams(RequestParams):
    model: str
    temperature: float | _Omit
    tools: list[dict[str, object]] = field(default_factory=list)
    reasoning: _Nested | _Omit = field(default_factory=_Omit)
    messages: list[dict[str, object]] = field(default_factory=list)

    @override
    def as_json(self) -> str:
        return request_params_json(self, omitted_class=_Omit)


@pytest.mark.parametrize(
    ("request_params", "expected"),
    [
        (
            _SampleRequestParams(
                model="m",
                temperature=_Omit(),
                tools=[{"name": "t", "cache_control": _Omit()}],
                messages=[{"role": "user", "content": "hi"}],
            ),
            {
                "model": "m",
                "tools": [{"name": "t"}],
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
        (
            _SampleRequestParams(model="m", temperature=0.5, reasoning=_Nested(depth=2)),
            {
                "model": "m",
                "temperature": 0.5,
                "tools": [],
                "reasoning": {"depth": 2},
                "messages": [],
            },
        ),
    ],
    ids=["omitted_fields", "nested_model"],
)
def test_request_params_json(
    request_params: _SampleRequestParams, expected: dict[str, object]
) -> None:
    assert json.loads(request_params.as_json()) == expected


@dataclass(frozen=True, kw_only=True)
class _OtherAdapterRequestParams(RequestParams):
    @override
    def as_json(self) -> str:
        raise NotImplementedError


def test_narrowed_request_params_hands_back_the_adapters_own_and_refuses_every_other() -> None:
    own = _SampleRequestParams(model="m", temperature=0.5)
    assert narrowed_request_params(own, _SampleRequestParams) is own
    with pytest.raises(TypeError, match="_OtherAdapterRequestParams"):
        _ = narrowed_request_params(_OtherAdapterRequestParams(), _SampleRequestParams)
