"""Test provider-neutral adapter helpers.

retry_after_seconds_from_headers tests header precedence, units, and HTTP dates that email.utils cannot convert.
request_params_json and narrowed_request_params tests use local request params values.
"""

import json
from dataclasses import dataclass, field
from typing import override

import pytest
from pydantic import BaseModel

from langchaint.adapter import (
    RequestParams,
    narrowed_request_params,
    request_params_json,
    retry_after_seconds_from_headers,
)


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
    """Parse positive retry-after headers in seconds with millisecond precedence."""
    monkeypatch.setattr("langchaint.adapter.time.time", lambda: 1767225600.0)
    assert retry_after_seconds_from_headers(headers) == expected


class _Omit:
    """Stands in for the SDK class whose instances mean "send no such field"."""


class _Nested(BaseModel):
    """Provide a nested pydantic request params value."""

    depth: int


@dataclass(frozen=True, kw_only=True)
class _SampleRequestParams(RequestParams):
    """Provide each request params value shape under test."""

    model: str
    temperature: float | _Omit
    tools: list[dict[str, object]] = field(default_factory=list)
    reasoning: _Nested | _Omit = field(default_factory=_Omit)
    messages: list[dict[str, object]] = field(default_factory=list)

    @override
    def as_json(self) -> str:
        """Serialize through request_params_json."""
        return request_params_json(self, omitted_class=_Omit)


def test_request_params_json_drops_omitted_fields_and_keeps_every_sent_one() -> None:
    """request_params_json recursively removes omitted fields."""
    request_params = _SampleRequestParams(
        model="m",
        temperature=_Omit(),
        tools=[{"name": "t", "cache_control": _Omit()}],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert json.loads(request_params.as_json()) == {
        "model": "m",
        "tools": [{"name": "t"}],
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_request_params_json_renders_a_model_an_adapter_passes_by_instance() -> None:
    """request_params_json serializes nested pydantic values by field."""
    request_params = _SampleRequestParams(model="m", temperature=0.5, reasoning=_Nested(depth=2))
    assert json.loads(request_params.as_json()) == {
        "model": "m",
        "temperature": 0.5,
        "tools": [],
        "reasoning": {"depth": 2},
        "messages": [],
    }


@dataclass(frozen=True, kw_only=True)
class _OtherAdapterRequestParams(RequestParams):
    """Request params some other adapter built, which narrowing to _SampleRequestParams must refuse."""

    @override
    def as_json(self) -> str:
        """Unreachable: the narrowing raises before anything renders this."""
        raise NotImplementedError


def test_narrowed_request_params_hands_back_the_adapters_own_and_refuses_every_other() -> None:
    """Request params another adapter built raise rather than reaching that adapter's own open_stream.

    Mixing them raises before I/O and names the unexpected class.
    """
    own = _SampleRequestParams(model="m", temperature=0.5)
    assert narrowed_request_params(own, _SampleRequestParams) is own
    with pytest.raises(TypeError, match="_OtherAdapterRequestParams"):
        _ = narrowed_request_params(_OtherAdapterRequestParams(), _SampleRequestParams)
