"""Test provider-neutral adapter helpers.

retry_after_seconds_from_headers tests header precedence and units.
request_json and narrowed_request tests use local request values.
"""

import json
from dataclasses import dataclass, field
from typing import override

import pytest
from pydantic import BaseModel

from langchaint.adapter import (
    RequestParams,
    narrowed_request,
    request_json,
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
        ({"retry-after": "0"}, None),
        ({"retry-after": "-5"}, None),
        ({"retry-after-ms": "0", "retry-after": "49"}, 49.0),
        ({"retry-after-ms": "-1000", "retry-after": "49"}, 49.0),
        ({"retry-after-ms": "soon", "retry-after": "49"}, 49.0),
        ({"retry-after-ms": "0"}, None),
        ({"retry-after-ms": "soon"}, None),
        ({"retry-after-ms": "0", "retry-after": "soon"}, None),
    ],
    ids=[
        "no_headers",
        "seconds_whole",
        "seconds_fractional",
        "milliseconds",
        "milliseconds_preferred_over_seconds",
        "http_date_is_not_parsed",
        "zero_seconds_is_absent",
        "negative_seconds_is_absent",
        "zero_milliseconds_falls_through",
        "negative_milliseconds_falls_through",
        "unparseable_milliseconds_falls_through",
        "zero_milliseconds_alone",
        "unparseable_milliseconds_alone",
        "unusable_milliseconds_then_unparseable_seconds",
    ],
)
def test_retry_after_seconds_from_headers(headers: dict[str, str], expected: float | None) -> None:
    """Parse positive retry-after headers in seconds with millisecond precedence."""
    assert retry_after_seconds_from_headers(headers) == expected


class _Omit:
    """Stands in for the SDK class whose instances mean "send no such field"."""


class _Nested(BaseModel):
    """Provide a nested pydantic request value."""

    depth: int


@dataclass(frozen=True, kw_only=True)
class _Request(RequestParams):
    """Provide each request value shape under test."""

    model: str
    temperature: float | _Omit
    tools: list[dict[str, object]] = field(default_factory=list)
    reasoning: _Nested | _Omit = field(default_factory=_Omit)
    messages: list[dict[str, object]] = field(default_factory=list)

    @override
    def as_json(self) -> str:
        """Serialize through request_json."""
        return request_json(self, omitted_class=_Omit)


def test_request_json_drops_omitted_fields_and_keeps_every_sent_one() -> None:
    """request_json recursively removes omitted fields."""
    request = _Request(
        model="m",
        temperature=_Omit(),
        tools=[{"name": "t", "cache_control": _Omit()}],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert json.loads(request.as_json()) == {
        "model": "m",
        "tools": [{"name": "t"}],
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_request_json_renders_a_model_an_adapter_passes_by_instance() -> None:
    """request_json serializes nested pydantic values by field."""
    request = _Request(model="m", temperature=0.5, reasoning=_Nested(depth=2))
    assert json.loads(request.as_json()) == {
        "model": "m",
        "temperature": 0.5,
        "tools": [],
        "reasoning": {"depth": 2},
        "messages": [],
    }


@dataclass(frozen=True, kw_only=True)
class _OtherRequest(RequestParams):
    """A request some other adapter built, which narrowing to _Request must refuse."""

    @override
    def as_json(self) -> str:
        """Unreachable: the narrowing raises before anything renders this."""
        raise NotImplementedError


def test_narrowed_request_hands_back_the_adapters_own_and_refuses_every_other() -> None:
    """A request another adapter built raises rather than reaching that adapter's own open_stream.

    Mixing them raises before I/O and names the unexpected class.
    """
    own = _Request(model="m", temperature=0.5)
    assert narrowed_request(own, _Request) is own
    with pytest.raises(TypeError, match="_OtherRequest"):
        _ = narrowed_request(_OtherRequest(), _Request)
