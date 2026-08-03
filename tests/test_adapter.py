"""The neutral adapter helpers, driven directly rather than through either backend.

retry_after_seconds_from_headers is reached in production only through parse_anthropic and
parse_openai, so the backend test modules cover each parse reading its own SDK's
exception. What they cannot state is the parsing itself, which is one function shared by both:
the header precedence, the units, and which malformed value falls through to the other header.
request_json and narrowed_request are the same shape: each adapter calls them with its own SDK's
omit class and its own request subclass, and what is shared is the walk and the narrowing, exercised
here against stand-ins for both.
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
    """The precise header wins, values are seconds, and an unusable value is absent rather than zero.

    A returned 0.0 would be a wait the rate limiter honors as "no wait", so a server sending 0 or a
    negative must be indistinguishable from a server sending nothing.
    An unusable retry-after-ms falls through to retry-after, because the millisecond header is the
    optional refinement of the standard one, not a replacement that voids it. A malformed
    retry-after ends the parse there, including where the millisecond header already fell through.
    """
    assert retry_after_seconds_from_headers(headers) == expected


class _Omit:
    """Stands in for the SDK class whose instances mean "send no such field"."""


class _Nested(BaseModel):
    """Stands in for an SDK model an adapter passes by instance, which json rejects."""

    depth: int


@dataclass(frozen=True, kw_only=True)
class _Request(RequestParams):
    """One request holding a value of every kind the walk has to place."""

    model: str
    temperature: float | _Omit
    tools: list[dict[str, object]] = field(default_factory=list)
    reasoning: _Nested | _Omit = field(default_factory=_Omit)
    messages: list[dict[str, object]] = field(default_factory=list)

    @override
    def as_json(self) -> str:
        """Render through the shared walk, as each adapter's own as_json does."""
        return request_json(self, omitted_class=_Omit)


def test_request_json_drops_omitted_fields_and_keeps_every_sent_one() -> None:
    """An omitted field is absent from the object, which is what the request body does with it.

    Writing it as null would say the request sent null, a value the providers reject, and writing the
    sentinel's repr would put a memory address in an archive.
    A nested omit goes too: the walk recurses, so a field of a field is placed by the same rule.
    """
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
    """A pydantic value inside the request becomes its fields, not its repr.

    json rejects it outright, so without this the whole cell would be one unreadable string.
    """
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

    Mixing them is a defect in langchaint, not anything a provider did, so it stops before any I/O
    and names the class that arrived.
    """
    own = _Request(model="m", temperature=0.5)
    assert narrowed_request(own, _Request) is own
    with pytest.raises(TypeError, match="_OtherRequest"):
        _ = narrowed_request(_OtherRequest(), _Request)
