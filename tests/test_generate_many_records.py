"""Test which saved result records a resumed `generate_many_records` call generates again."""

from langchaint import (
    AuthErrorRecord,
    InvalidRequestErrorRecord,
    ResponseRecord,
    RetriesExhaustedErrorRecord,
    TimedOutErrorRecord,
)
from langchaint.generation._generate_many_records import _regenerates


def test_a_resumed_call_regenerates_only_missing_and_retryable_records() -> None:
    """Only `kind` decides, so records built by `model_construct` stand in for saved ones."""
    records = [
        None,
        RetriesExhaustedErrorRecord.model_construct(),
        TimedOutErrorRecord.model_construct(),
        AuthErrorRecord.model_construct(),
        InvalidRequestErrorRecord.model_construct(),
        ResponseRecord[str].model_construct(),
    ]
    assert [_regenerates(record) for record in records] == [True, True, True, True, False, False]
