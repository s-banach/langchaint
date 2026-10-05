"""Test which saved outcome records a resumed `generate_many_records` call generates again."""

from langchaint import GenerationWithoutToolCallsRecord, PlainErrorRecord
from langchaint.generation._generate_many_records import _regenerates


def test_a_resumed_call_regenerates_only_missing_and_retryable_records() -> None:
    """Only `kind` decides, so records built by `model_construct` stand in for saved ones."""
    records = [
        None,
        PlainErrorRecord.model_construct(kind="retries_exhausted_error"),
        PlainErrorRecord.model_construct(kind="timed_out_error"),
        PlainErrorRecord.model_construct(kind="auth_error"),
        PlainErrorRecord.model_construct(kind="rejected_error"),
        GenerationWithoutToolCallsRecord[str].model_construct(),
    ]
    assert [_regenerates(record) for record in records] == [True, True, True, True, False, False]
