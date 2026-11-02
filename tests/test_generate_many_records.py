"""Test which saved outcome records a resumed `generate_many_records` call returns."""

from langchaint import PlainErrorRecord, PlainGenerationRecord
from langchaint.generation._generate_many_records import _reuses


def test_a_resumed_call_reuses_only_records_other_than_retryable_errors() -> None:
    """Only `kind` decides, so records built by `model_construct` stand in for saved ones."""
    records = [
        PlainErrorRecord.model_construct(kind="retries_exhausted_error"),
        PlainErrorRecord.model_construct(kind="timed_out_error"),
        PlainErrorRecord.model_construct(kind="auth_error"),
        PlainErrorRecord.model_construct(kind="rejected_error"),
        PlainGenerationRecord[str].model_construct(),
    ]
    assert [_reuses(record) for record in records] == [False, False, False, True, True]
