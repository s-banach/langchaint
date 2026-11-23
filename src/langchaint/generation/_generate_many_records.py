"""Resume file reading and writing for `LLM.generate_many_records`.

A resume file holds JSON Lines.
The first line is a `_ResumeHeader`.
Every later line is a `_RecordLine`, and the last line for an input index holds that input's record.
"""

import asyncio
import os
import tempfile
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Literal, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from langchaint.common.checked_copy import CheckedCopyModel
from langchaint.concurrency.cancellation import await_task_cancellation_safe
from langchaint.generation.response import GenerationOutcomeRecord

_RESUME_IO_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_RESUME_MODEL_CONFIG = ConfigDict(ser_json_inf_nan="strings")


async def _run_resume_io[ReturnT](function: Callable[[], ReturnT]) -> ReturnT:
    async def run_in_resume_io_executor() -> ReturnT:
        return await asyncio.get_running_loop().run_in_executor(_RESUME_IO_EXECUTOR, function)

    task = asyncio.create_task(run_in_resume_io_executor())
    return await await_task_cancellation_safe(task)


class _ResumeHeader(CheckedCopyModel):
    """Identify the binding and the inputs whose records follow the header.

    `input_ids=None` identifies each input by its position in `input_fingerprints`.
    Otherwise `input_ids` holds one unique string per entry of `input_fingerprints`.
    """

    model_config = _RESUME_MODEL_CONFIG

    format_version: Literal[8] = 8
    config_fingerprint: str
    input_ids: tuple[str, ...] | None
    input_fingerprints: tuple[str, ...]

    @model_validator(mode="after")
    def _require_one_unique_input_id_per_input(self) -> "_ResumeHeader":
        if self.input_ids is None:
            return self
        if len(self.input_ids) != len(self.input_fingerprints):
            raise ValueError("input_ids must contain one value per generation input")
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("input_ids must be unique")
        return self


class _RecordLine[OutputT](CheckedCopyModel):
    """Validation reconstructs the record of the input at `input_index`."""

    model_config = _RESUME_MODEL_CONFIG

    input_index: int = Field(ge=0)
    outcome_record: GenerationOutcomeRecord[OutputT]


_CLAIMED_RESUME_PATHS: set[Path] = set()
_CLAIMED_RESUME_PATHS_LOCK = Lock()


@contextmanager
def claim_resume_path(resolved_resume_path: Path) -> Generator[None]:
    """Claim one resolved path until the surrounding `generate_many_records` call ends.

    Raises:
        RuntimeError: Another active `generate_many_records` call in this process has claimed the path.
    """
    with _CLAIMED_RESUME_PATHS_LOCK:
        if resolved_resume_path in _CLAIMED_RESUME_PATHS:
            raise RuntimeError(
                f"another active generate_many_records call uses {resolved_resume_path}"
            )
        _CLAIMED_RESUME_PATHS.add(resolved_resume_path)
    try:
        yield
    finally:
        with _CLAIMED_RESUME_PATHS_LOCK:
            _CLAIMED_RESUME_PATHS.remove(resolved_resume_path)


def _reuses[OutputT](outcome_record: GenerationOutcomeRecord[OutputT]) -> bool:
    """Report whether a resumed `generate_many_records` call returns this saved record.

    Retry exhaustion, a timeout, and an auth failure can end differently when sent again.
    """
    return outcome_record.kind not in ("retries_exhausted_error", "timed_out_error", "auth_error")


@overload
def read_generation_outcome_records(resume_path: Path) -> list[GenerationOutcomeRecord[str]]: ...


@overload
def read_generation_outcome_records[OutputT: str | BaseModel](
    resume_path: Path, *, response_format: type[OutputT]
) -> list[GenerationOutcomeRecord[OutputT]]: ...


def read_generation_outcome_records(
    resume_path: Path, *, response_format: type[str] | type[BaseModel] = str
) -> Sequence[GenerationOutcomeRecord[str | BaseModel]]:
    """Read the saved records from an `LLM.generate_many_records` resume file in input order.

    Each input contributes its last newline-terminated record, including errors that generation would retry.
    Inputs without a saved record are omitted.
    For structured output, pass the Pydantic model used as the binding's `response_format`.

    Raises:
        TypeError: `response_format` is a proper subclass of `str`.
        pydantic.PydanticUserError: `response_format` is not fully defined when a complete record line is validated.
        FileNotFoundError: The resume file does not exist.
        OSError: The resume file cannot be read.
        ValueError: The header is invalid or specifies an unsupported format.
        ValueError: A complete record line is invalid for the header or `response_format`.
    """
    if response_format is not str and not issubclass(response_format, BaseModel):
        raise TypeError(
            f"response_format must be str or a pydantic.BaseModel subclass, got {response_format!r}"
        )
    header, record_lines_bytes = _read_resume_header(resume_path)
    records = _records_by_input_index(
        resume_path, header, record_lines_bytes, _RecordLine[response_format]
    )
    return [records[input_index] for input_index in sorted(records)]


def prepare_resume_file[OutputT](
    *,
    resume_path: Path,
    output_type: type[OutputT],
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    input_ids: tuple[str, ...] | None,
) -> dict[int, GenerationOutcomeRecord[OutputT]]:
    """Validate the resume file, then replace it with a new header and one line per reused record.

    Returns each reused record by its current input index.
    The replacement drops superseded lines, records that generate again, and a last line cut off by a process failure.
    Plain generation records hold `output_type` output.

    Raises:
        ValueError: Caller `input_ids` or existing resume data are invalid.
    """
    header = _ResumeHeader(
        config_fingerprint=config_fingerprint,
        input_ids=input_ids,
        input_fingerprints=input_fingerprints,
    )
    record_line_type = _RecordLine[output_type]
    reused_record_by_input_index = _reused_record_by_input_index(
        resume_path=resume_path,
        header=header,
        record_line_type=record_line_type,
    )
    _replace_file(
        resume_path,
        header.model_dump_json().encode()
        + b"\n"
        + b"".join(
            _record_line_json(
                record_line_type(input_index=input_index, outcome_record=outcome_record)
            )
            for input_index, outcome_record in reused_record_by_input_index.items()
        ),
    )
    return reused_record_by_input_index


def append_outcome_record[OutputT](
    *,
    resume_path: Path,
    output_type: type[OutputT],
    input_index: int,
    outcome_record: GenerationOutcomeRecord[OutputT],
) -> None:
    """Append one record line to the resume file and flush it to disk.

    Raises:
        ValueError: The record does not validate from its own JSON.
        OSError: The resume file cannot be appended to.
    """
    line_json = _record_line_json(
        _RecordLine[output_type](input_index=input_index, outcome_record=outcome_record)
    )
    with resume_path.open("ab") as resume_file:
        _ = resume_file.write(line_json)
        resume_file.flush()
        os.fsync(resume_file.fileno())


def _stored_input_index_by_input_index(
    *, stored: _ResumeHeader, current: _ResumeHeader
) -> dict[int, int]:
    """Map each current input index to the stored input index whose record it reuses.

    Position identity reuses records only when the complete ordered fingerprint sequence is unchanged.
    `input_ids` identity reuses the record of the stored input with the same `input_id` and fingerprint.
    """
    if stored.config_fingerprint != current.config_fingerprint:
        return {}
    if stored.input_ids is None and current.input_ids is None:
        if stored.input_fingerprints != current.input_fingerprints:
            return {}
        return {input_index: input_index for input_index in range(len(current.input_fingerprints))}
    if stored.input_ids is None or current.input_ids is None:
        return {}
    stored_input_index_by_identity = {
        identity: stored_input_index
        for stored_input_index, identity in enumerate(
            zip(stored.input_ids, stored.input_fingerprints, strict=True)
        )
    }
    return {
        input_index: stored_input_index_by_identity[identity]
        for input_index, identity in enumerate(
            zip(current.input_ids, current.input_fingerprints, strict=True)
        )
        if identity in stored_input_index_by_identity
    }


def _read_resume_header(resume_path: Path) -> tuple[_ResumeHeader, bytes]:
    header_json, _, record_lines_bytes = resume_path.read_bytes().partition(b"\n")
    try:
        return _ResumeHeader.model_validate_json(header_json), record_lines_bytes
    except ValidationError as error:
        raise ValueError(f"{resume_path} is not a valid resume file") from error


def _records_by_input_index[OutputT](
    resume_path: Path,
    header: _ResumeHeader,
    record_lines_bytes: bytes,
    record_line_type: type[_RecordLine[OutputT]],
) -> dict[int, GenerationOutcomeRecord[OutputT]]:
    *record_line_jsons, _ = record_lines_bytes.split(b"\n")
    records: dict[int, GenerationOutcomeRecord[OutputT]] = {}
    for record_line_json in record_line_jsons:
        try:
            record_line = record_line_type.model_validate_json(record_line_json)
        except ValidationError as error:
            raise ValueError(f"{resume_path} has an invalid resume record line") from error
        if record_line.input_index >= len(header.input_fingerprints):
            raise ValueError(f"{resume_path} has a resume record for an input outside its header")
        records[record_line.input_index] = record_line.outcome_record
    return records


def _reused_record_by_input_index[OutputT](
    *,
    resume_path: Path,
    header: _ResumeHeader,
    record_line_type: type[_RecordLine[OutputT]],
) -> dict[int, GenerationOutcomeRecord[OutputT]]:
    """Return reusable saved records keyed by their current input index.

    Bytes after the last newline are ignored.
    Record lines are validated only when some current input matches a stored input.

    Raises:
        ValueError: The first line is not a valid `_ResumeHeader`, or a record line of a matched file is invalid.
    """
    try:
        stored_header, record_lines_bytes = _read_resume_header(resume_path)
    except FileNotFoundError:
        return {}
    stored_input_index_by_input_index = _stored_input_index_by_input_index(
        stored=stored_header, current=header
    )
    if not stored_input_index_by_input_index:
        return {}
    outcome_record_by_stored_input_index = _records_by_input_index(
        resume_path, stored_header, record_lines_bytes, record_line_type
    )
    return {
        input_index: outcome_record
        for input_index, stored_input_index in stored_input_index_by_input_index.items()
        if (outcome_record := outcome_record_by_stored_input_index.get(stored_input_index))
        is not None
        and _reuses(outcome_record)
    }


def _record_line_json[OutputT](record_line: _RecordLine[OutputT]) -> bytes:
    """Serialize one record line and confirm that it validates from that JSON.

    Raises:
        ValueError: The record does not validate from its own JSON.
    """
    line_json = record_line.model_dump_json().encode()
    _ = type(record_line).model_validate_json(line_json)
    return line_json + b"\n"


def _replace_file(resume_path: Path, file_bytes: bytes) -> None:
    """Atomically replace the resume file after its new bytes reach the disk."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=resume_path.parent,
            prefix=f".{resume_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            _ = temporary_file.write(file_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _ = temporary_path.replace(resume_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
