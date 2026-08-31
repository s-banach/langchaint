"""Validated resume state for `BoundLLM.generate_many_records`."""

import asyncio
import os
import tempfile
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, overload

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from langchaint.cancellation import await_task_cancellation_safe
from langchaint.checked_copy import CheckedCopyModel
from langchaint.exceptions import RetriesExhaustedErrorRecord, TimedOutErrorRecord
from langchaint.messages import JsonValue
from langchaint.response import CallResultRecord

_RESUME_FORMAT_VERSION = 1
_RESUME_IO_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_RESUME_MODEL_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    ser_json_inf_nan="strings",
)


async def _run_resume_io[ResultT](function: Callable[[], ResultT]) -> ResultT:
    async def run_in_resume_io_executor() -> ResultT:
        return await asyncio.get_running_loop().run_in_executor(_RESUME_IO_EXECUTOR, function)

    task = asyncio.create_task(run_in_resume_io_executor())
    return await await_task_cancellation_safe(task)


class _PositionItem[OutputT](CheckedCopyModel):
    """Validation reconstructs one result record and rejects unknown fields."""

    model_config = _RESUME_MODEL_CONFIG

    input_fingerprint: str
    result_record: CallResultRecord[OutputT] | None


class _SampleIdItem[OutputT](CheckedCopyModel):
    """Validation reconstructs one identified result record and rejects unknown fields."""

    model_config = _RESUME_MODEL_CONFIG

    sample_id: str
    input_fingerprint: str
    result_record: CallResultRecord[OutputT] | None


class _PositionDocument[OutputT](CheckedCopyModel):
    """Validation fixes the position resume document shape."""

    model_config = _RESUME_MODEL_CONFIG

    format_version: Literal[1] = 1
    binding_fingerprint: str
    identity_mode: Literal["position"] = "position"
    items: tuple[_PositionItem[OutputT], ...]


class _SampleIdDocument[OutputT](CheckedCopyModel):
    """Validation fixes the `sample_id` resume document shape and rejects duplicates."""

    model_config = _RESUME_MODEL_CONFIG

    format_version: Literal[1] = 1
    binding_fingerprint: str
    identity_mode: Literal["sample_id"] = "sample_id"
    items: tuple[_SampleIdItem[OutputT], ...]

    @model_validator(mode="after")
    def _require_unique_sample_ids(self) -> "_SampleIdDocument[OutputT]":
        sample_ids = tuple(item.sample_id for item in self.items)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("resume file sample_id values must be unique")
        return self


type _ResumeDocument[OutputT] = Annotated[
    _PositionDocument[OutputT] | _SampleIdDocument[OutputT],
    Field(discriminator="identity_mode"),
]

_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_BROAD_DOCUMENT_ADAPTER: TypeAdapter[_ResumeDocument[JsonValue]] = TypeAdapter(
    _ResumeDocument[JsonValue]
)


@dataclass(frozen=True)
class _LoadedDocument:
    document_json: bytes
    document: _PositionDocument[JsonValue] | _SampleIdDocument[JsonValue]


_CLAIMED_RESUME_PATHS: set[Path] = set()
_CLAIMED_RESUME_PATHS_LOCK = Lock()


@contextmanager
def claim_resume_path(resolved_resume_path: Path) -> Generator[None]:
    """Claim one resolved path until the surrounding generation call ends.

    Raises:
        RuntimeError: Another active call in this process has claimed the path.
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


class ResumeState[OutputT]:
    """Hold one validated document while generated records replace pending entries."""

    def __init__(
        self,
        *,
        resume_path: Path,
        document: _PositionDocument[OutputT] | _SampleIdDocument[OutputT],
        document_adapter: TypeAdapter[_ResumeDocument[OutputT]],
    ) -> None:
        self._resume_path = resume_path
        self._document = document
        self._document_adapter = document_adapter
        self._pending_index_set = {
            index
            for index, item in enumerate(document.items)
            if item.result_record is None
            or isinstance(
                item.result_record,
                (RetriesExhaustedErrorRecord, TimedOutErrorRecord),
            )
        }

    def pending_indices(self) -> tuple[int, ...]:
        """Return current input indices that require generation."""
        return tuple(sorted(self._pending_index_set))

    def store_result_record(
        self,
        index: int,
        result_record: CallResultRecord[OutputT],
    ) -> None:
        """Atomically replace one entry and mark its generation attempt complete."""
        if index < 0 or index >= len(self._document.items):
            raise IndexError(f"result index {index} is outside the generation input sequence")
        candidate_document = self._document_with_result(index, result_record)
        validated_document = _write_document(
            resume_path=self._resume_path,
            document=candidate_document,
            document_adapter=self._document_adapter,
        )
        self._document = validated_document
        self._pending_index_set.discard(index)

    def result_records(self) -> list[CallResultRecord[OutputT]]:
        """Return records in current input order after each pending item settles.

        Raises:
            RuntimeError: A current input has not completed its generation attempt.
        """
        if self._pending_index_set:
            raise RuntimeError("resume state still has pending generation inputs")
        result_records: list[CallResultRecord[OutputT]] = []
        for item in self._document.items:
            if item.result_record is None:
                raise RuntimeError("resume state has a missing result record")
            result_records.append(item.result_record)
        return result_records

    def _document_with_result(
        self,
        index: int,
        result_record: CallResultRecord[OutputT],
    ) -> _PositionDocument[OutputT] | _SampleIdDocument[OutputT]:
        items = list(self._document.items)
        items[index] = items[index].model_copy(update={"result_record": result_record})
        return self._document.model_copy(update={"items": tuple(items)})


@overload
def prepare_resume_state(
    *,
    resume_path: Path,
    response_format: None,
    binding_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    sample_ids: tuple[str, ...] | None,
) -> ResumeState[str]: ...


@overload
def prepare_resume_state[OutputT](
    *,
    resume_path: Path,
    response_format: type[OutputT],
    binding_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    sample_ids: tuple[str, ...] | None,
) -> ResumeState[OutputT]: ...


def prepare_resume_state[OutputT](
    *,
    resume_path: Path,
    response_format: type[OutputT] | None,
    binding_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    sample_ids: tuple[str, ...] | None,
) -> ResumeState[OutputT] | ResumeState[str]:
    """Validate or replace one resume document before generation starts.

    Raises:
        ValueError: Caller `sample_ids` or existing resume data are invalid.
    """
    if sample_ids is not None:
        if len(sample_ids) != len(input_fingerprints):
            raise ValueError("sample_ids must contain one value per generation input")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
    if response_format is None:
        return _prepare_resume_state(
            resume_path=resume_path,
            output_type=str,
            binding_fingerprint=binding_fingerprint,
            input_fingerprints=input_fingerprints,
            sample_ids=sample_ids,
        )
    return _prepare_resume_state(
        resume_path=resume_path,
        output_type=response_format,
        binding_fingerprint=binding_fingerprint,
        input_fingerprints=input_fingerprints,
        sample_ids=sample_ids,
    )


def _prepare_resume_state[OutputT](
    *,
    resume_path: Path,
    output_type: type[OutputT],
    binding_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    sample_ids: tuple[str, ...] | None,
) -> ResumeState[OutputT]:
    document_adapter = _document_adapter(output_type)
    loaded_document = _load_document(resume_path)
    if sample_ids is None:
        document = _prepare_position_document(
            loaded_document=loaded_document,
            document_adapter=document_adapter,
            binding_fingerprint=binding_fingerprint,
            input_fingerprints=input_fingerprints,
        )
    else:
        document = _prepare_sample_id_document(
            loaded_document=loaded_document,
            document_adapter=document_adapter,
            binding_fingerprint=binding_fingerprint,
            input_fingerprints=input_fingerprints,
            sample_ids=sample_ids,
        )
    validated_document = _write_document(
        resume_path=resume_path,
        document=document,
        document_adapter=document_adapter,
    )
    return ResumeState(
        resume_path=resume_path,
        document=validated_document,
        document_adapter=document_adapter,
    )


def _document_adapter[OutputT](
    output_type: type[OutputT],
) -> TypeAdapter[_ResumeDocument[OutputT]]:
    return TypeAdapter(_ResumeDocument[output_type])


def _load_document(resume_path: Path) -> _LoadedDocument | None:
    try:
        document_json = resume_path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        document_object = _JSON_OBJECT_ADAPTER.validate_json(document_json)
    except ValidationError as error:
        raise ValueError(f"{resume_path} is not a valid resume JSON object") from error
    format_version = document_object.get("format_version")
    if type(format_version) is not int or format_version != _RESUME_FORMAT_VERSION:
        raise ValueError(f"{resume_path} has an unsupported resume format_version")
    try:
        document = _BROAD_DOCUMENT_ADAPTER.validate_python(document_object)
    except ValidationError as error:
        raise ValueError(f"{resume_path} is not a valid version 1 resume document") from error
    return _LoadedDocument(document_json=document_json, document=document)


def _prepare_position_document[OutputT](
    *,
    loaded_document: _LoadedDocument | None,
    document_adapter: TypeAdapter[_ResumeDocument[OutputT]],
    binding_fingerprint: str,
    input_fingerprints: tuple[str, ...],
) -> _PositionDocument[OutputT]:
    if (
        loaded_document is not None
        and isinstance(loaded_document.document, _PositionDocument)
        and loaded_document.document.binding_fingerprint == binding_fingerprint
        and tuple(item.input_fingerprint for item in loaded_document.document.items)
        == input_fingerprints
    ):
        restored = document_adapter.validate_json(loaded_document.document_json)
        if not isinstance(restored, _PositionDocument):
            raise TypeError("the position discriminator changed during validation")
        return restored
    return _PositionDocument(
        binding_fingerprint=binding_fingerprint,
        items=tuple(
            _PositionItem(input_fingerprint=input_fingerprint, result_record=None)
            for input_fingerprint in input_fingerprints
        ),
    )


def _prepare_sample_id_document[OutputT](
    *,
    loaded_document: _LoadedDocument | None,
    document_adapter: TypeAdapter[_ResumeDocument[OutputT]],
    binding_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> _SampleIdDocument[OutputT]:
    stored_items: dict[str, _SampleIdItem[OutputT]] = {}
    if (
        loaded_document is not None
        and isinstance(loaded_document.document, _SampleIdDocument)
        and loaded_document.document.binding_fingerprint == binding_fingerprint
    ):
        restored = document_adapter.validate_json(loaded_document.document_json)
        if not isinstance(restored, _SampleIdDocument):
            raise TypeError("the sample_id discriminator changed during validation")
        stored_items = {item.sample_id: item for item in restored.items}
    reconciled_items: list[_SampleIdItem[OutputT]] = []
    for sample_id, input_fingerprint in zip(sample_ids, input_fingerprints, strict=True):
        stored_item = stored_items.get(sample_id)
        result_record = (
            stored_item.result_record
            if stored_item is not None and stored_item.input_fingerprint == input_fingerprint
            else None
        )
        reconciled_items.append(
            _SampleIdItem(
                sample_id=sample_id,
                input_fingerprint=input_fingerprint,
                result_record=result_record,
            )
        )
    return _SampleIdDocument(
        binding_fingerprint=binding_fingerprint,
        items=tuple(reconciled_items),
    )


def _write_document[OutputT](
    *,
    resume_path: Path,
    document: _PositionDocument[OutputT] | _SampleIdDocument[OutputT],
    document_adapter: TypeAdapter[_ResumeDocument[OutputT]],
) -> _PositionDocument[OutputT] | _SampleIdDocument[OutputT]:
    validated_document = document_adapter.validate_python(document)
    document_json = document_adapter.dump_json(validated_document, indent=2) + b"\n"
    validated_document = document_adapter.validate_json(document_json)
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
            _ = temporary_file.write(document_json)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _ = temporary_path.replace(resume_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return validated_document
