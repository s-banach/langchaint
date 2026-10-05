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

from langchaint.common.checked_copy import CheckedCopyModel
from langchaint.common.messages import JsonValue
from langchaint.concurrency.cancellation import await_task_cancellation_safe
from langchaint.generation.response import GenerationOutcomeRecord

_RESUME_FORMAT_VERSION = 2
"""The resume file version that this code reads and writes.

Restoring a file of this version gives the records that rerunning its binding and inputs would give.
That equality assumes the provider answers the same way both times.
Any change to the records written for the same binding and inputs requires a new version.
"""
_RESUME_IO_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_RESUME_MODEL_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    ser_json_inf_nan="strings",
)


async def _run_resume_io[ReturnT](function: Callable[[], ReturnT]) -> ReturnT:
    async def run_in_resume_io_executor() -> ReturnT:
        return await asyncio.get_running_loop().run_in_executor(_RESUME_IO_EXECUTOR, function)

    task = asyncio.create_task(run_in_resume_io_executor())
    return await await_task_cancellation_safe(task)


class _PositionItem[OutputT, WithToolCallsOutputT](CheckedCopyModel):
    """Validation reconstructs one outcome record and rejects unknown fields."""

    model_config = _RESUME_MODEL_CONFIG

    input_fingerprint: str
    outcome_record: GenerationOutcomeRecord[OutputT, WithToolCallsOutputT] | None


class _InputIdItem[OutputT, WithToolCallsOutputT](CheckedCopyModel):
    """Validation reconstructs one identified outcome record and rejects unknown fields."""

    model_config = _RESUME_MODEL_CONFIG

    input_id: str
    input_fingerprint: str
    outcome_record: GenerationOutcomeRecord[OutputT, WithToolCallsOutputT] | None


class _PositionDocument[OutputT, WithToolCallsOutputT](CheckedCopyModel):
    """Validation fixes the position resume document shape."""

    model_config = _RESUME_MODEL_CONFIG

    format_version: Literal[2] = 2
    config_fingerprint: str
    identity_mode: Literal["position"] = "position"
    items: tuple[_PositionItem[OutputT, WithToolCallsOutputT], ...]


class _InputIdDocument[OutputT, WithToolCallsOutputT](CheckedCopyModel):
    """Validation fixes the `input_id` resume document shape and rejects duplicates."""

    model_config = _RESUME_MODEL_CONFIG

    format_version: Literal[2] = 2
    config_fingerprint: str
    identity_mode: Literal["input_id"] = "input_id"
    items: tuple[_InputIdItem[OutputT, WithToolCallsOutputT], ...]

    @model_validator(mode="after")
    def _require_unique_input_ids(self) -> "_InputIdDocument[OutputT, WithToolCallsOutputT]":
        input_ids = tuple(item.input_id for item in self.items)
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("resume file input_id values must be unique")
        return self


type _ResumeDocument[OutputT, WithToolCallsOutputT] = Annotated[
    _PositionDocument[OutputT, WithToolCallsOutputT]
    | _InputIdDocument[OutputT, WithToolCallsOutputT],
    Field(discriminator="identity_mode"),
]

_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_BROAD_DOCUMENT_ADAPTER: TypeAdapter[_ResumeDocument[JsonValue, JsonValue]] = TypeAdapter(
    _ResumeDocument[JsonValue, JsonValue]
)


@dataclass(frozen=True)
class _LoadedDocument:
    document_json: bytes
    document: _PositionDocument[JsonValue, JsonValue] | _InputIdDocument[JsonValue, JsonValue]


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


def _regenerates[OutputT, WithToolCallsOutputT](
    outcome_record: GenerationOutcomeRecord[OutputT, WithToolCallsOutputT] | None,
) -> bool:
    """Report whether a resumed `generate_many_records` call generates this entry again.

    An entry without a record was never generated.
    Retry exhaustion, a timeout, and an auth failure can end differently when sent again.
    """
    if outcome_record is None:
        return True
    match outcome_record.kind:
        case "retries_exhausted_error" | "timed_out_error" | "auth_error":
            return True
        case _:
            return False


class ResumeState[OutputT, WithToolCallsOutputT]:
    """Hold one validated document while generated records replace pending entries."""

    def __init__(
        self,
        *,
        resume_path: Path,
        document: _PositionDocument[OutputT, WithToolCallsOutputT]
        | _InputIdDocument[OutputT, WithToolCallsOutputT],
        document_adapter: TypeAdapter[_ResumeDocument[OutputT, WithToolCallsOutputT]],
    ) -> None:
        self._resume_path = resume_path
        self._document = document
        self._document_adapter = document_adapter
        self._pending_index_set = {
            index for index, item in enumerate(document.items) if _regenerates(item.outcome_record)
        }

    def pending_indices(self) -> tuple[int, ...]:
        """Return current input indices that require generation."""
        return tuple(sorted(self._pending_index_set))

    def store_outcome_record(
        self,
        index: int,
        outcome_record: GenerationOutcomeRecord[OutputT, WithToolCallsOutputT],
    ) -> None:
        """Atomically replace one entry and mark its input handled."""
        if index < 0 or index >= len(self._document.items):
            raise IndexError(f"outcome index {index} is outside the generation input sequence")
        document = self._document_with_outcome_record(index, outcome_record)
        validated_document = _write_document(
            resume_path=self._resume_path,
            document=document,
            document_adapter=self._document_adapter,
        )
        self._document = validated_document
        self._pending_index_set.discard(index)

    def outcome_records(self) -> list[GenerationOutcomeRecord[OutputT, WithToolCallsOutputT]]:
        """Return records in current input order after each pending item settles.

        Raises:
            RuntimeError: A current input has no outcome record yet.
        """
        if self._pending_index_set:
            raise RuntimeError("resume state still has pending generation inputs")
        outcome_records: list[GenerationOutcomeRecord[OutputT, WithToolCallsOutputT]] = []
        for item in self._document.items:
            if item.outcome_record is None:
                raise RuntimeError("resume state has a missing outcome record")
            outcome_records.append(item.outcome_record)
        return outcome_records

    def _document_with_outcome_record(
        self,
        index: int,
        outcome_record: GenerationOutcomeRecord[OutputT, WithToolCallsOutputT],
    ) -> (
        _PositionDocument[OutputT, WithToolCallsOutputT]
        | _InputIdDocument[OutputT, WithToolCallsOutputT]
    ):
        items = list(self._document.items)
        items[index] = items[index].model_copy(update={"outcome_record": outcome_record})
        return self._document.model_copy(update={"items": tuple(items)})


@overload
def prepare_resume_state(
    *,
    resume_path: Path,
    response_format: None,
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    input_ids: tuple[str, ...] | None,
) -> ResumeState[str, str]: ...


@overload
def prepare_resume_state[OutputT](
    *,
    resume_path: Path,
    response_format: type[OutputT],
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    input_ids: tuple[str, ...] | None,
) -> ResumeState[OutputT, OutputT | None]: ...


def prepare_resume_state[OutputT](
    *,
    resume_path: Path,
    response_format: type[OutputT] | None,
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    input_ids: tuple[str, ...] | None,
) -> ResumeState[OutputT, OutputT | None] | ResumeState[str, str]:
    """Validate or replace one resume document before generation starts.

    Raises:
        ValueError: Caller `input_ids` or existing resume data are invalid.
    """
    if input_ids is not None:
        if len(input_ids) != len(input_fingerprints):
            raise ValueError("input_ids must contain one value per generation input")
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("input_ids must be unique")
    if response_format is None:
        return _prepare_resume_state(
            resume_path=resume_path,
            document_adapter=TypeAdapter(_ResumeDocument[str, str]),
            config_fingerprint=config_fingerprint,
            input_fingerprints=input_fingerprints,
            input_ids=input_ids,
        )
    return _prepare_resume_state(
        resume_path=resume_path,
        document_adapter=TypeAdapter(_ResumeDocument[response_format, response_format | None]),
        config_fingerprint=config_fingerprint,
        input_fingerprints=input_fingerprints,
        input_ids=input_ids,
    )


def _prepare_resume_state[OutputT, WithToolCallsOutputT](
    *,
    resume_path: Path,
    document_adapter: TypeAdapter[_ResumeDocument[OutputT, WithToolCallsOutputT]],
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    input_ids: tuple[str, ...] | None,
) -> ResumeState[OutputT, WithToolCallsOutputT]:
    loaded_document = _load_document(resume_path)
    if input_ids is None:
        document = _prepare_position_document(
            loaded_document=loaded_document,
            document_adapter=document_adapter,
            config_fingerprint=config_fingerprint,
            input_fingerprints=input_fingerprints,
        )
    else:
        document = _prepare_input_id_document(
            loaded_document=loaded_document,
            document_adapter=document_adapter,
            config_fingerprint=config_fingerprint,
            input_fingerprints=input_fingerprints,
            input_ids=input_ids,
        )
    document = _write_document(
        resume_path=resume_path,
        document=document,
        document_adapter=document_adapter,
    )
    return ResumeState(
        resume_path=resume_path,
        document=document,
        document_adapter=document_adapter,
    )


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
        raise ValueError(
            f"{resume_path} is not a valid version {_RESUME_FORMAT_VERSION} resume document"
        ) from error
    return _LoadedDocument(document_json=document_json, document=document)


def _prepare_position_document[OutputT, WithToolCallsOutputT](
    *,
    loaded_document: _LoadedDocument | None,
    document_adapter: TypeAdapter[_ResumeDocument[OutputT, WithToolCallsOutputT]],
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
) -> _PositionDocument[OutputT, WithToolCallsOutputT]:
    if (
        loaded_document is not None
        and isinstance(loaded_document.document, _PositionDocument)
        and loaded_document.document.config_fingerprint == config_fingerprint
        and tuple(item.input_fingerprint for item in loaded_document.document.items)
        == input_fingerprints
    ):
        restored = document_adapter.validate_json(loaded_document.document_json)
        if not isinstance(restored, _PositionDocument):
            raise TypeError("the position discriminator changed during validation")
        return restored
    return _PositionDocument(
        config_fingerprint=config_fingerprint,
        items=tuple(
            _PositionItem(input_fingerprint=input_fingerprint, outcome_record=None)
            for input_fingerprint in input_fingerprints
        ),
    )


def _prepare_input_id_document[OutputT, WithToolCallsOutputT](
    *,
    loaded_document: _LoadedDocument | None,
    document_adapter: TypeAdapter[_ResumeDocument[OutputT, WithToolCallsOutputT]],
    config_fingerprint: str,
    input_fingerprints: tuple[str, ...],
    input_ids: tuple[str, ...],
) -> _InputIdDocument[OutputT, WithToolCallsOutputT]:
    stored_items: dict[str, _InputIdItem[OutputT, WithToolCallsOutputT]] = {}
    if (
        loaded_document is not None
        and isinstance(loaded_document.document, _InputIdDocument)
        and loaded_document.document.config_fingerprint == config_fingerprint
    ):
        restored = document_adapter.validate_json(loaded_document.document_json)
        if not isinstance(restored, _InputIdDocument):
            raise TypeError("the input_id discriminator changed during validation")
        stored_items = {item.input_id: item for item in restored.items}
    reconciled_items: list[_InputIdItem[OutputT, WithToolCallsOutputT]] = []
    for input_id, input_fingerprint in zip(input_ids, input_fingerprints, strict=True):
        stored_item = stored_items.get(input_id)
        outcome_record = (
            stored_item.outcome_record
            if stored_item is not None and stored_item.input_fingerprint == input_fingerprint
            else None
        )
        reconciled_items.append(
            _InputIdItem(
                input_id=input_id,
                input_fingerprint=input_fingerprint,
                outcome_record=outcome_record,
            )
        )
    return _InputIdDocument(
        config_fingerprint=config_fingerprint,
        items=tuple(reconciled_items),
    )


def _write_document[OutputT, WithToolCallsOutputT](
    *,
    resume_path: Path,
    document: _PositionDocument[OutputT, WithToolCallsOutputT]
    | _InputIdDocument[OutputT, WithToolCallsOutputT],
    document_adapter: TypeAdapter[_ResumeDocument[OutputT, WithToolCallsOutputT]],
) -> (
    _PositionDocument[OutputT, WithToolCallsOutputT]
    | _InputIdDocument[OutputT, WithToolCallsOutputT]
):
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
