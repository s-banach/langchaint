"""A strict, frozen pydantic base that rejects unknown keys during construction and `model_copy`.

Subclasses inherit `strict=True`, `frozen=True`, and `extra="forbid"`.
They must keep `extra="forbid"` and use pydantic's generated constructor.
`model_construct` retains pydantic's unvalidated behavior.
"""

from collections.abc import Mapping
from typing import Self, override

from pydantic import BaseModel, ConfigDict


def _bad_update_key_message(model_class: type[BaseModel], key: str) -> str:
    if isinstance(getattr(model_class, key, None), property):
        return f"model_copy cannot update property {key!r} of {model_class.__name__}; construct a new {model_class.__name__} with changed field values"
    return f"model_copy update key {key!r} is not a field of {model_class.__name__}; fields: {sorted(model_class.model_fields)}"


class CheckedCopyModel(BaseModel):
    """Validate fields strictly, freeze instances, and reject unknown keys during construction and `model_copy`."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    @classmethod
    @override
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Require `extra="forbid"` and pydantic's generated constructor.

        Raises:
            TypeError: The subclass sets `extra` to a value other than `"forbid"`, or defines `__init__`.
        """
        super().__pydantic_init_subclass__(**kwargs)
        extra = cls.model_config.get("extra")
        if extra != "forbid":
            raise TypeError(
                f'{cls.__name__} sets extra={extra!r}; CheckedCopyModel subclasses must use extra="forbid"'
            )
        if cls.__pydantic_custom_init__:
            raise TypeError(
                f"{cls.__name__} defines __init__; CheckedCopyModel subclasses must use pydantic's generated constructor"
            )

    @override
    def model_copy(
        self, *, update: Mapping[str, object] | None = None, deep: bool = False
    ) -> Self:
        """Copy after checking that every update key names a field.

        Field values retain pydantic's unvalidated `model_copy` behavior.

        Args:
            update: Field values to replace, or `None` to retain every field.
            deep: Whether pydantic deep-copies field values.

        Raises:
            TypeError: An update key does not name a field.
        """
        if update:
            for key in update:
                if key not in type(self).model_fields:
                    raise TypeError(_bad_update_key_message(type(self), key))
        return super().model_copy(update=update, deep=deep)
